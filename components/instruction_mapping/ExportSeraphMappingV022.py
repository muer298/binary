# -*- coding: utf-8 -*-
# Ghidra-backed syntax ancestry export. No inferred or nearest-address mapping.
#@category Research2.SERAPH
#@runtime Jython
from __future__ import print_function

import json
import os
from java.lang import System
from ghidra.app.decompiler import DecompInterface
from ghidra.app.decompiler.component import DecompilerUtils


def text(value):
    try:
        return unicode(value)
    except NameError:
        return str(value)


def address(value):
    return None if value is None else "0x%x" % int(str(value), 16)


def safe_range(node):
    try:
        return address(node.getMinAddress()), address(node.getMaxAddress())
    except Exception:
        return None, None


def parent(node):
    try:
        return node.Parent()
    except Exception:
        return None


def direct_anchors(token):
    operation = token.getPcodeOp()
    if operation is not None:
        sequence = operation.getSeqnum()
        return [{
            "address": address(sequence.getTarget()),
            "kind": "TOKEN_PCODE_SEQ_TARGET",
            "pcode_order": int(sequence.getOrder()),
            "pcode_time": int(sequence.getTime()),
            "pcode": text(operation),
        }]
    minimum, maximum = safe_range(token)
    if minimum is not None and minimum == maximum:
        return [{
            "address": minimum,
            "kind": "TOKEN_EXACT_SINGLE_ADDRESS",
            "pcode_order": None,
            "pcode_time": None,
            "pcode": None,
        }]
    return []


def varnode_evidence(token):
    try:
        varnode = token.getVarnode()
    except Exception:
        varnode = None
    if varnode is None:
        return None
    result = {"text": text(varnode)}
    for name in ("getAddress", "getPCAddress"):
        try:
            result[name] = address(getattr(varnode, name)())
        except Exception:
            result[name] = None
    return result


def listing_catalog(function):
    output = []
    iterator = currentProgram.getListing().getInstructions(function.getBody(), True)
    while iterator.hasNext():
        instruction = iterator.next()
        try:
            pcode = [text(operation) for operation in instruction.getPcode()]
        except Exception:
            pcode = []
        output.append({
            "address": address(instruction.getAddress()),
            "assembly": text(instruction),
            "mnemonic": text(instruction.getMnemonicString()),
            "pcode": pcode,
        })
    return output


def export_markup(markup, max_depth):
    lines = []
    flat = []
    function_position = 0
    source_lines = list(DecompilerUtils.toLines(markup))
    for line_number, line in enumerate(source_lines, 1):
        indent = text(line.getIndentString())
        parts = [indent]
        line_position = len(indent)
        token_rows = []
        for token in line.getAllTokens():
            token_text = text(token.getText())
            token_min, token_max = safe_range(token)
            row = {
                "flat_token_index": len(flat),
                "text": token_text,
                "line_char_start": line_position,
                "line_char_end": line_position + len(token_text),
                "char_start": function_position + line_position,
                "char_end": function_position + line_position + len(token_text),
                "token_min_address": token_min,
                "token_max_address": token_max,
                "anchors": direct_anchors(token),
                "varnode": varnode_evidence(token),
                "syntax_node_ids": [],
                "_object": token,
            }
            token_rows.append(row)
            flat.append(row)
            parts.append(token_text)
            line_position += len(token_text)
        line_text = "".join(parts)
        lines.append({
            "line": line_number,
            "text": line_text,
            "char_start": function_position,
            "char_end": function_position + len(line_text),
            "tokens": token_rows,
        })
        function_position += len(line_text)
        if line_number < len(source_lines):
            function_position += 1

    nodes = {}
    for token in flat:
        node = token["_object"]
        depth = 0
        seen = set()
        while node is not None and depth < max_depth:
            identity = int(System.identityHashCode(node))
            if identity in seen:
                break
            seen.add(identity)
            node_id = "node_%08x" % (identity & 0xffffffff)
            minimum, maximum = safe_range(node)
            node_parent = parent(node)
            record = nodes.get(node_id)
            if record is None:
                record = {
                    "node_id": node_id,
                    "class": text(node.getClass().getSimpleName()),
                    "min_address": minimum,
                    "max_address": maximum,
                    "is_root": node_parent is None,
                    "descendant_token_indices": set(),
                    "direct_pcode_addresses": set(),
                    "depths_from_token": set(),
                }
                nodes[node_id] = record
            record["descendant_token_indices"].add(token["flat_token_index"])
            record["depths_from_token"].add(depth)
            for anchor in token["anchors"]:
                if anchor.get("address") is not None:
                    record["direct_pcode_addresses"].add(anchor["address"])
            token["syntax_node_ids"].append(node_id)
            node = node_parent
            depth += 1

    output_nodes = []
    for node_id, record in nodes.items():
        indices = sorted(record.pop("descendant_token_indices"))
        nonempty = [flat[index] for index in indices if flat[index]["char_end"] > flat[index]["char_start"]]
        record["descendant_token_indices"] = indices
        record["direct_pcode_addresses"] = sorted(record["direct_pcode_addresses"], key=lambda value: int(value, 16))
        record["depths_from_token"] = sorted(record["depths_from_token"])
        # Ghidra embeds Jython 2.7, whose min/max do not support default=.
        record["char_start"] = min([token["char_start"] for token in nonempty]) if nonempty else None
        record["char_end"] = max([token["char_end"] for token in nonempty]) if nonempty else None
        record["nonempty_descendant_tokens"] = len(nonempty)
        output_nodes.append(record)

    for token in flat:
        del token["_object"]
    return lines, sorted(output_nodes, key=lambda row: row["node_id"])


def main():
    arguments = list(getScriptArgs())
    if len(arguments) != 5:
        raise ValueError("OUTPUT_JSON SAMPLE_ID ENTRIES_TXT TIMEOUT_SEC MAX_ANCESTOR_DEPTH required")
    output, sample_id, entry_file = arguments[0], arguments[1], arguments[2]
    timeout, max_depth = int(arguments[3]), int(arguments[4])
    with open(entry_file) as handle:
        entries = [line.strip() for line in handle if line.strip()]
    if len(entries) != len(set(entries)):
        raise ValueError("duplicate entries")

    decompiler = DecompInterface()
    decompiler.toggleCCode(True)
    decompiler.toggleSyntaxTree(True)
    decompiler.setSimplificationStyle("decompile")
    if not decompiler.openProgram(currentProgram):
        raise RuntimeError(text(decompiler.getLastMessage()))

    functions = []
    try:
        manager = currentProgram.getFunctionManager()
        space = currentProgram.getAddressFactory().getDefaultAddressSpace()
        for index, entry in enumerate(entries, 1):
            function = manager.getFunctionAt(space.getAddress(entry.replace("0x", "")))
            row = {
                "sample_id": sample_id, "function_entry": entry,
                "status": "NOT_DECOMPILED", "function_text": "",
                "lines": [], "syntax_nodes": [], "address_catalog": [],
            }
            if function is None:
                row["status"] = "FUNCTION_ENTRY_NOT_FOUND"
            elif function.isExternal():
                row["status"] = "EXTERNAL_FUNCTION"
            else:
                result = decompiler.decompileFunction(function, timeout, monitor)
                row["message"] = text(result.getErrorMessage())
                if not result.decompileCompleted():
                    row["status"] = "DECOMPILE_INCOMPLETE"
                elif result.getCCodeMarkup() is None:
                    row["status"] = "NO_CODE_MARKUP"
                else:
                    row["lines"], row["syntax_nodes"] = export_markup(result.getCCodeMarkup(), max_depth)
                    row["function_text"] = "\n".join(line["text"] for line in row["lines"])
                    row["address_catalog"] = listing_catalog(function)
                    row["status"] = "OK" if row["lines"] else "NO_CODE_LINES"
            functions.append(row)
            print("[SERAPH V022 MAPPING]", index, "/", len(entries), entry, row["status"])
    finally:
        decompiler.dispose()

    parent_dir = os.path.dirname(os.path.abspath(output))
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    payload = {
        "schema": "seraph_code_mapping_export_v0_2_2",
        "sample_id": sample_id,
        "completed": True,
        "requested_entries": entries,
        "image_base": address(currentProgram.getImageBase()),
        "mapping_policy": "exact token anchor; syntax exact PC set; Ghidra node address range; never nearest address/line/string/order",
        "functions": functions,
    }
    with open(output, "w") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
    print("SERAPH_MAPPING_EXPORT_V022_DONE", sample_id, "functions", len(functions))


if __name__ == "__main__":
    main()
