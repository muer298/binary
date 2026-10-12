#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SymGen 外部验证使用的独立 DWARF/ELF Oracle。

Revision v5.4.2 additionally recovers a strict mapping from call instruction
addresses to imported API symbols.  The mapping is derived from the full ELF
binary's disassembly/PLT annotations and dynamic symbol table; it never reads
USG evidence-edge targets. 本模块只读取 full ELF，禁止参与候选生成。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

try:
    from elftools.dwarf.descriptions import describe_form_class
    from elftools.elf.elffile import ELFFile
except ImportError:  # Checked when extraction is requested.
    describe_form_class = None
    ELFFile = None


ORACLE_SCHEMA = "task2_symgen_elf_dwarf_oracle_v1"
CALL_MNEMONICS = frozenset(("call", "callq", "bl", "blx", "jal", "jalr"))
_INSTRUCTION_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+(.*)$")
_ANGLE_SYMBOL_RE = re.compile(r"<([^>]+)>")
_MNEMONIC_RE = re.compile(r"\b(callq?|blx?|jalr?)\b", re.IGNORECASE)


def parse_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"0x[0-9a-fA-F]+|[0-9a-fA-F]{4,}", text)
    if match:
        token = match.group(0)
        try:
            return int(token, 16)
        except ValueError:
            pass
    try:
        return int(text, 0)
    except ValueError:
        return None


def normalize_symbol(value: Any) -> str:
    name = str(value or "").strip()
    name = name.replace("@plt", "").replace(".plt", "")
    name = re.sub(r"@@?[^\s]+$", "", name)
    return name.lstrip("_").lower()


def _normalize_disassembly_symbol(value: Any) -> str:
    """Normalize an objdump label while retaining only its base symbol."""
    name = str(value or "").strip()
    name = re.sub(r"[+-]0x[0-9a-fA-F]+$", "", name)
    return normalize_symbol(name)


def parse_objdump_callsites(
    text: str,
    undefined_dynamic_symbols: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, Sequence[str]], Dict[str, int]]:
    """Parse unique imported-symbol calls from GNU/LLVM objdump text.

    Only symbols also present in the ELF undefined dynamic-symbol set are
    retained.  A callsite that resolves to more than one imported symbol is
    reported as a conflict and excluded from the strict map.
    """
    imported = {
        normalize_symbol(symbol)
        for symbol in undefined_dynamic_symbols
        if normalize_symbol(symbol)
    }
    candidates: Dict[int, Set[str]] = {}
    call_instructions = 0
    annotated_calls = 0
    for line in str(text or "").splitlines():
        instruction = _INSTRUCTION_RE.match(line)
        if instruction is None:
            continue
        address = int(instruction.group(1), 16)
        body = instruction.group(2)
        mnemonic_match = _MNEMONIC_RE.search(body)
        if mnemonic_match is None:
            continue
        mnemonic = mnemonic_match.group(1).lower()
        if mnemonic not in CALL_MNEMONICS:
            continue
        call_instructions += 1
        symbols = {
            _normalize_disassembly_symbol(raw)
            for raw in _ANGLE_SYMBOL_RE.findall(body)
        }
        symbols = {symbol for symbol in symbols if symbol in imported}
        if not symbols:
            continue
        annotated_calls += 1
        candidates.setdefault(address, set()).update(symbols)

    resolved = {
        hex(address): next(iter(symbols))
        for address, symbols in candidates.items()
        if len(symbols) == 1
    }
    conflicts = {
        hex(address): sorted(symbols)
        for address, symbols in candidates.items()
        if len(symbols) > 1
    }
    stats = {
        "call_instructions": call_instructions,
        "annotated_import_calls": annotated_calls,
        "resolved_callsites": len(resolved),
        "conflicting_callsites": len(conflicts),
    }
    return resolved, conflicts, stats


def _select_objdump(explicit: Optional[str] = None) -> Optional[str]:
    candidates = [explicit, os.environ.get("OBJDUMP"), "objdump", "llvm-objdump"]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(str(candidate))
        if resolved:
            return resolved
    return None


def extract_callsite_symbols(
    full_path: str,
    undefined_dynamic_symbols: Sequence[str],
    objdump_path: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, Sequence[str]], Dict[str, Any]]:
    """Recover callsite -> imported symbol using independent ELF disassembly."""
    tool = _select_objdump(objdump_path)
    if tool is None:
        return {}, {}, {
            "status": "objdump_not_found",
            "tool": None,
            "call_instructions": 0,
            "annotated_import_calls": 0,
            "resolved_callsites": 0,
            "conflicting_callsites": 0,
        }
    if not undefined_dynamic_symbols:
        return {}, {}, {
            "status": "no_undefined_dynamic_symbols",
            "tool": tool,
            "call_instructions": 0,
            "annotated_import_calls": 0,
            "resolved_callsites": 0,
            "conflicting_callsites": 0,
        }
    try:
        completed = subprocess.run(
            [tool, "-d", str(full_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired as error:
        return {}, {}, {
            "status": "objdump_timeout",
            "tool": tool,
            "error": str(error),
            "call_instructions": 0,
            "annotated_import_calls": 0,
            "resolved_callsites": 0,
            "conflicting_callsites": 0,
        }
    except OSError as error:
        return {}, {}, {
            "status": "objdump_error",
            "tool": tool,
            "error": str(error),
            "call_instructions": 0,
            "annotated_import_calls": 0,
            "resolved_callsites": 0,
            "conflicting_callsites": 0,
        }
    if completed.returncode != 0:
        return {}, {}, {
            "status": "objdump_failed",
            "tool": tool,
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-2000:],
            "call_instructions": 0,
            "annotated_import_calls": 0,
            "resolved_callsites": 0,
            "conflicting_callsites": 0,
        }
    resolved, conflicts, stats = parse_objdump_callsites(
        completed.stdout,
        undefined_dynamic_symbols,
    )
    return resolved, conflicts, {
        "status": "ok",
        "tool": tool,
        "returncode": int(completed.returncode),
        **stats,
    }


def normalize_type_name(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b(const|volatile|restrict|static|register)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def type_category(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        for item in value:
            category = type_category(item)
            if category != "unknown":
                return category
        return "unknown"
    text = normalize_type_name(value)
    if not text or text in ("unknown", "none", "null", "base", "default"):
        return "unknown"
    if "*" in text or "ptr" in text or "pointer" in text:
        return "ptr"
    if "struct" in text or "class" in text:
        return "struct"
    if "union" in text:
        return "union"
    if "enum" in text:
        return "enum"
    if "float" in text or "double" in text:
        return "float"
    if text == "void" or text.startswith("void "):
        return "void"
    if any(
        token in text
        for token in (
            "int", "long", "short", "char", "bool", "size_t",
            "ssize_t", "uint", "byte", "dword", "qword", "undefined",
        )
    ):
        return "int"
    return "other"


def _attr_text(die: Any, key: str) -> Optional[str]:
    attribute = die.attributes.get(key)
    if attribute is None:
        return None
    value = attribute.value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _reference_die(die: Any, cu: Any, by_offset: Mapping[int, Any]) -> Any:
    attribute = die.attributes.get("DW_AT_type")
    if attribute is None or describe_form_class(attribute.form) != "reference":
        return None
    for offset in (cu.cu_offset + attribute.value, attribute.value):
        if offset in by_offset:
            return by_offset[offset]
    return None


def _type_name(
    die: Any, cu: Any, by_offset: Mapping[int, Any], seen: Optional[Set[int]] = None
) -> str:
    if die is None:
        return "unknown"
    seen = set() if seen is None else seen
    if die.offset in seen:
        return "recursive"
    seen.add(die.offset)
    name = _attr_text(die, "DW_AT_name")
    if die.tag == "DW_TAG_base_type":
        return name or "base"
    if die.tag == "DW_TAG_pointer_type":
        return _type_name(_reference_die(die, cu, by_offset), cu, by_offset, seen) + "*"
    if die.tag == "DW_TAG_const_type":
        return "const " + _type_name(_reference_die(die, cu, by_offset), cu, by_offset, seen)
    if die.tag == "DW_TAG_typedef":
        return name or _type_name(_reference_die(die, cu, by_offset), cu, by_offset, seen)
    if die.tag == "DW_TAG_array_type":
        return _type_name(_reference_die(die, cu, by_offset), cu, by_offset, seen) + "[]"
    if die.tag == "DW_TAG_structure_type":
        return "struct " + (name or "anon_%x" % die.offset)
    if die.tag == "DW_TAG_union_type":
        return "union " + (name or "anon_%x" % die.offset)
    if die.tag == "DW_TAG_enumeration_type":
        return "enum " + (name or "anon_%x" % die.offset)
    return name or die.tag.replace("DW_TAG_", "")


def _struct_offsets(
    die: Any, cu: Any, by_offset: Mapping[int, Any], seen: Optional[Set[int]] = None
) -> Optional[tuple[int, ...]]:
    if die is None:
        return None
    seen = set() if seen is None else seen
    if die.offset in seen:
        return None
    seen.add(die.offset)
    if die.tag == "DW_TAG_structure_type":
        offsets = []
        for child in die.iter_children():
            if child.tag != "DW_TAG_member":
                continue
            attribute = child.attributes.get("DW_AT_data_member_location")
            if attribute is not None and isinstance(attribute.value, int):
                offsets.append(int(attribute.value))
        return tuple(sorted(set(offsets))) if offsets else None
    if die.tag in (
        "DW_TAG_pointer_type", "DW_TAG_const_type", "DW_TAG_typedef",
        "DW_TAG_array_type",
    ):
        return _struct_offsets(
            _reference_die(die, cu, by_offset), cu, by_offset, seen
        )
    return None


def _high_pc(die: Any, low_pc: int) -> Optional[int]:
    attribute = die.attributes.get("DW_AT_high_pc")
    if attribute is None:
        return None
    if describe_form_class(attribute.form) == "address":
        return int(attribute.value)
    try:
        return int(low_pc) + int(attribute.value)
    except (TypeError, ValueError):
        return None


def extract_oracle(
    full_path: str,
    objdump_path: Optional[str] = None,
) -> Dict[str, Any]:
    if ELFFile is None:
        raise RuntimeError(
            "pyelftools is required: python3 -m pip install --user pyelftools"
        )
    path = Path(full_path)
    result: Dict[str, Any] = {
        "schema": ORACLE_SCHEMA,
        "full_path": str(path),
        "has_dwarf": False,
        "functions": [],
        "global_struct_offsets": [],
        "undefined_dynamic_symbols": [],
        "all_function_symbols": [],
        "callsite_to_symbol": {},
        "callsite_conflicts": {},
        "callsite_extraction": {},
    }
    with path.open("rb") as stream:
        elf = ELFFile(stream)
        undefined = set()
        functions = set()
        for section_name in (".dynsym", ".symtab"):
            section = elf.get_section_by_name(section_name)
            if section is None or not hasattr(section, "iter_symbols"):
                continue
            for symbol in section.iter_symbols():
                name = normalize_symbol(symbol.name)
                if not name:
                    continue
                info_type = str(symbol["st_info"]["type"])
                if info_type in ("STT_FUNC", "STT_NOTYPE"):
                    functions.add(name)
                if section_name == ".dynsym" and str(symbol["st_shndx"]) == "SHN_UNDEF":
                    undefined.add(name)
        result["undefined_dynamic_symbols"] = sorted(undefined)
        result["all_function_symbols"] = sorted(functions)
        callsites, conflicts, extraction = extract_callsite_symbols(
            str(path),
            result["undefined_dynamic_symbols"],
            objdump_path=objdump_path,
        )
        result["callsite_to_symbol"] = callsites
        result["callsite_conflicts"] = conflicts
        result["callsite_extraction"] = extraction

        if not elf.has_dwarf_info():
            return result
        result["has_dwarf"] = True
        dwarf = elf.get_dwarf_info()
        global_offsets = set()
        dwarf_functions = []
        for cu in dwarf.iter_CUs():
            by_offset = {die.offset: die for die in cu.iter_DIEs()}
            for die in cu.iter_DIEs():
                if die.tag == "DW_TAG_structure_type":
                    offsets = _struct_offsets(die, cu, by_offset)
                    if offsets:
                        global_offsets.add(offsets)
                if die.tag != "DW_TAG_subprogram":
                    continue
                low_attribute = die.attributes.get("DW_AT_low_pc")
                if low_attribute is None:
                    continue
                low_pc = int(low_attribute.value)
                params = []
                local_offsets = set()
                for child in die.iter_children():
                    if child.tag == "DW_TAG_formal_parameter":
                        type_die = _reference_die(child, cu, by_offset)
                        params.append(type_category(_type_name(type_die, cu, by_offset)))
                        offsets = _struct_offsets(type_die, cu, by_offset)
                        if offsets:
                            local_offsets.add(offsets)
                    elif child.tag == "DW_TAG_variable":
                        offsets = _struct_offsets(
                            _reference_die(child, cu, by_offset), cu, by_offset
                        )
                        if offsets:
                            local_offsets.add(offsets)
                return_die = _reference_die(die, cu, by_offset)
                return_category = type_category(
                    _type_name(return_die, cu, by_offset)
                    if return_die is not None else "void"
                )
                dwarf_functions.append({
                    "name": _attr_text(die, "DW_AT_name")
                    or _attr_text(die, "DW_AT_linkage_name") or "",
                    "low_pc": low_pc,
                    "high_pc": _high_pc(die, low_pc),
                    "param_categories": params,
                    "param_count": len(params),
                    "return_category": return_category,
                    "struct_offsets": [list(value) for value in sorted(local_offsets)],
                })
        result["functions"] = dwarf_functions
        result["global_struct_offsets"] = [
            list(value) for value in sorted(global_offsets)
        ]
    return result


def find_function(oracle: Mapping[str, Any], entry: Optional[int]) -> Optional[Dict[str, Any]]:
    if entry is None:
        return None
    exact = None
    for function in oracle.get("functions", []) or []:
        low = parse_int(function.get("low_pc"))
        high = parse_int(function.get("high_pc"))
        if low == entry:
            exact = dict(function)
        if low is not None and high is not None and low <= entry < high:
            return dict(function)
    return exact


__all__ = [
    "ORACLE_SCHEMA", "extract_callsite_symbols", "extract_oracle",
    "find_function", "normalize_symbol", "parse_int",
    "parse_objdump_callsites", "type_category",
]
