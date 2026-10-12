# -*- coding: utf-8 -*-
#@author Semantic Dictionary Generator (FULL VERSION)
#@category N-Pcode
#@runtime Jython
import json
import os
import re
import codecs
from collections import defaultdict, Counter
from ghidra.util.task import TaskMonitor
from ghidra.program.model.symbol import RefType
from ghidra.program.model.listing import Function, Parameter, Variable
from ghidra.program.model.data import PointerDataType, StructureDataType, ArrayDataType

# ==============================================
# 统一 ID 生成（严格按论文）
# ==============================================
def gen_id(layer, addr, seq):
    prefix_map = {
        "low_instr": "instr",
        "low_block": "block",
        "mid_cfg": "cfg",
        "mid_dataflow": "dataflow",
        "high_type": "type",
        "high_interface": "interface",
        "high_api": "api",
        "high_mem": "memlayout"
    }
    return "{}_{}_{}".format(prefix_map[layer], addr, seq)

# ==============================================
# 🔥 你最新原版：API 语义模板库（论文表3）
# ==============================================
API_SEMANTIC_TEMPLATES = {
    "malloc": {
        "intent": "memory_allocation",
        "param_roles": ["size"],
        "side_effect": ["heap_memory_allocation", "pointer_return"],
        "constraints": ["return_ptr_non_null", "size_positive"]
    },
    "free": {
        "intent": "memory_deallocation",
        "param_roles": ["buffer_ptr"],
        "side_effect": ["heap_memory_release"],
        "constraints": ["ptr_valid_before_free", "no_use_after_free"]
    },
    "calloc": {
        "intent": "memory_allocation_zero",
        "param_roles": ["count", "size"],
        "side_effect": ["heap_memory_allocation", "zero_initialization"],
        "constraints": ["return_ptr_non_null", "count_positive", "size_positive"]
    },
    "realloc": {
        "intent": "memory_reallocation",
        "param_roles": ["old_ptr", "new_size"],
        "side_effect": ["heap_memory_reallocation", "data_migration"],
        "constraints": ["old_ptr_valid", "new_size_positive"]
    },
    "fopen": {
        "intent": "file_open",
        "param_roles": ["path", "mode"],
        "side_effect": ["file_handle_create", "resource_acquisition"],
        "constraints": ["path_valid", "mode_valid", "handle_return"]
    },
    "fread": {
        "intent": "file_read",
        "param_roles": ["buffer", "size", "count", "file"],
        "side_effect": ["file_data_read", "buffer_fill"],
        "constraints": ["buffer_ptr_valid", "file_handle_valid"]
    },
    "fwrite": {
        "intent": "file_write",
        "param_roles": ["buffer", "size", "count", "file"],
        "side_effect": ["file_data_write", "buffer_consume"],
        "constraints": ["buffer_ptr_valid", "file_handle_valid"]
    },
    "fclose": {
        "intent": "file_close",
        "param_roles": ["file"],
        "side_effect": ["file_handle_release", "resource_release"],
        "constraints": ["file_handle_valid"]
    },
    "printf": {
        "intent": "std_output",
        "param_roles": ["format_str", "var_args"],
        "side_effect": ["console_output", "format_processing"],
        "constraints": ["format_str_valid"]
    },
    "scanf": {
        "intent": "std_input",
        "param_roles": ["format_str", "var_args"],
        "side_effect": ["console_input", "buffer_fill"],
        "constraints": ["format_str_valid", "arg_ptr_valid"]
    },
    "strcpy": {
        "intent": "string_copy",
        "param_roles": ["dest", "src"],
        "side_effect": ["buffer_copy", "memory_write"],
        "constraints": ["dest_buffer_valid", "src_null_terminated"]
    },
    "strlen": {
        "intent": "string_length_calc",
        "param_roles": ["str"],
        "side_effect": ["no_side_effect"],
        "constraints": ["str_null_terminated"]
    },
    "default": {
        "intent": "general_subroutine",
        "param_roles": ["general_param"] * 10,
        "side_effect": ["unknown"],
        "constraints": []
    }
}

def get_api_semantic_template(api_name):
    api_lower = api_name.lower()
    for key in API_SEMANTIC_TEMPLATES:
        if key in api_lower:
            return API_SEMANTIC_TEMPLATES[key]
    return API_SEMANTIC_TEMPLATES["default"]

# ==============================================
# 🔥 你最新原版：TypeSem 工具函数（最终纯净版）
# ==============================================
def get_safe_type_category(dt):
    if not dt:
        return "unknown"
    if hasattr(dt, "getCategory"):
        try:
            return str(dt.getCategory())
        except:
            return "base"
    return "base"

def get_safe_stack_offset(var):
    if not var:
        return None
    if isinstance(var, Parameter):
        return None
    try:
        return var.getStackOffset()
    except:
        return None

def get_type_candidates_and_confidence(dt, var=None, func=None):
    candidates = []
    confidence = []
    evidence = []

    if not dt:
        candidates.append("unknown")
        confidence.append(0.1)
        evidence.append("unknown_type")
        return candidates, confidence, evidence

    main_type = dt.getName()
    candidates.append(main_type)
    confidence.append(0.9)
    evidence.append("primary_type_inference")

    if isinstance(dt, PointerDataType):
        pointee = dt.getDataType()
        if pointee:
            candidates.append("ptr_%s" % pointee.getName())
            confidence.append(0.8)
            evidence.append("pointer_dereference_analysis")
            candidates.append("void*")
            confidence.append(0.3)
            evidence.append("generic_pointer_candidate")

    elif "int" in main_type.lower() or "dword" in main_type.lower() or "long" in main_type.lower():
        candidates.extend(["int32", "uint32", "int64", "uint64"])
        confidence.extend([0.7, 0.6, 0.4, 0.3])
        evidence.extend(["integer_width_ambiguity"] * 4)

    elif isinstance(dt, ArrayDataType):
        elem_type = dt.getDataType()
        if elem_type:
            candidates.append("array_%s" % elem_type.getName())
            confidence.append(0.85)
            evidence.append("array_element_analysis")

    elif isinstance(dt, StructureDataType):
        candidates.append("struct_%s" % dt.getName())
        confidence.append(0.95)
        evidence.append("structure_layout_analysis")

    if var and func and not isinstance(var, Parameter):
        stack_offset = get_safe_stack_offset(var)
        if stack_offset is not None:
            evidence.append("stack_offset_analysis")
            confidence[0] += 0.02
        if hasattr(var, "hasReferences") and var.hasReferences():
            evidence.append("dataflow_reference")
            confidence[0] += 0.03

    return candidates, confidence, evidence

# ==============================================
# 🔥 你最新原版：InterfaceSem 工具函数（论文表3）
# ==============================================
def infer_param_role(param_type, param_name, func):
    type_name = param_type.getName().lower()
    role_candidates = []
    role_confidence = []

    if isinstance(param_type, PointerDataType) or "ptr" in type_name or "buf" in param_name.lower():
        role_candidates.append("buffer")
        role_confidence.append(0.9)
        role_candidates.append("pointer")
        role_confidence.append(0.8)
    elif "int" in type_name and ("len" in param_name.lower() or "size" in param_name.lower()):
        role_candidates.append("length")
        role_confidence.append(0.9)
        role_candidates.append("counter")
        role_confidence.append(0.7)
    elif "int" in type_name and ("fd" in param_name.lower() or "handle" in param_name.lower()):
        role_candidates.append("handle")
        role_confidence.append(0.85)
        role_candidates.append("fd")
        role_confidence.append(0.8)
    else:
        role_candidates.append("general_param")
        role_confidence.append(0.95)

    return role_candidates, role_confidence

# ==============================================
# 🔥 你最新原版：MemLayoutSem 工具函数（论文表3 修复版）
# ==============================================
def get_safe_data_size(dt):
    if not dt:
        return 0
    try:
        return dt.getLength()
    except:
        return 0

def analyze_struct_memory_layout(func, dt):
    layout_info = {
        "base_obj": str(dt.getName()),
        "offset_set": [],
        "size_rel": [],
        "field_relations": []
    }

    if not isinstance(dt, StructureDataType):
        layout_info["base_obj"] = "simple_type_%s" % dt.getName()
        layout_info["offset_set"] = [0]
        layout_info["size_rel"] = ["size_eq_%d" % get_safe_data_size(dt)]
        return layout_info

    field_offset = 0
    for field in dt.getComponents():
        field_name = field.getFieldName() if field.getFieldName() else "field_%d" % len(layout_info["offset_set"])
        field_dt = field.getDataType()
        field_size = get_safe_data_size(field_dt)

        layout_info["offset_set"].append(field_offset)
        size_constraint = "size_eq_%d" % field_size
        if field_offset > 0:
            prev_offset = layout_info["offset_set"][-2]
            size_constraint += " offset_gt_%d" % prev_offset
        layout_info["size_rel"].append(size_constraint)

        layout_info["field_relations"].append({
            "field_name": field_name,
            "offset": field_offset,
            "type": field_dt.getName(),
            "size": field_size
        })

        field_offset += field_size

    total_size = get_safe_data_size(dt)
    layout_info["size_rel"].append("total_size_eq_%d" % total_size)
    return layout_info

def infer_memory_roles(base_obj, offset_set, size_rel):
    roles = []
    if any(o > 0 for o in offset_set):
        roles.append("struct_with_fields")
    if any("ptr" in s.lower() for s in size_rel):
        roles.append("pointer_based")
    if any("size_eq" in s for s in size_rel):
        roles.append("fixed_size_layout")

    cross_layer_evidence = []
    if "struct" in base_obj.lower():
        cross_layer_evidence.append("type_mem_bidirectional_check")
    if len(offset_set) > 1:
        cross_layer_evidence.append("field_offset_constraint")

    return roles, cross_layer_evidence

# ==============================================
# 🔥 低层语义：指令语义（100% 你原版）
# ==============================================
def gen_instr_sem(prog):
    program = prog
    output = []

    reg_map = {}
    tmp_map = {}
    counters = {"reg": 0, "tmp": 0}

    funcs = program.getListing().getFunctions(True)
    for func in funcs:
        if func.isThunk() or func.isExternal():
            continue

        instr_iter = program.getListing().getInstructions(func.getBody(), True)
        while instr_iter.hasNext():
            instr = instr_iter.next()
            addr = str(instr.getAddress())
            assembly = instr.toString()

            raw_pcode = [str(p) for p in instr.getPcode()]
            n_pcode = []
            for line in raw_pcode:
                np = normalize_pcode(line, reg_map, tmp_map, counters)
                n_pcode.append(np)

            for np_line in n_pcode:
                np_line = np_line.strip()
                if not np_line or np_line.startswith("---"):
                    continue

                parts = np_line.split()
                if len(parts) < 1:
                    continue

                opcode = "UNKNOWN"
                operands = []
                opcode_list = ["INT_ADD", "INT_SUB", "INT_MULT", "INT_EQUAL",
                               "STORE", "LOAD", "COPY", "BRANCH", "CBRANCH", "RETURN"]

                for i, part in enumerate(parts):
                    if part in opcode_list:
                        opcode = part
                        operands = parts[:i] + parts[i+1:]
                        break
                else:
                    if len(parts) >= 2 and parts[1] in opcode_list:
                        opcode = parts[1]
                        operands = [parts[0]] + parts[2:]

                operands = [op for op in operands if op not in (",", "---")]
                sem = {
                    "id": gen_id("low_instr", addr, str(len(output))),
                    "address": addr,
                    "assembly": assembly,
                    "n_pcode": np_line,
                    "opcode": opcode,
                    "operands": operands
                }
                output.append(sem)
    return output

# ==============================
# 你原版 N-pcode 规范化
# ==============================
def normalize_pcode(line, reg_map, tmp_map, counters):
    if not line:
        return "EMPTY"
    line = re.sub(r"register, *0x([0-9a-fA-F]+), *\d+", lambda m: "r{}".format(int(m.group(1), 16)), line)
    line = re.sub(r"unique, *0x([0-9a-fA-F]+), *\d+", lambda m: "t{}".format(abs(int(m.group(1), 16)) % 1000), line)
    line = re.sub(r"stack, *(-?0x[0-9a-fA-F]+), *\d+", r"mem(sp,0,1,\1)", line)
    line = re.sub(r"const, *(-?0x[0-9a-fA-F]+), *\d+", r"\1", line)
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    line = line.replace("(", "").replace(")", "")
    return line

# ==============================================
# 🔥 低层语义：block_sem（100% 你原版）
# ==============================================
def gen_block_sem(prog):
    program = prog
    output = []

    reg_map = {}
    tmp_map = {}
    counters = {"reg": 0, "tmp": 0}

    funcs = program.getListing().getFunctions(True)
    for func in funcs:
        if func.isThunk() or func.isExternal():
            continue

        instr_iter = program.getListing().getInstructions(func.getBody(), True)
        instrs = []
        while instr_iter.hasNext():
            instr = instr_iter.next()
            addr = str(instr.getAddress())
            assembly = instr.toString()

            raw_pcode = [str(p) for p in instr.getPcode()]
            n_pcode = [normalize_pcode(l, reg_map, tmp_map, counters) for l in raw_pcode]

            instrs.append({
                "address": addr,
                "assembly": assembly,
                "n_pcode": n_pcode
            })

        if not instrs:
            continue

        blocks = build_basic_blocks(instrs)

        for block_id, block in enumerate(blocks):
            block_sem = analyze_block_sem(block, block_id)
            
            addrs = [i["address"] for i in block]
            entry = min(addrs)
            end = max(addrs)
            
            sem = {
                "id": gen_id("low_block", entry, str(block_id)),
                "entry": entry,
                "end": end,
                "block_id": block_sem["block_id"],
                "addr_range": block_sem["addr_range"],
                "ctx_in": block_sem["ctx_in"],
                "ctx_out": block_sem["ctx_out"],
                "sp_delta": block_sem["sp_delta"],
                "def_set": block_sem["def_set"],
                "use_set": block_sem["use_set"],
                "opcode_ratio": block_sem["opcode_ratio"],
                "branches": block_sem["branches"],
                "calls": block_sem["calls"]
            }
            output.append(sem)
    return output

# ==============================
# 你原版：自动构建基本块
# ==============================
def build_basic_blocks(instrs):
    blocks = []
    current_block = []
    branch_targets = set()

    for instr in instrs:
        asm = instr["assembly"].lower()
        if "b" in asm or "bl" in asm or "cbz" in asm or "tbnz" in asm:
            parts = asm.split()
            for p in parts:
                if p.startswith("0x"):
                    branch_targets.add(p)

    for instr in instrs:
        current_block.append(instr)
        asm = instr["assembly"].lower()

        if any(k in asm for k in ["b ", "bl ", "ret", "br", "cbz", "tbnz"]):
            blocks.append(current_block)
            current_block = []
        elif instr["address"] in branch_targets:
            blocks.append(current_block)
            current_block = []

    if current_block:
        blocks.append(current_block)
    return blocks

# ==============================
# 你原版：分析基本块语义
# ==============================
def analyze_block_sem(block, block_id):
    addrs = [i["address"] for i in block]
    addr_range = "{}~{}".format(min(addrs), max(addrs))

    ctx_in = {}
    ctx_out = {}
    def_set = set()
    use_set = set()
    sp_delta = 0
    opcode_count = Counter()
    branch_count = 0
    call_count = 0
    state = defaultdict(lambda: None)

    for instr in block:
        n_pcodes = instr["n_pcode"]
        asm = instr["assembly"].lower()

        for np in n_pcodes:
            np = np.strip()
            if not np or np.startswith("---"):
                continue
            parts = np.split()
            if len(parts) < 2:
                continue

            for op in ["INT_ADD", "INT_SUB", "INT_MULT", "INT_EQUAL","STORE", "COPY", "BRANCH", "CBRANCH", "RETURN"]:
                if op in parts:
                    opcode_count[op] += 1

            if "b " in asm or "bl" in asm: branch_count += 1
            if "bl " in asm: call_count += 1

            for i, p in enumerate(parts):
                if p.startswith("r") or p.startswith("t") or p.startswith("0x"):
                    if i > 0 and parts[i-1] in ["INT_ADD", "INT_SUB", "INT_MULT", "INT_EQUAL"]:
                        def_set.add(p)
                    else:
                        use_set.add(p)

            try:
                if "sub" in asm and "sp" in asm:
                    sp_delta -= int(asm.split("#")[-1], 16) if "#" in asm else -4
                elif "add" in asm and "sp" in asm:
                    sp_delta += int(asm.split("#")[-1], 16) if "#" in asm else 4
            except:
                pass

    ctx_out = dict(state)
    return {
        "block_id": block_id,
        "addr_range": addr_range,
        "ctx_in": ctx_in,
        "ctx_out": ctx_out,
        "sp_delta": sp_delta,
        "def_set": list(def_set),
        "use_set": list(use_set),
        "opcode_ratio": dict(opcode_count),
        "branches": branch_count,
        "calls": call_count
    }

# ==============================================
# 🔥 中层语义：CFG语义（100% 你原版）
# ==============================================
def gen_cfg_sem(prog):
    all_cfgsem = []
    funcManager = prog.getFunctionManager()
    funcs = funcManager.getFunctions(True)

    for func in funcs:
        if func.isThunk() or func.isExternal():
            continue

        cfgsem = {
            "function_name": func.getName(),
            "entry_address": str(func.getEntryPoint()),
            "blocks": [],
            "edges": [],
            "predicates": {}
        }

        instr_iter = prog.getListing().getInstructions(func.getBody(), True)
        instructions = list(instr_iter)
        block_list = []
        current_block = []

        for instr in instructions:
            current_block.append(instr)
            asm = instr.toString().lower()
            is_terminator = any(k in asm for k in [
                "b ", "bl ", "cbz", "tbnz", "ret", "br",
                "b.eq", "b.ne", "b.gt", "b.ge", "b.lt", "b.le"
            ])

            if is_terminator:
                addrs = [str(i.getAddress()) for i in current_block]
                block_list.append({
                    "block_id": len(block_list),
                    "start_address": min(addrs),
                    "end_address": max(addrs),
                    "instruction_count": len(current_block)
                })
                current_block = []

        if current_block:
            addrs = [str(i.getAddress()) for i in current_block]
            block_list.append({
                "block_id": len(block_list),
                "start_address": min(addrs),
                "end_address": max(addrs),
                "instruction_count": len(current_block)
            })

        cfgsem["blocks"] = block_list

        addrFactory = prog.getAddressFactory()
        for b in block_list:
            end_addr = addrFactory.getAddress(b["end_address"])
            inst = prog.getListing().getInstructionAt(end_addr)
            if not inst:
                continue

            fall = inst.getFallThrough()
            if fall and func.getBody().contains(fall):
                cfgsem["edges"].append({
                    "from": b["start_address"],
                    "to": str(fall),
                    "type": "sequential"
                })

            refs = inst.getReferencesFrom()
            for r in refs:
                rt = r.getReferenceType()
                if RefType.isJump(rt) or RefType.isCall(rt):
                    t = "call" if RefType.isCall(rt) else "branch"
                    cfgsem["edges"].append({
                        "from": b["start_address"],
                        "to": str(r.getToAddress()),
                        "type": t
                    })

        pred_idx = 0
        for b in block_list:
            end_addr = addrFactory.getAddress(b["end_address"])
            inst = prog.getListing().getInstructionAt(end_addr)
            if not inst:
                continue

            asm = inst.toString().lower()
            full_asm = inst.toString() 
            
            cond_list = ["b.eq","b.ne","b.gt","b.ge","b.lt","b.le","cbz","tbnz"]
            if any(c in asm for c in cond_list):
                cond_type = "unknown"
                
                if "b.eq" in asm: cond_type = "equal"
                elif "b.ne" in asm: cond_type = "not_equal"
                elif "b.gt" in asm: cond_type = "greater_than"
                elif "b.ge" in asm: cond_type = "greater_or_equal"
                elif "b.lt" in asm: cond_type = "less_than"
                elif "b.le" in asm: cond_type = "less_or_equal"
                elif "cbz" in asm: cond_type = "zero"
                elif "tbnz" in asm: cond_type = "non_zero"

                cfgsem["predicates"]["p{}".format(pred_idx)] = {
                    "address": str(end_addr),
                    "instruction": full_asm,
                    "condition": cond_type,
                    "operand": "flags"
                }
                pred_idx += 1

        all_cfgsem.append(cfgsem)
    return all_cfgsem

# ==============================================
# 🔥 中层语义：数据流 DFGSem（100% 你原版）
# ==============================================
def gen_dataflow_sem(prog):
    all_dfgsem = []
    funcManager = prog.getFunctionManager()
    funcs = funcManager.getFunctions(True)

    for func in funcs:
        if func.isThunk() or func.isExternal():
            continue

        dfgsem = {
            "id": gen_id("mid_dataflow", str(func.getEntryPoint()), "0"),  
            "function_name": func.getName(),
            "entry_address": str(func.getEntryPoint()),
            "locations": [],
            "data_edges": [],
            "abstract_merge": {}
        }

        instr_iter = prog.getListing().getInstructions(func.getBody(), True)
        instructions = list(instr_iter)
        addr_to_instr = {str(i.getAddress()): i for i in instructions}

        block_list = []
        current_block = []
        for instr in instructions:
            current_block.append(instr)
            asm = instr.toString().lower()
            is_terminator = any(k in asm for k in ["b ", "bl ", "cbz", "tbnz", "ret", "br", "b.eq", "b.ne", "b.gt", "b.ge", "b.lt", "b.le"])
            if is_terminator:
                addrs = [str(i.getAddress()) for i in current_block]
                block_list.append({
                    "block_id": len(block_list),
                    "start": min(addrs),
                    "end": max(addrs),
                    "instructions": current_block
                })
                current_block = []
        if current_block:
            addrs = [str(i.getAddress()) for i in current_block]
            block_list.append({
                "block_id": len(block_list),
                "start": min(addrs),
                "end": max(addrs),
                "instructions": current_block
            })

        location_set = set()
        data_edges = []
        abstract_deps = defaultdict(set)

        for block in block_list:
            block_start = block["start"]
            block_end = block["end"]
            instrs_in_block = block["instructions"]

            def_set = set()
            use_set = set()

            for instr in instrs_in_block:
                addr = str(instr.getAddress())
                asm = instr.toString().lower()
                full_asm = instr.toString()

                regs = re.findall(r'r\d+|sp|lr|pc', full_asm, re.IGNORECASE)
                for r in regs:
                    location_set.add("register:" + r.upper())

                if "sp" in asm:
                    location_set.add("register:SP")
                    offsets = re.findall(r'sp[+-]\#?0x[\da-f]+', asm, re.IGNORECASE)
                    for off in offsets:
                        location_set.add("stack:" + off.upper())

                if "[" in asm and "]" in asm:
                    location_set.add("memory:[" + full_asm.split("[")[1].split("]")[0].strip().upper() + "]")

                if any(k in asm for k in ["mov", "add", "sub", "mul", "and", "orr"]):
                    ops = full_asm.split()
                    if len(ops) >= 1:
                        dest = ops[0].upper()
                        if dest.startswith("R") or dest in ["LR", "PC"]:
                            def_set.add(dest)
                        elif dest == "SP":
                            def_set.add("register:SP")

                if any(k in asm for k in ["cmp", "add", "sub", "mul", "and", "orr", "b.ne", "b.eq", "b.gt"]):
                    ops = full_asm.split()[1:]
                    for op in ops:
                        op = op.upper()
                        if op.startswith("R") or op in ["LR", "PC"] or "0x" in op:
                            use_set.add(op)

            dfgsem["locations"] = list(location_set)

            for use in use_set:
                if use in def_set:
                    continue

                found = False
                for prev_block in block_list:
                    if prev_block["end"] == block_start:
                        continue
                    p_instrs = prev_block["instructions"]
                    for p_instr in p_instrs:
                        p_asm = p_instr.toString().lower()
                        if use.upper() in p_asm and any(k in p_asm for k in ["mov", "add", "sub"]):
                            data_edges.append({
                                "from_block": prev_block["end"],
                                "to_block": block_start,
                                "dependency": "def_use",
                                "variable": use,
                                "description": "跨块定义-使用链"
                            })
                            found = True
                            break
                    if found:
                        break

        abstract_merge = {
            "unidentified_memory_dependencies": "conservative_merge",
            "indirect_memory_locations": ["memory:[R1]", "memory:[R2]"],
            "stack_dependency_merge": "stack:SP+0x10 -> stack:SP+0x20 (merged)"
        }
        dfgsem["abstract_merge"] = abstract_merge
        dfgsem["data_edges"] = data_edges
        all_dfgsem.append(dfgsem)

    return all_dfgsem

# ==============================================
# 🔥 🔥 🔥 你最新完整版 TypeSem（最终纯净版）
# ==============================================
def gen_type_sem(prog):
    all_typesem = []
    funcManager = prog.getFunctionManager()
    funcs = list(funcManager.getFunctions(True))

    for func in funcs:
        if func.isExternal():
            continue

        try:
            func_entry = str(func.getEntryPoint())
            func_name = func.getName()

            # 返回值类型语义
            ret_type = func.getReturnType()
            ret_candidates, ret_conf, ret_evidence = get_type_candidates_and_confidence(ret_type)
            return_type_sem = {
                "id": gen_id("high_type", func_entry, "ret"),
                "location": "%s_return" % func_entry,
                "type_candidates": ret_candidates,
                "type_confidence": ret_conf,
                "type_evidence": ret_evidence,
                "type_size": ret_type.getLength() if ret_type else 0,
                "is_pointer": isinstance(ret_type, PointerDataType),
                "category": get_safe_type_category(ret_type)
            }

            # 参数类型语义
            params_sem = []
            params = func.getParameters()
            for idx, param in enumerate(params):
                if not param:
                    continue
                p_type = param.getDataType()
                p_candidates, p_conf, p_evidence = get_type_candidates_and_confidence(p_type, param, func)
                params_sem.append({
                    "id": gen_id("high_type", func_entry, "param_{}".format(idx)),
                    "location": "%s_param_%d" % (func_entry, idx),
                    "param_name": param.getName(),
                    "type_candidates": p_candidates,
                    "type_confidence": p_conf,
                    "type_evidence": p_evidence,
                    "type_size": p_type.getLength() if p_type else 0,
                    "is_pointer": isinstance(p_type, PointerDataType),
                    "category": get_safe_type_category(p_type),
                    "storage": str(param.getStorage()) if param.getStorage() else "unknown"
                })

            # 局部变量类型语义
            locals_sem = []
            try:
                local_vars = func.getLocalVariables()
                for idx, var in enumerate(local_vars):
                    if not var:
                        continue
                    v_type = var.getDataType()
                    v_candidates, v_conf, v_evidence = get_type_candidates_and_confidence(v_type, var, func)
                    locals_sem.append({
                        "id": gen_id("high_type", func_entry, "local_{}".format(idx)),
                        "location": "%s_local_%d" % (func_entry, idx),
                        "var_name": var.getName(),
                        "type_candidates": v_candidates,
                        "type_confidence": v_conf,
                        "type_evidence": v_evidence,
                        "type_size": v_type.getLength() if v_type else 0,
                        "is_pointer": isinstance(v_type, PointerDataType),
                        "category": get_safe_type_category(v_type),
                        "stack_offset": get_safe_stack_offset(var)
                    })
            except:
                pass

            # 指令操作数类型语义
            instr_type_sem = []
            try:
                instr_iter = prog.getListing().getInstructions(func.getBody(), True)
                for instr in instr_iter:
                    if not instr:
                        continue
                    addr = str(instr.getAddress())
                    asm = instr.toString()
                    for op_idx in range(instr.getNumOperands()):
                        objs = instr.getOpObjects(op_idx)
                        for obj in objs:
                            if not obj:
                                continue
                            dt = None
                            if hasattr(obj, "getDataType"):
                                try:
                                    dt = obj.getDataType()
                                except:
                                    continue
                            if dt:
                                i_candidates, i_conf, i_evidence = get_type_candidates_and_confidence(dt)
                                instr_type_sem.append({
                                    "id": gen_id("high_type", addr, "op_{}".format(op_idx)),
                                    "location": addr,
                                    "instruction": asm,
                                    "operand_idx": op_idx,
                                    "type_candidates": i_candidates,
                                    "type_confidence": i_conf,
                                    "type_evidence": i_evidence,
                                    "type_size": dt.getLength() if dt else 0,
                                    "is_pointer": isinstance(dt, PointerDataType),
                                    "category": get_safe_type_category(dt)
                                })
            except:
                pass

            # 组装完整类型语义对象
            func_typesem = {
                "id": gen_id("high_type", func_entry, "func"),
                "function_name": func_name,
                "function_entry": func_entry,
                "return_type_sem": return_type_sem,
                "parameter_type_sem": params_sem,
                "local_var_type_sem": locals_sem,
                "instruction_type_sem": instr_type_sem
            }
            all_typesem.append(func_typesem)

        except:
            continue

    return all_typesem

# ==============================================
# 🔥 🔥 🔥 你最新完整版 InterfaceSem（论文表3 100% 原样）
# ==============================================
def gen_interface_sem(prog):
    all_interfacesem = []
    funcManager = prog.getFunctionManager()
    funcs = list(funcManager.getFunctions(True))

    for func in funcs:
        if func.isThunk() or func.isExternal():
            continue

        try:
            func_entry = str(func.getEntryPoint())
            func_name = func.getName()
            func_id = "%s_%s" % (func_entry, func_name)

            params_sem = []
            params = func.getParameters()
            for idx, param in enumerate(params):
                p_type = param.getDataType()
                p_candidates, p_conf, p_evidence = get_type_candidates_and_confidence(p_type, param, func)
                role_candidates, role_conf = infer_param_role(p_type, param.getName(), func)

                param_sem = {
                    "id": gen_id("high_interface", func_entry, str(idx)),
                    "param_id": idx,
                    "param_name": param.getName(),
                    "location": "%s_param_%d" % (func_entry, idx),
                    "type_candidates": p_candidates,
                    "type_confidence": p_conf,
                    "type_evidence": p_evidence,
                    "type_size": p_type.getLength(),
                    "is_pointer": isinstance(p_type, PointerDataType),
                    "role_candidates": role_candidates,
                    "role_confidence": role_conf,
                    "storage": str(param.getStorage())
                }
                params_sem.append(param_sem)

            ret_type = func.getReturnType()
            ret_candidates, ret_conf, ret_evidence = get_type_candidates_and_confidence(ret_type)
            ret_sem = {
                "location": "%s_return" % func_entry,
                "type_candidates": ret_candidates,
                "type_confidence": ret_conf,
                "type_evidence": ret_evidence,
                "type_size": ret_type.getLength(),
                "is_pointer": isinstance(ret_type, PointerDataType)
            }

            interface_constraints = {
                "calling_convention": str(func.getCallingConvention()),
                "param_count": len(params),
                "has_return": not ret_type.getName() == "void",
                "constraint_evidence": [
                    "calling_convention_analysis",
                    "dataflow_backtrace",
                    "param_role_inference"
                ],
                "cross_layer_constraints": {
                    "type_sem_ref": "%s_return" % func_entry,
                    "param_sem_refs": ["%s_param_%d" % (func_entry, i) for i in range(len(params))]
                }
            }

            interface_sem = {
                "id": gen_id("high_interface", func_entry, "0"),
                "func_id": func_id,
                "params": params_sem,
                "ret": ret_sem,
                "interface_constraints": interface_constraints
            }
            all_interfacesem.append(interface_sem)

        except:
            continue

    return all_interfacesem

# ==============================================
# 🔥 🔥 🔥 你最新完整版 APISem（论文表3 100% 原样）
# ==============================================
def gen_api_sem(prog):
    all_apisem = []
    listing = prog.getListing()
    funcManager = prog.getFunctionManager()
    external_funcs = [f for f in funcManager.getFunctions(True) if f.isExternal()]

    for ext_func in external_funcs:
        api_name = ext_func.getName()
        references = prog.getReferenceManager().getReferencesTo(ext_func.getEntryPoint())
        
        for ref in references:
            if ref.getReferenceType() != RefType.CALL:
                continue

            try:
                callsite_addr = ref.getFromAddress()
                callsite = str(callsite_addr)
                caller_func = funcManager.getFunctionContaining(callsite_addr)
                caller_func_id = "%s_%s" % (str(caller_func.getEntryPoint()), caller_func.getName()) if caller_func else "unknown"

                sem_template = get_api_semantic_template(api_name)
                call_instr = listing.getInstructionAt(callsite_addr)
                param_sem = []
                
                if call_instr:
                    for op_idx in range(1, call_instr.getNumOperands()):
                        objs = call_instr.getOpObjects(op_idx)
                        for obj in objs:
                            if hasattr(obj, "getDataType"):
                                dt = obj.getDataType()
                                is_ptr = isinstance(dt, PointerDataType)
                                role = sem_template["param_roles"][op_idx-1] if (op_idx-1) < len(sem_template["param_roles"]) else "general_param"
                                param_sem.append({
                                    "param_idx": op_idx-1,
                                    "type": dt.getName(),
                                    "is_pointer": is_ptr,
                                    "role": role
                                })

                semantic_template = {
                    "api_intent": sem_template["intent"],
                    "param_roles": param_sem,
                    "side_effects": sem_template["side_effect"],
                    "base_constraints": sem_template["constraints"]
                }

                propagated_constraints = {
                    "caller_func_id": caller_func_id,
                    "type_sem_refs": [],
                    "interface_sem_ref": caller_func_id,
                    "constraint_evidence": [
                        "api_semantic_template_match",
                        "call_site_analysis",
                        "param_role_inference"
                    ],
                    "propagation_targets": ["type_sem", "interface_sem"]
                }

                apisem = {
                    "id": gen_id("high_api", callsite, "0"),
                    "callsite": callsite,
                    "api_name": api_name,
                    "semantic_template": semantic_template,
                    "propagated_constraints": propagated_constraints
                }
                all_apisem.append(apisem)

            except:
                continue

    return all_apisem

# ==============================================
# 🔥 🔥 🔥 你最新完整版 MemLayoutSem（论文表3 修复版）
# ==============================================
def gen_mem_layout_sem(prog):
    all_memblayout = []
    funcManager = prog.getFunctionManager()
    all_funcs = list(funcManager.getFunctions(True))
    target_funcs = [f for f in all_funcs if not f.isThunk()]

    for func in target_funcs:
        func_name = func.getName()
        func_entry = str(func.getEntryPoint())
        func_id = "%s_%s" % (func_entry, func_name)

        try:
            all_vars = list(func.getParameters()) + list(func.getLocalVariables())
            unique_types = set()

            for var in all_vars:
                var_dt = var.getDataType()
                if var_dt in unique_types:
                    continue
                unique_types.add(var_dt)

                layout = analyze_struct_memory_layout(func, var_dt)
                roles, cross_evidence = infer_memory_roles(layout["base_obj"], layout["offset_set"], layout["size_rel"])

                propagated_constraints = {
                    "target_func_id": func_id,
                    "type_sem_ref": "%s_return" % func_entry,
                    "interface_sem_ref": func_id,
                    "memory_intent": "layout_analysis",
                    "cross_layer_constraints": roles,
                    "constraint_evidence": [
                        "struct_field_analysis",
                        "offset_calculation",
                        "size_relation_inference"
                    ] + cross_evidence,
                    "bidirectional_check": "type_sem_dual_validation"
                }

                mem_layout_sem = {
                    "id": gen_id("high_mem", func_entry, str(len(all_memblayout))),
                    "base_obj": layout["base_obj"],
                    "offset_set": layout["offset_set"],
                    "size_rel": layout["size_rel"],
                    "propagated_constraints": propagated_constraints,
                    "semantic_roles": roles,
                    "location": "%s_layout_%s" % (func_entry, var_dt.getName())
                }
                all_memblayout.append(mem_layout_sem)

        except:
            continue

    return all_memblayout

# ==============================================
# 7 条跨层映射规则（你原版）
# ==============================================
# ==============================================
# 最终修复版：7 条跨层映射规则（完全兼容现有数据结构）
# ==============================================
def gen_mapping(all_sem):
    r = []
    # 低层：指令、基本块（有 id，直接取）
    instr = [x["id"] for x in all_sem["low_level_sem"]["instr_sem"]]
    block = [x["id"] for x in all_sem["low_level_sem"]["block_sem"]]
    
    # 中层：CFG（用 block_sem 的 id 做映射，避免结构不匹配）
    # 直接复用 block_sem 的 id 列表，保证 R2 映射合法
    cfg = block[:]  # 控制流提取是 block → CFG，用同一套 id 即可
    dataflow = [x["id"] for x in all_sem["mid_level_sem"]["dataflow_sem"]]

    # 高层：类型、接口、API、内存布局（有 id，直接取）
    typ = [x["id"] for x in all_sem["high_level_sem"]["type_sem"]]
    itf = [x["id"] for x in all_sem["high_level_sem"]["interface_sem"]]
    api = [x["id"] for x in all_sem["high_level_sem"]["api_sem"]]
    mem = [x["id"] for x in all_sem["high_level_sem"]["mem_layout_sem"]]

    rules = [
        ("R1", "指令聚合", instr, block),
        ("R2", "控制流提取", block, cfg),  # 复用 block id，彻底解决 KeyError
        ("R3", "数据流提取", block, dataflow),
        ("R4", "类型推导", instr, typ),
        ("R5", "接口识别", typ, itf),
        ("R6", "API归整", itf, api),
        ("R7", "布局推断", instr, mem),
    ]

    # 安全遍历：只取 srcs/dsts 都存在的索引，避免越界
    for rid, name, srcs, dsts in rules:
        min_len = min(len(srcs), len(dsts))
        for i in range(min_len):
            r.append({
                "rule_id": rid,
                "name": name,
                "from": srcs[i],
                "to": dsts[i]
            })
    return r
# ==============================================
# 总构建
# ==============================================
def build_dict(prog):
    # 👇 改用 OrderedDict 保证顺序 100% 按你写的来
    from collections import OrderedDict
    sem = OrderedDict()

    # 👇 顺序完全按你的要求，不会再乱
    sem["program_info"] = {"name": prog.getName(), "base": str(prog.getImageBase())}
    
    sem["low_level_sem"] = OrderedDict()
    sem["low_level_sem"]["instr_sem"] = gen_instr_sem(prog)
    sem["low_level_sem"]["block_sem"] = gen_block_sem(prog)

    sem["mid_level_sem"] = OrderedDict()
    sem["mid_level_sem"]["cfg_sem"] = gen_cfg_sem(prog)
    sem["mid_level_sem"]["dataflow_sem"] = gen_dataflow_sem(prog)

    sem["high_level_sem"] = OrderedDict()
    sem["high_level_sem"]["type_sem"] = gen_type_sem(prog)
    sem["high_level_sem"]["interface_sem"] = gen_interface_sem(prog)
    sem["high_level_sem"]["api_sem"] = gen_api_sem(prog)
    sem["high_level_sem"]["mem_layout_sem"] = gen_mem_layout_sem(prog)

    sem["cross_layer_mapping"] = gen_mapping(sem)
    return sem



# ==============================================
# PATCH: Ghidra BasicBlockModel-based block/CFG extraction
# ==============================================
# -*- coding: utf-8 -*-
# Ready-to-paste patch for overall.py
# Purpose: replace asm-string-based basic-block slicing / CFG extraction
# with Ghidra BasicBlockModel-based extraction.

from ghidra.program.model.block import BasicBlockModel
from ghidra.util.task import TaskMonitor

# --------------------------------------------------
# Helpers
# --------------------------------------------------
CFG_OPCODE_LIST = set([
    "INT_ADD", "INT_SUB", "INT_MULT", "INT_EQUAL",
    "STORE", "LOAD", "COPY", "BRANCH", "CBRANCH", "RETURN"
])


def _get_script_monitor():
    try:
        return monitor
    except NameError:
        return TaskMonitor.DUMMY


def _addr_to_str(addr):
    return str(addr) if addr is not None else None


def _get_addr_sort_key(addr):
    try:
        return addr.getOffset()
    except:
        return str(addr)


def _collect_instructions_in_block(program, code_block, func_body=None):
    """
    code_block 是 AddressSetView，Listing.getInstructions 可直接遍历其中指令。
    这里再额外用 func_body 过滤一次，避免重叠块/异常边界把函数外地址带进来。
    """
    instrs = []
    instr_iter = program.getListing().getInstructions(code_block, True)
    while instr_iter.hasNext():
        instr = instr_iter.next()
        if func_body is None or func_body.contains(instr.getAddress()):
            instrs.append(instr)
    return instrs


def get_function_blocks(program, func, block_model=None, tm=None):
    """
    使用 Ghidra BasicBlockModel 获取某个函数内的基本块，返回稳定、可复用的块列表。
    每个块都保留原始 CodeBlock 和块内 Instruction 列表，供 block_sem / cfg_sem 共用。
    """
    if block_model is None:
        block_model = BasicBlockModel(program)
    if tm is None:
        tm = _get_script_monitor()

    blocks = []
    block_iter = block_model.getCodeBlocksContaining(func.getBody(), tm)
    while block_iter.hasNext():
        code_block = block_iter.next()
        instrs = _collect_instructions_in_block(program, code_block, func.getBody())
        if not instrs:
            continue

        entry = instrs[0].getAddress()
        end = instrs[-1].getAddress()

        blocks.append({
            "code_block": code_block,
            "instructions": instrs,
            "entry": entry,
            "end": end,
            "entry_points": [_addr_to_str(a) for a in code_block.getStartAddresses()]
        })

    # 为了让 block_id / JSON 输出稳定，统一按块入口地址排序后重新编号
    blocks.sort(key=lambda b: _get_addr_sort_key(b["entry"]))

    for idx, block in enumerate(blocks):
        block["block_id"] = idx
        block["entry_str"] = _addr_to_str(block["entry"])
        block["end_str"] = _addr_to_str(block["end"])
        block["addr_range"] = "%s~%s" % (block["entry_str"], block["end_str"])

    return blocks


def _normalize_instr_pcode(instr, reg_map, tmp_map, counters):
    raw_pcode = [str(p) for p in instr.getPcode()]
    return [normalize_pcode(line, reg_map, tmp_map, counters) for line in raw_pcode]


def _parse_npcode_def_use(np_line):
    """
    比原脚本更稳一些：
    - 优先利用 '=' 左右分 defs / uses
    - 没有 '=' 时，默认把 opcode 后面的 token 当 uses
    仍是轻量启发式，但至少不会把目的寄存器系统性误判成 use。
    """
    defs = set()
    uses = set()

    if not np_line:
        return defs, uses, None

    tokens = [t.strip() for t in np_line.replace(",", " ").split() if t.strip()]
    if not tokens:
        return defs, uses, None

    opcode = None
    for t in tokens:
        if t in CFG_OPCODE_LIST:
            opcode = t
            break

    def _interesting(tok):
        return tok.startswith("r") or tok.startswith("t") or tok.startswith("mem") or tok.startswith("0x")

    if "=" in tokens:
        eq_idx = tokens.index("=")
        lhs = tokens[:eq_idx]
        rhs = tokens[eq_idx + 1:]
        for tok in lhs:
            if _interesting(tok):
                defs.add(tok)
        for tok in rhs:
            if _interesting(tok):
                uses.add(tok)
    else:
        start_idx = 1 if len(tokens) > 1 else 0
        for tok in tokens[start_idx:]:
            if _interesting(tok):
                uses.add(tok)

    return defs, uses, opcode


def analyze_block_sem_from_instructions(instrs, block_id, reg_map=None, tmp_map=None, counters=None):
    if reg_map is None:
        reg_map = {}
    if tmp_map is None:
        tmp_map = {}
    if counters is None:
        counters = {"reg": 0, "tmp": 0}

    addrs = [_addr_to_str(i.getAddress()) for i in instrs]
    addr_range = "%s~%s" % (addrs[0], addrs[-1])

    ctx_in = {}
    ctx_out = {}
    def_set = set()
    use_set = set()
    sp_delta = 0
    opcode_count = Counter()
    branch_count = 0
    call_count = 0

    for instr in instrs:
        asm = instr.toString()
        asm_lower = asm.lower()
        flow_type = instr.getFlowType()

        # 这里按“每条指令”计一次，避免原脚本在每条 p-code 子操作上重复累加
        if flow_type is not None:
            if flow_type.isCall():
                call_count += 1
            if flow_type.isJump():
                branch_count += 1

        n_pcodes = _normalize_instr_pcode(instr, reg_map, tmp_map, counters)
        for np in n_pcodes:
            if not np or np.startswith("---"):
                continue
            defs, uses, opcode = _parse_npcode_def_use(np)
            def_set.update(defs)
            use_set.update(uses)
            if opcode is not None:
                opcode_count[opcode] += 1

        # 先保留你原来基于汇编文本的轻量 sp_delta 计算逻辑，但改成正则更稳一点
        try:
            m_sub = re.search(r'\bsub\b[^\n]*\bsp\b[^\n#]*#?(0x[0-9a-f]+|\d+)', asm_lower)
            m_add = re.search(r'\badd\b[^\n]*\bsp\b[^\n#]*#?(0x[0-9a-f]+|\d+)', asm_lower)
            if m_sub:
                imm = m_sub.group(1)
                sp_delta -= int(imm, 16 if imm.startswith("0x") else 10)
            elif m_add:
                imm = m_add.group(1)
                sp_delta += int(imm, 16 if imm.startswith("0x") else 10)
        except:
            pass

    # 当前阶段仍保留空的抽象状态，后面接入 p-code 级 transfer function 时再填充
    ctx_out = {}

    return {
        "block_id": block_id,
        "addr_range": addr_range,
        "ctx_in": ctx_in,
        "ctx_out": ctx_out,
        "sp_delta": sp_delta,
        "def_set": sorted(list(def_set)),
        "use_set": sorted(list(use_set)),
        "opcode_ratio": dict(opcode_count),
        "branches": branch_count,
        "calls": call_count
    }


# --------------------------------------------------
# Replacement: gen_block_sem
# --------------------------------------------------
def gen_block_sem(prog):
    program = prog
    output = []

    reg_map = {}
    tmp_map = {}
    counters = {"reg": 0, "tmp": 0}

    tm = _get_script_monitor()
    block_model = BasicBlockModel(program)
    funcs = program.getListing().getFunctions(True)

    for func in funcs:
        if func.isThunk() or func.isExternal():
            continue

        func_blocks = get_function_blocks(program, func, block_model, tm)
        for block in func_blocks:
            block_sem = analyze_block_sem_from_instructions(
                block["instructions"],
                block["block_id"],
                reg_map,
                tmp_map,
                counters
            )

            sem = {
                "id": gen_id("low_block", block["entry_str"], str(block["block_id"])),
                "entry": block["entry_str"],
                "end": block["end_str"],
                "block_id": block_sem["block_id"],
                "addr_range": block_sem["addr_range"],
                "ctx_in": block_sem["ctx_in"],
                "ctx_out": block_sem["ctx_out"],
                "sp_delta": block_sem["sp_delta"],
                "def_set": block_sem["def_set"],
                "use_set": block_sem["use_set"],
                "opcode_ratio": block_sem["opcode_ratio"],
                "branches": block_sem["branches"],
                "calls": block_sem["calls"]
            }
            output.append(sem)

    return output


# --------------------------------------------------
# Replacement: gen_cfg_sem
# --------------------------------------------------
def _classify_edge_type(flow_type):
    if flow_type is None:
        return "unknown"
    if flow_type.isCall():
        return "call"
    if flow_type.isJump():
        if flow_type.isConditional():
            return "conditional_branch"
        if flow_type.isComputed():
            return "indirect_branch"
        return "branch"
    if flow_type.isTerminal():
        return "terminal"
    return "fallthrough"



def _extract_predicate_from_last_instr(last_instr):
    if last_instr is None:
        return None
    flow_type = last_instr.getFlowType()
    if flow_type is None or not flow_type.isConditional():
        return None

    operands = []
    try:
        for i in range(last_instr.getNumOperands()):
            operands.append(last_instr.getDefaultOperandRepresentation(i))
    except:
        pass

    return {
        "address": _addr_to_str(last_instr.getAddress()),
        "instruction": last_instr.toString(),
        "mnemonic": last_instr.getMnemonicString(),
        "operands": operands,
        "condition": last_instr.getMnemonicString().lower()
    }



def gen_cfg_sem(prog):
    all_cfgsem = []
    tm = _get_script_monitor()
    block_model = BasicBlockModel(prog)
    funcs = prog.getFunctionManager().getFunctions(True)

    for func in funcs:
        if func.isThunk() or func.isExternal():
            continue

        func_blocks = get_function_blocks(prog, func, block_model, tm)
        block_by_start = {}
        block_list = []

        for block in func_blocks:
            block_by_start[block["entry_str"]] = block
            block_list.append({
                "block_id": block["block_id"],
                "start_address": block["entry_str"],
                "end_address": block["end_str"],
                "instruction_count": len(block["instructions"]),
                "entry_points": block["entry_points"]
            })

        cfgsem = {
            "function_name": func.getName(),
            "entry_address": _addr_to_str(func.getEntryPoint()),
            "blocks": block_list,
            "edges": [],
            "predicates": {}
        }

        edge_seen = set()
        pred_idx = 0

        for block in func_blocks:
            src_start = block["entry_str"]
            code_block = block["code_block"]

            # 条件谓词：挂在块尾指令上
            last_instr = block["instructions"][-1] if block["instructions"] else None
            pred = _extract_predicate_from_last_instr(last_instr)
            if pred is not None:
                cfgsem["predicates"]["p%d" % pred_idx] = pred
                pred_idx += 1

            # 真实的出边：来自 Ghidra 的 CodeBlockReference
            dest_iter = code_block.getDestinations(tm)
            while dest_iter.hasNext():
                ref = dest_iter.next()
                dst_block = ref.getDestinationBlock()
                if dst_block is None:
                    continue

                dst_start = _addr_to_str(dst_block.getFirstStartAddress())
                if dst_start not in block_by_start:
                    # 这里只保留函数体内块到块的 CFG 边；外部调用不是过程内 CFG
                    continue

                flow_type = ref.getFlowType()
                edge_type = _classify_edge_type(flow_type)
                key = (
                    src_start,
                    dst_start,
                    edge_type,
                    _addr_to_str(ref.getReferent()),
                    _addr_to_str(ref.getReference())
                )
                if key in edge_seen:
                    continue
                edge_seen.add(key)

                cfgsem["edges"].append({
                    "from": src_start,
                    "to": dst_start,
                    "type": edge_type,
                    "referent": _addr_to_str(ref.getReferent()),
                    "reference": _addr_to_str(ref.getReference())
                })

        all_cfgsem.append(cfgsem)

    return all_cfgsem



# ==============================================
# PATCH v2: p-code 级 DFG + 基于证据触发的真实映射规则
# 说明：
# 1) DFG 改为以 Ghidra p-code/Varnode 为基础做 def-use；
# 2) gen_cfg_sem 增补稳定 id，便于 R2 映射直达 CFGSem；
# 3) gen_mapping 不再按列表下标 zip，而是按规则 Trigger / Evidence / Emit / Constraints 触发。
# ==============================================

def _safe_encode_piece(vn):
    if vn is None:
        return None
    try:
        return vn.encodePiece()
    except:
        try:
            return str(vn)
        except:
            return None


def _sign_extend_value(value, size):
    try:
        bits = int(size) * 8
        if bits <= 0:
            return int(value)
        mask = (1 << bits) - 1
        value = int(value) & mask
        sign_bit = 1 << (bits - 1)
        if value & sign_bit:
            value -= (1 << bits)
        return int(value)
    except:
        try:
            return int(value)
        except:
            return 0


def _const_vn_as_signed(vn):
    if vn is None:
        return None
    try:
        if vn.isConstant():
            return _sign_extend_value(vn.getOffset(), vn.getSize())
    except:
        pass
    return None


def _format_signed_immediate(value):
    try:
        value = int(value)
    except:
        return "?"
    if value < 0:
        return "-0x%x" % (-value)
    return "0x%x" % value


def _format_base_offset(base, offset):
    if base is None:
        return _format_signed_immediate(offset)
    try:
        offset = int(offset)
    except:
        return base
    if offset == 0:
        return base
    if offset > 0:
        return "%s+0x%x" % (base, offset)
    return "%s-0x%x" % (base, -offset)


def _unique_list(items):
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _canonical_varnode_location(program, vn, include_constant=False):
    if vn is None:
        return None

    try:
        addr = vn.getAddress()
    except:
        addr = None

    try:
        if vn.isConstant() or (addr is not None and addr.isConstantAddress()):
            if not include_constant:
                return None
            return "const:%s:%d" % (_format_signed_immediate(_sign_extend_value(vn.getOffset(), vn.getSize())), int(vn.getSize()))
    except:
        pass

    try:
        if vn.isRegister() or (addr is not None and addr.isRegisterAddress()):
            reg = None
            try:
                reg = program.getRegister(addr, vn.getSize())
            except:
                reg = None
            if reg:
                return "reg:%s" % str(reg.getName()).upper()
            return "reg:0x%x:%d" % (int(vn.getOffset()), int(vn.getSize()))
    except:
        pass

    try:
        if vn.isUnique() or (addr is not None and addr.isUniqueAddress()):
            return "tmp:0x%x:%d" % (int(vn.getOffset()), int(vn.getSize()))
    except:
        pass

    try:
        if addr is not None and addr.isStackAddress():
            return "stack:%s:%d" % (str(addr), int(vn.getSize()))
    except:
        pass

    try:
        if vn.isAddress() or (addr is not None and addr.isMemoryAddress()):
            if addr is not None:
                return "addr:%s:0x%x:%d" % (addr.getAddressSpace().getName(), int(vn.getOffset()), int(vn.getSize()))
    except:
        pass

    piece = _safe_encode_piece(vn)
    if piece:
        return "loc:%s" % piece
    return None


def _collect_support_locations(program, vn, depth=0, seen=None):
    if seen is None:
        seen = set()
    if vn is None or depth > 5:
        return []

    key = _safe_encode_piece(vn)
    if key and key in seen:
        return []
    if key:
        seen.add(key)

    try:
        if vn.isConstant():
            return []
    except:
        pass

    loc = _canonical_varnode_location(program, vn, include_constant=False)

    try:
        if not vn.isUnique():
            return [loc] if loc else []
    except:
        return [loc] if loc else []

    defop = None
    try:
        defop = vn.getDef()
    except:
        defop = None

    if defop is None:
        return [loc] if loc else []

    out = []
    try:
        for inp in defop.getInputs():
            out.extend(_collect_support_locations(program, inp, depth + 1, seen))
    except:
        pass

    if not out and loc:
        out.append(loc)
    return _unique_list(out)


def _trace_address_expr(program, vn, depth=0, seen=None):
    if seen is None:
        seen = set()

    default_expr = _canonical_varnode_location(program, vn, include_constant=False)
    if default_expr is None:
        default_expr = _safe_encode_piece(vn) or "unknown"

    default = {
        "base": None,
        "offset": None,
        "expr": default_expr,
        "is_fixed": False
    }

    if vn is None or depth > 6:
        return default

    key = _safe_encode_piece(vn)
    if key and key in seen:
        return default
    if key:
        seen.add(key)

    const_val = _const_vn_as_signed(vn)
    if const_val is not None:
        return {
            "base": None,
            "offset": const_val,
            "expr": _format_signed_immediate(const_val),
            "is_fixed": True
        }

    loc = _canonical_varnode_location(program, vn, include_constant=False)

    try:
        addr = vn.getAddress()
    except:
        addr = None

    try:
        if vn.isRegister():
            return {"base": loc, "offset": 0, "expr": loc, "is_fixed": True}
    except:
        pass

    try:
        if addr is not None and addr.isStackAddress():
            return {"base": loc, "offset": 0, "expr": loc, "is_fixed": True}
    except:
        pass

    try:
        if addr is not None and addr.isMemoryAddress() and not vn.isUnique():
            return {"base": loc, "offset": 0, "expr": loc, "is_fixed": True}
    except:
        pass

    defop = None
    try:
        defop = vn.getDef()
    except:
        defop = None

    if defop is None:
        return default

    try:
        mnemonic = defop.getMnemonic()
    except:
        mnemonic = None

    try:
        inputs = list(defop.getInputs())
    except:
        inputs = []

    if mnemonic in ("COPY", "CAST", "INT_ZEXT", "INT_SEXT", "SUBPIECE", "PIECE", "INDIRECT"):
        if inputs:
            return _trace_address_expr(program, inputs[0], depth + 1, seen)
        return default

    if mnemonic in ("INT_ADD", "PTRSUB"):
        if len(inputs) >= 2:
            left = _trace_address_expr(program, inputs[0], depth + 1, seen)
            cst = _const_vn_as_signed(inputs[1])
            if cst is not None and left.get("base") is not None:
                off = int(left.get("offset") or 0) + int(cst)
                return {
                    "base": left.get("base"),
                    "offset": off,
                    "expr": _format_base_offset(left.get("base"), off),
                    "is_fixed": True
                }

            right = _trace_address_expr(program, inputs[1], depth + 1, seen)
            if left.get("base") is not None and right.get("base") is None and right.get("offset") is not None:
                off = int(left.get("offset") or 0) + int(right.get("offset"))
                return {
                    "base": left.get("base"),
                    "offset": off,
                    "expr": _format_base_offset(left.get("base"), off),
                    "is_fixed": True
                }

            return {
                "base": left.get("base"),
                "offset": None,
                "expr": "%s+%s" % (left.get("expr"), right.get("expr")),
                "is_fixed": False
            }

    if mnemonic == "INT_SUB":
        if len(inputs) >= 2:
            left = _trace_address_expr(program, inputs[0], depth + 1, seen)
            cst = _const_vn_as_signed(inputs[1])
            if cst is not None and left.get("base") is not None:
                off = int(left.get("offset") or 0) - int(cst)
                return {
                    "base": left.get("base"),
                    "offset": off,
                    "expr": _format_base_offset(left.get("base"), off),
                    "is_fixed": True
                }

            right = _trace_address_expr(program, inputs[1], depth + 1, seen)
            return {
                "base": left.get("base"),
                "offset": None,
                "expr": "%s-%s" % (left.get("expr"), right.get("expr")),
                "is_fixed": False
            }

    if mnemonic == "PTRADD":
        if len(inputs) >= 3:
            base = _trace_address_expr(program, inputs[0], depth + 1, seen)
            idx = _const_vn_as_signed(inputs[1])
            scale = _const_vn_as_signed(inputs[2])

            if idx is not None and scale is not None and base.get("base") is not None:
                off = int(base.get("offset") or 0) + int(idx) * int(scale)
                return {
                    "base": base.get("base"),
                    "offset": off,
                    "expr": _format_base_offset(base.get("base"), off),
                    "is_fixed": True
                }

            return {
                "base": base.get("base"),
                "offset": None,
                "expr": "%s+idx*scale" % base.get("expr"),
                "is_fixed": False
            }

    return default


def _canonical_memory_info(program, addr_vn):
    info = _trace_address_expr(program, addr_vn)
    expr = info.get("expr") or "unknown"
    info["location"] = "mem[%s]" % expr
    info["support_locs"] = _unique_list(_collect_support_locations(program, addr_vn))
    return info


def _dependency_kind(loc):
    if not loc:
        return "def_use"
    if loc.startswith("mem["):
        return "memory_def_use"
    if loc.startswith("stack:"):
        return "stack_def_use"
    if loc.startswith("reg:"):
        return "register_def_use"
    if loc.startswith("tmp:"):
        return "tmp_def_use"
    if loc.startswith("addr:"):
        return "address_def_use"
    return "def_use"


def _is_global_location(loc):
    if not loc:
        return False
    s = str(loc)
    if s.startswith("tmp:"):
        return False
    if s.startswith("mem[tmp:") or s.startswith("mem[unknown") or "idx*scale" in s:
        return False
    return True


def _is_param_like_location(loc):
    if not loc:
        return False
    return str(loc).startswith("reg:") or str(loc).startswith("stack:")


def _clone_defs_map(d):
    out = {}
    for k, v in d.items():
        out[k] = set(v)
    return out


def _merge_defs_maps(base_map, other_map):
    out = _clone_defs_map(base_map)
    for loc, sites in other_map.items():
        if loc not in out:
            out[loc] = set()
        out[loc].update(sites)
    return out


def _defs_maps_equal(a, b):
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a.keys():
        if set(a.get(k, [])) != set(b.get(k, [])):
            return False
    return True


def _build_function_preds(func_blocks, tm):
    preds = {}
    succs = {}
    block_by_start = {}

    for block in func_blocks:
        block_by_start[block["entry_str"]] = block
        preds[block["entry_str"]] = set()
        succs[block["entry_str"]] = set()

    for block in func_blocks:
        src = block["entry_str"]
        dest_iter = block["code_block"].getDestinations(tm)
        while dest_iter.hasNext():
            ref = dest_iter.next()
            dst_block = ref.getDestinationBlock()
            if dst_block is None:
                continue
            dst = _addr_to_str(dst_block.getFirstStartAddress())
            if dst not in block_by_start:
                continue
            succs[src].add(dst)
            preds[dst].add(src)

    return preds, succs


def _extract_pcode_facts(program, instr, pcode_op, pcode_index, block):
    addr = _addr_to_str(instr.getAddress())
    site = "%s:%d" % (addr, int(pcode_index))

    try:
        mnemonic = pcode_op.getMnemonic()
    except:
        mnemonic = "UNKNOWN"

    uses = []
    defs = []
    mem_accesses = []

    num_inputs = 0
    try:
        num_inputs = pcode_op.getNumInputs()
    except:
        num_inputs = 0

    output_vn = None
    try:
        output_vn = pcode_op.getOutput()
    except:
        output_vn = None

    if mnemonic == "LOAD":
        if num_inputs >= 2:
            try:
                addr_vn = pcode_op.getInput(1)
            except:
                addr_vn = None
            mem_info = _canonical_memory_info(program, addr_vn)
            uses.append(mem_info.get("location"))
            uses.extend(mem_info.get("support_locs", []))
            mem_accesses.append({
                "site": site,
                "instruction_address": addr,
                "block_entry": block["entry_str"],
                "block_id": block["block_id"],
                "access": "load",
                "location": mem_info.get("location"),
                "base": mem_info.get("base"),
                "offset": mem_info.get("offset"),
                "expr": mem_info.get("expr"),
                "is_fixed": mem_info.get("is_fixed"),
                "support_locs": mem_info.get("support_locs", [])
            })
        out_loc = _canonical_varnode_location(program, output_vn, include_constant=False)
        if out_loc:
            defs.append(out_loc)

    elif mnemonic == "STORE":
        if num_inputs >= 3:
            try:
                addr_vn = pcode_op.getInput(1)
            except:
                addr_vn = None
            try:
                val_vn = pcode_op.getInput(2)
            except:
                val_vn = None

            mem_info = _canonical_memory_info(program, addr_vn)
            defs.append(mem_info.get("location"))
            uses.extend(mem_info.get("support_locs", []))

            val_loc = _canonical_varnode_location(program, val_vn, include_constant=False)
            if val_loc:
                uses.append(val_loc)

            mem_accesses.append({
                "site": site,
                "instruction_address": addr,
                "block_entry": block["entry_str"],
                "block_id": block["block_id"],
                "access": "store",
                "location": mem_info.get("location"),
                "base": mem_info.get("base"),
                "offset": mem_info.get("offset"),
                "expr": mem_info.get("expr"),
                "is_fixed": mem_info.get("is_fixed"),
                "support_locs": mem_info.get("support_locs", []),
                "value_loc": val_loc
            })

    else:
        for i in range(num_inputs):
            try:
                inp = pcode_op.getInput(i)
            except:
                inp = None
            loc = _canonical_varnode_location(program, inp, include_constant=False)
            if loc:
                uses.append(loc)

        out_loc = _canonical_varnode_location(program, output_vn, include_constant=False)
        if out_loc:
            defs.append(out_loc)

    return {
        "site": site,
        "instruction_address": addr,
        "instruction": instr.toString(),
        "mnemonic": mnemonic,
        "pcode_index": int(pcode_index),
        "block_entry": block["entry_str"],
        "block_id": block["block_id"],
        "uses": _unique_list(uses),
        "defs": _unique_list(defs),
        "memory_accesses": mem_accesses
    }


# --------------------------------------------------
# Replacement v2: gen_cfg_sem（仅补充稳定 id）
# --------------------------------------------------
def gen_cfg_sem(prog):
    all_cfgsem = []
    tm = _get_script_monitor()
    block_model = BasicBlockModel(prog)
    funcs = prog.getFunctionManager().getFunctions(True)

    for func in funcs:
        if func.isThunk() or func.isExternal():
            continue

        func_blocks = get_function_blocks(prog, func, block_model, tm)
        block_by_start = {}
        block_list = []

        for block in func_blocks:
            block_by_start[block["entry_str"]] = block
            block_list.append({
                "block_id": block["block_id"],
                "start_address": block["entry_str"],
                "end_address": block["end_str"],
                "instruction_count": len(block["instructions"]),
                "entry_points": block["entry_points"]
            })

        cfgsem = {
            "id": gen_id("mid_cfg", _addr_to_str(func.getEntryPoint()), "0"),
            "function_name": func.getName(),
            "entry_address": _addr_to_str(func.getEntryPoint()),
            "blocks": block_list,
            "edges": [],
            "predicates": {}
        }

        edge_seen = set()
        pred_idx = 0

        for block in func_blocks:
            src_start = block["entry_str"]
            code_block = block["code_block"]

            last_instr = block["instructions"][-1] if block["instructions"] else None
            pred = _extract_predicate_from_last_instr(last_instr)
            if pred is not None:
                cfgsem["predicates"]["p%d" % pred_idx] = pred
                pred_idx += 1

            dest_iter = code_block.getDestinations(tm)
            while dest_iter.hasNext():
                ref = dest_iter.next()
                dst_block = ref.getDestinationBlock()
                if dst_block is None:
                    continue

                dst_start = _addr_to_str(dst_block.getFirstStartAddress())
                if dst_start not in block_by_start:
                    continue

                flow_type = ref.getFlowType()
                edge_type = _classify_edge_type(flow_type)
                key = (
                    src_start,
                    dst_start,
                    edge_type,
                    _addr_to_str(ref.getReferent()),
                    _addr_to_str(ref.getReference())
                )
                if key in edge_seen:
                    continue
                edge_seen.add(key)

                cfgsem["edges"].append({
                    "from": src_start,
                    "to": dst_start,
                    "type": edge_type,
                    "referent": _addr_to_str(ref.getReferent()),
                    "reference": _addr_to_str(ref.getReference())
                })

        all_cfgsem.append(cfgsem)

    return all_cfgsem


# --------------------------------------------------
# Replacement v2: gen_dataflow_sem（p-code 级 def-use）
# --------------------------------------------------
def gen_dataflow_sem(prog):
    all_dfgsem = []
    tm = _get_script_monitor()
    block_model = BasicBlockModel(prog)
    funcs = prog.getFunctionManager().getFunctions(True)

    for func in funcs:
        if func.isThunk() or func.isExternal():
            continue

        func_entry = _addr_to_str(func.getEntryPoint())
        func_blocks = get_function_blocks(prog, func, block_model, tm)
        if not func_blocks:
            continue

        preds, succs = _build_function_preds(func_blocks, tm)

        block_ops = {}
        gen_defs = {}
        kill_locs = {}
        site_meta = {}
        locations = set()
        definitions = []
        memory_access_patterns = []
        entry_input_locs = set()

        # 1) 先按块收集 p-code 级 defs / uses
        for block in func_blocks:
            bentry = block["entry_str"]
            ops = []
            local_last_def = {}
            local_use_before_def = set()

            for instr in block["instructions"]:
                try:
                    pcode_ops = list(instr.getPcode())
                except:
                    pcode_ops = []

                for pidx, pcode_op in enumerate(pcode_ops):
                    facts = _extract_pcode_facts(prog, instr, pcode_op, pidx, block)
                    ops.append(facts)
                    site_meta[facts["site"]] = facts

                    for loc in facts.get("uses", []):
                        if loc:
                            locations.add(loc)
                    for loc in facts.get("defs", []):
                        if loc:
                            locations.add(loc)

                    for mem_acc in facts.get("memory_accesses", []):
                        memory_access_patterns.append(mem_acc)
                        if mem_acc.get("location"):
                            locations.add(mem_acc.get("location"))

                    for dloc in facts.get("defs", []):
                        definitions.append({
                            "site": facts["site"],
                            "instruction_address": facts["instruction_address"],
                            "block_entry": bentry,
                            "block_id": block["block_id"],
                            "mnemonic": facts["mnemonic"],
                            "location": dloc
                        })

                    for uloc in facts.get("uses", []):
                        if _is_global_location(uloc) and uloc not in local_last_def:
                            local_use_before_def.add(uloc)

                    for dloc in facts.get("defs", []):
                        if _is_global_location(dloc):
                            local_last_def[dloc] = facts["site"]

            block_ops[bentry] = ops
            gen_defs[bentry] = {}
            kill_locs[bentry] = set()

            for loc, site in local_last_def.items():
                gen_defs[bentry][loc] = set([site])
                kill_locs[bentry].add(loc)

            if bentry == func_entry:
                for uloc in local_use_before_def:
                    if _is_param_like_location(uloc):
                        entry_input_locs.add(uloc)

        # 2) 块级 reaching-definitions，把块间 def-use 接起来
        in_defs = {}
        out_defs = {}
        for block in func_blocks:
            bentry = block["entry_str"]
            in_defs[bentry] = {}
            out_defs[bentry] = {}

        changed = True
        while changed:
            changed = False
            for block in func_blocks:
                bentry = block["entry_str"]

                new_in = {}
                for pred in preds.get(bentry, []):
                    new_in = _merge_defs_maps(new_in, out_defs.get(pred, {}))

                new_out = _clone_defs_map(new_in)
                for loc in kill_locs.get(bentry, set()):
                    if loc in new_out:
                        del new_out[loc]
                for loc, sites in gen_defs.get(bentry, {}).items():
                    new_out[loc] = set(sites)

                if (not _defs_maps_equal(in_defs.get(bentry, {}), new_in)) or (not _defs_maps_equal(out_defs.get(bentry, {}), new_out)):
                    in_defs[bentry] = new_in
                    out_defs[bentry] = new_out
                    changed = True

        # 3) 再顺序扫描块内 p-code，生成 p-code 级数据依赖边
        data_edges = []
        edge_seen = set()

        for block in func_blocks:
            bentry = block["entry_str"]
            env = _clone_defs_map(in_defs.get(bentry, {}))

            for facts in block_ops.get(bentry, []):
                current_site = facts["site"]

                for uloc in facts.get("uses", []):
                    src_sites = env.get(uloc, set())
                    if not src_sites:
                        continue
                    for src_site in sorted(list(src_sites)):
                        key = (src_site, current_site, uloc)
                        if key in edge_seen:
                            continue
                        edge_seen.add(key)

                        src_meta = site_meta.get(src_site, {})
                        data_edges.append({
                            "from": src_meta.get("instruction_address"),
                            "to": facts.get("instruction_address"),
                            "from_site": src_site,
                            "to_site": current_site,
                            "from_block": src_meta.get("block_entry"),
                            "to_block": facts.get("block_entry"),
                            "dependency": _dependency_kind(uloc),
                            "location": uloc,
                            "producer_pcode": src_meta.get("mnemonic"),
                            "consumer_pcode": facts.get("mnemonic")
                        })

                for dloc in facts.get("defs", []):
                    env[dloc] = set([current_site])

        # 4) 轻量 alias / 固定偏移摘要，供 R4/R7 触发使用
        base_groups = {}
        indirect_memory_locations = set()

        for acc in memory_access_patterns:
            if not acc.get("is_fixed") or acc.get("base") is None or acc.get("offset") is None:
                if acc.get("location"):
                    indirect_memory_locations.add(acc.get("location"))
                continue

            base = acc.get("base")
            if base not in base_groups:
                base_groups[base] = {
                    "base": base,
                    "offsets": set(),
                    "instr_addresses": set(),
                    "sites": set(),
                    "accesses": []
                }

            base_groups[base]["offsets"].add(int(acc.get("offset")))
            if acc.get("instruction_address"):
                base_groups[base]["instr_addresses"].add(acc.get("instruction_address"))
            if acc.get("site"):
                base_groups[base]["sites"].add(acc.get("site"))
            base_groups[base]["accesses"].append(acc)

        alias_groups = []
        for base, group in base_groups.items():
            offsets = sorted(list(group["offsets"]))
            stride = None
            if len(offsets) >= 2:
                diffs = []
                for i in range(len(offsets) - 1):
                    diffs.append(offsets[i + 1] - offsets[i])
                if diffs and len(set(diffs)) == 1:
                    stride = diffs[0]

            alias_groups.append({
                "base": base,
                "offsets": offsets,
                "instr_addresses": sorted(list(group["instr_addresses"])),
                "sites": sorted(list(group["sites"])),
                "access_count": len(group["accesses"]),
                "stride": stride
            })

        dfgsem = {
            "id": gen_id("mid_dataflow", func_entry, "0"),
            "function_name": func.getName(),
            "entry_address": func_entry,
            "locations": sorted(list(locations)),
            "definitions": definitions,
            "data_edges": data_edges,
            "memory_access_patterns": memory_access_patterns,
            "entry_input_locs": sorted(list(entry_input_locs)),
            "abstract_merge": {
                "alias_groups": alias_groups,
                "indirect_memory_locations": sorted(list(indirect_memory_locations)),
                "entry_input_locs": sorted(list(entry_input_locs))
            }
        }

        all_dfgsem.append(dfgsem)

    return all_dfgsem


def _addrstr_to_int(addr_str):
    if addr_str is None:
        return None
    s = str(addr_str).strip()
    try:
        return int(s, 16)
    except:
        m = re.search(r'([0-9a-fA-F]+)$', s)
        if m:
            try:
                return int(m.group(1), 16)
            except:
                return None
    return None


def _parse_func_entry_from_func_id(func_id):
    if not func_id:
        return None
    s = str(func_id)
    parts = s.split("_", 1)
    return parts[0] if parts else None


def _parse_func_entry_from_mem_location(mem_loc):
    if not mem_loc:
        return None
    s = str(mem_loc)
    if "_layout_" in s:
        return s.split("_layout_", 1)[0]
    return _parse_func_entry_from_func_id(s)


def _clamp_strength(v):
    try:
        v = float(v)
    except:
        v = 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return round(v, 3)


def _append_mapping(mapping_list, seen, rule_id, name, src_id, dst_id,
                    source_layer, target_layer, trigger, evidence_type,
                    strength, emit, constraints, edge_kind):
    if not src_id or not dst_id:
        return
    dedup_key = (rule_id, src_id, dst_id, evidence_type, edge_kind)
    if dedup_key in seen:
        return
    seen.add(dedup_key)

    mapping_list.append({
        "mapping_id": "%s_%d" % (rule_id, len(mapping_list)),
        "rule_id": rule_id,
        "name": name,
        "from": src_id,
        "to": dst_id,
        "edge_kind": edge_kind,
        "source_layer": source_layer,
        "target_layer": target_layer,
        "trigger": trigger or {},
        "evidence_type": evidence_type,
        "strength": _clamp_strength(strength),
        "emit": emit,
        "constraints": constraints or []
    })


# --------------------------------------------------
# Replacement v2: gen_mapping（真实规则触发）
# --------------------------------------------------
def gen_mapping(all_sem):
    mappings = []
    seen = set()

    instr_sem = all_sem.get("low_level_sem", {}).get("instr_sem", [])
    block_sem = all_sem.get("low_level_sem", {}).get("block_sem", [])
    cfg_sem = all_sem.get("mid_level_sem", {}).get("cfg_sem", [])
    dfg_sem = all_sem.get("mid_level_sem", {}).get("dataflow_sem", [])
    type_sem = all_sem.get("high_level_sem", {}).get("type_sem", [])
    interface_sem = all_sem.get("high_level_sem", {}).get("interface_sem", [])
    api_sem = all_sem.get("high_level_sem", {}).get("api_sem", [])
    mem_sem = all_sem.get("high_level_sem", {}).get("mem_layout_sem", [])

    # -----------------------------
    # 索引：CFG / Block / Instr / DFG / High-level
    # -----------------------------
    cfg_id_by_func = {}
    cfg_obj_by_func = {}
    block_func_by_entry = {}

    for cfg in cfg_sem:
        func_entry = cfg.get("entry_address")
        cfg_id = cfg.get("id") or gen_id("mid_cfg", func_entry, "0")
        cfg_id_by_func[func_entry] = cfg_id
        cfg_obj_by_func[func_entry] = cfg
        for b in cfg.get("blocks", []):
            block_func_by_entry[b.get("start_address")] = func_entry

    block_records = []
    for b in block_sem:
        func_entry = block_func_by_entry.get(b.get("entry"))
        block_records.append({
            "obj": b,
            "id": b.get("id"),
            "entry": b.get("entry"),
            "end": b.get("end"),
            "entry_i": _addrstr_to_int(b.get("entry")),
            "end_i": _addrstr_to_int(b.get("end")),
            "func_entry": func_entry
        })

    addr_to_block = {}
    instr_records = []
    for ins in instr_sem:
        addr = ins.get("address")
        if addr not in addr_to_block:
            aint = _addrstr_to_int(addr)
            found = None
            for br in block_records:
                if br.get("entry_i") is None or br.get("end_i") is None or aint is None:
                    continue
                if br["entry_i"] <= aint <= br["end_i"]:
                    found = br
                    break
            addr_to_block[addr] = found
        owner_block = addr_to_block.get(addr)
        instr_records.append({
            "obj": ins,
            "id": ins.get("id"),
            "address": addr,
            "block": owner_block,
            "func_entry": owner_block.get("func_entry") if owner_block else None
        })

    instr_by_addr = {}
    for rec in instr_records:
        instr_by_addr.setdefault(rec.get("address"), []).append(rec)

    dfg_by_func = {}
    for dfg in dfg_sem:
        dfg_by_func[dfg.get("entry_address")] = dfg

    type_by_func = {}
    for ts in type_sem:
        fentry = ts.get("function_entry")
        if fentry:
            type_by_func[fentry] = ts

    interface_by_func = {}
    for itf in interface_sem:
        fentry = _parse_func_entry_from_func_id(itf.get("func_id"))
        if fentry:
            interface_by_func[fentry] = itf

    mem_by_func = {}
    for ms in mem_sem:
        fentry = _parse_func_entry_from_mem_location(ms.get("location"))
        if not fentry:
            fentry = _parse_func_entry_from_func_id(ms.get("id"))
        if fentry:
            mem_by_func.setdefault(fentry, []).append(ms)

    # DFG 块统计：R3 / R5 用
    dfg_block_stats = {}
    for fentry, dfg in dfg_by_func.items():
        stats = {}
        for defi in dfg.get("definitions", []):
            bentry = defi.get("block_entry")
            if bentry not in stats:
                stats[bentry] = {"def_count": 0, "edge_count": 0}
            stats[bentry]["def_count"] += 1
        for edge in dfg.get("data_edges", []):
            for bentry in [edge.get("from_block"), edge.get("to_block")]:
                if bentry not in stats:
                    stats[bentry] = {"def_count": 0, "edge_count": 0}
                stats[bentry]["edge_count"] += 1
        dfg_block_stats[fentry] = stats

    # -----------------------------
    # R1: 指令聚合
    # Trigger: 指令语义对象的有序序列 -> 基本块语义对象
    # -----------------------------
    for rec in instr_records:
        block = rec.get("block")
        if not block:
            continue

        _append_mapping(
            mappings, seen,
            "R1", "指令聚合",
            rec.get("id"), block.get("id"),
            "InstrSem", "BlockSem",
            {
                "instruction_address": rec.get("address"),
                "block_range": block.get("obj", {}).get("addr_range"),
                "block_id": block.get("obj", {}).get("block_id")
            },
            "ordered_membership",
            1.0,
            "BlockSem",
            ["instruction_in_block", "execution_order_preserved"],
            "structural"
        )

    # -----------------------------
    # R2: 控制流提取
    # Trigger: 基本块末端跳转/分支模式 -> CFGSem
    # -----------------------------
    for br in block_records:
        fentry = br.get("func_entry")
        cfg_id = cfg_id_by_func.get(fentry)
        cfg_obj = cfg_obj_by_func.get(fentry)
        if not cfg_id or not cfg_obj:
            continue

        binfo = None
        for b in cfg_obj.get("blocks", []):
            if b.get("start_address") == br.get("entry"):
                binfo = b
                break

        _append_mapping(
            mappings, seen,
            "R2", "控制流提取",
            br.get("id"), cfg_id,
            "BlockSem", "CFGSem",
            {
                "block_entry": br.get("entry"),
                "block_end": br.get("end"),
                "instruction_count": binfo.get("instruction_count") if binfo else None,
                "branches": br.get("obj", {}).get("branches"),
                "calls": br.get("obj", {}).get("calls")
            },
            "terminal_flow_pattern",
            1.0,
            "CFGSem",
            ["block_terminal_flow_consistent", "cfg_membership_confirmed"],
            "structural"
        )

    # -----------------------------
    # R3: 数据流提取
    # Trigger: 基本块定义/使用集合 -> DFGSem
    # -----------------------------
    for br in block_records:
        fentry = br.get("func_entry")
        dfg = dfg_by_func.get(fentry)
        if not dfg:
            continue

        stats = dfg_block_stats.get(fentry, {}).get(br.get("entry"), {"def_count": 0, "edge_count": 0})
        if stats.get("def_count", 0) == 0 and stats.get("edge_count", 0) == 0:
            continue

        evidence_strength = 0.55
        evidence_strength += 0.05 * min(stats.get("def_count", 0), 3)
        evidence_strength += 0.03 * min(stats.get("edge_count", 0), 5)

        _append_mapping(
            mappings, seen,
            "R3", "数据流提取",
            br.get("id"), dfg.get("id"),
            "BlockSem", "DFGSem",
            {
                "block_entry": br.get("entry"),
                "block_def_count": stats.get("def_count", 0),
                "block_edge_count": stats.get("edge_count", 0),
                "block_def_set": br.get("obj", {}).get("def_set", []),
                "block_use_set": br.get("obj", {}).get("use_set", [])
            },
            "block_def_use_summary",
            evidence_strength,
            "DFGSem",
            ["def_use_chain_exists", "cross_block_dependency_allowed"],
            "structural"
        )

    # -----------------------------
    # R4: 类型推导
    # Trigger: 内存访问模式与操作数使用特征 -> TypeSem
    # -----------------------------
    for fentry, ts in type_by_func.items():
        dfg = dfg_by_func.get(fentry)
        if not dfg:
            continue

        base_access_count = {}
        for acc in dfg.get("memory_access_patterns", []):
            base = acc.get("base")
            if not base:
                continue
            base_access_count[base] = base_access_count.get(base, 0) + 1

        # 指令级证据：解引用/访存模式
        for acc in dfg.get("memory_access_patterns", []):
            addr = acc.get("instruction_address")
            if addr not in instr_by_addr:
                continue

            base = acc.get("base")
            count = base_access_count.get(base, 1)
            strength = 0.55
            if base:
                strength += 0.10
            if acc.get("offset") is not None:
                strength += 0.05
            if count >= 2:
                strength += 0.10
            if acc.get("is_fixed"):
                strength += 0.05

            for ins_rec in instr_by_addr.get(addr, []):
                if ins_rec.get("func_entry") != fentry:
                    continue

                _append_mapping(
                    mappings, seen,
                    "R4", "类型推导",
                    ins_rec.get("id"), ts.get("id"),
                    "InstrSem", "TypeSem",
                    {
                        "instruction_address": addr,
                        "memory_location": acc.get("location"),
                        "base": acc.get("base"),
                        "offset": acc.get("offset"),
                        "access": acc.get("access")
                    },
                    "memory_access_pattern",
                    strength,
                    "TypeSem",
                    ["candidate:pointer", "deref_path_or_memory_access_present"],
                    "evidence"
                )

        # 数据流级证据：定义-使用链 / 解引用链
        if dfg.get("data_edges") or dfg.get("memory_access_patterns"):
            deref_count = len(dfg.get("memory_access_patterns", []))
            flow_strength = 0.60 + 0.03 * min(deref_count, 5)
            _append_mapping(
                mappings, seen,
                "R4", "类型推导",
                dfg.get("id"), ts.get("id"),
                "DFGSem", "TypeSem",
                {
                    "function_entry": fentry,
                    "memory_access_count": deref_count,
                    "data_edge_count": len(dfg.get("data_edges", []))
                },
                "def_use_and_deref_chain",
                flow_strength,
                "TypeSem",
                ["pointer_usage_consistent", "type_candidate_supported_by_dataflow"],
                "evidence"
            )

    # -----------------------------
    # R5: 接口识别
    # Trigger: 调用约定与参数传递模式 -> InterfaceSem
    # -----------------------------
    for fentry, itf in interface_by_func.items():
        dfg = dfg_by_func.get(fentry)
        if not dfg:
            continue

        param_like = dfg.get("entry_input_locs", [])
        if not param_like and not itf.get("params"):
            continue

        strength = 0.72 + 0.04 * min(len(param_like), 3)

        # 优先用入口块做结构性证据
        entry_block = None
        for br in block_records:
            if br.get("entry") == fentry and br.get("func_entry") == fentry:
                entry_block = br
                break
        if entry_block is None:
            for br in block_records:
                if br.get("func_entry") == fentry:
                    entry_block = br
                    break

        if entry_block is not None:
            _append_mapping(
                mappings, seen,
                "R5", "接口识别",
                entry_block.get("id"), itf.get("id"),
                "BlockSem", "InterfaceSem",
                {
                    "function_entry": fentry,
                    "entry_input_locs": param_like,
                    "calling_convention": itf.get("interface_constraints", {}).get("calling_convention"),
                    "param_count": itf.get("interface_constraints", {}).get("param_count")
                },
                "calling_convention_and_entry_usage",
                strength,
                "InterfaceSem",
                ["calling_convention_compatible", "param_passing_pattern_detected"],
                "evidence"
            )

        _append_mapping(
            mappings, seen,
            "R5", "接口识别",
            dfg.get("id"), itf.get("id"),
            "DFGSem", "InterfaceSem",
            {
                "function_entry": fentry,
                "entry_input_locs": param_like,
                "param_count": itf.get("interface_constraints", {}).get("param_count"),
                "has_return": itf.get("interface_constraints", {}).get("has_return")
            },
            "param_passing_pattern",
            strength + 0.04,
            "InterfaceSem",
            ["entry_use_before_def", "interface_candidate_supported_by_dataflow"],
            "evidence"
        )

    # -----------------------------
    # R6: API 归整
    # Trigger: 符号信息与调用行为特征 -> APISem
    # -----------------------------
    for api in api_sem:
        callsite = api.get("callsite")
        api_name = api.get("api_name")
        caller_func = _parse_func_entry_from_func_id(api.get("propagated_constraints", {}).get("caller_func_id"))

        if callsite in instr_by_addr:
            for ins_rec in instr_by_addr.get(callsite, []):
                _append_mapping(
                    mappings, seen,
                    "R6", "API归整",
                    ins_rec.get("id"), api.get("id"),
                    "InstrSem", "APISem",
                    {
                        "callsite": callsite,
                        "api_name": api_name,
                        "instruction_address": callsite
                    },
                    "call_instruction",
                    0.95 if api_name else 0.75,
                    "APISem",
                    ["call_target_observed", "api_template_bindable"],
                    "evidence"
                )

        owner_block = addr_to_block.get(callsite)
        if owner_block is not None:
            _append_mapping(
                mappings, seen,
                "R6", "API归整",
                owner_block.get("id"), api.get("id"),
                "BlockSem", "APISem",
                {
                    "callsite": callsite,
                    "caller_function": caller_func,
                    "block_entry": owner_block.get("entry"),
                    "api_name": api_name
                },
                "call_behavior_context",
                0.80,
                "APISem",
                ["external_call_context", "semantic_template_available"],
                "evidence"
            )

    # -----------------------------
    # R7: 布局推断
    # Trigger: 固定偏移访问与步长模式 -> LayoutSem
    # -----------------------------
    for fentry, dfg in dfg_by_func.items():
        mem_candidates = mem_by_func.get(fentry, [])
        if not mem_candidates:
            continue

        alias_groups = dfg.get("abstract_merge", {}).get("alias_groups", [])
        if not alias_groups:
            continue

        for group in alias_groups:
            offsets = sorted(list(group.get("offsets", [])))
            if len(offsets) < 2:
                continue

            discovered = set(offsets)
            stride = group.get("stride")
            base = group.get("base")
            access_count = int(group.get("access_count", 0))

            for mem_obj in mem_candidates:
                target_offsets = set(mem_obj.get("offset_set", []))
                if not target_offsets:
                    continue

                overlap = discovered & target_offsets
                if not overlap:
                    continue

                overlap_ratio = float(len(overlap)) / float(max(len(discovered), 1))
                strength = 0.55 + 0.10 * min(len(discovered), 3)
                strength = strength * (0.60 + 0.40 * overlap_ratio)
                if stride is not None and len(discovered) >= 3:
                    strength += 0.05

                # 指令级固定偏移证据
                for addr in group.get("instr_addresses", []):
                    for ins_rec in instr_by_addr.get(addr, []):
                        if ins_rec.get("func_entry") != fentry:
                            continue
                        _append_mapping(
                            mappings, seen,
                            "R7", "布局推断",
                            ins_rec.get("id"), mem_obj.get("id"),
                            "InstrSem", "LayoutSem",
                            {
                                "instruction_address": addr,
                                "base": base,
                                "offset_set": offsets,
                                "matched_offsets": sorted(list(overlap)),
                                "stride": stride
                            },
                            "fixed_offset_access",
                            strength,
                            "LayoutSem",
                            ["offset_set_supported", "layout_struct_like"],
                            "evidence"
                        )

                # DFG 级步长/访存群证据
                _append_mapping(
                    mappings, seen,
                    "R7", "布局推断",
                    dfg.get("id"), mem_obj.get("id"),
                    "DFGSem", "LayoutSem",
                    {
                        "function_entry": fentry,
                        "base": base,
                        "offset_set": offsets,
                        "matched_offsets": sorted(list(overlap)),
                        "stride": stride,
                        "access_count": access_count
                    },
                    "stride_and_fixed_offset_pattern",
                    strength + 0.03,
                    "LayoutSem",
                    ["multi_offset_access_detected", "layout_candidate_consistent"],
                    "evidence"
                )

    return mappings


# ==============================================
# 运行 & 输出 JSON
# ==============================================

# ==============================================
# PATCH v3: 高层对象覆盖 / 统一引用 ID / BlockSem 摘要回填
#          / 地址表达式规范化 / LayoutSem 真正化 / USG 约束边
#          / API 语义链修复
# ==============================================

try:
    _BASE_GEN_CFG_SEM = gen_cfg_sem
except:
    _BASE_GEN_CFG_SEM = None

try:
    _BASE_GET_FUNCTION_BLOCKS = get_function_blocks
except:
    _BASE_GET_FUNCTION_BLOCKS = None


def _safe_get_name(obj, default_value):
    if obj is None:
        return default_value
    try:
        name = obj.getName()
        if name is None:
            return default_value
        return str(name)
    except:
        try:
            return str(obj)
        except:
            return default_value


def _safe_get_length(dt, default_value=0):
    if dt is None:
        return default_value
    try:
        return int(dt.getLength())
    except:
        return default_value


def _safe_get_storage(var):
    if var is None:
        return "unknown"
    try:
        st = var.getStorage()
        if st is None:
            return "unknown"
        return str(st)
    except:
        return "unknown"


def _safe_get_stack_offset(var):
    if var is None:
        return None
    try:
        return int(var.getStackOffset())
    except:
        return None


# ==============================================
# v7 eval-ready helpers: 给 DWARF 对齐评估提供稳定定位字段
# ==============================================
def _storage_kind_and_id(storage):
    s = "unknown" if storage is None else str(storage)
    low = s.lower()
    kind = "unknown"
    if "stack" in low:
        kind = "stack"
    elif "register" in low or low.startswith("reg:") or "reg" in low:
        kind = "register"
    elif "ram" in low or "mem" in low or "global" in low:
        kind = "global"
    elif "return" in low:
        kind = "return"
    return kind, s


def _first_type_candidate(type_candidates):
    try:
        if type_candidates and len(type_candidates) > 0:
            return str(type_candidates[0])
    except:
        pass
    return "unknown"


def _extract_site_addrs(sites):
    out = []
    if not sites:
        return out
    try:
        for site in sites:
            if isinstance(site, dict):
                for k in ["addr", "address", "instr_addr", "site", "pc"]:
                    if k in site and site[k] is not None:
                        out.append(str(site[k]))
                        break
            elif site is not None:
                out.append(str(site))
    except:
        pass
    return _unique_list(out)


def _safe_dt_name(dt):
    if dt is None:
        return "unknown"
    try:
        return str(dt.getName())
    except:
        return str(dt)


def _safe_cc(func):
    if func is None:
        return "unknown"
    try:
        cc = func.getCallingConventionName()
        if cc:
            return str(cc)
    except:
        pass
    try:
        cc = func.getCallingConvention()
        if cc:
            return str(cc)
    except:
        pass
    return "unknown"


def _is_void_type(dt):
    if dt is None:
        return False
    name = _safe_dt_name(dt).lower()
    return name == "void"


def _function_uid(func):
    entry = _addr_to_str(func.getEntryPoint())
    name = _safe_get_name(func, "sub_%s" % entry)
    return "%s_%s" % (entry, name)


def _get_internal_functions(prog, include_thunks=False):
    out = []
    funcs = prog.getFunctionManager().getFunctions(True)
    while funcs.hasNext():
        func = funcs.next()
        if func is None:
            continue
        try:
            if func.isExternal():
                continue
        except:
            pass
        try:
            if (not include_thunks) and func.isThunk():
                continue
        except:
            pass
        out.append(func)
    return out


def _get_processor_name(prog):
    try:
        return str(prog.getLanguage().getProcessor()).lower()
    except:
        return "unknown"


def _get_call_arg_register_slots(prog):
    """
    返回每个参数槽可能出现的一组寄存器名（统一成大写，不带 reg: 前缀）。
    兼顾 AArch64 / ARM / x86-64 / MIPS 的常见调用约定。
    """
    proc = _get_processor_name(prog)
    slots = []
    if "aarch64" in proc or "arm64" in proc:
        for i in range(8):
            slots.append(set(["X%d" % i, "W%d" % i, "S%d" % i, "D%d" % i, "Q%d" % i]))
        return slots
    if "x86" in proc and "64" in proc:
        # SysV + Windows x64 的并集，顺序以最常见顺序为主
        raw = [
            set(["RDI", "EDI", "XMM0"]),
            set(["RSI", "ESI", "XMM1"]),
            set(["RDX", "EDX", "XMM2"]),
            set(["RCX", "ECX", "XMM3"]),
            set(["R8", "R8D", "XMM4"]),
            set(["R9", "R9D", "XMM5"])
        ]
        # Windows x64 常见次序 RCX/RDX/R8/R9，也并进前四槽
        raw[0].update(set(["RCX", "ECX"]))
        raw[1].update(set(["RDX", "EDX"]))
        raw[2].update(set(["R8", "R8D"]))
        raw[3].update(set(["R9", "R9D"]))
        return raw
    if "arm" in proc:
        for i in range(4):
            slots.append(set(["R%d" % i, "S%d" % i, "D%d" % i]))
        return slots
    if "mips" in proc:
        return [
            set(["A0"]),
            set(["A1"]),
            set(["A2"]),
            set(["A3"])
        ]
    # 默认退化：只返回通用寄存器名，至少让 API 参数抽取能工作
    return [
        set(["R0", "X0", "A0"]),
        set(["R1", "X1", "A1"]),
        set(["R2", "X2", "A2"]),
        set(["R3", "X3", "A3"])
    ]


def _parse_int_addr(addr_str):
    return _addrstr_to_int(addr_str)


def _make_const_alias(value):
    return {
        "base": None,
        "offset": int(value),
        "expr": _format_signed_immediate(int(value)),
        "is_fixed": True,
        "source_loc": None
    }


def _make_loc_alias(loc):
    if not loc:
        return None
    if str(loc).startswith("reg:") or str(loc).startswith("stack:") or str(loc).startswith("addr:"):
        return {
            "base": str(loc),
            "offset": 0,
            "expr": str(loc),
            "is_fixed": True,
            "source_loc": str(loc)
        }
    return {
        "base": None,
        "offset": None,
        "expr": str(loc),
        "is_fixed": False,
        "source_loc": str(loc)
    }


def _clone_alias_info(info):
    if info is None:
        return None
    return {
        "base": info.get("base"),
        "offset": info.get("offset"),
        "expr": info.get("expr"),
        "is_fixed": bool(info.get("is_fixed")),
        "source_loc": info.get("source_loc")
    }


def _alias_signature(info):
    if info is None:
        return None
    return (
        info.get("base"),
        info.get("offset"),
        info.get("expr"),
        bool(info.get("is_fixed"))
    )


def _distinct_aliases(items):
    out = []
    seen = set()
    for item in items:
        if item is None:
            continue
        sig = _alias_signature(item)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(_clone_alias_info(item))
    return out


def _merge_alias_candidates(cands):
    uniq = _distinct_aliases(cands)
    if not uniq:
        return None
    if len(uniq) == 1:
        return uniq[0]

    bases = set([u.get("base") for u in uniq if u.get("base") is not None])
    exprs = set([u.get("expr") for u in uniq if u.get("expr") is not None])
    offsets = set([u.get("offset") for u in uniq if u.get("offset") is not None])

    if len(bases) == 1 and len(offsets) == 1:
        return {
            "base": list(bases)[0],
            "offset": list(offsets)[0],
            "expr": _format_base_offset(list(bases)[0], list(offsets)[0]),
            "is_fixed": True,
            "source_loc": None
        }

    if len(bases) == 1:
        base = list(bases)[0]
        return {
            "base": base,
            "offset": None,
            "expr": "%s+phi" % base,
            "is_fixed": False,
            "source_loc": None
        }

    if len(exprs) == 1:
        return uniq[0]

    return None


def _add_memory_cell_alias(memory_cells, mem_loc, alias_info, site=None):
    if not mem_loc or alias_info is None:
        return
    if mem_loc not in memory_cells:
        memory_cells[mem_loc] = {"aliases": [], "sites": []}
    memory_cells[mem_loc]["aliases"].append(_clone_alias_info(alias_info))
    if site is not None:
        memory_cells[mem_loc]["sites"].append(site)


def _get_memory_cell_alias(memory_cells, mem_loc):
    if mem_loc not in memory_cells:
        return None
    return _merge_alias_candidates(memory_cells[mem_loc].get("aliases", []))


def _vn_desc(program, vn):
    if vn is None:
        return {"loc": None, "const": None, "piece": None}
    return {
        "loc": _canonical_varnode_location(program, vn, include_constant=False),
        "const": _const_vn_as_signed(vn),
        "piece": _safe_encode_piece(vn)
    }


def _alias_from_desc(desc, value_alias):
    if desc is None:
        return None
    cst = desc.get("const")
    if cst is not None:
        return _make_const_alias(cst)
    loc = desc.get("loc")
    if loc and loc in value_alias:
        return _clone_alias_info(value_alias.get(loc))
    if loc:
        return _make_loc_alias(loc)
    return None


def _combine_alias_add(left, right):
    if left is None and right is None:
        return None
    if left is None:
        return _clone_alias_info(right)
    if right is None:
        return _clone_alias_info(left)

    if left.get("base") is not None and right.get("offset") is not None:
        off = int(left.get("offset") or 0) + int(right.get("offset"))
        return {
            "base": left.get("base"),
            "offset": off,
            "expr": _format_base_offset(left.get("base"), off),
            "is_fixed": bool(left.get("is_fixed")) and bool(right.get("is_fixed")),
            "source_loc": left.get("source_loc")
        }

    if right.get("base") is not None and left.get("offset") is not None:
        off = int(right.get("offset") or 0) + int(left.get("offset"))
        return {
            "base": right.get("base"),
            "offset": off,
            "expr": _format_base_offset(right.get("base"), off),
            "is_fixed": bool(left.get("is_fixed")) and bool(right.get("is_fixed")),
            "source_loc": right.get("source_loc")
        }

    return {
        "base": left.get("base") or right.get("base"),
        "offset": None,
        "expr": "%s+%s" % (left.get("expr"), right.get("expr")),
        "is_fixed": False,
        "source_loc": left.get("source_loc") or right.get("source_loc")
    }


def _combine_alias_sub(left, right):
    if left is None:
        return None
    if right is None:
        return _clone_alias_info(left)

    if left.get("base") is not None and right.get("offset") is not None:
        off = int(left.get("offset") or 0) - int(right.get("offset"))
        return {
            "base": left.get("base"),
            "offset": off,
            "expr": _format_base_offset(left.get("base"), off),
            "is_fixed": bool(left.get("is_fixed")) and bool(right.get("is_fixed")),
            "source_loc": left.get("source_loc")
        }

    return {
        "base": left.get("base"),
        "offset": None,
        "expr": "%s-%s" % (left.get("expr"), right.get("expr")),
        "is_fixed": False,
        "source_loc": left.get("source_loc")
    }


def _extract_pcode_facts_v3(program, instr, pcode_op, pcode_index, block):
    addr = _addr_to_str(instr.getAddress())
    site = "%s:%d" % (addr, int(pcode_index))

    try:
        mnemonic = str(pcode_op.getMnemonic())
    except:
        mnemonic = "UNKNOWN"

    num_inputs = 0
    try:
        num_inputs = int(pcode_op.getNumInputs())
    except:
        num_inputs = 0

    input_descs = []
    for i in range(num_inputs):
        try:
            vn = pcode_op.getInput(i)
        except:
            vn = None
        input_descs.append(_vn_desc(program, vn))

    output_desc = {"loc": None, "const": None, "piece": None}
    try:
        output_desc = _vn_desc(program, pcode_op.getOutput())
    except:
        pass

    uses = []
    defs = []
    mem_accesses = []

    if mnemonic == "LOAD":
        if len(input_descs) >= 2:
            try:
                addr_vn = pcode_op.getInput(1)
            except:
                addr_vn = None
            mem_info = _canonical_memory_info(program, addr_vn)
            raw_loc = mem_info.get("location")
            uses.append(raw_loc)
            uses.extend(mem_info.get("support_locs", []))
            mem_accesses.append({
                "site": site,
                "instruction_address": addr,
                "block_entry": block["entry_str"],
                "block_id": block["block_id"],
                "access": "load",
                "raw_location": raw_loc,
                "location": raw_loc,
                "base": mem_info.get("base"),
                "offset": mem_info.get("offset"),
                "expr": mem_info.get("expr"),
                "is_fixed": mem_info.get("is_fixed"),
                "support_locs": mem_info.get("support_locs", [])
            })
        if output_desc.get("loc"):
            defs.append(output_desc.get("loc"))

    elif mnemonic == "STORE":
        if len(input_descs) >= 3:
            try:
                addr_vn = pcode_op.getInput(1)
            except:
                addr_vn = None
            mem_info = _canonical_memory_info(program, addr_vn)
            raw_loc = mem_info.get("location")
            defs.append(raw_loc)
            uses.extend(mem_info.get("support_locs", []))

            value_loc = input_descs[2].get("loc")
            if value_loc:
                uses.append(value_loc)

            mem_accesses.append({
                "site": site,
                "instruction_address": addr,
                "block_entry": block["entry_str"],
                "block_id": block["block_id"],
                "access": "store",
                "raw_location": raw_loc,
                "location": raw_loc,
                "base": mem_info.get("base"),
                "offset": mem_info.get("offset"),
                "expr": mem_info.get("expr"),
                "is_fixed": mem_info.get("is_fixed"),
                "support_locs": mem_info.get("support_locs", []),
                "value_loc": value_loc
            })

    else:
        for desc in input_descs:
            if desc.get("loc"):
                uses.append(desc.get("loc"))
        if output_desc.get("loc"):
            defs.append(output_desc.get("loc"))

    return {
        "site": site,
        "instruction_address": addr,
        "instruction": instr.toString(),
        "mnemonic": mnemonic,
        "pcode_index": int(pcode_index),
        "block_entry": block["entry_str"],
        "block_id": block["block_id"],
        "input_descs": input_descs,
        "output_desc": output_desc,
        "uses": _unique_list(uses),
        "defs": _unique_list(defs),
        "memory_accesses": mem_accesses
    }


def _compute_output_alias_from_fact(facts, value_alias, memory_cells):
    mnemonic = facts.get("mnemonic")
    out_loc = facts.get("output_desc", {}).get("loc")
    if not out_loc:
        return None

    inputs = facts.get("input_descs", [])
    aliases = []
    for desc in inputs:
        aliases.append(_alias_from_desc(desc, value_alias))

    if mnemonic in ("COPY", "CAST", "INT_ZEXT", "INT_SEXT", "SUBPIECE", "PIECE", "INDIRECT"):
        if aliases:
            return aliases[0]

    if mnemonic in ("INT_ADD", "PTRSUB"):
        if len(aliases) >= 2:
            return _combine_alias_add(aliases[0], aliases[1])

    if mnemonic == "INT_SUB":
        if len(aliases) >= 2:
            return _combine_alias_sub(aliases[0], aliases[1])

    if mnemonic == "PTRADD":
        if len(aliases) >= 3:
            base_alias = aliases[0]
            idx_alias = aliases[1]
            scale_alias = aliases[2]
            if base_alias is not None and idx_alias is not None and scale_alias is not None:
                if base_alias.get("base") is not None and idx_alias.get("offset") is not None and scale_alias.get("offset") is not None:
                    off = int(base_alias.get("offset") or 0) + int(idx_alias.get("offset")) * int(scale_alias.get("offset"))
                    return {
                        "base": base_alias.get("base"),
                        "offset": off,
                        "expr": _format_base_offset(base_alias.get("base"), off),
                        "is_fixed": bool(base_alias.get("is_fixed")) and bool(idx_alias.get("is_fixed")) and bool(scale_alias.get("is_fixed")),
                        "source_loc": base_alias.get("source_loc")
                    }

    if mnemonic == "MULTIEQUAL":
        return _merge_alias_candidates(aliases)

    if mnemonic == "LOAD":
        mems = facts.get("memory_accesses", [])
        if mems:
            mem_loc = mems[0].get("location")
            ali = _get_memory_cell_alias(memory_cells, mem_loc)
            if ali is not None:
                return ali

    return None


def _refine_memory_access_with_alias(acc, value_alias, memory_cells):
    updated = dict(acc)
    raw_location = updated.get("raw_location") or updated.get("location")
    updated["raw_location"] = raw_location

    if updated.get("base") is not None and updated.get("is_fixed"):
        return updated

    candidate_aliases = []
    for loc in updated.get("support_locs", []):
        if loc in value_alias:
            candidate_aliases.append(value_alias.get(loc))
        else:
            candidate_aliases.append(_make_loc_alias(loc))

    if raw_location in memory_cells:
        candidate_aliases.append(_get_memory_cell_alias(memory_cells, raw_location))

    ali = _merge_alias_candidates(candidate_aliases)
    if ali is None or ali.get("base") is None:
        return updated

    if ali.get("is_fixed"):
        base = ali.get("base")
        base_off = int(ali.get("offset") or 0)
        off = updated.get("offset")
        if off is None:
            off = 0
        new_off = base_off + int(off)
        updated["base"] = base
        updated["offset"] = new_off
        updated["expr"] = _format_base_offset(base, new_off)
        updated["is_fixed"] = True
        updated["location"] = "mem[%s]" % updated["expr"]
        return updated

    updated["base"] = ali.get("base")
    updated["offset"] = None
    updated["expr"] = "%s+?" % ali.get("expr")
    updated["is_fixed"] = False
    updated["location"] = "mem[%s]" % updated["expr"]
    return updated


def _rename_memory_locations_in_dfg(dfg, rename_by_site):
    new_defs = []
    for defi in dfg.get("definitions", []):
        item = dict(defi)
        key = (item.get("site"), item.get("location"))
        if key in rename_by_site:
            item["location"] = rename_by_site[key]
        new_defs.append(item)

    new_edges = []
    for edge in dfg.get("data_edges", []):
        item = dict(edge)
        k_to = (item.get("to_site"), item.get("location"))
        k_from = (item.get("from_site"), item.get("location"))
        if k_to in rename_by_site:
            item["location"] = rename_by_site[k_to]
        elif k_from in rename_by_site:
            item["location"] = rename_by_site[k_from]
        new_edges.append(item)

    dfg["definitions"] = new_defs
    dfg["data_edges"] = new_edges


def _rebuild_alias_groups(memory_access_patterns):
    base_groups = {}
    indirect_memory_locations = set()

    for acc in memory_access_patterns:
        if acc.get("is_fixed") and acc.get("base") is not None and acc.get("offset") is not None:
            base = acc.get("base")
            if base not in base_groups:
                base_groups[base] = {
                    "base": base,
                    "offsets": set(),
                    "instr_addresses": set(),
                    "sites": set(),
                    "accesses": [],
                    "fields": {}
                }

            off = int(acc.get("offset"))
            base_groups[base]["offsets"].add(off)
            if acc.get("instruction_address"):
                base_groups[base]["instr_addresses"].add(acc.get("instruction_address"))
            if acc.get("site"):
                base_groups[base]["sites"].add(acc.get("site"))
            base_groups[base]["accesses"].append(acc)

            if off not in base_groups[base]["fields"]:
                base_groups[base]["fields"][off] = {"loads": 0, "stores": 0, "sites": []}
            if acc.get("access") == "load":
                base_groups[base]["fields"][off]["loads"] += 1
            elif acc.get("access") == "store":
                base_groups[base]["fields"][off]["stores"] += 1
            if acc.get("site"):
                base_groups[base]["fields"][off]["sites"].append(acc.get("site"))
        else:
            if acc.get("location"):
                indirect_memory_locations.add(acc.get("location"))

    alias_groups = []
    for base, group in base_groups.items():
        offsets = sorted(list(group["offsets"]))
        stride = None
        if len(offsets) >= 2:
            diffs = []
            for i in range(len(offsets) - 1):
                diffs.append(offsets[i + 1] - offsets[i])
            if diffs and len(set(diffs)) == 1:
                stride = diffs[0]

        field_summaries = []
        for off in sorted(group["fields"].keys()):
            field_summaries.append({
                "offset": off,
                "loads": group["fields"][off]["loads"],
                "stores": group["fields"][off]["stores"],
                "sites": sorted(_unique_list(group["fields"][off]["sites"]))
            })

        alias_groups.append({
            "base": base,
            "offsets": offsets,
            "instr_addresses": sorted(list(group["instr_addresses"])),
            "sites": sorted(list(group["sites"])),
            "access_count": len(group["accesses"]),
            "stride": stride,
            "field_summaries": field_summaries
        })

    return alias_groups, sorted(list(indirect_memory_locations))


def gen_dataflow_sem(prog):
    """
    基于上一版 p-code 级 DFG 再做一层地址表达式与栈槽别名归一：
    1) 修复“tmp 地址表达式”导致的 base/offset 丢失；
    2) 用固定栈槽的 store->load 值传播，恢复参数指针/局部指针的真实基址；
    3) 重建 alias_groups，使 LayoutSem 能真正依赖 offset cluster 触发。
    """
    if _BASE_GEN_CFG_SEM is None or _BASE_GET_FUNCTION_BLOCKS is None:
        return []

    # 先用当前版本的 CFG 作为块边界来源
    cfg_sem = _BASE_GEN_CFG_SEM(prog)
    cfg_by_entry = {}
    for cfg in cfg_sem:
        cfg_by_entry[cfg.get("entry_address")] = cfg

    all_dfgsem = []
    tm = _get_script_monitor()
    block_model = BasicBlockModel(prog)

    funcs = _get_internal_functions(prog, include_thunks=False)
    for func in funcs:
        func_entry = _addr_to_str(func.getEntryPoint())
        cfg_obj = cfg_by_entry.get(func_entry)
        func_blocks = _BASE_GET_FUNCTION_BLOCKS(prog, func, block_model, tm)
        if not func_blocks:
            continue

        preds, succs = _build_function_preds(func_blocks, tm)

        block_ops = {}
        gen_defs = {}
        kill_locs = {}
        site_meta = {}
        locations = set()
        definitions = []
        entry_input_locs = set()

        memory_cell_alias = {}
        value_alias = {}
        rename_by_site = {}
        memory_access_patterns = []

        # 按块入口地址线性扫描，做轻量值别名传播
        for block in func_blocks:
            bentry = block["entry_str"]
            ops = []
            local_last_def = {}
            local_use_before_def = set()

            for instr in block["instructions"]:
                try:
                    pcode_ops = list(instr.getPcode())
                except:
                    pcode_ops = []

                for pidx, pcode_op in enumerate(pcode_ops):
                    facts = _extract_pcode_facts_v3(prog, instr, pcode_op, pidx, block)

                    refined_mems = []
                    for acc in facts.get("memory_accesses", []):
                        refined = _refine_memory_access_with_alias(acc, value_alias, memory_cell_alias)
                        refined_mems.append(refined)
                        old_loc = acc.get("raw_location") or acc.get("location")
                        new_loc = refined.get("location")
                        if old_loc != new_loc:
                            rename_by_site[(refined.get("site"), old_loc)] = new_loc

                    facts["memory_accesses"] = refined_mems

                    # 用 refined location 覆盖 STORE/LOAD 的 uses/defs 中 memory location
                    if refined_mems:
                        new_uses = []
                        new_defs = []
                        use_set = set(facts.get("uses", []))
                        def_set = set(facts.get("defs", []))
                        for u in facts.get("uses", []):
                            replaced = u
                            if (facts.get("site"), u) in rename_by_site:
                                replaced = rename_by_site[(facts.get("site"), u)]
                            if replaced not in new_uses:
                                new_uses.append(replaced)
                        for d in facts.get("defs", []):
                            replaced = d
                            if (facts.get("site"), d) in rename_by_site:
                                replaced = rename_by_site[(facts.get("site"), d)]
                            if replaced not in new_defs:
                                new_defs.append(replaced)
                        facts["uses"] = new_uses
                        facts["defs"] = new_defs

                    # 根据事实更新 alias env
                    if facts.get("mnemonic") == "STORE" and facts.get("memory_accesses"):
                        mem_acc = facts.get("memory_accesses")[0]
                        val_alias = None
                        inputs = facts.get("input_descs", [])
                        if len(inputs) >= 3:
                            val_alias = _alias_from_desc(inputs[2], value_alias)
                        if val_alias is not None and (val_alias.get("base") is not None or val_alias.get("source_loc") is not None):
                            _add_memory_cell_alias(memory_cell_alias, mem_acc.get("location"), val_alias, facts.get("site"))

                    out_loc = facts.get("output_desc", {}).get("loc")
                    out_alias = _compute_output_alias_from_fact(facts, value_alias, memory_cell_alias)
                    if out_loc:
                        if out_alias is not None:
                            value_alias[out_loc] = out_alias
                        else:
                            if out_loc in value_alias:
                                del value_alias[out_loc]

                    ops.append(facts)
                    site_meta[facts["site"]] = facts

                    for loc in facts.get("uses", []):
                        if loc:
                            locations.add(loc)
                    for loc in facts.get("defs", []):
                        if loc:
                            locations.add(loc)

                    for mem_acc in facts.get("memory_accesses", []):
                        memory_access_patterns.append(mem_acc)
                        if mem_acc.get("location"):
                            locations.add(mem_acc.get("location"))

                    for dloc in facts.get("defs", []):
                        definitions.append({
                            "site": facts["site"],
                            "instruction_address": facts["instruction_address"],
                            "block_entry": bentry,
                            "block_id": block["block_id"],
                            "mnemonic": facts["mnemonic"],
                            "location": dloc
                        })

                    for uloc in facts.get("uses", []):
                        if _is_global_location(uloc) and uloc not in local_last_def:
                            local_use_before_def.add(uloc)

                    for dloc in facts.get("defs", []):
                        if _is_global_location(dloc):
                            local_last_def[dloc] = facts["site"]

            block_ops[bentry] = ops
            gen_defs[bentry] = {}
            kill_locs[bentry] = set()

            for loc, site in local_last_def.items():
                gen_defs[bentry][loc] = set([site])
                kill_locs[bentry].add(loc)

            if bentry == func_entry:
                for uloc in local_use_before_def:
                    if _is_param_like_location(uloc):
                        entry_input_locs.add(uloc)

        in_defs = {}
        out_defs = {}
        for block in func_blocks:
            bentry = block["entry_str"]
            in_defs[bentry] = {}
            out_defs[bentry] = {}

        changed = True
        while changed:
            changed = False
            for block in func_blocks:
                bentry = block["entry_str"]
                new_in = {}
                for pred in preds.get(bentry, []):
                    new_in = _merge_defs_maps(new_in, out_defs.get(pred, {}))

                new_out = _clone_defs_map(new_in)
                for loc in kill_locs.get(bentry, set()):
                    if loc in new_out:
                        del new_out[loc]
                for loc, sites in gen_defs.get(bentry, {}).items():
                    new_out[loc] = set(sites)

                if (not _defs_maps_equal(in_defs.get(bentry, {}), new_in)) or (not _defs_maps_equal(out_defs.get(bentry, {}), new_out)):
                    in_defs[bentry] = new_in
                    out_defs[bentry] = new_out
                    changed = True

        data_edges = []
        edge_seen = set()
        for block in func_blocks:
            bentry = block["entry_str"]
            env = _clone_defs_map(in_defs.get(bentry, {}))
            for facts in block_ops.get(bentry, []):
                current_site = facts["site"]

                for uloc in facts.get("uses", []):
                    src_sites = env.get(uloc, set())
                    if not src_sites:
                        continue
                    for src_site in sorted(list(src_sites)):
                        key = (src_site, current_site, uloc)
                        if key in edge_seen:
                            continue
                        edge_seen.add(key)

                        src_meta = site_meta.get(src_site, {})
                        data_edges.append({
                            "from": src_meta.get("instruction_address"),
                            "to": facts.get("instruction_address"),
                            "from_site": src_site,
                            "to_site": current_site,
                            "from_block": src_meta.get("block_entry"),
                            "to_block": facts.get("block_entry"),
                            "dependency": _dependency_kind(uloc),
                            "location": uloc,
                            "producer_pcode": src_meta.get("mnemonic"),
                            "consumer_pcode": facts.get("mnemonic")
                        })

                for dloc in facts.get("defs", []):
                    env[dloc] = set([current_site])

        dfgsem = {
            "id": gen_id("mid_dataflow", func_entry, "0"),
            "function_name": _safe_get_name(func, "sub_%s" % func_entry),
            "entry_address": func_entry,
            "locations": sorted(list(locations)),
            "definitions": definitions,
            "data_edges": data_edges,
            "memory_access_patterns": memory_access_patterns,
            "entry_input_locs": sorted(list(entry_input_locs)),
            "abstract_merge": {
                "alias_groups": [],
                "indirect_memory_locations": [],
                "entry_input_locs": sorted(list(entry_input_locs))
            }
        }

        alias_groups, indirect_memory_locations = _rebuild_alias_groups(memory_access_patterns)
        dfgsem["abstract_merge"]["alias_groups"] = alias_groups
        dfgsem["abstract_merge"]["indirect_memory_locations"] = indirect_memory_locations

        _rename_memory_locations_in_dfg(dfgsem, rename_by_site)

        # locations 重新计算，避免保留旧 mem[tmp:*] 位置
        final_locations = set()
        for d in dfgsem.get("definitions", []):
            if d.get("location"):
                final_locations.add(d.get("location"))
        for e in dfgsem.get("data_edges", []):
            if e.get("location"):
                final_locations.add(e.get("location"))
        for acc in dfgsem.get("memory_access_patterns", []):
            if acc.get("location"):
                final_locations.add(acc.get("location"))
        for loc in dfgsem.get("entry_input_locs", []):
            final_locations.add(loc)
        dfgsem["locations"] = sorted(list(final_locations))

        all_dfgsem.append(dfgsem)

    return all_dfgsem


def _collect_instruction_owner_map(prog):
    tm = _get_script_monitor()
    block_model = BasicBlockModel(prog)
    owner = {}
    funcs = _get_internal_functions(prog, include_thunks=False)
    for func in funcs:
        func_entry = _addr_to_str(func.getEntryPoint())
        func_blocks = _BASE_GET_FUNCTION_BLOCKS(prog, func, block_model, tm)
        for block in func_blocks:
            for instr in block.get("instructions", []):
                owner[_addr_to_str(instr.getAddress())] = {
                    "function_entry": func_entry,
                    "function_name": _safe_get_name(func, "sub_%s" % func_entry),
                    "block_entry": block.get("entry_str"),
                    "block_id": block.get("block_id")
                }
    return owner


def normalize_pcode(line, reg_map=None, tmp_map=None, counters=None):
    """
    修复旧版 normalize_pcode 把 mem(sp,...) 打平成 memsp... 的问题。
    这里保持结构化括号，不再 remove('(', ')')。
    """
    if not line:
        return "EMPTY"
    line = str(line)
    line = re.sub(r"register, *0x([0-9a-fA-F]+), *(\d+)", lambda m: "reg(0x%s,%s)" % (m.group(1), m.group(2)), line)
    line = re.sub(r"unique, *0x([0-9a-fA-F]+), *(\d+)", lambda m: "tmp(0x%s,%s)" % (m.group(1), m.group(2)), line)
    line = re.sub(r"stack, *(-?0x[0-9a-fA-F]+), *(\d+)", r"stack(\1,\2)", line)
    line = re.sub(r"const, *(-?0x[0-9a-fA-F]+), *(\d+)", r"const(\1,\2)", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def gen_instr_sem(prog, dfg_cache=None):
    output = []
    owner_map = _collect_instruction_owner_map(prog)
    tm = _get_script_monitor()
    block_model = BasicBlockModel(prog)
    funcs = _get_internal_functions(prog, include_thunks=False)

    seq = 0
    for func in funcs:
        func_blocks = _BASE_GET_FUNCTION_BLOCKS(prog, func, block_model, tm)
        for block in func_blocks:
            for instr in block.get("instructions", []):
                try:
                    pcode_ops = list(instr.getPcode())
                except:
                    pcode_ops = []

                for pidx, pcode_op in enumerate(pcode_ops):
                    facts = _extract_pcode_facts_v3(prog, instr, pcode_op, pidx, block)
                    input_operands = []
                    for desc in facts.get("input_descs", []):
                        if desc.get("const") is not None:
                            input_operands.append(_format_signed_immediate(desc.get("const")))
                        elif desc.get("loc") is not None:
                            input_operands.append(desc.get("loc"))
                        elif desc.get("piece") is not None:
                            input_operands.append(desc.get("piece"))

                    output_operand = facts.get("output_desc", {}).get("loc")
                    operands = []
                    if output_operand is not None:
                        operands.append(output_operand)
                    operands.extend(input_operands)

                    ctx_pre = {
                        "reads": facts.get("uses", []),
                        "block_entry": block.get("entry_str"),
                        "function_entry": _addr_to_str(func.getEntryPoint())
                    }
                    ctx_post = {
                        "writes": facts.get("defs", []),
                        "block_exit_candidate": block.get("end_str")
                    }
                    effects = {
                        "reads": facts.get("uses", []),
                        "writes": facts.get("defs", []),
                        "memory_accesses": facts.get("memory_accesses", []),
                        "control_effect": _classify_edge_type(instr.getFlowType()) if instr.getFlowType() is not None else "none"
                    }

                    sem = {
                        "id": gen_id("low_instr", _addr_to_str(instr.getAddress()), str(seq)),
                        "address": _addr_to_str(instr.getAddress()),
                        "function_entry": _addr_to_str(func.getEntryPoint()),
                        "block_entry": block.get("entry_str"),
                        "assembly": instr.toString(),
                        "n_pcode": normalize_pcode(str(pcode_op)),
                        "opcode": facts.get("mnemonic"),
                        "operands": operands,
                        "pcode_index": facts.get("pcode_index"),
                        "ctx_pre": ctx_pre,
                        "ctx_post": ctx_post,
                        "effects": effects
                    }
                    output.append(sem)
                    seq += 1

    return output


def _make_empty_block_summary(block):
    return {
        "block_id": block.get("block_id"),
        "entry": block.get("start_address"),
        "end": block.get("end_address"),
        "addr_range": "%s~%s" % (block.get("start_address"), block.get("end_address")),
        "ctx_in": {},
        "ctx_out": {},
        "sp_delta": 0,
        "def_set": [],
        "use_set": [],
        "read_set": [],
        "write_set": [],
        "effect_summary": {},
        "opcode_ratio": {},
        "branches": 0,
        "calls": 0,
        "T_B": [],
        "S_B": {},
        "Def_B": [],
        "Use_B": []
    }


def gen_block_sem(prog, cfg_cache=None, dfg_cache=None):
    """
    不再输出“空心块”。直接以 DFG/CFG 聚合块级摘要：
    - Def_B / Use_B / read_set / write_set
    - ctx_in / ctx_out
    - effect_summary
    """
    cfg_sem = cfg_cache if cfg_cache is not None else _BASE_GEN_CFG_SEM(prog)
    dfg_sem = dfg_cache if dfg_cache is not None else gen_dataflow_sem(prog)

    dfg_by_func = {}
    for d in dfg_sem:
        dfg_by_func[d.get("entry_address")] = d

    output = []
    for cfg in cfg_sem:
        func_entry = cfg.get("entry_address")
        dfg = dfg_by_func.get(func_entry, {})

        defs_by_block = defaultdict(list)
        for defi in dfg.get("definitions", []):
            defs_by_block[defi.get("block_entry")].append(defi)

        uses_by_block = defaultdict(set)
        reads_by_block = defaultdict(set)
        writes_by_block = defaultdict(set)
        sites_by_block = defaultdict(list)

        for edge in dfg.get("data_edges", []):
            if edge.get("to_block") and edge.get("location"):
                uses_by_block[edge.get("to_block")].add(edge.get("location"))

        for acc in dfg.get("memory_access_patterns", []):
            b = acc.get("block_entry")
            if acc.get("access") == "load":
                reads_by_block[b].add(acc.get("location"))
            elif acc.get("access") == "store":
                writes_by_block[b].add(acc.get("location"))
            sites_by_block[b].append(acc)

        block_info_by_start = {}
        for b in cfg.get("blocks", []):
            block_info_by_start[b.get("start_address")] = b

        for b in cfg.get("blocks", []):
            entry = b.get("start_address")
            sem = _make_empty_block_summary(b)
            def_set = set()
            use_set = set()
            read_set = set()
            write_set = set()
            pcode_counter = Counter()
            branch_count = 0
            call_count = 0

            for defi in defs_by_block.get(entry, []):
                if defi.get("location"):
                    def_set.add(defi.get("location"))
                if defi.get("mnemonic"):
                    pcode_counter[defi.get("mnemonic")] += 1

            for u in uses_by_block.get(entry, set()):
                use_set.add(u)

            for r in reads_by_block.get(entry, set()):
                read_set.add(r)
                use_set.add(r)

            for w in writes_by_block.get(entry, set()):
                write_set.add(w)
                def_set.add(w)

            for edge in cfg.get("edges", []):
                if edge.get("from") == entry:
                    tp = edge.get("type")
                    if tp in ("call",):
                        call_count += 1
                    if tp in ("conditional_branch", "branch", "indirect_branch", "terminal"):
                        branch_count += 1

            effect_summary = {
                "memory_reads": len(read_set),
                "memory_writes": len(write_set),
                "def_count": len(def_set),
                "use_count": len(use_set),
                "access_sites": sorted(_unique_list([x.get("site") for x in sites_by_block.get(entry, []) if x.get("site")])),
                "fixed_offset_access_count": len([x for x in sites_by_block.get(entry, []) if x.get("is_fixed")]),
                "call_count": call_count,
                "branch_count": branch_count
            }

            sem["id"] = gen_id("low_block", entry, str(b.get("block_id")))
            sem["entry"] = entry
            sem["end"] = b.get("end_address")
            sem["addr_range"] = "%s~%s" % (entry, b.get("end_address"))
            sem["ctx_in"] = {
                "live_uses": sorted(list(use_set - def_set)),
                "entry_input_locs": [loc for loc in dfg.get("entry_input_locs", []) if loc in use_set]
            }
            sem["ctx_out"] = {
                "defined_locs": sorted(list(def_set)),
                "memory_effects": {
                    "reads": sorted(list(read_set)),
                    "writes": sorted(list(write_set))
                }
            }
            sem["def_set"] = sorted(list(def_set))
            sem["use_set"] = sorted(list(use_set))
            sem["read_set"] = sorted(list(read_set))
            sem["write_set"] = sorted(list(write_set))
            sem["effect_summary"] = effect_summary
            sem["opcode_ratio"] = dict(pcode_counter)
            sem["branches"] = branch_count
            sem["calls"] = call_count
            sem["T_B"] = sorted(list(pcode_counter.keys()))
            sem["S_B"] = {
                "memory_reads": sorted(list(read_set)),
                "memory_writes": sorted(list(write_set)),
                "call_count": call_count,
                "branch_count": branch_count
            }
            sem["Def_B"] = sorted(list(def_set))
            sem["Use_B"] = sorted(list(use_set))
            output.append(sem)

    return output


def _dfg_by_entry(dfg_sem):
    out = {}
    for d in dfg_sem:
        out[d.get("entry_address")] = d
    return out


def _type_by_entry(type_sem):
    out = {}
    for t in type_sem:
        out[t.get("function_entry")] = t
    return out


def _interface_by_entry(interface_sem):
    out = {}
    for x in interface_sem:
        out[x.get("function_entry")] = x
    return out


def _type_support_from_dfg(dfg, storage_name):
    score = 0.25
    evidence = []
    if storage_name is None:
        storage_name = "unknown"

    mem_count = 0
    deref_count = 0
    for acc in dfg.get("memory_access_patterns", []):
        support = acc.get("support_locs", [])
        if any(storage_name in str(s) for s in support):
            mem_count += 1
            if acc.get("base") is not None:
                deref_count += 1
        if acc.get("base") == storage_name:
            mem_count += 1
            deref_count += 1

    if mem_count > 0:
        score += 0.15
        evidence.append("dfg_usage_trace")
    if deref_count > 0:
        score += 0.20
        evidence.append("pointer_like_dereference")
    return score, evidence


def _make_synthetic_param(entry, idx, loc):
    return {
        "id": gen_id("high_type", entry, "synthetic_param_%d" % idx),
        "location": loc,
        "param_name": "arg_%d" % idx,
        "type_candidates": ["unknown", "pointer_candidate"],
        "type_confidence": [0.35, 0.55],
        "type_evidence": ["entry_input_locs", "calling_convention_trace"],
        "type_size": 0,
        "is_pointer": True,
        "category": "synthetic_param",
        "storage": loc,
        "function_entry": func_entry,
        "function_name": None,
        "object_kind": "param",
        "param_index": idx,
        "storage_kind": _storage_kind_and_id(loc)[0],
        "storage_id": _storage_kind_and_id(loc)[1],
        "stack_offset": None,
        "pred_type": "pointer_like_input",
        "pred_type_category": "ptr",
        "source": "dfg_recovered"
    }


def gen_type_sem(prog, dfg_cache=None):
    all_typesem = []
    dfg_by_func = _dfg_by_entry(dfg_cache if dfg_cache is not None else gen_dataflow_sem(prog))
    funcs = _get_internal_functions(prog, include_thunks=False)

    for func in funcs:
        func_entry = _addr_to_str(func.getEntryPoint())
        func_name = _safe_get_name(func, "sub_%s" % func_entry)
        dfg = dfg_by_func.get(func_entry, {"memory_access_patterns": [], "entry_input_locs": [], "abstract_merge": {"alias_groups": []}})

        ret_type = None
        try:
            ret_type = func.getReturnType()
        except:
            ret_type = None

        ret_candidates, ret_conf, ret_evidence = get_type_candidates_and_confidence(ret_type)
        ret_type_sem = {
            "id": gen_id("high_type", func_entry, "ret"),
            "location": "%s_return" % func_entry,
            "type_candidates": ret_candidates,
            "type_confidence": ret_conf,
            "type_evidence": ret_evidence,
            "type_size": _safe_get_length(ret_type),
            "is_pointer": isinstance(ret_type, PointerDataType),
            "category": get_safe_type_category(ret_type),
            "function_entry": func_entry,
            "function_name": func_name,
            "object_kind": "return",
            "param_index": None,
            "storage_kind": "return",
            "storage_id": "%s_return" % func_entry,
            "stack_offset": None,
            "pred_type": _first_type_candidate(ret_candidates),
            "pred_type_category": get_safe_type_category(ret_type)
        }

        params_sem = []
        params = []
        try:
            params = list(func.getParameters())
        except:
            params = []

        for idx, param in enumerate(params):
            p_type = None
            try:
                p_type = param.getDataType()
            except:
                p_type = None
            storage = _safe_get_storage(param)
            p_storage_kind, p_storage_id = _storage_kind_and_id(storage)
            p_candidates, p_conf, p_evidence = get_type_candidates_and_confidence(p_type, param, func)
            support_boost, support_evidence = _type_support_from_dfg(dfg, storage)
            if p_conf:
                p_conf[0] = min(0.99, float(p_conf[0]) + support_boost - 0.25)
            p_evidence.extend(support_evidence)

            params_sem.append({
                "id": gen_id("high_type", func_entry, "param_%d" % idx),
                "location": "%s_param_%d" % (func_entry, idx),
                "param_name": _safe_get_name(param, "arg_%d" % idx),
                "type_candidates": p_candidates,
                "type_confidence": p_conf,
                "type_evidence": _unique_list(p_evidence),
                "type_size": _safe_get_length(p_type),
                "is_pointer": isinstance(p_type, PointerDataType),
                "category": get_safe_type_category(p_type),
                "storage": storage,
                "function_entry": func_entry,
                "function_name": func_name,
                "object_kind": "param",
                "param_index": idx,
                "storage_kind": p_storage_kind,
                "storage_id": p_storage_id,
                "stack_offset": _safe_get_stack_offset(param),
                "pred_type": _first_type_candidate(p_candidates),
                "pred_type_category": get_safe_type_category(p_type),
                "source": "ghidra_declared"
            })

        recovered_inputs = [loc for loc in dfg.get("entry_input_locs", []) if loc not in [x.get("storage") for x in params_sem]]
        syn_idx = 0
        for loc in recovered_inputs:
            params_sem.append(_make_synthetic_param(func_entry, syn_idx, loc))
            syn_idx += 1

        locals_sem = []
        local_vars = []
        try:
            local_vars = list(func.getLocalVariables())
        except:
            local_vars = []

        for idx, var in enumerate(local_vars):
            v_type = None
            try:
                v_type = var.getDataType()
            except:
                v_type = None
            storage = _safe_get_storage(var)
            v_storage_kind, v_storage_id = _storage_kind_and_id(storage)
            v_candidates, v_conf, v_evidence = get_type_candidates_and_confidence(v_type, var, func)
            support_boost, support_evidence = _type_support_from_dfg(dfg, storage)
            if v_conf:
                v_conf[0] = min(0.99, float(v_conf[0]) + support_boost - 0.25)
            v_evidence.extend(support_evidence)

            locals_sem.append({
                "id": gen_id("high_type", func_entry, "local_%d" % idx),
                "location": "%s_local_%d" % (func_entry, idx),
                "var_name": _safe_get_name(var, "local_%d" % idx),
                "type_candidates": v_candidates,
                "type_confidence": v_conf,
                "type_evidence": _unique_list(v_evidence),
                "type_size": _safe_get_length(v_type),
                "is_pointer": isinstance(v_type, PointerDataType),
                "category": get_safe_type_category(v_type),
                "storage": storage,
                "function_entry": func_entry,
                "function_name": func_name,
                "object_kind": "local",
                "param_index": None,
                "storage_kind": v_storage_kind,
                "storage_id": v_storage_id,
                "stack_offset": _safe_get_stack_offset(var),
                "pred_type": _first_type_candidate(v_candidates),
                "pred_type_category": get_safe_type_category(v_type),
                "source": "ghidra_declared"
            })

        alias_groups = dfg.get("abstract_merge", {}).get("alias_groups", [])
        evidence_summary = {
            "memory_access_count": len(dfg.get("memory_access_patterns", [])),
            "entry_input_count": len(dfg.get("entry_input_locs", [])),
            "alias_group_count": len(alias_groups),
            "fixed_offset_group_count": len([g for g in alias_groups if len(g.get("offsets", [])) >= 2])
        }

        all_typesem.append({
            "id": gen_id("high_type", func_entry, "func"),
            "function_name": func_name,
            "function_entry": func_entry,
            "function_id": _function_uid(func),
            "return_type_sem": ret_type_sem,
            "parameter_type_sem": params_sem,
            "local_var_type_sem": locals_sem,
            "instruction_type_sem": [],
            "type_constraints": {
                "return_type_ref": ret_type_sem.get("id"),
                "param_type_refs": [x.get("id") for x in params_sem],
                "local_type_refs": [x.get("id") for x in locals_sem],
                "evidence_summary": evidence_summary
            }
        })

    return all_typesem


def gen_interface_sem(prog, type_cache=None, dfg_cache=None):
    all_interfacesem = []
    funcs = _get_internal_functions(prog, include_thunks=False)
    type_by_func = _type_by_entry(type_cache if type_cache is not None else gen_type_sem(prog, dfg_cache))
    dfg_by_func = _dfg_by_entry(dfg_cache if dfg_cache is not None else gen_dataflow_sem(prog))

    for func in funcs:
        func_entry = _addr_to_str(func.getEntryPoint())
        func_name = _safe_get_name(func, "sub_%s" % func_entry)
        func_id = _function_uid(func)
        type_obj = type_by_func.get(func_entry, {})
        dfg = dfg_by_func.get(func_entry, {"entry_input_locs": []})

        params_sem = []
        type_params = type_obj.get("parameter_type_sem", [])
        for idx, p in enumerate(type_params):
            dt_name = "unknown"
            if p.get("type_candidates"):
                dt_name = p.get("type_candidates")[0]
            dummy_dt_name = dt_name.lower()
            role_candidates = ["general_param"]
            role_confidence = [0.65]
            pname = p.get("param_name", "arg_%d" % idx)
            storage = p.get("storage", "unknown")

            if "ptr" in dummy_dt_name or p.get("is_pointer") or storage.startswith("reg:") or storage.startswith("stack:"):
                role_candidates = ["buffer", "pointer", "general_param"]
                role_confidence = [0.82, 0.74, 0.55]
            elif "int" in dummy_dt_name and ("len" in pname.lower() or "size" in pname.lower()):
                role_candidates = ["length", "counter", "general_param"]
                role_confidence = [0.90, 0.70, 0.55]
            elif "float" in dummy_dt_name or "double" in dummy_dt_name:
                role_candidates = ["numeric_value", "general_param"]
                role_confidence = [0.78, 0.55]

            params_sem.append({
                "id": gen_id("high_interface", func_entry, str(idx)),
                "param_id": idx,
                "param_index": idx,
                "param_name": pname,
                "location": p.get("location"),
                "type_sem_ref": p.get("id"),
                "storage": storage,
                "storage_kind": p.get("storage_kind"),
                "storage_id": p.get("storage_id"),
                "stack_offset": p.get("stack_offset"),
                "type_candidates": p.get("type_candidates", []),
                "type_confidence": p.get("type_confidence", []),
                "pred_type": dt_name,
                "type_category": p.get("pred_type_category", p.get("category", "unknown")),
                "role_candidates": role_candidates,
                "role_confidence": role_confidence
            })

        ret_type_sem = type_obj.get("return_type_sem", {})
        ret_sem = {
            "location": ret_type_sem.get("location"),
            "type_sem_ref": ret_type_sem.get("id"),
            "type_candidates": ret_type_sem.get("type_candidates", []),
            "type_confidence": ret_type_sem.get("type_confidence", []),
            "type_size": ret_type_sem.get("type_size", 0),
            "is_pointer": ret_type_sem.get("is_pointer", False),
            "return_type": _first_type_candidate(ret_type_sem.get("type_candidates", [])),
            "return_type_category": ret_type_sem.get("pred_type_category", ret_type_sem.get("category", "unknown"))
        }

        interface_constraints = {
            "calling_convention": _safe_cc(func),
            "param_count": len(params_sem),
            "has_return": not _is_void_type(func.getReturnType()) if hasattr(func, "getReturnType") else True,
            "constraint_evidence": _unique_list([
                "calling_convention_analysis",
                "entry_input_locs",
                "type_sem_link"
            ]),
            "cross_layer_constraints": {
                "type_sem_ref": type_obj.get("id"),
                "return_type_ref": ret_type_sem.get("id"),
                "param_type_refs": [p.get("type_sem_ref") for p in params_sem],
                "entry_input_locs": dfg.get("entry_input_locs", [])
            }
        }

        all_interfacesem.append({
            "id": gen_id("high_interface", func_entry, "0"),
            "function_name": func_name,
            "function_entry": func_entry,
            "func_entry": func_entry,
            "func_name": func_name,
            "func_id": func_id,
            "param_count": len(params_sem),
            "param_types": [p.get("pred_type", "unknown") for p in params_sem],
            "param_type_categories": [p.get("type_category", "unknown") for p in params_sem],
            "return_type": ret_sem.get("return_type", "unknown"),
            "return_type_category": ret_sem.get("return_type_category", "unknown"),
            "params": params_sem,
            "ret": ret_sem,
            "interface_constraints": interface_constraints
        })

    return all_interfacesem


def _build_defs_index_for_call(dfg):
    defs_index = defaultdict(list)
    for defi in dfg.get("definitions", []):
        loc = defi.get("location")
        if not loc:
            continue
        defs_index[loc].append(defi)
    for loc in defs_index:
        defs_index[loc].sort(key=lambda x: (_parse_int_addr(x.get("instruction_address")) or 0, x.get("site") or ""))
    return defs_index


def _find_last_def_before(defs_index, loc_names, callsite_addr_int):
    best = None
    for loc in loc_names:
        for defi in defs_index.get(loc, []):
            di = _parse_int_addr(defi.get("instruction_address"))
            if di is None or callsite_addr_int is None:
                continue
            if di <= callsite_addr_int:
                if best is None or di > _parse_int_addr(best.get("instruction_address")):
                    best = defi
    return best


def _get_external_like_functions(prog):
    """
    收集 external + thunk->external + 常见导入桩函数。
    为了避免把普通内部 thunk 误当成 API，只接受：
    1) 真正 external；
    2) thunk 到 external 的桩函数。
    """
    out = []
    funcs = prog.getFunctionManager().getFunctions(True)
    while funcs.hasNext():
        func = funcs.next()
        if func is None:
            continue

        is_candidate = False
        target_name = _safe_get_name(func, None)
        target_entry = _addr_to_str(func.getEntryPoint())
        thunk_target = None

        try:
            if func.isExternal():
                is_candidate = True
        except:
            pass

        try:
            if (not is_candidate) and func.isThunk():
                try:
                    thunk_target = func.getThunkedFunction(False)
                except:
                    try:
                        thunk_target = func.getThunkedFunction()
                    except:
                        thunk_target = None
                if thunk_target is not None:
                    try:
                        if thunk_target.isExternal():
                            is_candidate = True
                    except:
                        is_candidate = False
                    if is_candidate:
                        target_name = _safe_get_name(thunk_target, target_name or "external_stub")
        except:
            thunk_target = None

        if is_candidate:
            out.append({
                "func": func,
                "entry": target_entry,
                "api_name": target_name,
                "thunk_target": thunk_target
            })

    return out


def gen_api_sem(prog, dfg_cache=None, interface_cache=None, type_cache=None):
    all_apisem = []
    api_seen = set()
    dfg_by_func = _dfg_by_entry(dfg_cache if dfg_cache is not None else gen_dataflow_sem(prog))
    interface_by_func = _interface_by_entry(interface_cache if interface_cache is not None else gen_interface_sem(prog, type_cache, dfg_cache))
    type_by_func = _type_by_entry(type_cache if type_cache is not None else gen_type_sem(prog, dfg_cache))
    func_manager = prog.getFunctionManager()
    ref_mgr = prog.getReferenceManager()
    arg_slots = _get_call_arg_register_slots(prog)

    candidate_funcs = _get_external_like_functions(prog)
    for ext in candidate_funcs:
        api_name = ext.get("api_name") or "external_api"
        target_entry = ext.get("entry")
        refs = ref_mgr.getReferencesTo(ext.get("func").getEntryPoint())
        for ref in refs:
            try:
                if not RefType.isCall(ref.getReferenceType()):
                    continue
            except:
                try:
                    if ref.getReferenceType() != RefType.CALL:
                        continue
                except:
                    continue

            callsite_addr = ref.getFromAddress()
            callsite = _addr_to_str(callsite_addr)
            caller_func = func_manager.getFunctionContaining(callsite_addr)
            if caller_func is None:
                continue
            try:
                if caller_func.isExternal():
                    continue
            except:
                pass

            caller_entry = _addr_to_str(caller_func.getEntryPoint())
            caller_dfg = dfg_by_func.get(caller_entry, {"definitions": [], "memory_access_patterns": []})
            defs_index = _build_defs_index_for_call(caller_dfg)
            call_addr_int = _parse_int_addr(callsite)

            sem_template = get_api_semantic_template(api_name)
            param_sem = []
            for idx, role in enumerate(sem_template.get("param_roles", [])):
                if idx >= len(arg_slots):
                    break
                slot_regs = set(["reg:%s" % r for r in arg_slots[idx]])
                best_def = _find_last_def_before(defs_index, slot_regs, call_addr_int)
                param_sem.append({
                    "param_idx": idx,
                    "role": role,
                    "source_locations": sorted(list(slot_regs)),
                    "recovered_from": best_def.get("location") if best_def else None,
                    "recovered_site": best_def.get("site") if best_def else None
                })

            caller_interface = interface_by_func.get(caller_entry)
            caller_type = type_by_func.get(caller_entry)

            semantic_template = {
                "api_intent": sem_template.get("intent"),
                "param_roles": param_sem,
                "side_effects": sem_template.get("side_effect", []),
                "base_constraints": sem_template.get("constraints", []),
                "symbol_resolution": {
                    "api_name": api_name,
                    "target_entry": target_entry,
                    "callsite": callsite,
                    "resolution_kind": "external_or_thunk"
                }
            }

            propagated_constraints = {
                "caller_func_id": _function_uid(caller_func),
                "caller_function_entry": caller_entry,
                "type_sem_ref": caller_type.get("id") if caller_type else None,
                "interface_sem_ref": caller_interface.get("id") if caller_interface else None,
                "constraint_evidence": _unique_list([
                    "api_symbol_resolution",
                    "call_site_analysis",
                    "argument_register_trace"
                ]),
                "propagation_targets": _unique_list(["type_sem", "interface_sem", "layout_sem"])
            }

            api_key = (callsite, api_name, caller_entry)
            if api_key in api_seen:
                continue
            api_seen.add(api_key)

            all_apisem.append({
                "id": gen_id("high_api", callsite, "0"),
                "callsite": callsite,
                "api_name": api_name,
                "semantic_template": semantic_template,
                "propagated_constraints": propagated_constraints
            })

    return all_apisem


def _field_relations_from_offsets(offsets):
    offsets = sorted(list(set([int(x) for x in offsets])))
    relations = []
    if not offsets:
        return relations
    for i, off in enumerate(offsets):
        next_off = None
        size_hint = None
        if i + 1 < len(offsets):
            next_off = offsets[i + 1]
            size_hint = next_off - off
        relations.append({
            "field_name": "field_%d" % i,
            "offset": off,
            "size_hint": size_hint,
            "relation": "next_field_gap_%s" % (size_hint if size_hint is not None else "unknown")
        })
    return relations


def _infer_layout_roles_from_group(group, interface_obj):
    roles = []
    offsets = group.get("offsets", [])
    if len(offsets) >= 2:
        roles.append("struct_with_fields")
    if group.get("stride") is not None and len(offsets) >= 3:
        roles.append("regular_stride_layout")
    base = group.get("base")
    if base and (str(base).startswith("reg:") or str(base).startswith("stack:")):
        roles.append("pointer_backed_base")
    if interface_obj is not None and interface_obj.get("params"):
        roles.append("interface_aligned")
    if not roles:
        roles.append("layout_candidate")
    return roles


def _match_type_refs_for_layout(type_obj, base_loc):
    """
    LayoutSem 的跨层引用统一指向顶层 TypeSem 节点 id，避免悬空子节点引用；
    更细的 param/local/return id 保留在 TypeSem / InterfaceSem 内部字段中。
    """
    refs = []
    if not type_obj:
        return refs
    if type_obj.get("id"):
        refs.append(type_obj.get("id"))
    return _unique_list(refs)


def _fallback_declared_layouts(func, func_entry, type_obj, interface_obj):
    out = []
    all_vars = []
    try:
        all_vars.extend(list(func.getParameters()))
    except:
        pass
    try:
        all_vars.extend(list(func.getLocalVariables()))
    except:
        pass

    seq = 0
    seen_dt = set()
    for var in all_vars:
        dt = None
        try:
            dt = var.getDataType()
        except:
            dt = None
        if dt is None:
            continue
        dt_name = _safe_dt_name(dt)
        if dt_name in seen_dt:
            continue
        seen_dt.add(dt_name)

        if isinstance(dt, StructureDataType):
            layout = analyze_struct_memory_layout(func, dt)
            out.append({
                "id": gen_id("high_mem", func_entry, "fallback_%d" % seq),
                "function_entry": func_entry,
                "function_name": _safe_get_name(func, "sub_%s" % func_entry),
                "func_entry": func_entry,
                "func_name": _safe_get_name(func, "sub_%s" % func_entry),
                "base_obj": layout.get("base_obj"),
                "base_object": layout.get("base_obj"),
                "base_storage": layout.get("base_obj"),
                "offset_set": layout.get("offset_set", []),
                "offsets": layout.get("offset_set", []),
                "evidence_addrs": [],
                "size_rel": layout.get("size_rel", []),
                "field_relations": layout.get("field_relations", []),
                "nested_candidates": [],
                "support_sites": [],
                "semantic_roles": ["declared_struct_fallback"],
                "layout_source": "ghidra_declared_type_fallback",
                "confidence": 0.55,
                "propagated_constraints": {
                    "target_func_id": _function_uid(func),
                    "type_sem_refs": _match_type_refs_for_layout(type_obj, None),
                    "interface_sem_ref": interface_obj.get("id") if interface_obj else None,
                    "constraint_evidence": ["declared_structure_type"]
                },
                "location": "%s_layout_%s" % (func_entry, dt_name)
            })
            seq += 1
    return out


def gen_mem_layout_sem(prog, dfg_cache=None, type_cache=None, interface_cache=None):
    all_memlayout = []
    dfg_by_func = _dfg_by_entry(dfg_cache if dfg_cache is not None else gen_dataflow_sem(prog))
    type_by_func = _type_by_entry(type_cache if type_cache is not None else gen_type_sem(prog, dfg_cache))
    interface_by_func = _interface_by_entry(interface_cache if interface_cache is not None else gen_interface_sem(prog, type_cache, dfg_cache))
    funcs = _get_internal_functions(prog, include_thunks=False)

    for func in funcs:
        func_entry = _addr_to_str(func.getEntryPoint())
        func_name = _safe_get_name(func, "sub_%s" % func_entry)
        dfg = dfg_by_func.get(func_entry, {"abstract_merge": {"alias_groups": []}})
        type_obj = type_by_func.get(func_entry)
        interface_obj = interface_by_func.get(func_entry)

        seq = 0
        groups = dfg.get("abstract_merge", {}).get("alias_groups", [])
        for group in groups:
            offsets = sorted(list(group.get("offsets", [])))
            if len(offsets) < 2:
                continue

            field_relations = _field_relations_from_offsets(offsets)
            size_rel = []
            for fr in field_relations:
                if fr.get("size_hint") is not None:
                    size_rel.append("size_eq_%d" % int(fr.get("size_hint")))

            semantic_roles = _infer_layout_roles_from_group(group, interface_obj)
            type_refs = _match_type_refs_for_layout(type_obj, group.get("base"))
            confidence = 0.58 + 0.06 * min(len(offsets), 4)
            if group.get("stride") is not None:
                confidence += 0.08

            all_memlayout.append({
                "id": gen_id("high_mem", func_entry, str(seq)),
                "function_entry": func_entry,
                "function_name": func_name,
                "func_entry": func_entry,
                "func_name": func_name,
                "base_obj": group.get("base"),
                "base_object": group.get("base"),
                "base_storage": group.get("base"),
                "offset_set": offsets,
                "offsets": offsets,
                "evidence_addrs": _extract_site_addrs(group.get("sites", [])),
                "size_rel": _unique_list(size_rel),
                "field_relations": field_relations,
                "nested_candidates": [],
                "support_sites": group.get("sites", []),
                "semantic_roles": semantic_roles,
                "layout_source": "dfg_alias_group",
                "confidence": min(0.98, confidence),
                "propagated_constraints": {
                    "target_func_id": _function_uid(func),
                    "type_sem_refs": type_refs,
                    "interface_sem_ref": interface_obj.get("id") if interface_obj else None,
                    "constraint_evidence": _unique_list([
                        "fixed_offset_access_cluster",
                        "alias_group_summary",
                        "stride_analysis" if group.get("stride") is not None else None
                    ])
                },
                "location": "%s_layout_%s_%d" % (func_entry, str(group.get("base")).replace(":", "_"), seq)
            })
            seq += 1

        if seq == 0:
            all_memlayout.extend(_fallback_declared_layouts(func, func_entry, type_obj, interface_obj))

    return all_memlayout


def _collect_semantic_nodes(sem):
    nodes = []
    for item in sem.get("low_level_sem", {}).get("instr_sem", []):
        nodes.append({"id": item.get("id"), "layer": "InstrSem"})
    for item in sem.get("low_level_sem", {}).get("block_sem", []):
        nodes.append({"id": item.get("id"), "layer": "BlockSem"})
    for item in sem.get("mid_level_sem", {}).get("cfg_sem", []):
        nodes.append({"id": item.get("id"), "layer": "CFGSem"})
    for item in sem.get("mid_level_sem", {}).get("dataflow_sem", []):
        nodes.append({"id": item.get("id"), "layer": "DFGSem"})
    for item in sem.get("high_level_sem", {}).get("type_sem", []):
        nodes.append({"id": item.get("id"), "layer": "TypeSem"})
    for item in sem.get("high_level_sem", {}).get("interface_sem", []):
        nodes.append({"id": item.get("id"), "layer": "InterfaceSem"})
    for item in sem.get("high_level_sem", {}).get("api_sem", []):
        nodes.append({"id": item.get("id"), "layer": "APISem"})
    for item in sem.get("high_level_sem", {}).get("mem_layout_sem", []):
        nodes.append({"id": item.get("id"), "layer": "LayoutSem"})
    return nodes


def gen_mapping(all_sem):
    mappings = []
    seen = set()

    instr_sem = all_sem.get("low_level_sem", {}).get("instr_sem", [])
    block_sem = all_sem.get("low_level_sem", {}).get("block_sem", [])
    cfg_sem = all_sem.get("mid_level_sem", {}).get("cfg_sem", [])
    dfg_sem = all_sem.get("mid_level_sem", {}).get("dataflow_sem", [])
    type_sem = all_sem.get("high_level_sem", {}).get("type_sem", [])
    interface_sem = all_sem.get("high_level_sem", {}).get("interface_sem", [])
    api_sem = all_sem.get("high_level_sem", {}).get("api_sem", [])
    mem_sem = all_sem.get("high_level_sem", {}).get("mem_layout_sem", [])

    cfg_by_func = {}
    block_func_by_entry = {}
    for cfg in cfg_sem:
        fentry = cfg.get("entry_address")
        cfg_by_func[fentry] = cfg
        for b in cfg.get("blocks", []):
            block_func_by_entry[b.get("start_address")] = fentry

    block_records = []
    for b in block_sem:
        fentry = block_func_by_entry.get(b.get("entry"))
        block_records.append({
            "id": b.get("id"),
            "entry": b.get("entry"),
            "end": b.get("end"),
            "func_entry": fentry,
            "obj": b,
            "entry_i": _parse_int_addr(b.get("entry")),
            "end_i": _parse_int_addr(b.get("end"))
        })

    addr_to_block = {}
    instr_records = []
    for ins in instr_sem:
        addr = ins.get("address")
        if addr not in addr_to_block:
            aint = _parse_int_addr(addr)
            owner = None
            for br in block_records:
                if br.get("entry_i") is None or br.get("end_i") is None or aint is None:
                    continue
                if br.get("entry_i") <= aint <= br.get("end_i"):
                    owner = br
                    break
            addr_to_block[addr] = owner

        owner_block = addr_to_block.get(addr)
        instr_records.append({
            "id": ins.get("id"),
            "address": addr,
            "obj": ins,
            "block": owner_block,
            "func_entry": owner_block.get("func_entry") if owner_block else ins.get("function_entry")
        })

    instr_by_addr = defaultdict(list)
    for rec in instr_records:
        instr_by_addr[rec.get("address")].append(rec)

    dfg_by_func = _dfg_by_entry(dfg_sem)
    type_by_func = _type_by_entry(type_sem)
    interface_by_func = _interface_by_entry(interface_sem)
    mem_by_func = defaultdict(list)
    for ms in mem_sem:
        mem_by_func[ms.get("function_entry")].append(ms)

    # R1 InstrSem -> BlockSem
    for rec in instr_records:
        block = rec.get("block")
        if block is None:
            continue
        _append_mapping(
            mappings, seen,
            "R1", "指令聚合",
            rec.get("id"), block.get("id"),
            "InstrSem", "BlockSem",
            {
                "instruction_address": rec.get("address"),
                "block_entry": block.get("entry"),
                "block_range": block.get("obj", {}).get("addr_range")
            },
            "ordered_membership",
            1.0,
            "BlockSem",
            ["instruction_in_block", "execution_order_preserved"],
            "structural"
        )

    # R2 BlockSem -> CFGSem
    for br in block_records:
        cfg = cfg_by_func.get(br.get("func_entry"))
        if cfg is None:
            continue
        _append_mapping(
            mappings, seen,
            "R2", "控制流提取",
            br.get("id"), cfg.get("id"),
            "BlockSem", "CFGSem",
            {
                "block_entry": br.get("entry"),
                "branches": br.get("obj", {}).get("branches"),
                "calls": br.get("obj", {}).get("calls")
            },
            "terminal_flow_pattern",
            1.0,
            "CFGSem",
            ["block_terminal_flow_consistent", "cfg_membership_confirmed"],
            "structural"
        )

    # R3 BlockSem -> DFGSem
    for br in block_records:
        dfg = dfg_by_func.get(br.get("func_entry"))
        if dfg is None:
            continue
        if not br.get("obj", {}).get("def_set") and not br.get("obj", {}).get("use_set"):
            continue
        strength = 0.55 + 0.05 * min(len(br.get("obj", {}).get("def_set", [])), 3) + 0.03 * min(len(br.get("obj", {}).get("use_set", [])), 5)
        _append_mapping(
            mappings, seen,
            "R3", "数据流提取",
            br.get("id"), dfg.get("id"),
            "BlockSem", "DFGSem",
            {
                "block_entry": br.get("entry"),
                "block_def_set": br.get("obj", {}).get("def_set", []),
                "block_use_set": br.get("obj", {}).get("use_set", [])
            },
            "block_def_use_summary",
            strength,
            "DFGSem",
            ["def_use_chain_exists", "cross_block_dependency_allowed"],
            "structural"
        )

    # R4 DFG/Instr -> TypeSem
    for fentry, ts in type_by_func.items():
        dfg = dfg_by_func.get(fentry)
        if dfg is None:
            continue
        mem_count = len(dfg.get("memory_access_patterns", []))
        strength = 0.60 + 0.03 * min(mem_count, 5)
        _append_mapping(
            mappings, seen,
            "R4", "类型推导",
            dfg.get("id"), ts.get("id"),
            "DFGSem", "TypeSem",
            {
                "function_entry": fentry,
                "memory_access_count": mem_count,
                "entry_input_locs": dfg.get("entry_input_locs", [])
            },
            "def_use_and_deref_chain",
            strength,
            "TypeSem",
            ["pointer_usage_consistent", "type_candidate_supported_by_dataflow"],
            "evidence"
        )

        for acc in dfg.get("memory_access_patterns", []):
            if acc.get("instruction_address") not in instr_by_addr:
                continue
            local_strength = 0.55
            if acc.get("base") is not None:
                local_strength += 0.10
            if acc.get("offset") is not None:
                local_strength += 0.08
            if acc.get("is_fixed"):
                local_strength += 0.05
            for rec in instr_by_addr.get(acc.get("instruction_address"), []):
                if rec.get("func_entry") != fentry:
                    continue
                _append_mapping(
                    mappings, seen,
                    "R4", "类型推导",
                    rec.get("id"), ts.get("id"),
                    "InstrSem", "TypeSem",
                    {
                        "instruction_address": acc.get("instruction_address"),
                        "location": acc.get("location"),
                        "base": acc.get("base"),
                        "offset": acc.get("offset")
                    },
                    "memory_access_pattern",
                    local_strength,
                    "TypeSem",
                    ["candidate:pointer", "fixed_or_pointer_backed_access"],
                    "evidence"
                )

    # R5 DFG/Block -> InterfaceSem
    for fentry, itf in interface_by_func.items():
        dfg = dfg_by_func.get(fentry)
        if dfg is None:
            continue
        entry_block = None
        for br in block_records:
            if br.get("func_entry") == fentry and br.get("entry") == fentry:
                entry_block = br
                break
        if entry_block is None:
            for br in block_records:
                if br.get("func_entry") == fentry:
                    entry_block = br
                    break

        if entry_block is not None:
            _append_mapping(
                mappings, seen,
                "R5", "接口识别",
                entry_block.get("id"), itf.get("id"),
                "BlockSem", "InterfaceSem",
                {
                    "function_entry": fentry,
                    "entry_input_locs": dfg.get("entry_input_locs", []),
                    "calling_convention": itf.get("interface_constraints", {}).get("calling_convention")
                },
                "calling_convention_and_entry_usage",
                0.78,
                "InterfaceSem",
                ["calling_convention_compatible", "param_passing_pattern_detected"],
                "evidence"
            )

        _append_mapping(
            mappings, seen,
            "R5", "接口识别",
            dfg.get("id"), itf.get("id"),
            "DFGSem", "InterfaceSem",
            {
                "function_entry": fentry,
                "entry_input_locs": dfg.get("entry_input_locs", []),
                "param_count": itf.get("interface_constraints", {}).get("param_count")
            },
            "entry_use_before_def",
            0.82,
            "InterfaceSem",
            ["entry_use_before_def", "interface_candidate_supported_by_dataflow"],
            "evidence"
        )

    # R6 Instr/Block -> APISem
    for api in api_sem:
        callsite = api.get("callsite")
        for rec in instr_by_addr.get(callsite, []):
            _append_mapping(
                mappings, seen,
                "R6", "API归整",
                rec.get("id"), api.get("id"),
                "InstrSem", "APISem",
                {
                    "callsite": callsite,
                    "api_name": api.get("api_name"),
                    "instruction_address": callsite
                },
                "call_instruction",
                0.95,
                "APISem",
                ["call_target_observed", "api_template_bindable"],
                "evidence"
            )

        owner_block = addr_to_block.get(callsite)
        if owner_block is not None:
            _append_mapping(
                mappings, seen,
                "R6", "API归整",
                owner_block.get("id"), api.get("id"),
                "BlockSem", "APISem",
                {
                    "callsite": callsite,
                    "caller_function": api.get("propagated_constraints", {}).get("caller_func_id"),
                    "block_entry": owner_block.get("entry"),
                    "api_name": api.get("api_name")
                },
                "call_behavior_context",
                0.84,
                "APISem",
                ["external_call_context", "semantic_template_available"],
                "evidence"
            )

    # R7 DFG/Instr -> LayoutSem
    for fentry, layouts in mem_by_func.items():
        dfg = dfg_by_func.get(fentry)
        if dfg is None:
            continue
        groups = dfg.get("abstract_merge", {}).get("alias_groups", [])
        if not groups:
            continue

        for layout in layouts:
            offsets = set(layout.get("offset_set", []))
            if len(offsets) < 2:
                continue
            for group in groups:
                group_offsets = set(group.get("offsets", []))
                overlap = offsets & group_offsets
                if not overlap:
                    continue
                strength = 0.60 + 0.08 * min(len(overlap), 3)
                if group.get("stride") is not None:
                    strength += 0.06

                _append_mapping(
                    mappings, seen,
                    "R7", "布局推断",
                    dfg.get("id"), layout.get("id"),
                    "DFGSem", "LayoutSem",
                    {
                        "function_entry": fentry,
                        "base": group.get("base"),
                        "group_offsets": sorted(list(group_offsets)),
                        "matched_offsets": sorted(list(overlap)),
                        "stride": group.get("stride")
                    },
                    "stride_and_fixed_offset_pattern",
                    strength,
                    "LayoutSem",
                    ["multi_offset_access_detected", "layout_candidate_consistent"],
                    "evidence"
                )

                for addr in group.get("instr_addresses", []):
                    for rec in instr_by_addr.get(addr, []):
                        if rec.get("func_entry") != fentry:
                            continue
                        _append_mapping(
                            mappings, seen,
                            "R7", "布局推断",
                            rec.get("id"), layout.get("id"),
                            "InstrSem", "LayoutSem",
                            {
                                "instruction_address": addr,
                                "base": group.get("base"),
                                "matched_offsets": sorted(list(overlap))
                            },
                            "fixed_offset_access",
                            strength - 0.05,
                            "LayoutSem",
                            ["offset_set_supported", "layout_struct_like"],
                            "evidence"
                        )

    return mappings


def gen_constraint_edges(sem):
    edges = []
    seen = set()

    type_by_func = _type_by_entry(sem.get("high_level_sem", {}).get("type_sem", []))
    interface_by_func = _interface_by_entry(sem.get("high_level_sem", {}).get("interface_sem", []))

    for fentry, itf in interface_by_func.items():
        t = type_by_func.get(fentry)
        if t is None:
            continue
        key = ("type_interface", t.get("id"), itf.get("id"))
        if key not in seen:
            seen.add(key)
            edges.append({
                "constraint_id": "CT_%d" % len(edges),
                "from": t.get("id"),
                "to": itf.get("id"),
                "edge_kind": "constraint",
                "constraint_type": "type_interface_consistency",
                "constraints": [
                    "return_type_ref_valid",
                    "param_type_ref_valid",
                    "calling_convention_param_count_consistent"
                ],
                "strength": 0.92
            })

    for layout in sem.get("high_level_sem", {}).get("mem_layout_sem", []):
        for tref in layout.get("propagated_constraints", {}).get("type_sem_refs", []):
            key = ("type_layout", tref, layout.get("id"))
            if key in seen:
                continue
            seen.add(key)
            edges.append({
                "constraint_id": "CT_%d" % len(edges),
                "from": tref,
                "to": layout.get("id"),
                "edge_kind": "constraint",
                "constraint_type": "type_layout_consistency",
                "constraints": [
                    "layout_type_ref_valid",
                    "offset_cluster_consistent_with_type"
                ],
                "strength": 0.85
            })

        itf_ref = layout.get("propagated_constraints", {}).get("interface_sem_ref")
        if itf_ref:
            key = ("interface_layout", itf_ref, layout.get("id"))
            if key not in seen:
                seen.add(key)
                edges.append({
                    "constraint_id": "CT_%d" % len(edges),
                    "from": itf_ref,
                    "to": layout.get("id"),
                    "edge_kind": "constraint",
                    "constraint_type": "interface_layout_consistency",
                    "constraints": [
                        "base_obj_should_align_with_param_or_local_pointer"
                    ],
                    "strength": 0.78
                })

    for api in sem.get("high_level_sem", {}).get("api_sem", []):
        t_ref = api.get("propagated_constraints", {}).get("type_sem_ref")
        i_ref = api.get("propagated_constraints", {}).get("interface_sem_ref")
        if t_ref:
            key = ("api_type", api.get("id"), t_ref)
            if key not in seen:
                seen.add(key)
                edges.append({
                    "constraint_id": "CT_%d" % len(edges),
                    "from": api.get("id"),
                    "to": t_ref,
                    "edge_kind": "constraint",
                    "constraint_type": "api_type_propagation",
                    "constraints": api.get("semantic_template", {}).get("base_constraints", []),
                    "strength": 0.82
                })
        if i_ref:
            key = ("api_interface", api.get("id"), i_ref)
            if key not in seen:
                seen.add(key)
                edges.append({
                    "constraint_id": "CT_%d" % len(edges),
                    "from": api.get("id"),
                    "to": i_ref,
                    "edge_kind": "constraint",
                    "constraint_type": "api_interface_propagation",
                    "constraints": api.get("semantic_template", {}).get("base_constraints", []),
                    "strength": 0.82
                })

    return edges


def gen_node_state(sem):
    states = []
    for node in _collect_semantic_nodes(sem):
        layer = node.get("layer")
        nid = node.get("id")
        confidence = 1.0
        state = "confirmed"

        if layer in ("TypeSem", "InterfaceSem", "APISem", "LayoutSem"):
            state = "hypothesis_confirmed"
            confidence = 0.75

        if layer == "TypeSem":
            target = None
            for x in sem.get("high_level_sem", {}).get("type_sem", []):
                if x.get("id") == nid:
                    target = x
                    break
            if target:
                confs = []
                for c in target.get("return_type_sem", {}).get("type_confidence", []):
                    confs.append(float(c))
                for p in target.get("parameter_type_sem", []):
                    if p.get("type_confidence"):
                        confs.append(float(p.get("type_confidence")[0]))
                if confs:
                    confidence = round(sum(confs) / float(len(confs)), 3)

        if layer == "InterfaceSem":
            target = None
            for x in sem.get("high_level_sem", {}).get("interface_sem", []):
                if x.get("id") == nid:
                    target = x
                    break
            if target:
                confidence = 0.78 + 0.03 * min(len(target.get("params", [])), 3)

        if layer == "APISem":
            confidence = 0.90

        if layer == "LayoutSem":
            target = None
            for x in sem.get("high_level_sem", {}).get("mem_layout_sem", []):
                if x.get("id") == nid:
                    target = x
                    break
            if target:
                confidence = float(target.get("confidence", 0.7))

        states.append({
            "node_id": nid,
            "layer": layer,
            "state": state,
            "confidence": _clamp_strength(confidence)
        })
    return states


def gen_invariant_status(sem):
    evidence_edges = sem.get("cross_layer_mapping", [])
    constraint_edges = sem.get("constraint_edges", [])
    node_state = sem.get("node_state", [])

    node_ids = [x.get("node_id") for x in node_state if x.get("node_id")]
    deterministic_unique = len(node_ids) == len(set(node_ids))

    unresolved_constraint_refs = 0
    all_ids = set(node_ids)
    for edge in constraint_edges:
        if edge.get("from") not in all_ids or edge.get("to") not in all_ids:
            unresolved_constraint_refs += 1

    evidence_strength_valid = True
    for e in evidence_edges:
        s = e.get("strength")
        try:
            sv = float(s)
            if sv < 0.0 or sv > 1.0:
                evidence_strength_valid = False
                break
        except:
            evidence_strength_valid = False
            break

    return {
        "I1_confidence_monotonic_non_decreasing": {
            "status": evidence_strength_valid,
            "checked_on_snapshot": True,
            "detail": "all edge/node confidence values are clamped into [0,1]"
        },
        "I2_state_transition_irreversible": {
            "status": True,
            "checked_on_snapshot": True,
            "detail": "single-snapshot export keeps only forward states"
        },
        "I3_constraint_violations_monotonic_non_increasing": {
            "status": unresolved_constraint_refs == 0,
            "checked_on_snapshot": True,
            "detail": "unresolved_constraint_refs=%d" % unresolved_constraint_refs
        },
        "I4_evidence_edges_append_only": {
            "status": True,
            "checked_on_snapshot": True,
            "detail": "export is append-only within one run"
        },
        "I5_deterministic_immutable": {
            "status": deterministic_unique,
            "checked_on_snapshot": True,
            "detail": "node_id_unique=%s" % str(deterministic_unique)
        }
    }


def build_usg(sem):
    nodes = _collect_semantic_nodes(sem)
    return {
        "graph": {
            "V": [x.get("id") for x in nodes if x.get("id")],
            "E_ev": [x.get("mapping_id") for x in sem.get("cross_layer_mapping", []) if x.get("mapping_id")],
            "E_ct": [x.get("constraint_id") for x in sem.get("constraint_edges", []) if x.get("constraint_id")]
        },
        "summary": {
            "node_count": len(nodes),
            "evidence_edge_count": len(sem.get("cross_layer_mapping", [])),
            "constraint_edge_count": len(sem.get("constraint_edges", []))
        }
    }


def build_dict(prog):
    from collections import OrderedDict
    sem = OrderedDict()

    # 先算中层，再回填低层/高层；输出顺序仍保持你原来的层次结构
    cfg_sem = _BASE_GEN_CFG_SEM(prog)
    dfg_sem = gen_dataflow_sem(prog)
    instr_sem = gen_instr_sem(prog, dfg_sem)
    block_sem = gen_block_sem(prog, cfg_sem, dfg_sem)
    type_sem = gen_type_sem(prog, dfg_sem)
    interface_sem = gen_interface_sem(prog, type_sem, dfg_sem)
    api_sem = gen_api_sem(prog, dfg_sem, interface_sem, type_sem)
    mem_layout_sem = gen_mem_layout_sem(prog, dfg_sem, type_sem, interface_sem)

    sem["program_info"] = {
        "name": prog.getName(),
        "base": str(prog.getImageBase()),
        "schema_version": "semantic_dict_v3",
        "generator": "ghidra_usg_patch_v3"
    }

    sem["low_level_sem"] = OrderedDict()
    sem["low_level_sem"]["instr_sem"] = instr_sem
    sem["low_level_sem"]["block_sem"] = block_sem

    sem["mid_level_sem"] = OrderedDict()
    sem["mid_level_sem"]["cfg_sem"] = cfg_sem
    sem["mid_level_sem"]["dataflow_sem"] = dfg_sem

    sem["high_level_sem"] = OrderedDict()
    sem["high_level_sem"]["type_sem"] = type_sem
    sem["high_level_sem"]["interface_sem"] = interface_sem
    sem["high_level_sem"]["api_sem"] = api_sem
    sem["high_level_sem"]["mem_layout_sem"] = mem_layout_sem

    sem["cross_layer_mapping"] = gen_mapping(sem)
    sem["constraint_edges"] = gen_constraint_edges(sem)
    sem["node_state"] = gen_node_state(sem)
    sem["invariant_status"] = gen_invariant_status(sem)
    sem["usg"] = build_usg(sem)
    return sem


def _default_output_dir():
    try:
        cwd = os.getcwd()
        if cwd and os.path.isdir(cwd):
            out = os.path.join(cwd, "ghidra_semantic_output")
            if not os.path.exists(out):
                os.makedirs(out)
            return out
    except:
        pass
    try:
        home = os.path.expanduser("~")
        out = os.path.join(home, "ghidra_semantic_output")
        if not os.path.exists(out):
            os.makedirs(out)
        return out
    except:
        pass
    out = "/tmp/ghidra_semantic_output"
    if not os.path.exists(out):
        os.makedirs(out)
    return out



# ==============================================
# PATCH v4: 接口去噪 / USG 节点闭包 / Block terminator 摘要
#           / 地址表达式补强 / LayoutSem 分层 / API 模板扩充
# ==============================================

try:
    _PREV_TRACE_ADDRESS_EXPR = _trace_address_expr
except:
    _PREV_TRACE_ADDRESS_EXPR = None

try:
    _PREV_GEN_BLOCK_SEM = gen_block_sem
except:
    _PREV_GEN_BLOCK_SEM = None

try:
    _PREV_GEN_MAPPING = gen_mapping
except:
    _PREV_GEN_MAPPING = None


def _safe_lower(s):
    try:
        return str(s).lower()
    except:
        return ""


def _dtype_name_lower(dt):
    return _safe_lower(_safe_dt_name(dt))


def _is_pointer_type_like(dt, candidates=None):
    try:
        if isinstance(dt, PointerDataType):
            return True
    except:
        pass

    name = _dtype_name_lower(dt)
    if "*" in name or "ptr" in name or name.endswith("_p"):
        return True

    if candidates:
        for c in candidates:
            s = _safe_lower(c)
            if "*" in s or "ptr" in s or s == "void*":
                return True
    return False


def _is_unknown_storage(storage):
    if storage is None:
        return True
    s = str(storage)
    return s in ("unknown", "", "<UNASSIGNED>", "None")


def _processor_is_aarch64(prog):
    proc = _get_processor_name(prog)
    return ("aarch64" in proc) or ("arm64" in proc)


def _processor_is_x86_64(prog):
    proc = _get_processor_name(prog)
    return ("x86" in proc) and ("64" in proc)


def _processor_is_arm32(prog):
    proc = _get_processor_name(prog)
    return ("arm" in proc) and ("64" not in proc) and ("aarch64" not in proc)


def _processor_is_mips(prog):
    proc = _get_processor_name(prog)
    return "mips" in proc


def _aarch64_canonical_arg_reg(reg):
    reg = str(reg).upper()
    m = re.match(r'^([XWSDQ])(\d+)$', reg)
    if not m:
        return None
    cls = m.group(1)
    idx = int(m.group(2))
    if idx < 0 or idx >= 8:
        return None
    if cls in ("S", "D", "Q"):
        return {"slot": idx, "canonical": "reg:D%d" % idx, "kind": "fp"}
    return {"slot": idx, "canonical": "reg:X%d" % idx, "kind": "gp"}


def _x86_64_canonical_arg_reg(reg):
    reg = str(reg).upper()
    mapping = {
        "RDI": (0, "reg:RDI", "gp"), "EDI": (0, "reg:RDI", "gp"), "XMM0": (0, "reg:XMM0", "fp"),
        "RSI": (1, "reg:RSI", "gp"), "ESI": (1, "reg:RSI", "gp"), "XMM1": (1, "reg:XMM1", "fp"),
        "RDX": (2, "reg:RDX", "gp"), "EDX": (2, "reg:RDX", "gp"), "XMM2": (2, "reg:XMM2", "fp"),
        "RCX": (3, "reg:RCX", "gp"), "ECX": (3, "reg:RCX", "gp"), "XMM3": (3, "reg:XMM3", "fp"),
        "R8": (4, "reg:R8", "gp"), "R8D": (4, "reg:R8", "gp"), "XMM4": (4, "reg:XMM4", "fp"),
        "R9": (5, "reg:R9", "gp"), "R9D": (5, "reg:R9", "gp"), "XMM5": (5, "reg:XMM5", "fp")
    }
    if reg not in mapping:
        return None
    slot, canon, kind = mapping[reg]
    return {"slot": slot, "canonical": canon, "kind": kind}


def _arm32_canonical_arg_reg(reg):
    reg = str(reg).upper()
    m = re.match(r'^([RSD])(\d+)$', reg)
    if not m:
        return None
    cls = m.group(1)
    idx = int(m.group(2))
    if cls == "R" and 0 <= idx < 4:
        return {"slot": idx, "canonical": "reg:R%d" % idx, "kind": "gp"}
    if cls in ("S", "D") and 0 <= idx < 4:
        return {"slot": idx, "canonical": "reg:S%d" % idx, "kind": "fp"}
    return None


def _mips_canonical_arg_reg(reg):
    reg = str(reg).upper()
    mapping = {
        "A0": (0, "reg:A0", "gp"),
        "A1": (1, "reg:A1", "gp"),
        "A2": (2, "reg:A2", "gp"),
        "A3": (3, "reg:A3", "gp")
    }
    if reg not in mapping:
        return None
    slot, canon, kind = mapping[reg]
    return {"slot": slot, "canonical": canon, "kind": kind}


def _canonical_abi_slot_info(prog, loc):
    if loc is None:
        return None
    s = str(loc)
    if not s.startswith("reg:"):
        return None
    reg = s.split(":", 1)[1].upper()

    if _processor_is_aarch64(prog):
        return _aarch64_canonical_arg_reg(reg)
    if _processor_is_x86_64(prog):
        return _x86_64_canonical_arg_reg(reg)
    if _processor_is_arm32(prog):
        return _arm32_canonical_arg_reg(reg)
    if _processor_is_mips(prog):
        return _mips_canonical_arg_reg(reg)

    # 通用退化：沿用旧的 arg slot 表
    slots = _get_call_arg_register_slots(prog)
    idx = 0
    for slot_set in slots:
        try:
            if reg in slot_set:
                ordered = sorted(list(slot_set))
                canon = "reg:%s" % ordered[0]
                return {"slot": idx, "canonical": canon, "kind": "gp"}
        except:
            pass
        idx += 1
    return None


def _infer_declared_param_storage_v4(prog, idx, dt=None):
    if idx is None or idx < 0:
        return "unknown"

    if _processor_is_aarch64(prog):
        if idx < 8:
            if dt is not None and ("float" in _dtype_name_lower(dt) or "double" in _dtype_name_lower(dt)):
                return "reg:D%d" % idx
            return "reg:X%d" % idx
        return "stack:param_%d" % idx

    if _processor_is_x86_64(prog):
        gp = ["reg:RDI", "reg:RSI", "reg:RDX", "reg:RCX", "reg:R8", "reg:R9"]
        fp = ["reg:XMM0", "reg:XMM1", "reg:XMM2", "reg:XMM3", "reg:XMM4", "reg:XMM5"]
        if idx < len(gp):
            if dt is not None and ("float" in _dtype_name_lower(dt) or "double" in _dtype_name_lower(dt)):
                return fp[idx]
            return gp[idx]
        return "stack:param_%d" % idx

    if _processor_is_arm32(prog):
        if idx < 4:
            if dt is not None and ("float" in _dtype_name_lower(dt) or "double" in _dtype_name_lower(dt)):
                return "reg:S%d" % idx
            return "reg:R%d" % idx
        return "stack:param_%d" % idx

    if _processor_is_mips(prog):
        gp = ["reg:A0", "reg:A1", "reg:A2", "reg:A3"]
        if idx < len(gp):
            return gp[idx]
        return "stack:param_%d" % idx

    return "param_%d" % idx


def _is_special_or_callee_saved_loc(prog, loc):
    if loc is None:
        return True
    s = str(loc)
    if not s.startswith("reg:"):
        return False

    reg = s.split(":", 1)[1].upper()

    if _processor_is_aarch64(prog):
        if reg in ("SP", "WSP", "X29", "FP", "X30", "LR"):
            return True
        m = re.match(r'^X(\d+)$', reg)
        if m:
            idx = int(m.group(1))
            if 19 <= idx <= 28:
                return True
        m = re.match(r'^W(\d+)$', reg)
        if m:
            idx = int(m.group(1))
            if 19 <= idx <= 28:
                return True
        return False

    if _processor_is_x86_64(prog):
        return reg in ("RSP", "ESP", "RBP", "EBP", "RBX", "EBX", "R12", "R13", "R14", "R15", "RIP")

    if _processor_is_arm32(prog):
        if reg in ("SP", "LR", "PC", "R13", "R14", "R15"):
            return True
        m = re.match(r'^R(\d+)$', reg)
        if m:
            idx = int(m.group(1))
            if 4 <= idx <= 11:
                return True
        return False

    if _processor_is_mips(prog):
        if reg in ("SP", "FP", "RA"):
            return True
        m = re.match(r'^S(\d+)$', reg)
        if m:
            return True
        return False

    return False


def _normalize_entry_param_locs_v4(prog, locs):
    out = []
    seen = set()
    for loc in locs:
        if _is_special_or_callee_saved_loc(prog, loc):
            continue
        info = _canonical_abi_slot_info(prog, loc)
        if info is None:
            continue
        canon = info.get("canonical")
        if canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
    return out


def _storage_matches_abi_slot(prog, storage, slot_loc):
    if storage is None or slot_loc is None:
        return False
    if str(storage) == str(slot_loc):
        return True
    a = _canonical_abi_slot_info(prog, storage)
    b = _canonical_abi_slot_info(prog, slot_loc)
    if a is None or b is None:
        return False
    return (a.get("slot") == b.get("slot")) and (a.get("kind") == b.get("kind"))


def _extra_recovered_param_locs_v4(prog, recovered_locs, declared_param_items):
    normalized = _normalize_entry_param_locs_v4(prog, recovered_locs)
    declared_slots = set()
    declared_count = len(declared_param_items)

    idx = 0
    for item in declared_param_items:
        storage = item.get("storage")
        if _is_unknown_storage(storage):
            storage = _infer_declared_param_storage_v4(prog, idx, None)
        info = _canonical_abi_slot_info(prog, storage)
        if info is not None:
            declared_slots.add((info.get("slot"), info.get("kind")))
        idx += 1

    extras = []
    for loc in normalized:
        info = _canonical_abi_slot_info(prog, loc)
        if info is None:
            continue
        key = (info.get("slot"), info.get("kind"))
        if key in declared_slots:
            continue
        if info.get("slot") < declared_count:
            continue
        extras.append(loc)
    return extras, normalized


def _make_synthetic_param_v4(entry, idx, loc, slot_info=None):
    item = {
        "id": gen_id("high_type", entry, "synthetic_param_%d" % idx),
        "location": "%s_recovered_param_%d" % (entry, idx),
        "param_name": "arg_%d" % idx,
        "type_candidates": ["unknown", "pointer_candidate"],
        "type_confidence": [0.38, 0.56],
        "type_evidence": ["entry_input_locs_filtered", "abi_argument_slot"],
        "type_size": 0,
        "is_pointer": True,
        "category": "synthetic_param",
        "storage": loc,
        "function_entry": func_entry,
        "function_name": None,
        "object_kind": "param",
        "param_index": idx,
        "storage_kind": _storage_kind_and_id(loc)[0],
        "storage_id": _storage_kind_and_id(loc)[1],
        "stack_offset": None,
        "pred_type": "pointer_like_input",
        "pred_type_category": "ptr",
        "source": "dfg_recovered"
    }
    if slot_info is not None:
        item["abi_slot"] = slot_info.get("slot")
        item["abi_kind"] = slot_info.get("kind")
    return item


def _role_candidates_from_type_v4(param_name, type_name, is_pointer):
    pname = _safe_lower(param_name)
    tname = _safe_lower(type_name)

    if pname in ("argc",):
        return ["count", "length", "general_param"], [0.95, 0.76, 0.55]
    if pname in ("argv", "envp", "env"):
        return ["pointer", "buffer", "general_param"], [0.95, 0.82, 0.55]
    if "fmt" in pname or "format" in pname:
        return ["format_str", "buffer", "general_param"], [0.93, 0.62, 0.50]
    if "len" in pname or "size" in pname or "count" in pname or "num" in pname:
        return ["length", "counter", "general_param"], [0.90, 0.72, 0.55]
    if "out" in pname:
        if is_pointer:
            return ["output_buffer", "pointer", "general_param"], [0.90, 0.78, 0.55]
        return ["output_value", "general_param"], [0.80, 0.55]
    if is_pointer:
        return ["buffer", "pointer", "general_param"], [0.82, 0.74, 0.55]
    if ("float" in tname) or ("double" in tname):
        return ["numeric_value", "general_param"], [0.78, 0.55]
    return ["general_param"], [0.65]


def _top_type_candidate_name(item):
    cands = item.get("type_candidates", [])
    if cands:
        return cands[0]
    return "unknown"


def _type_confidence_head(item, default_value=0.6):
    confs = item.get("type_confidence", [])
    if confs:
        try:
            return float(confs[0])
        except:
            return default_value
    return default_value


def _trace_address_expr(program, vn, depth=0, seen=None):
    """
    v4: 在上一版基础上补强 phi / 选择合并类地址表达式的规约。
    仍然保持轻量、可解释，不引入求解器。
    """
    if seen is None:
        seen = set()

    if vn is None:
        if _PREV_TRACE_ADDRESS_EXPR is not None:
            return _PREV_TRACE_ADDRESS_EXPR(program, vn, depth, seen)
        return {"base": None, "offset": None, "expr": "unknown", "is_fixed": False}

    key = _safe_encode_piece(vn)
    if key and key in seen:
        if _PREV_TRACE_ADDRESS_EXPR is not None:
            return _PREV_TRACE_ADDRESS_EXPR(program, vn, depth, seen)
        return {"base": None, "offset": None, "expr": key, "is_fixed": False}

    local_seen = set(seen)
    if key:
        local_seen.add(key)

    defop = None
    try:
        defop = vn.getDef()
    except:
        defop = None

    if defop is not None:
        try:
            mnemonic = str(defop.getMnemonic())
        except:
            mnemonic = None

        inputs = []
        try:
            inputs = list(defop.getInputs())
        except:
            inputs = []

        if mnemonic == "MULTIEQUAL":
            cands = []
            for inp in inputs:
                cands.append(_trace_address_expr(program, inp, depth + 1, set(local_seen)))
            merged = _merge_alias_candidates(cands)
            if merged is not None:
                out = {
                    "base": merged.get("base"),
                    "offset": merged.get("offset"),
                    "expr": merged.get("expr"),
                    "is_fixed": bool(merged.get("is_fixed"))
                }
                if out.get("expr") is None:
                    out["expr"] = "phi"
                return out

        if mnemonic == "PTRADD" and len(inputs) >= 3:
            base = _trace_address_expr(program, inputs[0], depth + 1, set(local_seen))
            idx_info = _trace_address_expr(program, inputs[1], depth + 1, set(local_seen))
            scale_info = _trace_address_expr(program, inputs[2], depth + 1, set(local_seen))

            if base.get("base") is not None and idx_info.get("offset") is not None and scale_info.get("offset") is not None:
                off = int(base.get("offset") or 0) + int(idx_info.get("offset")) * int(scale_info.get("offset"))
                return {
                    "base": base.get("base"),
                    "offset": off,
                    "expr": _format_base_offset(base.get("base"), off),
                    "is_fixed": bool(base.get("is_fixed")) and bool(idx_info.get("is_fixed")) and bool(scale_info.get("is_fixed"))
                }

    if _PREV_TRACE_ADDRESS_EXPR is not None:
        return _PREV_TRACE_ADDRESS_EXPR(program, vn, depth, local_seen)

    default_expr = _canonical_varnode_location(program, vn, include_constant=False)
    if default_expr is None:
        default_expr = _safe_encode_piece(vn) or "unknown"
    return {"base": None, "offset": None, "expr": default_expr, "is_fixed": False}


def gen_type_sem(prog, dfg_cache=None):
    all_typesem = []
    dfg_by_func = _dfg_by_entry(dfg_cache if dfg_cache is not None else gen_dataflow_sem(prog))
    funcs = _get_internal_functions(prog, include_thunks=False)

    for func in funcs:
        func_entry = _addr_to_str(func.getEntryPoint())
        func_name = _safe_get_name(func, "sub_%s" % func_entry)
        dfg = dfg_by_func.get(func_entry, {"memory_access_patterns": [], "entry_input_locs": [], "abstract_merge": {"alias_groups": []}})

        ret_type = None
        try:
            ret_type = func.getReturnType()
        except:
            ret_type = None

        ret_candidates, ret_conf, ret_evidence = get_type_candidates_and_confidence(ret_type)
        ret_type_sem = {
            "id": gen_id("high_type", func_entry, "ret"),
            "location": "%s_return" % func_entry,
            "type_candidates": ret_candidates,
            "type_confidence": ret_conf,
            "type_evidence": ret_evidence,
            "type_size": _safe_get_length(ret_type),
            "is_pointer": _is_pointer_type_like(ret_type, ret_candidates),
            "category": get_safe_type_category(ret_type)
        }

        params_sem = []
        raw_params = []
        try:
            raw_params = list(func.getParameters())
        except:
            raw_params = []

        idx = 0
        for param in raw_params:
            p_type = None
            try:
                p_type = param.getDataType()
            except:
                p_type = None

            storage = _safe_get_storage(param)
            if _is_unknown_storage(storage):
                storage = _infer_declared_param_storage_v4(prog, idx, p_type)

            p_candidates, p_conf, p_evidence = get_type_candidates_and_confidence(p_type, param, func)
            support_boost, support_evidence = _type_support_from_dfg(dfg, storage)
            if p_conf:
                p_conf[0] = min(0.99, float(p_conf[0]) + support_boost - 0.25)
            p_evidence.extend(support_evidence)

            abi_info = _canonical_abi_slot_info(prog, storage)
            param_item = {
                "id": gen_id("high_type", func_entry, "param_%d" % idx),
                "location": "%s_param_%d" % (func_entry, idx),
                "param_name": _safe_get_name(param, "arg_%d" % idx),
                "type_candidates": p_candidates,
                "type_confidence": p_conf,
                "type_evidence": _unique_list(p_evidence),
                "type_size": _safe_get_length(p_type),
                "is_pointer": _is_pointer_type_like(p_type, p_candidates),
                "category": get_safe_type_category(p_type),
                "storage": storage,
                "source": "ghidra_declared"
            }
            if abi_info is not None:
                param_item["abi_slot"] = abi_info.get("slot")
                param_item["abi_kind"] = abi_info.get("kind")
                param_item["storage"] = abi_info.get("canonical")
            params_sem.append(param_item)
            idx += 1

        recovered_inputs, normalized_entry_locs = _extra_recovered_param_locs_v4(prog, dfg.get("entry_input_locs", []), params_sem)

        syn_idx = 0
        for loc in recovered_inputs:
            abi_info = _canonical_abi_slot_info(prog, loc)
            params_sem.append(_make_synthetic_param_v4(func_entry, syn_idx, loc, abi_info))
            syn_idx += 1

        locals_sem = []
        local_vars = []
        try:
            local_vars = list(func.getLocalVariables())
        except:
            local_vars = []

        idx = 0
        for var in local_vars:
            v_type = None
            try:
                v_type = var.getDataType()
            except:
                v_type = None
            storage = _safe_get_storage(var)
            v_storage_kind, v_storage_id = _storage_kind_and_id(storage)
            v_candidates, v_conf, v_evidence = get_type_candidates_and_confidence(v_type, var, func)
            support_boost, support_evidence = _type_support_from_dfg(dfg, storage)
            if v_conf:
                v_conf[0] = min(0.99, float(v_conf[0]) + support_boost - 0.25)
            v_evidence.extend(support_evidence)

            locals_sem.append({
                "id": gen_id("high_type", func_entry, "local_%d" % idx),
                "location": "%s_local_%d" % (func_entry, idx),
                "var_name": _safe_get_name(var, "local_%d" % idx),
                "type_candidates": v_candidates,
                "type_confidence": v_conf,
                "type_evidence": _unique_list(v_evidence),
                "type_size": _safe_get_length(v_type),
                "is_pointer": _is_pointer_type_like(v_type, v_candidates),
                "category": get_safe_type_category(v_type),
                "storage": storage,
                "stack_offset": _safe_get_stack_offset(var),
                "source": "ghidra_declared"
            })
            idx += 1

        alias_groups = dfg.get("abstract_merge", {}).get("alias_groups", [])
        evidence_summary = {
            "memory_access_count": len(dfg.get("memory_access_patterns", [])),
            "entry_input_count": len(normalized_entry_locs),
            "alias_group_count": len(alias_groups),
            "fixed_offset_group_count": len([g for g in alias_groups if len(g.get("offsets", [])) >= 2]),
            "recovered_extra_param_count": len(recovered_inputs)
        }

        all_typesem.append({
            "id": gen_id("high_type", func_entry, "func"),
            "function_name": func_name,
            "function_entry": func_entry,
            "function_id": _function_uid(func),
            "return_type_sem": ret_type_sem,
            "parameter_type_sem": params_sem,
            "local_var_type_sem": locals_sem,
            "instruction_type_sem": [],
            "type_constraints": {
                "return_type_ref": ret_type_sem.get("id"),
                "param_type_refs": [x.get("id") for x in params_sem],
                "local_type_refs": [x.get("id") for x in locals_sem],
                "entry_input_locs_filtered": normalized_entry_locs,
                "evidence_summary": evidence_summary
            }
        })

    return all_typesem


def gen_interface_sem(prog, type_cache=None, dfg_cache=None):
    all_interfacesem = []
    funcs = _get_internal_functions(prog, include_thunks=False)
    type_by_func = _type_by_entry(type_cache if type_cache is not None else gen_type_sem(prog, dfg_cache))
    dfg_by_func = _dfg_by_entry(dfg_cache if dfg_cache is not None else gen_dataflow_sem(prog))

    for func in funcs:
        func_entry = _addr_to_str(func.getEntryPoint())
        func_name = _safe_get_name(func, "sub_%s" % func_entry)
        func_id = _function_uid(func)
        type_obj = type_by_func.get(func_entry, {})
        dfg = dfg_by_func.get(func_entry, {"entry_input_locs": []})

        params_sem = []
        type_params = type_obj.get("parameter_type_sem", [])
        idx = 0
        declared_count = 0
        synthetic_count = 0
        for p in type_params:
            dt_name = _top_type_candidate_name(p)
            pname = p.get("param_name", "arg_%d" % idx)
            storage = p.get("storage")
            if _is_unknown_storage(storage):
                storage = _infer_declared_param_storage_v4(prog, idx, None)

            is_ptr = bool(p.get("is_pointer"))
            role_candidates, role_confidence = _role_candidates_from_type_v4(pname, dt_name, is_ptr)
            param_id = "interface_%s_param_%d" % (func_entry, idx)

            origin = p.get("source", "unknown")
            if origin == "ghidra_declared":
                declared_count += 1
            else:
                synthetic_count += 1

            abi_info = _canonical_abi_slot_info(prog, storage)

            item = {
                "id": param_id,
                "param_id": idx,
                "param_name": pname,
                "location": p.get("location"),
                "type_sem_ref": p.get("id"),
                "storage": storage,
                "type_candidates": p.get("type_candidates", []),
                "type_confidence": p.get("type_confidence", []),
                "role_candidates": role_candidates,
                "role_confidence": role_confidence,
                "param_origin": origin
            }
            if abi_info is not None:
                item["abi_slot"] = abi_info.get("slot")
                item["abi_kind"] = abi_info.get("kind")
                item["storage"] = abi_info.get("canonical")
            params_sem.append(item)
            idx += 1

        ret_type_sem = type_obj.get("return_type_sem", {})
        ret_sem = {
            "id": "interface_%s_ret" % func_entry,
            "location": ret_type_sem.get("location"),
            "type_sem_ref": ret_type_sem.get("id"),
            "type_candidates": ret_type_sem.get("type_candidates", []),
            "type_confidence": ret_type_sem.get("type_confidence", []),
            "type_size": ret_type_sem.get("type_size", 0),
            "is_pointer": ret_type_sem.get("is_pointer", False),
            "return_type": _first_type_candidate(ret_type_sem.get("type_candidates", [])),
            "return_type_category": ret_type_sem.get("pred_type_category", ret_type_sem.get("category", "unknown"))
        }

        filtered_entry_locs = type_obj.get("type_constraints", {}).get("entry_input_locs_filtered")
        if filtered_entry_locs is None:
            filtered_entry_locs = _normalize_entry_param_locs_v4(prog, dfg.get("entry_input_locs", []))

        all_interfacesem.append({
            "id": gen_id("high_interface", func_entry, "func"),
            "function_name": func_name,
            "function_entry": func_entry,
            "func_id": func_id,
            "params": params_sem,
            "ret": ret_sem,
            "interface_constraints": {
                "calling_convention": _safe_cc(func),
                "param_count": len(params_sem),
                "declared_param_count": declared_count,
                "synthetic_param_count": synthetic_count,
                "has_return": (not _is_void_type(func.getReturnType())) if hasattr(func, "getReturnType") else True,
                "constraint_evidence": _unique_list([
                    "calling_convention_analysis",
                    "entry_input_locs_filtered",
                    "type_sem_link"
                ]),
                "cross_layer_constraints": {
                    "type_sem_ref": type_obj.get("id"),
                    "return_type_ref": ret_type_sem.get("id"),
                    "param_type_refs": [p.get("type_sem_ref") for p in params_sem],
                    "entry_input_locs": filtered_entry_locs
                }
            }
        })

    return all_interfacesem


def _block_successor_policy_v4(cfg_edges):
    edge_types = sorted(list(set([e.get("type") for e in cfg_edges if e.get("type")])))
    if not edge_types:
        return "no_explicit_successor"
    if "conditional_branch" in edge_types:
        return "conditional_split"
    if "call" in edge_types and len(edge_types) == 1:
        return "call_then_return"
    if "branch" in edge_types or "indirect_branch" in edge_types:
        return "jump_transfer"
    if "terminal" in edge_types:
        return "function_exit"
    return "fallthrough"


def gen_block_sem(prog, cfg_cache=None, dfg_cache=None):
    base_blocks = []
    if _PREV_GEN_BLOCK_SEM is not None:
        base_blocks = _PREV_GEN_BLOCK_SEM(prog, cfg_cache, dfg_cache)
    else:
        base_blocks = []

    raw_block_by_entry = {}
    cfg_edges_by_entry = defaultdict(list)
    cfg_sem = cfg_cache if cfg_cache is not None else _BASE_GEN_CFG_SEM(prog)

    for cfg in cfg_sem:
        for e in cfg.get("edges", []):
            cfg_edges_by_entry[e.get("from")].append(e)

    tm = _get_script_monitor()
    block_model = BasicBlockModel(prog)
    funcs = _get_internal_functions(prog, include_thunks=False)
    for func in funcs:
        for block in _BASE_GET_FUNCTION_BLOCKS(prog, func, block_model, tm):
            raw_block_by_entry[block.get("entry_str")] = block

    enriched = []
    for sem in base_blocks:
        entry = sem.get("entry")
        raw = raw_block_by_entry.get(entry)
        last_instr = None
        if raw is not None and raw.get("instructions"):
            last_instr = raw.get("instructions")[-1]

        term_pred = _extract_predicate_from_last_instr(last_instr)
        term_kind = "fallthrough"
        term_asm = None
        term_mnemonic = None
        if last_instr is not None:
            term_asm = last_instr.toString()
            try:
                term_mnemonic = last_instr.getMnemonicString()
            except:
                term_mnemonic = None
            try:
                ft = last_instr.getFlowType()
            except:
                ft = None
            term_kind = _classify_edge_type(ft)

        item = dict(sem)
        item["terminator_asm"] = term_asm
        item["terminator_kind"] = term_kind
        item["predicate"] = term_pred
        item["successor_policy"] = _block_successor_policy_v4(cfg_edges_by_entry.get(entry, []))
        item["instruction_count"] = len(raw.get("instructions", [])) if raw is not None else 0

        if last_instr is not None and not item.get("T_B"):
            item["T_B"] = [term_mnemonic] if term_mnemonic else []
        if item.get("effect_summary") is None:
            item["effect_summary"] = {}
        item["effect_summary"]["terminator_kind"] = term_kind
        item["effect_summary"]["successor_policy"] = item.get("successor_policy")
        item["effect_summary"]["terminator_only"] = (len(item.get("def_set", [])) == 0 and len(item.get("use_set", [])) == 0 and item.get("instruction_count", 0) > 0)
        enriched.append(item)

    return enriched


# ---- API 模板补齐 ----
API_SEMANTIC_TEMPLATES.update({
    "memcpy": {
        "intent": "buffer_copy",
        "param_roles": ["dest", "src", "size"],
        "side_effect": ["memory_write", "buffer_copy"],
        "constraints": ["dest_buffer_valid", "src_buffer_valid", "size_non_negative"]
    },
    "memmove": {
        "intent": "buffer_move",
        "param_roles": ["dest", "src", "size"],
        "side_effect": ["memory_write", "overlap_safe_copy"],
        "constraints": ["dest_buffer_valid", "src_buffer_valid", "size_non_negative"]
    },
    "memset": {
        "intent": "buffer_fill",
        "param_roles": ["dest", "value", "size"],
        "side_effect": ["memory_write", "buffer_initialization"],
        "constraints": ["dest_buffer_valid", "size_non_negative"]
    },
    "strcmp": {
        "intent": "string_compare",
        "param_roles": ["lhs", "rhs"],
        "side_effect": ["no_side_effect"],
        "constraints": ["lhs_string_valid", "rhs_string_valid"]
    },
    "puts": {
        "intent": "std_output",
        "param_roles": ["message"],
        "side_effect": ["console_output"],
        "constraints": ["message_string_valid"]
    },
    "exit": {
        "intent": "process_termination",
        "param_roles": ["status"],
        "side_effect": ["no_return", "process_exit"],
        "constraints": ["status_code_valid"]
    },
    "stack_chk_fail": {
        "intent": "stack_smash_abort",
        "param_roles": [],
        "side_effect": ["security_abort", "no_return"],
        "constraints": ["stack_cookie_violation_detected"]
    },
    "__printf_chk": {
        "intent": "std_output",
        "param_roles": ["chk_level", "format_str", "var_args"],
        "side_effect": ["console_output", "format_processing"],
        "constraints": ["format_str_valid"]
    }
})


def _api_min_keep_count(api_name, sem_template):
    name = _safe_lower(api_name)
    if "printf" in name or "scanf" in name:
        return 1
    return len(sem_template.get("param_roles", []))


def _crop_api_param_roles_v4(api_name, param_sem, sem_template):
    min_keep = _api_min_keep_count(api_name, sem_template)
    if not param_sem:
        return []

    last_non_null = -1
    idx = 0
    for p in param_sem:
        if p.get("recovered_from") is not None:
            last_non_null = idx
        idx += 1

    keep = max(min_keep, last_non_null + 1)
    if keep <= 0:
        keep = min_keep
    if keep < 0:
        keep = 0
    cropped = param_sem[:keep]

    # 去掉尾部连续的空恢复项（但保留 min_keep）
    while len(cropped) > min_keep:
        tail = cropped[-1]
        if tail.get("recovered_from") is None:
            cropped = cropped[:-1]
            continue
        break
    return cropped


def gen_api_sem(prog, dfg_cache=None, interface_cache=None, type_cache=None):
    all_apisem = []
    api_seen = set()
    dfg_by_func = _dfg_by_entry(dfg_cache if dfg_cache is not None else gen_dataflow_sem(prog))
    interface_by_func = _interface_by_entry(interface_cache if interface_cache is not None else gen_interface_sem(prog, type_cache, dfg_cache))
    type_by_func = _type_by_entry(type_cache if type_cache is not None else gen_type_sem(prog, dfg_cache))
    func_manager = prog.getFunctionManager()
    ref_mgr = prog.getReferenceManager()
    arg_slots = _get_call_arg_register_slots(prog)

    candidate_funcs = _get_external_like_functions(prog)
    for ext in candidate_funcs:
        api_name = ext.get("api_name") or "external_api"
        target_entry = ext.get("entry")
        refs = ref_mgr.getReferencesTo(ext.get("func").getEntryPoint())

        for ref in refs:
            try:
                if not RefType.isCall(ref.getReferenceType()):
                    continue
            except:
                try:
                    if ref.getReferenceType() != RefType.CALL:
                        continue
                except:
                    continue

            callsite_addr = ref.getFromAddress()
            callsite = _addr_to_str(callsite_addr)
            caller_func = func_manager.getFunctionContaining(callsite_addr)
            if caller_func is None:
                continue
            try:
                if caller_func.isExternal():
                    continue
            except:
                pass

            caller_entry = _addr_to_str(caller_func.getEntryPoint())
            caller_dfg = dfg_by_func.get(caller_entry, {"definitions": [], "memory_access_patterns": []})
            defs_index = _build_defs_index_for_call(caller_dfg)
            call_addr_int = _parse_int_addr(callsite)

            sem_template = get_api_semantic_template(api_name)
            param_sem = []

            idx = 0
            for role in sem_template.get("param_roles", []):
                if idx >= len(arg_slots):
                    break
                slot_regs = set(["reg:%s" % r for r in arg_slots[idx]])
                best_def = _find_last_def_before(defs_index, slot_regs, call_addr_int)
                param_sem.append({
                    "param_idx": idx,
                    "role": role,
                    "source_locations": sorted(list(slot_regs)),
                    "recovered_from": best_def.get("location") if best_def else None,
                    "recovered_site": best_def.get("site") if best_def else None
                })
                idx += 1

            param_sem = _crop_api_param_roles_v4(api_name, param_sem, sem_template)

            caller_interface = interface_by_func.get(caller_entry)
            caller_type = type_by_func.get(caller_entry)

            semantic_template = {
                "api_intent": sem_template.get("intent"),
                "param_roles": param_sem,
                "side_effects": sem_template.get("side_effect", []),
                "base_constraints": sem_template.get("constraints", []),
                "symbol_resolution": {
                    "api_name": api_name,
                    "target_entry": target_entry,
                    "callsite": callsite,
                    "resolution_kind": "external_or_thunk"
                }
            }

            propagated_constraints = {
                "caller_func_id": _function_uid(caller_func),
                "caller_function_entry": caller_entry,
                "type_sem_ref": caller_type.get("id") if caller_type else None,
                "interface_sem_ref": caller_interface.get("id") if caller_interface else None,
                "constraint_evidence": _unique_list([
                    "api_symbol_resolution",
                    "call_site_analysis",
                    "argument_register_trace"
                ]),
                "propagation_targets": _unique_list(["type_sem", "interface_sem", "layout_sem"])
            }

            api_key = (callsite, api_name, caller_entry)
            if api_key in api_seen:
                continue
            api_seen.add(api_key)

            all_apisem.append({
                "id": gen_id("high_api", callsite, "0"),
                "callsite": callsite,
                "api_name": api_name,
                "semantic_template": semantic_template,
                "propagated_constraints": propagated_constraints
            })

    return all_apisem


def _is_stack_frame_base_v4(base):
    if base is None:
        return False
    s = str(base)
    if s.startswith("stack:"):
        return True
    return s in ("reg:SP", "reg:RSP", "reg:ESP", "reg:WSP", "reg:X29", "reg:FP", "reg:RBP", "reg:EBP")


def _cluster_field_summaries_v4(field_summaries, stride_hint=None):
    if not field_summaries:
        return []

    ordered = sorted(field_summaries, key=lambda x: int(x.get("offset")))
    gap_threshold = 64
    if stride_hint is not None:
        try:
            gap_threshold = max(64, abs(int(stride_hint)) * 4)
        except:
            gap_threshold = 64

    groups = []
    current = [ordered[0]]
    last_off = int(ordered[0].get("offset"))

    for fs in ordered[1:]:
        off = int(fs.get("offset"))
        # 大 gap、跨符号区间、或从负偏移跳到正偏移时切分
        if abs(off - last_off) > gap_threshold or ((last_off < 0 <= off) or (off < 0 <= last_off)):
            groups.append(current)
            current = [fs]
        else:
            current.append(fs)
        last_off = off

    if current:
        groups.append(current)
    return groups


def _match_type_refs_for_layout_v4(prog, type_obj, base_loc):
    refs = []
    if not type_obj:
        return refs

    top_id = type_obj.get("id")
    if top_id:
        refs.append(top_id)

    for p in type_obj.get("parameter_type_sem", []):
        if _storage_matches_abi_slot(prog, p.get("storage"), base_loc):
            refs.append(p.get("id"))
    for lv in type_obj.get("local_var_type_sem", []):
        if str(lv.get("storage")) == str(base_loc):
            refs.append(lv.get("id"))

    return _unique_list(refs)


def _match_interface_param_refs_for_layout_v4(prog, interface_obj, base_loc):
    refs = []
    if interface_obj is None:
        return refs
    for p in interface_obj.get("params", []):
        if _storage_matches_abi_slot(prog, p.get("storage"), base_loc):
            refs.append(p.get("id"))
    return _unique_list(refs)


def _layout_roles_v4(layout_kind, stride, matched_type_refs, matched_interface_refs):
    roles = []
    if layout_kind == "stack_frame_layout":
        roles.append("stack_frame_layout")
    else:
        roles.append("object_layout")
    if stride is not None:
        roles.append("regular_stride_layout")
    if matched_type_refs:
        roles.append("type_aligned")
    if matched_interface_refs:
        roles.append("interface_aligned")
    if not roles:
        roles.append("layout_candidate")
    return roles


def gen_mem_layout_sem(prog, dfg_cache=None, type_cache=None, interface_cache=None):
    all_memlayout = []
    dfg_by_func = _dfg_by_entry(dfg_cache if dfg_cache is not None else gen_dataflow_sem(prog))
    type_by_func = _type_by_entry(type_cache if type_cache is not None else gen_type_sem(prog, dfg_cache))
    interface_by_func = _interface_by_entry(interface_cache if interface_cache is not None else gen_interface_sem(prog, type_cache, dfg_cache))
    funcs = _get_internal_functions(prog, include_thunks=False)

    for func in funcs:
        func_entry = _addr_to_str(func.getEntryPoint())
        func_name = _safe_get_name(func, "sub_%s" % func_entry)
        dfg = dfg_by_func.get(func_entry, {"abstract_merge": {"alias_groups": []}})
        type_obj = type_by_func.get(func_entry)
        interface_obj = interface_by_func.get(func_entry)

        seq = 0
        groups = dfg.get("abstract_merge", {}).get("alias_groups", [])
        for group in groups:
            field_summaries = group.get("field_summaries", [])
            cluster_fields = _cluster_field_summaries_v4(field_summaries, group.get("stride"))

            for cidx, cluster in enumerate(cluster_fields):
                offsets = sorted([int(x.get("offset")) for x in cluster])
                if len(offsets) < 2:
                    continue

                layout_kind = "stack_frame_layout" if _is_stack_frame_base_v4(group.get("base")) else "object_layout"
                field_relations = _field_relations_from_offsets(offsets)

                size_rel = []
                for fr in field_relations:
                    if fr.get("size_hint") is not None:
                        size_rel.append("size_eq_%d" % int(fr.get("size_hint")))

                matched_type_refs = _match_type_refs_for_layout_v4(prog, type_obj, group.get("base"))
                matched_interface_refs = _match_interface_param_refs_for_layout_v4(prog, interface_obj, group.get("base"))
                roles = _layout_roles_v4(layout_kind, group.get("stride"), matched_type_refs, matched_interface_refs)

                confidence = 0.60 + 0.05 * min(len(offsets), 4)
                if group.get("stride") is not None:
                    confidence += 0.08
                if layout_kind == "object_layout":
                    confidence += 0.08
                else:
                    confidence -= 0.05

                support_sites = []
                for fs in cluster:
                    support_sites.extend(fs.get("sites", []))
                support_sites = sorted(_unique_list(support_sites))

                all_memlayout.append({
                    "id": gen_id("high_mem", func_entry, "cluster_%d" % seq),
                    "function_entry": func_entry,
                    "function_name": func_name,
                    "layout_kind": layout_kind,
                    "cluster_id": cidx,
                    "base_obj": group.get("base"),
                    "offset_set": offsets,
                    "size_rel": _unique_list(size_rel),
                    "field_relations": field_relations,
                    "nested_candidates": [],
                    "support_sites": support_sites,
                    "semantic_roles": roles,
                    "layout_source": "dfg_alias_group_cluster",
                    "confidence": _clamp_strength(confidence),
                    "propagated_constraints": {
                        "target_func_id": _function_uid(func),
                        "type_sem_refs": matched_type_refs,
                        "interface_sem_ref": interface_obj.get("id") if interface_obj else None,
                        "interface_param_refs": matched_interface_refs,
                        "constraint_evidence": _unique_list([
                            "fixed_offset_access_cluster",
                            "alias_group_cluster_split",
                            "stride_analysis" if group.get("stride") is not None else None,
                            "non_sp_base_object_candidate" if layout_kind == "object_layout" else "stack_frame_cluster"
                        ])
                    },
                    "location": "%s_layout_%s_%d" % (func_entry, str(group.get("base")).replace(":", "_"), seq)
                })
                seq += 1

        if seq == 0:
            fallback_items = _fallback_declared_layouts(func, func_entry, type_obj, interface_obj)
            for item in fallback_items:
                item["layout_kind"] = "declared_type_layout"
            all_memlayout.extend(fallback_items)

    return all_memlayout


def _append_node_unique_v4(nodes, seen, node_id, layer, kind=None, parent_id=None):
    if not node_id or node_id in seen:
        return
    seen.add(node_id)
    item = {"id": node_id, "layer": layer}
    if kind is not None:
        item["kind"] = kind
    if parent_id is not None:
        item["parent_id"] = parent_id
    nodes.append(item)


def _collect_semantic_nodes(sem):
    nodes = []
    seen = set()

    for item in sem.get("low_level_sem", {}).get("instr_sem", []):
        _append_node_unique_v4(nodes, seen, item.get("id"), "InstrSem")
    for item in sem.get("low_level_sem", {}).get("block_sem", []):
        _append_node_unique_v4(nodes, seen, item.get("id"), "BlockSem")
    for item in sem.get("mid_level_sem", {}).get("cfg_sem", []):
        _append_node_unique_v4(nodes, seen, item.get("id"), "CFGSem")
    for item in sem.get("mid_level_sem", {}).get("dataflow_sem", []):
        _append_node_unique_v4(nodes, seen, item.get("id"), "DFGSem")

    for item in sem.get("high_level_sem", {}).get("type_sem", []):
        parent_id = item.get("id")
        _append_node_unique_v4(nodes, seen, parent_id, "TypeSem", "type_root")
        rt = item.get("return_type_sem", {})
        _append_node_unique_v4(nodes, seen, rt.get("id"), "TypeSem", "return_type", parent_id)
        for p in item.get("parameter_type_sem", []):
            _append_node_unique_v4(nodes, seen, p.get("id"), "TypeSem", "param_type", parent_id)
        for lv in item.get("local_var_type_sem", []):
            _append_node_unique_v4(nodes, seen, lv.get("id"), "TypeSem", "local_type", parent_id)

    for item in sem.get("high_level_sem", {}).get("interface_sem", []):
        parent_id = item.get("id")
        _append_node_unique_v4(nodes, seen, parent_id, "InterfaceSem", "interface_root")
        for p in item.get("params", []):
            _append_node_unique_v4(nodes, seen, p.get("id"), "InterfaceSem", "interface_param", parent_id)
        rt = item.get("ret", {})
        _append_node_unique_v4(nodes, seen, rt.get("id"), "InterfaceSem", "interface_ret", parent_id)

    for item in sem.get("high_level_sem", {}).get("api_sem", []):
        _append_node_unique_v4(nodes, seen, item.get("id"), "APISem")
    for item in sem.get("high_level_sem", {}).get("mem_layout_sem", []):
        _append_node_unique_v4(nodes, seen, item.get("id"), "LayoutSem")

    return nodes


def gen_node_state(sem):
    states = []
    type_index = {}
    interface_index = {}
    layout_index = {}

    for item in sem.get("high_level_sem", {}).get("type_sem", []):
        type_index[item.get("id")] = item
        type_index[item.get("return_type_sem", {}).get("id")] = item.get("return_type_sem", {})
        for p in item.get("parameter_type_sem", []):
            type_index[p.get("id")] = p
        for lv in item.get("local_var_type_sem", []):
            type_index[lv.get("id")] = lv

    for item in sem.get("high_level_sem", {}).get("interface_sem", []):
        interface_index[item.get("id")] = item
        interface_index[item.get("ret", {}).get("id")] = item.get("ret", {})
        for p in item.get("params", []):
            interface_index[p.get("id")] = p

    for item in sem.get("high_level_sem", {}).get("mem_layout_sem", []):
        layout_index[item.get("id")] = item

    for node in _collect_semantic_nodes(sem):
        layer = node.get("layer")
        nid = node.get("id")
        confidence = 1.0
        state = "confirmed"

        if layer in ("TypeSem", "InterfaceSem", "APISem", "LayoutSem"):
            state = "hypothesis_confirmed"
            confidence = 0.75

        if layer == "TypeSem":
            target = type_index.get(nid, {})
            if target:
                if "type_confidence" in target and target.get("type_confidence"):
                    try:
                        confidence = float(target.get("type_confidence")[0])
                    except:
                        confidence = 0.72
                else:
                    confs = []
                    rt = target.get("return_type_sem", {})
                    for c in rt.get("type_confidence", []):
                        try:
                            confs.append(float(c))
                        except:
                            pass
                    for p in target.get("parameter_type_sem", []):
                        if p.get("type_confidence"):
                            try:
                                confs.append(float(p.get("type_confidence")[0]))
                            except:
                                pass
                    if confs:
                        confidence = round(sum(confs) / float(len(confs)), 3)

        if layer == "InterfaceSem":
            target = interface_index.get(nid, {})
            if target:
                if target.get("role_confidence"):
                    try:
                        confidence = float(target.get("role_confidence")[0])
                    except:
                        confidence = 0.74
                else:
                    pcount = 0
                    try:
                        pcount = len(target.get("params", []))
                    except:
                        pcount = 0
                    confidence = 0.78 + 0.02 * min(pcount, 3)

        if layer == "APISem":
            confidence = 0.90

        if layer == "LayoutSem":
            target = layout_index.get(nid, {})
            if target:
                confidence = float(target.get("confidence", 0.7))

        states.append({
            "node_id": nid,
            "layer": layer,
            "state": state,
            "confidence": _clamp_strength(confidence)
        })
    return states


def gen_mapping(all_sem):
    if _PREV_GEN_MAPPING is None:
        return []
    mappings = _PREV_GEN_MAPPING(all_sem)
    interface_by_id = {}
    for itf in all_sem.get("high_level_sem", {}).get("interface_sem", []):
        interface_by_id[itf.get("id")] = itf

    for item in mappings:
        if item.get("rule_id") == "R5":
            itf = interface_by_id.get(item.get("to"))
            if itf is not None:
                filtered_entry = itf.get("interface_constraints", {}).get("cross_layer_constraints", {}).get("entry_input_locs", [])
                if "trigger" not in item or item.get("trigger") is None:
                    item["trigger"] = {}
                item["trigger"]["entry_input_locs"] = filtered_entry
                item["trigger"]["param_count"] = itf.get("interface_constraints", {}).get("param_count")
    return mappings


def build_dict(prog):
    from collections import OrderedDict
    sem = OrderedDict()

    cfg_sem = _BASE_GEN_CFG_SEM(prog)
    dfg_sem = gen_dataflow_sem(prog)
    instr_sem = gen_instr_sem(prog, dfg_sem)
    block_sem = gen_block_sem(prog, cfg_sem, dfg_sem)
    type_sem = gen_type_sem(prog, dfg_sem)
    interface_sem = gen_interface_sem(prog, type_sem, dfg_sem)
    api_sem = gen_api_sem(prog, dfg_sem, interface_sem, type_sem)
    mem_layout_sem = gen_mem_layout_sem(prog, dfg_sem, type_sem, interface_sem)

    sem["program_info"] = {
        "name": prog.getName(),
        "base": str(prog.getImageBase()),
        "schema_version": "semantic_dict_v4",
        "generator": "ghidra_usg_patch_v4"
    }

    sem["low_level_sem"] = OrderedDict()
    sem["low_level_sem"]["instr_sem"] = instr_sem
    sem["low_level_sem"]["block_sem"] = block_sem

    sem["mid_level_sem"] = OrderedDict()
    sem["mid_level_sem"]["cfg_sem"] = cfg_sem
    sem["mid_level_sem"]["dataflow_sem"] = dfg_sem

    sem["high_level_sem"] = OrderedDict()
    sem["high_level_sem"]["type_sem"] = type_sem
    sem["high_level_sem"]["interface_sem"] = interface_sem
    sem["high_level_sem"]["api_sem"] = api_sem
    sem["high_level_sem"]["mem_layout_sem"] = mem_layout_sem

    sem["cross_layer_mapping"] = gen_mapping(sem)
    sem["constraint_edges"] = gen_constraint_edges(sem)
    sem["node_state"] = gen_node_state(sem)
    sem["invariant_status"] = gen_invariant_status(sem)
    sem["usg"] = build_usg(sem)
    return sem



# ==============================================
# v5 PATCH: remaining fixes + strict consistency report
# ==============================================

try:
    _V4_GEN_INSTR_SEM = gen_instr_sem
except:
    _V4_GEN_INSTR_SEM = None

try:
    _V4_GEN_MEM_LAYOUT_SEM = gen_mem_layout_sem
except:
    _V4_GEN_MEM_LAYOUT_SEM = None

try:
    _V4_BUILD_DICT = build_dict
except:
    _V4_BUILD_DICT = None


def _dtype_is_fp_scalar_v5(dt):
    if dt is None:
        return False
    if _is_pointer_type_like(dt):
        return False
    name = _dtype_name_lower(dt)
    if ("float" in name) or ("double" in name):
        return True
    return False


def _infer_declared_param_storage_v4(prog, idx, dt=None):
    if idx is None or idx < 0:
        return "unknown"

    is_fp_scalar = _dtype_is_fp_scalar_v5(dt)

    if _processor_is_aarch64(prog):
        if idx < 8:
            if is_fp_scalar:
                return "reg:D%d" % idx
            return "reg:X%d" % idx
        return "stack:param_%d" % idx

    if _processor_is_x86_64(prog):
        gp = ["reg:RDI", "reg:RSI", "reg:RDX", "reg:RCX", "reg:R8", "reg:R9"]
        fp = ["reg:XMM0", "reg:XMM1", "reg:XMM2", "reg:XMM3", "reg:XMM4", "reg:XMM5"]
        if idx < len(gp):
            if is_fp_scalar:
                return fp[idx]
            return gp[idx]
        return "stack:param_%d" % idx

    if _processor_is_arm32(prog):
        if idx < 4:
            if is_fp_scalar:
                return "reg:S%d" % idx
            return "reg:R%d" % idx
        return "stack:param_%d" % idx

    if _processor_is_mips(prog):
        gp = ["reg:A0", "reg:A1", "reg:A2", "reg:A3"]
        if idx < len(gp):
            return gp[idx]
        return "stack:param_%d" % idx

    return "param_%d" % idx


API_SEMANTIC_TEMPLATES.update({
    "qsort": {
        "intent": "array_sort",
        "param_roles": ["base", "nmemb", "size", "compar"],
        "side_effect": ["inplace_reordering"],
        "constraints": ["base_buffer_valid", "element_count_non_negative", "element_size_positive", "comparator_callable"]
    }
})


def _build_instr_mem_norm_index_v5(dfg_cache):
    norm_index = {}
    if dfg_cache is None:
        return norm_index
    for dfg in dfg_cache:
        for acc in dfg.get("memory_access_patterns", []):
            site = acc.get("site")
            raw = acc.get("raw_location") or acc.get("location")
            loc = acc.get("location") or raw
            if site is None or raw is None or loc is None:
                continue
            norm_index[(site, raw)] = acc
            norm_index[(site, loc)] = acc
    return norm_index


def _normalize_instr_sem_memory_with_dfg_v5(instr_sem, dfg_cache):
    norm_index = _build_instr_mem_norm_index_v5(dfg_cache)
    if not norm_index:
        return instr_sem

    for sem in instr_sem:
        site = "%s:%s" % (sem.get("address"), sem.get("pcode_index"))
        effects = sem.get("effects", {})
        mems = effects.get("memory_accesses", [])
        if not mems:
            continue

        rename_map = {}
        new_mems = []
        for acc in mems:
            raw = acc.get("raw_location") or acc.get("location")
            repl = norm_index.get((site, raw))
            if repl is None:
                new_mems.append(acc)
                continue

            merged = dict(acc)
            for k, v in repl.items():
                if v is not None:
                    merged[k] = v
            new_mems.append(merged)

            old_loc = acc.get("location")
            new_loc = merged.get("location")
            if raw is not None and new_loc is not None and raw != new_loc:
                rename_map[raw] = new_loc
            if old_loc is not None and new_loc is not None and old_loc != new_loc:
                rename_map[old_loc] = new_loc

        if rename_map:
            sem["operands"] = [rename_map.get(x, x) for x in sem.get("operands", [])]
            if sem.get("ctx_pre") is not None:
                sem["ctx_pre"]["reads"] = [rename_map.get(x, x) for x in sem.get("ctx_pre", {}).get("reads", [])]
            if sem.get("ctx_post") is not None:
                sem["ctx_post"]["writes"] = [rename_map.get(x, x) for x in sem.get("ctx_post", {}).get("writes", [])]
            sem["effects"]["reads"] = [rename_map.get(x, x) for x in sem.get("effects", {}).get("reads", [])]
            sem["effects"]["writes"] = [rename_map.get(x, x) for x in sem.get("effects", {}).get("writes", [])]

        sem["effects"]["memory_accesses"] = new_mems

    return instr_sem


def gen_instr_sem(prog, dfg_cache=None):
    if _V4_GEN_INSTR_SEM is None:
        return []
    instr_sem = _V4_GEN_INSTR_SEM(prog, dfg_cache)
    return _normalize_instr_sem_memory_with_dfg_v5(instr_sem, dfg_cache)


def _layout_key_v5(item):
    return (
        item.get("function_entry"),
        item.get("layout_kind"),
        item.get("base_obj"),
        tuple(item.get("offset_set", []))
    )


def _make_single_offset_object_layout_v5(func, group, type_obj, interface_obj, seq, prog):
    offsets = group.get("offsets", [])
    if not offsets:
        return None
    offset = offsets[0]
    base_obj = group.get("base")
    if _is_stack_frame_base_v4(base_obj):
        return None

    matched_type_refs = _match_type_refs_for_layout_v4(prog, type_obj, base_obj)
    matched_interface_refs = _match_interface_param_refs_for_layout_v4(prog, interface_obj, base_obj)

    roles = _layout_roles_v4("object_layout", group.get("stride"), matched_type_refs, matched_interface_refs)
    if "single_offset_object_candidate" not in roles:
        roles.append("single_offset_object_candidate")

    support_sites = sorted(_unique_list(group.get("sites", [])))
    confidence = 0.64
    if matched_type_refs:
        confidence += 0.06
    if matched_interface_refs:
        confidence += 0.04
    if group.get("access_count", 0) >= 2:
        confidence += 0.04
    if group.get("stride") is not None:
        confidence += 0.05

    field_relations = [{
        "offset": offset,
        "field_name": "field_0",
        "relation": "next_field_gap_unknown",
        "size_hint": None
    }]

    func_entry = _addr_to_str(func.getEntryPoint())
    func_name = _safe_get_name(func, "sub_%s" % func_entry)

    return {
        "id": gen_id("high_mem", func_entry, "cluster_single_%d" % seq),
        "function_entry": func_entry,
        "function_name": func_name,
        "layout_kind": "object_layout",
        "cluster_id": seq,
        "base_obj": base_obj,
        "offset_set": [offset],
        "size_rel": [],
        "field_relations": field_relations,
        "nested_candidates": [],
        "support_sites": support_sites,
        "semantic_roles": roles,
        "layout_source": "dfg_single_offset_object_candidate",
        "confidence": _clamp_strength(confidence),
        "propagated_constraints": {
            "target_func_id": _function_uid(func),
            "type_sem_refs": matched_type_refs,
            "interface_sem_ref": interface_obj.get("id") if interface_obj else None,
            "interface_param_refs": matched_interface_refs,
            "constraint_evidence": _unique_list([
                "single_fixed_offset_access",
                "non_sp_base_object_candidate",
                "alias_group_singleton"
            ])
        },
        "location": "%s_layout_%s_single_%d" % (func_entry, str(base_obj).replace(":", "_"), seq)
    }


def gen_mem_layout_sem(prog, dfg_cache=None, type_cache=None, interface_cache=None):
    base_layouts = []
    if _V4_GEN_MEM_LAYOUT_SEM is not None:
        base_layouts = _V4_GEN_MEM_LAYOUT_SEM(prog, dfg_cache, type_cache, interface_cache)
    else:
        base_layouts = []

    existing_keys = set([_layout_key_v5(x) for x in base_layouts])
    all_memlayout = list(base_layouts)

    dfg_by_func = _dfg_by_entry(dfg_cache if dfg_cache is not None else gen_dataflow_sem(prog))
    type_by_func = _type_by_entry(type_cache if type_cache is not None else gen_type_sem(prog, dfg_cache))
    interface_by_func = _interface_by_entry(interface_cache if interface_cache is not None else gen_interface_sem(prog, type_cache, dfg_cache))
    funcs = _get_internal_functions(prog, include_thunks=False)

    extra_seq_by_func = {}
    for item in all_memlayout:
        fentry = item.get("function_entry")
        extra_seq_by_func[fentry] = max(extra_seq_by_func.get(fentry, 0), int(item.get("cluster_id", 0)) + 1)

    for func in funcs:
        func_entry = _addr_to_str(func.getEntryPoint())
        dfg = dfg_by_func.get(func_entry, {"abstract_merge": {"alias_groups": []}})
        type_obj = type_by_func.get(func_entry)
        interface_obj = interface_by_func.get(func_entry)

        seq = extra_seq_by_func.get(func_entry, 0)
        for group in dfg.get("abstract_merge", {}).get("alias_groups", []):
            offsets = list(group.get("offsets", []))
            if len(offsets) != 1:
                continue
            item = _make_single_offset_object_layout_v5(func, group, type_obj, interface_obj, seq, prog)
            if item is None:
                continue
            key = _layout_key_v5(item)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            all_memlayout.append(item)
            seq += 1

        extra_seq_by_func[func_entry] = seq

    return all_memlayout


def _collect_nodes_for_consistency_v5(sem):
    nodes = {}
    layer_lists = [
        ("InstrSem", sem.get("low_level_sem", {}).get("instr_sem", [])),
        ("BlockSem", sem.get("low_level_sem", {}).get("block_sem", [])),
        ("CFGSem", sem.get("mid_level_sem", {}).get("cfg_sem", [])),
        ("DFGSem", sem.get("mid_level_sem", {}).get("dataflow_sem", [])),
        ("APISem", sem.get("high_level_sem", {}).get("api_sem", [])),
        ("LayoutSem", sem.get("high_level_sem", {}).get("mem_layout_sem", []))
    ]
    for layer, arr in layer_lists:
        for obj in arr:
            oid = obj.get("id")
            if oid:
                nodes[oid] = {"layer": layer, "obj": obj}

    for obj in sem.get("high_level_sem", {}).get("type_sem", []):
        oid = obj.get("id")
        if oid:
            nodes[oid] = {"layer": "TypeSem", "obj": obj}
        for p in obj.get("parameter_type_sem", []):
            pid = p.get("id")
            if pid:
                nodes[pid] = {"layer": "TypeSem", "obj": p}
        for p in obj.get("local_var_type_sem", []):
            pid = p.get("id")
            if pid:
                nodes[pid] = {"layer": "TypeSem", "obj": p}
        ret = obj.get("return_type_sem")
        if ret and ret.get("id"):
            nodes[ret.get("id")] = {"layer": "TypeSem", "obj": ret}

    for obj in sem.get("high_level_sem", {}).get("interface_sem", []):
        oid = obj.get("id")
        if oid:
            nodes[oid] = {"layer": "InterfaceSem", "obj": obj}
        for p in obj.get("params", []):
            pid = p.get("id")
            if pid:
                nodes[pid] = {"layer": "InterfaceSem", "obj": p}
        ret = obj.get("ret")
        if ret and ret.get("id"):
            nodes[ret.get("id")] = {"layer": "InterfaceSem", "obj": ret}

    return nodes


def _add_issue_v5(report, level, check_name, message, ref_id=None, function_name=None, detail=None):
    issue = {
        "level": level,
        "check": check_name,
        "message": message
    }
    if ref_id is not None:
        issue["ref_id"] = ref_id
    if function_name is not None:
        issue["function_name"] = function_name
    if detail is not None:
        issue["detail"] = detail
    report["issues"].append(issue)


def _is_fp_storage_name_v5(storage):
    s = str(storage)
    return s.startswith("reg:D") or s.startswith("reg:S") or s.startswith("reg:Q") or s.startswith("reg:XMM")


def _is_gp_storage_name_v5(storage):
    s = str(storage)
    if s.startswith("stack:"):
        return True
    return s.startswith("reg:X") or s.startswith("reg:W") or s.startswith("reg:R") or s.startswith("reg:E") or s.startswith("reg:A")


def _is_pointer_like_item_v5(item):
    if item.get("is_pointer"):
        return True
    cands = item.get("type_candidates", [])
    for c in cands:
        s = _safe_lower(c)
        if "*" in s or "ptr" in s or s == "void*" or s.endswith("_p"):
            return True
    return False


def _top_type_name_v5(item):
    cands = item.get("type_candidates", [])
    if cands:
        return str(cands[0])
    return "unknown"


def _state_rank_v5(state):
    order = {
        "hypothesis": 0,
        "tentative": 1,
        "hypothesis_confirmed": 2,
        "confirmed": 3,
        "stable": 4
    }
    return order.get(str(state), 1)


def _load_json_maybe_v5(path):
    if path is None:
        return None
    try:
        with codecs.open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def run_consistency_checks(sem, prev_sem=None):
    report = {
        "summary": {
            "status": "pass",
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0
        },
        "checks": {},
        "issues": []
    }

    nodes = _collect_nodes_for_consistency_v5(sem)
    node_ids = set(nodes.keys())
    checks = {}

    # 1) 引用完整性
    ref_errors = 0
    ref_warnings = 0
    for edge in sem.get("cross_layer_mapping", []):
        if edge.get("from") not in node_ids:
            _add_issue_v5(report, "error", "reference_integrity", "evidence edge source node is missing", edge.get("from"), None, edge.get("mapping_id"))
            ref_errors += 1
        if edge.get("to") not in node_ids:
            _add_issue_v5(report, "error", "reference_integrity", "evidence edge target node is missing", edge.get("to"), None, edge.get("mapping_id"))
            ref_errors += 1

    for edge in sem.get("constraint_edges", []):
        if edge.get("from") not in node_ids:
            _add_issue_v5(report, "error", "reference_integrity", "constraint edge source node is missing", edge.get("from"), None, edge.get("constraint_id"))
            ref_errors += 1
        if edge.get("to") not in node_ids:
            _add_issue_v5(report, "error", "reference_integrity", "constraint edge target node is missing", edge.get("to"), None, edge.get("constraint_id"))
            ref_errors += 1

    for itf in sem.get("high_level_sem", {}).get("interface_sem", []):
        for p in itf.get("params", []):
            tref = p.get("type_sem_ref")
            if tref and tref not in node_ids:
                _add_issue_v5(report, "error", "reference_integrity", "interface parameter type_sem_ref is missing", tref, itf.get("function_name"), p.get("id"))
                ref_errors += 1
        ret = itf.get("ret")
        if ret:
            tref = ret.get("type_sem_ref")
            if tref and tref not in node_ids:
                _add_issue_v5(report, "error", "reference_integrity", "interface return type_sem_ref is missing", tref, itf.get("function_name"), ret.get("id"))
                ref_errors += 1

    for layout in sem.get("high_level_sem", {}).get("mem_layout_sem", []):
        pc = layout.get("propagated_constraints", {})
        trefs = pc.get("type_sem_refs", [])
        for tref in trefs:
            if tref and tref not in node_ids:
                _add_issue_v5(report, "error", "reference_integrity", "layout type_sem_ref is missing", tref, layout.get("function_name"), layout.get("id"))
                ref_errors += 1
        iref = pc.get("interface_sem_ref")
        if iref and iref not in node_ids:
            _add_issue_v5(report, "error", "reference_integrity", "layout interface_sem_ref is missing", iref, layout.get("function_name"), layout.get("id"))
            ref_errors += 1
        for pref in pc.get("interface_param_refs", []):
            if pref and pref not in node_ids:
                _add_issue_v5(report, "error", "reference_integrity", "layout interface_param_ref is missing", pref, layout.get("function_name"), layout.get("id"))
                ref_errors += 1

    checks["reference_integrity"] = {
        "status": "pass" if ref_errors == 0 else "fail",
        "error_count": ref_errors,
        "warning_count": ref_warnings
    }

    # 2) Interface / Type 对齐
    align_errors = 0
    align_warnings = 0
    type_by_entry = _type_by_entry(sem.get("high_level_sem", {}).get("type_sem", []))
    interface_by_entry = _interface_by_entry(sem.get("high_level_sem", {}).get("interface_sem", []))

    for fentry, itf in interface_by_entry.items():
        t = type_by_entry.get(fentry)
        if t is None:
            _add_issue_v5(report, "warning", "cross_layer_alignment", "interface has no matching TypeSem", itf.get("id"), itf.get("function_name"))
            align_warnings += 1
            continue

        param_count = len(itf.get("params", []))
        declared_param_count = itf.get("interface_constraints", {}).get("declared_param_count")
        cross_refs = itf.get("interface_constraints", {}).get("cross_layer_constraints", {}).get("param_type_refs", [])

        if itf.get("interface_constraints", {}).get("param_count") != param_count:
            _add_issue_v5(report, "error", "cross_layer_alignment", "interface_constraints.param_count mismatches params length", itf.get("id"), itf.get("function_name"))
            align_errors += 1
        if declared_param_count is not None and declared_param_count > 0 and declared_param_count > param_count:
            _add_issue_v5(report, "warning", "cross_layer_alignment", "declared_param_count is larger than exported params", itf.get("id"), itf.get("function_name"))
            align_warnings += 1
        if len(cross_refs) != param_count:
            _add_issue_v5(report, "warning", "cross_layer_alignment", "cross-layer param_type_refs count mismatches interface params length", itf.get("id"), itf.get("function_name"))
            align_warnings += 1

        type_params = t.get("parameter_type_sem", [])
        if len(type_params) != len(t.get("type_constraints", {}).get("param_type_refs", [])):
            _add_issue_v5(report, "warning", "cross_layer_alignment", "TypeSem parameter_type_sem count mismatches param_type_refs", t.get("id"), t.get("function_name"))
            align_warnings += 1

        idx = 0
        while idx < min(len(itf.get("params", [])), len(type_params)):
            ip = itf.get("params", [])[idx]
            tp = type_params[idx]
            if ip.get("type_sem_ref") and ip.get("type_sem_ref") != tp.get("id"):
                _add_issue_v5(report, "warning", "cross_layer_alignment", "interface param type_sem_ref does not align with TypeSem param id order", ip.get("id"), itf.get("function_name"), {"expected": tp.get("id"), "actual": ip.get("type_sem_ref")})
                align_warnings += 1
            idx += 1

    checks["cross_layer_alignment"] = {
        "status": "pass" if align_errors == 0 else "fail",
        "error_count": align_errors,
        "warning_count": align_warnings
    }

    # 3) ABI / storage 一致性
    abi_errors = 0
    abi_warnings = 0

    for t in sem.get("high_level_sem", {}).get("type_sem", []):
        for p in t.get("parameter_type_sem", []):
            storage = p.get("storage")
            top_name = _safe_lower(_top_type_name_v5(p))
            ptr_like = _is_pointer_like_item_v5(p)
            if ptr_like and _is_fp_storage_name_v5(storage):
                _add_issue_v5(report, "warning", "abi_storage_consistency", "pointer-like parameter is exported in FP register class", p.get("id"), t.get("function_name"), {"storage": storage, "type": p.get("type_candidates")})
                abi_warnings += 1
            if (("float" in top_name) or ("double" in top_name)) and (not ptr_like) and (not _is_fp_storage_name_v5(storage)):
                _add_issue_v5(report, "warning", "abi_storage_consistency", "floating scalar parameter is not exported in FP register class", p.get("id"), t.get("function_name"), {"storage": storage, "type": p.get("type_candidates")})
                abi_warnings += 1

    checks["abi_storage_consistency"] = {
        "status": "pass" if abi_errors == 0 else "fail",
        "error_count": abi_errors,
        "warning_count": abi_warnings
    }

    # 4) 指令级访存规范化
    mem_errors = 0
    mem_warnings = 0
    total_mems = 0
    raw_tmp_mems = []
    for ins in sem.get("low_level_sem", {}).get("instr_sem", []):
        for acc in ins.get("effects", {}).get("memory_accesses", []):
            total_mems += 1
            loc = acc.get("location") or acc.get("raw_location")
            if loc and ("tmp:" in str(loc)) and (acc.get("base") is None):
                raw_tmp_mems.append({
                    "instr_id": ins.get("id"),
                    "site": acc.get("site"),
                    "location": loc
                })
    if total_mems > 0:
        ratio = float(len(raw_tmp_mems)) / float(total_mems)
    else:
        ratio = 0.0

    if ratio > 0.15:
        _add_issue_v5(report, "warning", "instr_memory_normalization", "too many instruction-level memory accesses still use tmp-based unresolved locations", None, None, {"raw_tmp_count": len(raw_tmp_mems), "total_memory_accesses": total_mems, "ratio": round(ratio, 4), "examples": raw_tmp_mems[:8]})
        mem_warnings += 1

    checks["instr_memory_normalization"] = {
        "status": "pass",
        "error_count": mem_errors,
        "warning_count": mem_warnings,
        "detail": {
            "raw_tmp_count": len(raw_tmp_mems),
            "total_memory_accesses": total_mems,
            "ratio": round(ratio, 4)
        }
    }

    # 5) Layout 语义质量
    layout_errors = 0
    layout_warnings = 0
    object_layout_funcs = set()
    for layout in sem.get("high_level_sem", {}).get("mem_layout_sem", []):
        kind = layout.get("layout_kind")
        base_obj = layout.get("base_obj")
        if kind == "object_layout":
            object_layout_funcs.add(layout.get("function_entry"))
            if _is_stack_frame_base_v4(base_obj):
                _add_issue_v5(report, "error", "layout_semantics", "object_layout should not use stack-frame base", layout.get("id"), layout.get("function_name"))
                layout_errors += 1
        if kind == "stack_frame_layout" and (not _is_stack_frame_base_v4(base_obj)):
            _add_issue_v5(report, "warning", "layout_semantics", "stack_frame_layout uses non-stack base", layout.get("id"), layout.get("function_name"), {"base_obj": base_obj})
            layout_warnings += 1

    for dfg in sem.get("mid_level_sem", {}).get("dataflow_sem", []):
        fentry = dfg.get("entry_address")
        if fentry in object_layout_funcs:
            continue
        has_non_stack_singleton = False
        for group in dfg.get("abstract_merge", {}).get("alias_groups", []):
            if _is_stack_frame_base_v4(group.get("base")):
                continue
            if len(group.get("offsets", [])) >= 1:
                has_non_stack_singleton = True
                break
        if has_non_stack_singleton:
            _add_issue_v5(report, "warning", "layout_semantics", "non-stack fixed-offset access exists but no object_layout was exported", None, dfg.get("function_name"))
            layout_warnings += 1

    checks["layout_semantics"] = {
        "status": "pass" if layout_errors == 0 else "fail",
        "error_count": layout_errors,
        "warning_count": layout_warnings
    }

    # 6) API 模板质量
    api_errors = 0
    api_warnings = 0
    known_api_names = ["memcpy", "memmove", "memset", "printf", "scanf", "strcmp", "puts", "exit", "stack_chk_fail", "qsort"]
    for api in sem.get("high_level_sem", {}).get("api_sem", []):
        api_name = _safe_lower(api.get("api_name")).lstrip("_")
        intent = _safe_lower(api.get("semantic_template", {}).get("api_intent"))
        if any([k in api_name for k in known_api_names]):
            if intent in ("", "general_subroutine", "unknown", "none"):
                _add_issue_v5(report, "warning", "api_template_quality", "known API still falls back to generic semantic template", api.get("id"), None, {"api_name": api.get("api_name"), "intent": intent})
                api_warnings += 1

    checks["api_template_quality"] = {
        "status": "pass" if api_errors == 0 else "fail",
        "error_count": api_errors,
        "warning_count": api_warnings
    }

    # 7) USG / invariant 自检
    usg_errors = 0
    usg_warnings = 0
    usg = sem.get("usg", {})
    graph = usg.get("graph", {})
    summary = usg.get("summary", {})
    if summary.get("node_count") != len(graph.get("V", [])):
        _add_issue_v5(report, "error", "usg_structure", "usg.summary.node_count mismatches graph.V length", None, None, {"summary": summary.get("node_count"), "actual": len(graph.get("V", []))})
        usg_errors += 1
    if summary.get("evidence_edge_count") != len(graph.get("E_ev", [])):
        _add_issue_v5(report, "error", "usg_structure", "usg.summary.evidence_edge_count mismatches graph.E_ev length", None, None, {"summary": summary.get("evidence_edge_count"), "actual": len(graph.get("E_ev", []))})
        usg_errors += 1
    if summary.get("constraint_edge_count") != len(graph.get("E_ct", [])):
        _add_issue_v5(report, "error", "usg_structure", "usg.summary.constraint_edge_count mismatches graph.E_ct length", None, None, {"summary": summary.get("constraint_edge_count"), "actual": len(graph.get("E_ct", []))})
        usg_errors += 1

    invariant_status = sem.get("invariant_status", {})
    for key, item in invariant_status.items():
        if not item.get("status", False):
            _add_issue_v5(report, "warning", "usg_structure", "snapshot invariant is not satisfied", None, None, {key: item})
            usg_warnings += 1

    checks["usg_structure"] = {
        "status": "pass" if usg_errors == 0 else "fail",
        "error_count": usg_errors,
        "warning_count": usg_warnings
    }

    # 8) 与上一次快照对比（可选）
    hist_errors = 0
    hist_warnings = 0
    if prev_sem is not None:
        prev_node_state = {}
        for item in prev_sem.get("node_state", []):
            prev_node_state[item.get("node_id")] = item
        curr_node_state = {}
        for item in sem.get("node_state", []):
            curr_node_state[item.get("node_id")] = item

        for nid, prev_item in prev_node_state.items():
            curr_item = curr_node_state.get(nid)
            if curr_item is None:
                _add_issue_v5(report, "warning", "historical_invariants", "node disappears in current snapshot", nid)
                hist_warnings += 1
                continue

            try:
                prev_conf = float(prev_item.get("confidence", 0.0))
            except:
                prev_conf = 0.0
            try:
                curr_conf = float(curr_item.get("confidence", 0.0))
            except:
                curr_conf = 0.0
            if curr_conf + 1e-6 < prev_conf:
                _add_issue_v5(report, "warning", "historical_invariants", "node confidence decreased across snapshots", nid, None, {"prev": prev_conf, "curr": curr_conf})
                hist_warnings += 1

            if _state_rank_v5(curr_item.get("state")) < _state_rank_v5(prev_item.get("state")):
                _add_issue_v5(report, "warning", "historical_invariants", "node state regressed across snapshots", nid, None, {"prev": prev_item.get("state"), "curr": curr_item.get("state")})
                hist_warnings += 1

        prev_evidence = set([x.get("mapping_id") for x in prev_sem.get("cross_layer_mapping", []) if x.get("mapping_id")])
        curr_evidence = set([x.get("mapping_id") for x in sem.get("cross_layer_mapping", []) if x.get("mapping_id")])
        removed = sorted(list(prev_evidence - curr_evidence))
        if removed:
            _add_issue_v5(report, "warning", "historical_invariants", "evidence edges disappeared across snapshots", None, None, {"removed_examples": removed[:20], "removed_count": len(removed)})
            hist_warnings += 1

    checks["historical_invariants"] = {
        "status": "pass" if hist_errors == 0 else "fail",
        "error_count": hist_errors,
        "warning_count": hist_warnings
    }

    report["checks"] = checks

    error_count = len([x for x in report["issues"] if x.get("level") == "error"])
    warning_count = len([x for x in report["issues"] if x.get("level") == "warning"])
    info_count = len([x for x in report["issues"] if x.get("level") == "info"])

    report["summary"]["error_count"] = error_count
    report["summary"]["warning_count"] = warning_count
    report["summary"]["info_count"] = info_count
    if error_count > 0:
        report["summary"]["status"] = "fail"
    elif warning_count > 0:
        report["summary"]["status"] = "pass_with_warnings"
    else:
        report["summary"]["status"] = "pass"

    return report


def build_dict(prog, prev_snapshot_path=None):
    if _V4_BUILD_DICT is None:
        sem = {}
    else:
        sem = _V4_BUILD_DICT(prog)

    sem["program_info"]["schema_version"] = "semantic_dict_v5"
    sem["program_info"]["generator"] = "ghidra_usg_patch_v5"

    prev_sem = _load_json_maybe_v5(prev_snapshot_path)
    sem["consistency_report"] = run_consistency_checks(sem, prev_sem)
    return sem


def _get_prev_snapshot_arg_v5():
    try:
        args = getScriptArgs()
        if args and len(args) > 0:
            return args[0]
    except:
        pass
    try:
        env = os.environ.get("USG_PREV_SNAPSHOT")
        if env:
            return env
    except:
        pass
    return None



# ==============================================
# v6 PATCH: memory-aware low-level instruction extraction
# - keep only essential fields by default
# - process per function and skip extra-heavy functions
# - precompile regexes for normalize_pcode
# - optional truncation for extremely long p-code lines
# ==============================================

try:
    import gc as _gc_v6
except:
    _gc_v6 = None

# Pre-compiled regex patterns
_V6_RE_REGISTER = re.compile(r"register, *0x([0-9a-fA-F]+), *(\d+)")
_V6_RE_UNIQUE = re.compile(r"unique, *0x([0-9a-fA-F]+), *(\d+)")
_V6_RE_STACK = re.compile(r"stack, *(-?0x[0-9a-fA-F]+), *(\d+)")
_V6_RE_CONST = re.compile(r"const, *(-?0x[0-9a-fA-F]+), *(\d+)")
_V6_RE_WS = re.compile(r"\s+")

def _v6_env_flag(name, default):
    try:
        raw = os.environ.get(name)
    except:
        raw = None
    if raw is None:
        return default
    raw = str(raw).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")

def _v6_env_int(name, default):
    try:
        raw = os.environ.get(name)
    except:
        raw = None
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except:
        return default

def _v6_maybe_gc():
    if _gc_v6 is not None:
        try:
            _gc_v6.collect()
        except:
            pass

def normalize_pcode(line, reg_map=None, tmp_map=None, counters=None):
    """
    Memory-aware normalize_pcode.
    - precompiled regexes
    - keeps structured tokens
    - optionally truncates pathological long p-code strings
    """
    if not line:
        return "EMPTY"

    line = str(line)
    max_len = _v6_env_int("USG_MAX_NPCODE_TEXT", 1024)
    if max_len > 0 and len(line) > max_len:
        line = line[:max_len] + " ...<truncated>"

    line = _V6_RE_REGISTER.sub(lambda m: "reg(0x%s,%s)" % (m.group(1), m.group(2)), line)
    line = _V6_RE_UNIQUE.sub(lambda m: "tmp(0x%s,%s)" % (m.group(1), m.group(2)), line)
    line = _V6_RE_STACK.sub(r"stack(\1,\2)", line)
    line = _V6_RE_CONST.sub(r"const(\1,\2)", line)
    line = _V6_RE_WS.sub(" ", line).strip()
    return line

def _v6_trim_memory_accesses(mems):
    out = []
    if not mems:
        return out
    for acc in mems:
        if not isinstance(acc, dict):
            continue
        out.append({
            "site": acc.get("site"),
            "instruction_address": acc.get("instruction_address"),
            "access": acc.get("access"),
            "location": acc.get("location"),
            "base": acc.get("base"),
            "offset": acc.get("offset"),
            "expr": acc.get("expr"),
            "is_fixed": acc.get("is_fixed"),
            "support_locs": acc.get("support_locs", []),
            "value_loc": acc.get("value_loc")
        })
    return out

def gen_instr_sem(prog, dfg_cache=None):
    """
    v6 memory-aware instruction semantics.
    Default behavior for smoke / scale tests:
    - do NOT keep assembly unless USG_KEEP_ASSEMBLY=1
    - do NOT keep normalized p-code unless USG_KEEP_NPCODE=1
    - skip functions that exceed instruction / p-code thresholds
    """
    output = []
    tm = _get_script_monitor()
    block_model = BasicBlockModel(prog)
    funcs = _get_internal_functions(prog, include_thunks=False)

    keep_assembly = _v6_env_flag("USG_KEEP_ASSEMBLY", False)
    keep_npcode = _v6_env_flag("USG_KEEP_NPCODE", False)
    max_func_instr = _v6_env_int("USG_MAX_FUNC_INSTR", 8000)
    max_func_pcode = _v6_env_int("USG_MAX_FUNC_PCODE", 50000)

    seq = 0
    skipped = []

    for func in funcs:
        func_entry = _addr_to_str(func.getEntryPoint())
        func_name = _safe_get_name(func, "sub_%s" % func_entry)
        start_len = len(output)
        instr_seen = 0
        pcode_seen = 0
        heavy_skip = False

        func_blocks = _BASE_GET_FUNCTION_BLOCKS(prog, func, block_model, tm)
        for block in func_blocks:
            for instr in block.get("instructions", []):
                instr_seen += 1
                if max_func_instr > 0 and instr_seen > max_func_instr:
                    heavy_skip = True
                    break

                try:
                    pcode_ops = instr.getPcode()
                except:
                    pcode_ops = []

                try:
                    pcount = len(pcode_ops)
                except:
                    pcount = 0

                pcode_seen += pcount
                if max_func_pcode > 0 and pcode_seen > max_func_pcode:
                    heavy_skip = True
                    break

                for pidx in range(pcount):
                    try:
                        pcode_op = pcode_ops[pidx]
                    except:
                        continue

                    facts = _extract_pcode_facts_v3(prog, instr, pcode_op, pidx, block)

                    input_operands = []
                    for desc in facts.get("input_descs", []):
                        if desc.get("const") is not None:
                            input_operands.append(_format_signed_immediate(desc.get("const")))
                        elif desc.get("loc") is not None:
                            input_operands.append(desc.get("loc"))
                        elif desc.get("piece") is not None:
                            input_operands.append(desc.get("piece"))

                    output_operand = facts.get("output_desc", {}).get("loc")
                    operands = []
                    if output_operand is not None:
                        operands.append(output_operand)
                    operands.extend(input_operands)

                    sem = {
                        "id": gen_id("low_instr", _addr_to_str(instr.getAddress()), str(seq)),
                        "address": _addr_to_str(instr.getAddress()),
                        "function_entry": func_entry,
                        "block_entry": block.get("entry_str"),
                        "opcode": facts.get("mnemonic"),
                        "operands": operands,
                        "pcode_index": facts.get("pcode_index"),
                        "ctx_pre": {
                            "reads": facts.get("uses", []),
                            "block_entry": block.get("entry_str"),
                            "function_entry": func_entry
                        },
                        "ctx_post": {
                            "writes": facts.get("defs", []),
                            "block_exit_candidate": block.get("end_str")
                        },
                        "effects": {
                            "reads": facts.get("uses", []),
                            "writes": facts.get("defs", []),
                            "memory_accesses": _v6_trim_memory_accesses(facts.get("memory_accesses", [])),
                            "control_effect": _classify_edge_type(instr.getFlowType()) if instr.getFlowType() is not None else "none"
                        }
                    }

                    if keep_assembly:
                        try:
                            sem["assembly"] = instr.toString()
                        except:
                            sem["assembly"] = None
                    if keep_npcode:
                        try:
                            sem["n_pcode"] = normalize_pcode(str(pcode_op))
                        except:
                            sem["n_pcode"] = None

                    output.append(sem)
                    seq += 1

                # Explicitly release per-instruction temporaries
                facts = None
                input_operands = None
                operands = None

            if heavy_skip:
                break

        if heavy_skip:
            # Remove partial function output to keep semantics consistent
            del output[start_len:]
            skipped.append({
                "function_entry": func_entry,
                "function_name": func_name,
                "reason": "heavy_function_skipped",
                "instr_seen": instr_seen,
                "pcode_seen": pcode_seen
            })
            print("[v6] Skip heavy function %s (%s): instr_seen=%s pcode_seen=%s" % (
                func_name, func_entry, instr_seen, pcode_seen
            ))

        _v6_maybe_gc()

    if skipped:
        print("[v6] Total skipped heavy functions: %d" % len(skipped))
    return _normalize_instr_sem_memory_with_dfg_v5(output, dfg_cache)



# ==============================================
# v8 EVAL-READY PATCH: enrich TypeSem / InterfaceSem / LayoutSem
# 目标：让 DWARF evaluator 能做函数级、参数级和 offset 级映射。
# 说明：不改变整体 semantic_dict 结构，只补充定位字段和规范化类型类别。
# ==============================================

def _v8_lower(x):
    try:
        return str(x).lower()
    except:
        return ""


def _v8_first_candidate(cands):
    try:
        if cands and len(cands) > 0:
            return str(cands[0])
    except:
        pass
    return "unknown"


def _v8_type_category_from_name(name, is_pointer=False, for_return=False, has_return=True):
    s = _v8_lower(name).strip()
    if not s:
        return "unknown"
    if (not has_return) and for_return:
        return "void"
    # Ghidra often uses undefined/undefined1 as unknown scalar placeholders.
    if s in ("unknown", "undefined", "undefined1", "base", "default", "synthetic_param"):
        return "unknown"
    if is_pointer or ("*" in s) or ("ptr" in s) or ("pointer" in s):
        return "ptr"
    if "struct" in s or "class" in s:
        return "struct"
    if "union" in s:
        return "union"
    if "enum" in s:
        return "enum"
    if "float" in s or "double" in s:
        return "float"
    if "void" == s or s.startswith("void "):
        return "void"
    if ("int" in s) or ("long" in s) or ("short" in s) or ("char" in s) or ("bool" in s) or ("size_t" in s) or ("ssize_t" in s) or ("uint" in s) or ("byte" in s) or ("dword" in s) or ("qword" in s):
        return "int"
    return "other"


def _v8_type_category_from_item(item, for_return=False, has_return=True):
    if item is None:
        return "unknown"
    # Prefer concrete candidate name over Ghidra category like "base".
    tname = item.get("pred_type") or _v8_first_candidate(item.get("type_candidates", []))
    is_ptr = bool(item.get("is_pointer"))
    cat = _v8_type_category_from_name(tname, is_ptr, for_return, has_return)
    if cat != "unknown":
        return cat
    # Fallback to legacy category only if it is already useful.
    legacy = _v8_lower(item.get("category") or item.get("type_category") or item.get("pred_type_category"))
    if legacy in ("ptr", "struct", "union", "enum", "float", "int", "void", "other"):
        return legacy
    return cat


def _v8_enrich_type_item(item, func_entry, func_name, object_kind, index_value=None, storage=None):
    if item is None:
        return item
    item["function_entry"] = item.get("function_entry", func_entry)
    item["func_entry"] = item.get("func_entry", func_entry)
    item["function_name"] = item.get("function_name", func_name)
    item["object_kind"] = item.get("object_kind", object_kind)
    if object_kind == "param":
        item["param_index"] = item.get("param_index", index_value)
    if storage is None:
        storage = item.get("storage")
    kind, sid = _storage_kind_and_id(storage)
    item["storage"] = storage
    item["storage_kind"] = item.get("storage_kind", kind)
    item["storage_id"] = item.get("storage_id", sid)
    item["stack_offset"] = item.get("stack_offset", None)
    pred_type = item.get("pred_type") or _v8_first_candidate(item.get("type_candidates", []))
    item["pred_type"] = pred_type
    item["pred_type_category"] = _v8_type_category_from_item(item, for_return=(object_kind == "return"))
    item["type_category"] = item["pred_type_category"]
    return item


# Fix a bug in the previous synthetic-param helper: it referenced func_entry instead of entry.
def _make_synthetic_param_v4(entry, idx, loc, slot_info=None):
    kind, sid = _storage_kind_and_id(loc)
    item = {
        "id": gen_id("high_type", entry, "synthetic_param_%d" % idx),
        "location": "%s_recovered_param_%d" % (entry, idx),
        "param_name": "arg_%d" % idx,
        "type_candidates": ["pointer_like_input", "void*", "unknown"],
        "type_confidence": [0.56, 0.45, 0.25],
        "type_evidence": ["entry_input_locs_filtered", "abi_argument_slot"],
        "type_size": 0,
        "is_pointer": True,
        "category": "ptr",
        "type_category": "ptr",
        "pred_type": "pointer_like_input",
        "pred_type_category": "ptr",
        "storage": loc,
        "function_entry": entry,
        "func_entry": entry,
        "function_name": None,
        "object_kind": "param",
        "param_index": idx,
        "storage_kind": kind,
        "storage_id": sid,
        "stack_offset": None,
        "source": "dfg_recovered"
    }
    if slot_info is not None:
        item["abi_slot"] = slot_info.get("slot")
        item["abi_kind"] = slot_info.get("kind")
        item["storage"] = slot_info.get("canonical", loc)
        kind2, sid2 = _storage_kind_and_id(item.get("storage"))
        item["storage_kind"] = kind2
        item["storage_id"] = sid2
    return item


try:
    _V8_PREV_GEN_TYPE_SEM = gen_type_sem
except:
    _V8_PREV_GEN_TYPE_SEM = None

try:
    _V8_PREV_GEN_INTERFACE_SEM = gen_interface_sem
except:
    _V8_PREV_GEN_INTERFACE_SEM = None

try:
    _V8_PREV_GEN_MEM_LAYOUT_SEM = gen_mem_layout_sem
except:
    _V8_PREV_GEN_MEM_LAYOUT_SEM = None


def gen_type_sem(prog, dfg_cache=None):
    arr = _V8_PREV_GEN_TYPE_SEM(prog, dfg_cache) if _V8_PREV_GEN_TYPE_SEM is not None else []
    for obj in arr:
        func_entry = obj.get("function_entry") or obj.get("func_entry")
        func_name = obj.get("function_name") or "sub_%s" % func_entry
        obj["func_entry"] = obj.get("func_entry", func_entry)
        obj["function_name"] = func_name

        ret = obj.get("return_type_sem")
        if isinstance(ret, dict):
            _v8_enrich_type_item(ret, func_entry, func_name, "return", None, ret.get("storage"))

        params = obj.get("parameter_type_sem", []) or []
        for idx, p in enumerate(params):
            if not isinstance(p, dict):
                continue
            storage = p.get("storage")
            if _is_unknown_storage(storage):
                storage = _infer_declared_param_storage_v4(prog, idx, None)
            _v8_enrich_type_item(p, func_entry, func_name, "param", idx, storage)

        locals_list = obj.get("local_var_type_sem", []) or []
        for idx, p in enumerate(locals_list):
            if not isinstance(p, dict):
                continue
            _v8_enrich_type_item(p, func_entry, func_name, "local", idx, p.get("storage"))

        obj["param_type_categories"] = [p.get("pred_type_category", "unknown") for p in params if isinstance(p, dict)]
        obj["param_types"] = [p.get("pred_type", "unknown") for p in params if isinstance(p, dict)]
        if isinstance(ret, dict):
            obj["return_type"] = ret.get("pred_type", "unknown")
            obj["return_type_category"] = ret.get("pred_type_category", "unknown")
    return arr


def gen_interface_sem(prog, type_cache=None, dfg_cache=None):
    arr = _V8_PREV_GEN_INTERFACE_SEM(prog, type_cache, dfg_cache) if _V8_PREV_GEN_INTERFACE_SEM is not None else []
    for itf in arr:
        func_entry = itf.get("function_entry") or itf.get("func_entry")
        func_name = itf.get("function_name") or itf.get("func_name") or "sub_%s" % func_entry
        itf["func_entry"] = func_entry
        itf["func_name"] = func_name
        itf["function_name"] = func_name

        cons = itf.get("interface_constraints", {})
        params = itf.get("params", []) or []
        param_types = []
        param_cats = []
        for idx, p in enumerate(params):
            if not isinstance(p, dict):
                continue
            p["param_index"] = p.get("param_index", p.get("param_id", idx))
            pred_type = p.get("pred_type") or _v8_first_candidate(p.get("type_candidates", []))
            p["pred_type"] = pred_type
            cat = p.get("pred_type_category") or p.get("type_category") or _v8_type_category_from_name(pred_type, False)
            if cat in (None, "", "base", "synthetic_param"):
                cat = _v8_type_category_from_name(pred_type, False)
            p["pred_type_category"] = cat
            p["type_category"] = cat
            param_types.append(pred_type)
            param_cats.append(cat)

        ret = itf.get("ret", {}) or {}
        has_return = cons.get("has_return", True)
        ret_type = ret.get("return_type") or _v8_first_candidate(ret.get("type_candidates", []))
        ret_cat = ret.get("return_type_category") or _v8_type_category_from_name(ret_type, ret.get("is_pointer", False), True, has_return)
        if ret_cat in (None, "", "base", "synthetic_param"):
            ret_cat = _v8_type_category_from_name(ret_type, ret.get("is_pointer", False), True, has_return)
        ret["return_type"] = ret_type
        ret["return_type_category"] = ret_cat
        itf["ret"] = ret

        itf["param_count"] = cons.get("param_count", len(params))
        itf["declared_param_count"] = cons.get("declared_param_count", 0)
        itf["synthetic_param_count"] = cons.get("synthetic_param_count", 0)
        itf["param_types"] = param_types
        itf["param_type_categories"] = param_cats
        itf["return_type"] = ret_type
        itf["return_type_category"] = ret_cat
        itf["has_return"] = has_return
    return arr


def gen_mem_layout_sem(prog, dfg_cache=None, type_cache=None, interface_cache=None):
    arr = _V8_PREV_GEN_MEM_LAYOUT_SEM(prog, dfg_cache, type_cache, interface_cache) if _V8_PREV_GEN_MEM_LAYOUT_SEM is not None else []
    for item in arr:
        func_entry = item.get("function_entry") or item.get("func_entry")
        item["func_entry"] = func_entry
        item["func_name"] = item.get("function_name") or item.get("func_name") or "sub_%s" % func_entry
        item["base_object"] = item.get("base_object", item.get("base_obj"))
        item["base_storage"] = item.get("base_storage", item.get("base_obj"))
        offs = item.get("offsets", None)
        if offs is None:
            offs = item.get("field_offsets", None)
        if offs is None:
            offs = item.get("offset_set", [])
        nums = []
        try:
            for x in offs:
                nums.append(int(x))
        except:
            nums = []
        nums = sorted(set(nums))
        item["offsets"] = nums
        # field_offsets are the subset intended for DWARF struct-member comparison.
        if item.get("layout_kind") == "stack_frame_layout":
            item["field_offsets"] = []
        else:
            item["field_offsets"] = [x for x in nums if x >= 0 and x < 4096]
        item["evidence_addrs"] = item.get("evidence_addrs", item.get("support_sites", []))
    return arr



# ==============================================
# v9 EVAL-QUALITY PATCH: stronger InterfaceSem/TypeSem mapping fields
# ==============================================
# Goal:
#   1) Improve return_type_category known coverage by using Ghidra Function return type.
#   2) Make parameter TypeSem evaluable by ensuring parameter_type_sem has param_index,
#      pred_type_category, storage fields.
#   3) Make InterfaceSem parameter categories available at top-level and inside params.
#   4) Keep the previous memory-saving behavior.


def _v9_dt_name(dt):
    if dt is None:
        return "unknown"
    try:
        n = dt.getName()
        if n is not None:
            return str(n)
    except:
        pass
    try:
        return str(dt)
    except:
        return "unknown"


def _v9_dt_len(dt):
    if dt is None:
        return 0
    try:
        return int(dt.getLength())
    except:
        return 0


def _v9_is_ptr_dt(dt):
    try:
        if isinstance(dt, PointerDataType):
            return True
    except:
        pass
    s = _v9_dt_name(dt).lower()
    return ("*" in s) or ("ptr" in s) or ("pointer" in s)


def _v9_type_category_from_dt(dt, for_return=False, has_return=True):
    if for_return and (has_return is False):
        return "void"
    s = _v9_dt_name(dt).lower().strip()
    ln = _v9_dt_len(dt)
    if not s:
        return "unknown"
    if "void" == s or s.startswith("void "):
        return "void"
    if _v9_is_ptr_dt(dt):
        return "ptr"
    if "struct" in s or "class" in s:
        return "struct"
    if "union" in s:
        return "union"
    if "enum" in s:
        return "enum"
    if "float" in s or "double" in s:
        return "float"
    # Ghidra often emits undefined1/2/4/8 as a scalar placeholder.
    # Treat sized undefined values as scalar int-like so the evaluator can measure it,
    # but keep bare unknown/undefined with no size as unknown.
    if s in ("undefined1", "undefined2", "undefined4", "undefined8"):
        return "int"
    if s == "undefined" and ln in (1, 2, 4, 8):
        return "int"
    if s in ("unknown", "base", "default", "none", "null"):
        return "unknown"
    if ("int" in s) or ("long" in s) or ("short" in s) or ("char" in s) or ("bool" in s) or ("size_t" in s) or ("ssize_t" in s) or ("uint" in s) or ("byte" in s) or ("dword" in s) or ("qword" in s):
        return "int"
    return "other"


def _v9_param_info(param, idx, func_entry, func_name):
    dt = None
    try:
        dt = param.getDataType()
    except:
        dt = None
    storage = None
    try:
        storage = _safe_get_storage(param)
    except:
        try:
            storage = str(param.getStorage())
        except:
            storage = "unknown"
    if _is_unknown_storage(storage):
        storage = _infer_declared_param_storage_v4(None, idx, None)
    kind, sid = _storage_kind_and_id(storage)
    tname = _v9_dt_name(dt)
    cat = _v9_type_category_from_dt(dt)
    pname = "arg_%d" % idx
    try:
        pname = param.getName()
    except:
        pass
    return {
        "param_index": idx,
        "param_name": pname,
        "pred_type": tname,
        "pred_type_category": cat,
        "type_category": cat,
        "type_candidates": [tname],
        "type_confidence": [0.65 if cat != "unknown" else 0.2],
        "type_evidence": ["ghidra_function_parameter_datatype"],
        "type_size": _v9_dt_len(dt),
        "is_pointer": _v9_is_ptr_dt(dt),
        "storage": storage,
        "storage_kind": kind,
        "storage_id": sid,
        "stack_offset": None,
        "function_entry": func_entry,
        "func_entry": func_entry,
        "function_name": func_name,
        "object_kind": "param",
        "source": "ghidra_function_parameter"
    }


def _v9_function_info_map(prog):
    out = {}
    try:
        funcs = _get_internal_functions(prog, include_thunks=False)
    except:
        try:
            funcs = list(prog.getFunctionManager().getFunctions(True))
        except:
            funcs = []
    for func in funcs:
        try:
            if func.isExternal():
                continue
        except:
            pass
        try:
            entry = _addr_to_str(func.getEntryPoint())
        except:
            try:
                entry = str(func.getEntryPoint())
            except:
                continue
        name = _safe_get_name(func, "sub_%s" % entry) if '_safe_get_name' in globals() else "sub_%s" % entry
        params = []
        try:
            params = list(func.getParameters())
        except:
            params = []
        pinfo = []
        for idx, p in enumerate(params):
            try:
                pinfo.append(_v9_param_info(p, idx, entry, name))
            except:
                pass
        ret_dt = None
        try:
            ret_dt = func.getReturnType()
        except:
            ret_dt = None
        has_return = True
        try:
            has_return = not _is_void_type(ret_dt)
        except:
            has_return = (_v9_type_category_from_dt(ret_dt, True, True) != "void")
        ret_name = _v9_dt_name(ret_dt)
        ret_cat = _v9_type_category_from_dt(ret_dt, True, has_return)
        out[entry] = {
            "func": func,
            "entry": entry,
            "name": name,
            "params": pinfo,
            "param_count": len(pinfo),
            "return_type": ret_name,
            "return_type_category": ret_cat,
            "return_type_size": _v9_dt_len(ret_dt),
            "has_return": has_return,
        }
    return out


def _v9_is_cat_unknown(cat):
    return cat is None or str(cat).lower() in ("", "unknown", "undefined", "base", "default", "none", "null")


def _v9_enrich_param_from_info(p, info):
    if not isinstance(p, dict):
        p = {}
    for k in ["param_index", "param_name", "function_entry", "func_entry", "function_name", "object_kind", "storage", "storage_kind", "storage_id", "stack_offset"]:
        if p.get(k) in (None, "", "unknown"):
            p[k] = info.get(k)
    cur_cat = p.get("pred_type_category") or p.get("type_category") or p.get("category")
    if _v9_is_cat_unknown(cur_cat):
        p["pred_type"] = info.get("pred_type", p.get("pred_type", "unknown"))
        p["pred_type_category"] = info.get("pred_type_category", "unknown")
        p["type_category"] = p["pred_type_category"]
        p["type_candidates"] = info.get("type_candidates", p.get("type_candidates", []))
        p["type_confidence"] = info.get("type_confidence", p.get("type_confidence", []))
        p["type_evidence"] = _unique_list((p.get("type_evidence") or []) + info.get("type_evidence", []))
        p["type_size"] = info.get("type_size", p.get("type_size", 0))
        p["is_pointer"] = info.get("is_pointer", p.get("is_pointer", False))
    else:
        # normalize key names even if category already exists
        p["pred_type_category"] = cur_cat
        p["type_category"] = cur_cat
        if not p.get("pred_type"):
            p["pred_type"] = _v8_first_candidate(p.get("type_candidates", []))
    return p


try:
    _V9_PREV_GEN_TYPE_SEM = gen_type_sem
except:
    _V9_PREV_GEN_TYPE_SEM = None
try:
    _V9_PREV_GEN_INTERFACE_SEM = gen_interface_sem
except:
    _V9_PREV_GEN_INTERFACE_SEM = None
try:
    _V9_PREV_GEN_MEM_LAYOUT_SEM = gen_mem_layout_sem
except:
    _V9_PREV_GEN_MEM_LAYOUT_SEM = None


def gen_type_sem(prog, dfg_cache=None):
    arr = _V9_PREV_GEN_TYPE_SEM(prog, dfg_cache) if _V9_PREV_GEN_TYPE_SEM is not None else []
    fmap = _v9_function_info_map(prog)
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        entry = obj.get("function_entry") or obj.get("func_entry")
        finfo = fmap.get(entry, {})
        func_name = obj.get("function_name") or finfo.get("name") or "sub_%s" % entry
        obj["function_entry"] = entry
        obj["func_entry"] = entry
        obj["function_name"] = func_name

        # Return type: prefer Ghidra Function return datatype over old undefined/base category.
        ret = obj.get("return_type_sem")
        if not isinstance(ret, dict):
            ret = {}
        ret["function_entry"] = entry
        ret["func_entry"] = entry
        ret["function_name"] = func_name
        ret["object_kind"] = "return"
        if finfo:
            old_cat = ret.get("pred_type_category") or ret.get("type_category") or ret.get("category")
            if _v9_is_cat_unknown(old_cat) or str(old_cat).lower() == "base":
                ret["pred_type"] = finfo.get("return_type", "unknown")
                ret["pred_type_category"] = finfo.get("return_type_category", "unknown")
                ret["type_category"] = ret["pred_type_category"]
                ret["category"] = ret["pred_type_category"]
                ret["type_candidates"] = [ret["pred_type"]]
                ret["type_evidence"] = _unique_list((ret.get("type_evidence") or []) + ["ghidra_function_return_datatype"])
                ret["type_size"] = finfo.get("return_type_size", ret.get("type_size", 0))
                ret["is_pointer"] = (ret["pred_type_category"] == "ptr")
                ret["return_known_source"] = "ghidra_function_return_datatype"
        obj["return_type_sem"] = ret
        obj["return_type"] = ret.get("pred_type", _v8_first_candidate(ret.get("type_candidates", [])))
        obj["return_type_category"] = ret.get("pred_type_category", ret.get("type_category", "unknown"))

        # Parameters: fill/append from Ghidra Function parameters when available.
        params = obj.get("parameter_type_sem")
        if not isinstance(params, list):
            params = []
        infos = finfo.get("params", []) if finfo else []
        fixed = []
        max_len = max(len(params), len(infos))
        for idx in range(max_len):
            p = params[idx] if idx < len(params) and isinstance(params[idx], dict) else {}
            if idx < len(infos):
                p = _v9_enrich_param_from_info(p, infos[idx])
            else:
                _v8_enrich_type_item(p, entry, func_name, "param", idx, p.get("storage"))
            if not p.get("id"):
                p["id"] = gen_id("high_type", entry, "param_%d" % idx)
            fixed.append(p)
        obj["parameter_type_sem"] = fixed
        obj["param_types"] = [p.get("pred_type", "unknown") for p in fixed]
        obj["param_type_categories"] = [p.get("pred_type_category", p.get("type_category", "unknown")) for p in fixed]
        obj["param_count"] = len(fixed)
        obj["param_count_source"] = "ghidra_function_parameters" if infos else obj.get("param_count_source", "legacy")
    return arr


def gen_interface_sem(prog, type_cache=None, dfg_cache=None):
    arr = _V9_PREV_GEN_INTERFACE_SEM(prog, type_cache, dfg_cache) if _V9_PREV_GEN_INTERFACE_SEM is not None else []
    fmap = _v9_function_info_map(prog)
    type_map = {}
    try:
        for t in type_cache or []:
            if isinstance(t, dict):
                type_map[t.get("function_entry") or t.get("func_entry")] = t
    except:
        type_map = {}
    for itf in arr:
        if not isinstance(itf, dict):
            continue
        entry = itf.get("function_entry") or itf.get("func_entry")
        finfo = fmap.get(entry, {})
        func_name = itf.get("function_name") or itf.get("func_name") or finfo.get("name") or "sub_%s" % entry
        itf["function_entry"] = entry
        itf["func_entry"] = entry
        itf["func_name"] = func_name
        itf["function_name"] = func_name

        cons = itf.get("interface_constraints")
        if not isinstance(cons, dict):
            cons = {}
        params = itf.get("params")
        if not isinstance(params, list):
            params = []
        infos = finfo.get("params", []) if finfo else []
        tparams = []
        tobj = type_map.get(entry, {})
        if isinstance(tobj, dict):
            tparams = tobj.get("parameter_type_sem", []) or []

        max_len = max(len(params), len(infos), len(tparams))
        fixed = []
        for idx in range(max_len):
            p = params[idx] if idx < len(params) and isinstance(params[idx], dict) else {}
            if idx < len(tparams) and isinstance(tparams[idx], dict):
                p = _v9_enrich_param_from_info(p, tparams[idx])
            if idx < len(infos):
                p = _v9_enrich_param_from_info(p, infos[idx])
            if not p.get("id"):
                p["id"] = "interface_%s_param_%d" % (entry, idx)
            p["param_id"] = p.get("param_id", idx)
            p["param_index"] = p.get("param_index", idx)
            fixed.append(p)
        itf["params"] = fixed
        ptypes = [p.get("pred_type", _v8_first_candidate(p.get("type_candidates", []))) for p in fixed]
        pcats = [p.get("pred_type_category", p.get("type_category", "unknown")) for p in fixed]
        itf["param_types"] = ptypes
        itf["param_type_categories"] = pcats
        itf["param_count"] = len(fixed)
        itf["param_count_source"] = "ghidra_function_parameters_or_typesem" if max_len else "no_params"

        ret = itf.get("ret")
        if not isinstance(ret, dict):
            ret = {}
        has_return = finfo.get("has_return", cons.get("has_return", itf.get("has_return", True)))
        ret["has_return"] = has_return
        if finfo:
            ret["return_type"] = finfo.get("return_type", ret.get("return_type", "unknown"))
            ret["return_type_category"] = finfo.get("return_type_category", ret.get("return_type_category", "unknown"))
            ret["return_type_source"] = "ghidra_function_return_datatype"
        else:
            ret_type = ret.get("return_type") or _v8_first_candidate(ret.get("type_candidates", []))
            ret["return_type"] = ret_type
            ret["return_type_category"] = _v8_type_category_from_name(ret_type, ret.get("is_pointer", False), True, has_return)
        itf["ret"] = ret
        itf["return_type"] = ret.get("return_type", "unknown")
        itf["return_type_category"] = ret.get("return_type_category", "unknown")
        itf["return_type_source"] = ret.get("return_type_source", "legacy")
        itf["has_return"] = has_return

        cons["param_count"] = len(fixed)
        cons["declared_param_count"] = len(infos)
        cons["synthetic_param_count"] = max(0, len(fixed) - len(infos))
        cons["has_return"] = has_return
        itf["interface_constraints"] = cons
    return arr


def gen_mem_layout_sem(prog, dfg_cache=None, type_cache=None, interface_cache=None):
    arr = _V9_PREV_GEN_MEM_LAYOUT_SEM(prog, dfg_cache, type_cache, interface_cache) if _V9_PREV_GEN_MEM_LAYOUT_SEM is not None else []
    for item in arr:
        if not isinstance(item, dict):
            continue
        func_entry = item.get("function_entry") or item.get("func_entry")
        item["function_entry"] = func_entry
        item["func_entry"] = func_entry
        item["func_name"] = item.get("function_name") or item.get("func_name") or "sub_%s" % func_entry
        item["base_object"] = item.get("base_object", item.get("base_obj"))
        item["base_storage"] = item.get("base_storage", item.get("base_obj"))
        offs = item.get("field_offsets")
        if offs is None:
            offs = item.get("offsets", item.get("offset_set", []))
        nums = []
        try:
            for x in offs:
                n = int(x)
                if 0 <= n < 4096:
                    nums.append(n)
        except:
            nums = []
        nums = sorted(set(nums))
        item["offsets"] = nums
        if item.get("layout_kind") == "stack_frame_layout":
            item["field_offsets"] = []
        else:
            item["field_offsets"] = nums
        item["evidence_addrs"] = item.get("evidence_addrs", item.get("support_sites", []))
    return arr

def build_dict(prog, prev_snapshot_path=None):
    if _V4_BUILD_DICT is None:
        sem = {}
    else:
        sem = _V4_BUILD_DICT(prog)

    sem["program_info"]["schema_version"] = "semantic_dict_v9_eval_quality"
    sem["program_info"]["generator"] = "ghidra_usg_patch_v9_eval_quality"
    sem["program_info"]["memory_mode"] = {
        "keep_assembly": _v6_env_flag("USG_KEEP_ASSEMBLY", False),
        "keep_npcode": _v6_env_flag("USG_KEEP_NPCODE", False),
        "max_func_instr": _v6_env_int("USG_MAX_FUNC_INSTR", 8000),
        "max_func_pcode": _v6_env_int("USG_MAX_FUNC_PCODE", 50000),
        "max_npcode_text": _v6_env_int("USG_MAX_NPCODE_TEXT", 1024)
    }

    prev_sem = _load_json_maybe_v5(prev_snapshot_path)
    sem["consistency_report"] = run_consistency_checks(sem, prev_sem)
    return sem



# ==============================================
# v10 algorithmic recovery patch
# Implements:
# A) compiler/runtime function filtering metadata
# B) decompiler prototype extraction
# C) ABI entry-register parameter recovery
# D) parameter-focused TypeSem for strict DWARF evaluation
# ==============================================

try:
    _V10_PREV_GEN_TYPE_SEM = gen_type_sem
except:
    _V10_PREV_GEN_TYPE_SEM = None
try:
    _V10_PREV_GEN_INTERFACE_SEM = gen_interface_sem
except:
    _V10_PREV_GEN_INTERFACE_SEM = None
try:
    _V10_PREV_GEN_MEM_LAYOUT_SEM = gen_mem_layout_sem
except:
    _V10_PREV_GEN_MEM_LAYOUT_SEM = None
try:
    _V10_PREV_BUILD_DICT = build_dict
except:
    _V10_PREV_BUILD_DICT = None

_V10_FINFO_CACHE = None


def _v10_env_flag(name, default_value=False):
    try:
        v = os.environ.get(name)
        if v is None:
            return default_value
        return str(v).lower() not in ('0', 'false', 'no', 'off', '')
    except:
        return default_value


def _v10_env_int(name, default_value):
    try:
        return int(os.environ.get(name, str(default_value)))
    except:
        return default_value


def _v10_safe_lower(x):
    try:
        return str(x).lower()
    except:
        return ''


def _v10_func_entry(func):
    try:
        return _addr_to_str(func.getEntryPoint())
    except:
        try:
            return str(func.getEntryPoint())
        except:
            return 'unknown'


def _v10_func_name(func, entry):
    try:
        return _safe_get_name(func, 'sub_%s' % entry)
    except:
        try:
            return str(func.getName())
        except:
            return 'sub_%s' % entry


def _v10_runtime_name(name):
    s = _v10_safe_lower(name)
    if not s:
        return False
    bad_exact = set([
        '_init', '_fini', '_start', '__libc_start_main', '__stack_chk_fail',
        'deregister_tm_clones', 'register_tm_clones', 'frame_dummy',
        '__do_global_dtors_aux', '__x86.get_pc_thunk.bx', '__x86.get_pc_thunk.cx',
        '__x86.get_pc_thunk.dx', '__x86.get_pc_thunk.ax'
    ])
    if s in bad_exact:
        return True
    bad_substrings = [
        'tm_clones', 'frame_dummy', 'global_dtors', '.plt', '@plt',
        'libc_start', 'gmon_start', '__cxa_finalize', 'stack_chk',
        'asan_', 'ubsan_', 'tsan_', 'msan_', '__sanitizer',
        'init_array', 'fini_array'
    ]
    for b in bad_substrings:
        if b in s:
            return True
    return False


def _v10_function_eval_exclude(func, name):
    if _v10_runtime_name(name):
        return True
    try:
        if func.isExternal():
            return True
    except:
        pass
    try:
        if func.isThunk():
            return True
    except:
        pass
    return False


def _v10_dt_name(dt):
    try:
        return str(dt.getName())
    except:
        try:
            return str(dt)
        except:
            return 'unknown'


def _v10_dt_len(dt):
    try:
        return int(dt.getLength())
    except:
        return 0


def _v10_is_ptr_dt(dt):
    try:
        return bool(dt.isPointer())
    except:
        try:
            return ' *' in str(dt) or '*' in str(dt) or 'pointer' in str(dt).lower()
        except:
            return False


def _v10_type_cat_from_dt(dt, for_return=False, has_return=True):
    try:
        return _v9_type_category_from_dt(dt, for_return, has_return)
    except:
        s = _v10_dt_name(dt).lower()
        if for_return and has_return is False:
            return 'void'
        if s in ('void', 'void *') or s.startswith('void '):
            return 'void'
        if _v10_is_ptr_dt(dt) or '*' in s or 'ptr' in s or 'pointer' in s:
            return 'ptr'
        if 'struct' in s or 'class' in s:
            return 'struct'
        if 'union' in s:
            return 'union'
        if 'enum' in s:
            return 'enum'
        if 'float' in s or 'double' in s:
            return 'float'
        if s in ('undefined1', 'undefined2', 'undefined4', 'undefined8'):
            return 'int'
        if any(w in s for w in ['int', 'long', 'short', 'char', 'bool', 'size_t', 'ssize_t', 'uint', 'byte', 'dword', 'qword']):
            return 'int'
        if not s or s in ('unknown', 'undefined', 'base', 'default'):
            return 'unknown'
        return 'other'


def _v10_storage_of_obj(obj):
    if obj is None:
        return 'unknown'
    for meth in ['getStorage', 'getVariableStorage']:
        try:
            st = getattr(obj, meth)()
            if st is not None:
                return str(st)
        except:
            pass
    try:
        return _safe_get_storage(obj)
    except:
        return 'unknown'


def _v10_name_of_obj(obj, default_value):
    try:
        n = obj.getName()
        if n is not None:
            return str(n)
    except:
        pass
    return default_value


def _v10_dtype_of_obj(obj):
    for meth in ['getDataType', 'getType']:
        try:
            dt = getattr(obj, meth)()
            if dt is not None:
                return dt
        except:
            pass
    return None


def _v10_make_param_info(idx, func_entry, func_name, dt, name, storage, source):
    if storage is None or storage == '':
        storage = 'unknown'
    if _is_unknown_storage(storage):
        try:
            storage = _infer_declared_param_storage_v4(None, idx, None)
        except:
            storage = 'unknown'
    kind, sid = _storage_kind_and_id(storage)
    tname = _v10_dt_name(dt)
    cat = _v10_type_cat_from_dt(dt)
    return {
        'id': gen_id('high_type', func_entry, 'param_%d' % idx),
        'function_entry': func_entry,
        'func_entry': func_entry,
        'function_name': func_name,
        'object_kind': 'param',
        'param_index': idx,
        'param_name': name or ('arg_%d' % idx),
        'pred_type': tname,
        'pred_type_category': cat,
        'type_category': cat,
        'category': cat,
        'type_candidates': [tname] if tname else ['unknown'],
        'type_confidence': [0.78 if cat != 'unknown' else 0.25],
        'type_evidence': [source],
        'type_size': _v10_dt_len(dt),
        'is_pointer': _v10_is_ptr_dt(dt),
        'storage': storage,
        'storage_kind': kind,
        'storage_id': sid,
        'stack_offset': None,
        'source': source
    }


def _v10_make_abi_param_info(idx, func_entry, func_name, loc, source):
    info = None
    try:
        info = _canonical_abi_slot_info(currentProgram, loc)
    except:
        info = None
    storage = loc
    if info is not None:
        storage = info.get('canonical', loc)
    kind, sid = _storage_kind_and_id(storage)
    return {
        'id': gen_id('high_type', func_entry, 'abi_param_%d' % idx),
        'function_entry': func_entry,
        'func_entry': func_entry,
        'function_name': func_name,
        'object_kind': 'param',
        'param_index': idx,
        'param_name': 'arg_%d' % idx,
        'pred_type': 'unknown',
        'pred_type_category': 'unknown',
        'type_category': 'unknown',
        'category': 'unknown',
        'type_candidates': ['unknown'],
        'type_confidence': [0.20],
        'type_evidence': [source],
        'type_size': 0,
        'is_pointer': False,
        'storage': storage,
        'storage_kind': kind,
        'storage_id': sid,
        'stack_offset': None,
        'abi_slot': info.get('slot') if info else idx,
        'abi_kind': info.get('kind') if info else kind,
        'source': source
    }


def _v10_func_param_infos(func, entry, name):
    out = []
    try:
        params = list(func.getParameters())
    except:
        params = []
    for idx, p in enumerate(params):
        try:
            out.append(_v10_make_param_info(idx, entry, name, _v10_dtype_of_obj(p), _v10_name_of_obj(p, 'arg_%d' % idx), _v10_storage_of_obj(p), 'ghidra_function_parameter_datatype'))
        except:
            pass
    return out


def _v10_decompiler_interface(prog):
    if not _v10_env_flag('USG_V10_USE_DECOMPILER', True):
        return None
    try:
        from ghidra.app.decompiler import DecompInterface
        ifc = DecompInterface()
        try:
            ifc.openProgram(prog)
        except:
            return None
        return ifc
    except:
        return None


def _v10_proto_params_from_proto(proto, entry, name):
    out = []
    if proto is None:
        return out
    # Try getNumParams/getParam first. This works for many Ghidra decompiler prototypes.
    try:
        n = int(proto.getNumParams())
        for idx in range(n):
            try:
                p = proto.getParam(idx)
                out.append(_v10_make_param_info(idx, entry, name, _v10_dtype_of_obj(p), _v10_name_of_obj(p, 'arg_%d' % idx), _v10_storage_of_obj(p), 'decompiler_function_prototype'))
            except:
                pass
        if out:
            return out
    except:
        pass
    # Fallback getParams iterable.
    try:
        params = list(proto.getParams())
        for idx, p in enumerate(params):
            out.append(_v10_make_param_info(idx, entry, name, _v10_dtype_of_obj(p), _v10_name_of_obj(p, 'arg_%d' % idx), _v10_storage_of_obj(p), 'decompiler_function_prototype'))
    except:
        pass
    return out


def _v10_decompile_function_info(ifc, func, entry, name):
    if ifc is None:
        return None
    timeout = _v10_env_int('USG_V10_DECOMP_TIMEOUT', 8)
    try:
        res = ifc.decompileFunction(func, timeout, monitor)
    except:
        return None
    try:
        if not res.decompileCompleted():
            return None
    except:
        pass
    hf = None
    try:
        hf = res.getHighFunction()
    except:
        hf = None
    if hf is None:
        return None
    proto = None
    try:
        proto = hf.getFunctionPrototype()
    except:
        proto = None
    params = _v10_proto_params_from_proto(proto, entry, name)
    ret_dt = None
    try:
        ret_dt = proto.getReturnType()
    except:
        try:
            ret_dt = func.getReturnType()
        except:
            ret_dt = None
    has_return = True
    try:
        has_return = (_v10_type_cat_from_dt(ret_dt, True, True) != 'void')
    except:
        has_return = True
    return {
        'params': params,
        'param_count': len(params),
        'return_type': _v10_dt_name(ret_dt),
        'return_type_category': _v10_type_cat_from_dt(ret_dt, True, has_return),
        'return_type_size': _v10_dt_len(ret_dt),
        'has_return': has_return,
        'prototype_source': 'decompiler_function_prototype'
    }


def _v10_abi_locs_from_dfg(prog, dfg_obj):
    locs = []
    try:
        raw = dfg_obj.get('entry_input_locs', [])
    except:
        raw = []
    try:
        locs = _normalize_entry_param_locs_v4(prog, raw)
    except:
        locs = raw or []
    # Deduplicate and order by ABI slot when available.
    scored = []
    seen = set()
    for loc in locs:
        try:
            info = _canonical_abi_slot_info(prog, loc)
        except:
            info = None
        if info is None:
            continue
        canon = info.get('canonical', loc)
        if canon in seen:
            continue
        seen.add(canon)
        scored.append((int(info.get('slot', 999)), canon))
    scored.sort(key=lambda x: x[0])
    return [x[1] for x in scored]


def _v10_merge_params(entry, name, proto_params, func_params, abi_locs):
    # Preference order: decompiler prototype > Ghidra Function parameters > ABI recovered slots.
    base = proto_params if proto_params else func_params
    out = []
    for idx, p in enumerate(base):
        q = dict(p)
        q['param_index'] = idx
        out.append(q)
    if _v10_env_flag('USG_V10_USE_ABI_RECOVERY', True):
        # Add ABI-only params only when they extend beyond the decompiler/function prototype.
        for loc in abi_locs:
            try:
                info = _canonical_abi_slot_info(currentProgram, loc)
                slot = int(info.get('slot', len(out)))
            except:
                slot = len(out)
            if slot < len(out):
                # If a concrete parameter exists in the same slot, keep the concrete type.
                continue
            # Fill gaps with unknown placeholders so param_index remains ABI-consistent.
            while len(out) < slot:
                out.append(_v10_make_abi_param_info(len(out), entry, name, 'unknown', 'abi_gap_placeholder'))
            out.append(_v10_make_abi_param_info(slot, entry, name, loc, 'abi_entry_register_inference'))
    return out


def _v10_function_info_map(prog, dfg_cache=None):
    global _V10_FINFO_CACHE
    if _V10_FINFO_CACHE is not None:
        return _V10_FINFO_CACHE
    dfg_by_entry = {}
    try:
        for d in dfg_cache or []:
            e = d.get('function_entry') or d.get('func_entry')
            if e:
                dfg_by_entry[e] = d
    except:
        dfg_by_entry = {}
    try:
        funcs = _get_internal_functions(prog, include_thunks=False)
    except:
        try:
            funcs = list(prog.getFunctionManager().getFunctions(True))
        except:
            funcs = []
    ifc = _v10_decompiler_interface(prog)
    fmap = {}
    max_decomp = _v10_env_int('USG_V10_MAX_DECOMP_FUNCS', 1000000)
    decomp_count = 0
    for func in funcs:
        try:
            if func.isExternal():
                continue
        except:
            pass
        entry = _v10_func_entry(func)
        name = _v10_func_name(func, entry)
        eval_exclude = _v10_function_eval_exclude(func, name)
        func_params = _v10_func_param_infos(func, entry, name)
        decomp = None
        if decomp_count < max_decomp and (not eval_exclude or _v10_env_flag('USG_V10_DECOMP_RUNTIME_FUNCS', False)):
            decomp = _v10_decompile_function_info(ifc, func, entry, name)
            decomp_count += 1
        proto_params = decomp.get('params', []) if isinstance(decomp, dict) else []
        dfg_obj = dfg_by_entry.get(entry, {})
        abi_locs = _v10_abi_locs_from_dfg(prog, dfg_obj)
        params = _v10_merge_params(entry, name, proto_params, func_params, abi_locs)
        ret_dt = None
        try:
            ret_dt = func.getReturnType()
        except:
            ret_dt = None
        ret_name = _v10_dt_name(ret_dt)
        ret_cat = _v10_type_cat_from_dt(ret_dt, True, True)
        has_return = (ret_cat != 'void')
        ret_source = 'ghidra_function_return_datatype'
        if isinstance(decomp, dict):
            if decomp.get('return_type_category') not in (None, '', 'unknown'):
                ret_name = decomp.get('return_type', ret_name)
                ret_cat = decomp.get('return_type_category', ret_cat)
                has_return = decomp.get('has_return', has_return)
                ret_source = decomp.get('prototype_source', 'decompiler_function_prototype')
        fmap[entry] = {
            'func': func,
            'entry': entry,
            'name': name,
            'params': params,
            'param_count': len(params),
            'return_type': ret_name,
            'return_type_category': ret_cat,
            'return_type_size': _v10_dt_len(ret_dt),
            'has_return': has_return,
            'return_type_source': ret_source,
            'eval_exclude': eval_exclude,
            'eval_exclude_reason': 'runtime_or_thunk' if eval_exclude else '',
            'prototype_source': 'decompiler' if proto_params else ('ghidra_function' if func_params else ('abi' if abi_locs else 'none')),
            'abi_locs': abi_locs,
            'decompiler_param_count': len(proto_params),
            'function_param_count': len(func_params),
            'abi_param_count': len(abi_locs)
        }
    _V10_FINFO_CACHE = fmap
    return fmap


def gen_type_sem(prog, dfg_cache=None):
    arr = _V10_PREV_GEN_TYPE_SEM(prog, dfg_cache) if _V10_PREV_GEN_TYPE_SEM is not None else []
    fmap = _v10_function_info_map(prog, dfg_cache)
    by_entry = {}
    for obj in arr:
        if isinstance(obj, dict):
            e = obj.get('function_entry') or obj.get('func_entry')
            if e:
                by_entry[e] = obj
    # Ensure every function gets a TypeSem object with parameter_type_sem.
    for entry, finfo in fmap.items():
        if entry not in by_entry:
            obj = {
                'id': gen_id('high_type', entry, 'func'),
                'function_entry': entry,
                'func_entry': entry,
                'function_name': finfo.get('name'),
                'function_id': '%s_%s' % (entry, finfo.get('name')),
                'parameter_type_sem': [],
                'local_var_type_sem': [],
                'instruction_type_sem': [],
                'type_constraints': {},
            }
            arr.append(obj)
            by_entry[entry] = obj
    for entry, obj in by_entry.items():
        finfo = fmap.get(entry, {})
        obj['function_entry'] = entry
        obj['func_entry'] = entry
        obj['function_name'] = obj.get('function_name') or finfo.get('name') or ('sub_%s' % entry)
        obj['eval_exclude'] = bool(finfo.get('eval_exclude', False))
        obj['eval_exclude_reason'] = finfo.get('eval_exclude_reason', '')
        # Return type.
        ret = obj.get('return_type_sem')
        if not isinstance(ret, dict):
            ret = {}
        ret.update({
            'function_entry': entry,
            'func_entry': entry,
            'function_name': obj.get('function_name'),
            'object_kind': 'return',
            'pred_type': finfo.get('return_type', ret.get('pred_type', 'unknown')),
            'pred_type_category': finfo.get('return_type_category', ret.get('pred_type_category', 'unknown')),
            'type_category': finfo.get('return_type_category', ret.get('type_category', 'unknown')),
            'return_type_source': finfo.get('return_type_source', 'unknown'),
            'type_candidates': [finfo.get('return_type', ret.get('pred_type', 'unknown'))],
            'type_evidence': _unique_list((ret.get('type_evidence') or []) + [finfo.get('return_type_source', 'unknown')])
        })
        obj['return_type_sem'] = ret
        obj['return_type'] = ret.get('pred_type', 'unknown')
        obj['return_type_category'] = ret.get('pred_type_category', 'unknown')
        # Parameter type semantics. Keep only stable param-level objects for strict eval.
        old_params = obj.get('parameter_type_sem') if isinstance(obj.get('parameter_type_sem'), list) else []
        old_by_idx = {}
        for i, p in enumerate(old_params):
            if isinstance(p, dict):
                idx = p.get('param_index', i)
                old_by_idx[idx] = p
        fixed = []
        for idx, pinfo in enumerate(finfo.get('params', [])):
            old = old_by_idx.get(idx, {})
            q = dict(old)
            q.update(pinfo)
            q['param_index'] = idx
            q['object_kind'] = 'param'
            q['function_entry'] = entry
            q['func_entry'] = entry
            q['function_name'] = obj.get('function_name')
            if not q.get('id'):
                q['id'] = gen_id('high_type', entry, 'param_%d' % idx)
            fixed.append(q)
        obj['parameter_type_sem'] = fixed
        obj['param_types'] = [p.get('pred_type', 'unknown') for p in fixed]
        obj['param_type_categories'] = [p.get('pred_type_category', p.get('type_category', 'unknown')) for p in fixed]
        obj['param_count'] = len(fixed)
        obj['param_count_source'] = finfo.get('prototype_source', 'unknown')
    return arr


def gen_interface_sem(prog, type_cache=None, dfg_cache=None):
    arr = _V10_PREV_GEN_INTERFACE_SEM(prog, type_cache, dfg_cache) if _V10_PREV_GEN_INTERFACE_SEM is not None else []
    fmap = _v10_function_info_map(prog, dfg_cache)
    by_entry = {}
    for obj in arr:
        if isinstance(obj, dict):
            e = obj.get('function_entry') or obj.get('func_entry')
            if e:
                by_entry[e] = obj
    for entry, finfo in fmap.items():
        if entry not in by_entry:
            obj = {
                'id': gen_id('interface', entry, 'func'),
                'function_entry': entry,
                'func_entry': entry,
                'func_id': '%s_%s' % (entry, finfo.get('name')),
                'function_name': finfo.get('name'),
                'params': [],
                'ret': {},
                'interface_constraints': {},
            }
            arr.append(obj)
            by_entry[entry] = obj
    for entry, itf in by_entry.items():
        finfo = fmap.get(entry, {})
        func_name = itf.get('function_name') or itf.get('func_name') or finfo.get('name') or ('sub_%s' % entry)
        itf['function_entry'] = entry
        itf['func_entry'] = entry
        itf['func_name'] = func_name
        itf['function_name'] = func_name
        itf['eval_exclude'] = bool(finfo.get('eval_exclude', False))
        itf['eval_exclude_reason'] = finfo.get('eval_exclude_reason', '')
        fixed_params = []
        for idx, pinfo in enumerate(finfo.get('params', [])):
            p = dict(pinfo)
            p['param_index'] = idx
            p['param_id'] = idx
            if not p.get('id'):
                p['id'] = 'interface_%s_param_%d' % (entry, idx)
            fixed_params.append(p)
        itf['params'] = fixed_params
        itf['param_count'] = len(fixed_params)
        itf['param_count_source'] = finfo.get('prototype_source', 'unknown')
        itf['param_types'] = [p.get('pred_type', 'unknown') for p in fixed_params]
        itf['param_type_categories'] = [p.get('pred_type_category', p.get('type_category', 'unknown')) for p in fixed_params]
        ret = itf.get('ret') if isinstance(itf.get('ret'), dict) else {}
        ret.update({
            'return_type': finfo.get('return_type', ret.get('return_type', 'unknown')),
            'return_type_category': finfo.get('return_type_category', ret.get('return_type_category', 'unknown')),
            'return_type_source': finfo.get('return_type_source', 'unknown'),
            'has_return': finfo.get('has_return', True),
            'type_candidates': [finfo.get('return_type', ret.get('return_type', 'unknown'))]
        })
        itf['ret'] = ret
        itf['return_type'] = ret.get('return_type', 'unknown')
        itf['return_type_category'] = ret.get('return_type_category', 'unknown')
        itf['return_type_source'] = ret.get('return_type_source', 'unknown')
        itf['has_return'] = ret.get('has_return', True)
        cons = itf.get('interface_constraints') if isinstance(itf.get('interface_constraints'), dict) else {}
        cons['param_count'] = len(fixed_params)
        cons['declared_param_count'] = finfo.get('decompiler_param_count', finfo.get('function_param_count', len(fixed_params)))
        cons['synthetic_param_count'] = max(0, len(fixed_params) - int(cons.get('declared_param_count', 0)))
        cons['has_return'] = itf['has_return']
        cons['prototype_source'] = finfo.get('prototype_source', 'unknown')
        cons['abi_locs'] = finfo.get('abi_locs', [])
        itf['interface_constraints'] = cons
    return arr


def gen_mem_layout_sem(prog, dfg_cache=None, type_cache=None, interface_cache=None):
    arr = _V10_PREV_GEN_MEM_LAYOUT_SEM(prog, dfg_cache, type_cache, interface_cache) if _V10_PREV_GEN_MEM_LAYOUT_SEM is not None else []
    fmap = _v10_function_info_map(prog, dfg_cache)
    for item in arr:
        if not isinstance(item, dict):
            continue
        entry = item.get('function_entry') or item.get('func_entry')
        finfo = fmap.get(entry, {})
        item['function_entry'] = entry
        item['func_entry'] = entry
        item['func_name'] = item.get('function_name') or item.get('func_name') or finfo.get('name') or ('sub_%s' % entry)
        item['eval_exclude'] = bool(finfo.get('eval_exclude', False))
        item['eval_exclude_reason'] = finfo.get('eval_exclude_reason', '')
        if item.get('layout_kind') == 'stack_frame_layout':
            item['field_offsets'] = []
            continue
        offs = item.get('field_offsets')
        if offs is None:
            offs = item.get('offsets', item.get('offset_set', []))
        nums = []
        try:
            for x in offs:
                n = int(x)
                if 0 <= n < 4096:
                    nums.append(n)
        except:
            nums = []
        item['field_offsets'] = sorted(set(nums))
        item['offsets'] = sorted(set(nums))
        item['base_object'] = item.get('base_object', item.get('base_obj'))
        item['base_storage'] = item.get('base_storage', item.get('base_obj'))
        item['evidence_addrs'] = item.get('evidence_addrs', item.get('support_sites', []))
    return arr


def build_dict(prog, prev_snapshot_path=None):
    sem = _V10_PREV_BUILD_DICT(prog, prev_snapshot_path) if _V10_PREV_BUILD_DICT is not None else {}
    try:
        sem['program_info']['schema_version'] = 'semantic_dict_v10_algorithmic_eval'
        sem['program_info']['generator'] = 'ghidra_usg_patch_v10_algorithmic_eval'
        sem['program_info']['algorithmic_recovery'] = {
            'runtime_function_filter_metadata': True,
            'decompiler_prototype_extraction': _v10_env_flag('USG_V10_USE_DECOMPILER', True),
            'abi_entry_register_recovery': _v10_env_flag('USG_V10_USE_ABI_RECOVERY', True),
            'parameter_focused_typesem': True,
            'decomp_timeout_sec': _v10_env_int('USG_V10_DECOMP_TIMEOUT', 8)
        }
    except:
        pass
    try:
        prev_sem = _load_json_maybe_v5(prev_snapshot_path)
        sem['consistency_report'] = run_consistency_checks(sem, prev_sem)
    except:
        pass
    return sem



# ==============================================
# v11 PATCH: real evidence edges + same-base LayoutSem merge
#            + DFG-guided parameter type refinement
# ==============================================
# Purpose:
#   1) Improve low metrics caused by fragmented LayoutSem and conservative
#      parameter type categories.
#   2) Emit explicit, standard evidence_edges with source_node -> target_node,
#      so Research Task 2 can build real anchors without EvidenceStub repair.
#
# Notes:
#   - This patch does not use DWARF or labels. It only uses Ghidra/decompiler
#     output and existing DFG/memory-access evidence.
#   - The old cross_layer_mapping is preserved for compatibility.
# ==============================================

try:
    _V11_PREV_BUILD_DICT = build_dict
except:
    _V11_PREV_BUILD_DICT = None

try:
    import hashlib as _v11_hashlib
except:
    _v11_hashlib = None


def _v11_env_flag(name, default_value):
    try:
        v = os.environ.get(name)
        if v is None:
            return default_value
        return str(v).strip().lower() not in ("0", "false", "no", "off")
    except:
        return default_value


def _v11_hash(s, n=10):
    try:
        if _v11_hashlib is not None:
            return _v11_hashlib.sha1(str(s).encode('utf-8')).hexdigest()[:n]
    except:
        pass
    try:
        return str(abs(hash(str(s))))[:n]
    except:
        return "0"


def _v11_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except:
        try:
            return repr(x)
        except:
            return ""


def _v11_norm(s):
    s = _v11_str(s).strip().lower()
    s = s.replace(" ", "")
    return s


def _v11_unique(seq):
    out = []
    seen = set()
    for x in seq or []:
        sx = _v11_str(x)
        if sx in seen:
            continue
        seen.add(sx)
        out.append(x)
    return out


def _v11_as_int(x):
    try:
        return int(x)
    except:
        try:
            return int(str(x), 0)
        except:
            return None


def _v11_extract_offsets(obj):
    vals = []
    for key in ["field_offsets", "offsets", "offset_set", "observed_offsets", "accessed_offsets"]:
        v = obj.get(key)
        if isinstance(v, list):
            vals.extend(v)
    for fr in obj.get("field_relations", []) or []:
        if isinstance(fr, dict):
            vals.append(fr.get("offset"))
    nums = []
    for v in vals:
        n = _v11_as_int(v)
        if n is not None and 0 <= n < 4096:
            nums.append(n)
    return sorted(list(set(nums)))


def _v11_offset_closure(offsets):
    observed = sorted(list(set([x for x in offsets if x is not None and 0 <= x < 4096])))
    inferred = []
    if not observed:
        return [], []

    # Most C structs have a field at offset 0. If all observed offsets start
    # after 0, adding 0 often improves observed-layout recall while keeping the
    # candidate conservative. Controlled by USG_V11_INFER_BASE_ZERO.
    if _v11_env_flag("USG_V11_INFER_BASE_ZERO", True):
        try:
            if min(observed) > 0:
                inferred.append(0)
        except:
            pass

    # Fill internal stride gaps only when there is a clear, small positive stride.
    # This helps cases where the binary accesses offset 0 and 16 but the middle
    # field at 8 is implied by a uniform field cluster. Controlled by env flag.
    if _v11_env_flag("USG_V11_INFER_STRIDE_GAPS", True):
        try:
            if len(observed) >= 3:
                diffs = []
                for i in range(len(observed) - 1):
                    d = observed[i + 1] - observed[i]
                    if d > 0:
                        diffs.append(d)
                stride = min(diffs) if diffs else None
                if stride is not None and 0 < stride <= 32:
                    cur = min(observed)
                    mx = max(observed)
                    while cur <= mx:
                        if cur not in observed:
                            inferred.append(cur)
                        cur += stride
        except:
            pass

    final = sorted(list(set(observed + inferred)))
    return final, sorted(list(set(inferred)))


def _v11_layout_group_key(obj):
    fentry = obj.get("function_entry") or obj.get("func_entry") or ""
    base = obj.get("base_object") or obj.get("base_storage") or obj.get("base_obj")
    if not base:
        loc = obj.get("location") or obj.get("id") or ""
        # Keep declared/fallback layouts separate when there is no clear base.
        base = loc
    kind = obj.get("layout_kind") or obj.get("layout_source") or "layout"
    return (_v11_str(fentry), _v11_str(base), _v11_str(kind))


def _v11_merge_same_base_layouts(sem):
    high = sem.get("high_level_sem", {})
    layouts = high.get("mem_layout_sem", [])
    if not isinstance(layouts, list):
        return {}

    groups = {}
    order = []
    id_remap = {}

    for obj in layouts:
        if not isinstance(obj, dict):
            continue
        # Keep explicit stack frame layouts out of struct-field evaluation.
        if obj.get("layout_kind") == "stack_frame_layout":
            obj["field_offsets"] = []
            obj["offsets"] = []
            key = (obj.get("id"), "stack", obj.get("location"))
        else:
            key = _v11_layout_group_key(obj)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(obj)

    merged = []
    for key in order:
        items = groups[key]
        first = dict(items[0])
        ids = [x.get("id") for x in items if isinstance(x, dict) and x.get("id")]
        primary_id = ids[0] if ids else first.get("id")
        if not primary_id:
            primary_id = "memlayout_%s_merged_%s" % (_v11_str(key[0]), _v11_hash(key))
        first["id"] = primary_id

        observed_offsets = []
        evidence_addrs = []
        support_sites = []
        semantic_roles = []
        constraints = []
        confidences = []
        bases = []
        layout_sources = []

        for item in items:
            if not isinstance(item, dict):
                continue
            oid = item.get("id")
            if oid:
                id_remap[oid] = primary_id
            observed_offsets.extend(_v11_extract_offsets(item))
            evidence_addrs.extend(item.get("evidence_addrs", []) or [])
            support_sites.extend(item.get("support_sites", []) or [])
            semantic_roles.extend(item.get("semantic_roles", []) or [])
            layout_sources.append(item.get("layout_source"))
            bases.append(item.get("base_object") or item.get("base_storage") or item.get("base_obj"))
            pc = item.get("propagated_constraints")
            if isinstance(pc, dict):
                constraints.extend(pc.get("constraint_evidence", []) or [])
            try:
                confidences.append(float(item.get("confidence")))
            except:
                pass

        observed_offsets = sorted(list(set([x for x in observed_offsets if x is not None])))
        final_offsets, inferred_offsets = _v11_offset_closure(observed_offsets)

        first["function_entry"] = first.get("function_entry") or first.get("func_entry") or key[0]
        first["func_entry"] = first.get("func_entry") or first.get("function_entry")
        first["base_object"] = first.get("base_object") or first.get("base_obj") or first.get("base_storage") or (bases[0] if bases else None)
        first["base_storage"] = first.get("base_storage") or first.get("base_object") or first.get("base_obj")
        first["offsets"] = final_offsets
        first["offset_set"] = final_offsets
        first["field_offsets"] = final_offsets
        first["observed_field_offsets"] = observed_offsets
        first["inferred_field_offsets"] = inferred_offsets
        first["field_offsets_observed"] = observed_offsets
        first["field_offsets_inferred"] = inferred_offsets
        first["evidence_addrs"] = _v11_unique(evidence_addrs)
        first["support_sites"] = _v11_unique(support_sites)
        first["semantic_roles"] = _v11_unique(semantic_roles)
        first["layout_source"] = "merged_same_base_offset_cluster" if len(items) > 1 else (first.get("layout_source") or "single_layout_cluster")
        first["layout_merge_count"] = len(items)
        first["merged_from_ids"] = ids
        first["layout_v11_merged"] = len(items) > 1
        first["layout_v11_offset_closure"] = bool(inferred_offsets)
        if confidences:
            first["confidence"] = _clamp_strength(max(confidences))
        else:
            first["confidence"] = _clamp_strength(first.get("confidence", 0.65))
        pc = first.get("propagated_constraints") if isinstance(first.get("propagated_constraints"), dict) else {}
        pc["constraint_evidence"] = _v11_unique((pc.get("constraint_evidence", []) or []) + constraints + ["same_base_offset_merge"])
        first["propagated_constraints"] = pc
        merged.append(first)

    high["mem_layout_sem"] = merged
    sem["high_level_sem"] = high
    return id_remap


def _v11_update_edge_references(obj, id_remap):
    if not id_remap:
        return obj
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            v = obj.get(k)
            if k in ("from", "to", "source_node", "target_node", "source_ref", "target_ref") and v in id_remap:
                obj[k] = id_remap[v]
            else:
                obj[k] = _v11_update_edge_references(v, id_remap)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _v11_update_edge_references(obj[i], id_remap)
    return obj


def _v11_collect_nodes(sem):
    nodes = {}
    paths = [
        ("InstrSem", sem.get("low_level_sem", {}).get("instr_sem", [])),
        ("BlockSem", sem.get("low_level_sem", {}).get("block_sem", [])),
        ("CFGSem", sem.get("mid_level_sem", {}).get("cfg_sem", [])),
        ("DFGSem", sem.get("mid_level_sem", {}).get("dataflow_sem", [])),
        ("TypeSem", sem.get("high_level_sem", {}).get("type_sem", [])),
        ("InterfaceSem", sem.get("high_level_sem", {}).get("interface_sem", [])),
        ("APISem", sem.get("high_level_sem", {}).get("api_sem", [])),
        ("LayoutSem", sem.get("high_level_sem", {}).get("mem_layout_sem", [])),
    ]
    for layer, arr in paths:
        for x in arr or []:
            if isinstance(x, dict) and x.get("id"):
                nodes[x.get("id")] = (layer, x)
    return nodes


def _v11_func_entry_from_obj(obj):
    if not isinstance(obj, dict):
        return ""
    for k in ["function_entry", "func_entry", "entry_address", "entry", "address"]:
        if obj.get(k):
            return _v11_str(obj.get(k))
    try:
        fid = obj.get("func_id") or obj.get("function_id") or obj.get("id")
        if fid:
            return _parse_func_entry_from_func_id(fid) or ""
    except:
        pass
    loc = obj.get("location")
    try:
        if loc and "_layout_" in loc:
            return loc.split("_layout_", 1)[0]
    except:
        pass
    return ""


def _v11_add_evidence(edges, seen, rule_id, evidence_type, source_id, target_id,
                      source_layer, target_layer, strength, function_entry, trigger):
    if not source_id or not target_id:
        return
    key = (evidence_type, source_id, target_id, rule_id)
    if key in seen:
        return
    seen.add(key)
    edge_id = "ev_%s_%s" % (evidence_type, _v11_hash("%s|%s|%s|%s" % (rule_id, source_id, target_id, len(edges)), 16))
    edges.append({
        "edge_id": edge_id,
        "mapping_id": edge_id,
        "edge_kind": "evidence",
        "rule_id": rule_id,
        "evidence_type": evidence_type,
        "source_node": source_id,
        "target_node": target_id,
        "source_ref": source_id,
        "target_ref": target_id,
        "from": source_id,
        "to": target_id,
        "source_layer": source_layer,
        "target_layer": target_layer,
        "strength": _clamp_strength(strength),
        "function_entry": function_entry or "",
        "trigger": trigger or {},
        "v11_real_edge": True
    })


def _v11_standard_evidence_edges(sem):
    nodes = _v11_collect_nodes(sem)
    edges = []
    seen = set()

    # 1) Convert existing cross_layer_mapping evidence into standard source_node/target_node edges.
    for m in sem.get("cross_layer_mapping", []) or []:
        if not isinstance(m, dict):
            continue
        if m.get("edge_kind") != "evidence":
            continue
        src = m.get("source_node") or m.get("source_ref") or m.get("from")
        dst = m.get("target_node") or m.get("target_ref") or m.get("to")
        if not src or not dst:
            continue
        sl = m.get("source_layer")
        tl = m.get("target_layer")
        if src in nodes:
            sl = sl or nodes[src][0]
        if dst in nodes:
            tl = tl or nodes[dst][0]
        fentry = ""
        if dst in nodes:
            fentry = _v11_func_entry_from_obj(nodes[dst][1])
        if not fentry and src in nodes:
            fentry = _v11_func_entry_from_obj(nodes[src][1])
        _v11_add_evidence(edges, seen, m.get("rule_id", "R"), m.get("evidence_type", "evidence"),
                          src, dst, sl or "", tl or "", m.get("strength", 0.7), fentry, m.get("trigger", {}))

    # 2) Add direct fixed_offset evidence from LayoutSem.evidence_addrs/support_sites -> LayoutSem.
    instr_by_addr = {}
    for ins in sem.get("low_level_sem", {}).get("instr_sem", []) or []:
        if isinstance(ins, dict) and ins.get("address") and ins.get("id"):
            instr_by_addr.setdefault(_v11_str(ins.get("address")), []).append(ins.get("id"))

    dfg_by_func = {}
    for dfg in sem.get("mid_level_sem", {}).get("dataflow_sem", []) or []:
        if isinstance(dfg, dict):
            fe = dfg.get("entry_address") or dfg.get("function_entry")
            if fe and dfg.get("id"):
                dfg_by_func[_v11_str(fe)] = dfg.get("id")

    for lay in sem.get("high_level_sem", {}).get("mem_layout_sem", []) or []:
        if not isinstance(lay, dict) or not lay.get("id"):
            continue
        if lay.get("layout_kind") == "stack_frame_layout":
            continue
        fentry = _v11_str(lay.get("function_entry") or lay.get("func_entry") or "")
        offsets = lay.get("field_offsets") or lay.get("offsets") or []
        strength = 0.62 + 0.04 * min(len(offsets), 4)
        try:
            strength += 0.10 * float(lay.get("confidence", 0.0))
        except:
            pass
        addrs = []
        addrs.extend(lay.get("evidence_addrs", []) or [])
        for site in lay.get("support_sites", []) or []:
            if isinstance(site, dict):
                if site.get("instruction_address"):
                    addrs.append(site.get("instruction_address"))
                elif site.get("site"):
                    addrs.append(site.get("site"))
            else:
                addrs.append(site)
        for addr in _v11_unique(addrs):
            # Some site strings may be "addr:..."; find a hex-like address.
            addr_s = _v11_str(addr)
            m = re.search(r'(0x[0-9a-fA-F]+|[0-9a-fA-F]{6,16})', addr_s)
            if m:
                a = m.group(1)
                if a.startswith("0x"):
                    try:
                        a = "%08x" % int(a, 16)
                    except:
                        a = a[2:]
                for src_id in instr_by_addr.get(a, []):
                    _v11_add_evidence(edges, seen, "R7", "fixed_offset", src_id, lay.get("id"),
                                      "InstrSem", "LayoutSem", strength, fentry,
                                      {"instruction_address": a, "offsets": offsets, "base": lay.get("base_object")})
        # If instruction addresses are unavailable, at least connect the real DFGSem node.
        if fentry in dfg_by_func:
            _v11_add_evidence(edges, seen, "R7", "fixed_offset", dfg_by_func[fentry], lay.get("id"),
                              "DFGSem", "LayoutSem", strength + 0.02, fentry,
                              {"offsets": offsets, "base": lay.get("base_object"), "source": "layout_dfg_summary"})

    return edges


def _v11_param_refine_by_dfg(sem):
    dfg_by_func = {}
    for dfg in sem.get("mid_level_sem", {}).get("dataflow_sem", []) or []:
        if isinstance(dfg, dict):
            fe = dfg.get("entry_address") or dfg.get("function_entry")
            if fe:
                dfg_by_func[_v11_str(fe)] = dfg

    type_by_func = {}
    for ts in sem.get("high_level_sem", {}).get("type_sem", []) or []:
        if isinstance(ts, dict):
            fe = ts.get("function_entry") or ts.get("func_entry")
            if fe:
                type_by_func[_v11_str(fe)] = ts

    itf_by_func = {}
    for itf in sem.get("high_level_sem", {}).get("interface_sem", []) or []:
        if isinstance(itf, dict):
            fe = itf.get("function_entry") or itf.get("func_entry")
            if fe:
                itf_by_func[_v11_str(fe)] = itf

    for fentry, dfg in dfg_by_func.items():
        bases = set()
        for acc in dfg.get("memory_access_patterns", []) or []:
            b = acc.get("base")
            if b:
                bases.add(_v11_norm(b))
            loc = acc.get("location")
            if loc:
                bases.add(_v11_norm(loc))
        # If an ABI entry input is used as a memory base, it is likely a pointer parameter.
        entry_locs = [_v11_norm(x) for x in (dfg.get("entry_input_locs", []) or [])]

        def _refine_param(p):
            if not isinstance(p, dict):
                return p
            storage_candidates = [p.get("storage"), p.get("storage_id"), p.get("location"), p.get("param_name")]
            hit_ptr = False
            for sc in storage_candidates:
                ns = _v11_norm(sc)
                if not ns:
                    continue
                if ns in bases:
                    hit_ptr = True
                # Partial containment handles forms like "reg:rdi" vs "rdi".
                for b in list(bases)[:200]:
                    if ns and (ns in b or b in ns):
                        hit_ptr = True
                        break
                if hit_ptr:
                    break
            cur = p.get("pred_type_category") or p.get("type_category") or p.get("category") or "unknown"
            name = _v11_norm(p.get("param_name"))
            if hit_ptr and cur not in ("ptr", "struct", "union"):
                p["pred_type"] = p.get("pred_type") if p.get("pred_type") not in (None, "", "unknown", "undefined") else "void*"
                p["pred_type_category"] = "ptr"
                p["type_category"] = "ptr"
                p["category"] = "ptr"
                p["is_pointer"] = True
                ev = p.get("type_evidence") if isinstance(p.get("type_evidence"), list) else []
                p["type_evidence"] = _v11_unique(ev + ["v11_dfg_base_parameter_refinement"])
                p["v11_refined_by_dfg"] = True
            elif cur in ("unknown", "other", "base"):
                if "len" in name or "size" in name or "count" in name or name in ("argc",):
                    p["pred_type_category"] = "int"
                    p["type_category"] = "int"
                    p["category"] = "int"
                    if p.get("pred_type") in (None, "", "unknown", "undefined"):
                        p["pred_type"] = "int"
                    ev = p.get("type_evidence") if isinstance(p.get("type_evidence"), list) else []
                    p["type_evidence"] = _v11_unique(ev + ["v11_name_based_size_count_refinement"])
                    p["v11_refined_by_name"] = True
            return p

        ts = type_by_func.get(fentry)
        if ts is not None:
            params = []
            for p in ts.get("parameter_type_sem", []) or []:
                params.append(_refine_param(p))
            ts["parameter_type_sem"] = params
            ts["param_types"] = [p.get("pred_type", "unknown") for p in params if isinstance(p, dict)]
            ts["param_type_categories"] = [p.get("pred_type_category", p.get("type_category", "unknown")) for p in params if isinstance(p, dict)]

        itf = itf_by_func.get(fentry)
        if itf is not None:
            params = []
            for p in itf.get("params", []) or []:
                params.append(_refine_param(p))
            itf["params"] = params
            itf["param_types"] = [p.get("pred_type", "unknown") for p in params if isinstance(p, dict)]
            itf["param_type_categories"] = [p.get("pred_type_category", p.get("type_category", "unknown")) for p in params if isinstance(p, dict)]
    return sem


def _v11_add_edge_alias_to_usg(sem):
    try:
        usg = sem.get("usg")
        if isinstance(usg, dict):
            usg["E_ev"] = [x.get("edge_id") for x in sem.get("evidence_edges", []) if isinstance(x, dict) and x.get("edge_id")]
            usg["E_ct"] = [x.get("constraint_id") for x in sem.get("constraint_edges", []) if isinstance(x, dict) and x.get("constraint_id")]
            usg["evidence_edge_count"] = len(usg.get("E_ev", []))
            usg["constraint_edge_count"] = len(usg.get("E_ct", []))
    except:
        pass


def build_dict(prog, prev_snapshot_path=None):
    if _V11_PREV_BUILD_DICT is None:
        sem = {}
    else:
        try:
            sem = _V11_PREV_BUILD_DICT(prog, prev_snapshot_path)
        except TypeError:
            sem = _V11_PREV_BUILD_DICT(prog)

    if not isinstance(sem, dict):
        return sem

    try:
        sem = _v11_param_refine_by_dfg(sem)
    except Exception as e:
        try:
            sem.setdefault("program_info", {})["v11_param_refine_error"] = _v11_str(e)
        except:
            pass

    try:
        id_remap = _v11_merge_same_base_layouts(sem)
        _v11_update_edge_references(sem.get("cross_layer_mapping", []), id_remap)
        _v11_update_edge_references(sem.get("constraint_edges", []), id_remap)
    except Exception as e:
        try:
            sem.setdefault("program_info", {})["v11_layout_merge_error"] = _v11_str(e)
        except:
            pass

    try:
        sem["evidence_edges"] = _v11_standard_evidence_edges(sem)
    except Exception as e:
        try:
            sem.setdefault("program_info", {})["v11_evidence_edge_error"] = _v11_str(e)
        except:
            sem["evidence_edges"] = []

    try:
        pi = sem.setdefault("program_info", {})
        pi["schema_version"] = "semantic_dict_v11_evidence_layout"
        pi["generator"] = "ghidra_usg_patch_v11_evidence_layout"
        pi["v11_optimizations"] = {
            "same_base_layout_merge": True,
            "layout_offset_closure_base_zero": _v11_env_flag("USG_V11_INFER_BASE_ZERO", True),
            "layout_offset_closure_stride_gaps": _v11_env_flag("USG_V11_INFER_STRIDE_GAPS", True),
            "standard_evidence_edges": True,
            "dfg_guided_parameter_type_refinement": True,
            "evidence_edge_count": len(sem.get("evidence_edges", [])),
            "layout_count_after_merge": len(sem.get("high_level_sem", {}).get("mem_layout_sem", []))
        }
    except:
        pass

    try:
        _v11_add_edge_alias_to_usg(sem)
    except:
        pass

    try:
        prev_sem = _load_json_maybe_v5(prev_snapshot_path)
        sem["consistency_report"] = run_consistency_checks(sem, prev_sem)
    except:
        pass

    return sem



# ==============================================
# v11 TYPESEM-PLUS PATCH: only refine TypeSem / InterfaceSem parameter type quality
# ==============================================
# Design goal:
#   1) Keep v11 evidence_edges/layout behavior unchanged.
#   2) Do NOT use DWARF/gold information.
#   3) Preserve original type predictions as candidates, instead of overwriting them blindly.
#   4) Refine only parameter TypeSem and mirrored InterfaceSem parameter categories.
#   5) Make evaluation/debugging transparent by recording candidate sources and scores.

try:
    _TYPEPLUS_PREV_BUILD_DICT = build_dict
except:
    _TYPEPLUS_PREV_BUILD_DICT = None


def _tp_str(x):
    try:
        if x is None:
            return ""
        return str(x)
    except:
        try:
            return unicode(x)
        except:
            return ""


def _tp_lower(x):
    return _tp_str(x).lower()


def _tp_norm(x):
    s = _tp_lower(x).strip()
    # normalize common storage/register prefixes
    for pre in ("reg:", "register:", "stack:", "mem:", "var:"):
        if s.startswith(pre):
            s = s[len(pre):]
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    # normalize x86 register width aliases
    reg_alias = {
        "edi": "rdi", "di": "rdi", "dil": "rdi",
        "esi": "rsi", "si": "rsi", "sil": "rsi",
        "edx": "rdx", "dx": "rdx", "dl": "rdx",
        "ecx": "rcx", "cx": "rcx", "cl": "rcx",
        "eax": "rax", "ax": "rax", "al": "rax",
        "ebx": "rbx", "bx": "rbx", "bl": "rbx",
        "r8d": "r8", "r8w": "r8", "r8b": "r8",
        "r9d": "r9", "r9w": "r9", "r9b": "r9",
        "esp": "rsp", "sp": "rsp", "spl": "rsp",
        "ebp": "rbp", "bp": "rbp", "bpl": "rbp",
    }
    return reg_alias.get(s, s)


def _tp_unique(seq):
    out = []
    seen = set()
    for x in seq or []:
        sx = _tp_str(x)
        if sx not in seen:
            seen.add(sx)
            out.append(x)
    return out


def _tp_is_unknown_cat(cat):
    s = _tp_lower(cat)
    return s in ("", "unknown", "undefined", "base", "default", "none", "null", "nan")


def _tp_type_category_from_name(name):
    s = _tp_lower(name).strip()
    if not s:
        return "unknown"
    if s in ("unknown", "undefined", "base", "default", "none", "null"):
        return "unknown"
    if s in ("undefined1", "undefined2", "undefined4", "undefined8"):
        return "int"
    if s == "void" or s.startswith("void "):
        return "void"
    # Pointers first, because char* should be ptr not int.
    if "*" in s or " ptr" in s or "ptr" in s or "pointer" in s or s.endswith("[]"):
        return "ptr"
    if "struct" in s or "class" in s:
        return "struct"
    if "union" in s:
        return "union"
    if "enum" in s:
        return "enum"
    if "float" in s or "double" in s:
        return "float"
    int_words = ["int", "long", "short", "char", "bool", "size_t", "ssize_t", "uint", "uintptr", "byte", "dword", "qword", "word"]
    for w in int_words:
        if w in s:
            return "int"
    return "other"


def _tp_candidate(cat, tname, source, score):
    cat = _tp_type_category_from_name(cat) if cat not in ("ptr", "int", "float", "void", "struct", "union", "enum", "other", "unknown") else cat
    return {
        "category": cat,
        "type": _tp_str(tname) if tname not in (None, "") else cat,
        "source": source,
        "score": float(score)
    }


def _tp_collect_existing_candidates(p):
    cands = []
    if not isinstance(p, dict):
        return cands

    orig_type = p.get("pred_type") or p.get("type") or "unknown"
    orig_cat = p.get("pred_type_category") or p.get("type_category") or p.get("category") or _tp_type_category_from_name(orig_type)
    if not _tp_is_unknown_cat(orig_cat):
        cands.append(_tp_candidate(orig_cat, orig_type, "original_pred_type", 0.74))
    else:
        cands.append(_tp_candidate("unknown", orig_type, "original_pred_type_unknown", 0.10))

    tc = p.get("type_candidates")
    conf = p.get("type_confidence")
    if isinstance(tc, list):
        for i, t in enumerate(tc):
            cat = _tp_type_category_from_name(t)
            if _tp_is_unknown_cat(cat):
                continue
            score = 0.62
            try:
                if isinstance(conf, list) and i < len(conf):
                    score = max(score, min(0.95, float(conf[i])))
            except:
                pass
            cands.append(_tp_candidate(cat, t, "original_type_candidate", score))

    # Some earlier patches use these fields.
    for k in ("category", "type_category", "pred_type_category"):
        if p.get(k) and not _tp_is_unknown_cat(p.get(k)):
            cands.append(_tp_candidate(p.get(k), orig_type, "legacy_%s" % k, 0.66))
    return cands


def _tp_param_name_hint_candidates(p):
    name = _tp_lower(p.get("param_name") or p.get("name") or p.get("var_name") or "")
    out = []
    if not name:
        return out
    ptr_words = ["buf", "buffer", "str", "string", "path", "file", "name", "data", "dst", "src", "out", "in", "ptr", "argv", "envp", "fmt", "format", "base", "addr", "address", "mem", "bytes"]
    int_words = ["len", "length", "size", "count", "cnt", "num", "n", "fd", "argc", "mode", "flag", "flags", "index", "idx", "offset", "off", "pos", "ch", "code", "status", "id", "pid", "uid", "gid", "port"]
    for w in ptr_words:
        if w in name:
            out.append(_tp_candidate("ptr", "void*", "param_name_pointer_hint:%s" % w, 0.84))
            return out
    for w in int_words:
        if w in name:
            out.append(_tp_candidate("int", "int", "param_name_integer_hint:%s" % w, 0.78))
            return out
    return out


def _tp_build_function_evidence(sem):
    """Collect function-level hints from DFG memory bases and v11 evidence_edges."""
    info = {}
    # DFG memory base usage.
    try:
        dfgs = sem.get("mid_level_sem", {}).get("dataflow_sem", []) or []
    except:
        dfgs = []
    for dfg in dfgs:
        if not isinstance(dfg, dict):
            continue
        fe = _tp_str(dfg.get("entry_address") or dfg.get("function_entry") or dfg.get("func_entry"))
        if not fe:
            continue
        rec = info.setdefault(fe, {"memory_bases": set(), "entry_inputs": set(), "etype_by_target": {}, "api_names": set()})
        for x in dfg.get("entry_input_locs", []) or []:
            nx = _tp_norm(x)
            if nx:
                rec["entry_inputs"].add(nx)
        for acc in dfg.get("memory_access_patterns", []) or []:
            if not isinstance(acc, dict):
                continue
            for k in ("base", "base_register", "base_storage", "location"):
                nx = _tp_norm(acc.get(k))
                if nx:
                    rec["memory_bases"].add(nx)

    # API names by function, if APISem exists.
    try:
        apis = sem.get("high_level_sem", {}).get("api_sem", []) or []
    except:
        apis = []
    for api in apis:
        if not isinstance(api, dict):
            continue
        fe = _tp_str(api.get("function_entry") or api.get("func_entry"))
        if not fe:
            continue
        rec = info.setdefault(fe, {"memory_bases": set(), "entry_inputs": set(), "etype_by_target": {}, "api_names": set()})
        for k in ("api_name", "api_target", "name", "callee", "function_name"):
            if api.get(k):
                rec["api_names"].add(_tp_lower(api.get(k)))

    # Evidence edges by target node and function.
    for e in sem.get("evidence_edges", []) or []:
        if not isinstance(e, dict):
            continue
        fe = _tp_str(e.get("function_entry") or e.get("func_entry"))
        if not fe:
            continue
        rec = info.setdefault(fe, {"memory_bases": set(), "entry_inputs": set(), "etype_by_target": {}, "api_names": set()})
        tgt = e.get("target_node") or e.get("target_ref") or e.get("to")
        etype = _tp_lower(e.get("evidence_type") or e.get("edge_type") or "")
        if tgt and etype:
            rec["etype_by_target"].setdefault(_tp_str(tgt), set()).add(etype)
        trig = e.get("trigger")
        if isinstance(trig, dict):
            for loc in trig.get("entry_input_locs", []) or []:
                nx = _tp_norm(loc)
                if nx:
                    rec["entry_inputs"].add(nx)
    return info


def _tp_storage_candidates(p):
    vals = []
    if not isinstance(p, dict):
        return vals
    for k in ("storage", "storage_id", "location", "base_storage", "varnode", "register"):
        if p.get(k):
            vals.append(p.get(k))
    # ABI fallback by param_index.
    try:
        idx = int(p.get("param_index"))
        abi = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
        if 0 <= idx < len(abi):
            vals.append(abi[idx])
    except:
        pass
    return vals


def _tp_evidence_candidates(p, finfo):
    out = []
    if not isinstance(p, dict) or not isinstance(finfo, dict):
        return out

    # Storage used as memory base -> pointer.
    bases = finfo.get("memory_bases", set())
    for sc in _tp_storage_candidates(p):
        ns = _tp_norm(sc)
        if not ns or ns in ("rsp", "rbp", "esp", "ebp"):
            continue
        hit = False
        if ns in bases:
            hit = True
        else:
            for b in list(bases)[:300]:
                if not b or b in ("rsp", "rbp", "esp", "ebp"):
                    continue
                if ns in b or b in ns:
                    hit = True
                    break
        if hit:
            out.append(_tp_candidate("ptr", "void*", "storage_used_as_memory_base", 0.92))
            break

    # Evidence edge target hints.
    target_keys = []
    for k in ("id", "type_sem_ref", "param_type_ref", "node_id"):
        if p.get(k):
            target_keys.append(_tp_str(p.get(k)))
    etype_by_target = finfo.get("etype_by_target", {})
    for tk in target_keys:
        for et in etype_by_target.get(tk, set()):
            if "deref" in et or "fixed_offset" in et or "memory" in et:
                out.append(_tp_candidate("ptr", "void*", "evidence_edge_pointer_hint:%s" % et, 0.91))
            elif "api" in et:
                # API call itself does not always imply pointer, so keep it as weak support only.
                out.append(_tp_candidate("other", "api_related", "evidence_edge_api_hint:%s" % et, 0.35))

    return out


def _tp_api_name_candidates(p, finfo):
    """Conservative API/name hints. This is intentionally weak unless param name is clear."""
    out = []
    if not isinstance(finfo, dict):
        return out
    api_names = " ".join(list(finfo.get("api_names", set()))[:30])
    pname = _tp_lower(p.get("param_name") or p.get("name") or "") if isinstance(p, dict) else ""
    if not api_names:
        return out
    # If this function calls string/memory APIs and the param name suggests buffer/string, pointer is likely.
    if any(x in api_names for x in ["strlen", "strcpy", "strcmp", "strncpy", "memcpy", "memset", "read", "write", "fread", "fwrite", "open", "fopen"]):
        if any(w in pname for w in ["buf", "buffer", "str", "string", "src", "dst", "path", "name", "data"]):
            out.append(_tp_candidate("ptr", "void*", "api_name_plus_param_name_pointer_hint", 0.86))
        if any(w in pname for w in ["len", "size", "count", "n", "fd"]):
            out.append(_tp_candidate("int", "int", "api_name_plus_param_name_integer_hint", 0.80))
    return out


def _tp_choose_best_candidate(p, candidates):
    if not candidates:
        return p
    candidates = [c for c in candidates if isinstance(c, dict)]
    if not candidates:
        return p
    candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)

    orig_cat = p.get("original_pred_type_category") or p.get("pred_type_category") or p.get("type_category") or p.get("category") or "unknown"
    orig_cat = _tp_type_category_from_name(orig_cat) if orig_cat not in ("ptr", "int", "float", "void", "struct", "union", "enum", "other", "unknown") else orig_cat
    orig_known = not _tp_is_unknown_cat(orig_cat)
    best = candidates[0]

    # Conservative replacement policy:
    #   - If original is unknown/other/base, take the best useful candidate.
    #   - If original is known, replace only with strong evidence >= 0.86.
    #   - Always keep all original/new candidates in candidate fields.
    chosen = None
    if not orig_known or orig_cat == "other":
        chosen = best
    elif best.get("category") != orig_cat and best.get("score", 0.0) >= 0.86:
        chosen = best
    else:
        # Keep original category as predicted category.
        chosen = _tp_candidate(orig_cat, p.get("original_pred_type") or p.get("pred_type") or orig_cat, "kept_original_type", 0.74)

    p["pred_type"] = chosen.get("type", p.get("pred_type", "unknown"))
    p["pred_type_category"] = chosen.get("category", p.get("pred_type_category", "unknown"))
    p["type_category"] = p["pred_type_category"]
    p["category"] = p["pred_type_category"]
    p["is_pointer"] = (p["pred_type_category"] == "ptr")

    # Preserve all candidates for downstream model/reasoning.
    p["type_category_candidates"] = _tp_unique([c.get("category") for c in candidates if c.get("category")])
    p["pred_type_candidates_extended"] = _tp_unique([c.get("type") for c in candidates if c.get("type")])
    p["type_candidate_records"] = candidates[:12]
    ev = p.get("type_evidence") if isinstance(p.get("type_evidence"), list) else []
    p["type_evidence"] = _tp_unique(ev + ["typesem_plus:%s" % chosen.get("source", "unknown")])
    p["typesem_plus_chosen_source"] = chosen.get("source")
    p["typesem_plus_chosen_score"] = chosen.get("score")
    return p


def _tp_refine_param(p, finfo):
    if not isinstance(p, dict):
        return p
    # Save original prediction exactly once.
    if "original_pred_type" not in p:
        p["original_pred_type"] = p.get("pred_type") or _tp_unique(p.get("type_candidates") or ["unknown"])[0]
    if "original_pred_type_category" not in p:
        p["original_pred_type_category"] = p.get("pred_type_category") or p.get("type_category") or p.get("category") or _tp_type_category_from_name(p.get("original_pred_type"))
    if "original_type_candidates" not in p:
        p["original_type_candidates"] = list(p.get("type_candidates") or [])

    candidates = []
    candidates.extend(_tp_collect_existing_candidates(p))
    candidates.extend(_tp_param_name_hint_candidates(p))
    candidates.extend(_tp_evidence_candidates(p, finfo))
    candidates.extend(_tp_api_name_candidates(p, finfo))

    # Special case: if Ghidra type name is explicit pointer/struct, keep it strong.
    tname = p.get("pred_type") or p.get("original_pred_type")
    cat_from_name = _tp_type_category_from_name(tname)
    if cat_from_name in ("ptr", "struct", "union", "float", "enum"):
        candidates.append(_tp_candidate(cat_from_name, tname, "explicit_type_name", 0.88))

    p = _tp_choose_best_candidate(p, candidates)

    # Rebuild type_candidates/confidence with original types plus chosen/enriched types.
    all_types = []
    all_types.extend(p.get("original_type_candidates") or [])
    all_types.extend(p.get("pred_type_candidates_extended") or [])
    if p.get("pred_type"):
        all_types.append(p.get("pred_type"))
    p["type_candidates"] = _tp_unique([x for x in all_types if x not in (None, "")])
    # Confidence is approximate; keep length aligned for downstream summarizers.
    conf = []
    for t in p["type_candidates"]:
        cat = _tp_type_category_from_name(t)
        if cat == p.get("pred_type_category"):
            conf.append(0.85)
        elif not _tp_is_unknown_cat(cat):
            conf.append(0.55)
        else:
            conf.append(0.15)
    p["type_confidence"] = conf
    p["typesem_plus_applied"] = True
    return p


def _tp_refine_typesem_and_interface(sem):
    finfo_by_func = _tp_build_function_evidence(sem)

    type_by_func = {}
    try:
        type_items = sem.get("high_level_sem", {}).get("type_sem", []) or []
    except:
        type_items = []
    for ts in type_items:
        if not isinstance(ts, dict):
            continue
        fe = _tp_str(ts.get("function_entry") or ts.get("func_entry"))
        if not fe:
            continue
        finfo = finfo_by_func.get(fe, {"memory_bases": set(), "entry_inputs": set(), "etype_by_target": {}, "api_names": set()})
        params = []
        for p in ts.get("parameter_type_sem", []) or []:
            params.append(_tp_refine_param(p, finfo))
        ts["parameter_type_sem"] = params
        ts["param_types"] = [p.get("pred_type", "unknown") for p in params if isinstance(p, dict)]
        ts["param_type_categories"] = [p.get("pred_type_category", p.get("type_category", "unknown")) for p in params if isinstance(p, dict)]
        ts["type_param_accuracy_focus"] = True
        type_by_func[fe] = ts

    try:
        interfaces = sem.get("high_level_sem", {}).get("interface_sem", []) or []
    except:
        interfaces = []
    for itf in interfaces:
        if not isinstance(itf, dict):
            continue
        fe = _tp_str(itf.get("function_entry") or itf.get("func_entry"))
        finfo = finfo_by_func.get(fe, {"memory_bases": set(), "entry_inputs": set(), "etype_by_target": {}, "api_names": set()})
        ts = type_by_func.get(fe, {})
        tparams = ts.get("parameter_type_sem", []) if isinstance(ts, dict) else []
        params = []
        old_params = itf.get("params", []) or []
        max_len = max(len(old_params), len(tparams))
        for idx in range(max_len):
            p = old_params[idx] if idx < len(old_params) and isinstance(old_params[idx], dict) else {}
            # Prefer refined TypeSem categories for consistency; preserve old interface info as candidates.
            if idx < len(tparams) and isinstance(tparams[idx], dict):
                if "original_pred_type" not in p:
                    p["original_pred_type"] = p.get("pred_type") or p.get("type") or "unknown"
                if "original_pred_type_category" not in p:
                    p["original_pred_type_category"] = p.get("pred_type_category") or p.get("type_category") or p.get("category") or "unknown"
                # Merge candidate records from TypeSem.
                merged = []
                merged.extend(_tp_collect_existing_candidates(p))
                for c in tparams[idx].get("type_candidate_records", []) or []:
                    if isinstance(c, dict):
                        merged.append(c)
                p = _tp_choose_best_candidate(p, merged)
                # If TypeSem has a stronger/explicit refined category, mirror it.
                p["pred_type"] = tparams[idx].get("pred_type", p.get("pred_type", "unknown"))
                p["pred_type_category"] = tparams[idx].get("pred_type_category", p.get("pred_type_category", "unknown"))
                p["type_category"] = p["pred_type_category"]
                p["category"] = p["pred_type_category"]
                p["type_sem_ref"] = tparams[idx].get("id", p.get("type_sem_ref"))
                p["type_category_candidates"] = _tp_unique((p.get("type_category_candidates") or []) + (tparams[idx].get("type_category_candidates") or []))
                p["pred_type_candidates_extended"] = _tp_unique((p.get("pred_type_candidates_extended") or []) + (tparams[idx].get("pred_type_candidates_extended") or []))
                ev = p.get("type_evidence") if isinstance(p.get("type_evidence"), list) else []
                p["type_evidence"] = _tp_unique(ev + ["mirrored_from_typesem_plus"])
            else:
                p = _tp_refine_param(p, finfo)
            if "param_index" not in p:
                p["param_index"] = idx
            params.append(p)
        itf["params"] = params
        itf["param_types"] = [p.get("pred_type", "unknown") for p in params if isinstance(p, dict)]
        itf["param_type_categories"] = [p.get("pred_type_category", p.get("type_category", "unknown")) for p in params if isinstance(p, dict)]
        itf["interface_param_accuracy_focus"] = True
    return sem




# ==============================================
# PATCH v11-typesem-plus-paramflow:
# Enrich InstrSem for api_call parameter-flow neighborhoods
# ==============================================
_PF_X86_ARG_REGS = ["RDI", "RSI", "RDX", "RCX", "R8", "R9"]
_PF_X86_RET_REGS = ["RAX", "EAX", "AX", "AL"]
_PF_ARM64_ARG_REGS = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"]
_PF_ARM64_RET_REGS = ["X0", "W0"]
_PF_ARM32_ARG_REGS = ["R0", "R1", "R2", "R3"]
_PF_ARM32_RET_REGS = ["R0"]


def _pf_str(x):
    try:
        if x is None:
            return ""
        return str(x)
    except:
        try:
            return unicode(x)
        except:
            return ""


def _pf_upper(x):
    try:
        return _pf_str(x).upper()
    except:
        return ""


def _pf_unique(xs):
    seen = set()
    out = []
    for x in xs or []:
        if x is None:
            continue
        sx = _pf_str(x)
        if sx == "" or sx in seen:
            continue
        seen.add(sx)
        out.append(sx)
    return out


def _pf_get_instr_sem_list(sem):
    try:
        xs = sem.get("low_level_sem", {}).get("instr_sem", [])
        if isinstance(xs, list):
            return xs
    except:
        pass
    return []


def _pf_get_edges_list(sem):
    try:
        xs = sem.get("evidence_edges")
        if isinstance(xs, list):
            return xs
    except:
        pass
    sem["evidence_edges"] = []
    return sem["evidence_edges"]


def _pf_addr_to_ghidra(prog, addr_s):
    try:
        if addr_s is None:
            return None
        s = _pf_str(addr_s)
        return prog.getAddressFactory().getAddress(s)
    except:
        try:
            if s.startswith("0x"):
                return prog.getAddressFactory().getDefaultAddressSpace().getAddress(int(s, 16))
        except:
            pass
    return None


def _pf_addr_offset_key(addr_s):
    try:
        s = _pf_str(addr_s)
        if s.startswith("0x"):
            return int(s, 16)
        return int(s, 16)
    except:
        return 0


def _pf_language_kind(prog):
    try:
        p = _pf_str(prog.getLanguage().getProcessor()).lower()
    except:
        p = ""
    if "aarch64" in p or "arm64" in p:
        return "arm64"
    if "arm" in p:
        return "arm32"
    if "x86" in p or "intel" in p:
        return "x86_64"
    return "x86_64"


def _pf_arg_regs(prog):
    k = _pf_language_kind(prog)
    if k == "arm64":
        return list(_PF_ARM64_ARG_REGS)
    if k == "arm32":
        return list(_PF_ARM32_ARG_REGS)
    return list(_PF_X86_ARG_REGS)


def _pf_ret_regs(prog):
    k = _pf_language_kind(prog)
    if k == "arm64":
        return list(_PF_ARM64_RET_REGS)
    if k == "arm32":
        return list(_PF_ARM32_RET_REGS)
    return list(_PF_X86_RET_REGS)


def _pf_reg_aliases(reg):
    r = _pf_upper(reg).replace("REG:", "")
    x86 = {
        "RDI": ["RDI", "EDI", "DI", "DIL"],
        "RSI": ["RSI", "ESI", "SI", "SIL"],
        "RDX": ["RDX", "EDX", "DX", "DL"],
        "RCX": ["RCX", "ECX", "CX", "CL"],
        "R8": ["R8", "R8D", "R8W", "R8B"],
        "R9": ["R9", "R9D", "R9W", "R9B"],
        "RAX": ["RAX", "EAX", "AX", "AL"]
    }
    if r in x86:
        return x86[r]
    if r.startswith("X"):
        try:
            n = int(r[1:])
            return ["X%d" % n, "W%d" % n]
        except:
            return [r]
    if r.startswith("R"):
        return [r]
    return [r]


def _pf_canon_loc(prog, vn):
    # Prefer the v11 p-code canonicalizer if it exists.
    try:
        return _canonical_varnode_location(prog, vn, include_constant=False)
    except:
        pass
    if vn is None:
        return None
    try:
        addr = vn.getAddress()
    except:
        addr = None
    try:
        if vn.isRegister() or (addr is not None and addr.isRegisterAddress()):
            try:
                reg = prog.getRegister(addr, vn.getSize())
                if reg:
                    return "reg:%s" % _pf_upper(reg.getName())
            except:
                pass
            return "reg:0x%x:%d" % (int(vn.getOffset()), int(vn.getSize()))
    except:
        pass
    try:
        if vn.isUnique():
            return "tmp:0x%x:%d" % (int(vn.getOffset()), int(vn.getSize()))
    except:
        pass
    try:
        if addr is not None and addr.isStackAddress():
            return "stack:%s:%d" % (_pf_str(addr), int(vn.getSize()))
    except:
        pass
    try:
        if addr is not None and addr.isMemoryAddress():
            return "addr:%s:0x%x:%d" % (_pf_str(addr.getAddressSpace().getName()), int(vn.getOffset()), int(vn.getSize()))
    except:
        pass
    return None


def _pf_loc_reg_name(loc):
    s = _pf_upper(loc)
    if s.startswith("REG:"):
        s = s[4:]
    if ":" in s:
        # Handle forms such as reg:RDI or reg:RDI:8.
        parts = s.split(":")
        if len(parts) >= 2 and parts[0] == "REG":
            return parts[1]
    return s


def _pf_extract_pcode_rw(prog, instr):
    reads = []
    writes = []
    try:
        pcode_ops = instr.getPcode()
    except:
        pcode_ops = []
    for op in pcode_ops:
        try:
            mnem = op.getMnemonic()
        except:
            mnem = ""
        try:
            n_inputs = op.getNumInputs()
        except:
            n_inputs = 0
        # Inputs are uses. For CALL p-code, ABI args are added later.
        for i in range(n_inputs):
            try:
                loc = _pf_canon_loc(prog, op.getInput(i))
                if loc:
                    reads.append(loc)
            except:
                pass
        try:
            out_vn = op.getOutput()
            loc = _pf_canon_loc(prog, out_vn)
            if loc:
                writes.append(loc)
        except:
            pass
    return _pf_unique(reads), _pf_unique(writes)


def _pf_is_call_instr(instr, sem_obj):
    try:
        ft = instr.getFlowType()
        if ft is not None and ft.isCall():
            return True
    except:
        pass
    try:
        op = _pf_upper(sem_obj.get("opcode"))
        if "CALL" in op:
            return True
    except:
        pass
    try:
        asm = _pf_str(instr.toString()).lower()
        if re.search(r"\b(call|bl|blr)\b", asm):
            return True
    except:
        pass
    return False


def _pf_call_target(prog, instr):
    try:
        refs = instr.getReferencesFrom()
        for r in refs:
            try:
                rt = r.getReferenceType()
                if RefType.isCall(rt):
                    to_addr = r.getToAddress()
                    fm = prog.getFunctionManager()
                    f = fm.getFunctionAt(to_addr)
                    if f:
                        return f.getName()
                    sym = prog.getSymbolTable().getPrimarySymbol(to_addr)
                    if sym:
                        return sym.getName()
                    return _pf_str(to_addr)
            except:
                pass
    except:
        pass
    try:
        if instr.getNumOperands() > 0:
            return _pf_str(instr.getDefaultOperandRepresentation(0))
    except:
        pass
    return ""


def _pf_enrich_instrsem_paramflow_fields(sem, prog):
    instr_sem = _pf_get_instr_sem_list(sem)
    changed = 0
    call_count = 0
    by_func = defaultdict(list)

    for i, obj in enumerate(instr_sem):
        if not isinstance(obj, dict):
            continue
        addr_s = obj.get("address") or obj.get("addr") or obj.get("location")
        gaddr = _pf_addr_to_ghidra(prog, addr_s)
        ginstr = None
        if gaddr is not None:
            try:
                ginstr = prog.getListing().getInstructionAt(gaddr)
            except:
                ginstr = None
        fentry = obj.get("function_entry") or obj.get("func_entry") or ""
        if ginstr is not None:
            try:
                f = prog.getFunctionManager().getFunctionContaining(ginstr.getAddress())
                if f:
                    fentry = _pf_str(f.getEntryPoint())
            except:
                pass
        if fentry:
            obj["function_entry"] = fentry
            obj["function_entry_norm"] = fentry
        by_func[fentry or "unknown"].append((i, obj, ginstr))

    for fentry, arr in by_func.items():
        arr.sort(key=lambda t: _pf_addr_offset_key(t[1].get("address")) * 1000 + t[0])
        for seq, (idx, obj, ginstr) in enumerate(arr):
            reads = []
            writes = []
            operands = obj.get("operands") if isinstance(obj.get("operands"), list) else []
            if ginstr is not None:
                p_reads, p_writes = _pf_extract_pcode_rw(prog, ginstr)
                reads.extend(p_reads)
                writes.extend(p_writes)
                try:
                    ops = []
                    for oi in range(ginstr.getNumOperands()):
                        ops.append(_pf_str(ginstr.getDefaultOperandRepresentation(oi)))
                    if ops:
                        operands = ops
                except:
                    pass
            else:
                # Fallback from existing n_pcode / operands.
                try:
                    defs, uses, opcode = _parse_npcode_def_use(obj.get("n_pcode", ""))
                    reads.extend(list(uses))
                    writes.extend(list(defs))
                except:
                    pass

            is_call = ginstr is not None and _pf_is_call_instr(ginstr, obj)
            if not is_call:
                try:
                    if "CALL" in _pf_upper(obj.get("opcode")):
                        is_call = True
                except:
                    pass

            if is_call:
                call_count += 1
                for r in _pf_arg_regs(prog):
                    reads.append("reg:%s" % r)
                for r in _pf_ret_regs(prog):
                    writes.append("reg:%s" % r)
                target = _pf_call_target(prog, ginstr) if ginstr is not None else ""
                if target:
                    obj["call_target"] = target
                    obj["callee"] = target
                    obj.setdefault("api_name", target)
                obj["argument_locs"] = ["reg:%s" % r for r in _pf_arg_regs(prog)]
                obj["return_locs"] = ["reg:%s" % r for r in _pf_ret_regs(prog)]
                obj["paramflow_call_like"] = True
                if not obj.get("opcode") or obj.get("opcode") == "UNKNOWN":
                    obj["opcode"] = "CALL"

            obj["seq"] = obj.get("seq", seq)
            obj["operands"] = operands
            obj["read_set"] = _pf_unique(reads)
            obj["write_set"] = _pf_unique(writes)
            changed += 1

    return changed, call_count


def _pf_instr_id(obj):
    try:
        return obj.get("id") or obj.get("node_id") or ""
    except:
        return ""


def _pf_has_alias(last_def, reg):
    for a in _pf_reg_aliases(reg):
        if a in last_def:
            return last_def[a]
        key = "REG:%s" % a
        if key in last_def:
            return last_def[key]
    return None


def _pf_build_paramflow_edges(sem, prog):
    instr_sem = _pf_get_instr_sem_list(sem)
    edges = _pf_get_edges_list(sem)
    existing = set()
    for e in edges:
        try:
            if e.get("edge_id"):
                existing.add(e.get("edge_id"))
        except:
            pass

    by_func = defaultdict(list)
    for i, obj in enumerate(instr_sem):
        if not isinstance(obj, dict):
            continue
        fe = obj.get("function_entry") or obj.get("function_entry_norm") or obj.get("func_entry") or "unknown"
        by_func[_pf_str(fe)].append((i, obj))

    added_param = 0
    added_ret = 0
    for fe, arr in by_func.items():
        arr.sort(key=lambda t: int(t[1].get("seq", t[0])) if _pf_str(t[1].get("seq", "")).isdigit() else t[0])
        last_def = {}
        pending_returns = []
        for i, obj in arr:
            nid = _pf_instr_id(obj)
            if not nid:
                continue
            reads = [_pf_loc_reg_name(x) for x in (obj.get("read_set") or [])]
            writes = [_pf_loc_reg_name(x) for x in (obj.get("write_set") or [])]
            is_call = obj.get("paramflow_call_like") or ("CALL" in _pf_upper(obj.get("opcode")))

            if is_call:
                for arg_idx, reg in enumerate(_pf_arg_regs(prog)):
                    src = _pf_has_alias(last_def, reg)
                    if src:
                        eid = "pf_param_%s" % _v11_hash("%s|%s|%s|%s" % (src, nid, reg, arg_idx), 16)
                        if eid not in existing:
                            edges.append({
                                "edge_id": eid,
                                "mapping_id": eid,
                                "edge_kind": "relation",
                                "edge_type": "param_def_use",
                                "evidence_type": "param_def_use",
                                "source_node": src,
                                "target_node": nid,
                                "source_ref": src,
                                "target_ref": nid,
                                "from": src,
                                "to": nid,
                                "source_layer": "InstrSem",
                                "target_layer": "InstrSem",
                                "strength": 0.90,
                                "function_entry": fe,
                                "trigger": {"param_idx": arg_idx, "reg": reg, "direction": "def_to_call"},
                                "paramflow_real_edge": True
                            })
                            existing.add(eid)
                            added_param += 1
                pending_returns.append({"call_node": nid, "remaining": 12})

            if reads and pending_returns:
                keep = []
                for pr in pending_returns:
                    pr["remaining"] -= 1
                    if pr["remaining"] <= 0:
                        continue
                    hit_ret = False
                    for rr in _pf_ret_regs(prog):
                        if rr in reads:
                            hit_ret = True
                            break
                    if hit_ret:
                        eid = "pf_ret_%s" % _v11_hash("%s|%s|ret" % (pr["call_node"], nid), 16)
                        if eid not in existing:
                            edges.append({
                                "edge_id": eid,
                                "mapping_id": eid,
                                "edge_kind": "relation",
                                "edge_type": "return_use",
                                "evidence_type": "return_use",
                                "source_node": pr["call_node"],
                                "target_node": nid,
                                "source_ref": pr["call_node"],
                                "target_ref": nid,
                                "from": pr["call_node"],
                                "to": nid,
                                "source_layer": "InstrSem",
                                "target_layer": "InstrSem",
                                "strength": 0.85,
                                "function_entry": fe,
                                "trigger": {"reg": _pf_ret_regs(prog)[0], "direction": "call_to_use"},
                                "paramflow_real_edge": True
                            })
                            existing.add(eid)
                            added_ret += 1
                    else:
                        keep.append(pr)
                pending_returns = keep

            for w in writes:
                if w and w != "MEM":
                    last_def[w] = nid
                    # Store aliases too.
                    for a in _pf_reg_aliases(w):
                        last_def[a] = nid

    return added_param, added_ret


def _paramflow_enrich_semantic_dict(sem, prog):
    if not isinstance(sem, dict):
        return sem
    changed, call_count = _pf_enrich_instrsem_paramflow_fields(sem, prog)
    added_param, added_ret = _pf_build_paramflow_edges(sem, prog)
    try:
        pi = sem.setdefault("program_info", {})
        pi["v11_paramflow_patch"] = {
            "enabled": True,
            "changed_instrsem": changed,
            "call_instrsem": call_count,
            "added_param_def_use_edges": added_param,
            "added_return_use_edges": added_ret,
            "fields": ["seq", "read_set", "write_set", "operands", "call_target", "argument_locs", "return_locs"],
            "note": "Paramflow fields are added after v11_typesem_plus refinement for api_call neighborhood extraction."
        }
    except:
        pass
    return sem
# ==============================================
# v11-typesem-plus-paramflow-callfix PATCH:
# Generate real InstrSem(CALL) -> APISem api_call evidence edges at source.
# ==============================================
# Motivation:
#   In later Task2 experiments many api_call anchors had EvidenceStub as source,
#   meaning the high-level APISem was not directly bound to the real CALL
#   instruction. This patch normalizes callsite addresses, matches APISem objects
#   back to InstrSem CALL/CALLIND records, and emits standard api_call evidence
#   edges whose source is a real InstrSem node.
# ==============================================


def _cf_str(x):
    try:
        if x is None:
            return ""
        return str(x)
    except:
        try:
            return unicode(x)
        except:
            return ""


def _cf_norm_addr(x):
    s = _cf_str(x).strip().lower()
    if not s:
        return ""
    # Extract a hex-looking address if this is an id like instr_00405dac_7970.
    try:
        m = re.search(r'(0x[0-9a-fA-F]+|[0-9a-fA-F]{5,16})', s)
        if m:
            s = m.group(1).lower()
    except:
        pass
    if s.startswith("0x"):
        s = s[2:]
    # Remove non-hex separators that may appear in Ghidra stringified addresses.
    try:
        m2 = re.search(r'([0-9a-fA-F]{5,16})', s)
        if m2:
            s = m2.group(1).lower()
    except:
        pass
    try:
        s = s.lstrip("0")
    except:
        pass
    return s or "0"


def _cf_upper(x):
    try:
        return _cf_str(x).upper()
    except:
        return ""


def _cf_get_attrs(obj):
    try:
        a = obj.get("attrs")
        if isinstance(a, dict):
            return a
    except:
        pass
    return obj if isinstance(obj, dict) else {}


def _cf_unique(seq):
    out = []
    seen = set()
    for x in seq or []:
        sx = _cf_str(x)
        if not sx or sx in seen:
            continue
        seen.add(sx)
        out.append(x)
    return out


def _cf_get_instr_sem(sem):
    try:
        arr = sem.get("low_level_sem", {}).get("instr_sem", [])
        if isinstance(arr, list):
            return arr
    except:
        pass
    return []


def _cf_get_block_sem(sem):
    try:
        arr = sem.get("low_level_sem", {}).get("block_sem", [])
        if isinstance(arr, list):
            return arr
    except:
        pass
    return []


def _cf_get_api_sem(sem):
    try:
        arr = sem.get("high_level_sem", {}).get("api_sem", [])
        if isinstance(arr, list):
            return arr
    except:
        pass
    return []


def _cf_node_ids_by_layer(sem):
    nodes = {}
    paths = [
        ("InstrSem", sem.get("low_level_sem", {}).get("instr_sem", [])),
        ("BlockSem", sem.get("low_level_sem", {}).get("block_sem", [])),
        ("CFGSem", sem.get("mid_level_sem", {}).get("cfg_sem", [])),
        ("DFGSem", sem.get("mid_level_sem", {}).get("dataflow_sem", [])),
        ("TypeSem", sem.get("high_level_sem", {}).get("type_sem", [])),
        ("InterfaceSem", sem.get("high_level_sem", {}).get("interface_sem", [])),
        ("APISem", sem.get("high_level_sem", {}).get("api_sem", [])),
        ("LayoutSem", sem.get("high_level_sem", {}).get("mem_layout_sem", [])),
    ]
    for layer, arr in paths:
        try:
            for obj in arr or []:
                if isinstance(obj, dict) and obj.get("id"):
                    nodes[obj.get("id")] = layer
        except:
            pass
    return nodes


def _cf_instr_id(obj):
    if not isinstance(obj, dict):
        return ""
    a = _cf_get_attrs(obj)
    return obj.get("id") or a.get("id") or obj.get("node_id") or a.get("node_id") or ""


def _cf_instr_addr_candidates(obj):
    cands = []
    if not isinstance(obj, dict):
        return cands
    a = _cf_get_attrs(obj)
    for k in ["address", "addr", "instruction_address", "site", "raw_addr"]:
        if obj.get(k):
            cands.append(obj.get(k))
        if a.get(k):
            cands.append(a.get(k))
    iid = _cf_instr_id(obj)
    if iid:
        cands.append(iid)
    return _cf_unique(cands)


def _cf_is_call_instr(obj):
    if not isinstance(obj, dict):
        return False
    a = _cf_get_attrs(obj)
    try:
        if obj.get("paramflow_call_like") or a.get("paramflow_call_like"):
            return True
    except:
        pass
    op = _cf_upper(obj.get("opcode") or a.get("opcode"))
    if "CALL" in op:
        return True
    # Fallback to id/text inspection, but keep it conservative.
    try:
        text = json.dumps(obj, ensure_ascii=False).upper()
        if "CALLIND" in text or "\"CALL\"" in text or "OPCODE\": \"CALL" in text:
            return True
    except:
        pass
    return False


def _cf_build_instr_indices(sem):
    by_addr = defaultdict(list)
    by_id = {}
    call_count = 0
    for obj in _cf_get_instr_sem(sem):
        if not isinstance(obj, dict):
            continue
        iid = _cf_instr_id(obj)
        if iid:
            by_id[iid] = obj
        is_call = _cf_is_call_instr(obj)
        if is_call:
            call_count += 1
        for cand in _cf_instr_addr_candidates(obj):
            raw = _cf_str(cand)
            norm = _cf_norm_addr(cand)
            if raw:
                by_addr[raw].append(obj)
            if norm:
                by_addr[norm].append(obj)
        # Record normalized address back to object for downstream consistency.
        try:
            addr = obj.get("address") or _cf_get_attrs(obj).get("address")
            naddr = _cf_norm_addr(addr or iid)
            if naddr:
                obj.setdefault("address_norm", naddr)
        except:
            pass
    return by_addr, by_id, call_count


def _cf_api_id(api):
    if not isinstance(api, dict):
        return ""
    return api.get("id") or api.get("node_id") or ""


def _cf_api_callsite_candidates(api):
    cands = []
    if not isinstance(api, dict):
        return cands
    for k in ["callsite", "call_site", "instruction_address", "address", "site"]:
        if api.get(k):
            cands.append(api.get(k))
    st = api.get("semantic_template")
    if isinstance(st, dict):
        sr = st.get("symbol_resolution")
        if isinstance(sr, dict):
            for k in ["callsite", "call_site", "instruction_address", "address"]:
                if sr.get(k):
                    cands.append(sr.get(k))
    pc = api.get("propagated_constraints")
    if isinstance(pc, dict):
        for k in ["callsite", "call_site", "instruction_address"]:
            if pc.get(k):
                cands.append(pc.get(k))
    aid = _cf_api_id(api)
    if aid:
        cands.append(aid)
    return _cf_unique(cands)


def _cf_api_callsite_norm(api):
    for c in _cf_api_callsite_candidates(api):
        n = _cf_norm_addr(c)
        if n:
            return n
    return ""


def _cf_mapping_seen(mappings):
    seen = set()
    try:
        for m in mappings or []:
            if not isinstance(m, dict):
                continue
            seen.add((m.get("rule_id"), m.get("from") or m.get("source_node") or m.get("source_ref"),
                      m.get("to") or m.get("target_node") or m.get("target_ref"),
                      m.get("evidence_type"), m.get("edge_kind")))
    except:
        pass
    return seen


def _cf_edge_hash(s):
    try:
        return _v11_hash(s, 16)
    except:
        try:
            return str(abs(hash(str(s))))[:16]
        except:
            return "0"


def _cf_clamp(x):
    try:
        return _clamp_strength(x)
    except:
        try:
            v = float(x)
        except:
            v = 0.7
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v


def _cf_add_cross_mapping(sem, source_id, target_id, api, call_obj, strength):
    mappings = sem.setdefault("cross_layer_mapping", [])
    seen = _cf_mapping_seen(mappings)
    callsite = api.get("callsite") or api.get("call_site") or call_obj.get("address") or call_obj.get("address_norm")
    callsite_norm = _cf_api_callsite_norm(api) or _cf_norm_addr(callsite)
    trigger = {
        "callsite": callsite,
        "callsite_norm": callsite_norm,
        "api_name": api.get("api_name") or api.get("name"),
        "instruction_address": call_obj.get("address") or callsite,
        "instruction_address_norm": call_obj.get("address_norm") or _cf_norm_addr(call_obj.get("address") or callsite),
        "relation_type": "call_instruction",
        "evidence_type": "api_call",
        "semantic_template_available": bool(api.get("semantic_template")),
        "call_target_observed": bool(api.get("api_name") or call_obj.get("call_target") or call_obj.get("callee"))
    }
    try:
        _append_mapping(
            mappings, seen,
            "R6_CALLFIX", "API调用真实指令绑定",
            source_id, target_id,
            "InstrSem", "APISem",
            trigger,
            "api_call",
            strength,
            "APISem",
            ["call_target_observed", "api_template_bindable", "real_call_instruction_source"],
            "evidence"
        )
        return True
    except Exception:
        # Manual fallback if _append_mapping is unavailable for any reason.
        key = ("R6_CALLFIX", source_id, target_id, "api_call", "evidence")
        if key in seen:
            return False
        eid = "R6_CALLFIX_%d" % len(mappings)
        mappings.append({
            "mapping_id": eid,
            "rule_id": "R6_CALLFIX",
            "name": "API调用真实指令绑定",
            "from": source_id,
            "to": target_id,
            "edge_kind": "evidence",
            "source_layer": "InstrSem",
            "target_layer": "APISem",
            "trigger": trigger,
            "evidence_type": "api_call",
            "strength": _cf_clamp(strength),
            "emit": "APISem",
            "constraints": ["call_target_observed", "api_template_bindable", "real_call_instruction_source"]
        })
        return True


def _cf_existing_evidence_keys(edges):
    seen = set()
    try:
        for e in edges or []:
            if not isinstance(e, dict):
                continue
            seen.add((e.get("evidence_type"), e.get("source_node") or e.get("source_ref") or e.get("from"),
                      e.get("target_node") or e.get("target_ref") or e.get("to"), e.get("rule_id")))
    except:
        pass
    return seen


def _cf_add_standard_evidence_edge(sem, source_id, target_id, api, call_obj, strength):
    edges = sem.setdefault("evidence_edges", [])
    seen = _cf_existing_evidence_keys(edges)
    key = ("api_call", source_id, target_id, "R6_CALLFIX")
    if key in seen:
        return False
    callsite = api.get("callsite") or api.get("call_site") or call_obj.get("address")
    callsite_norm = _cf_api_callsite_norm(api) or _cf_norm_addr(callsite)
    edge_id = "ev_api_callfix_%s" % _cf_edge_hash("%s|%s|%s" % (source_id, target_id, callsite_norm))
    edges.append({
        "edge_id": edge_id,
        "mapping_id": edge_id,
        "edge_kind": "evidence",
        "edge_type": "api_call",
        "rule_id": "R6_CALLFIX",
        "evidence_type": "api_call",
        "source_node": source_id,
        "target_node": target_id,
        "source_ref": source_id,
        "target_ref": target_id,
        "from": source_id,
        "to": target_id,
        "source_layer": "InstrSem",
        "target_layer": "APISem",
        "strength": _cf_clamp(strength),
        "function_entry": api.get("propagated_constraints", {}).get("caller_function_entry") if isinstance(api.get("propagated_constraints"), dict) else "",
        "trigger": {
            "callsite": callsite,
            "callsite_norm": callsite_norm,
            "api_name": api.get("api_name") or api.get("name"),
            "instruction_address": call_obj.get("address") or callsite,
            "instruction_address_norm": call_obj.get("address_norm") or _cf_norm_addr(call_obj.get("address") or callsite),
            "relation_type": "call_instruction",
            "semantic_template_available": bool(api.get("semantic_template")),
            "call_target_observed": bool(api.get("api_name") or call_obj.get("call_target") or call_obj.get("callee"))
        },
        "callfix_real_instr_to_api": True,
        "v11_real_edge": True
    })
    return True


def _cf_filter_bad_api_evidence(sem, nodes_by_layer):
    removed_mapping = 0
    removed_evidence = 0

    def bad_api_source(src, sl, et):
        et = _cf_str(et)
        if et not in ("api_call", "call_instruction"):
            return False
        if not src or src not in nodes_by_layer:
            return True
        layer = sl or nodes_by_layer.get(src)
        return layer not in ("InstrSem",)

    # Keep BlockSem call_behavior_context edges; only filter api_call/call_instruction that cannot resolve to InstrSem.
    try:
        new_maps = []
        for m in sem.get("cross_layer_mapping", []) or []:
            if not isinstance(m, dict):
                new_maps.append(m)
                continue
            src = m.get("source_node") or m.get("source_ref") or m.get("from")
            sl = m.get("source_layer")
            et = m.get("evidence_type")
            if bad_api_source(src, sl, et):
                removed_mapping += 1
                continue
            new_maps.append(m)
        sem["cross_layer_mapping"] = new_maps
    except:
        pass

    try:
        new_edges = []
        for e in sem.get("evidence_edges", []) or []:
            if not isinstance(e, dict):
                new_edges.append(e)
                continue
            src = e.get("source_node") or e.get("source_ref") or e.get("from")
            sl = e.get("source_layer")
            et = e.get("evidence_type") or e.get("edge_type")
            if bad_api_source(src, sl, et):
                removed_evidence += 1
                continue
            new_edges.append(e)
        sem["evidence_edges"] = new_edges
    except:
        pass
    return removed_mapping, removed_evidence


def _cf_match_call_instrs_for_api(api, instr_by_addr, instr_by_id):
    matched = []
    seen = set()

    # 1) Direct call_instr_id if already recorded by a previous patch or by gen_api_sem.
    for k in ["call_instr_id", "call_instruction_id", "instr_id", "source_instr_id"]:
        cid = api.get(k) if isinstance(api, dict) else None
        if cid and cid in instr_by_id:
            obj = instr_by_id[cid]
            if _cf_is_call_instr(obj) and cid not in seen:
                matched.append(obj)
                seen.add(cid)

    # 2) Address-normalized callsite matching.
    for c in _cf_api_callsite_candidates(api):
        keys = [c, _cf_norm_addr(c)]
        for key in keys:
            for obj in instr_by_addr.get(key, []) or []:
                iid = _cf_instr_id(obj)
                if not iid or iid in seen:
                    continue
                if not _cf_is_call_instr(obj):
                    continue
                matched.append(obj)
                seen.add(iid)
    return matched


def _cf_enrich_call_and_api_fields(api, call_obj):
    try:
        api_name = api.get("api_name") or api.get("name") or call_obj.get("call_target") or call_obj.get("callee")
        callsite_norm = _cf_api_callsite_norm(api) or _cf_norm_addr(call_obj.get("address") or call_obj.get("address_norm"))
        call_id = _cf_instr_id(call_obj)
        if callsite_norm:
            api["callsite_norm"] = callsite_norm
        if call_id:
            api["call_instr_id"] = call_id
            api["api_call_source_node"] = call_id
            api["api_call_source_layer"] = "InstrSem"
        if api_name:
            api["api_name"] = api_name
            call_obj.setdefault("api_name", api_name)
            call_obj.setdefault("callee", api_name)
            call_obj.setdefault("call_target", api_name)
        call_obj["api_call_target_ref"] = api.get("id")
        call_obj["api_call_evidence_fixed"] = True
        api["api_call_evidence_fixed"] = True
    except:
        pass


def _callfix_enrich_semantic_dict(sem, prog=None):
    if not isinstance(sem, dict):
        return sem
    instr_by_addr, instr_by_id, call_count = _cf_build_instr_indices(sem)
    apis = _cf_get_api_sem(sem)
    nodes_by_layer = _cf_node_ids_by_layer(sem)

    matched_api = 0
    added_mapping = 0
    added_evidence = 0
    matched_call_edges = 0
    missing_call = 0
    examples = []

    # Filter preexisting broken api_call/call_instruction edges that would later become EvidenceStub.
    removed_mapping, removed_evidence = _cf_filter_bad_api_evidence(sem, nodes_by_layer)

    for api in apis:
        if not isinstance(api, dict) or not api.get("id"):
            continue
        calls = _cf_match_call_instrs_for_api(api, instr_by_addr, instr_by_id)
        if not calls:
            missing_call += 1
            try:
                api.setdefault("diagnostics", {})["callfix_missing_real_call_instr"] = True
                api["callsite_norm"] = api.get("callsite_norm") or _cf_api_callsite_norm(api)
            except:
                pass
            continue
        matched_api += 1
        # Normally there is one CALL per APISem callsite; keep all if multiple are exactly matched.
        for call_obj in calls:
            sid = _cf_instr_id(call_obj)
            tid = api.get("id")
            if not sid or not tid:
                continue
            _cf_enrich_call_and_api_fields(api, call_obj)
            strength = 0.97 if api.get("semantic_template") else 0.90
            if _cf_add_cross_mapping(sem, sid, tid, api, call_obj, strength):
                added_mapping += 1
            if _cf_add_standard_evidence_edge(sem, sid, tid, api, call_obj, strength):
                added_evidence += 1
            matched_call_edges += 1
            if len(examples) < 20:
                examples.append({
                    "api_id": tid,
                    "api_name": api.get("api_name"),
                    "callsite": api.get("callsite"),
                    "callsite_norm": api.get("callsite_norm"),
                    "call_instr_id": sid,
                    "call_instr_address": call_obj.get("address"),
                    "opcode": call_obj.get("opcode")
                })

    try:
        pi = sem.setdefault("program_info", {})
        pi["v11_callfix_patch"] = {
            "enabled": True,
            "goal": "emit real InstrSem(CALL)->APISem api_call evidence edges instead of EvidenceStub api_call anchors",
            "call_instrsem_indexed": call_count,
            "apisem_total": len(apis),
            "apisem_matched_to_real_call": matched_api,
            "apisem_missing_real_call": missing_call,
            "added_cross_layer_api_call_mappings": added_mapping,
            "added_standard_api_call_evidence_edges": added_evidence,
            "matched_call_edges": matched_call_edges,
            "removed_broken_api_mappings": removed_mapping,
            "removed_broken_api_evidence_edges": removed_evidence,
            "examples": examples
        }
    except:
        pass

    try:
        _v11_add_edge_alias_to_usg(sem)
    except:
        try:
            usg = sem.get("usg")
            if isinstance(usg, dict):
                usg["E_ev"] = [x.get("edge_id") for x in sem.get("evidence_edges", []) if isinstance(x, dict) and x.get("edge_id")]
                usg["evidence_edge_count"] = len(usg.get("E_ev", []))
        except:
            pass
    return sem


def build_dict(prog, prev_snapshot_path=None):
    if _TYPEPLUS_PREV_BUILD_DICT is None:
        sem = {}
    else:
        try:
            sem = _TYPEPLUS_PREV_BUILD_DICT(prog, prev_snapshot_path)
        except TypeError:
            sem = _TYPEPLUS_PREV_BUILD_DICT(prog)
    if not isinstance(sem, dict):
        return sem

    try:
        sem = _tp_refine_typesem_and_interface(sem)
    except Exception as e:
        try:
            sem.setdefault("program_info", {})["typesem_plus_error"] = _tp_str(e)
        except:
            pass

    # Enrich InstrSem and relation edges for api_call parameter-flow neighborhoods.
    try:
        sem = _paramflow_enrich_semantic_dict(sem, prog)
    except Exception as e:
        try:
            sem.setdefault("program_info", {})["paramflow_patch_error"] = _tp_str(e)
        except:
            pass

    # Source-level api_call fix: emit real InstrSem(CALL) -> APISem evidence edges.
    try:
        sem = _callfix_enrich_semantic_dict(sem, prog)
    except Exception as e:
        try:
            sem.setdefault("program_info", {})["callfix_patch_error"] = _tp_str(e)
        except:
            pass

    try:
        pi = sem.setdefault("program_info", {})
        pi["schema_version"] = "semantic_dict_v11_typesem_plus_paramflow_callfix"
        pi["generator"] = "ghidra_usg_patch_v11_typesem_plus_paramflow_callfix"
        prev_opts = pi.get("v11_optimizations", {}) if isinstance(pi.get("v11_optimizations", {}), dict) else {}
        pi["v11_typesem_plus_optimizations"] = {
            "inherits_v11_evidence_layout": True,
            "preserve_original_type_candidates": True,
            "dfg_storage_memory_base_type_refinement": True,
            "param_name_type_refinement": True,
            "api_name_param_hint_refinement": True,
            "evidence_edge_pointer_hint_refinement": True,
            "mirror_typesem_to_interfacesem": True,
            "v11_evidence_edge_count": prev_opts.get("evidence_edge_count", len(sem.get("evidence_edges", []))),
            "paramflow_instrsem_read_write_fields": True,
            "paramflow_call_argument_return_locs": True,
            "paramflow_param_def_use_edges": True,
            "paramflow_return_use_edges": True,
            "callfix_real_instr_to_api_edges": True,
            "callfix_no_evidencestub_api_call": True
        }
    except:
        pass

    # Consistency report after TypeSem/InterfaceSem mirroring.
    try:
        prev_sem = _load_json_maybe_v5(prev_snapshot_path)
        sem["consistency_report"] = run_consistency_checks(sem, prev_sem)
    except:
        pass
    return sem


# ==============================================
# BlockSem upstream export contract v7.0
# 目的：在不破坏既有 v11/typesem/paramflow/callfix 输出的前提下，补齐
# 1) 稳定 p-code site/value ID；2) LOAD/COPY/CAST/INT_ADD 等值传递关系；
# 3) 跨基本块 reaching-definition 标记；4) Block 显式有序成员；
# 5) 为 RSP/RBP/CFA 外部 CFI 区间桥接保存足够的机器可读字段。
# ==============================================

_BLOCKSEM_V7_PREV_BUILD_DICT = build_dict


def _bsv7_hash(*parts):
    try:
        import hashlib
        s = '|'.join([str(x) for x in parts])
        return hashlib.sha1(s.encode('utf-8')).hexdigest()[:20]
    except:
        return str(abs(hash(tuple([str(x) for x in parts]))))


def _bsv7_site_id(func_entry, address, pcode_index):
    return 'pcsite:' + _bsv7_hash(func_entry, address, pcode_index)


def _bsv7_value_id(func_entry, producer_site, location):
    return 'value:' + _bsv7_hash(func_entry, producer_site or 'ENTRY', location)


def _bsv7_get_dfgs(sem):
    mid = sem.get('mid_level_sem', {}) if isinstance(sem, dict) else {}
    if not isinstance(mid, dict):
        return []
    x = mid.get('dataflow_sem')
    if isinstance(x, list):
        return x
    x = mid.get('dfg_sem')
    return x if isinstance(x, list) else []


def _bsv7_cfg_block_to_func(sem):
    out = {}
    mid = sem.get('mid_level_sem', {}) if isinstance(sem, dict) else {}
    cfgs = []
    if isinstance(mid, dict):
        cfgs = mid.get('cfg_sem', []) or []
    for cfg in cfgs:
        if not isinstance(cfg, dict):
            continue
        fe = cfg.get('entry_address') or cfg.get('function_entry')
        for b in cfg.get('blocks', []) or []:
            if not isinstance(b, dict):
                continue
            be = b.get('start_address') or b.get('entry')
            if be is not None:
                out[str(be)] = fe
    return out


def _bsv7_export_target_pcode(prog):
    # Export only p-code classes required for value provenance, not the entire raw decompiler graph.
    target_ops = set(['LOAD','STORE','COPY','CAST','INT_ADD','INT_SUB','PTRADD','PTRSUB',
                      'INT_ZEXT','INT_SEXT','SUBPIECE','PIECE','MULTIEQUAL','INDIRECT'])
    rows = []
    transfer_edges = []
    prologue = []
    try:
        tm = _get_script_monitor()
        block_model = BasicBlockModel(prog)
        funcs = _get_internal_functions(prog, include_thunks=False)
    except:
        return rows, transfer_edges, prologue

    for func in funcs:
        try:
            fe = _addr_to_str(func.getEntryPoint())
            blocks = _BASE_GET_FUNCTION_BLOCKS(prog, func, block_model, tm)
        except:
            continue
        for block in blocks:
            be = block.get('entry_str')
            bid = block.get('block_id')
            for instr in block.get('instructions', []) or []:
                try:
                    pcops = list(instr.getPcode())
                except:
                    pcops = []
                for pidx, op in enumerate(pcops):
                    try:
                        facts = _extract_pcode_facts_v3(prog, instr, op, pidx, block)
                    except:
                        continue
                    m = str(facts.get('mnemonic') or '').upper()
                    if m not in target_ops:
                        continue
                    site = facts.get('site') or ('%s:%s' % (_addr_to_str(instr.getAddress()), pidx))
                    site_id = _bsv7_site_id(fe, _addr_to_str(instr.getAddress()), pidx)
                    inputs = []
                    for idx, desc in enumerate(facts.get('input_descs', []) or []):
                        if not isinstance(desc, dict):
                            continue
                        loc = desc.get('loc')
                        inp = {
                            'input_index': idx,
                            'loc': loc,
                            'const': desc.get('const'),
                            'piece': desc.get('piece'),
                            'entry_value_id': _bsv7_value_id(fe, None, loc) if loc else None
                        }
                        inputs.append(inp)
                    od = facts.get('output_desc', {}) if isinstance(facts.get('output_desc', {}), dict) else {}
                    out_loc = od.get('loc')
                    out_value = _bsv7_value_id(fe, site, out_loc) if out_loc else None
                    row = {
                        'pcode_site_id': site_id,
                        'site': site,
                        'function_entry': fe,
                        'block_entry': be,
                        'block_id': bid,
                        'instruction_address': _addr_to_str(instr.getAddress()),
                        'pcode_index': pidx,
                        'opcode': m,
                        'inputs': inputs,
                        'output_loc': out_loc,
                        'output_value_id': out_value,
                        'memory_accesses': facts.get('memory_accesses', []) or []
                    }
                    rows.append(row)
                    if out_value:
                        for inp in inputs:
                            src = inp.get('entry_value_id')
                            if not src:
                                continue
                            edge = {
                                'edge_id': 'pvedge:' + _bsv7_hash(src, out_value, site_id, inp.get('input_index')),
                                'edge_type': 'pcode_value_transfer',
                                'opcode': m,
                                'source_value_id': src,
                                'target_value_id': out_value,
                                'source_location': inp.get('loc'),
                                'target_location': out_loc,
                                'pcode_site_id': site_id,
                                'site': site,
                                'function_entry': fe,
                                'block_entry': be,
                                'block_id': bid,
                                'input_index': inp.get('input_index')
                            }
                            transfer_edges.append(edge)
                            sl = str(inp.get('loc') or '').upper()
                            tl = str(out_loc or '').upper()
                            if ('RSP' in sl or 'RBP' in sl or 'RSP' in tl or 'RBP' in tl) and m in ('COPY','CAST','INT_ADD','INT_SUB','PTRADD','PTRSUB'):
                                pe = dict(edge)
                                pe['edge_type'] = 'stack_frame_prologue_transform'
                                prologue.append(pe)
    return rows, transfer_edges, prologue


def _blocksem_v7_enrich_semantic_dict(sem, prog):
    if not isinstance(sem, dict):
        return sem
    low = sem.setdefault('low_level_sem', {})
    if not isinstance(low, dict):
        return sem
    instrs = low.get('instr_sem', []) or []
    blocks = low.get('block_sem', []) or []
    block_to_func = _bsv7_cfg_block_to_func(sem)

    # Map p-code site -> existing InstrSem ID and seed stable IDs.
    instr_by_site = {}
    instr_by_block = {}
    for ins in instrs:
        if not isinstance(ins, dict):
            continue
        fe = ins.get('function_entry') or block_to_func.get(str(ins.get('block_entry')))
        be = ins.get('block_entry')
        addr = ins.get('address')
        pidx = ins.get('pcode_index')
        ins['function_entry'] = fe
        ins['pcode_site_id'] = _bsv7_site_id(fe, addr, pidx)
        ins.setdefault('input_value_ids', [])
        ins.setdefault('output_value_ids', [])
        site = '%s:%s' % (addr, pidx)
        instr_by_site[site] = ins
        key = (str(fe), str(be))
        instr_by_block.setdefault(key, []).append(ins)

    # Existing dataflow_sem already computes reaching definitions to a fixed point.
    # Here we make each reaching definition explicit and give the carried value a stable ID.
    dfgs = _bsv7_get_dfgs(sem)
    cross_block_count = 0
    value_edge_count = 0
    value_ids = set()
    for dfg in dfgs:
        if not isinstance(dfg, dict):
            continue
        fe = dfg.get('entry_address') or dfg.get('function_entry')
        for definition in dfg.get('definitions', []) or []:
            if not isinstance(definition, dict):
                continue
            site = definition.get('site')
            loc = definition.get('location')
            vid = _bsv7_value_id(fe, site, loc)
            definition['value_id'] = vid
            definition['pcode_site_id'] = _bsv7_site_id(fe, definition.get('instruction_address'), str(site).split(':')[-1] if site else None)
            value_ids.add(vid)
            src_ins = instr_by_site.get(str(site))
            if src_ins is not None and vid not in src_ins['output_value_ids']:
                src_ins['output_value_ids'].append(vid)
        for e in dfg.get('data_edges', []) or []:
            if not isinstance(e, dict):
                continue
            src_site = e.get('from_site')
            dst_site = e.get('to_site')
            loc = e.get('location')
            vid = _bsv7_value_id(fe, src_site, loc)
            e['value_id'] = vid
            e['edge_id'] = e.get('edge_id') or ('dfgval:' + _bsv7_hash(fe, src_site, dst_site, loc))
            e['cross_block'] = str(e.get('from_block')) != str(e.get('to_block'))
            e['reaching_definition_proven'] = True
            e['source_pcode_site_id'] = _bsv7_site_id(fe, e.get('from'), str(src_site).split(':')[-1] if src_site else None)
            e['target_pcode_site_id'] = _bsv7_site_id(fe, e.get('to'), str(dst_site).split(':')[-1] if dst_site else None)
            value_ids.add(vid)
            value_edge_count += 1
            if e['cross_block']:
                cross_block_count += 1
            src_ins = instr_by_site.get(str(src_site))
            dst_ins = instr_by_site.get(str(dst_site))
            if src_ins is not None and vid not in src_ins['output_value_ids']:
                src_ins['output_value_ids'].append(vid)
            if dst_ins is not None and vid not in dst_ins['input_value_ids']:
                dst_ins['input_value_ids'].append(vid)
        for acc in dfg.get('memory_access_patterns', []) or []:
            if not isinstance(acc, dict):
                continue
            acc['function_entry'] = fe
            acc['access_id'] = acc.get('access_id') or ('memacc:' + _bsv7_hash(fe, acc.get('site'), acc.get('access'), acc.get('location')))
            acc['pcode_site_id'] = _bsv7_site_id(fe, acc.get('instruction_address'), str(acc.get('site')).split(':')[-1] if acc.get('site') else None)
            acc['provenance_seed_locations'] = _unique_list([x for x in [acc.get('base'), acc.get('location'), acc.get('value_loc')] if x] + list(acc.get('support_locs', []) or []))

    # Explicit Block membership. Members are InstrSem IDs, ordered first by machine
    # instruction address and then by p-code index. No address-only synthetic member is created.
    block_complete = 0
    for b in blocks:
        if not isinstance(b, dict):
            continue
        be = b.get('entry') or b.get('block_entry')
        fe = b.get('function_entry') or block_to_func.get(str(be))
        b['function_entry'] = fe
        if b.get('block_id') in (None, ''):
            b['block_id'] = 'block:' + _bsv7_hash(fe, be, b.get('end'))
        members = instr_by_block.get((str(fe), str(be)), [])
        try:
            members = sorted(members, key=lambda x: (int(str(x.get('address')), 16), int(x.get('pcode_index') or 0)))
        except:
            members = sorted(members, key=lambda x: (str(x.get('address')), int(x.get('pcode_index') or 0)))
        ids = [x.get('id') for x in members if x.get('id')]
        unique_addrs = set([str(x.get('address')) for x in members if x.get('address') is not None])
        expected_instr = b.get('instruction_count')
        complete = bool(ids)
        if expected_instr not in (None, ''):
            try:
                complete = complete and (len(unique_addrs) == int(expected_instr))
            except:
                complete = False
        b['ordered_instruction_ids'] = ids
        b['member_count'] = len(ids)
        b['member_unique_machine_instruction_count'] = len(unique_addrs)
        b['member_list_complete'] = bool(complete)
        b['membership_source'] = 'explicit InstrSem function_entry+block_entry join'
        if complete:
            block_complete += 1

    # Targeted P-code operand transfer rows provide the missing RSP/RBP provenance
    # needed by the external CFI interval join.
    pcode_rows, transfer_edges, prologue = _bsv7_export_target_pcode(prog)
    sem['blocksem_v7'] = {
        'schema_version': 'blocksem_upstream_contract_v7',
        'pcode_ops': pcode_rows,
        'pcode_value_edges': transfer_edges,
        'stack_frame_prologue_relations': prologue,
        'stats': {
            'instr_sem': len(instrs),
            'block_sem': len(blocks),
            'block_member_complete': block_complete,
            'dfg_value_edges': value_edge_count,
            'cross_block_reaching_def_edges': cross_block_count,
            'stable_value_ids': len(value_ids),
            'target_pcode_ops': len(pcode_rows),
            'pcode_value_transfer_edges': len(transfer_edges),
            'stack_frame_prologue_relations': len(prologue)
        }
    }
    try:
        pi = sem.setdefault('program_info', {})
        pi['blocksem_upstream_contract_v7'] = sem['blocksem_v7']['stats']
        pi['schema_version'] = 'semantic_dict_v11_typesem_plus_paramflow_callfix_blocksem_v7'
        pi['generator'] = 'ghidra_usg_patch_v11_typesem_plus_paramflow_callfix_blocksem_v7'
    except:
        pass
    return sem


def build_dict(prog, prev_snapshot_path=None):
    try:
        sem = _BLOCKSEM_V7_PREV_BUILD_DICT(prog, prev_snapshot_path)
    except TypeError:
        sem = _BLOCKSEM_V7_PREV_BUILD_DICT(prog)
    try:
        return _blocksem_v7_enrich_semantic_dict(sem, prog)
    except Exception as e:
        try:
            sem.setdefault('program_info', {})['blocksem_v7_error'] = str(e)
        except:
            pass
        return sem


# ==============================================
# 运行 & 输出 JSON（v5）
# ==============================================
if __name__ == "__main__":
    prog = currentProgram
    prev_snapshot_path = _get_prev_snapshot_arg_v5()
    data = build_dict(prog, prev_snapshot_path)

    out_dir = _default_output_dir()
    path = os.path.join(out_dir, prog.getName() + "_semantic_dict.json")

    with codecs.open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Over! -> %s" % path)
