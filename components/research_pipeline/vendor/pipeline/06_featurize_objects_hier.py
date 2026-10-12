#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_featurize_objects_hier.py

2.2.1.1 完整分层桥接映射的工程增强版。

目标：
把原来的 one-hot + numeric + hash baseline，升级为“分层语义特征编码”：

低层 InstrSem / BlockSem：
  - opcode
  - read_set / write_set
  - operand / memory access
  - load/store/call/branch 标志
  - call target / api name

中层 CFGSem / DFGSem：
  - 控制流 / 数据流结构统计
  - degree / def-use 数量
  - entry input / memory base / deref chain 等结构特征

高层 TypeSem / InterfaceSem / APISem / LayoutSem：
  - type candidates + confidence
  - param type categories
  - return type category
  - api_name / api_family / semantic_template
  - layout offsets / stride / observed fields

输出仍然兼容现有 07_train_alignment.py / 07_train_alignment_api_head.py：
  node_features_*.pt
  feature_vocab_*.json
  node_features_*.jsonl

用法示例：
python3 task2_exp/src/06_featurize_objects_hier.py \
  --nodes $TASK2_ROOT/data/graphs/usg_nodes_selected_40k_api_filtered.jsonl \
  --neighborhoods $TASK2_ROOT/data/anchors/anchor_neighborhoods_balanced_40k_api_filtered.jsonl \
  --out-pt $TASK2_ROOT/data/features/node_features_40k_hier.pt \
  --vocab-out $TASK2_ROOT/data/features/feature_vocab_40k_hier.json \
  --node-feature-jsonl $TASK2_ROOT/data/features/node_features_40k_hier.jsonl \
  --hash-dim-low 192 \
  --hash-dim-mid 128 \
  --hash-dim-high 192
"""

import argparse
import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


NODE_TYPES = [
    "InstrSem", "BlockSem", "CFGSem", "DFGSem",
    "TypeSem", "InterfaceSem", "APISem", "LayoutSem",
    "EvidenceStub", "Unresolved", "unknown"
]

LAYERS = ["instruction", "block", "cfg", "dfg", "low", "mid", "high", "stub", "unknown"]

LOW_TYPES = {"InstrSem", "BlockSem"}
MID_TYPES = {"CFGSem", "DFGSem"}
HIGH_TYPES = {"TypeSem", "InterfaceSem", "APISem", "LayoutSem"}

API_FAMILY_KEYWORDS = {
    "memory_alloc": ["malloc", "calloc", "realloc", "operator.new", "new"],
    "memory_free": ["free", "operator.delete", "delete"],
    "memory_copy": ["memcpy", "memmove", "memset", "bcopy", "bzero"],
    "string": ["strlen", "strcmp", "strncmp", "strcpy", "strncpy", "strstr", "strchr", "atoi", "strtol"],
    "format_io": ["printf", "fprintf", "sprintf", "snprintf", "scanf", "sscanf", "puts", "putchar"],
    "file_io": ["open", "read", "write", "close", "fopen", "fread", "fwrite", "fclose", "lseek"],
    "math": ["sqrt", "sqrtf", "pow", "sin", "cos", "tan", "exp", "log", "floor", "ceil"],
    "process": ["exit", "abort", "system", "exec", "fork"],
    "ctype": ["isdigit", "isalpha", "isalnum", "tolower", "toupper"],
    "crypto": ["md5", "sha", "aes", "des", "camellia"],
    "time": ["time", "gettimeofday", "clock"],
}


GENERAL_NUMERIC = [
    "num_list_fields",
    "num_scalar_fields",
    "confidence_max",
    "confidence_mean",
    "eval_exclude",
    "has_function_entry",
    "has_location",
    "anchor_degree",
    "source_anchor_count",
    "target_anchor_count",
]

LOW_NUMERIC = [
    "is_instr",
    "is_block",
    "is_load",
    "is_store",
    "is_call",
    "is_branch",
    "is_return",
    "has_memory_access",
    "memory_access_count",
    "read_set_count",
    "write_set_count",
    "operand_count",
    "constant_count",
    "offset_count",
    "stack_access_count",
    "instr_count",
    "load_count",
    "store_count",
    "call_count",
    "branch_count",
    "opcode_ratio_mem",
    "opcode_ratio_call",
    "opcode_ratio_branch",
]

MID_NUMERIC = [
    "is_cfg",
    "is_dfg",
    "node_count",
    "edge_count",
    "in_degree",
    "out_degree",
    "successor_count",
    "predecessor_count",
    "branch_count_mid",
    "loop_flag",
    "back_edge_count",
    "def_use_count",
    "source_count",
    "sink_count",
    "entry_input_count",
    "memory_base_count",
    "deref_chain_length",
    "dataflow_depth",
    "used_as_address_count",
]

HIGH_NUMERIC = [
    "is_type",
    "is_interface",
    "is_api",
    "is_layout",
    "candidate_count",
    "param_count",
    "param_type_count",
    "return_known",
    "pointer_score",
    "int_score",
    "float_score",
    "struct_score",
    "unknown_score",
    "api_name_present",
    "api_family_present",
    "semantic_template_present",
    "argument_role_count",
    "return_role_present",
    "offset_count_high",
    "field_offset_count",
    "observed_field_offset_count",
    "min_offset_norm",
    "max_offset_norm",
    "stride_like",
    "layout_array_like",
]

NUMERIC_NAMES = GENERAL_NUMERIC + LOW_NUMERIC + MID_NUMERIC + HIGH_NUMERIC


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stable_hash_int(s):
    return int(hashlib.md5(str(s).encode("utf-8", errors="replace")).hexdigest(), 16)


def hash_index(token, dim):
    if dim <= 0:
        return 0
    return stable_hash_int(token) % dim


def as_float(x):
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        if math.isfinite(float(x)):
            return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except Exception:
            return None
    return None


def norm01(x, scale=10.0):
    if x is None:
        return 0.0
    try:
        return math.log1p(max(0.0, float(x))) / scale
    except Exception:
        return 0.0


def get_attrs(n):
    a = n.get("attrs", {})
    return a if isinstance(a, dict) else {}


def node_type(n):
    return n.get("node_type", "unknown")


def layer_of(n):
    return n.get("layer", "unknown")


def list_len(v):
    return len(v) if isinstance(v, list) else 0


def flatten_tokens(x, prefix=""):
    tokens = []
    if x is None:
        return tokens
    if isinstance(x, (str, int, float, bool)):
        s = str(x)
        if len(s) <= 256:
            tokens.append(prefix + s)
    elif isinstance(x, list):
        for v in x[:120]:
            tokens.extend(flatten_tokens(v, prefix=prefix))
    elif isinstance(x, dict):
        for k, v in list(x.items())[:120]:
            if isinstance(v, (str, int, float, bool)):
                tokens.append(f"{prefix}{k}={v}")
            elif isinstance(v, list):
                for vv in v[:80]:
                    if isinstance(vv, (str, int, float, bool)):
                        tokens.append(f"{prefix}{k}:{vv}")
            elif isinstance(v, dict):
                for kk, vv in list(v.items())[:50]:
                    if isinstance(vv, (str, int, float, bool)):
                        tokens.append(f"{prefix}{k}.{kk}={vv}")
    return tokens


def confidence_values(attrs):
    vals = []
    for k in [
        "confidence", "type_confidence", "type_confidences",
        "score", "strength", "candidate_confidence",
        "confidence_distribution",
    ]:
        v = attrs.get(k)
        if isinstance(v, list):
            for x in v:
                fx = as_float(x)
                if fx is not None:
                    vals.append(max(0.0, min(1.0, fx)))
        elif isinstance(v, dict):
            for x in v.values():
                fx = as_float(x)
                if fx is not None:
                    vals.append(max(0.0, min(1.0, fx)))
        else:
            fx = as_float(v)
            if fx is not None:
                vals.append(max(0.0, min(1.0, fx)))
    return vals


def confidence_stats(attrs):
    vals = confidence_values(attrs)
    if not vals:
        return 0.0, 0.0
    return max(vals), sum(vals) / len(vals)


def normalize_api_name(x):
    if not x:
        return ""
    s = str(x).strip()
    s = s.replace("@plt", "").replace(".plt", "")
    s = re.sub(r"@@?GLIBC[^,\s]*", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def api_family(name):
    s = normalize_api_name(name).lower()
    if not s:
        return ""
    for fam, keys in API_FAMILY_KEYWORDS.items():
        for k in keys:
            if k.lower() in s:
                return fam
    return "other_api"


def collect_api_strings(attrs):
    vals = []
    for k in [
        "api_name", "api_target", "call_target", "callee",
        "function_name", "func_name", "semantic_template",
        "api_template", "template", "name",
        "api_aware_normalized_api_name", "api_aware_family",
        "api_aware_semantic_template",
    ]:
        v = attrs.get(k)
        if v:
            if isinstance(v, list):
                vals.extend(str(x) for x in v[:30] if x)
            else:
                vals.append(str(v))
    return vals


def canonical_type_cat(x):
    if x is None:
        return "unknown"
    s = str(x).lower().strip()
    if not s or s in {"?", "none", "null", "undefined", "unknown"}:
        return "unknown"
    if any(k in s for k in ["ptr", "pointer", "char*", "void*", "struct*", "buffer", "buf", "array"]):
        return "ptr"
    if any(k in s for k in ["struct", "record", "layout", "object"]):
        return "struct"
    if any(k in s for k in ["size_t", "ssize_t", "uint", "int", "long", "short", "fd", "descriptor", "count", "len", "length", "offset", "argc"]):
        return "int"
    if any(k in s for k in ["float", "double", "real"]):
        return "float"
    if any(k in s for k in ["bool", "flag"]):
        return "bool"
    if "string" in s:
        return "ptr"
    return "other"


def type_candidate_items(attrs):
    items = []
    for k in [
        "type_candidates", "pred_type_candidates_extended", "type_category_candidates",
        "param_type_categories", "param_types", "return_type_category",
        "pred_type_category", "pred_type", "type_category",
    ]:
        v = attrs.get(k)
        if isinstance(v, list):
            items.extend(v[:80])
        elif isinstance(v, dict):
            items.extend(v.keys())
        elif v:
            items.append(v)
    # nested TypeSem parameter_type_sem
    pts = attrs.get("parameter_type_sem")
    if isinstance(pts, list):
        for p in pts[:80]:
            if isinstance(p, dict):
                for k in ["pred_type_category", "type_category", "pred_type", "category"]:
                    if p.get(k):
                        items.append(p.get(k))
                for k in ["type_candidates", "type_category_candidates"]:
                    if isinstance(p.get(k), list):
                        items.extend(p.get(k)[:20])
    return items


def type_distribution(attrs):
    cats = Counter()
    items = type_candidate_items(attrs)
    for x in items:
        cats[canonical_type_cat(x)] += 1

    total = sum(cats.values()) or 1
    return {
        "ptr": cats["ptr"] / total,
        "int": cats["int"] / total,
        "float": cats["float"] / total,
        "struct": cats["struct"] / total,
        "unknown": cats["unknown"] / total,
    }


def offset_values(attrs):
    vals = []
    for k in ["offsets", "field_offsets", "observed_field_offsets", "inferred_field_offsets", "eval_field_offsets", "offset_set"]:
        v = attrs.get(k)
        if isinstance(v, list):
            for x in v[:200]:
                if isinstance(x, dict):
                    for kk in ["offset", "field_offset", "value"]:
                        fx = as_float(x.get(kk))
                        if fx is not None:
                            vals.append(fx)
                else:
                    fx = as_float(x)
                    if fx is not None:
                        vals.append(fx)
    return vals


def count_text_occurs(attrs, words):
    text = json.dumps(attrs, ensure_ascii=False).lower()
    return sum(1 for w in words if w in text)


def is_call_like(attrs):
    text = json.dumps(attrs, ensure_ascii=False).lower()
    return str(attrs.get("opcode", "")).upper() == "CALL" or "call" in text or "callee" in text


def opcode_token(attrs):
    v = attrs.get("opcode") or attrs.get("op") or attrs.get("mnemonic")
    if v:
        return str(v).upper()
    text = json.dumps(attrs, ensure_ascii=False).lower()
    for op in ["load", "store", "call", "branch", "ret", "cmp", "add", "sub", "mul", "div", "mov"]:
        if op in text:
            return op.upper()
    return ""


def collect_read_write(attrs):
    reads = []
    writes = []
    for k in ["read_set", "reads", "input_regs", "use_set", "uses", "src_regs"]:
        v = attrs.get(k)
        if isinstance(v, list):
            reads.extend(str(x) for x in v[:100])
        elif v:
            reads.append(str(v))
    for k in ["write_set", "writes", "output_regs", "def_set", "defs", "dst_regs"]:
        v = attrs.get(k)
        if isinstance(v, list):
            writes.extend(str(x) for x in v[:100])
        elif v:
            writes.append(str(v))
    return reads, writes


def collect_anchor_degrees(neighborhoods_path):
    deg = defaultdict(lambda: {"degree": 0, "source": 0, "target": 0})
    p = Path(neighborhoods_path) if neighborhoods_path else None
    if not p or not p.exists():
        return deg
    for a in load_jsonl(p):
        src = a.get("source_node")
        tgt = a.get("target_node")
        if src:
            deg[src]["degree"] += 1
            deg[src]["source"] += 1
        if tgt:
            deg[tgt]["degree"] += 1
            deg[tgt]["target"] += 1
        for side in ["low_nodes", "high_nodes"]:
            for n in a.get(side, [])[:300]:
                nid = n.get("node_id")
                if nid:
                    deg[nid]["degree"] += 0.1
    return deg


def general_numeric(n, anchor_degrees):
    attrs = get_attrs(n)
    nid = n.get("node_id", "")
    num_list = sum(1 for v in attrs.values() if isinstance(v, list))
    num_scalar = sum(1 for v in attrs.values() if isinstance(v, (str, int, float, bool, type(None))))
    cmax, cmean = confidence_stats(attrs)
    ad = anchor_degrees.get(nid, {})
    return {
        "num_list_fields": norm01(num_list, 5.0),
        "num_scalar_fields": norm01(num_scalar, 10.0),
        "confidence_max": cmax,
        "confidence_mean": cmean,
        "eval_exclude": 1.0 if attrs.get("eval_exclude") else 0.0,
        "has_function_entry": 1.0 if (n.get("function_entry") or attrs.get("function_entry") or attrs.get("func_entry")) else 0.0,
        "has_location": 1.0 if attrs.get("location") or attrs.get("address") or attrs.get("addr") else 0.0,
        "anchor_degree": norm01(ad.get("degree", 0), 20.0),
        "source_anchor_count": norm01(ad.get("source", 0), 20.0),
        "target_anchor_count": norm01(ad.get("target", 0), 20.0),
    }


def low_numeric(n):
    attrs = get_attrs(n)
    nt = node_type(n)
    op = opcode_token(attrs)
    reads, writes = collect_read_write(attrs)
    offsets = offset_values(attrs)

    text = json.dumps(attrs, ensure_ascii=False).lower()
    is_load = op == "LOAD" or "load" in text
    is_store = op == "STORE" or "store" in text
    is_call = op == "CALL" or is_call_like(attrs)
    is_branch = op in {"BRANCH", "CBRANCH"} or "branch" in text or "jump" in text
    is_ret = op in {"RET", "RETURN"} or "ret" in text

    instr_count = attrs.get("instr_count") or attrs.get("instruction_count") or list_len(attrs.get("instructions"))
    load_count = attrs.get("load_count") or count_text_occurs(attrs, ["load"])
    store_count = attrs.get("store_count") or count_text_occurs(attrs, ["store"])
    call_count = attrs.get("call_count") or count_text_occurs(attrs, ["call"])
    branch_count = attrs.get("branch_count") or count_text_occurs(attrs, ["branch", "jump"])

    mem_count = attrs.get("memory_access_count") or attrs.get("mem_access_count") or count_text_occurs(attrs, ["mem", "[", "load", "store"])

    return {
        "is_instr": 1.0 if nt == "InstrSem" else 0.0,
        "is_block": 1.0 if nt == "BlockSem" else 0.0,
        "is_load": 1.0 if is_load else 0.0,
        "is_store": 1.0 if is_store else 0.0,
        "is_call": 1.0 if is_call else 0.0,
        "is_branch": 1.0 if is_branch else 0.0,
        "is_return": 1.0 if is_ret else 0.0,
        "has_memory_access": 1.0 if mem_count or offsets else 0.0,
        "memory_access_count": norm01(mem_count or 0, 10.0),
        "read_set_count": norm01(len(reads), 8.0),
        "write_set_count": norm01(len(writes), 8.0),
        "operand_count": norm01(attrs.get("operand_count") or list_len(attrs.get("operands")), 8.0),
        "constant_count": norm01(attrs.get("constant_count") or count_text_occurs(attrs, ["const", "0x"]), 8.0),
        "offset_count": norm01(len(offsets), 8.0),
        "stack_access_count": norm01(attrs.get("stack_access_count") or count_text_occurs(attrs, ["sp", "rsp", "rbp", "stack"]), 8.0),
        "instr_count": norm01(instr_count or 0, 10.0),
        "load_count": norm01(load_count or 0, 10.0),
        "store_count": norm01(store_count or 0, 10.0),
        "call_count": norm01(call_count or 0, 10.0),
        "branch_count": norm01(branch_count or 0, 10.0),
        "opcode_ratio_mem": float(mem_count or 0) / max(1.0, float(instr_count or 1)),
        "opcode_ratio_call": float(call_count or 0) / max(1.0, float(instr_count or 1)),
        "opcode_ratio_branch": float(branch_count or 0) / max(1.0, float(instr_count or 1)),
    }


def mid_numeric(n):
    attrs = get_attrs(n)
    nt = node_type(n)
    text = json.dumps(attrs, ensure_ascii=False).lower()

    node_count = attrs.get("node_count") or list_len(attrs.get("nodes"))
    edge_count = attrs.get("edge_count") or list_len(attrs.get("edges"))
    in_degree = attrs.get("in_degree") or attrs.get("indegree") or list_len(attrs.get("predecessors"))
    out_degree = attrs.get("out_degree") or attrs.get("outdegree") or list_len(attrs.get("successors"))

    def_use_count = attrs.get("def_use_count") or attrs.get("du_count") or count_text_occurs(attrs, ["def_use", "def-use", "use"])
    entry_input_count = attrs.get("entry_input_count") or list_len(attrs.get("entry_input_locs")) or count_text_occurs(attrs, ["entry_input"])
    memory_base_count = attrs.get("memory_base_count") or count_text_occurs(attrs, ["memory_base", "base_reg", "used_as_base"])
    deref_len = attrs.get("deref_chain_length") or attrs.get("deref_depth") or count_text_occurs(attrs, ["deref", "load"])

    return {
        "is_cfg": 1.0 if nt == "CFGSem" else 0.0,
        "is_dfg": 1.0 if nt == "DFGSem" else 0.0,
        "node_count": norm01(node_count or 0, 10.0),
        "edge_count": norm01(edge_count or 0, 10.0),
        "in_degree": norm01(in_degree or 0, 8.0),
        "out_degree": norm01(out_degree or 0, 8.0),
        "successor_count": norm01(attrs.get("successor_count") or list_len(attrs.get("successors")), 8.0),
        "predecessor_count": norm01(attrs.get("predecessor_count") or list_len(attrs.get("predecessors")), 8.0),
        "branch_count_mid": norm01(attrs.get("branch_count") or count_text_occurs(attrs, ["branch"]), 8.0),
        "loop_flag": 1.0 if attrs.get("loop_flag") or "loop" in text or "back_edge" in text else 0.0,
        "back_edge_count": norm01(attrs.get("back_edge_count") or count_text_occurs(attrs, ["back_edge"]), 8.0),
        "def_use_count": norm01(def_use_count or 0, 10.0),
        "source_count": norm01(attrs.get("source_count") or list_len(attrs.get("sources")), 8.0),
        "sink_count": norm01(attrs.get("sink_count") or list_len(attrs.get("sinks")), 8.0),
        "entry_input_count": norm01(entry_input_count or 0, 8.0),
        "memory_base_count": norm01(memory_base_count or 0, 8.0),
        "deref_chain_length": norm01(deref_len or 0, 8.0),
        "dataflow_depth": norm01(attrs.get("dataflow_depth") or attrs.get("dfg_depth") or 0, 8.0),
        "used_as_address_count": norm01(attrs.get("used_as_address_count") or count_text_occurs(attrs, ["used_as_address", "address_use"]), 8.0),
    }


def high_numeric(n):
    attrs = get_attrs(n)
    nt = node_type(n)
    items = type_candidate_items(attrs)
    dist = type_distribution(attrs)
    api_strings = collect_api_strings(attrs)
    families = [api_family(x) for x in api_strings if api_family(x)]
    offsets = offset_values(attrs)
    observed_offsets = []
    for k in ["observed_field_offsets", "eval_field_offsets"]:
        v = attrs.get(k)
        if isinstance(v, list):
            observed_offsets.extend(v)

    param_count = attrs.get("param_count") or attrs.get("declared_param_count") or list_len(attrs.get("params"))
    param_types = attrs.get("param_type_categories") or attrs.get("param_types") or []
    return_type = attrs.get("return_type_category") or attrs.get("return_type") or attrs.get("ret")

    min_off = min(offsets) if offsets else 0
    max_off = max(offsets) if offsets else 0
    stride_like = 0.0
    if len(offsets) >= 3:
        so = sorted(set(offsets))
        diffs = [so[i+1] - so[i] for i in range(len(so)-1)]
        if diffs:
            common = Counter(diffs).most_common(1)[0][1]
            stride_like = common / len(diffs)

    return {
        "is_type": 1.0 if nt == "TypeSem" else 0.0,
        "is_interface": 1.0 if nt == "InterfaceSem" else 0.0,
        "is_api": 1.0 if nt == "APISem" else 0.0,
        "is_layout": 1.0 if nt == "LayoutSem" else 0.0,
        "candidate_count": norm01(len(items), 8.0),
        "param_count": norm01(param_count or 0, 8.0),
        "param_type_count": norm01(len(param_types) if isinstance(param_types, list) else 0, 8.0),
        "return_known": 0.0 if canonical_type_cat(return_type) == "unknown" else 1.0,
        "pointer_score": dist["ptr"],
        "int_score": dist["int"],
        "float_score": dist["float"],
        "struct_score": dist["struct"],
        "unknown_score": dist["unknown"],
        "api_name_present": 1.0 if api_strings else 0.0,
        "api_family_present": 1.0 if families else 0.0,
        "semantic_template_present": 1.0 if attrs.get("semantic_template") or attrs.get("api_template") or attrs.get("api_aware_semantic_template") else 0.0,
        "argument_role_count": norm01(list_len(attrs.get("argument_roles")) or list_len(attrs.get("api_roles")), 8.0),
        "return_role_present": 1.0 if attrs.get("return_role") else 0.0,
        "offset_count_high": norm01(len(offsets), 10.0),
        "field_offset_count": norm01(list_len(attrs.get("field_offsets")), 10.0),
        "observed_field_offset_count": norm01(len(observed_offsets), 10.0),
        "min_offset_norm": norm01(abs(min_off), 12.0),
        "max_offset_norm": norm01(abs(max_off), 12.0),
        "stride_like": stride_like,
        "layout_array_like": 1.0 if attrs.get("layout_array_like") else 0.0,
    }


def numeric_features(n, anchor_degrees):
    vals = {}
    vals.update(general_numeric(n, anchor_degrees))
    vals.update(low_numeric(n))
    vals.update(mid_numeric(n))
    vals.update(high_numeric(n))
    return [float(vals.get(k, 0.0)) for k in NUMERIC_NAMES]


def category_onehot(n, node_type_to_idx, layer_to_idx):
    nt_vec = [0.0] * len(NODE_TYPES)
    ly_vec = [0.0] * len(LAYERS)
    nt = node_type(n)
    ly = layer_of(n)
    nt_vec[node_type_to_idx.get(nt, node_type_to_idx["unknown"])] = 1.0
    ly_vec[layer_to_idx.get(ly, layer_to_idx["unknown"])] = 1.0
    return nt_vec + ly_vec


def low_tokens(n):
    attrs = get_attrs(n)
    tokens = []
    op = opcode_token(attrs)
    if op:
        tokens.append("low_opcode:" + op)
    reads, writes = collect_read_write(attrs)
    for r in reads[:50]:
        tokens.append("read:" + str(r))
    for w in writes[:50]:
        tokens.append("write:" + str(w))
    for k in ["operands", "operand_types", "base_reg", "index_reg", "scale", "offset", "address", "location"]:
        if k in attrs:
            tokens.extend(flatten_tokens(attrs[k], prefix=f"{k}:"))
    for s in collect_api_strings(attrs):
        ns = normalize_api_name(s)
        if ns:
            tokens.append("call_api:" + ns)
            fam = api_family(ns)
            if fam:
                tokens.append("call_api_family:" + fam)
    return tokens


def mid_tokens(n):
    attrs = get_attrs(n)
    tokens = []
    for k in [
        "edge_type", "cfg_edge_type", "branch_type", "predicate",
        "def_use", "def", "use", "source", "sink",
        "entry_input_locs", "memory_bases", "alias_groups",
        "deref_chain", "dataflow_path",
    ]:
        if k in attrs:
            tokens.extend(flatten_tokens(attrs[k], prefix=f"mid_{k}:"))
    return tokens


def high_tokens(n):
    attrs = get_attrs(n)
    tokens = []
    nt = node_type(n)
    tokens.append("high_node_type:" + nt)

    # Type/Interface candidates
    for x in type_candidate_items(attrs):
        cat = canonical_type_cat(x)
        tokens.append("type_candidate:" + str(x))
        tokens.append("type_cat:" + cat)

    # API tokens
    for s in collect_api_strings(attrs):
        ns = normalize_api_name(s)
        if ns:
            tokens.append("api_name:" + ns)
            fam = api_family(ns)
            if fam:
                tokens.append("api_family:" + fam)

    for k in ["semantic_template", "api_template", "api_aware_semantic_template", "argument_roles", "return_role"]:
        if k in attrs:
            tokens.extend(flatten_tokens(attrs[k], prefix=f"api_{k}:"))

    # Layout tokens
    for k in ["layout_kind", "offsets", "field_offsets", "observed_field_offsets", "eval_field_offsets", "base_object", "base_storage"]:
        if k in attrs:
            tokens.extend(flatten_tokens(attrs[k], prefix=f"layout_{k}:"))

    return tokens


def generic_tokens(n):
    attrs = get_attrs(n)
    tokens = [
        "node_type=" + node_type(n),
        "layer=" + layer_of(n),
    ]
    for k in [
        "category", "sem_type", "eval_exclude_reason",
        "function_entry", "func_entry", "function_name", "func_name",
        "raw_id", "location", "address",
    ]:
        if k in attrs:
            tokens.extend(flatten_tokens(attrs[k], prefix=f"{k}:"))
    fe = n.get("function_entry") or n.get("function_entry_norm") or attrs.get("function_entry") or attrs.get("func_entry")
    if fe:
        tokens.append("function_entry=" + str(fe))
    return tokens


def hashed_vec(tokens, dim):
    vec = [0.0] * dim
    if dim <= 0:
        return vec
    for tok in tokens:
        idx = hash_index(tok, dim)
        vec[idx] += 1.0
    for i, x in enumerate(vec):
        if x:
            vec[i] = math.log1p(x)
    return vec


def hierarchical_hash_features(n, dim_low, dim_mid, dim_high, dim_generic):
    nt = node_type(n)
    lv = [0.0] * dim_low
    mv = [0.0] * dim_mid
    hv = [0.0] * dim_high
    gv = hashed_vec(generic_tokens(n), dim_generic)

    if nt in LOW_TYPES:
        lv = hashed_vec(low_tokens(n), dim_low)
    elif nt in MID_TYPES:
        mv = hashed_vec(mid_tokens(n), dim_mid)
    elif nt in HIGH_TYPES:
        hv = hashed_vec(high_tokens(n), dim_high)
    else:
        # unknown/stub 同时使用通用字段和少量 high/low 字段，避免完全空向量
        lv = hashed_vec(low_tokens(n), dim_low)
        hv = hashed_vec(high_tokens(n), dim_high)

    return lv + mv + hv + gv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--neighborhoods", default="")
    ap.add_argument("--out-pt", required=True)
    ap.add_argument("--vocab-out", required=True)
    ap.add_argument("--node-feature-jsonl", default="")
    ap.add_argument("--hash-dim-low", type=int, default=192)
    ap.add_argument("--hash-dim-mid", type=int, default=128)
    ap.add_argument("--hash-dim-high", type=int, default=192)
    ap.add_argument("--hash-dim-generic", type=int, default=64)
    args = ap.parse_args()

    nodes_path = Path(args.nodes).resolve()
    out_pt = Path(args.out_pt).resolve()
    vocab_out = Path(args.vocab_out).resolve()
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    vocab_out.parent.mkdir(parents=True, exist_ok=True)
    if args.node_feature_jsonl:
        Path(args.node_feature_jsonl).parent.mkdir(parents=True, exist_ok=True)

    node_type_to_idx = {x: i for i, x in enumerate(NODE_TYPES)}
    layer_to_idx = {x: i for i, x in enumerate(LAYERS)}
    anchor_degrees = collect_anchor_degrees(args.neighborhoods)

    node_ids = []
    features = []
    rows = []
    node_type_cnt = Counter()
    layer_cnt = Counter()
    api_family_cnt = Counter()

    for n in load_jsonl(nodes_path):
        nid = n.get("node_id")
        if not nid:
            continue
        node_ids.append(nid)
        node_type_cnt[node_type(n)] += 1
        layer_cnt[layer_of(n)] += 1

        attrs = get_attrs(n)
        for s in collect_api_strings(attrs):
            fam = api_family(s)
            if fam:
                api_family_cnt[fam] += 1

        vec = []
        vec.extend(category_onehot(n, node_type_to_idx, layer_to_idx))
        vec.extend(numeric_features(n, anchor_degrees))
        vec.extend(hierarchical_hash_features(
            n,
            args.hash_dim_low,
            args.hash_dim_mid,
            args.hash_dim_high,
            args.hash_dim_generic,
        ))
        features.append(vec)

        if args.node_feature_jsonl:
            rows.append({
                "node_id": nid,
                "sample_id": n.get("sample_id", ""),
                "node_type": node_type(n),
                "layer": layer_of(n),
                "feature": vec,
            })

    node_index = {nid: i for i, nid in enumerate(node_ids)}
    feature_dim = len(features[0]) if features else (
        len(NODE_TYPES) + len(LAYERS) + len(NUMERIC_NAMES)
        + args.hash_dim_low + args.hash_dim_mid + args.hash_dim_high + args.hash_dim_generic
    )

    backend = "pickle"
    try:
        import torch
        tensor = torch.tensor(features, dtype=torch.float32)
        payload = {
            "node_ids": node_ids,
            "node_index": node_index,
            "features": tensor,
            "feature_dim": feature_dim,
            "backend": "torch",
            "feature_schema": "hierarchical_v1",
        }
        torch.save(payload, out_pt)
        backend = "torch"
    except Exception:
        payload = {
            "node_ids": node_ids,
            "node_index": node_index,
            "features": features,
            "feature_dim": feature_dim,
            "backend": "pickle",
            "feature_schema": "hierarchical_v1",
        }
        with out_pt.open("wb") as f:
            pickle.dump(payload, f)

    if args.node_feature_jsonl:
        with Path(args.node_feature_jsonl).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    prefix_dim = len(NODE_TYPES) + len(LAYERS)
    num_start = prefix_dim
    num_end = num_start + len(NUMERIC_NAMES)
    low_start = num_end
    low_end = low_start + args.hash_dim_low
    mid_start = low_end
    mid_end = mid_start + args.hash_dim_mid
    high_start = mid_end
    high_end = high_start + args.hash_dim_high
    gen_start = high_end
    gen_end = gen_start + args.hash_dim_generic

    vocab = {
        "nodes": str(nodes_path),
        "neighborhoods": str(args.neighborhoods),
        "out_pt": str(out_pt),
        "backend": backend,
        "feature_schema": "hierarchical_v1",
        "num_nodes": len(node_ids),
        "feature_dim": feature_dim,
        "node_type_to_idx": node_type_to_idx,
        "layer_to_idx": layer_to_idx,
        "numeric_feature_names": NUMERIC_NAMES,
        "hash_dims": {
            "low": args.hash_dim_low,
            "mid": args.hash_dim_mid,
            "high": args.hash_dim_high,
            "generic": args.hash_dim_generic,
        },
        "feature_layout": {
            "node_type_onehot": [0, len(NODE_TYPES)],
            "layer_onehot": [len(NODE_TYPES), prefix_dim],
            "numeric": [num_start, num_end],
            "low_hashed_instr_block": [low_start, low_end],
            "mid_hashed_cfg_dfg": [mid_start, mid_end],
            "high_hashed_type_interface_api_layout": [high_start, high_end],
            "generic_hashed": [gen_start, gen_end],
        },
        "node_type_counts": dict(node_type_cnt),
        "layer_counts": dict(layer_cnt),
        "api_family_counts": dict(api_family_cnt),
        "note": "Hierarchical bridge mapping: low/mid/high semantic objects are encoded with different feature blocks, but output remains compatible with existing training scripts."
    }
    vocab_out.write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")

    print("wrote", out_pt)
    print("wrote", vocab_out)
    if args.node_feature_jsonl:
        print("wrote", args.node_feature_jsonl)
    print(json.dumps({
        "num_nodes": len(node_ids),
        "feature_dim": feature_dim,
        "backend": backend,
        "feature_schema": "hierarchical_v1",
        "out_pt": str(out_pt),
        "vocab_out": str(vocab_out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
