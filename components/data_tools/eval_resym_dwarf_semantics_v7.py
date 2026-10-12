#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DWARF evaluator v7 for ReSym USG outputs.

Improvements over first-pass evaluator:
1) Prefer explicit full_path from runner meta manifest_row.full_path.
2) Map InterfaceSem to DWARF subprograms by function entry address range when possible.
3) Evaluate parameter count, return type category, parameter type category.
4) Evaluate TypeSem parameter type categories using function_entry + param_index.
5) Evaluate LayoutSem offset Precision/Recall/F1 by raw, merged and observed-offset variants.
6) Use one-to-one InterfaceSem↔DWARF function matching to avoid duplicate gold mappings.

Install dependency:
python3 -m pip install --user pyelftools

Usage:
python3 scripts/eval_resym_dwarf_semantics_v3.py \
  --json-root usg_outputs/c80_10_v2 \
  --full-root samples_c80/full \
  --out reports/c80_10_v3_eval.csv \
  --summary reports/c80_10_v3_eval_summary.json
"""
import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

try:
    from elftools.elf.elffile import ELFFile
    from elftools.dwarf.descriptions import describe_form_class
except Exception as e:
    raise SystemExit("Missing pyelftools. Install with: python3 -m pip install --user pyelftools\n" + str(e))


def hex_to_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if not s:
        return None
    # Keep final hex-like address if strings contain labels.
    m = re.search(r'0x[0-9a-fA-F]+|[0-9a-fA-F]{4,}', s)
    if not m:
        return None
    try:
        return int(m.group(0), 16)
    except Exception:
        try:
            return int(s, 0)
        except Exception:
            return None


def norm_type(s):
    s = "" if s is None else str(s).lower()
    s = re.sub(r'\b(const|volatile|restrict|static|register)\b', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def type_cat(x):
    """Normalize type string/list/category into evaluator categories."""
    if isinstance(x, (list, tuple)):
        for item in x:
            c = type_cat(item)
            if c != 'unknown':
                return c
        return 'unknown'
    s = norm_type(x)
    if not s or s in ('unknown', 'base', 'default', 'none', 'null'):
        return 'unknown'
    if s in ('undefined1', 'undefined2', 'undefined4', 'undefined8'):
        return 'int'
    if s == 'undefined':
        return 'unknown'
    if '*' in s or 'ptr' in s or 'pointer' in s or 'pointer_like' in s:
        return 'ptr'
    if 'struct' in s or 'class' in s:
        return 'struct'
    if 'union' in s:
        return 'union'
    if 'enum' in s:
        return 'enum'
    if 'float' in s or 'double' in s:
        return 'float'
    if s == 'void' or s.startswith('void '):
        return 'void'
    if any(w in s for w in ['int', 'long', 'short', 'char', 'bool', 'size_t', 'ssize_t', 'uint', 'byte', 'dword', 'qword']):
        return 'int'
    return 'other'



RUNTIME_NAME_SUBSTRINGS = [
    'tm_clones', 'frame_dummy', 'global_dtors', '.plt', '@plt',
    'libc_start', 'gmon_start', '__cxa_finalize', 'stack_chk',
    'asan_', 'ubsan_', 'tsan_', 'msan_', '__sanitizer',
    'init_array', 'fini_array'
]
RUNTIME_NAME_EXACT = set([
    '_init', '_fini', '_start', '__libc_start_main', '__stack_chk_fail',
    'deregister_tm_clones', 'register_tm_clones', 'frame_dummy', '__do_global_dtors_aux',
    '__x86.get_pc_thunk.bx', '__x86.get_pc_thunk.cx', '__x86.get_pc_thunk.dx', '__x86.get_pc_thunk.ax'
])

def runtime_name_filter(name):
    s = norm_type(name)
    if not s:
        return False
    if s in RUNTIME_NAME_EXACT:
        return True
    return any(x in s for x in RUNTIME_NAME_SUBSTRINGS)


def first_non_unknown(*vals):
    for v in vals:
        c = type_cat(v)
        if c != 'unknown':
            return c
    return 'unknown'

def attr_str(die, key):
    a = die.attributes.get(key)
    if not a:
        return None
    v = a.value
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='replace')
    return str(v)


def ref_die(die, cu, by_off):
    a = die.attributes.get('DW_AT_type')
    if not a:
        return None
    if describe_form_class(a.form) != 'reference':
        return None
    ref = a.value
    for off in [cu.cu_offset + ref, ref]:
        if off in by_off:
            return by_off[off]
    return None


def die_type_name(die, cu, by_off, seen=None):
    if die is None:
        return 'unknown'
    if seen is None:
        seen = set()
    if die.offset in seen:
        return 'recursive'
    seen.add(die.offset)
    tag = die.tag
    name = attr_str(die, 'DW_AT_name')
    if tag == 'DW_TAG_base_type':
        return name or 'base'
    if tag == 'DW_TAG_pointer_type':
        return die_type_name(ref_die(die, cu, by_off), cu, by_off, seen) + '*'
    if tag == 'DW_TAG_const_type':
        return 'const ' + die_type_name(ref_die(die, cu, by_off), cu, by_off, seen)
    if tag == 'DW_TAG_typedef':
        return name or die_type_name(ref_die(die, cu, by_off), cu, by_off, seen)
    if tag == 'DW_TAG_array_type':
        return die_type_name(ref_die(die, cu, by_off), cu, by_off, seen) + '[]'
    if tag == 'DW_TAG_structure_type':
        return 'struct ' + (name or f'anon_{die.offset:x}')
    if tag == 'DW_TAG_union_type':
        return 'union ' + (name or f'anon_{die.offset:x}')
    if tag == 'DW_TAG_enumeration_type':
        return 'enum ' + (name or f'anon_{die.offset:x}')
    return name or tag.replace('DW_TAG_', '')


def underlying_struct_offsets(die, cu, by_off, seen=None):
    """Return struct member offset tuple if type resolves to/passes through a structure."""
    if die is None:
        return None
    if seen is None:
        seen = set()
    if die.offset in seen:
        return None
    seen.add(die.offset)
    if die.tag == 'DW_TAG_structure_type':
        offs = []
        for ch in die.iter_children():
            if ch.tag == 'DW_TAG_member':
                a = ch.attributes.get('DW_AT_data_member_location')
                if a and isinstance(a.value, int):
                    offs.append(a.value)
        return tuple(sorted(set(offs))) if offs else None
    if die.tag in ['DW_TAG_pointer_type', 'DW_TAG_const_type', 'DW_TAG_typedef', 'DW_TAG_array_type']:
        return underlying_struct_offsets(ref_die(die, cu, by_off), cu, by_off, seen)
    return None


def high_pc_value(die, low_pc):
    hp = die.attributes.get('DW_AT_high_pc')
    if not hp:
        return None
    if describe_form_class(hp.form) == 'address':
        return int(hp.value)
    # constant offset from low_pc
    try:
        return int(low_pc) + int(hp.value)
    except Exception:
        return None


def extract_dwarf(full_path):
    gold = {'has_dwarf': False, 'functions': [], 'global_struct_offsets': []}
    with open(full_path, 'rb') as f:
        elf = ELFFile(f)
        if not elf.has_dwarf_info():
            return gold
        dwarf = elf.get_dwarf_info()
        gold['has_dwarf'] = True
        for cu in dwarf.iter_CUs():
            by_off = {d.offset: d for d in cu.iter_DIEs()}
            for die in cu.iter_DIEs():
                if die.tag == 'DW_TAG_structure_type':
                    offs = underlying_struct_offsets(die, cu, by_off)
                    if offs:
                        gold['global_struct_offsets'].append(offs)
                if die.tag != 'DW_TAG_subprogram':
                    continue
                name = attr_str(die, 'DW_AT_name') or attr_str(die, 'DW_AT_linkage_name') or ''
                lp = die.attributes.get('DW_AT_low_pc')
                if not lp:
                    continue
                low = int(lp.value)
                high = high_pc_value(die, low)
                params = []
                local_structs = []
                for ch in die.iter_children():
                    if ch.tag == 'DW_TAG_formal_parameter':
                        tdie = ref_die(ch, cu, by_off)
                        tname = die_type_name(tdie, cu, by_off)
                        params.append(type_cat(tname))
                        st = underlying_struct_offsets(tdie, cu, by_off)
                        if st:
                            local_structs.append(st)
                    elif ch.tag == 'DW_TAG_variable':
                        tdie = ref_die(ch, cu, by_off)
                        st = underlying_struct_offsets(tdie, cu, by_off)
                        if st:
                            local_structs.append(st)
                ret = ref_die(die, cu, by_off)
                ret_cat = type_cat(die_type_name(ret, cu, by_off) if ret else 'void')
                gold['functions'].append({
                    'name': name,
                    'low_pc': low,
                    'high_pc': high,
                    'param_cats': params,
                    'param_count': len(params),
                    'return_cat': ret_cat,
                    'struct_offsets': list(sorted(set(local_structs))),
                    'eval_exclude': runtime_name_filter(name),
                })
    return gold


def find_gold_func(gold, entry):
    if entry is None:
        return None
    for g in gold['functions']:
        if g.get('eval_exclude'):
            continue
        if g['high_pc'] is not None and g['low_pc'] <= entry < g['high_pc']:
            return g
    # fallback exact low_pc
    for g in gold['functions']:
        if g.get('eval_exclude'):
            continue
        if g['low_pc'] == entry:
            return g
    return None


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def infer_full_path(json_path, data, full_root):
    sid = json_path.name.replace('_semantic_dict.json', '')
    meta = json_path.parent / f'{sid}.meta.json'
    if meta.exists():
        try:
            m = load_json(meta)
            row = m.get('manifest_row', {})
            fp = row.get('full_path')
            if fp and Path(fp).exists():
                return Path(fp)
            sp = row.get('stripped_path') or m.get('binary')
            if sp and '/stripped/' in str(sp):
                fp2 = str(sp).replace('/stripped/', '/full/')
                if Path(fp2).exists():
                    return Path(fp2)
        except Exception:
            pass
    pname = data.get('program_info', {}).get('name')
    if pname:
        hits = list(Path(full_root).rglob(pname))
        if hits:
            return hits[0]
    return None


def pred_interfaces(data):
    hs = data.get('high_level_sem', {}) if isinstance(data, dict) else {}
    out = []
    for itf in hs.get('interface_sem', []) or []:
        if not isinstance(itf, dict):
            continue
        cons = itf.get('interface_constraints', {}) if isinstance(itf.get('interface_constraints', {}), dict) else {}
        ret_obj = itf.get('ret', {}) if isinstance(itf.get('ret', {}), dict) else {}
        entry = hex_to_int(itf.get('func_entry') or itf.get('function_entry'))
        name = itf.get('function_name') or itf.get('func_name') or ''
        eval_exclude = bool(itf.get('eval_exclude')) or runtime_name_filter(name)
        params = itf.get('params') or []
        # Support v9 output and older nested structures.
        if not isinstance(params, list):
            params = []
        param_count = itf.get('param_count')
        if param_count is None:
            param_count = cons.get('param_count')
        if param_count is None:
            param_count = len(params) if isinstance(params, list) else None

        param_cats = itf.get('param_type_categories')
        if not isinstance(param_cats, list):
            param_cats = []
        if not param_cats:
            for p in params:
                if isinstance(p, dict):
                    cat = first_non_unknown(
                        p.get('pred_type_category'),
                        p.get('type_category'),
                        p.get('category'),
                        p.get('pred_type'),
                        p.get('return_type'),
                        p.get('type_candidates'),
                    )
                    param_cats.append(cat)
        # If param_count is larger than available param categories, pad with unknown so
        # coverage is measured honestly instead of silently ignored.
        try:
            pc_int = int(param_count) if param_count is not None else len(param_cats)
            while len(param_cats) < pc_int:
                param_cats.append('unknown')
        except Exception:
            pass

        ret_cat = first_non_unknown(
            itf.get('return_type_category'),
            itf.get('return_type'),
            ret_obj.get('return_type_category'),
            ret_obj.get('return_type'),
            ret_obj.get('type_candidates'),
        )
        has_return = itf.get('has_return')
        if has_return is None:
            has_return = cons.get('has_return')
        if has_return is False:
            ret_cat = 'void'

        out.append({
            'entry': entry,
            'param_count': param_count,
            'return_cat': ret_cat,
            'param_cats': [str(x) for x in param_cats],
            'name': name,
            'eval_exclude': eval_exclude,
        })
    return out


def pred_type_params(data):
    hs = data.get('high_level_sem', {}) if isinstance(data, dict) else {}
    out = []
    for ts in hs.get('type_sem', []) or []:
        if not isinstance(ts, dict):
            continue
        entry = hex_to_int(ts.get('function_entry') or ts.get('func_entry'))
        name = ts.get('function_name') or ts.get('func_name') or ''
        eval_exclude = bool(ts.get('eval_exclude')) or runtime_name_filter(name)
        nested = ts.get('parameter_type_sem', []) or []
        if nested:
            for idx, p in enumerate(nested):
                if not isinstance(p, dict):
                    continue
                pi = p.get('param_index', idx)
                cat = first_non_unknown(
                    p.get('pred_type_category'),
                    p.get('type_category'),
                    p.get('category'),
                    p.get('pred_type'),
                    p.get('type_candidates'),
                )
                out.append({'entry': entry, 'param_index': pi, 'cat': cat, 'name': name, 'eval_exclude': eval_exclude})
        else:
            cats = ts.get('param_type_categories') or []
            if isinstance(cats, list):
                for idx, cat0 in enumerate(cats):
                    out.append({'entry': entry, 'param_index': idx, 'cat': type_cat(cat0), 'name': name, 'eval_exclude': eval_exclude})
    return out


def pred_layouts(data):
    hs = data.get('high_level_sem', {}) if isinstance(data, dict) else {}
    out = []
    for ml in hs.get('mem_layout_sem', []) or hs.get('layout_sem', []) or []:
        if not isinstance(ml, dict):
            continue
        # Stack-frame layout is not a DWARF struct-member layout; keep it out of struct F1.
        if ml.get('layout_kind') == 'stack_frame_layout':
            continue
        entry = hex_to_int(ml.get('func_entry') or ml.get('function_entry'))
        name = ml.get('function_name') or ml.get('func_name') or ''
        if bool(ml.get('eval_exclude')) or runtime_name_filter(name):
            continue

        # v11/v12 may provide several offset variants:
        #   field_offsets / offsets       : final predicted offsets, possibly with closure
        #   observed_field_offsets        : offsets directly observed in binary evidence
        #   inferred_field_offsets        : offsets inferred by closure/merge
        #   eval_field_offsets            : optional explicit evaluator preference
        def nums_from(v):
            nums = []
            if isinstance(v, list):
                for x in v:
                    try:
                        n = int(x)
                        if 0 <= n < 4096:
                            nums.append(n)
                    except Exception:
                        pass
            return tuple(sorted(set(nums)))

        final = nums_from(ml.get('eval_field_offsets'))
        if not final:
            final = nums_from(ml.get('field_offsets'))
        if not final:
            final = nums_from(ml.get('offsets'))
        if not final:
            final = nums_from(ml.get('offset_set'))

        observed = nums_from(ml.get('observed_field_offsets'))
        if not observed:
            observed = nums_from(ml.get('field_offsets_observed'))
        if not observed:
            observed = final
        inferred = nums_from(ml.get('inferred_field_offsets'))
        if not inferred:
            inferred = nums_from(ml.get('field_offsets_inferred'))

        base = ml.get('base_object') or ml.get('base_storage') or ml.get('base_obj') or ml.get('location') or ml.get('id') or ''
        kind = ml.get('layout_kind') or ml.get('layout_source') or 'layout'
        out.append({
            'entry': entry,
            'offsets': final,
            'observed_offsets': observed,
            'inferred_offsets': inferred,
            'base': str(base),
            'kind': str(kind),
            'id': ml.get('id', ''),
        })
    return out

def prf(pred, gold):
    pred, gold = set(pred), set(gold)
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f1



def candidate_gold_indices(gold, entry):
    """Return candidate gold function indices for a predicted function entry."""
    if entry is None:
        return []
    hits = []
    for i, g in enumerate(gold['functions']):
        if g.get('eval_exclude'):
            continue
        if g['high_pc'] is not None and g['low_pc'] <= entry < g['high_pc']:
            hits.append(i)
    if hits:
        return hits
    for i, g in enumerate(gold['functions']):
        if g.get('eval_exclude'):
            continue
        if g['low_pc'] == entry:
            hits.append(i)
    return hits


def interface_match_score(pi, gf):
    """Score how well a predicted InterfaceSem matches a gold subprogram."""
    score = 0
    try:
        if pi.get('param_count') == gf.get('param_count'):
            score += 4
    except Exception:
        pass
    if pi.get('return_cat') == gf.get('return_cat'):
        score += 3
    pcats = pi.get('param_cats') or []
    gcats = gf.get('param_cats') or []
    for i, gcat in enumerate(gcats):
        if i < len(pcats) and pcats[i] == gcat:
            score += 2
        elif i < len(pcats) and pcats[i] != 'unknown':
            score += 0.25
    # Prefer known returns/params and avoid completely unknown predictions.
    if pi.get('return_cat') != 'unknown':
        score += 0.5
    score += 0.1 * sum(1 for c in pcats if c != 'unknown')
    return score


def match_interfaces_one_to_one(p_itfs, gold):
    """Greedy one-to-one matching between predicted InterfaceSem and DWARF subprograms."""
    candidates = []
    for pi_idx, pi in enumerate(p_itfs):
        for gi in candidate_gold_indices(gold, pi.get('entry')):
            gf = gold['functions'][gi]
            candidates.append((interface_match_score(pi, gf), pi_idx, gi))
    candidates.sort(reverse=True)
    used_p = set()
    used_g = set()
    matches = []
    for score, pi_idx, gi in candidates:
        if pi_idx in used_p or gi in used_g:
            continue
        used_p.add(pi_idx)
        used_g.add(gi)
        matches.append((pi_idx, gi, score))
    return matches


def best_layout_score(offsets, candidates):
    if not offsets:
        return (0.0, 0.0, 0.0)
    best = (0.0, 0.0, 0.0)
    for goffs in candidates or []:
        p0, r0, f10 = prf(offsets, goffs)
        if f10 > best[2]:
            best = (p0, r0, f10)
    return best


def layout_candidates_for(gold, entry):
    gf = find_gold_func(gold, entry)
    if gf and gf.get('struct_offsets'):
        return gf['struct_offsets']
    return gold.get('global_struct_offsets', [])


def merged_layouts_by_base(p_lays):
    groups = {}
    for pl in p_lays:
        key = (pl.get('entry'), pl.get('base') or '', pl.get('kind') or '')
        groups.setdefault(key, {'entry': pl.get('entry'), 'base': pl.get('base'), 'kind': pl.get('kind'),
                                'offsets': set(), 'observed_offsets': set(), 'inferred_offsets': set(), 'count': 0})
        g = groups[key]
        g['count'] += 1
        g['offsets'].update(pl.get('offsets') or [])
        g['observed_offsets'].update(pl.get('observed_offsets') or [])
        g['inferred_offsets'].update(pl.get('inferred_offsets') or [])
    out = []
    for g in groups.values():
        out.append({
            'entry': g['entry'],
            'base': g['base'],
            'kind': g['kind'],
            'offsets': tuple(sorted(g['offsets'])),
            'observed_offsets': tuple(sorted(g['observed_offsets'])),
            'inferred_offsets': tuple(sorted(g['inferred_offsets'])),
            'merge_count': g['count'],
        })
    return out


def summarize_layout_scores(layouts, gold, offset_key='offsets'):
    scores = []
    for pl in layouts:
        offs = pl.get(offset_key) or ()
        if not offs:
            continue
        cands = layout_candidates_for(gold, pl.get('entry'))
        scores.append(best_layout_score(offs, cands))
    if not scores:
        return None, None, None, 0
    return (sum(x[0] for x in scores)/len(scores),
            sum(x[1] for x in scores)/len(scores),
            sum(x[2] for x in scores)/len(scores),
            len(scores))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-root', required=True)
    ap.add_argument('--full-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--summary', required=True)
    args = ap.parse_args()

    rows = []
    status = Counter()
    global_counts = Counter()
    sums = Counter()

    for jp in sorted(Path(args.json_root).glob('*_semantic_dict.json')):
        data = load_json(jp)
        fp = infer_full_path(jp, data, args.full_root)
        if not fp:
            status['full_not_found'] += 1
            rows.append({'json': str(jp), 'status': 'full_not_found'})
            continue
        try:
            gold = extract_dwarf(fp)
        except Exception as e:
            status['dwarf_extract_error'] += 1
            rows.append({'json': str(jp), 'full_path': str(fp), 'status': 'dwarf_extract_error', 'error': str(e)})
            continue
        if not gold['has_dwarf']:
            status['no_dwarf'] += 1
            rows.append({'json': str(jp), 'full_path': str(fp), 'status': 'no_dwarf'})
            continue

        p_itfs = [x for x in pred_interfaces(data) if not x.get('eval_exclude')]
        p_tps = [x for x in pred_type_params(data) if not x.get('eval_exclude')]
        p_lays = pred_layouts(data)
        gold_eval_functions = [g for g in gold['functions'] if not g.get('eval_exclude')]

        # Interface metrics with one-to-one prediction↔gold matching.
        # v6 allowed several predicted InterfaceSem nodes to map to the same DWARF function,
        # which could inflate coverage and make precision hard to interpret. v7 fixes that.
        interface_matches = match_interfaces_one_to_one(p_itfs, gold)
        mapped_itf = len(interface_matches)
        param_count_ok = 0
        ret_ok = 0
        ret_known_total = 0
        ret_known_ok = 0
        param_type_total = 0
        param_type_ok = 0
        param_type_known_total = 0
        param_type_known_ok = 0
        for pi_idx, gi, _score in interface_matches:
            pi = p_itfs[pi_idx]
            gf = gold['functions'][gi]
            if pi['param_count'] == gf['param_count']:
                param_count_ok += 1
            if pi['return_cat'] == gf['return_cat']:
                ret_ok += 1
            if pi['return_cat'] != 'unknown':
                ret_known_total += 1
                if pi['return_cat'] == gf['return_cat']:
                    ret_known_ok += 1
            for i, gcat in enumerate(gf['param_cats']):
                if i < len(pi['param_cats']):
                    param_type_total += 1
                    if pi['param_cats'][i] == gcat:
                        param_type_ok += 1
                    if pi['param_cats'][i] != 'unknown':
                        param_type_known_total += 1
                        if pi['param_cats'][i] == gcat:
                            param_type_known_ok += 1

        # TypeSem parameter metrics
        type_param_total = 0
        type_param_ok = 0
        type_param_known_total = 0
        type_param_known_ok = 0
        for pp in p_tps:
            gf = find_gold_func(gold, pp['entry'])
            if not gf:
                continue
            idx = pp['param_index']
            try:
                idx = int(idx)
            except Exception:
                continue
            if 0 <= idx < len(gf['param_cats']):
                type_param_total += 1
                if pp['cat'] == gf['param_cats'][idx]:
                    type_param_ok += 1
                if pp['cat'] != 'unknown':
                    type_param_known_total += 1
                    if pp['cat'] == gf['param_cats'][idx]:
                        type_param_known_ok += 1

        # Layout offset metrics.
        # raw:     each predicted LayoutSem independently vs best DWARF struct
        # merged:  merge LayoutSem nodes with same function/base/kind before matching
        # observed: evaluate only directly observed offsets when v11/v12 provides them
        raw_p, raw_r, raw_f1, raw_n = summarize_layout_scores(p_lays, gold, offset_key='offsets')
        merged_lays = merged_layouts_by_base(p_lays)
        merged_p, merged_r, merged_f1, merged_n = summarize_layout_scores(merged_lays, gold, offset_key='offsets')
        obs_p, obs_r, obs_f1, obs_n = summarize_layout_scores(p_lays, gold, offset_key='observed_offsets')

        # Backward-compatible field names use the merged score because it is the fairer
        # measure when one structure/base is split into several LayoutSem candidates.
        layout_precision_main = merged_p if merged_p is not None else raw_p
        layout_recall_main = merged_r if merged_r is not None else raw_r
        layout_f1_main = merged_f1 if merged_f1 is not None else raw_f1
        layout_main_n = merged_n if merged_n else raw_n

        def ratio(a, b):
            return (a / b) if b else None

        row = {
            'json': str(jp),
            'full_path': str(fp),
            'status': 'ok',
            'gold_function_count': len(gold_eval_functions),
            'pred_interface_count': len(p_itfs),
            'mapped_interface_count': mapped_itf,
            'interface_param_count_acc': ratio(param_count_ok, mapped_itf),
            'interface_return_cat_acc': ratio(ret_ok, mapped_itf),
            'interface_return_cat_acc_known': ratio(ret_known_ok, ret_known_total),
            'interface_return_known_coverage': ratio(ret_known_total, mapped_itf),
            'interface_param_type_cat_acc': ratio(param_type_ok, param_type_total),
            'interface_param_type_cat_acc_known': ratio(param_type_known_ok, param_type_known_total),
            'interface_param_type_known_coverage': ratio(param_type_known_total, param_type_total),
            'type_param_cat_acc': ratio(type_param_ok, type_param_total),
            'type_param_cat_acc_known': ratio(type_param_known_ok, type_param_known_total),
            'type_param_known_coverage': ratio(type_param_known_total, type_param_total),
            'layout_pred_count': len(p_lays),
            'layout_mapped_count': layout_main_n,
            'layout_merged_count': len(merged_lays),
            'layout_offset_precision': layout_precision_main,
            'layout_offset_recall': layout_recall_main,
            'layout_offset_f1': layout_f1_main,
            'layout_raw_offset_precision': raw_p,
            'layout_raw_offset_recall': raw_r,
            'layout_raw_offset_f1': raw_f1,
            'layout_raw_count': raw_n,
            'layout_merged_offset_precision': merged_p,
            'layout_merged_offset_recall': merged_r,
            'layout_merged_offset_f1': merged_f1,
            'layout_observed_offset_precision': obs_p,
            'layout_observed_offset_recall': obs_r,
            'layout_observed_offset_f1': obs_f1,
            'layout_observed_count': obs_n,
        }
        rows.append(row)
        status['ok'] += 1

        for k in ['mapped_interface_count', 'pred_interface_count', 'gold_function_count', 'layout_pred_count', 'layout_mapped_count']:
            global_counts[k] += int(row[k] or 0)
        for k in ['interface_param_count_acc','interface_return_cat_acc','interface_return_cat_acc_known','interface_return_known_coverage','interface_param_type_cat_acc','interface_param_type_cat_acc_known','interface_param_type_known_coverage','type_param_cat_acc','type_param_cat_acc_known','type_param_known_coverage','layout_offset_precision','layout_offset_recall','layout_offset_f1','layout_raw_offset_precision','layout_raw_offset_recall','layout_raw_offset_f1','layout_merged_offset_precision','layout_merged_offset_recall','layout_merged_offset_f1','layout_observed_offset_precision','layout_observed_offset_recall','layout_observed_offset_f1']:
            if row[k] is not None:
                sums[k] += row[k]
                global_counts[k + '_n'] += 1

    fields = [
        'json','full_path','status','gold_function_count','pred_interface_count','mapped_interface_count',
        'interface_param_count_acc','interface_return_cat_acc','interface_return_cat_acc_known','interface_return_known_coverage',
        'interface_param_type_cat_acc','interface_param_type_cat_acc_known','interface_param_type_known_coverage',
        'type_param_cat_acc','type_param_cat_acc_known','type_param_known_coverage',
        'layout_pred_count','layout_mapped_count','layout_merged_count','layout_offset_precision','layout_offset_recall','layout_offset_f1','layout_raw_offset_precision','layout_raw_offset_recall','layout_raw_offset_f1','layout_raw_count','layout_merged_offset_precision','layout_merged_offset_recall','layout_merged_offset_f1','layout_observed_offset_precision','layout_observed_offset_recall','layout_observed_offset_f1','layout_observed_count','error'
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

    summary = {'rows': len(rows), 'status_counts': dict(status)}
    for k in ['interface_param_count_acc','interface_return_cat_acc','interface_return_cat_acc_known','interface_return_known_coverage','interface_param_type_cat_acc','interface_param_type_cat_acc_known','interface_param_type_known_coverage','type_param_cat_acc','type_param_cat_acc_known','type_param_known_coverage','layout_offset_precision','layout_offset_recall','layout_offset_f1','layout_raw_offset_precision','layout_raw_offset_recall','layout_raw_offset_f1','layout_merged_offset_precision','layout_merged_offset_recall','layout_merged_offset_f1','layout_observed_offset_precision','layout_observed_offset_recall','layout_observed_offset_f1']:
        n = global_counts.get(k + '_n', 0)
        summary[k + '_avg'] = (sums[k] / n) if n else None
    for k in ['mapped_interface_count','pred_interface_count','gold_function_count','layout_pred_count','layout_mapped_count','layout_merged_count','layout_raw_count','layout_observed_count']:
        summary[k + '_sum'] = global_counts.get(k, 0)
    summary['note'] = 'v7 evaluator: one-to-one InterfaceSem matching; raw/merged/observed LayoutSem metrics; consumes v10/v11 fields; still not full local-variable accuracy.'
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('wrote', out)
    print('wrote', args.summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
