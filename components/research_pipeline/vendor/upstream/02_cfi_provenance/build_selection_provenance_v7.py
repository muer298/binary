#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from bisect import bisect_right
from collections import defaultdict, deque
from pathlib import Path

def read_jsonl(p):
    if not p or not Path(p).exists(): return []
    return [json.loads(x) for x in Path(p).read_text(errors='replace').splitlines() if x.strip()]

def parse_addr(x):
    if x is None: return None
    try: return int(str(x),0)
    except Exception:
        try: return int(str(x),16)
        except Exception: return None

def canon_reg(x):
    if x is None: return None
    s=str(x).upper().replace('%','')
    return {'ESP':'RSP','SP':'RSP','EBP':'RBP','BP':'RBP','FP':'RBP'}.get(s,s)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--semantic-json',required=True)
    ap.add_argument('--cfi',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    sem=json.load(open(a.semantic_json))
    cfi=read_jsonl(a.cfi)
    cfi=sorted(cfi,key=lambda r:int(r['pc_start_rva']))
    starts=[int(x['pc_start_rva']) for x in cfi]
    def cfi_at(pc):
        if pc is None or not starts: return None
        i=bisect_right(starts,pc)-1
        if i<0:return None
        r=cfi[i]
        return r if pc<int(r['pc_end_rva']) else None
    low=sem.get('low_level_sem',{}) or {}
    instrs=low.get('instr_sem',[]) or []
    blocks=low.get('block_sem',[]) or []
    bs7=sem.get('blocksem_v7',{}) or {}
    vedges=bs7.get('pcode_value_edges',[]) or []
    prolog=bs7.get('stack_frame_prologue_relations',[]) or []
    # Value graph is directional input->output; reverse lets us trace a loaded pointer to RSP/RBP roots.
    rev=defaultdict(list)
    for e in vedges:
        rev[e.get('target_value_id')].append(e)
    pcode_by_site={x.get('pcode_site_id'):x for x in bs7.get('pcode_ops',[]) or []}
    records=[]
    for ins in instrs:
        eff=ins.get('effects',{}) or {}
        for acc in eff.get('memory_accesses',[]) or []:
            pc=parse_addr(ins.get('address'))
            # semantic addresses are VAs in the legacy exporter; use CFI VA interval when possible.
            match=None
            for r in cfi:
                s=parse_addr(r.get('pc_start_va')); e=parse_addr(r.get('pc_end_va'))
                if s is not None and e is not None and pc is not None and s<=pc<e:
                    match=r; break
            seeds=list(acc.get('provenance_seed_locations',[]) or [])
            seeds += list(acc.get('support_locs',[]) or [])
            if acc.get('base'): seeds.append(acc.get('base'))
            # We retain exact reported fixed offsets; no inference is promoted to strong here.
            base=str(acc.get('base') or '')
            cfa_ok=False; cfa_rel=None
            if match and acc.get('offset') is not None:
                b=base.upper()
                cr=canon_reg(match.get('cfa_reg'))
                if cr and cr in b:
                    cfa_ok=True; cfa_rel=int(acc.get('offset'))-int(match.get('cfa_offset') or 0)
            records.append({
                'schema_version':'blocksem_selection_evidence_v7',
                'instr_id':ins.get('id'),'pcode_site_id':ins.get('pcode_site_id'),
                'function_entry':ins.get('function_entry'),'block_entry':ins.get('block_entry'),
                'address':ins.get('address'),'access':acc,
                'cfi_interval':match,
                'cfi_direct_base_match':cfa_ok,
                'cfa_relative_offset':cfa_rel,
                'prologue_relation_count':len(prolog),
                'block_contract_complete': any(str(b.get('entry'))==str(ins.get('block_entry')) and b.get('member_list_complete') for b in blocks),
                'eligible_upstream_evidence': bool(match) and bool(ins.get('input_value_ids') or ins.get('output_value_ids'))
            })
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    stats={'records':len(records),
           'cfi_interval_match':sum(bool(x['cfi_interval']) for x in records),
           'direct_cfa_match':sum(bool(x['cfi_direct_base_match']) for x in records),
           'block_contract_complete':sum(bool(x['block_contract_complete']) for x in records),
           'eligible_upstream_evidence':sum(bool(x['eligible_upstream_evidence']) for x in records)}
    Path(str(out)+'.stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
