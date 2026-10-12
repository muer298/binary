#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Balanced anchor sampling with runtime/startup/helper API-call filtering."""
import argparse, json, random
from pathlib import Path
from collections import Counter

BAD_API_NAMES = {
    '_start','_init','_fini','main','__libc_start_main','__gmon_start__',
    '__do_global_dtors_aux','register_tm_clones','deregister_tm_clones',
    'frame_dummy','__cxa_finalize','__stack_chk_fail','__stack_chk_fail_local'
}
BAD_PREFIXES = ('__sub_I_', '__static_initialization', '_GLOBAL__sub_I_')

def collect_strings(obj):
    vals=[]
    if isinstance(obj, dict):
        for k,v in obj.items():
            if isinstance(v,(str,int,float)): vals.append(str(v))
            elif isinstance(v,list): vals.extend(str(x) for x in v[:20] if isinstance(x,(str,int,float)))
            elif isinstance(v,dict): vals.extend(collect_strings(v))
    return vals

def is_bad_api_anchor(a):
    candidates=[]
    for k in ['api_name','api_target','function_name','source_raw','target_raw','edge_type_raw','source_node','target_node']:
        if a.get(k): candidates.append(str(a.get(k)))
    for side in ['source_attrs','target_attrs']:
        candidates.extend(collect_strings(a.get(side,{})))
    joined=' '.join(candidates)
    for bad in BAD_API_NAMES:
        if bad in joined: return bad
    for pre in BAD_PREFIXES:
        if pre in joined: return pre
    return None

def get_api_name_for_stats(a):
    for side in ['target_attrs','source_attrs']:
        attrs=a.get(side,{})
        if isinstance(attrs,dict):
            for k in ['api_name','api_target','function_name','semantic_template']:
                if attrs.get(k): return str(attrs.get(k))
    for k in ['api_name','api_target','function_name','target_raw','source_raw']:
        if a.get(k): return str(a.get(k))
    return 'unknown'

ap=argparse.ArgumentParser()
ap.add_argument('--anchors', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--stats', required=True)
ap.add_argument('--limits', default='fixed_offset=10000,deref_chain=10000,def_use=10000,api_call=10000')
ap.add_argument('--seed', type=int, default=1337)
args=ap.parse_args()
limits={}
for part in args.limits.split(','):
    if part.strip():
        k,v=part.split('='); limits[k.strip()]=int(v)
reservoir={k:[] for k in limits}; seen=Counter(); resolved=Counter(); filtered=Counter(); kept_api=Counter()
rnd=random.Random(args.seed)
with Path(args.anchors).open('r',encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        a=json.loads(line); et=a.get('evidence_type','unknown'); seen[et]+=1
        if et not in limits: continue
        if not (a.get('source_resolved') and a.get('target_resolved')): continue
        if et=='api_call':
            bad=is_bad_api_anchor(a)
            if bad:
                filtered[bad]+=1; continue
            kept_api[get_api_name_for_stats(a)]+=1
        resolved[et]+=1
        arr=reservoir[et]; limit=limits[et]
        if len(arr)<limit: arr.append(a)
        else:
            j=rnd.randint(0,resolved[et]-1)
            if j<limit: arr[j]=a
picked=[]
for arr in reservoir.values(): picked.extend(arr)
rnd.shuffle(picked)
out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w',encoding='utf-8') as f:
    for a in picked: f.write(json.dumps(a, ensure_ascii=False)+'\n')
stats={'input':args.anchors,'output':args.out,'limits':limits,'seen_type_counts':dict(seen),'resolved_type_counts_after_filter':dict(resolved),'filtered_bad_api':dict(filtered),'kept_api_name_top30':kept_api.most_common(30),'picked_type_counts':dict(Counter(a.get('evidence_type','unknown') for a in picked)),'picked_total':len(picked)}
Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
Path(args.stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(stats, indent=2, ensure_ascii=False))
