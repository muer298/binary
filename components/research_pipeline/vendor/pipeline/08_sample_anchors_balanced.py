#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Balanced reservoir sampling for Task2 anchors.

Reads anchors.jsonl and samples at most N anchors per evidence type while keeping
only source_resolved && target_resolved anchors by default.
"""
import argparse, json, random
from pathlib import Path
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument('--anchors', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--stats', required=True)
ap.add_argument('--limits', default='fixed_offset=10000,deref_chain=10000,def_use=10000,api_call=10000')
ap.add_argument('--seed', type=int, default=1337)
ap.add_argument('--allow-unresolved', action='store_true')
args = ap.parse_args()

limits = {}
for part in args.limits.split(','):
    if part.strip():
        k,v = part.split('=')
        limits[k.strip()] = int(v)
reservoir = {k: [] for k in limits}
seen = Counter(); resolved = Counter()
rnd = random.Random(args.seed)

with Path(args.anchors).open('r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        a = json.loads(line); et = a.get('evidence_type','unknown')
        seen[et]+=1
        if et not in limits: continue
        if not args.allow_unresolved and not (a.get('source_resolved') and a.get('target_resolved')): continue
        resolved[et]+=1
        arr = reservoir[et]; limit = limits[et]
        if len(arr) < limit:
            arr.append(a)
        else:
            j = rnd.randint(0, resolved[et]-1)
            if j < limit: arr[j] = a
picked=[]
for arr in reservoir.values(): picked.extend(arr)
rnd.shuffle(picked)
out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', encoding='utf-8') as f:
    for a in picked: f.write(json.dumps(a, ensure_ascii=False)+'\n')
stats={'input':args.anchors,'output':args.out,'limits':limits,'seen_type_counts':dict(seen),'resolved_type_counts':dict(resolved),'picked_type_counts':dict(Counter(a.get('evidence_type','unknown') for a in picked)),'picked_total':len(picked)}
Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
Path(args.stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(stats, indent=2, ensure_ascii=False))
