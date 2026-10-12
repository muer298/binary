#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select only USG nodes that appear in anchor neighborhoods."""
import argparse, json
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--nodes', required=True)
ap.add_argument('--neighborhoods', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--stats', required=True)
args=ap.parse_args()
needed=set()
with Path(args.neighborhoods).open('r',encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        d=json.loads(line)
        for k in ['source_node','target_node']:
            if d.get(k): needed.add(d.get(k))
        for side in ['low_nodes','high_nodes']:
            for n in d.get(side,[]):
                if n.get('node_id'): needed.add(n.get('node_id'))
written=0; seen=0
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with Path(args.nodes).open('r',encoding='utf-8') as fin, Path(args.out).open('w',encoding='utf-8') as fout:
    for line in fin:
        if not line.strip(): continue
        seen+=1; n=json.loads(line)
        if n.get('node_id') in needed:
            fout.write(json.dumps(n, ensure_ascii=False)+'\n'); written+=1
stats={'needed_node_ids':len(needed),'input_nodes_seen':seen,'selected_nodes_written':written,'out':args.out}
Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
Path(args.stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(stats, indent=2, ensure_ascii=False))
