#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path

def read(p):return [json.loads(x) for x in Path(p).read_text(errors='replace').splitlines() if x.strip()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selection-dir',required=True);a=ap.parse_args();d=Path(a.selection_dir)
    gate=json.load(open(d/'selection_gate.json')); rows=read(d/'selection_all.jsonl')
    err=[];seen=set(); groups=defaultdict(set)
    for i,x in enumerate(rows):
        q=x.get('query_id')
        if q in seen:err.append('duplicate query_id '+str(q))
        seen.add(q); groups[x.get('project_id')].add(x.get('split'))
        if not x.get('baseline_covered'):err.append('baseline missing '+str(q))
        if not x.get('gold_id'):err.append('gold missing '+str(q))
    leak={k:list(v) for k,v in groups.items() if len(v)>1}
    if leak:err.append('project leakage: '+json.dumps(leak)[:1000])
    report={'rows':len(rows),'gate_pass':gate.get('pass'),'errors':err,'valid':not err and gate.get('pass')}
    (d/'STATIC_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['valid'] else 2
if __name__=='__main__':raise SystemExit(main())
