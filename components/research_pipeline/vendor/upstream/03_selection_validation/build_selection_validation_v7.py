#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, random
from collections import Counter,defaultdict
from pathlib import Path

def rows(path):
    p=Path(path)
    if p.suffix=='.jsonl': return [json.loads(x) for x in p.read_text(errors='replace').splitlines() if x.strip()]
    if p.suffix=='.json':
        x=json.load(open(p));
        if isinstance(x,list): return x
        for k in ('rows','items','queries','examples','data'):
            if isinstance(x.get(k),list): return x[k]
        return [x]
    with p.open(newline='') as f: return list(csv.DictReader(f))

def first(r,names,default=None):
    for k in names:
        if r.get(k) not in (None,''):return r[k]
    return default

def truthy(x):
    if isinstance(x,bool):return x
    return str(x).lower() in ('1','true','yes','strong','pass','passed')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description='Build fresh project-isolated Selection Validation. Gold must be independent of repaired model scores.')
    ap.add_argument('--gold',required=True,help='Independent object Gold JSONL/JSON/CSV')
    ap.add_argument('--baseline',required=True,help='Frozen PRE-REPAIR baseline rankings')
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--seed',type=int,default=20260910)
    ap.add_argument('--group-by',choices=('project','sample'),default='project')
    ap.add_argument('--min-strong',type=int,default=100)
    ap.add_argument('--min-baseline-errors',type=int,default=10)
    a=ap.parse_args()
    gold=[]
    for r in rows(a.gold):
        q=first(r,['query_id','target_id','object_id','access_id','id'])
        g=first(r,['gold_id','gold_candidate_id','gold_layout_id','label_id','target_candidate_id','gold'])
        strong=truthy(first(r,['strong','is_strong','strong_label','gold_strong','eligible'],False))
        if q is None or g is None or not strong:continue
        gold.append({'query_id':str(q),'gold_id':str(g),
                     'sample_id':str(first(r,['sample_id','sample','binary_id','binary'],'unknown')),
                     'project_id':str(first(r,['project_id','project','package'],first(r,['sample_id','sample'],'unknown'))),
                     'gold_source':first(r,['gold_source','label_source','evidence_source']),
                     'raw_gold':r})
    bmap={}
    for r in rows(a.baseline):
        q=first(r,['query_id','target_id','object_id','access_id','id'])
        if q is None:continue
        ranked=first(r,['ranked_ids','candidate_ids','ranking','candidates','topk'],[])
        if isinstance(ranked,str):
            try:ranked=json.loads(ranked)
            except Exception:ranked=[x for x in ranked.replace(',',' ').split() if x]
        if ranked and isinstance(ranked[0],dict):ranked=[first(x,['id','candidate_id','layout_id','node_id']) for x in ranked]
        top=first(r,['top1_id','top1','prediction','predicted_id'],ranked[0] if ranked else None)
        bmap[str(q)]={'top1':None if top is None else str(top),'ranked':[str(x) for x in ranked if x is not None]}
    for x in gold:
        b=bmap.get(x['query_id']); x['baseline_covered']=b is not None
        x['baseline_top1']=None if b is None else b['top1']
        x['baseline_top1_correct']=bool(b and b['top1']==x['gold_id'])
        x['baseline_ranked_ids']=[] if b is None else b['ranked']
    groups=defaultdict(list)
    for x in gold:groups[x['project_id'] if a.group_by=='project' else x['sample_id']].append(x)
    keys=sorted(groups); random.Random(a.seed).shuffle(keys)
    n=sum(map(len,groups.values())); ct=Counter(); split={}
    for k in keys:
        s='train' if ct['train']<.70*n else ('validation' if ct['validation']<.15*n else 'test')
        split[k]=s; ct[s]+=len(groups[k])
    for x in gold:x['split']=split[x['project_id'] if a.group_by=='project' else x['sample_id']]
    errors=[x for x in gold if x['baseline_covered'] and not x['baseline_top1_correct']]
    gate={'strong':len(gold),'baseline_covered':sum(x['baseline_covered'] for x in gold),
          'baseline_top1_errors':len(errors),'group_count':len(groups),'split_counts':dict(ct),
          'gold_sha256':sha(a.gold),'baseline_sha256':sha(a.baseline),
          'strong_gate':len(gold)>=a.min_strong,
          'baseline_error_gate':len(errors)>=a.min_baseline_errors,
          'baseline_coverage_gate':sum(x['baseline_covered'] for x in gold)==len(gold),
          'split_gate':all(ct[s]>0 for s in ('train','validation','test')),
          'independence_rule':'repaired model predictions are not inputs to sample selection; baseline is used only for the fixed diagnostic error subset'}
    gate['pass']=all(gate[k] for k in ('strong_gate','baseline_error_gate','baseline_coverage_gate','split_gate'))
    gate['decision']='READY_FOR_ONE_TIME_PARAMETER_SELECTION' if gate['pass'] else 'STOP_AND_AUGMENT_SELECTION_VALIDATION'
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    def dump(name,xs):
        with (out/name).open('w') as f:
            for x in xs:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
    dump('selection_all.jsonl',gold);dump('selection_baseline_errors.jsonl',errors)
    for s in ('train','validation','test'):dump('selection_'+s+'.jsonl',[x for x in gold if x['split']==s])
    (out/'selection_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(gate,ensure_ascii=False,indent=2))
    return 0 if gate['pass'] else 2
if __name__=='__main__':raise SystemExit(main())
