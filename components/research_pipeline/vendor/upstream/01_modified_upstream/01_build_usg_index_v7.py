#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--json-root',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--stats',required=True)
    a=ap.parse_args()
    root=Path(a.json_root).resolve(); out=Path(a.out); stats=Path(a.stats)
    files=sorted(root.rglob('*_semantic_dict.json'))
    rows=[]; errors=[]
    for p in files:
        sid=p.name[:-len('_semantic_dict.json')]
        try:
            d=json.load(open(p,encoding='utf-8'))
            pi=d.get('program_info',{}) or {}
            rows.append({'sample_id':sid,'json_path':str(p.resolve()),'status':'ok',
                         'schema_version':pi.get('schema_version'),'generator':pi.get('generator')})
        except Exception as e:
            errors.append({'file':str(p),'error':repr(e)})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    stats.parent.mkdir(parents=True,exist_ok=True)
    stats.write_text(json.dumps({'json_root':str(root),'samples':len(rows),'errors':errors},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'samples':len(rows),'errors':len(errors),'out':str(out)},ensure_ascii=False,indent=2))
    return 0 if rows and not errors else (2 if not rows else 1)
if __name__=='__main__':raise SystemExit(main())
