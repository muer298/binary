#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

HARD = {"import_table","symbol_table","exact_signature"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nodes",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    counts=Counter()
    examples=[]
    total=0
    with open(args.nodes,encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            n=json.loads(line); total+=1
            if n.get("node_type")!="APISem": continue
            counts["apisem_nodes"]+=1
            a=n.get("attrs") or {}
            src=str(a.get("api_source") or "unknown").strip().lower()
            counts["source:"+src]+=1
            deterministic=bool(a.get("deterministic_api_source"))
            if src in HARD and deterministic:
                counts["hard_source_nodes"]+=1
                if len(examples)<5:
                    examples.append({
                        "node_id":n.get("node_id"),
                        "api_name":a.get("api_name"),
                        "api_source":src,
                        "api_source_path":a.get("api_source_path"),
                    })
    result={
        "schema":"subset100_api_provenance_preflight_v1",
        "nodes_seen":total,
        "apisem_nodes":counts["apisem_nodes"],
        "hard_source_nodes":counts["hard_source_nodes"],
        "source_counts":{k.split(":",1)[1]:v for k,v in counts.items() if k.startswith("source:")},
        "examples":examples,
        "pass":counts["hard_source_nodes"]>0,
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result["pass"]:
        raise SystemExit("No deterministic API provenance survived into model_input; stop before Stage05")
if __name__=="__main__":
    main()
