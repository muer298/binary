from __future__ import annotations
import csv, hashlib, json, os, random, re, time
from pathlib import Path

MODULES=("APISem","TypeSem","LayoutSem","InterfaceSem")
CODE_KEYS=("pseudoc","pseudo_c","decompiled_code","decompiled_c","code_text","code","text")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def dump_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    os.replace(tmp,p)

def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def write_rows(path,items):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f".tmp.{os.getpid()}")
    with tmp.open("w",encoding="utf-8") as f:
        for x in items:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n")
    os.replace(tmp,p)

def sha_text(s): return hashlib.sha256(str(s).encode()).hexdigest()
def stable_float(key): return int(sha_text(key)[:16],16)/float(16**16)
def split_for(group,ratios):
    x=stable_float(group); a=float(ratios['train']); b=a+float(ratios['val'])
    return 'train' if x<a else ('val' if x<b else 'test')

def norm_entry(v):
    s=str(v or '').strip().lower()
    if s.startswith('0x'): s=s[2:]
    return s.lstrip('0') or '0'

def find_code_in_obj(obj):
    found=[]
    def rec(x,depth=0):
        if depth>3:return
        if isinstance(x,dict):
            for k,v in x.items():
                if k in CODE_KEYS and isinstance(v,str) and len(v.strip())>=20: found.append((k,v.strip()))
                elif isinstance(v,(dict,list)): rec(v,depth+1)
        elif isinstance(x,list):
            for v in x[:20]: rec(v,depth+1)
    rec(obj)
    uniq=[];seen=set()
    for k,v in found:
        h=sha_text(v)
        if h not in seen:uniq.append((k,v));seen.add(h)
    if len(uniq)==1:return uniq[0]
    return None

def _sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def _pseudoc_from_seraph_code_export(path, expected_sid):
    """Read the exact v7.3.0 Ghidra code export referenced by code_reference.json.

    Returns (records, diagnostics). No heuristic scanning of functions.jsonl is used.
    """
    x=load_json(path)
    if not isinstance(x,dict):
        raise ValueError('SERAPH code export root must be an object: '+str(path))
    if x.get('schema')!='seraph_code_export_v7_3_0':
        raise ValueError('Unexpected SERAPH code export schema in '+str(path)+': '+str(x.get('schema')))
    if str(x.get('sample_id',''))!=str(expected_sid):
        raise ValueError('SERAPH code export sample_id mismatch: '+str(path))
    if x.get('completed') is not True:
        raise ValueError('SERAPH code export is not completed: '+str(path))
    funs=x.get('functions')
    if not isinstance(funs,list):
        raise ValueError('SERAPH code export functions must be a list: '+str(path))
    out=[]; stats={'functions_total':0,'functions_ok':0,'functions_non_ok':0,'empty_code':0,'bad_lines':0}
    for i,f in enumerate(funs):
        stats['functions_total']+=1
        if not isinstance(f,dict):
            stats['bad_lines']+=1; continue
        if f.get('status')!='OK':
            stats['functions_non_ok']+=1; continue
        fe=f.get('function_entry')
        lines=f.get('lines')
        if fe in (None,'') or not isinstance(lines,list):
            stats['bad_lines']+=1; continue
        texts=[]; valid=True
        for line in lines:
            if not isinstance(line,dict) or not isinstance(line.get('text'),str):
                valid=False; break
            texts.append(line['text'])
        if not valid:
            stats['bad_lines']+=1; continue
        code='\n'.join(texts).strip()
        if len(code)<20:
            stats['empty_code']+=1; continue
        stats['functions_ok']+=1
        out.append({
            'sample_id':str(expected_sid),
            'function_entry':norm_entry(fe),
            'code_key':'seraph_code_export_v7_3_0.functions[%d].lines[].text'%i,
            'pseudoc':code,
            'status':'OK',
            'export_path':str(Path(path).resolve()),
        })
    return out,stats


def load_pseudoc_index(task1_dir):
    """Load stripped pseudo-C from the *referenced Ghidra export* used by SERAPH v7.3.0.

    v7.3.0 samples/<sid>/functions.jsonl contains semantic/query views, not pseudo-C text.
    samples/<sid>/code_reference.json points to the authoritative
    seraph_code_export_v7_3_0 JSON. We follow that pointer and verify its sha256.
    """
    root=Path(task1_dir); out={}; ambiguous=[]
    samples=root/'samples'
    if not samples.is_dir(): raise FileNotFoundError(samples)
    diagnostics={}
    for sd in sorted(p for p in samples.iterdir() if p.is_dir()):
        sid=sd.name
        refp=sd/'code_reference.json'
        if not refp.is_file():
            diagnostics[sid]={'status':'MISSING_CODE_REFERENCE'}
            continue
        ref=load_json(refp)
        if not isinstance(ref,dict) or not isinstance(ref.get('path'),str) or not ref.get('path'):
            diagnostics[sid]={'status':'BAD_CODE_REFERENCE'}
            continue
        codepath=Path(ref['path'])
        if not codepath.is_absolute():
            # Prefer path relative to the task1 output root, then sample dir.
            p1=(root/codepath).resolve(); p2=(sd/codepath).resolve()
            codepath=p1 if p1.is_file() else p2
        if not codepath.is_file():
            diagnostics[sid]={'status':'MISSING_CODE_EXPORT','path':str(codepath)}
            continue
        want=ref.get('sha256')
        got=_sha_file(codepath)
        if want and str(want)!=got:
            raise ValueError('SERAPH code export sha256 mismatch for '+sid+': '+str(codepath))
        recs,stats=_pseudoc_from_seraph_code_export(codepath,sid)
        diagnostics[sid]={'status':'OK','path':str(codepath),'sha256':got,**stats}
        for rec in recs:
            key=(sid,rec['function_entry'])
            if key in out and out[key]['pseudoc']!=rec['pseudoc']:
                ambiguous.append(key);out.pop(key,None)
            elif key not in ambiguous:
                out[key]=rec
    return out,ambiguous

def load_binary_manifest(path):
    out={}
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            sid=r.get('sample_id')
            if sid:out[sid]=r
    return out

def choose_alias(records,aliases,min_fraction=.8):
    scores=[]
    n=max(1,len(records))
    for a in aliases:
        hit=sum(1 for r in records if r.get(a) not in (None,''))
        scores.append((hit/n,hit,a))
    scores.sort(reverse=True)
    if not scores or scores[0][0]<min_fraction:return None,scores
    if len(scores)>1 and scores[1][0]==scores[0][0] and scores[1][0]>0:return None,scores
    return scores[0][2],scores

def load_embedding_index(path):
    x=load_json(path)
    if isinstance(x,dict):
        for k in ('index','node_to_index','node_embedding_index'):
            if isinstance(x.get(k),dict):x=x[k];break
        if all(isinstance(v,int) for v in x.values()):return {str(k):int(v) for k,v in x.items()}
        if all(isinstance(k,str) and isinstance(v,dict) for k,v in x.items()):
            out={}
            for k,v in x.items():
                idx=v.get('index',v.get('row'))
                if isinstance(idx,int):out[k]=idx
            if out:return out
    if isinstance(x,list):
        out={}
        for i,v in enumerate(x):
            if isinstance(v,str):out[v]=i
            elif isinstance(v,dict):
                nid=v.get('node_id') or v.get('id'); idx=v.get('index',v.get('row',i))
                if nid is not None and isinstance(idx,int):out[str(nid)]=idx
        if out:return out
    raise ValueError('Unsupported node embedding index schema: '+str(path))

def infer_query_node_field(query_records,index_keys):
    fields={}
    n=len(query_records)
    for q in query_records:
        for k,v in q.items():
            if isinstance(v,str):fields.setdefault(k,[0,0]);fields[k][1]+=1;fields[k][0]+=int(v in index_keys)
    scored=[]
    for k,(hit,present) in fields.items():
        if present:scored.append((hit/present,hit,present,k))
    scored.sort(reverse=True)
    if not scored or scored[0][0]<.90:return None,scored[:10]
    if len(scored)>1 and scored[1][0]>=scored[0][0]-.01 and scored[1][1]>10:return None,scored[:10]
    return scored[0][3],scored[:10]
