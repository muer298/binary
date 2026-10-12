#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, subprocess
from pathlib import Path

FDE = re.compile(r"FDE.*?pc=([0-9a-fA-Fx]+)\.\.([0-9a-fA-Fx]+)")
ROW = re.compile(r"^\s*([0-9a-fA-F]{4,})\s+(\S+)")
CFA = re.compile(r"^([A-Za-z][A-Za-z0-9]*)\s*([+-])\s*(0x[0-9a-fA-F]+|\d+)$")

def canon_reg(x):
    x=str(x).upper().replace('%','')
    return {'ESP':'RSP','SP':'RSP','EBP':'RBP','BP':'RBP','FP':'RBP'}.get(x,x)

def parse_cfa(s):
    m=CFA.match(s)
    if not m: return None,None
    off=int(m.group(3),0)
    if m.group(2)=='-': off=-off
    return canon_reg(m.group(1)),off

def load_base(binary):
    try:
        from elftools.elf.elffile import ELFFile
        with open(binary,'rb') as f:
            e=ELFFile(f)
            vs=[int(s['p_vaddr']) for s in e.iter_segments() if s['p_type']=='PT_LOAD']
            return min(vs) if vs else 0
    except Exception:
        return 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--binary',required=True)
    ap.add_argument('--sample-id',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--readelf',default='readelf')
    a=ap.parse_args()
    cp=subprocess.run([a.readelf,'--debug-dump=frames-interp','--wide',a.binary],text=True,capture_output=True)
    if cp.returncode!=0:
        raise SystemExit(cp.stderr or 'readelf failed')
    base=load_base(a.binary)
    rows=[]; current=[]; fstart=fend=None
    def flush():
        nonlocal current
        if not current: return
        current.sort(key=lambda x:x['pc_start_va'])
        for i,r in enumerate(current):
            end=current[i+1]['pc_start_va'] if i+1<len(current) else fend
            r['pc_end_va']=end
            r['pc_start_rva']=r['pc_start_va']-base
            r['pc_end_rva']=None if end is None else end-base
            if end is not None and end>r['pc_start_va']:
                rows.append(r)
        current=[]
    for line in cp.stdout.splitlines():
        m=FDE.search(line)
        if m:
            flush(); fstart=int(m.group(1),16); fend=int(m.group(2),16); continue
        if fstart is None: continue
        m=ROW.match(line)
        if not m: continue
        pc=int(m.group(1),16)
        if not (fstart<=pc<=fend): continue
        reg,off=parse_cfa(m.group(2))
        if reg is None: continue
        current.append({'schema_version':'blocksem_cfi_v7','sample_id':a.sample_id,
                        'fde_start_va':fstart,'fde_end_va':fend,
                        'pc_start_va':pc,'cfa_reg':reg,'cfa_offset':off})
    flush()
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
    print(json.dumps({'sample_id':a.sample_id,'rows':len(rows),'load_base':base},indent=2))
    return 0 if rows else 3
if __name__=='__main__': raise SystemExit(main())
