"""Read the verified source-index contract. Full/DWARF inputs are never opened."""
from __future__ import annotations
import csv, os, struct
from pathlib import Path
from .common import *
BASE='seraph_task1_v7_3_0'
RUN_DEFAULT='/data/chaoni/mjy/resym_exp/validation_runs/blocksem_subset20_v7_1_0'
GH_DEFAULT='/data/chaoni/mjy/resym_exp/ghidra_12.0.4_PUBLIC/support/analyzeHeadless'

def elf_sections(path):
    p=require(path)
    with p.open('rb') as f:
        h=f.read(64)
        if len(h)<52 or h[:4]!=b'\x7fELF' or h[4] not in (1,2) or h[5] not in (1,2):raise ValueError('Expected valid ELF: '+str(p))
        e='<' if h[5]==1 else '>'
        v=struct.unpack(e+('HHIQQQIHHHHHH' if h[4]==2 else 'HHIIIIIHHHHHH'),h[16:64] if h[4]==2 else h[16:52])
        off,ent,n,ns=v[5],v[10],v[11],v[12];fmt=e+('IIQQQQIIQQ' if h[4]==2 else 'IIIIIIIIII');size=p.stat().st_size
        if not off:return []
        z=struct.calcsize(fmt)
        if ent<z or off+ent>size:raise ValueError('Invalid ELF section table')
        f.seek(off);zero=struct.unpack(fmt,f.read(z))
        if n==0:n=zero[5]
        if ns==65535:ns=zero[6]
        if n>100000 or ns>=n or off+n*ent>size:raise ValueError('Invalid ELF section bounds')
        f.seek(off+ns*ent);sec=struct.unpack(fmt,f.read(z));a,l=sec[4:6]
        if l>16*(1<<20) or a+l>size:raise ValueError('Invalid ELF string table')
        f.seek(a);strings=f.read(l);names=[]
        for i in range(n):
            f.seek(off+i*ent);s=struct.unpack(fmt,f.read(z));start=s[0];end=strings.find(b'\0',start)
            if start>=l or end<0:raise ValueError('Invalid ELF section name')
            names.append(strings[start:end].decode('ascii','replace'))
        return names

def admit(path):
    names=elf_sections(path)
    forbidden=[n for n in names if n.startswith(('.debug','.zdebug')) or n.endswith('.dwo') or n in ('.gnu_debuglink','.gnu_debugaltlink')]
    if forbidden:raise ValueError('Stripped prediction input contains debug data/link: '+str(forbidden))
    return {'debug_sections_present':False,'contains_symtab':'.symtab' in names}

def configure(args,ghidra=True):
    run=Path(args.run_dir).expanduser().resolve()
    if not run.is_dir():raise FileNotFoundError('RUN_DIR: '+str(run))
    prev=inside(args.previous_dir or run/'downstream_readiness_v7_2_1/all_scopefix',run)
    required=[run/'selection_report.json',run/'snapshot.json',run/'gold_zone/pairs.csv',run/'candidate_rankings.csv',run/'checkpoints/07_rank_export.json',prev/'state.json',prev/'plan.json',prev/'source_index.sqlite3',prev/'source_index_audit.json',prev/'checkpoints/03_source_index.json']
    for p in required:require(p)
    if load(prev/'state.json').get('status')!='DONE':raise ValueError('Previous all_scopefix not DONE')
    sel=load(run/'selection_report.json')
    if sel.get('selection_uses_gold') is not False or sel.get('selection_uses_model_scores') is not False:raise ValueError('Selection firewall flags missing')
    with (run/'gold_zone/pairs.csv').open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f)
        if not {'sample_id','stripped_path'}.issubset(rd.fieldnames or []):raise ValueError('pairs schema mismatch')
        rows=[{'sample_id':r['sample_id'],'stripped_path':r['stripped_path']} for r in rd]
    ids=[r['sample_id'] for r in rows]
    if not ids or len(set(ids))!=len(ids) or ids!=sel.get('selected_sample_ids'):raise ValueError('Frozen selection/pairs mismatch')
    if load(run/'snapshot.json').get('selected_samples')!=len(ids) or load(prev/'plan.json').get('samples')!=ids:raise ValueError('Source index/snapshot scope mismatch')
    for r in rows:
        sid=r['sample_id']
        if Path(sid).name!=sid or sid in ('.','..'):raise ValueError('Unsafe sample id')
        r['stripped_path']=str(require(r['stripped_path']))
    n=args.limit_samples or (3 if args.action=='pilot' else 0)
    if n<0 or n>len(rows):raise ValueError('Invalid sample limit')
    tag=tag_ok(args.output_tag or ('pilot3' if args.action=='pilot' else 'all'))
    out=inside(run/BASE/tag,run);exe=None;ver=None
    if ghidra:
        exe=require(args.analyze_headless or os.environ.get('ANALYZE_HEADLESS') or GH_DEFAULT)
        if not os.access(str(exe),os.X_OK):raise ValueError('analyzeHeadless not executable')
        props=require(exe.parent.parent/'Ghidra/application.properties')
        ver=next((s.split('=',1)[1].strip() for s in props.read_text().splitlines() if s.startswith('application.version=')),'unknown')
        required.append(props)
    db=connect_ro(prev/'source_index.sqlite3')
    needed={'nodes':{'id','sid','typ','fn','pc','raw','attrs','raw_source'},'edges':{'id','sid','src','dst','kind','typ','path','origin_valid'},'ranks':{'sid','qid','module','rank','tid','before','after','cstatus','dstatus','dtid'}}
    try:
        for t,cols in needed.items():
            real={r[1] for r in db.execute('PRAGMA table_info('+t+')')}
            if not cols.issubset(real):raise ValueError('SQL contract missing '+t+': '+str(cols-real))
    finally:db.close()
    return {'run':run,'previous':prev,'index':prev/'source_index.sqlite3','base':run/BASE,'out':out,'all_samples':rows,'samples':rows[:n] if n else rows,'headless':exe,'ghidra_version':ver,'required':required}

def freeze(c,min_free):
    free_check(c['run'],min_free)
    cp=load(c['previous']/'checkpoints/03_source_index.json');want=cp.get('outputs',{}).get(str(c['index']))
    if not want or sha(c['index'])!=want:raise ValueError('Original source_index SHA mismatch')
    want=load(c['run']/'checkpoints/07_rank_export.json').get('outputs',{}).get('csv',{}).get('sha256')
    if not want or sha(c['run']/'candidate_rankings.csv')!=want:raise ValueError('Original rankings SHA mismatch')
    cert={'files':{str(p):record(p) for p in c['required']},'binaries':{},'full_binary_read':False,'Gold_read':False}
    for r in c['samples']:
        p=Path(r['stripped_path']);a=admit(p);cert['files'][str(p)]=record(p)
        cert['binaries'][r['sample_id']]={'stripped_sha256':cert['files'][str(p)]['sha256'],'elf':a}
    return cert

def unchanged(cert):
    for p,r in cert['files'].items():
        s=require(p).stat()
        if s.st_size!=r['bytes'] or s.st_mtime_ns!=r['mtime_ns']:raise ValueError('Original input changed: '+p)
