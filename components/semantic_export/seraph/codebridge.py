from __future__ import annotations
import os, signal, subprocess, tempfile
from pathlib import Path
from .common import *

def validate(path,sid,entries):
    d=load(path)
    if d.get('schema')!='seraph_code_export_v7_3_0' or d.get('sample_id')!=sid or d.get('completed') is not True or d.get('requested_entries')!=entries:raise ValueError('Ghidra exporter schema/scope mismatch')
    if len(d.get('functions',[]))!=len(entries):raise ValueError('Ghidra function count mismatch')
    seen=set()
    for f in d['functions']:
        if f.get('sample_id')!=sid or f.get('function_entry') not in entries or f['function_entry'] in seen:raise ValueError('Ghidra function identity mismatch')
        seen.add(f['function_entry'])
        if f['status'] not in {'OK','FUNCTION_ENTRY_NOT_FOUND','EXTERNAL_FUNCTION','DECOMPILE_INCOMPLETE','NO_CODE_MARKUP','NO_CODE_LINES'}:raise ValueError('Unsupported Ghidra status')
        if f['status']=='OK' and not f['lines']:raise ValueError('Empty OK function')
        for i,l in enumerate(f['lines'],1):
            if l['line']!=i or not isinstance(l['text'],str):raise ValueError('Code line schema mismatch')
            end=0
            for t in l['tokens']:
                a,b=t['start_char'],t['end_char']
                if type(a) is not int or type(b) is not int or not end<=a<=b<=len(l['text']) or l['text'][a:b]!=t['text']:raise ValueError('Token text/offset mismatch')
                end=b
                for link in t['anchors']:
                    if hx(link.get('address')) is None or link.get('kind') not in ('pcode_target','token_single_address'):raise ValueError('Bad direct address anchor')
    return d

def export(c,r,entries,script,timeout=45):
    sid=r['sample_id'];binary=Path(r['stripped_path']);params={'sid':sid,'stripped_sha':sha(binary),'entries':entries,'script_sha':sha(script),'ghidra_version':c['ghidra_version'],'timeout':timeout}
    cache=c['base']/'assets'/sid/digest(params);done=cache/'done.json';out=cache/'code.json'
    if done.is_file():
        cp=load(done)
        if cp['parameters']!=params or sha(out)!=cp['sha256']:raise ValueError('Code cache changed')
        validate(out,sid,entries);print('[REUSE VERIFIED CODE]',sid,flush=True);return out
    cache.mkdir(parents=True,exist_ok=True);attempt=Path(tempfile.mkdtemp(prefix='attempt_',dir=str(cache)));pending=attempt/'pending.json';log=attempt/'ghidra.log';req=attempt/'entries.txt';req.write_text('\n'.join(entries)+'\n',encoding='ascii')
    if not entries:
        dump(pending,{'schema':'seraph_code_export_v7_3_0','sample_id':sid,'completed':True,'image_base':None,'requested_entries':[],'functions':[]})
    else:
        project=attempt/'project';project.mkdir()
        cmd=[str(c['headless']),str(project),'seraph_'+digest(params)[:12],'-import',str(binary),'-scriptPath',str(script.parent),'-postScript',script.name,str(pending),sid,str(req),str(timeout),'-analysisTimeoutPerFile','600','-max-cpu','2','-deleteProject']
        env=os.environ.copy();env['LC_ALL']='C'
        with log.open('w') as f:
            proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            try:rc=proc.wait(timeout=780+len(entries)*(timeout+5))
            except BaseException:
                try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
                except (ProcessLookupError,subprocess.TimeoutExpired):
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                raise
        dump(attempt/'command.json',{'command':cmd,'returncode':rc,'log':str(log)})
        text=log.read_text(errors='replace')[-2000000:]
        if rc!=0 or not pending.is_file() or 'SERAPH_CODE_EXPORT_DONE '+sid not in text or 'GhidraScriptLoadException' in text or 'ERROR REPORT SCRIPT ERROR' in text:
            raise RuntimeError('Ghidra export failed; actual per-sample log: '+str(log))
    validate(pending,sid,entries)
    if sha(binary)!=params['stripped_sha']:raise ValueError('Binary changed during export')
    os.replace(pending,out);dump(done,{'parameters':params,'sha256':sha(out)});return out
