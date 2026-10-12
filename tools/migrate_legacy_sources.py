"""One-shot, recoverable migration for this inspected server release only.

Never recursively deletes files. The reviewed plan carries per-target hashes.
The default is a read-only preflight; --apply is required for relocation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1048576),b''):h.update(b)
    return h.hexdigest()

def fingerprint(p):
    fs=sorted(x for x in p.rglob('*') if x.is_file() and not x.is_symlink() and '__pycache__' not in x.parts) if p.is_dir() else [p]
    h=hashlib.sha256();size=0
    for f in fs:
        h.update((str(f.relative_to(p)) if p.is_dir() else p.name).encode());h.update(b'\0')
        h.update(bytes.fromhex(digest(f)));size+=f.stat().st_size
    return len(fs),size,h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('plan',type=Path);ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    plan=json.loads(a.plan.read_text());root=Path(plan['root']);archive=Path(plan['archive']);release=Path(plan['release'])
    if str(root)!='/data/chaoni/mjy/resym_exp' or str(release)!='/data/chaoni/mjy/binary' or str(archive)!='/data/chaoni/mjy_code_archive_20261012':
        raise RuntimeError('Plan is not the reviewed exact target set')
    if archive.exists() or archive.is_symlink():raise RuntimeError('Archive already exists; inspect journal, do not rerun blindly')
    if release.resolve()!=release or root.resolve()!=root:raise RuntimeError('Unexpected symlink in root')
    subprocess.run([os.sys.executable,str(release/'tools/verify_release.py')],check=True)
    report=root/'training_runs/seraph_task2/v0_3_8/TASK2_V0_3_8_REPORT.md'
    if 'EXPANDED_LOCALIZATION_VALIDATION_COMPLETE' not in report.read_text():raise RuntimeError('v038 is not complete')
    rows=json.loads((release/'release/source_manifest.json').read_text())['files']
    for r in rows:
        p=Path(r['source_path'])
        if not p.is_file() or digest(p)!=r['sha256']:raise RuntimeError('Source changed after publication: '+str(p))
    # Exclude this Python process and its launch ancestors, but not other jobs.
    ancestors={os.getpid()};pid=os.getppid()
    while pid>1:
        ancestors.add(pid)
        try:pid=int(Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split()[1])
        except (OSError,ValueError,IndexError):break
    active=[]
    needles=[str(x['source'])+'/' for x in plan['targets'] if x['kind']=='directory']
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in ancestors:continue
        try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if any(n in cmd for n in needles):active.append({'pid':int(p.name),'command':cmd[:400]})
    if active:raise RuntimeError('Active consumers; no files moved: '+json.dumps(active))
    seen=set()
    for x in plan['targets']:
        p=Path(x['source']);rel=Path(x['relative'])
        if p in seen or p!=root/rel or rel.is_absolute() or '..' in rel.parts or str(rel)=='.' or p.is_symlink():raise RuntimeError('Unsafe/duplicate target')
        if rel.parts[0] in plan['protected']:raise RuntimeError('Protected target')
        if not p.exists():raise RuntimeError('Missing target '+str(p))
        got=fingerprint(p)
        if got!=(x['files'],x['bytes'],x['tree_sha256']):raise RuntimeError('Target changed since planning: '+str(p))
        seen.add(p)
    print(json.dumps({'preflight':'PASSED','targets':len(seen),'bytes':sum(x['bytes'] for x in plan['targets']),'apply':a.apply}),flush=True)
    if not a.apply:return
    archive.mkdir(mode=0o700)
    journal={'status':'IN_PROGRESS','started_epoch':time.time(),'archive':str(archive),'moves':[],'links':[],'permanent_deletion':False}
    def save():
        (archive/'MIGRATION_JOURNAL.json').write_text(json.dumps(journal,ensure_ascii=False,indent=2)+'\n')
    save()
    try:
        for x in plan['targets']:
            src=Path(x['source']);dst=archive/x['relative'];dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists() or dst.is_symlink():raise RuntimeError('Archive collision')
            src.rename(dst);journal['moves'].append({'from':str(src),'to':str(dst),'tree_sha256':x['tree_sha256']});save()
            target=release/'packages' if x['relative']=='packages' else release/'components/data_tools' if x['relative']=='scripts' else None
            if target is not None:
                src.symlink_to(target,target_is_directory=True);journal['links'].append({'path':str(src),'target':str(target)});save()
        for row in rows:
            if digest(Path(row['source_path']))!=row['sha256']:raise RuntimeError('Compatibility resolution failed: '+row['source_path'])
        journal['status']='COMPLETE';journal['completed_epoch']=time.time();save()
        (release/'release/cleanup_status.json').write_text(json.dumps(journal,ensure_ascii=False,indent=2)+'\n')
    except Exception:
        for x in reversed(journal['links']):
            p=Path(x['path'])
            if p.is_symlink() and str(p.readlink())==x['target']:p.unlink()
        for x in reversed(journal['moves']):
            src=Path(x['from']);dst=Path(x['to'])
            if not src.exists() and not src.is_symlink():dst.rename(src)
        journal['status']='ROLLED_BACK';save();raise
    print(json.dumps({'status':'COMPLETE','moved_targets':len(journal['moves']),'archive':str(archive),'permanent_deletion':False}),flush=True)

if __name__=='__main__':main()
