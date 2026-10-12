#!/usr/bin/env python3
from __future__ import annotations
import argparse,os,traceback
from pathlib import Path
from collections import deque
from seraph.common import *
from seraph.source import BASE,RUN_DEFAULT,configure
from seraph.runner import prepare
from seraph.client import ask
from seraph.evaluation import evaluate
ROOT=Path(__file__).resolve().parent
MUTATING={'pilot','run','continue','ask','evaluate'}

def parser():
    p=argparse.ArgumentParser(description='Task1 first integration; not full SERAPH training, not a security detector')
    p.add_argument('action',choices=['check','pilot','run','continue','status','log','report','ask','evaluate'])
    p.add_argument('--run-dir',default=RUN_DEFAULT);p.add_argument('--previous-dir',default='');p.add_argument('--output-tag',default='');p.add_argument('--analyze-headless',default='')
    p.add_argument('--limit-samples',type=int,default=0);p.add_argument('--max-requests',type=int,default=24);p.add_argument('--function-timeout',type=int,default=45);p.add_argument('--min-free-gib',type=float,default=5)
    p.add_argument('--llm-config',default='');p.add_argument('--session',default='primary');p.add_argument('--allow-send-code',action='store_true');p.add_argument('--gold',default='');p.add_argument('--baseline-only',action='store_true');p.add_argument('--evaluation-tag',default='posthoc_primary');p.add_argument('--_job-dir',default='',help=argparse.SUPPRESS)
    return p

def paths(a):
    run=Path(a.run_dir).expanduser().resolve()
    if not run.is_dir():raise FileNotFoundError('RUN_DIR: '+str(run))
    base=run/BASE;tag=a.output_tag or ('pilot3' if a.action=='pilot' else 'all')
    return run,base,inside(base/tag_ok(tag),run)

def main(argv=None):
    a=parser().parse_args(argv);run,base,out=paths(a)
    if a.max_requests<1 or a.max_requests>10000 or not 1<=a.function_timeout<=600 or a.min_free_gib<0:raise ValueError('Invalid budgets')
    if a.action=='report':
        f=out/'task1_report.md'
        if not f.is_file():raise FileNotFoundError(str(f)+'; specify --output-tag pilot3 for pilot report')
        print(f.read_text());return 0
    if a.action=='log':
        if not (base/'latest_job.json').is_file():print('NO_BACKGROUND_JOB');return 0
        x=load(base/'latest_job.json');print(canonical(x))
        with Path(x['log']).open(errors='replace') as f:print(''.join(deque(f,maxlen=100)),end='')
        return 0
    if a.action=='status':
        for p in sorted(base.glob('*/state.json')):print(p);print(p.read_text())
        for p in sorted(base.glob('*/llm/*/summary.json')):print(p);print(p.read_text())
        if (base/'.run.lock').exists():
            try:
                with locked(base/'.run.lock'):print('ACTIVE_LOCK: NO')
            except RuntimeError:print('ACTIVE_LOCK: YES')
        else:print('ACTIVE_LOCK: NO')
        return 0
    if a.action=='check':
        c=configure(a)
        with locked(run/'.pipeline.lock',shared=True),locked(c['previous']/'.audit.lock',shared=True):free_check(run,a.min_free_gib)
        print(canonical({'check':'PASSED','source_index_dir':str(c['previous']),'output_dir':str(c['out']),'active_samples':len(c['samples']),'frozen_parent_samples':len(c['all_samples']),'Ghidra':str(c['headless']),'Ghidra_version':c['ghidra_version'],'old_inference_rerun':False,'note':'Full digests and debug-section admission run before export'}));return 0
    job=inside(a._job_dir,base/'jobs') if a._job_dir else None
    try:
        with locked(base/'.run.lock',create=True):
            if job:dump(job/'state.json',{'status':'RUNNING','action':a.action,'pid':os.getpid(),'output_dir':str(out)})
            if a.action in ('pilot','run','continue'):rc=prepare(configure(a),a,ROOT)
            elif a.action=='ask':
                if not a.llm_config:raise ValueError('ask requires actual --llm-config')
                rc=ask(out,a.llm_config,a.session,a.allow_send_code)
            else:
                if not a.gold:raise ValueError('evaluate requires explicit --gold after predictions frozen')
                rc=evaluate(out,a.gold,a.session,a.evaluation_tag,a.baseline_only)
            if job:dump(job/'state.json',{'status':'DONE' if rc==0 else 'PARTIAL_OR_FAILED','action':a.action,'returncode':rc,'output_dir':str(out),'finished_utc':stamp()})
            return rc
    except Exception as e:
        if job:dump(job/'state.json',{'status':'FAILED','action':a.action,'error':str(e),'output_dir':str(out),'finished_utc':stamp()})
        raise

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:traceback.print_exc();raise SystemExit(1)
