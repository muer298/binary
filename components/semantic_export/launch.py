#!/usr/bin/env python3
from __future__ import annotations
import os,subprocess,sys,time
from pathlib import Path
from cli import parser,paths,main,MUTATING
from seraph.common import *

def launch(argv):
    a=parser().parse_args(argv)
    if a.action not in MUTATING:return main(argv)
    run,base,out=paths(a);base.mkdir(parents=True,exist_ok=True)
    try:
        with locked(base/'.launch.lock',create=True):
            with locked(base/'.run.lock',create=True):pass
            job=base/'jobs'/(stamp()+'_'+a.action+'_'+str(os.getpid()));job.mkdir(parents=True);log=job/'run.log'
            cmd=[sys.executable,'-S','-u',str(Path(__file__).with_name('cli.py'))]+argv+['--_job-dir',str(job)]
            with log.open('w') as f:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            dump(base/'latest_job.json',{'pid':p.pid,'action':a.action,'output_dir':str(out),'log':str(log),'job':str(job)})
            print('SUBMITTED PID=%s\nOUTPUT=%s\nLOG=%s'%(p.pid,out,log),flush=True)
            for _ in range(40):
                if (job/'state.json').exists() or p.poll() is not None:break
                time.sleep(.05)
            if p.poll() is not None and p.returncode:print(log.read_text(errors='replace')[-12000:]);return p.returncode
            return 0
    except RuntimeError as e:print('ALREADY_RUNNING: no duplicate submitted. '+str(e));return 3
if __name__=='__main__':raise SystemExit(launch(sys.argv[1:]))
