from __future__ import annotations
from collections import Counter
from pathlib import Path
from .common import *
from .source import freeze,unchanged
from .baseline import extract
from .codebridge import export
from .ssq import build

def code_hash(root):return digest({str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*.py')) if 'tests' not in p.parts and '__pycache__' not in p.parts})

def prepare(c,args,root):
    out=c['out'];out.mkdir(parents=True,exist_ok=True)
    with locked(c['run']/'.pipeline.lock',shared=True),locked(c['previous']/'.audit.lock',shared=True):
        plan={'schema':'seraph_task1_plan_v7_3_0','sample_ids':[r['sample_id'] for r in c['samples']],'parent_sample_ids':[r['sample_id'] for r in c['all_samples']],'source_run':str(c['run']),'previous_dir':str(c['previous']),'code_sha':code_hash(root),'ghidra_version':c['ghidra_version'],'headless':str(c['headless']),'max_requests':args.max_requests,'function_timeout':args.function_timeout,'Gold_read_for_generation':False,'phase2_gated_embeddings':'NOT_IMPLEMENTED_THIS_BATCH','closed_loop_LLM_integration':'NOT_IMPLEMENTED_THIS_BATCH'}
        if (out/'plan.json').exists() and load(out/'plan.json')!=plan:raise ValueError('Plan/code/scope changed; use a new --output-tag, never edit checkpoints')
        dump(out/'plan.json',plan);state={'status':'RUNNING','stage':'01_freeze_source','pid':__import__('os').getpid(),'started_utc':stamp()};dump(out/'state.json',state)
        try:
            print('[STAGE 01] Verify source index and stripped inputs; Gold not opened',flush=True)
            cert=freeze(c,args.min_free_gib)
            if (out/'input_certificate.json').exists() and load(out/'input_certificate.json')!=cert:raise ValueError('Original input identity changed')
            dump(out/'input_certificate.json',cert)
            if (out/'prepare_checkpoint.json').exists():
                verify_outputs(load(out/'prepare_checkpoint.json'));report=load(out/'task1_report.json');dump(out/'state.json',{**state,'status':report['status'],'stage':'05_report','completed_utc':stamp(),'reused':True});print('SERAPH TASK1 PREPARED (REUSED VERIFIED OUTPUTS)',flush=True);return 0
            counts=Counter();dirs=[]
            for i,r in enumerate(c['samples'],1):
                sid=r['sample_id'];d=out/'samples'/sid;cp=out/'checkpoints'/(sid+'.json');fp={'plan':digest(plan),'index':cert['files'][str(c['index'])]['sha256'],'stripped':cert['binaries'][sid]['stripped_sha256']}
                state.update(stage='02_function_views_and_03_code_bridge',sample=sid,progress='%s/%s'%(i,len(c['samples'])));dump(out/'state.json',state)
                if cp.is_file():
                    x=load(cp)
                    if x['fingerprint']!=fp:raise ValueError('Sample checkpoint identity changed')
                    verify_outputs(x);count=load(d/'sample_summary.json');print('[SKIP VERIFIED SAMPLE]',sid,flush=True)
                else:
                    if d.exists():
                        a=out/'attempts'/(sid+'_'+stamp()+'_'+str(__import__('os').getpid()));a.parent.mkdir(parents=True,exist_ok=True);d.rename(a)
                    free_check(out,args.min_free_gib);print('[SAMPLE]',i,'/',len(c['samples']),sid,flush=True);count=extract(c['index'],sid,d)
                    entries=sorted({f['function_entry'] for f in jsonl(d/'functions.jsonl')},key=lambda x:int(x,16));codefile=export(c,r,entries,root/'ghidra_scripts/ExportSeraphCodeV730.py',args.function_timeout);code=load(codefile)
                    base=load(c['previous']/'source_index_audit.json').get('addresses',{}).get(sid,{}).get('ghidra_image_base')
                    if base is not None and code.get('image_base') is not None and address(base)!=address(code['image_base']):raise ValueError('Fresh Ghidra image base differs; no silent relocation')
                    dump(d/'code_reference.json',{'path':str(codefile),'sha256':sha(codefile),'raw_Npcode_equals_decompiler_index':False,'LLM_token_vectors_created':False});count['code_OK']=sum(f['status']=='OK' for f in code['functions']);count['code_unavailable']=len(entries)-count['code_OK'];dump(d/'sample_summary.json',count)
                    outputs={str(p):sha(p) for p in d.rglob('*') if p.is_file()};outputs[str(codefile)]=sha(codefile);dump(cp,{'fingerprint':fp,'outputs':outputs})
                dirs.append(d);counts.update(count)
            state.update(stage='04_SSQ');dump(out/'state.json',state);print('[STAGE 04] Prepare SSQ (offline)',flush=True);ssq=build(dirs,out,args.max_requests)
            def functions():
                for d in dirs:
                    for f in jsonl(d/'functions.jsonl'):
                        f['query_bundle_file']=str((d/'queries.jsonl').relative_to(out));f['edge_catalog_file']=str((d/'edges.jsonl').relative_to(out));f['code_reference_file']=str((d/'code_reference.json').relative_to(out));yield f
            write_jsonl(out/'function_semantics_baseline.jsonl',functions())
            status='PREPARED_WITH_GAPS' if counts['code_unavailable'] or counts['unlocated_query_module_groups'] or not ssq['selected_requests'] else 'PREPARED'
            report={'schema':'seraph_task1_report_v7_3_0','status':status,'samples':len(dirs),'counts':dict(counts),'SSQ':ssq,'source_index_dir':str(c['previous']),'source_run':str(c['run']),'output_dir':str(out),'LLM_called':False,'LLM_gated_embeddings_trained':False,'LoRA_joint_training_completed':False,'LLM_calibrated':False,'document_four_evidence_fusion_completed':False,'S_global_recomputed':False,'original_scores_modified':False,'security_detector':'OUT_OF_SCOPE_TASK1_ONLY'}
            dump(out/'task1_report.json',report)
            text=['# 任务1 SERAPH 第一批集成 v7.3.0','','状态：'+status,'样本：%s；函数：%s；查询-模块组：%s。'%(len(dirs),counts['functions'],counts['query_module_groups']),'伪C成功函数：%s；不可用函数：%s；准备的SSQ：%s。'%(counts['code_OK'],counts['code_unavailable'],ssq['selected_requests']),'','## 已完成','冻结已有基线、函数级四模块query候选视图、stripped伪C/直接PC-token桥、四区段SSQ队列、同范围输出。','','## 未完成且未用占位结果冒充','没有生成LLM隐藏向量、门控融合训练、LoRA联合训练、校准或把第四证据接回原闭环；本次准备不发送网络请求。','代码token地址关联不是LLM hidden-state对齐。未匹配位置保留未知，不使用最近地址猜测。','函数级视图保留query语义，不将不同变量候选用投票混成一个签名。','','## SSQ未入队原因',canonical(ssq['reason_counts']),'','## 下一步','检查code_OK及SSQ队列，再配置实际模型；原confirmed/rejected状态保持。']
            (out/'task1_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8');unchanged(cert)
            if sha(c['run']/'candidate_rankings.csv')!=cert['files'][str(c['run']/'candidate_rankings.csv')]['sha256']:raise ValueError('Original ranking changed')
            outputs={str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('state.json','prepare_checkpoint.json') and 'attempts' not in p.parts}
            for d in dirs:
                ref=load(d/'code_reference.json');outputs[ref['path']]=ref['sha256']
            dump(out/'prepare_checkpoint.json',{'fingerprint':digest(plan),'outputs':outputs});dump(out/'state.json',{**state,'stage':'05_report','status':status,'completed_utc':stamp(),'inference_rerun':False,'original_ranking_unchanged':True})
            print('SERAPH TASK1 PREPARED (LLM NOT CALLED; NOT FULL SERAPH TRAINING)\n'+str(out/'task1_report.md'),flush=True);return 0
        except Exception as e:
            dump(out/'state.json',{**state,'status':'FAILED','error':str(e),'failed_utc':stamp(),'resume':'same action/tag; do not edit old checkpoints'});raise
