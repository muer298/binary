#!/usr/bin/env python3
"""Resume an immutable partial snapshot through the original frozen inference scripts.

No training, Gold-based selection, or promotion to Formal Test is performed.
Checkpoints are whole-stage checkpoints, not arbitrary instruction checkpoints.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict

from control import HERE, dump, read_json, sha, stamp, lock, FROZEN_FILES, safe_id
VENDOR=HERE/'vendor/pipeline'
HELPERS=HERE/'helpers'
TYPES=('APISem','TypeSem','LayoutSem','InterfaceSem')


def code_hash(upstream):
    h=hashlib.sha256()
    roots=[HERE,Path(upstream)/'01_modified_upstream']
    for root in roots:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.py','.sh') and '__pycache__' not in p.parts:
                h.update(str(p.relative_to(root)).encode());h.update(bytes.fromhex(sha(p)))
    return h.hexdigest()


def verify_snapshot(root):
    report=read_json(root/'snapshot.json')
    for name,digest in report['files_sha256'].items():
        p=root/name
        if sha(p)!=digest:raise RuntimeError('固定子集文件被修改：'+str(p))
    return report


class Steps:
    def __init__(self, root, up):
        self.root=root;self.code=code_hash(up)
        self.steps=root/'steps';self.steps.mkdir(exist_ok=True)
        self.checkpoints=root/'checkpoints';self.checkpoints.mkdir(exist_ok=True)
        self.env=dict(os.environ)
        # Remove evaluation labels and inherited previous-run inputs. Only inputs
        # explicitly supplied below reach child inference processes.
        for key in list(self.env):
            if ('GOLD' in key or key in {'API_PROVENANCE','QUERY_WHITELIST_JSONL',
                'SAMPLE_SPLIT','SAMPLE_SPLIT_MANIFEST','MODULE_WEIGHTS','MODULE_HEAD_CHECKPOINT',
                'EXPECTED_RECORDS','MAX_QUERIES','MAX_NODES','MAX_EDGES','TASK2_ROOT','WORK_ROOT',
                'LAYOUT_RERANK_CONFIG','CONFIRM_POLICY_FILE','EMBEDDINGS','INDEX','RESOLUTION_MANIFEST',
                'REPRESENTATION_MANIFEST','OUT_2216','OUT_222_V5_6','OUT_223_V6_6','BINDING_OUT',
                'MIN_HARD_API_SEEDS','ALLOW_ZERO_HARD_API_SEEDS','DIRECTION_WEIGHT_OVERRIDES'}):
                self.env.pop(key,None)
        for key in ('MAX_ITERATIONS','GLOBAL_THRESHOLD','STABLE_ROUNDS','REPRESENTATION_STEP',
            'EMBEDDING_RESCORE_BLEND','SCORE_FEEDBACK_BLEND','MAX_SCORE_DROP_PER_ROUND',
            'MAX_SCORE_RISE_PER_ROUND','SOFT_SWITCH_MARGIN','SOFT_RELEASE_MARGIN',
            'SOFT_RELEASE_GAP_MARGIN','RETRIEVAL_INTERVAL','RETRIEVAL_TOP_K_PER_TYPE',
            'RETRIEVAL_MAX_CANDIDATES_PER_TYPE','RETRIEVAL_MIN_PROBABILITY',
            'RETRIEVAL_REPLACEMENT_MARGIN','RETRIEVAL_QUERY_BATCH_SIZE',
            'RETRIEVAL_TARGET_BATCH_SIZE','RETRIEVAL_DEVICE','ITERATIVE_RETRIEVAL',
            'LEGACY_MONOTONIC_SCORE_UPDATES','LOCK_SOFT_CONFIRMATIONS','ENFORCE_RAW_MONOTONIC',
            'MIN_GLOBAL_DELTA','MAX_STATE_CHANGE_RATE_FOR_STABLE','TOP_K_PER_TYPE',
            'MAX_TOTAL_CANDIDATES','QUERY_BATCH_SIZE','TARGET_BATCH_SIZE',
            'REQUIRE_RETRIEVED_CANDIDATE','REQUIRE_SCORE_DECREASE',
            'MAX_NODES_PER_SIDE','CALLCTX_PRE','CALLCTX_POST','MAX_LOW_NODES','PARAMFLOW_PRE',
            'PARAMFLOW_POST','PARAMFLOW_HOPS','MAX_FLOW_NODES','MAX_CALL_CANDIDATES',
            'TRACE_NODE_ID','COMMIT_EVERY','MIN_RESULTS','PROGRESS_EVERY'):
            self.env.pop(key,None)
        self.env['PYTHON_BIN']=sys.executable
        self.env['PYTHONUNBUFFERED']='1'
        # Old launchers use python3 literally. Use the same active Python environment.
        self.env['PATH']=str(Path(sys.executable).parent)+os.pathsep+self.env.get('PATH','')
        self.env.update(PACKAGE_ROOT=str(VENDOR),TASK2_SRC=str(VENDOR),EXT_DIR=str(HELPERS))
        self.active=None
        self.lock_fd=None
        self.progress=self.root/'pipeline_progress.json'

    def call(self, command, extra=None, accepted=(0,)):
        cmd=[str(x) for x in command]
        env=dict(self.env);env.update({k:str(v) for k,v in (extra or {}).items()})
        print('$ '+shlex.join(cmd),flush=True)
        with (self.active/'commands.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'cmd':cmd,'environment_overrides':extra or {}},ensure_ascii=False)+'\n')
        with (self.active/'stage.log').open('a',encoding='utf-8') as log:
            log.write('$ '+shlex.join(cmd)+'\n');log.flush()
            proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1,
                pass_fds=() if self.lock_fd is None else (self.lock_fd,))
            try:
                for line in proc.stdout:
                    log.write(line);log.flush();print(line,end='',flush=True)
                rc=proc.wait()
            finally:
                if proc.poll() is None:proc.terminate();proc.wait()
        if rc not in accepted:raise RuntimeError('命令失败 rc=%s；日志 %s'%(rc,self.active/'stage.log'))
        return rc

    def step(self, name, inputs, callback, options=None):
        fingerprint={'code_sha256':self.code,'inputs':{str(Path(p)):sha(p) for p in inputs},'options':options or {}}
        mark=self.checkpoints/(name+'.json')
        if mark.exists():
            m=read_json(mark)
            if m['fingerprint']!=fingerprint:
                raise RuntimeError('检查点输入/参数/代码发生变化：'+name+'；请用新快照/新实验目录，不能沿用旧结果。')
            for key,item in m['outputs'].items():
                if not Path(item['path']).is_file() or sha(item['path'])!=item['sha256']:
                    raise RuntimeError('已完成阶段产物改变或缺失：'+item['path'])
            print('[SKIP VERIFIED STAGE] '+name,flush=True)
            return {k:Path(v['path']) for k,v in m['outputs'].items()}
        dest=self.steps/name
        if dest.exists():
            archive=self.root/'attempts';archive.mkdir(exist_ok=True)
            dest.rename(archive/(name+'_'+stamp()+'_'+str(os.getpid())))
        dest.mkdir();self.active=dest
        t=time.time();dump(self.progress,{'stage':name,'status':'RUNNING','started_utc':stamp()})
        print('\n========== STAGE '+name+' ==========',flush=True)
        try:
            outputs=callback(dest)
            for k,p in outputs.items():
                if not Path(p).is_file():raise RuntimeError('缺少阶段产物：'+str(p))
            packed={k:{'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size} for k,p in outputs.items()}
            dump(mark,{'stage':name,'elapsed_sec':time.time()-t,'fingerprint':fingerprint,'outputs':packed})
            dump(self.progress,{'stage':name,'status':'DONE','elapsed_sec':time.time()-t})
            print('[STAGE DONE] %s elapsed=%.1fs'%(name,time.time()-t),flush=True)
            return {k:Path(p) for k,p in outputs.items()}
        except Exception:
            dump(self.progress,{'stage':name,'status':'FAILED','elapsed_sec':time.time()-t,
                'resume':'same command; only this incomplete stage is retried'})
            raise


def node_metadata(path):
    result={}
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            n=json.loads(line);a=n.get('attrs') or {}
            result[n['node_id']]={'sample_id':n.get('sample_id'),'node_type':n.get('node_type'),
                'function_entry':n.get('function_entry') or a.get('function_entry'),
                'address':a.get('address') or a.get('location'),'raw_id':n.get('raw_id')}
    return result


def export_rankings(final_path, nodes_path, out, snapshot_report, summary223):
    meta=node_metadata(nodes_path);query_count=0;groups=Counter();cand_counts=Counter();top_changes=Counter();qtypes=Counter()
    csv_path=out/'candidate_rankings.csv';jl=out/'candidate_rankings.jsonl'
    cols=['sample_id','query_id','query_type','function_entry','address','semantic_type','rank',
          'target_id','score_before_223','score_after_223','candidate_status','decision_status','decision_target_id']
    with open(final_path,encoding='utf-8') as f,csv_path.open('w',newline='',encoding='utf-8') as cf,jl.open('w',encoding='utf-8') as jf:
        writer=csv.DictWriter(cf,fieldnames=cols);writer.writeheader()
        for line in f:
            if not line.strip():continue
            row=json.loads(line);qid=str(row.get('query_id') or '')
            qm=meta.get(qid,{})
            qt=qm.get('node_type') or row.get('query_type') or 'unknown'
            qtypes[qt]+=1;query_count+=1;by=defaultdict(list)
            for c in row.get('candidates') or []:
                typ=c.get('target_semantic_type')
                if typ in TYPES:by[typ].append(c)
            for typ,candidates in by.items():
                ordered=sorted(candidates,key=lambda c:(-float(c.get('score_223_v5_3',c.get('confidence',0.0)) or 0.0),str(c.get('target_id') or '')))
                old=sorted(candidates,key=lambda c:(-float(c.get('score_before_223_v5_3',c.get('confidence',0.0)) or 0.0),str(c.get('target_id') or '')))
                groups[typ]+=1;cand_counts[typ]+=len(ordered)
                top_changes[typ]+=int(str(old[0].get('target_id'))!=str(ordered[0].get('target_id')))
                dec=(row.get('semantic_decisions') or {}).get(typ) or {}
                for rank,c in enumerate(ordered,1):
                    rec={'sample_id':row.get('sample_id') or qm.get('sample_id'),'query_id':qid,
                        'query_type':qt,'function_entry':qm.get('function_entry'),'address':qm.get('address'),
                        'semantic_type':typ,'rank':rank,'target_id':c.get('target_id'),
                        'score_before_223':c.get('score_before_223_v5_3',c.get('confidence')),
                        'score_after_223':c.get('score_223_v5_3',c.get('confidence')),
                        'candidate_status':c.get('candidate_status'),'decision_status':dec.get('status'),
                        'decision_target_id':dec.get('best_target_id')}
                    writer.writerow(rec);jf.write(json.dumps(rec,ensure_ascii=False)+'\n')
    if not query_count:raise RuntimeError('没有候选查询；不能标记排名完成。')
    result={'schema':'partial_candidate_ranking_summary_v1','scope':'partial_diagnostic',
        'snapshot_samples':snapshot_report['selected_samples'],'full_manifest_ok_rows':snapshot_report['requested_ok_rows'],
        'raw_instruction_nodes':snapshot_report['counts']['instructions'],
        'feature_node_count':len(meta),'query_rows':query_count,'query_types':dict(qtypes),
        'candidate_universe':'high-level nodes in this partial feature graph; not the full-corpus candidate pool',
        'modules':{t:{'query_groups':groups[t],'candidates':cand_counts[t],
            'top1_changed_since_222':top_changes[t], 'recall_at_1':None,'recall_at_5':None,'MRR':None} for t in TYPES},
        'gold_evaluation':'NOT_RUN','strong_label_count':None,'selection_gate':'NOT_RUN',
        'formal_test_status':'NOT_RUN',
        'score_sort':'all candidates sorted descending score_223_v5_3, target_id tie-break, consistent with original evaluator',
        'confirmation_note':'score rank is not the locked semantic decision; rejected candidate flags retained',
        'provenance_sidecar_used_for_scoring':False,
        'collaboration_summary':str(summary223)}
    dump(out/'rank_summary.json',result)
    return {'csv':csv_path,'jsonl':jl,'summary':out/'rank_summary.json'}


def run_pipeline(root, up, frozen=None, mode='rank'):
    with lock(root/'.pipeline.lock') as lease:
        snap=verify_snapshot(root);S=Steps(root,up);S.lock_fd=lease.fileno()
        def graph(d):
            idx=d/'usg_index.jsonl';nodes=d/'usg_nodes.jsonl';edges=d/'usg_edges.jsonl'
            S.call([sys.executable,up/'01_modified_upstream/01_build_usg_index_v7.py',
                '--json-root',root/'semantic_json','--out',idx,'--stats',d/'index_stats.json'])
            S.call([sys.executable,up/'01_modified_upstream/02_extract_nodes_edges_blocksem_v7.py',
                '--index',idx,'--nodes-out',nodes,'--edges-out',edges,'--stats',d/'graph_stats.json'])
            a=read_json(d/'index_stats.json');b=read_json(d/'graph_stats.json')
            if a.get('errors') or b.get('errors'):raise RuntimeError('图构建报告含错误；请看 graph_stats.json')
            if a.get('samples')!=snap['selected_samples']:raise RuntimeError('图样本数与快照不一致')
            if nodes.stat().st_size==0 or edges.stat().st_size==0:raise RuntimeError('空 USG')
            # Catch the exact v7.0.1 regression before spending hours on features/embeddings:
            # deterministic API provenance must survive semantic_dict -> USG.
            if int(b.get('hard_api_source_count') or 0)<=0:
                raise RuntimeError('USG 中 deterministic API provenance=0；拒绝继续。必须使用 v7.0.2+ extractor。')
            if int(b.get('direct_edges_with_api_provenance') or 0)<=0:
                raise RuntimeError('USG 中 direct API provenance edges=0；2.2.2 hard-seed path 将不可用。')
            return {'index':idx,'nodes':nodes,'edges':edges,'graph_stats':d/'graph_stats.json','index_stats':d/'index_stats.json'}
        g=S.step('01_graph',[root/'snapshot.json'],graph)
        if mode=='graph':return
        if frozen is None:raise RuntimeError('冻结模型未指定')
        import torch
        missing=[n for n in FROZEN_FILES if not (frozen/n).is_file()]
        if missing:raise RuntimeError('冻结模型缺文件：'+str(missing))
        conf=read_json(frozen/'layout_reranker_config.json')
        if conf.get('test_used_for_selection') is not False:
            raise RuntimeError('冻结 Layout config 不满足 test_used_for_selection=false；不自动修改配置。')
        head=frozen/'module_head_checkpoint.pt'
        directions=''
        migration=frozen/'migration_manifest.json'
        if migration.is_file():
            dm=(read_json(migration).get('protocol') or {}).get('direction_weight_overrides') or {}
            if dm:directions=','.join('%s=%s'%(k,v) for k,v in sorted(dm.items()))
        query_types=os.environ.get('PARTIAL_QUERY_TYPES','InstrSem')
        device=os.environ.get('DEVICE','auto')
        frozen_info={n:sha(frozen/n) for n in FROZEN_FILES}
        if head.is_file():frozen_info[head.name]=sha(head)
        plan={'protocol':'subset100_diagnostic_frozen_inference','source_pipeline':'research2_pure_scripts_final_v6_7_4',
            'frozen_dir':str(frozen),'frozen_files_sha256':frozen_info,'training_invoked':False,
            'gold_used_for_inference':False,'parameter_selection_invoked':False,
            'query_types':query_types,'device':device,'optional_module_heads':head.is_file(),
            'directions':directions or 'original_pure_script_defaults',
            'selection_and_formal_gates':'not run, not lowered, not passed',
            'CFI_provenance':'separate audit; not an automatic input to original ranker',
            'scope':'deterministic 100-sample diagnostic subset; not Formal Test'}
        plan_path=root/'inference_plan.json'
        if plan_path.exists() and read_json(plan_path)!=plan:
            raise RuntimeError('冻结推理计划变化；请新建实验，不复用已有阶段。')
        dump(plan_path,plan)
        S.env.update(CONFIRM_POLICY_FILE=str(frozen/'confirmation_policy.json'),
            LAYOUT_RERANK_CONFIG=str(frozen/'layout_reranker_config.json'),DEVICE=device)
        def features(d):
            env={'RUN_TAG':'partial','ANCHOR_DIR':str(d/'anchors'),'GRAPH_DIR':str(d/'graphs'),
                'FEATURE_DIR':str(d/'features'),'REPORT_DIR':str(d/'reports'),
                'RAW_NODES':str(g['nodes']),'EDGES':str(g['edges']),
                'FROZEN_FEATURE_VOCAB':str(frozen/'feature_vocab.json'),
                'SELECTED_NODES':str(d/'graphs/selected.jsonl'),
                'API_ENRICHED_NODES':str(d/'graphs/api_enriched.jsonl'),
                'NODES':str(d/'graphs/model_input.jsonl'),
                'NEIGHBORHOODS':str(d/'anchors/final.jsonl'),
                'NODE_FEATURES':str(d/'features/node_features.pt'),
                'EXTERNAL_FEATURE_VOCAB':str(d/'features/feature_vocab.json'),
                'ANCHOR_LIMITS':'fixed_offset=10000,deref_chain=10000,def_use=10000,api_call=10000'}
            S.call(['bash',HELPERS/'06_features.sh'],env)
            api_audit=d/'reports/api_provenance_model_input.json'
            S.call([sys.executable,HELPERS/'audit_api_provenance_nodes.py',
                '--nodes',env['NODES'],'--out',api_audit])
            return {'nodes':Path(env['NODES']),'features':Path(env['NODE_FEATURES']),
                'vocab':Path(env['EXTERNAL_FEATURE_VOCAB']),'neighborhoods':Path(env['NEIGHBORHOODS']),
                'contract':d/'reports/frozen_feature_contract_partial.json','api_provenance':api_audit}
        f=S.step('02_features',[g['nodes'],g['edges'],frozen/'feature_vocab.json',plan_path],features)
        def embeddings(d):
            S.call([sys.executable,VENDOR/'10_export_node_embeddings_v6_3.py',
                '--checkpoint',frozen/'alignment_model_best.pt','--stage3-metrics',frozen/'alignment_metrics.json',
                '--node-features',f['features'],'--nodes',f['nodes'],'--out-dir',d,'--device',device,
                '--batch-size','8192','--max-nodes','0','--input-hash-policy','sha256',
                '--require-node-metadata-coverage','--require-exact-node-set'])
            S.call([sys.executable,VENDOR/'11_verify_node_embedding_export_v6_3.py',
                '--embeddings',d/'node_embeddings.pt','--index',d/'node_embedding_index.json',
                '--manifest',d/'node_embedding_export_manifest.json','--report',d/'verification.json','--require-formal'])
            return {'embeddings':d/'node_embeddings.pt','index':d/'node_embedding_index.json',
                'manifest':d/'node_embedding_export_manifest.json','verification':d/'verification.json'}
        e=S.step('03_embeddings',[f['features'],f['nodes'],frozen/'alignment_model_best.pt',frozen/'alignment_metrics.json',plan_path],embeddings)
        def binding_conflict(d):
            env={'WORK_ROOT':str(d),'MODE':'full','RUN_TAG':'partial','NODES':str(f['nodes']),'EDGES':str(g['edges']),
                'EMBED_DIR':str(e['manifest'].parent),'EMBEDDINGS':str(e['embeddings']),
                'INDEX':str(e['index']),'EMBED_MANIFEST':str(e['manifest']),
                'BINDING_OUT':str(d/'binding'),'BINDING_STORE':str(d/'binding/node_binding_store.sqlite3'),
                'OUT_2216':str(d/'conflict'),'APPLY_EVENTS':'1'}
            S.call(['bash',VENDOR/'run_221_binding.sh'],env)
            S.call(['bash',VENDOR/'run_221_conflict.sh'],env)
            return {'embeddings':d/'conflict/node_embeddings_resolved_v6_5.pt',
                'index':d/'conflict/node_embedding_index_resolved_v6_5.json',
                'manifest':d/'conflict/resolution_manifest_v6_5.json',
                'store':d/'binding/node_binding_store.sqlite3'}
        b=S.step('04_binding_conflict',[f['nodes'],g['edges'],e['embeddings'],e['index'],e['manifest'],plan_path],binding_conflict)
        def reasoning(d):
            (d/'reports').mkdir()
            env={'TASK2_ROOT':str(d),'OUT_222_V5_6':str(d/'reasoning'),
                'NODES':str(f['nodes']),'EDGES':str(g['edges']),'EMBEDDINGS':str(b['embeddings']),
                'INDEX':str(b['index']),'RESOLUTION_MANIFEST':str(b['manifest']),
                'REPRESENTATION_MANIFEST':str(e['manifest']),'FORMAL_REPRESENTATION_INPUT':'1',
                'MAX_QUERIES':'0','QUERY_TYPES':query_types,'MIN_HARD_API_SEEDS':'1',
                'ALLOW_ZERO_HARD_API_SEEDS':'0'}
            S.call(['bash',VENDOR/'run_222_v5_6.sh'],env)
            ev=read_json(d/'reasoning/evaluation_v5_3.json')
            val=read_json(d/'reasoning/validation_summary_v5_3.json')
            if int(ev.get('gold_queries') or 0)!=0:raise RuntimeError('Gold unexpectedly reached 2.2.2')
            if int(val.get('hard_api_seed_count') or 0)<1:
                raise RuntimeError('2.2.2 finished without hard API seeds; this clean package must not lower the gate.')
            if ev.get('pass_structural_checks') is not True:
                raise RuntimeError('2.2.2 evaluator structural checks failed; inspect evaluation_v5_3.json')
            return {'scope':d/'reasoning/reasoning_scope_for_223_v5_3.jsonl',
                'summary':d/'reasoning/reasoning_summary_v5_3.json',
                'final':d/'reasoning/reasoning_final_v5_3.jsonl','evaluation':d/'reasoning/evaluation_v5_3.json'}
        r=S.step('05_reasoning_222',[f['nodes'],g['edges'],b['embeddings'],b['index'],b['manifest'],e['manifest'],plan_path],reasoning)
        def collaboration(d):
            args=[sys.executable,VENDOR/'19_run_four_module_inference_v6_6.py','--reasoning',r['scope'],
                '--nodes',f['nodes'],'--edges',g['edges'],'--layout-config',frozen/'layout_reranker_config.json',
                '--out-dir',d/'modules','--summary-out',d/'four_module_summary_v6_6.json']
            if head.is_file():args+=['--head-checkpoint',head]
            S.call(args)
            sm=read_json(d/'four_module_summary_v6_6.json')
            if int(sm.get('loaded_node_metadata') or 0)<=0:raise RuntimeError('2.2.3 未读取节点属性')
            if int(sm.get('loaded_incident_edge_ids') or 0)<=0:raise RuntimeError('2.2.3 未读取邻接边上下文')
            # Scoring diagnostics only; do NOT label this as passing the Layout promotion gate.
            dump(d/'promotion_status.json',{'status':'NOT_RUN','reason':'diagnostic candidate rankings only',
                'fixed_selection_thresholds':{'strong':100,'sortable_baseline_top1_errors':10},
                'module_counts':sm.get('module_counts')})
            mw=float(conf.get('recommended_module_weight',.25))
            weights='APISem=0.25,TypeSem=0.25,LayoutSem=%s,InterfaceSem=0.25'%mw
            args=[sys.executable,VENDOR/'20_iterative_module_collaboration_v6_6.py',
                '--reasoning',r['scope'],'--edges',g['edges'],
                '--confirm-policy',frozen/'confirmation_policy.json',
                '--max-iterations','10','--stable-rounds','2','--global-threshold','0.85',
                '--module-weight','0.25','--module-weights',weights,
                '--propagation-step','0.25','--max-score-change-per-round','0.08','--ablation','full',
                '--out',d/'collaborative_result_v6_6.jsonl','--summary-out',d/'collaboration_summary_v6_6.json']
            if directions:args+=['--direction-weight-overrides',directions]
            for mod in ('apiinfer','typeinfer','layoutinfer','interfaceinfer'):
                args+=['--module-result',d/'modules'/(mod+'_v6_6.jsonl')]
            S.call(args)
            S.call([sys.executable,VENDOR/'21_evaluate_223_v6_6.py','--baseline',r['scope'],
                '--input',d/'collaborative_result_v6_6.jsonl','--collaboration-summary',d/'collaboration_summary_v6_6.json',
                '--ablation','full','--out',d/'evaluation_no_gold.json'])
            ev=read_json(d/'evaluation_no_gold.json')
            if int((ev.get('gold_audit') or {}).get('unique_queries') or 0)!=0:raise RuntimeError('Gold unexpectedly reached 2.2.3')
            return {'final':d/'collaborative_result_v6_6.jsonl','summary':d/'collaboration_summary_v6_6.json',
                'module_summary':d/'four_module_summary_v6_6.json','evaluation':d/'evaluation_no_gold.json',
                'promotion':d/'promotion_status.json'}
        c=S.step('06_collaboration_223',[r['scope'],g['edges'],f['nodes'],plan_path],collaboration)
        ranks=S.step('07_rank_export',[c['final'],f['nodes'],c['summary'],root/'snapshot.json'],
            lambda d:export_rankings(c['final'],f['nodes'],d,snap,c['summary']))
        for key,path in ranks.items():
            if key in ('csv','jsonl','summary'):shutil.copy2(path,root/path.name)
        dump(root/'RANKING_DONE.json',{'status':'SUBSET100_DIAGNOSTIC_RANKING_DONE','outputs':{k:str(root/p.name) for k,p in ranks.items()},
            'accuracy_evaluation':'NOT_RUN','selection_gate':'NOT_RUN','formal_test':'NOT_RUN'})
        print('\nSUBSET100 DIAGNOSTIC RANKING DONE\n'+str(root/'candidate_rankings.csv'),flush=True)


def run_evidence(root,up):
    """Optional independent CFI/provenance task; does not feed Gold to inference."""
    with lock(root/'.evidence.lock') as lease:
        verify_snapshot(root)
        import elftools  # Do not let the legacy extractor silently fall back to load_base=0.
        if not shutil.which('readelf'):raise RuntimeError('readelf 不在 PATH')
        with (root/'inference_manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        S=Steps(root,up);S.lock_fd=lease.fileno();S.progress=root/'evidence_progress.json';total=Counter()
        for r in rows:
            sid=safe_id(r['sample_id']);sem=root/'semantic_json'/(sid+'_semantic_dict.json')
            def sample(d,r=r,sid=sid,sem=sem):
                cfi=d/'cfi.jsonl';prov=d/'provenance.jsonl'
                rc=S.call([sys.executable,up/'02_cfi_provenance/extract_cfi_intervals_v7.py',
                    '--binary',r['stripped_path'],'--sample-id',sid,'--readelf','readelf','--out',cfi],accepted=(0,3))
                S.call([sys.executable,up/'02_cfi_provenance/build_selection_provenance_v7.py',
                    '--semantic-json',sem,'--cfi',cfi,'--out',prov])
                dump(d/'cfi_status.json',{'returncode':rc,'has_rows':cfi.stat().st_size>0})
                return {'stats':Path(str(prov)+'.stats.json'),'cfi':cfi,'provenance':prov,'cfi_status':d/'cfi_status.json'}
            outputs=S.step('evidence_'+sid,[sem,Path(r['stripped_path'])],sample)
            total.update(read_json(outputs['stats']))
        dump(root/'evidence_summary.json',{'samples':len(rows),'counts':dict(total),
            'is_strong_gold':False,'automatically_used_by_ranker':False})
