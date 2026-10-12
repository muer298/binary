"""Function-indexed views of existing hypotheses, without conflating variables."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
from .common import *
CONFIRMED={'confirmed_hard','confirmed_soft'}
HIGH={'id','function_entry','entry_address','api_name','api_source','semantic_template','parameter_type_sem','return_type_sem','local_var_type_sem','params','ret','interface_constraints','type_candidates','type_confidence','type_category','type_name','pred_type','type_size','byte_size','bit_width','signedness','is_signed','is_pointer','pointer_depth','pointee_type','pointee_kind','pointee_byte_size','param_index','parameter_index','param_count','parameter_count','role_candidates','role_confidence','base_obj','object_size','offset_set','field_relations','eval_field_offsets','access_patterns','memory_access_patterns','support_sites','layout_kind','symbol_resolution','deterministic_api_source'}
LOW={'id','address','instruction_address','function_entry','pcode_index','pcode_site_id','opcode','mnemonic','assembly','npcode','np_code','semantics','effects','input_value_ids','output_value_ids','inputs','output'}
BAD=('gold','dwarf','oracle','ground_truth','full_path','full_binary','source_code_path')

def clean(x,depth=0):
    if depth>12:return {'truncated':'depth_limit'}
    if isinstance(x,dict):return {k:clean(v,depth+1) for k,v in x.items() if not any(b in str(k).lower() for b in BAD)}
    if isinstance(x,list):return [clean(v,depth+1) for v in x]
    return x

def payload(n,low=False):
    if not n:return {},'node_missing'
    raw=loads(n['raw']) if n['raw'] else {};a=raw or loads(n['attrs'] or '{}')
    return clean({k:v for k,v in a.items() if k in (LOW if low else HIGH)}),n['raw_source'] or 'graph_attrs'

def unit_note(d):
    x={};b=d.get('type_size',d.get('byte_size'));w=d.get('bit_width')
    if type(b) is int and b>0:x.update(declared_byte_size=b,bit_width_from_declared_byte_size=b*8)
    if type(w) is int and w>0:
        x['explicit_bit_width']=w
        if type(b) is int and b>0 and b*8!=w:x['unit_conflict']=True
    x['policy']='annotation_only; no original field or score overwrite'
    return x

def extract(dbpath,sid,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);db=connect_ro(dbpath)
    try:
        groups=defaultdict(list)
        for r in db.execute('SELECT * FROM ranks WHERE sid=? ORDER BY qid,module,rank',(sid,)):
            if r['module'] not in MODULES:raise ValueError('Unknown module')
            groups[(r['qid'],r['module'])].append(dict(r))
        if not groups:raise ValueError('No ranking groups for sample '+sid)
        ids=sorted({q for q,m in groups}|{r['tid'] for rows in groups.values() for r in rows});wanted=set(ids);nodes={}
        for k in range(0,len(ids),400):
            part=ids[k:k+400]
            for r in db.execute('SELECT * FROM nodes WHERE sid=? AND id IN ('+','.join('?' for _ in part)+')',[sid]+part):nodes[r['id']]=dict(r)
        edges={};adj=defaultdict(list)
        for r in db.execute('SELECT * FROM edges WHERE sid=? ORDER BY id',(sid,)):
            if r['src'] not in wanted and r['dst'] not in wanted:continue
            eid=r['id'];edges[eid]={'edge_id':eid,'source_node':r['src'],'target_node':r['dst'],'edge_kind':r['kind'],'edge_type':r['typ'],'raw_origin_validated':bool(r['origin_valid']),'source_locator':r['path'],'predicate_satisfaction_available':False}
            adj[r['src']].append(eid)
            if r['dst']!=r['src']:adj[r['dst']].append(eid)
        rows=[];functions={};stats=Counter()
        for (qid,m),rs in sorted(groups.items()):
            if [r['rank'] for r in rs]!=list(range(1,len(rs)+1)) or len({r['tid'] for r in rs})!=len(rs):raise ValueError('Candidate ranks not consecutive/unique')
            if len({(r['dstatus'],r['dtid']) for r in rs})!=1:raise ValueError('Conflicting decisions')
            q=nodes.get(qid);qp,how=payload(q,True);fn=hx(q['fn'] if q else None);pc=hx(q['pc'] if q else None) or hx(qp.get('address'));gid=digest([sid,qid,m]);candidates=[]
            for r in rs:
                n=nodes.get(r['tid']);a,source=payload(n);tfn=hx(n['fn'] if n else None)
                scope='unknown' if not(fn and tfn) else 'same_function' if fn==tfn else 'other_function_template'
                candidates.append({'target_id':r['tid'],'rank':r['rank'],'score_before_223':finite(r['before']),'score_after_223':finite(r['after']),'candidate_status':r['cstatus'],'origin_function_entry':tfn,'scope_relation':scope,'attributes':a,'attribute_source':source,'unit_annotations':unit_note(a),'encoder_consumption_proven':False})
                stats['candidate_scope_'+scope]+=1
            rec={'schema':'seraph_query_v7_3_0','group_id':gid,'sample_id':sid,'query_id':qid,'semantic_type':m,'function_entry':fn,'instruction_address':pc,'instruction':qp,'instruction_source':how,'candidates':candidates,'decision':{'status':rs[0]['dstatus'],'target_id':rs[0]['dtid'] or None,'locked':rs[0]['dstatus'] in CONFIRMED,'preserved_from_original':True},'evidence_edge_ids':sorted(set(adj[qid])),'constraint_edge_ids':sorted({eid for c in candidates for eid in adj[c['target_id']] if 'constraint' in (edges[eid]['edge_kind'] or '').lower()}),'full_binding_proof_exported':False}
            rows.append(rec);stats['query_module_groups']+=1;stats['candidates']+=len(rs);stats['module_'+m]+=1
            if fn:
                f=functions.setdefault(fn,{'schema':'seraph_function_baseline_v7_3_0','sample_id':sid,'function_entry':fn,'function_id':sid+'::fn::'+fn,'query_scoped_semantics':{x:[] for x in MODULES},'query_bundle_file':'queries.jsonl','edge_catalog_file':'edges.jsonl','unified_function_signature_proven':False})
                f['query_scoped_semantics'][m].append(gid)
            else:stats['unlocated_query_module_groups']+=1
        stats['functions']=len(functions);stats['queries']=len({q for q,m in groups})
        write_jsonl(dest/'queries.jsonl',rows);write_jsonl(dest/'functions.jsonl',[functions[f] for f in sorted(functions)]);write_jsonl(dest/'edges.jsonl',[edges[e] for e in sorted(edges)])
        dump(dest/'baseline_summary.json',{'sample_id':sid,'counts':dict(stats),'Gold_read':False,'original_scores_unchanged':True,'scope_note':'Other-function candidates are templates, not facts of this function.'})
        return dict(stats)
    finally:db.close()
