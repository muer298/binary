"""Post-hoc SSQ-only ranking ablation, not calibrated fusion or closed-loop SERAPH."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
from .common import *
from .client import verified,recommendations

def metrics(rs):
    n=len(rs)
    return {'gold_n':n,'candidate_coverage':sum(r is not None for r in rs)/n if n else None,'R1':sum(r==1 for r in rs)/n if n else None,'R5':sum(r is not None and r<=5 for r in rs)/n if n else None,'MRR':sum(1/r for r in rs if r is not None)/n if n else None}

def evaluate(out,gold,session,tag,baseline_only=False):
    verified(out);plan=load(out/'plan.json');active=set(plan['sample_ids']);qs={};qid_sid={}
    for sid in plan['sample_ids']:
        for q in jsonl(out/'samples'/sid/'queries.jsonl'):
            key=(sid,q['query_id'],q['semantic_type'])
            if key in qs:raise ValueError('Duplicate query module')
            qs[key]=q
            if q['query_id'] in qid_sid and qid_sid[q['query_id']]!=sid:raise ValueError('Query ID cross-sample collision')
            qid_sid[q['query_id']]=sid
    recs={} if baseline_only else recommendations(out,session)
    if not baseline_only and not recs:raise ValueError('No validated LLM output; use --baseline-only explicitly')
    labels={};excluded=Counter()
    for g in jsonl(gold):
        qid=str(g.get('query_id') or '');m=g.get('target_semantic_type');tid=g.get('target_id');sid=g.get('sample_id') or qid_sid.get(qid)
        if not qid or m not in MODULES or not tid or not sid:raise ValueError('Gold schema missing query/sample/target/module')
        k=(sid,qid,m)
        if sid not in active:excluded['outside_sample_scope']+=1;continue
        if k not in qs:excluded['outside_original_query_scope']+=1;continue
        if k in labels and labels[k]['target_id']!=tid:raise ValueError('Conflicting strict Gold labels')
        if k in labels:excluded['identical_duplicate_label']+=1
        labels[k]=g
    if not labels:raise ValueError('No overlapping Gold; cannot report zero accuracy')
    b=defaultdict(list);a=defaultdict(list);tr=defaultdict(Counter);rows=[];calls=Counter();by_sample=defaultdict(lambda:([],[]))
    for k,g in sorted(labels.items()):
        sid,qid,m=k;q=qs[k];old=[x['target_id'] for x in q['candidates']];new=list(old);r=recs.get(q['group_id'])
        if r:
            calls[m]+=1
            if not r['abstain']:
                if q['decision']['locked'] or set(r['ordered_target_ids'])!=set(old):raise ValueError('Recommendation violates candidate/decision contract')
                new=r['ordered_target_ids']
        tid=g['target_id'];br=old.index(tid)+1 if tid in old else None;ar=new.index(tid)+1 if tid in new else None
        trans='right_to_right' if br==1 and ar==1 else 'right_to_wrong' if br==1 else 'wrong_to_right' if ar==1 else 'wrong_to_wrong'
        b[m].append(br);a[m].append(ar);tr[m][trans]+=1;tr[m]['top1_changed']+=int(old[0]!=new[0]);by_sample[(sid,m)][0].append(br);by_sample[(sid,m)][1].append(ar)
        rows.append({'sample_id':sid,'query_id':qid,'module':m,'gold_target_id':tid,'gold_source':g.get('gold_source'),'baseline_rank':br,'SSQ_rank':ar,'baseline_top1':old[0],'SSQ_top1':new[0],'transition':trans,'LLM_validated':bool(r)})
    dest=out/'evaluation'/tag_ok(tag);fingerprint={'prepared_sha':sha(out/'prepare_checkpoint.json'),'gold_sha':sha(gold),'responses':digest(recs),'baseline_only':baseline_only}
    if (dest/'done.json').exists():
        cp=load(dest/'done.json')
        if cp['fingerprint']!=fingerprint:raise ValueError('Evaluation changed; use new --evaluation-tag')
        verify_outputs(cp);print((dest/'accuracy_report.md').read_text());return 0
    dest.mkdir(parents=True,exist_ok=True);mods={};table=[]
    for m in MODULES:
        mb,ma=metrics(b[m]),metrics(a[m]);net=tr[m]['wrong_to_right']-tr[m]['right_to_wrong']
        mods[m]={'baseline_223':mb,'SSQ_order_ablation':ma,'transitions':dict(tr[m]),'net_correct':net,'LLM_gold_n':calls[m]}
        table.append({'module':m,'gold_n':mb['gold_n'],'candidate_coverage':mb['candidate_coverage'],'baseline_R1':mb['R1'],'SSQ_R1':ma['R1'],'baseline_R5':mb['R5'],'SSQ_R5':ma['R5'],'baseline_MRR':mb['MRR'],'SSQ_MRR':ma['MRR'],'net_correct':net,'LLM_gold_n':calls[m]})
    report={'schema':'seraph_posthoc_SSQ_ablation_v7_3_0','modules':mods,'all_query_module_groups':len(qs),'strict_gold_groups':len(labels),'excluded_Gold':dict(excluded),'inputs':fingerprint,'scope':'DEVELOPMENT_DIAGNOSTIC_NOT_FORMAL_TEST','denominator':'Same original queries intersected with existing mapped Gold; missing candidate=miss. No Gold=unknown.','LLM_failure_unrequested_abstain_policy':'retain original baseline, never silently drop','confidence_calibrated':False,'document_four_evidence_fusion_completed':False,'S_global_recomputed':False,'original_decisions_modified':False}
    dump(dest/'accuracy_report.json',report);write_jsonl(dest/'query_outcomes.jsonl',rows);write_csv(dest/'module_metrics.csv',table,list(table[0]))
    write_jsonl(dest/'per_sample_metrics.jsonl',[{'sample_id':s,'module':m,'baseline':metrics(x[0]),'SSQ':metrics(x[1])} for (s,m),x in sorted(by_sample.items())])
    def fmt(x):return '未知' if x is None else '%.2f%%'%(x*100)
    text=['# 任务1：基线与SSQ独立排序建议的只读对照','','本表不是完整SERAPH四证据融合。未调用/失败/弃权保留基线；未校准分数不写回原状态。','','|模块|Gold n|候选覆盖|基线R@1|SSQ建议R@1|净正确|','|---|---:|---:|---:|---:|---:|']
    for r in table:text.append('|%s|%s|%s|%s|%s|%s|'%(r['module'],r['gold_n'],fmt(r['candidate_coverage']),fmt(r['baseline_R1']),fmt(r['SSQ_R1']),r['net_correct']))
    (dest/'accuracy_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8');dump(dest/'done.json',{'fingerprint':fingerprint,'outputs':{str(p):sha(p) for p in dest.iterdir() if p.is_file() and p.name!='done.json'}});print('\n'.join(text));return 0
