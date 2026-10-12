"""Gold-limited DESIGN-only evaluation for the frozen External SSQ branch.

HOLDOUT Gold is not JSON-decoded. No CALIBRATION/TEST path is used.
This script does not perform API inference or modify model artifacts.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


BASE = Path('/data/chaoni/mjy/resym_exp/training_runs/seraph_local')
OLD = BASE/'external_ssq_v0_4_0_dev'
SMOKE = BASE/'external_ssq_v0_4_0b_api_smoke_repair'
OUT = BASE/'external_ssq_v0_4_0c_design'
MANIFESTS = (BASE/'clean_joint_v0_2_10/dev_manifest.jsonl',
             BASE/'ssq_layout_dev_supplement_v0_3_1a/layout_records/mapped_layout_records.jsonl')
MODULES = ('TypeSem','InterfaceSem','LayoutSem')
ID_RE = re.compile(r'"record_id"\s*:\s*"([^"]+)"')


def write_new(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')


def write_lines_new(path,rows):
    if path.exists():
        raise FileExistsError(path)
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')


def design_split_rows():
    split = json.loads((OLD/'EXTERNAL_SSQ_SCORING_SPLIT_V2.json').read_text())
    rows = [r for r in split['records'] if r['external_ssq_split']=='EXTERNAL_SSQ_DESIGN']
    if len(rows)!=628:
        raise RuntimeError('DESIGN split count mismatch')
    return rows


def design_gold(ids):
    found={}
    for path in MANIFESTS:
        with path.open() as f:
            for line in f:
                match=ID_RE.search(line)
                if not match or match.group(1) not in ids:
                    continue
                row=json.loads(line)  # Only DESIGN rows are JSON-decoded.
                rid=row['record_id']
                if rid in found:
                    raise RuntimeError('duplicate DESIGN Gold record')
                found[rid]=row['gold_candidate_id']
    if set(found)!=set(ids):
        raise RuntimeError(f'DESIGN Gold coverage {len(found)}/{len(ids)}')
    return found


def rank(values,ids):
    return sorted(ids,key=lambda cid:(-values[cid],cid))


def normalized_scores(scores,ids):
    if set(scores)!=set(ids):
        raise RuntimeError('API candidate key mismatch')
    total=sum(float(scores[c]) for c in ids)
    if total<=0:
        return {c:1/len(ids) for c in ids}
    return {c:float(scores[c])/total for c in ids}


def metrics_for_module(rows):
    total=len(rows)
    valid=sum(r['api_valid'] for r in rows)
    stage_correct=sum(r['stage2_correct'] for r in rows)
    api_correct=sum(r['api_correct'] for r in rows)
    abstain=sum(r['api_abstain'] for r in rows)
    invalid=sum(not r['api_valid'] for r in rows)
    categories=collections.Counter(r['paired_category'] for r in rows)
    wrong_stage=[r for r in rows if not r['stage2_correct']]
    correct_stage=[r for r in rows if r['stage2_correct']]
    wc=categories['wrong_to_correct']
    cw=categories['correct_to_wrong']
    nonabstain=[r for r in rows if r['api_valid'] and not r['api_abstain']]
    good_margins=[r['api_margin'] for r in nonabstain if r['api_correct']]
    bad_margins=[r['api_margin'] for r in nonabstain if not r['api_correct']]
    wc_margins=[r['api_margin'] for r in rows if r['paired_category']=='wrong_to_correct']
    cw_margins=[r['api_margin'] for r in rows if r['paired_category']=='correct_to_wrong']
    return {
        'n':total,'valid':valid,'valid_rate':valid/total,'invalid':invalid,
        'abstain':abstain,'abstain_rate':abstain/total,
        'stage2':{'R@1':stage_correct/total,'MRR':sum(r['stage2_reciprocal_rank'] for r in rows)/total,
                  'R@5':sum(r['stage2_gold_rank']<=5 for r in rows)/total},
        'api_raw':{'R@1':api_correct/total,'MRR':sum(r['api_reciprocal_rank'] for r in rows)/total,
                   'R@5':sum(r['api_gold_rank'] is not None and r['api_gold_rank']<=5 for r in rows)/total},
        'paired':dict(categories),'wrong_to_correct':wc,'correct_to_wrong':cw,
        'both_correct':categories['both_correct'],'both_wrong':categories['both_wrong'],
        'net_correction':wc-cw,
        'stage2_wrong_count':len(wrong_stage),'stage2_correct_count':len(correct_stage),
        'api_top1_correct_within_stage2_wrong':wc,
        'api_gold_rank_improved_within_stage2_wrong':sum(r['gold_rank_improved'] for r in wrong_stage),
        'api_gold_highest_score_within_stage2_wrong':sum(r['gold_score_tied_highest'] for r in wrong_stage),
        'api_gold_unique_highest_score_within_stage2_wrong':sum(r['gold_score_unique_highest'] for r in wrong_stage),
        'rescue_precision':wc/(wc+cw) if wc+cw else None,
        'rescue_coverage':wc/len(wrong_stage) if wrong_stage else None,
        'harm_rate':cw/len(correct_stage) if correct_stage else None,
        'api_margin':{
            'all_nonabstain_mean':statistics.mean(r['api_margin'] for r in nonabstain) if nonabstain else None,
            'correct_mean':statistics.mean(good_margins) if good_margins else None,
            'wrong_mean':statistics.mean(bad_margins) if bad_margins else None,
            'wrong_to_correct_mean':statistics.mean(wc_margins) if wc_margins else None,
            'correct_to_wrong_mean':statistics.mean(cw_margins) if cw_margins else None,
        },
        'candidate_count_distribution':dict(collections.Counter(str(r['candidate_count']) for r in rows)),
        'function_group_count':len({r['function_group_id'] for r in rows}),
    }


def main():
    status=json.loads((SMOKE/'v0_4_0b_final_status.json').read_text())
    if status['status']!='READY_FOR_EXTERNAL_SSQ_DESIGN':
        raise RuntimeError('smoke gate invalid')
    split=design_split_rows()
    ids={r['record_id'] for r in split}
    by_id={r['record_id']:r for r in split}
    with (OLD/'EXTERNAL_SSQ_API_REQUESTS_V2.jsonl').open() as request_handle:
        request_by_id={r['request_id']:r['request'] for r in (json.loads(line) for line in request_handle if line.strip())}
    api_paths=list((OUT/'attempts').glob('*.json'))
    if len(api_paths)!=len(ids):
        raise RuntimeError(f'DESIGN inference incomplete {len(api_paths)}/{len(ids)}')
    api={p.stem:json.loads(p.read_text()) for p in api_paths}
    if set(api)!=ids:
        raise RuntimeError('API result set differs from DESIGN split')
    gold=design_gold(ids)
    cases=[]
    request_logs=[]
    raw=[]
    parsed=[]
    usage=collections.Counter()
    total_latency=0.0
    total_elapsed=0.0
    total_attempts=0
    invented_any=0
    parseable=complete=legal=0
    failures=collections.Counter()
    for rid in sorted(ids):
        source=by_id[rid]
        item=api[rid]
        members=source['candidate_ids']
        target=gold[rid]
        if target not in members or item['candidate_ids']!=members:
            raise RuntimeError(f'Gold/candidate mismatch {rid}')
        probs=source['candidate_probabilities']
        if set(probs)!=set(members):
            raise RuntimeError('Stage2 score set mismatch')
        stage_rank=source['ranked_candidate_ids']
        if set(stage_rank)!=set(members):
            raise RuntimeError('Stage2 ranking set mismatch')
        stage_gold_rank=stage_rank.index(target)+1
        final=item['attempts'][-1]
        failures[final.get('error_kind') or 'none']+=1
        if final.get('parse_valid'):
            parseable+=1
            complete+=bool(final.get('candidate_completeness'))
            legal+=bool(final.get('candidate_legality'))
        for attempt in item['attempts']:
            total_attempts+=1
            invented_any+=bool(attempt.get('invented_candidate'))
            total_latency+=float(attempt.get('latency_seconds') or 0)
            u=attempt.get('usage') or {}
            for key in ('prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens'):
                usage[key]+=int(u.get(key) or 0)
            usage['reasoning_tokens']+=int((u.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0)
        total_elapsed+=float(item.get('elapsed_seconds') or 0)
        valid=bool(item['valid'])
        output=item['parsed'] if valid else None
        abstain=bool(output['abstain']) if valid else False
        api_rank=None
        norm=None
        gold_rank=None
        margin=None
        gold_highest=False
        gold_unique=False
        if valid:
            norm=normalized_scores(output['scores'],members)
            api_rank=rank(norm,members)
            if not abstain:
                gold_rank=api_rank.index(target)+1
            ordered=sorted(norm.values(),reverse=True)
            margin=ordered[0]-ordered[1]
            top=max(norm.values())
            gold_highest=math.isclose(norm[target],top,rel_tol=0,abs_tol=1e-12)
            gold_unique=gold_highest and sum(math.isclose(v,top,rel_tol=0,abs_tol=1e-12) for v in norm.values())==1
        s_ok=stage_gold_rank==1
        a_ok=gold_rank==1
        if not valid:
            category='invalid_output'
        elif abstain:
            category='abstain'
        elif s_ok and a_ok:
            category='both_correct'
        elif s_ok and not a_ok:
            category='correct_to_wrong'
        elif not s_ok and a_ok:
            category='wrong_to_correct'
        else:
            category='both_wrong'
        row={'record_id':rid,'module':source['module'],'function_group_id':source['function_group_id'],
             'candidate_count':len(members),'stage2_gold_rank':stage_gold_rank,
             'stage2_reciprocal_rank':1/stage_gold_rank,'stage2_correct':s_ok,
             'api_valid':valid,'api_abstain':abstain,'api_gold_rank':gold_rank,
             'api_reciprocal_rank':1/gold_rank if gold_rank else 0,
             'api_correct':a_ok,'paired_category':category,
             'gold_rank_improved':bool(gold_rank is not None and gold_rank<stage_gold_rank),
             'gold_score_tied_highest':gold_highest,
             'gold_score_unique_highest':gold_unique,
             'api_top1_probability':max(norm.values()) if norm else None,
             'api_top2_probability':sorted(norm.values(),reverse=True)[1] if norm else None,
             'api_margin':margin,'api_normalized_scores':norm}
        cases.append(row)
        request_logs.append({'record_id':rid,'module':source['module'],
                             'query_id':source['query_id'],'candidate_count':len(members),
                             'request_sha256':item['request_sha256'],
                             'request':request_by_id[rid],
                             'http_status':final.get('http_status'),
                             'api_valid_response':final.get('api_valid_response'),
                             'parse_valid':final.get('parse_valid'),
                             'candidate_completeness':final.get('candidate_completeness'),
                             'candidate_legality':final.get('candidate_legality'),
                             'timeout':any(a.get('timeout') for a in item['attempts']),
                             'retry_count':item['retry_count'],
                             'empty_response':final.get('empty_response'),
                             'elapsed_seconds':item.get('elapsed_seconds'),
                             'valid':valid,'error_kind':item.get('final_error_kind')})
        raw.append({'record_id':rid,'module':source['module'],'attempts':item['attempts']})
        parsed.append({'record_id':rid,'module':source['module'],'candidate_ids':members,
                       'valid':valid,'scores':output['scores'] if output else None,
                       'abstain':output['abstain'] if output else None})
    module_rows={m:[r for r in cases if r['module']==m] for m in MODULES}
    mod_metrics={m:metrics_for_module(module_rows[m]) for m in MODULES}
    per_mod_engineering={m:(v['valid_rate']>=0.95 and
                            all(r['api_valid'] or not api[r['record_id']]['attempts'][-1].get('parse_valid') for r in module_rows[m]))
                         for m,v in mod_metrics.items()}
    candidate_ok=(parseable>0 and complete==parseable and legal==parseable and invented_any==0)
    for m in MODULES:
        per_mod_engineering[m]=per_mod_engineering[m] and candidate_ok
    hit,miss,out=(usage['prompt_cache_hit_tokens'],usage['prompt_cache_miss_tokens'],usage['completion_tokens'])
    offpeak=(hit*0.003+miss*0.15+out*0.6)/1e6
    metrics={'schema':'seraph_external_ssq_design_metrics_v0_4_0c',
             'split_sha256':hashlib.sha256((OLD/'EXTERNAL_SSQ_SCORING_SPLIT_V2.json').read_bytes()).hexdigest(),
             'api_config_sha256':hashlib.sha256((SMOKE/'API_SMOKE_CONFIGURATION_CONTRACT.json').read_bytes()).hexdigest(),
             'module_metrics':mod_metrics,'module_engineering_pass':per_mod_engineering,
             'candidate_complete_among_parseable':f'{complete}/{parseable}',
             'candidate_legal_among_parseable':f'{legal}/{parseable}',
             'invented_candidate_across_attempts':invented_any,
             'failure_kinds_final':dict(failures),'total_requests':len(cases),
             'total_attempts':total_attempts,'total_retries':total_attempts-len(cases),
             'empty_final_responses':sum(api[r]['attempts'][-1].get('empty_response') for r in ids),
             'parse_failures_final':sum(a['attempts'][-1].get('api_valid_response') and not a['attempts'][-1].get('parse_valid') for a in api.values()),
             'token_usage_all_attempts':dict(usage),
             'api_latency_seconds_all_attempts':round(total_latency,3),
             'record_elapsed_seconds_total':round(total_elapsed,3),
             'estimated_cost_USD_offpeak':round(offpeak,6),
             'estimated_cost_USD_peak':round(offpeak*2,6),
             'estimated_cost_note':'NOT BILLING RECORD; 2026-10-05 DeepSeek listed rates, retained usage only',
             'abstain_metric_policy':'API abstain/invalid contributes 0 to raw R@1 and MRR; paired correction/harm excludes abstain and invalid.',
             'score_normalization':'L1 over nonnegative scores; all-zero vector maps to uniform; ties break by lexical candidate ID',
             'HOLDOUT_inference_count':0,'HOLDOUT_evaluation_count':0,
             'CALIBRATION_opened':0,'FINAL_TEST_V2_opened':0,'FINAL_TEST_V3_opened':0}
    paired={m:{k:v for k,v in mod_metrics[m].items() if k in ('n','wrong_to_correct','correct_to_wrong','both_correct','both_wrong','net_correction','abstain','rescue_precision','rescue_coverage','harm_rate','stage2_wrong_count','stage2_correct_count')} for m in MODULES}
    write_lines_new(OUT/'EXTERNAL_SSQ_DESIGN_REQUESTS.jsonl',request_logs)
    write_lines_new(OUT/'EXTERNAL_SSQ_DESIGN_RAW_RESPONSES.jsonl',raw)
    write_lines_new(OUT/'EXTERNAL_SSQ_DESIGN_PARSED_SCORES.jsonl',parsed)
    write_new(OUT/'EXTERNAL_SSQ_DESIGN_METRICS.json',metrics)
    write_new(OUT/'EXTERNAL_SSQ_CORRECTION_HARM.json',paired)
    # Case rows are DESIGN-only and contain no raw Gold IDs; they support reproducible analyses.
    write_lines_new(OUT/'EXTERNAL_SSQ_DESIGN_CASE_METRICS.jsonl',cases)
    rescue_lines=['# DESIGN rescue analysis','','Definitions: rescue precision = wrong→correct / (wrong→correct + correct→wrong); rescue coverage = wrong→correct / Stage2-wrong; harm rate = correct→wrong / Stage2-correct. Paired categories exclude explicit abstain and invalid API output.','']
    for m in MODULES:
        v=mod_metrics[m]
        rescue_lines += [f'## {m}','',f'Stage2 wrong: {v["stage2_wrong_count"]}; API top1 correct: {v["api_top1_correct_within_stage2_wrong"]}; Gold rank improved: {v["api_gold_rank_improved_within_stage2_wrong"]}; Gold tied-highest score: {v["api_gold_highest_score_within_stage2_wrong"]}; unique highest: {v["api_gold_unique_highest_score_within_stage2_wrong"]}.',f'Rescue precision: {v["rescue_precision"]}; rescue coverage: {v["rescue_coverage"]}; harm rate: {v["harm_rate"]}.','']
    (OUT/'EXTERNAL_SSQ_RESCUE_ANALYSIS.md').write_text('\n'.join(rescue_lines))
    margin_lines=['# DESIGN normalized API margin analysis','','Normalization: L1; all-zero scores become uniform. Margin = top1 - top2. Abstain/invalid omitted from margin cohorts.','']
    for m in MODULES:
        margin_lines += [f'## {m}','',json.dumps(mod_metrics[m]['api_margin'],sort_keys=True,indent=2),'']
    (OUT/'EXTERNAL_SSQ_MARGIN_ANALYSIS.md').write_text('\n'.join(margin_lines))
    print(json.dumps({'design_records':len(cases),'modules':{m:{'stage2_R1':mod_metrics[m]['stage2']['R@1'],'api_R1':mod_metrics[m]['api_raw']['R@1'],'net':mod_metrics[m]['net_correction'],'engineering_pass':per_mod_engineering[m]} for m in MODULES},'holdout_evaluation':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('evaluate',))
    args=parser.parse_args()
    main()
