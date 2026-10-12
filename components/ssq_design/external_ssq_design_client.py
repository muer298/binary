"""Frozen v0_4_0b External SSQ API contract applied to DESIGN only.

No Gold file is read in this module. Completed request IDs are checkpointed and
never called again on resume. HOLDOUT has no execution path.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time


PKG_B = Path('/data/chaoni/mjy/resym_exp/packages/seraph_external_ssq_v0_4_0b_api_smoke_repair')
sys.path.insert(0, str(PKG_B))
import external_api_smoke_repair as fixed  # noqa: E402


BASE = Path('/data/chaoni/mjy/resym_exp/training_runs/seraph_local')
OLD = BASE/'external_ssq_v0_4_0_dev'
SMOKE = BASE/'external_ssq_v0_4_0b_api_smoke_repair'
OUT = BASE/'external_ssq_v0_4_0c_design'
PACKAGE = Path('/data/chaoni/mjy/resym_exp/packages/seraph_external_ssq_v0_4_0c_design')
EXPECTED_SPLIT_SHA = 'ab0555dd3d3767ab0b70ddd92d3a97532395fd1be5f1a41a0d4a8587b60a8c24'
EXPECTED_CONTRACT_SHA = '79bf0773568492468725d99c7232364cd7791a74e5e2930d00e211ce4ce7d476'
EXPECTED_STAGE2_SELECTION_SHA = '5a79d599c1eb2c680aaa73d0e8b14fcdcaaef2dc97c37e2571ef06a918445985'
POLICY = {'TypeSem': 'unified_llm_gate_clean',
          'InterfaceSem': 'module_aware_clean',
          'LayoutSem': 'unified_llm_gate_clean'}
SEEDS = ('1337', '3407', '20260929')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def write_new(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)+'\n')


def design_rows():
    with (OLD/'EXTERNAL_SSQ_API_REQUESTS_V2.jsonl').open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [r for r in rows if r['split'] == 'EXTERNAL_SSQ_DESIGN']


def preflight():
    if sha(OLD/'EXTERNAL_SSQ_SCORING_SPLIT_V2.json') != EXPECTED_SPLIT_SHA:
        raise RuntimeError('split SHA changed')
    if sha(SMOKE/'API_SMOKE_CONFIGURATION_CONTRACT.json') != EXPECTED_CONTRACT_SHA:
        raise RuntimeError('smoke config SHA changed')
    selected_path = BASE/'clean_joint_v0_2_10/STAGE2_FINAL_SELECTION_CONTRACT_CLEAN_V2.json'
    if sha(selected_path) != EXPECTED_STAGE2_SELECTION_SHA:
        raise RuntimeError('Stage2 selection contract hash changed')
    selected = json.loads(selected_path.read_text())
    if tuple(map(str,selected['seeds'])) != SEEDS:
        raise RuntimeError('selected seed contract changed')
    for module, arm in POLICY.items():
        module_policy = selected['module_policy'][module]
        if module_policy['selected_arm'] != arm:
            raise RuntimeError('selected Stage2 arm changed')
        for checkpoint in module_policy['checkpoint_artifacts']:
            if sha(Path(checkpoint['path'])) != checkpoint['sha256']:
                raise RuntimeError('selected Stage2 checkpoint hash changed')
    status = json.loads((SMOKE/'v0_4_0b_final_status.json').read_text())
    if status['status'] != 'READY_FOR_EXTERNAL_SSQ_DESIGN' or status['new_smoke_valid'] != 30:
        raise RuntimeError('smoke gate not passed')
    contract = json.loads((SMOKE/'API_SMOKE_CONFIGURATION_CONTRACT.json').read_text())
    if (contract['model'] != fixed.MODEL or contract['prompt_sha256'] != fixed.PROMPT_SHA
        or contract['thinking'] != {'type': 'disabled'} or contract['temperature'] != 0
        or contract['max_tokens'] != fixed.MAX_TOKENS or contract['timeout_seconds'] != fixed.TIMEOUT
        or contract['max_attempts'] != fixed.MAX_ATTEMPTS):
        raise RuntimeError('API configuration drift')
    if sha(OLD/'EXTERNAL_SSQ_API_REQUESTS_V2.jsonl') != contract['input_request_file_sha256']:
        raise RuntimeError('API input evidence drift')
    split = json.loads((OLD/'EXTERNAL_SSQ_SCORING_SPLIT_V2.json').read_text())
    if split['function_group_overlap'] != 0 or split['CALIBRATION_opened'] or split['FINAL_TEST_V2_opened']:
        raise RuntimeError('data split gate failed')
    split_by_id = {r['record_id']: r for r in split['records']}
    rows = design_rows()
    if len(rows) != 628 or collections.Counter(r['module'] for r in rows) != {'TypeSem':215,'InterfaceSem':28,'LayoutSem':385}:
        raise RuntimeError('DESIGN count/distribution drift')
    if len({r['request_id'] for r in rows}) != len(rows):
        raise RuntimeError('duplicate DESIGN request ID')
    scope = collections.Counter()
    audit_rows = []
    for r in rows:
        s = split_by_id[r['request_id']]
        q = r['request']
        if s['external_ssq_split'] != 'EXTERNAL_SSQ_DESIGN' or r['split'] != 'EXTERNAL_SSQ_DESIGN':
            raise RuntimeError('HOLDOUT request encountered')
        if s['module'] != r['module'] or s['candidate_ids'] != [c['candidate_id'] for c in q['candidates']]:
            raise RuntimeError('candidate contract drift')
        if s['selected_arm'] != POLICY[r['module']] or s['trigger_threshold'] != 0.15 or not s['gap'] < 0.15:
            raise RuntimeError('Stage2 policy/trigger drift')
        if tuple(sorted(s['per_seed_probabilities'])) != tuple(sorted(SEEDS)):
            raise RuntimeError('seed set drift')
        if any(term in json.dumps(q,ensure_ascii=False).lower() for term in ('gold_candidate_id','gold_position','confirmed_hard','confirmed_soft','candidate_status')):
            raise RuntimeError('Gold or answer marker in API request')
        scope[(r['module'], q['code']['scope'])] += 1
        audit_rows.append({'record_id': r['request_id'], 'module': r['module'],
                           'query_id': s['query_id'], 'candidate_count': len(q['candidates']),
                           'candidate_set_sha256': hashlib.sha256(canonical(s['candidate_ids'])).hexdigest(),
                           'request_sha256': hashlib.sha256(canonical(q)).hexdigest(),
                           'context_scope': q['code']['scope']})
    audit = {'schema':'seraph_external_ssq_design_preflight_v0_4_0c',
             'design_count':len(rows), 'module_counts':dict(collections.Counter(r['module'] for r in rows)),
             'context_scopes':{str(k):v for k,v in scope.items()},
             'split_sha256':EXPECTED_SPLIT_SHA, 'smoke_config_sha256':EXPECTED_CONTRACT_SHA,
             'request_file_sha256':sha(OLD/'EXTERNAL_SSQ_API_REQUESTS_V2.jsonl'),
             'function_group_overlap':0, 'trigger_threshold':0.15,
             'stage2_policy':POLICY, 'seeds':list(SEEDS),
             'Gold_in_API_requests':False, 'HOLDOUT_request_count':0,
             'CALIBRATION_opened':False, 'FINAL_TEST_V2_opened':False, 'FINAL_TEST_V3_opened':False,
             'rows':audit_rows}
    return rows, audit


def prepare():
    rows, audit = preflight()
    if any(OUT.iterdir()):
        raise RuntimeError('new output directory not empty')
    write_new(OUT/'DESIGN_PREFLIGHT.json',audit)
    print(json.dumps({'preflight':'PASS','design_records':len(rows),
                      'context_scopes':audit['context_scopes'],
                      'split_sha256':EXPECTED_SPLIT_SHA,
                      'smoke_config_sha256':EXPECTED_CONTRACT_SHA}))


def run():
    rows, audit = preflight()
    frozen = json.loads((OUT/'DESIGN_PREFLIGHT.json').read_text())
    if frozen['request_file_sha256'] != audit['request_file_sha256'] or frozen['split_sha256'] != audit['split_sha256']:
        raise RuntimeError('DESIGN preflight changed')
    key = os.environ.get('DEEPSEEK_API_KEY','').strip()
    print('API_KEY_AVAILABLE='+('true' if key else 'false'),flush=True)
    if not key:
        raise SystemExit(2)
    attempts_dir = OUT/'attempts'
    attempts_dir.mkdir(exist_ok=True)
    completed = 0
    for index, row in enumerate(rows,1):
        rid = row['request_id']
        target = attempts_dir/(rid+'.json')
        request_hash = hashlib.sha256(canonical(row['request'])).hexdigest()
        if target.exists():
            prior = json.loads(target.read_text())
            if prior['request_sha256'] != request_hash or prior['prompt_sha256'] != fixed.PROMPT_SHA:
                raise RuntimeError('resume checkpoint hash mismatch')
            completed += 1
            continue
        started = time.monotonic()
        attempts = []
        parsed = None
        for attempt_number in range(1,fixed.MAX_ATTEMPTS+1):
            response,error = fixed.api_call(key,row['request'])
            if error is not None:
                attempt = {'attempt_number':attempt_number,**error,
                           'api_valid_response':False,'parse_valid':False,
                           'candidate_completeness':False,'candidate_legality':False,
                           'invented_candidate':False,'empty_response':False,
                           'usage':{},'raw_content':None}
            else:
                parsed,flags,invalid = fixed.validate(response['content'],row['request'])
                attempt = {'attempt_number':attempt_number,
                           'http_status':response['http_status'],
                           'api_valid_response':True,'parse_valid':flags['parse_valid'],
                           'candidate_completeness':flags['candidate_complete'],
                           'candidate_legality':flags['candidate_legal'],
                           'invented_candidate':flags['invented_candidate'],
                           'empty_response':not flags['nonempty'],
                           'score_numeric_finite':flags['score_numeric_finite'],
                           'abstain_valid':flags['abstain_valid'],
                           'usage':response['usage'],'finish_reason':response['finish_reason'],
                           'response_model':response['response_model'],
                           'latency_seconds':response['latency_seconds'],
                           'timeout':False,'error_kind':invalid,'raw_content':response['content']}
            if key in json.dumps(attempt,ensure_ascii=False):
                raise RuntimeError('secret appeared in response; refusing to save')
            attempts.append(attempt)
            if parsed is not None:
                break
            if error is not None and not error['retryable']:
                break
            if attempt_number < fixed.MAX_ATTEMPTS:
                time.sleep(2**attempt_number)
        final = attempts[-1]
        record = {'record_id':rid,'module':row['module'],
                  'query_id':audit['rows'][index-1]['query_id'],
                  'candidate_ids':[c['candidate_id'] for c in row['request']['candidates']],
                  'candidate_count':len(row['request']['candidates']),
                  'request_sha256':request_hash,'prompt_sha256':fixed.PROMPT_SHA,
                  'model':fixed.MODEL,'valid':parsed is not None,'parsed':parsed,
                  'retry_count':len(attempts)-1,'attempts':attempts,
                  'elapsed_seconds':round(time.monotonic()-started,3),
                  'final_http_status':final.get('http_status'),
                  'final_error_kind':final.get('error_kind')}
        write_new(target,record)
        completed += 1
        print(json.dumps({'index':index,'total':len(rows),'completed':completed,
                          'module':row['module'],'valid':record['valid'],
                          'retry_count':record['retry_count'],
                          'error_kind':record['final_error_kind']}),flush=True)
        if final.get('http_status') in (400,401,403):
            print('STOP_UNRETRYABLE_API_STATUS',flush=True)
            break
    print(json.dumps({'completed_ids':completed,'expected':len(rows)}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','run'))
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:run()


if __name__=='__main__':main()
