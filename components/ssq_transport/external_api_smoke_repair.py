"""One fixed 30-record External DeepSeek SSQ configuration-repair smoke.

This program never reads Gold and has no DESIGN/HOLDOUT execution path.
The API key is read only from DEEPSEEK_API_KEY and never serialized.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request


OLD = Path('/data/chaoni/mjy/resym_exp/training_runs/seraph_local/external_ssq_v0_4_0_dev')
NEW = Path('/data/chaoni/mjy/resym_exp/training_runs/seraph_local/external_ssq_v0_4_0b_api_smoke_repair')
ENDPOINT = 'https://api.deepseek.com/chat/completions'
MODEL = 'deepseek-flash'
SYSTEM = ('You are a binary-semantics evidence scorer. Candidate IDs are opaque and order is not evidence. '
          'Use only the supplied pseudo-C and candidate semantic descriptions. '
          'Output JSON only: one object with scores mapping every candidate ID to a finite number between 0 and 1, '
          'and abstain as a boolean. Do not include any other fields.')
USER_PREFIX = ('Score every candidate. Return exactly {"scores":{"<each supplied candidate_id>":0.0},"abstain":false}. '
               'Return JSON only. Do not explain. Do not use Markdown. Do not repeat the prompt. '
               'Do not omit candidates. Do not invent candidate IDs. Input JSON:\n')
PROMPT_SHA = hashlib.sha256((SYSTEM + '\n' + USER_PREFIX).encode()).hexdigest()
MAX_TOKENS = 4096
TIMEOUT = 90
MAX_ATTEMPTS = 3


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


HTTP = urllib.request.build_opener(NoRedirect())


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def write_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def write_lines_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def source_rows() -> list[dict]:
    with (OLD/'EXTERNAL_SSQ_API_REQUESTS_V2.jsonl').open() as handle:
        requests = {row['request_id']: row for row in (json.loads(line) for line in handle if line.strip())}
    ids = json.loads((OLD/'EXTERNAL_SSQ_SMOKE_IDS_V2.json').read_text())['request_ids']
    if len(ids) != 30 or len(set(ids)) != 30 or set(ids) - set(requests):
        raise RuntimeError('old smoke set invalid')
    result = [requests[rid] for rid in ids]
    if collections.Counter(row['module'] for row in result) != {'TypeSem': 10, 'InterfaceSem': 10, 'LayoutSem': 10}:
        raise RuntimeError('module distribution mismatch')
    return result


def prepare():
    if any(NEW.iterdir()):
        raise RuntimeError('new output directory is not empty')
    rows = source_rows()
    split = json.loads((OLD/'EXTERNAL_SSQ_SCORING_SPLIT_V2.json').read_text())
    by_id = {row['record_id']: row for row in split['records']}
    audit_rows = []
    for old in rows:
        rid = old['request_id']
        src = by_id[rid]
        if old['split'] != 'EXTERNAL_SSQ_DESIGN' or src['external_ssq_split'] != 'EXTERNAL_SSQ_DESIGN':
            raise RuntimeError('smoke record not in DESIGN')
        request = old['request']
        candidate_ids = [c['candidate_id'] for c in request['candidates']]
        if candidate_ids != src['candidate_ids'] or request['module'] != src['module']:
            raise RuntimeError('candidate/source mismatch')
        if request['request_id'] != rid or len(candidate_ids) != len(set(candidate_ids)):
            raise RuntimeError('request identity/legality mismatch')
        audit_rows.append({'record_id': rid, 'query_id': src['query_id'], 'module': src['module'],
                           'candidate_ids': candidate_ids, 'candidate_count': len(candidate_ids),
                           'request_input_sha256': sha_bytes(canonical(request)),
                           'candidate_set_sha256': sha_bytes(canonical(candidate_ids)),
                           'source_split': old['split']})
    old_cfg = json.loads((OLD/'EXTERNAL_SSQ_API_CONFIG_V2.json').read_text())
    if old_cfg['request_file_sha256'] != sha_file(OLD/'EXTERNAL_SSQ_API_REQUESTS_V2.jsonl'):
        raise RuntimeError('old request file changed')
    old_smoke = json.loads((OLD/'EXTERNAL_SSQ_SMOKE_REPORT.json').read_text())
    if old_smoke['valid_structured_output'] != 6 or old_smoke['total'] != 30:
        raise RuntimeError('old failure baseline changed')
    audit = {
        'schema': 'seraph_external_ssq_smoke_manifest_reuse_v0_4_0b',
        'record_count': 30, 'record_ids_exact_order_reused': True,
        'query_ids_reused': True, 'candidate_sets_and_order_reused': True,
        'module_distribution': dict(collections.Counter(row['module'] for row in rows)),
        'old_smoke_manifest_sha256': sha_file(OLD/'EXTERNAL_SSQ_SMOKE_IDS_V2.json'),
        'old_request_file_sha256': sha_file(OLD/'EXTERNAL_SSQ_API_REQUESTS_V2.jsonl'),
        'old_split_sha256': sha_file(OLD/'EXTERNAL_SSQ_SCORING_SPLIT_V2.json'),
        'clean_stage2_scores_and_context_policy': 'unchanged; all request objects reused byte-equivalent after canonical serialization',
        'gold_evaluation_subset': 'unchanged; no Gold opened for this smoke',
        'old_valid_baseline': '6/30', 'rows': audit_rows,
    }
    config = {
        'schema': 'seraph_external_ssq_api_smoke_configuration_v0_4_0b',
        'status': 'FROZEN_BEFORE_API',
        'endpoint': ENDPOINT, 'model': MODEL,
        'thinking': {'type': 'disabled'}, 'reasoning_effort': None,
        'temperature': 0, 'top_p': 'omitted; API default 1.0',
        'max_tokens': MAX_TOKENS, 'max_completion_tokens': 'not used on chat/completions',
        'response_format': {'type': 'json_object'},
        'timeout_seconds': TIMEOUT, 'max_attempts': MAX_ATTEMPTS,
        'retry_policy': 'same request/config only; retry network, 408/429/5xx, empty/invalid/incomplete output; no retry on 400/401/403',
        'prompt_sha256': PROMPT_SHA, 'input_request_file_sha256': audit['old_request_file_sha256'],
        'smoke_manifest_sha256': audit['old_smoke_manifest_sha256'],
        'output_contract': {'scores': 'exact candidate-id key set, finite numbers in [0,1]', 'abstain': 'boolean'},
        'design_allowed': False, 'holdout_allowed': False,
        'documentation': ['https://api-docs.deepseek.com/api/create-chat-completion/',
                          'https://api-docs.deepseek.com/guides/thinking_mode/',
                          'https://api-docs.deepseek.com/quick_start/pricing/'],
    }
    audit_md = '''# API configuration audit\n\nPrevious smoke used `https://api.deepseek.com/chat/completions`, `deepseek-flash`, `temperature=0`, `max_tokens=1000`, `response_format=json_object`; `top_p` was omitted (API default 1), thinking/reasoning parameters were omitted (API default thinking enabled/high). `max_completion_tokens` was not used. Timeout was 90 seconds, with at most 3 attempts and bounded backoff.\n\nAll 30 final responses had HTTP 200, but only 6/30 were valid. 21/30 saved final contents were empty. In the saved final responses, 26,560 of 28,806 completion tokens were reasoning tokens. The prior client did not retain usage for retry responses, so these are lower bounds. In thinking mode `temperature` has no effect. The official API documents thinking as enabled by default and supports `thinking.type=disabled` for non-thinking mode; `max_tokens` bounds generated tokens, which empirically included the reasoning allocation in the previous smoke.\n\nThis is an output-contract failure, not a semantic-performance result. No API key is recorded here.\n\nSources: https://api-docs.deepseek.com/api/create-chat-completion/ ; https://api-docs.deepseek.com/guides/thinking_mode/\n'''
    write_new(NEW/'SMOKE_MANIFEST_REUSE_AUDIT.json', audit)
    write_new(NEW/'API_SMOKE_CONFIGURATION_CONTRACT.json', config)
    (NEW/'API_CONFIGURATION_AUDIT.md').write_text(audit_md)
    print(json.dumps({'prepared': True, 'records': 30, 'model': MODEL,
                      'thinking': 'disabled', 'max_tokens': MAX_TOKENS,
                      'prompt_sha256': PROMPT_SHA}))


def validate(content, request):
    flags = {'nonempty': bool(content and content.strip()), 'parse_valid': False,
             'candidate_complete': False, 'candidate_legal': False,
             'invented_candidate': False, 'score_numeric_finite': False,
             'abstain_valid': False}
    if not flags['nonempty']:
        return None, flags, 'empty_response'
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        return None, flags, 'json_parse_failed'
    flags['parse_valid'] = True
    if not isinstance(obj, dict) or not isinstance(obj.get('scores'), dict):
        return None, flags, 'scores_missing_or_wrong_type'
    original = [c['candidate_id'] for c in request['candidates']]
    got = set(obj['scores'])
    expected = set(original)
    flags['invented_candidate'] = bool(got - expected)
    flags['candidate_complete'] = expected <= got
    flags['candidate_legal'] = got <= expected
    if got != expected:
        return None, flags, 'candidate_key_set_mismatch'
    scores = {}
    for cid in original:
        val = obj['scores'][cid]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or not 0 <= val <= 1:
            return None, flags, 'invalid_score'
        scores[cid] = float(val)
    flags['score_numeric_finite'] = True
    if not isinstance(obj.get('abstain'), bool):
        return None, flags, 'invalid_abstain'
    flags['abstain_valid'] = True
    return {'scores': scores, 'abstain': obj['abstain']}, flags, None


def api_call(key: str, request: dict):
    body = {'model': MODEL, 'thinking': {'type': 'disabled'}, 'temperature': 0,
            'max_tokens': MAX_TOKENS, 'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': USER_PREFIX + json.dumps(request, ensure_ascii=False, sort_keys=True)}]}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={'Authorization': 'Bearer ' + key,
                                          'Content-Type': 'application/json'}, method='POST')
    start = time.monotonic()
    try:
        with HTTP.open(req, timeout=TIMEOUT) as response:
            envelope = json.loads(response.read().decode())
            choice = envelope['choices'][0]
            return {'http_status': int(response.status),
                    'content': choice['message'].get('content'),
                    'finish_reason': choice.get('finish_reason'),
                    'usage': envelope.get('usage') or {},
                    'response_model': envelope.get('model'),
                    'latency_seconds': round(time.monotonic()-start, 3)}, None
    except urllib.error.HTTPError as exc:
        code = int(exc.code)
        return None, {'error_kind': 'http_error', 'http_status': code,
                      'retryable': code in (408, 429, 500, 502, 503, 504),
                      'timeout': code == 408, 'latency_seconds': round(time.monotonic()-start, 3)}
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        return None, {'error_kind': type(exc).__name__, 'http_status': None,
                      'retryable': True, 'timeout': isinstance(exc, (socket.timeout, TimeoutError)),
                      'latency_seconds': round(time.monotonic()-start, 3)}
    except (ValueError, KeyError, TypeError, IndexError):
        return None, {'error_kind': 'invalid_api_envelope', 'http_status': None,
                      'retryable': True, 'timeout': False,
                      'latency_seconds': round(time.monotonic()-start, 3)}


def run_smoke():
    config = json.loads((NEW/'API_SMOKE_CONFIGURATION_CONTRACT.json').read_text())
    audit = json.loads((NEW/'SMOKE_MANIFEST_REUSE_AUDIT.json').read_text())
    if config['status'] != 'FROZEN_BEFORE_API' or config['prompt_sha256'] != PROMPT_SHA:
        raise RuntimeError('configuration changed')
    if config['input_request_file_sha256'] != sha_file(OLD/'EXTERNAL_SSQ_API_REQUESTS_V2.jsonl'):
        raise RuntimeError('request evidence changed')
    rows = source_rows()
    if [row['request_id'] for row in rows] != [row['record_id'] for row in audit['rows']]:
        raise RuntimeError('smoke manifest changed')
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    print('API_KEY_AVAILABLE=' + ('true' if key else 'false'), flush=True)
    if not key:
        raise SystemExit(2)
    attempt_dir = NEW/'attempts'
    attempt_dir.mkdir(exist_ok=True)
    for index, row in enumerate(rows, 1):
        rid = row['request_id']
        path = attempt_dir/(rid+'.json')
        if path.exists():
            existing = json.loads(path.read_text())
            if existing['request_sha256'] != sha_bytes(canonical(row['request'])):
                raise RuntimeError('resume request mismatch')
            print(json.dumps({'index': index, 'total': 30, 'module': row['module'],
                              'valid': existing['valid'], 'resumed': True}), flush=True)
            continue
        attempts = []
        valid_output = None
        for attempt_number in range(1, MAX_ATTEMPTS+1):
            result, error = api_call(key, row['request'])
            if error is not None:
                attempt = {'attempt_number': attempt_number, **error,
                           'api_valid_response': False, 'parse_valid': False,
                           'candidate_completeness': False, 'candidate_legality': False,
                           'invented_candidate': False, 'empty_response': False,
                           'usage': {}, 'raw_content': None}
            else:
                parsed, flags, invalid = validate(result['content'], row['request'])
                attempt = {'attempt_number': attempt_number,
                           'http_status': result['http_status'],
                           'api_valid_response': True, 'parse_valid': flags['parse_valid'],
                           'candidate_completeness': flags['candidate_complete'],
                           'candidate_legality': flags['candidate_legal'],
                           'invented_candidate': flags['invented_candidate'],
                           'empty_response': not flags['nonempty'],
                           'score_numeric_finite': flags['score_numeric_finite'],
                           'abstain_valid': flags['abstain_valid'],
                           'timeout': False, 'usage': result['usage'],
                           'finish_reason': result['finish_reason'],
                           'response_model': result['response_model'],
                           'latency_seconds': result['latency_seconds'],
                           'error_kind': invalid, 'raw_content': result['content']}
                if parsed is not None:
                    valid_output = parsed
            if key in json.dumps(attempt, ensure_ascii=False):
                raise RuntimeError('secret unexpectedly appeared in API response')
            attempts.append(attempt)
            if valid_output is not None:
                break
            if error is not None and not error['retryable']:
                break
            if attempt_number < MAX_ATTEMPTS:
                time.sleep(2 ** attempt_number)
        final = attempts[-1]
        record = {'record_id': rid, 'module': row['module'],
                  'query_id': audit['rows'][index-1]['query_id'],
                  'candidate_count': len(row['request']['candidates']),
                  'candidate_ids': [c['candidate_id'] for c in row['request']['candidates']],
                  'request_sha256': sha_bytes(canonical(row['request'])),
                  'prompt_sha256': PROMPT_SHA, 'model': MODEL,
                  'valid': valid_output is not None, 'parsed': valid_output,
                  'retry_count': len(attempts)-1, 'attempts': attempts,
                  'final_http_status': final.get('http_status'),
                  'final_error_kind': final.get('error_kind')}
        write_new(path, record)
        print(json.dumps({'index': index, 'total': 30, 'module': row['module'],
                          'valid': record['valid'], 'retry_count': record['retry_count'],
                          'error_kind': record['final_error_kind']}), flush=True)
        if final.get('http_status') in (400, 401, 403):
            print('STOP_UNRETRYABLE_API_STATUS', flush=True)
            break


def finalize():
    rows = source_rows()
    audit = json.loads((NEW/'SMOKE_MANIFEST_REUSE_AUDIT.json').read_text())
    files = {p.stem: p for p in (NEW/'attempts').glob('*.json')}
    results = [json.loads(files[row['request_id']].read_text()) for row in rows if row['request_id'] in files]
    if len(results) != 30:
        raise RuntimeError(f'only {len(results)}/30 responses recorded; cannot finalize smoke')
    request_logs = []
    parsed_outputs = []
    module_counts = collections.defaultdict(lambda: collections.Counter())
    failures = collections.Counter()
    usage = collections.Counter()
    latency = 0.0
    empty = 0
    parse_failure = 0
    invented = 0
    parseable_count = 0
    parseable_complete = 0
    parseable_legal = 0
    first_attempt_valid = 0
    total_attempts = 0
    for record in results:
        mod = record['module']
        module_counts[mod]['total'] += 1
        module_counts[mod]['valid'] += bool(record['valid'])
        final = record['attempts'][-1]
        first_attempt_valid += bool(record['attempts'][0].get('error_kind') is None)
        empty += bool(final.get('empty_response'))
        parse_failure += bool(final.get('api_valid_response') and not final.get('parse_valid'))
        if final.get('parse_valid'):
            parseable_count += 1
            parseable_complete += bool(final.get('candidate_completeness'))
            parseable_legal += bool(final.get('candidate_legality'))
        failures[final.get('error_kind') or 'none'] += 1
        for attempt in record['attempts']:
            total_attempts += 1
            invented += bool(attempt.get('invented_candidate'))
            latency += attempt.get('latency_seconds') or 0
            u = attempt.get('usage') or {}
            for key in ('prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens'):
                usage[key] += int(u.get(key) or 0)
            usage['reasoning_tokens'] += int((u.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0)
        request_logs.append({'record_id': record['record_id'], 'module': mod,
                             'query_id': record['query_id'], 'candidate_count': record['candidate_count'],
                             'request_sha256': record['request_sha256'],
                             'HTTP_status': final.get('http_status'),
                             'API_valid_response': final.get('api_valid_response'),
                             'parse_valid': final.get('parse_valid'),
                             'candidate_completeness': final.get('candidate_completeness'),
                             'candidate_legality': final.get('candidate_legality'),
                             'empty_response': final.get('empty_response'),
                             'timeout': any(a.get('timeout') for a in record['attempts']),
                             'retry_count': record['retry_count'],
                             'prompt_tokens': sum((a.get('usage') or {}).get('prompt_tokens',0) for a in record['attempts']),
                             'reasoning_tokens': sum(((a.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens',0) for a in record['attempts']),
                             'completion_tokens': sum((a.get('usage') or {}).get('completion_tokens',0) for a in record['attempts']),
                             'latency_seconds': round(sum(a.get('latency_seconds') or 0 for a in record['attempts']),3),
                             'valid': record['valid'], 'final_error_kind': record['final_error_kind']})
        parsed_outputs.append({'record_id': record['record_id'], 'module': mod,
                               'candidate_ids': record['candidate_ids'], 'valid': record['valid'],
                               'output': record['parsed']})
    valid = sum(r['valid'] for r in results)
    modules_ok = all(module_counts[m]['valid'] >= 9 for m in ('TypeSem','InterfaceSem','LayoutSem'))
    candidates_ok = (parseable_count > 0 and parseable_complete == parseable_count
                     and parseable_legal == parseable_count and invented == 0)
    status = 'READY_FOR_EXTERNAL_SSQ_DESIGN' if valid >= 29 and modules_ok and candidates_ok else 'BLOCKED_EXTERNAL_SSQ_API_SMOKE'
    hit, miss, output = (usage['prompt_cache_hit_tokens'],usage['prompt_cache_miss_tokens'],usage['completion_tokens'])
    # Official 2026-10-05 list rates, USD per million tokens. This is an estimate, not an invoice.
    cost_offpeak = (hit*0.003 + miss*0.15 + output*0.6)/1_000_000
    cost_peak = cost_offpeak*2
    report = {
        'schema': 'seraph_external_ssq_v0_4_0b_api_smoke_repair_report',
        'status': status, 'old_smoke_valid': 6, 'new_smoke_valid': valid,
        'total': 30, 'valid_rate': valid/30, 'first_attempt_valid': first_attempt_valid,
        'by_module': {k: dict(v) for k,v in sorted(module_counts.items())},
        'empty_final_responses': empty, 'parse_failures_final': parse_failure,
        'parseable_final_responses': parseable_count,
        'candidate_complete_among_parseable': parseable_complete,
        'candidate_legal_among_parseable': parseable_legal,
        'invented_candidate_across_all_attempts': invented,
        'candidate_gate_passed': candidates_ok,
        'total_api_attempts': total_attempts, 'usage_all_retained_attempts': dict(usage),
        'api_latency_seconds_all_attempts': round(latency,3),
        'estimated_cost_USD_offpeak': round(cost_offpeak,6),
        'estimated_cost_USD_peak': round(cost_peak,6),
        'cost_note': 'Official DeepSeek list-rate estimate from retained HTTP-200 usage only; not an invoice. Non-200 attempts may lack usage.',
        'failure_kinds_final': dict(failures),
        'model': MODEL, 'thinking': 'disabled', 'max_tokens': MAX_TOKENS,
        'prompt_sha256': PROMPT_SHA,
        'old_smoke_manifest_sha256': audit['old_smoke_manifest_sha256'],
        'upstream_stage2_changed': False, 'gold_evaluated': False,
        'design_called': False, 'holdout_called': False, 'calibration_opened': False,
        'final_test_v2_opened': False, 'final_test_v3_opened': False,
    }
    write_lines_new(NEW/'API_SMOKE_REQUEST_LOG.jsonl', request_logs)
    write_lines_new(NEW/'API_SMOKE_PARSED_OUTPUTS.jsonl', parsed_outputs)
    write_new(NEW/'v0_4_0b_final_status.json', report)
    failure_md = ['# API smoke failure analysis', '',
                  f'Status: **{status}**.',
                  f'Final failure kinds: {dict(failures)}.',
                  f'First-attempt valid: {first_attempt_valid}/30; after fixed retries: {valid}/30.',
                  f'Empty final: {empty}; parse failures: {parse_failure}; invented candidate across attempts: {invented}.',
                  'No semantic metrics or Gold were used in this smoke.', '']
    (NEW/'API_SMOKE_FAILURE_ANALYSIS.md').write_text('\n'.join(failure_md))
    lines = ['# v0_4_0b External API configuration-repair smoke', '',
             f'Final status: **{status}**.', '',
             f'Frozen configuration: `{MODEL}`, `thinking.disabled`, `temperature=0`, `max_tokens={MAX_TOKENS}`, `response_format=json_object`, timeout {TIMEOUT}s, at most {MAX_ATTEMPTS} attempts.',
             f'Original 30 records reused exactly; input evidence and candidate sets unchanged. Manifest audit: `SMOKE_MANIFEST_REUSE_AUDIT.json`.',
             f'Valid structured output: **{valid}/30 ({valid/30:.1%})**; required >=29/30 and >=9/10 per module.',
             '', '| Module | Valid / 10 |', '|---|---:|']
    for module in ('TypeSem','InterfaceSem','LayoutSem'):
        lines.append(f'| {module} | {module_counts[module]["valid"]}/10 |')
    lines.extend(['', f'Empty final responses: {empty}; parse failures: {parse_failure}.',
                  f'Candidate completeness among parseable final outputs: {parseable_complete}/{parseable_count}; legality: {parseable_legal}/{parseable_count}; invented across any attempt: {invented}.',
                  f'All retained-attempt usage: {dict(usage)}.',
                  f'Accumulated API latency: {latency:.1f}s across {total_attempts} attempts.',
                  f'Estimated listed API cost: USD ${cost_offpeak:.6f} off-peak to ${cost_peak:.6f} peak (not invoice).',
                  '', 'No DESIGN, HOLDOUT, CALIBRATION, FINAL_TEST_V2 or FINAL_TEST_V3 calls. Stage1/Stage2, Gold, input evidence and split remain unchanged.',
                  'The v0_4_0_dev 6/30 failure baseline remains intact.', ''])
    (NEW/'V0_4_0B_API_SMOKE_REPAIR_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'status': status, 'valid': valid, 'total': 30,
                      'by_module': {k: dict(v) for k,v in module_counts.items()},
                      'empty': empty, 'parse_failures': parse_failure,
                      'candidate_complete': f'{parseable_complete}/{parseable_count}',
                      'candidate_legal': f'{parseable_legal}/{parseable_count}',
                      'reasoning_tokens': usage['reasoning_tokens'],
                      'completion_tokens': usage['completion_tokens'],
                      'latency_seconds': round(latency,1)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare','smoke','finalize'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'smoke':
        run_smoke()
    else:
        finalize()


if __name__ == '__main__':
    main()
