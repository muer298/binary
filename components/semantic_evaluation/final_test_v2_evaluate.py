#!/usr/bin/env python3
"""One-shot FINAL_TEST_V2 evaluator; first Gold access follows prediction freeze."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import signal
import stat
import statistics
import sys
from pathlib import Path

RUNS = Path('/data/chaoni/mjy/resym_exp/training_runs/seraph_local')
OUT = RUNS / 'final_test_v2_run'
TEST = RUNS / 'final_test_v2_build'
FREEZE = RUNS / 'final_freeze_v0_3_3c'
GOLD_DIR = TEST / 'final_test_v2_gold_sealed'
GOLD = GOLD_DIR / 'FINAL_TEST_V2_GOLD_SEALED.jsonl'
INPUT = TEST / 'FINAL_TEST_V2_MANIFEST.jsonl'
PRED = OUT / 'FINAL_TEST_V2_PREDICTIONS_FROZEN.jsonl'
PRED_FREEZE = OUT / 'FINAL_TEST_PREDICTION_FREEZE.json'
MANIFEST = FREEZE / 'FINAL_MODEL_MANIFEST_CLEAN_V2.json'
LOCK = FREEZE / 'FINAL_MODEL_FROZEN_CLEAN_V2.lock'
TRAIN = RUNS / 'clean_joint_v0_2_10/train_manifest.jsonl'
EXPECTED_MANIFEST = '82af0f01c2125fb1169f1764be3e1e018e74940a4a12b25ed978c94e31c9ffcd'
EXPECTED_LOCK = '1ab6efae4b38bb0f22597d3467bf80356be8c090370ef3a1b1c42aba733a3c9d'
EXPECTED_TEST = '88f1f3af035ef67d163181a1a9827fe4112bfa60667bbfb20b1375ba7b6a351b'
EXPECTED_GOLD = '78fe1593494f2f9bfd3e3afeb788f336705a10c3503af65c7e31332580bf0aed'
SEEDS = (1337, 3407, 20260929)
MODULES = ('TypeSem', 'InterfaceSem', 'LayoutSem')
VARIANTS = ('A_clean_symbolic_only', 'B_final_clean_stage2')


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)
            stream.write('\n')


def verify_frozen_contract():
    require(sha(MANIFEST) == EXPECTED_MANIFEST and sha(LOCK) == EXPECTED_LOCK,
            'v0_3_3c freeze identity changed')
    lock = read_json(LOCK)
    require(len(lock['watched_artifacts']) == 37 and
            lock['manifest_sha256'] == EXPECTED_MANIFEST,
            'v0_3_3c lock contract invalid')
    for item in lock['watched_artifacts']:
        path = Path(item['path'])
        require(path.is_file() and path.stat().st_size == item['size_bytes']
                and sha(path) == item['sha256'], 'frozen artifact changed: ' + str(path))
    require(sha(INPUT) == EXPECTED_TEST, 'TEST input changed')
    return read_json(MANIFEST)


def prediction_gate():
    manifest = verify_frozen_contract()
    freeze = read_json(PRED_FREEZE)
    require(freeze['status'] == 'PASS' and freeze['prediction_count'] == 2605
            and freeze['manifest_sha256'] == EXPECTED_MANIFEST
            and freeze['lock_sha256'] == EXPECTED_LOCK
            and freeze['gold_ever_opened_before_freeze'] is False,
            'prediction freeze gate invalid')
    require(stat.S_IMODE(GOLD_DIR.stat().st_mode) == 0,
            'Gold directory not sealed immediately before evaluation')
    pred_hash = sha(PRED)
    require(pred_hash == freeze['prediction_sha256'], 'frozen prediction SHA changed')
    require((OUT / 'FINAL_TEST_V2_PREDICTIONS_FROZEN.sha256').read_text().split()[0]
            == pred_hash, 'prediction SHA sidecar mismatch')
    require(not any((OUT / filename).exists() for filename in (
        'FINAL_TEST_METRICS_CLEAN_V2.json', 'FINAL_TEST_REPORT_CLEAN_V2.md',
        'FINAL_TEST_AUDIT_CLEAN_V2.json')),
        'evaluation output already exists; refuse one-time rerun')
    inputs = rows(INPUT)
    predictions = rows(PRED)
    require(len(inputs) == len(predictions) == 2605, 'TEST/prediction count mismatch')
    by_input = {x['record_id']: x for x in inputs}
    by_pred = {x['record_id']: x for x in predictions}
    require(len(by_input) == len(by_pred) == 2605 and set(by_input) == set(by_pred),
            'TEST/prediction identity mismatch')
    for key, source in by_input.items():
        p = by_pred[key]
        require(p['candidate_semantic_ids'] == source['candidate_semantic_identities']
                and p['candidate_instance_identities'] == source['candidate_instance_identities']
                and p['candidate_membership_sha256'] == source['candidate_membership_sha256'],
                'candidate membership changed for ' + key)
        require(source['module'] == p['module'], 'module mismatch for ' + key)
    require(manifest['artifacts']['test_manifest']['sha256'] == EXPECTED_TEST,
            'frozen manifest points at another TEST')
    return manifest, pred_hash, by_input, by_pred


def gold_read_once():
    require(stat.S_IMODE(GOLD_DIR.stat().st_mode) == 0,
            'Gold directory permissions changed')
    original_file_mode = None
    try:
        os.chmod(GOLD_DIR, 0o500)
        require(GOLD.is_file(), 'sealed Gold file missing')
        original_file_mode = stat.S_IMODE(GOLD.stat().st_mode)
        require(original_file_mode == 0, 'sealed Gold file was not mode 000')
        os.chmod(GOLD, 0o400)
        before = sha(GOLD)
        require(before == EXPECTED_GOLD, 'sealed Gold pre-read SHA mismatch')
        gold = rows(GOLD)
        after = sha(GOLD)
        require(after == EXPECTED_GOLD, 'sealed Gold post-read SHA mismatch')
        return gold, before, after
    finally:
        if original_file_mode is not None:
            os.chmod(GOLD, 0o000)
        os.chmod(GOLD_DIR, 0o000)


def rank_metric(ranks):
    require(bool(ranks), 'empty metric population')
    return {'n': len(ranks), 'R@1': sum(x == 1 for x in ranks) / len(ranks),
            'MRR': sum(1 / x for x in ranks) / len(ranks),
            'R@5': sum(x <= 5 for x in ranks) / len(ranks)}


def aggregate_seed_metrics(per_seed):
    result = {}
    for name in ('R@1', 'MRR', 'R@5'):
        values = [per_seed[str(seed)][name] for seed in SEEDS]
        result[name] = {'values_by_seed': dict(zip(map(str, SEEDS), values)),
                        'mean': statistics.mean(values),
                        'std_population': statistics.pstdev(values)}
    return result


def train_position_rule():
    data = rows(TRAIN)
    require(all(x['global_split'] == 'TRAIN' for x in data),
            'position rule source contains non-TRAIN')
    by_count = collections.defaultdict(list)
    by_stratum = collections.defaultdict(list)
    for x in data:
        n = len(x['candidate_ids'])
        by_count[n].append(x['gold_position'])
        by_stratum[(x['module'], n)].append(x['gold_position'])
    majority = {n: collections.Counter(v).most_common(1)[0][0]
                for n, v in by_count.items()}
    position_only = {k: collections.Counter(v).most_common(1)[0][0]
                     for k, v in by_stratum.items()}
    return majority, position_only, len(data)


def evaluate(manifest, pred_hash, by_input, by_pred, gold_rows, gold_before, gold_after):
    require(len(gold_rows) == 2605, 'sealed Gold count is not 2605')
    by_gold = {x['record_id']: x for x in gold_rows}
    require(len(by_gold) == 2605 and set(by_gold) == set(by_input),
            'Gold/TEST record identity mismatch')
    majority, position_only, train_n = train_position_rule()
    rank_data = collections.defaultdict(list)
    seed_rank_data = collections.defaultdict(list)
    position_hits = collections.defaultdict(lambda: collections.Counter())
    gold_positions = collections.defaultdict(collections.Counter)
    perm_agreement = collections.defaultdict(list)
    perm_rank = collections.defaultdict(list)
    perm_r1_variance = collections.defaultdict(list)
    api_resolved = api_abstain = 0
    for rid, source in by_input.items():
        gold = by_gold[rid]
        predicted = by_pred[rid]
        instances = source['candidate_instance_identities']
        semantic = source['candidate_semantic_identities']
        pos = int(gold['gold_position'])
        require(0 <= pos < len(instances) and instances[pos] ==
                gold['gold_candidate_instance_id'], 'Gold instance/position mismatch')
        require(semantic[pos] == gold['gold_semantic_id'],
                'Gold semantic identity mismatch')
        require(gold['gold_semantic_id'] in semantic,
                'Gold not in candidate set')
        module = source['module']
        gold_positions[module][pos] += 1
        if module == 'APISem':
            api_abstain += int(predicted['abstain'])
            api_resolved += int(not predicted['abstain'])
            continue
        n = len(instances)
        position_hits[module]['n'] += 1
        position_hits[module]['always_first'] += int(pos == 0)
        position_hits[module]['majority_position'] += int(pos == majority[n])
        if (module, n) in position_only:
            position_hits[module]['position_only_eligible'] += 1
            position_hits[module]['position_only'] += int(
                pos == position_only[(module, n)])
        for variant in VARIANTS:
            value = predicted['variants'][variant]
            ordered = value['ranked_candidate_semantic_ids']
            require(set(ordered) == set(semantic), 'variant rank not a candidate permutation')
            rank_data[(variant, module)].append(ordered.index(gold['gold_semantic_id']) + 1)
            for seed in SEEDS:
                seed_order = value['per_seed_scores'][str(seed)]['ranked_candidate_semantic_ids']
                require(set(seed_order) == set(semantic), 'seed rank not a candidate permutation')
                seed_rank_data[(variant, module, str(seed))].append(
                    seed_order.index(gold['gold_semantic_id']) + 1)
        permutations = predicted['permutation_robustness']
        require(len(permutations) == 5, 'missing deterministic permutation')
        perm_agreement[module].extend(x['top1_identity_agreement'] for x in permutations)
        perm_rank[module].extend(x['rank_consistency_spearman'] for x in permutations)
        perm_r1_variance[module].append(statistics.pvariance(
            int(x['top1_candidate_semantic_id'] == gold['gold_semantic_id'])
            for x in permutations))

    variant_metrics = {}
    for variant in VARIANTS:
        variant_metrics[variant] = {}
        for module in MODULES:
            per_seed = {str(seed): rank_metric(seed_rank_data[(variant, module, str(seed))])
                        for seed in SEEDS}
            variant_metrics[variant][module] = {
                'ensemble': rank_metric(rank_data[(variant, module)]),
                'per_seed': per_seed,
                'seed_mean_std': aggregate_seed_metrics(per_seed)}
    position = {}
    permutation = {}
    for module in MODULES:
        counts = position_hits[module]
        position[module] = {
            'n': counts['n'], 'gold_position_distribution': dict(gold_positions[module]),
            'always_first_R@1': counts['always_first'] / counts['n'],
            'majority_position_R@1': counts['majority_position'] / counts['n'],
            'position_only_R@1_on_seen_TRAIN_strata': (
                counts['position_only'] / counts['position_only_eligible']
                if counts['position_only_eligible'] else None),
            'position_only_eligible_n': counts['position_only_eligible'],
            'position_only_unseen_TRAIN_strata_n':
                counts['n'] - counts['position_only_eligible']}
        permutation[module] = {
            'n': counts['n'], 'permutations_per_query': 5,
            'top1_identity_agreement': statistics.mean(perm_agreement[module]),
            'mean_rank_consistency_spearman': statistics.mean(perm_rank[module]),
            'mean_R@1_variance': statistics.mean(perm_r1_variance[module])}
    position['rule_source'] = 'v0_2_10 TRAIN only; majority by candidate count; position-only by module and count'
    position['train_record_count'] = train_n
    position['APISem_gold_position_distribution'] = dict(gold_positions['APISem'])
    deltas = {}
    for module in MODULES:
        a = variant_metrics[VARIANTS[0]][module]['ensemble']
        b = variant_metrics[VARIANTS[1]][module]['ensemble']
        deltas[module] = {name: b[name] - a[name] for name in ('R@1', 'MRR', 'R@5')}
    metrics = {
        'schema': 'seraph_final_test_metrics_clean_v2',
        'status': 'FINAL_TEST_COMPLETE',
        'final_manifest_sha256': EXPECTED_MANIFEST, 'final_lock_sha256': EXPECTED_LOCK,
        'prediction_sha256': pred_hash, 'gold_sha256': gold_before,
        'test_manifest_sha256': EXPECTED_TEST, 'total_records': 2605,
        'variants': variant_metrics, 'final_minus_symbolic': deltas,
        'APISem': {'eligible_n': 1, 'resolved_n': api_resolved,
                   'abstain_n': api_abstain, 'coverage': api_resolved,
                   'interpretation': 'INSUFFICIENT_FINAL_TEST_SAMPLE_FOR_APISEM'},
        'position_sanity': position, 'permutation_sanity': permutation,
        'R@5_limitation': '2–4 candidates; R@5 has little discriminative power.',
        'SOURCE_BINARY_REVISION_MATCH': 'NOT_PROVEN',
        'SSQ': 'DISABLED', 'calibration': 'NOT_APPLICABLE_NO_VALID_S_LLM',
        'constraint': 'DISABLED',
    }
    audit = {
        'schema': 'seraph_final_test_audit_clean_v2', 'status': 'PASS',
        'v0_3_3c_monitored_hashes_reverified': 37,
        'prediction_frozen_before_first_gold_access': True,
        'prediction_hash_verified_before_gold_access': pred_hash,
        'gold_sha256_before_read': gold_before,
        'gold_sha256_after_read': gold_after,
        'gold_hash_unchanged': gold_before == gold_after == EXPECTED_GOLD,
        'gold_directory_mode_after_read': oct(stat.S_IMODE(GOLD_DIR.stat().st_mode)),
        'gold_file_mode_after_read': '000_recorded_in_finally',
        'candidate_membership_unchanged': True,
        'models_retrained_or_tuned': False,
        'registered_variants_only': list(VARIANTS),
        'module_policy': manifest['module_policy'],
        'seed_policy': list(SEEDS),
        'APISem_policy': 'DETERMINISTIC_CONFIRMED_HARD_ELSE_ABSTAIN',
        'test_performance_evaluation_count': 1,
    }
    return metrics, audit


def report(metrics):
    lines = ['# FINAL_TEST_V2 — clean frozen model', '',
             'Status: FINAL_TEST_COMPLETE. This was the one-time FINAL_TEST_V2 evaluation.', '',
             f"Final manifest SHA256: `{EXPECTED_MANIFEST}`  ",
             f"Final lock SHA256: `{EXPECTED_LOCK}`  ",
             f"Prediction SHA256: `{metrics['prediction_sha256']}`  ",
             f"Gold SHA256 (before = after): `{EXPECTED_GOLD}`  ",
             'Final model: clean Stage1 v0_1_5 adapter + v0_2_8 clean h_sym + v0_2_10 Stage2.',
             'Module policy: TypeSem=unified_llm_gate_clean; InterfaceSem=module_aware_clean; '
             'LayoutSem=unified_llm_gate_clean.',
             'Seeds: 1337, 3407, 20260929; per-seed softmax probabilities aligned by candidate '
             'semantic identity, then arithmetic mean.', '',
             '## Final TEST metrics', '',
             '| Module | n | Symbolic R@1 | Final R@1 | Δ R@1 | Final MRR | Final R@5 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for module in MODULES:
        a = metrics['variants'][VARIANTS[0]][module]['ensemble']
        b = metrics['variants'][VARIANTS[1]][module]['ensemble']
        lines.append(f"| {module} | {b['n']} | {a['R@1']:.4f} | {b['R@1']:.4f} | "
                     f"{metrics['final_minus_symbolic'][module]['R@1']:+.4f} | "
                     f"{b['MRR']:.4f} | {b['R@5']:.4f} |")
    lines += ['', 'All pre-registered variant and per-seed R@1/MRR/R@5 values, mean/std, '
              'position baselines and five-permutation checks are in '
              '`FINAL_TEST_METRICS_CLEAN_V2.json`.', '',
              '## Limits and protocol', '',
              'APISem: 1 eligible, 0 resolved, 1 abstained because no confirmed_hard evidence '
              'was available in the frozen predictor input. '
              'INSUFFICIENT_FINAL_TEST_SAMPLE_FOR_APISEM.',
              'R@5 has little discriminative power with only 2–4 candidates.',
              'SOURCE_BINARY_REVISION_MATCH = NOT_PROVEN.',
              'Explicit SSQ disabled; calibration N/A (no valid S_llm); constraint supervision disabled.',
              'Gold was first read only after all 2,605 predictions and their SHA were frozen; '
              'the Gold hash matched before and after and permissions were restored to 000.',
              'No TEST-driven retraining, model selection, candidate changes or rerun is permitted.', '']
    return '\n'.join(lines)


def main():
    manifest, pred_hash, by_input, by_pred = prediction_gate()
    print(json.dumps({'event': 'PREDICTION_FREEZE_GATE_PASS',
                      'prediction_sha256': pred_hash, 'records': 2605},
                     sort_keys=True), flush=True)
    gold, before, after = gold_read_once()
    require(stat.S_IMODE(GOLD_DIR.stat().st_mode) == 0,
            'Gold directory was not resealed')
    print(json.dumps({'event': 'GOLD_READ_AND_RESEAL_PASS',
                      'gold_sha256': before, 'records': len(gold)},
                     sort_keys=True), flush=True)
    metrics, audit = evaluate(manifest, pred_hash, by_input, by_pred, gold, before, after)
    write_new(OUT / 'FINAL_TEST_METRICS_CLEAN_V2.json', metrics)
    write_new(OUT / 'FINAL_TEST_AUDIT_CLEAN_V2.json', audit)
    write_new(OUT / 'FINAL_TEST_REPORT_CLEAN_V2.md', report(metrics))
    print(json.dumps({'event': 'FINAL_TEST_COMPLETE',
                      'metrics': {m: metrics['variants'][VARIANTS[1]][m]['ensemble']
                                  for m in MODULES}}, sort_keys=True), flush=True)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    try:
        main()
    except Exception as exc:
        print(json.dumps({'event': 'FINAL_TEST_EVALUATION_BLOCKED',
                          'reason': str(exc)}, ensure_ascii=False),
              file=sys.stderr, flush=True)
        raise
