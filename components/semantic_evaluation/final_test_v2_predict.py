#!/usr/bin/env python3
"""One-shot, Gold-blind FINAL_TEST_V2 inference against the v0_3_3c freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path('/data/chaoni/mjy/resym_exp')
RUNS = ROOT / 'training_runs/seraph_local'
FREEZE = RUNS / 'final_freeze_v0_3_3c'
TEST = RUNS / 'final_test_v2_build'
OUT = RUNS / 'final_test_v2_run'
PACKAGE = ROOT / 'packages/seraph_clean_joint_v0_2_10'
MANIFEST = FREEZE / 'FINAL_MODEL_MANIFEST_CLEAN_V2.json'
LOCK = FREEZE / 'FINAL_MODEL_FROZEN_CLEAN_V2.lock'
INPUT = TEST / 'FINAL_TEST_V2_MANIFEST.jsonl'
GOLD_PARENT = TEST / 'final_test_v2_gold_sealed'
EMBEDDING = TEST / 'final_test_v2_inputs/clean/node_embeddings_clean.pt'
INDEX = TEST / 'final_test_v2_inputs/clean/node_embedding_index_clean.json'
EMBED_REPORT = TEST / 'final_test_v2_inputs/clean/clean_embedding_export_report.json'
CFG = PACKAGE / 'config.json'
EXPECTED_MANIFEST = '82af0f01c2125fb1169f1764be3e1e018e74940a4a12b25ed978c94e31c9ffcd'
EXPECTED_LOCK = '1ab6efae4b38bb0f22597d3467bf80356be8c090370ef3a1b1c42aba733a3c9d'
EXPECTED_TEST = '88f1f3af035ef67d163181a1a9827fe4112bfa60667bbfb20b1375ba7b6a351b'
EXPECTED_GOLD = '78fe1593494f2f9bfd3e3afeb788f336705a10c3503af65c7e31332580bf0aed'
SEEDS = (1337, 3407, 20260929)
MODULES = ('TypeSem', 'InterfaceSem', 'LayoutSem')

sys.path.insert(0, str(PACKAGE / 'src'))
from common import sha256_json  # noqa: E402
from model import CleanJointRanker  # noqa: E402
from permutation_eval import _permutation, _ranks, _spearman  # noqa: E402
from train_clean_joint import FrozenLLMRepresentations, _tensor_payload  # noqa: E402


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze():
    require(sha(MANIFEST) == EXPECTED_MANIFEST, 'v0_3_3c manifest SHA mismatch')
    require(sha(LOCK) == EXPECTED_LOCK, 'v0_3_3c lock SHA mismatch')
    manifest, lock = read_json(MANIFEST), read_json(LOCK)
    require(lock['manifest_sha256'] == EXPECTED_MANIFEST, 'lock/manifest mismatch')
    watched = lock['watched_artifacts']
    require(len(watched) == 37 and len({x['path'] for x in watched}) == 37,
            'not 37 distinct frozen dependencies')
    for item in watched:
        path = Path(item['path'])
        require(path.is_file(), 'missing frozen dependency: ' + str(path))
        require(path.stat().st_size == item['size_bytes'] and sha(path) == item['sha256'],
                'frozen dependency size/SHA mismatch: ' + str(path))
    require(sha(INPUT) == EXPECTED_TEST, 'FINAL_TEST_V2 input SHA mismatch')
    require(manifest['artifacts']['test_manifest']['sha256'] == EXPECTED_TEST,
            'freeze references a different TEST input')
    require(manifest['gold_seal']['preseal_sha256'] == EXPECTED_GOLD,
            'recorded sealed Gold SHA mismatch')
    require(stat.S_IMODE(GOLD_PARENT.stat().st_mode) == 0,
            'Gold parent is not sealed at mode 000')
    require(manifest['status'] == 'FINAL_MODEL_FROZEN_READY_FOR_TEST_AUTHORIZATION',
            'freeze status is not authorized-ready')
    require(manifest['module_policy'] == {
        'TypeSem': 'unified_llm_gate_clean',
        'InterfaceSem': 'module_aware_clean',
        'LayoutSem': 'unified_llm_gate_clean'}, 'module policy drift')
    require(manifest['ensemble']['seeds'] == list(SEEDS), 'ensemble seeds changed')
    require(manifest['ensemble']['normalization'] == 'per-seed softmax over candidate scores',
            'ensemble normalization changed')
    require(manifest['ensemble']['aggregation'] == 'arithmetic mean of aligned per-seed probabilities',
            'ensemble aggregation changed')
    require(not manifest['explicit_ssq_enabled'] and not manifest['calibration_enabled']
            and not manifest['constraint_loss_enabled'], 'disabled branch was enabled')
    require(manifest['APISem_mode'] == 'DETERMINISTIC_CONFIRMED_HARD', 'APISem policy drift')
    require(shutil.disk_usage('/data').free >= 1024**3, 'less than 1 GiB /data free')
    cfg = read_json(CFG)
    require(cfg['llm']['max_length'] == 512 and cfg['llm']['hidden_dim'] == 4096,
            'frozen LLM representation configuration drift')
    require(cfg['llm']['expected_stage1_adapter_sha256'] ==
            manifest['artifacts']['adapter']['sha256'], 'adapter config/freeze mismatch')
    require(read_json(EMBED_REPORT)['checkpoint'] == manifest['artifacts']['hsym_checkpoint']['path'],
            'TEST clean h_sym derived from unexpected checkpoint')
    require(read_json(EMBED_REPORT)['shape'] == [391932, 64],
            'TEST clean h_sym export shape drift')
    require(read_json(TEST / 'CLEAN_HSYM_STAGE_STATUS.json')['status'] == 'PASSED',
            'TEST clean h_sym status not PASSED')
    selection = read_json(manifest['artifacts']['selection_contract']['path'])
    require(selection['candidate_order_sha256'] ==
            manifest['artifacts']['candidate_order_contract_sha256'],
            'frozen candidate-order contract mismatch')
    registry = read_json(manifest['variant_registry']['path'])
    require(set(registry).issuperset({'A', 'B'}) and registry['A']['arm'] == 'symbolic_only_clean'
            and registry['B']['module_policy'] == manifest['module_policy'],
            'variant registry drift')
    print(json.dumps({'event': 'PREFLIGHT_PASS', 'watched': 37,
                      'data_free_bytes': shutil.disk_usage('/data').free,
                      'gold_parent_mode': '000'}, sort_keys=True), flush=True)
    return manifest, cfg, registry


def load_rows(index):
    rows = [json.loads(line) for line in INPUT.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(rows) == 2605, 'TEST record count is not 2605')
    require(len({r['record_id'] for r in rows}) == 2605, 'duplicate TEST record ID')
    require(Counter(r['module'] for r in rows) == {
        'TypeSem': 1024, 'InterfaceSem': 812, 'LayoutSem': 768, 'APISem': 1},
        'TEST module counts drift')
    for row in rows:
        ids = row['candidate_instance_identities']
        semantic = row['candidate_semantic_identities']
        positions = row['clean_hsym_rows']
        n = row['candidate_count']
        require(n == len(ids) == len(semantic) == len(positions['candidates'])
                and 2 <= n <= 4, 'candidate size mismatch: ' + row['record_id'])
        require(len(set(ids)) == n and len(set(semantic)) == n,
                'duplicate candidate identity: ' + row['record_id'])
        require(row['candidate_order'] == ids,
                'candidate order differs from embedding order: ' + row['record_id'])
        require(index[row['query_identity']] == positions['query'],
                'query h_sym index mismatch: ' + row['record_id'])
        require([index[c] for c in ids] == positions['candidates'],
                'candidate h_sym index mismatch: ' + row['record_id'])
        require(isinstance(row['pseudo_c'], str) and row['pseudo_c'].strip(),
                'missing pseudo-C: ' + row['record_id'])
        require(not any('gold' in key.lower() or 'label' in key.lower() for key in row),
                'Gold/label field in TEST input: ' + row['record_id'])
    print(json.dumps({'event': 'TEST_INPUT_PASS', 'rows': len(rows),
                      'unique_pseudo_c': len({sha256_json(r['pseudo_c']) for r in rows})},
                     sort_keys=True), flush=True)
    return rows


def load_checkpoint(item, expected_arm, expected_seed):
    require(sha(item['path']) == item['sha256'], 'checkpoint changed: ' + item['path'])
    payload = torch.load(item['path'], map_location='cpu', weights_only=False)
    require(payload['arm'] == expected_arm and payload['seed'] == expected_seed,
            'checkpoint arm/seed mismatch')
    require(payload['symbolic_dim'] == 64 and payload['llm_dim'] == 4096,
            'checkpoint dimensions differ')
    model = CleanJointRanker(payload['arm'], payload['seed'],
                             symbolic_dim=payload['symbolic_dim'],
                             llm_dim=payload['llm_dim'],
                             bridge_dim=payload['bridge_dim'],
                             modules=payload['modules'],
                             apisem_enabled=payload['apisem_enabled'],
                             temperature=payload['temperature'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def rank(score_map):
    return sorted(score_map, key=lambda key: (-score_map[key], key))


def model_logits(model, module, query, candidates, h_llm):
    fused, _, _ = model.fuse(module, query, h_llm)
    values, _ = model.score(module, fused, candidates)
    result = [float(x) for x in values[0].tolist()]
    require(all(math.isfinite(x) for x in result), 'NaN/Inf model logits')
    return result


def variant(model_map, module, semantic, query, candidates, h_llm):
    per_seed = {}
    probabilities = []
    for seed in SEEDS:
        logits = model_logits(model_map[seed], module, query, candidates, h_llm)
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float64), 0).tolist()
        per_seed[str(seed)] = {
            'logits': dict(zip(semantic, logits)),
            'probabilities': dict(zip(semantic, probs)),
            'predicted_candidate_semantic_id': rank(dict(zip(semantic, probs)))[0],
            'ranked_candidate_semantic_ids': rank(dict(zip(semantic, probs))),
        }
        probabilities.append(probs)
    mean = [sum(x[i] for x in probabilities) / len(SEEDS) for i in range(len(semantic))]
    score_map = dict(zip(semantic, mean))
    require(all(math.isfinite(x) for x in mean), 'NaN/Inf ensemble probability')
    return {'per_seed_scores': per_seed, 'ensemble_scores': score_map,
            'predicted_candidate_semantic_id': rank(score_map)[0],
            'ranked_candidate_semantic_ids': rank(score_map)}


def permute_final(model_map, row, query, candidates, h_llm, reference):
    semantic = row['candidate_semantic_identities']
    reference_ranks = _ranks(reference['ensemble_scores'])
    top1 = reference['predicted_candidate_semantic_id']
    outcomes = []
    for i in range(5):
        order = _permutation(semantic, 20260929, row['record_id'], i)
        indices = [semantic.index(item) for item in order]
        permuted = candidates[indices]
        result = variant(model_map, row['module'], order, query, permuted, h_llm)
        ranks = _ranks(result['ensemble_scores'])
        outcomes.append({'permutation': i, 'candidate_semantic_order': order,
                         'top1_candidate_semantic_id': result['predicted_candidate_semantic_id'],
                         'top1_identity_agreement':
                         int(result['predicted_candidate_semantic_id'] == top1),
                         'rank_consistency_spearman': _spearman(reference_ranks, ranks)})
    return outcomes


def predict():
    started = time.monotonic()
    manifest, cfg, registry = verify_freeze()
    require(not OUT.exists(), 'FINAL_TEST_V2 output already exists; one-shot run refuses overwrite')
    index = read_json(INDEX)
    require(len(index) == 391932, 'TEST h_sym index length drift')
    rows = load_rows(index)
    symbolic = _tensor_payload(EMBEDDING)
    require(list(symbolic.shape) == [391932, 64] and bool(torch.isfinite(symbolic).all()),
            'TEST h_sym shape/finite validation failed')
    require(torch.cuda.is_available(), 'no CUDA GPU available')
    free, total = torch.cuda.mem_get_info(0)
    require(free >= 40 * 1024**3, 'selected GPU 0 has <40 GiB free VRAM')
    torch.cuda.reset_peak_memory_stats(0)
    OUT.mkdir(parents=True, exist_ok=False)
    proof = {'schema': 'seraph_final_test_v2_authorization_and_preflight',
             'authorization': 'explicit user authorization in current task',
             'final_test_authorized': True, 'manifest_sha256': EXPECTED_MANIFEST,
             'lock_sha256': EXPECTED_LOCK, 'monitored_hashes_verified': 37,
             'test_manifest_sha256': EXPECTED_TEST, 'gold_opened': False,
             'gold_parent_mode': '000', 'gpu_index': 0,
             'gpu_free_bytes_before': free, 'data_free_bytes_before': shutil.disk_usage('/data').free,
             'clean_test_embedding_sha256': sha(EMBEDDING),
             'clean_test_index_sha256': sha(INDEX)}
    (OUT / 'FINAL_TEST_PREFLIGHT.json').write_text(
        json.dumps(proof, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'GPU_READY', 'gpu_index': 0, 'gpu_free_bytes': free,
                      'data_free_bytes': proof['data_free_bytes_before']}, sort_keys=True), flush=True)

    contract = {'manifest_sha256': EXPECTED_TEST, 'training_config_sha256': sha(CFG)}
    llm = FrozenLLMRepresentations(cfg, 'unified_llm_gate_clean',
                                   [r for r in rows if r['module'] in MODULES], [],
                                   torch.device('cuda:0'), OUT / 'representation_cache', contract)
    require(llm.adapter_hash_before == llm.adapter_hash_after ==
            manifest['artifacts']['adapter']['sha256'], 'Stage1 adapter changed during extraction')
    llm.release()
    print(json.dumps({'event': 'LLM_REPRESENTATIONS_READY',
                      'unique_functions': len(llm.function_cache),
                      'elapsed_seconds': round(time.monotonic() - started, 1),
                      'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                      'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(0)},
                     sort_keys=True), flush=True)

    models = {'A': {}, 'B': {}}
    for item in registry['A']['checkpoints']:
        models['A'][item['seed']] = load_checkpoint(item, 'symbolic_only_clean', item['seed'])
    for module in MODULES:
        models['B'][module] = {}
        for item in registry['B']['checkpoints'][module]:
            arm = manifest['module_policy'][module]
            models['B'][module][item['seed']] = load_checkpoint(item, arm, item['seed'])
    require(all(set(models['A']) == set(SEEDS) and set(models['B'][m]) == set(SEEDS)
                for m in MODULES), 'missing registered checkpoint seed')

    pending = OUT / 'FINAL_TEST_V2_PREDICTIONS_FROZEN.pending.jsonl'
    with torch.no_grad(), pending.open('x', encoding='utf-8') as stream:
        for i, row in enumerate(rows, 1):
            semantic = row['candidate_semantic_identities']
            record = {
                'schema': 'seraph_final_test_v2_frozen_prediction',
                'record_id': row['record_id'], 'module': row['module'],
                'candidate_instance_identities': row['candidate_instance_identities'],
                'candidate_semantic_ids': semantic,
                'candidate_membership_sha256': row['candidate_membership_sha256'],
                'candidate_count': row['candidate_count'],
                'final_model_manifest_sha256': EXPECTED_MANIFEST,
                'variants': {}, 'abstain': False,
            }
            if row['module'] == 'APISem':
                # The frozen TEST input has no confirmed_hard evidence field.
                record['abstain'] = True
                record['abstain_reason'] = 'NO_CONFIRMED_HARD_EVIDENCE_IN_FROZEN_TEST_INPUT'
                record['predicted_candidate_semantic_id'] = None
                record['ranked_candidate_semantic_ids'] = []
                record['permutation_robustness'] = []
            else:
                q = symbolic[int(row['clean_hsym_rows']['query'])].unsqueeze(0)
                c = symbolic[row['clean_hsym_rows']['candidates']]
                h = llm.function_cache[sha256_json(row['pseudo_c'])].unsqueeze(0)
                a = variant(models['A'], row['module'], semantic, q, c, None)
                b = variant(models['B'][row['module']], row['module'], semantic, q, c, h)
                record['variants'] = {'A_clean_symbolic_only': a, 'B_final_clean_stage2': b}
                record['predicted_candidate_semantic_id'] = b['predicted_candidate_semantic_id']
                record['ranked_candidate_semantic_ids'] = b['ranked_candidate_semantic_ids']
                record['permutation_robustness'] = permute_final(
                    models['B'][row['module']], row, q, c, h, b)
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False) + '\n')
            if i == 150 or i % 500 == 0 or i == len(rows):
                stream.flush()
                elapsed = time.monotonic() - started
                rate = i / max(elapsed, 1e-9)
                print(json.dumps({'event': 'PREDICTION_PROGRESS', 'processed_records': i,
                                  'total_records': len(rows), 'records_per_second': rate,
                                  'elapsed_seconds': elapsed,
                                  'eta_seconds': (len(rows) - i) / max(rate, 1e-9),
                                  'gpu_index': 0,
                                  'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                                  'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
                                  'data_free_bytes': shutil.disk_usage('/data').free},
                                 sort_keys=True), flush=True)
    check_prediction(pending, rows)
    frozen = OUT / 'FINAL_TEST_V2_PREDICTIONS_FROZEN.jsonl'
    require(not frozen.exists(), 'prediction destination already exists')
    os.replace(pending, frozen)
    prediction_sha = sha(frozen)
    (OUT / 'FINAL_TEST_V2_PREDICTIONS_FROZEN.sha256').write_text(
        prediction_sha + '  ' + frozen.name + '\n', encoding='utf-8')
    require(sha(frozen) == prediction_sha, 'prediction SHA changed after freeze')
    verify_freeze()
    require(stat.S_IMODE(GOLD_PARENT.stat().st_mode) == 0,
            'Gold parent opened before prediction freeze')
    freeze = {'schema': 'seraph_final_test_v2_prediction_freeze',
              'status': 'PASS', 'prediction_path': str(frozen),
              'prediction_sha256': prediction_sha, 'prediction_count': 2605,
              'duplicate_count': 0, 'missing_count': 0, 'invalid_candidate_id_count': 0,
              'nonfinite_count': 0, 'candidate_membership_unchanged': True,
              'manifest_sha256': EXPECTED_MANIFEST, 'lock_sha256': EXPECTED_LOCK,
              'model_hashes_reverified_after_prediction': True,
              'gold_ever_opened_before_freeze': False,
              'gold_parent_mode_before_evaluation': '000',
              'gold_preseal_sha256_recorded_only': EXPECTED_GOLD,
              'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
              'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
              'elapsed_seconds': time.monotonic() - started,
              'data_free_bytes': shutil.disk_usage('/data').free}
    (OUT / 'FINAL_TEST_PREDICTION_FREEZE.json').write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'PREDICTION_FROZEN', 'prediction_sha256': prediction_sha,
                      'records': 2605, 'elapsed_seconds': freeze['elapsed_seconds']},
                     sort_keys=True), flush=True)


def check_prediction(path, rows):
    expected = {r['record_id']: r for r in rows}
    got = {}
    with Path(path).open('r', encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            rid = item['record_id']
            require(rid not in got and rid in expected, 'duplicate/unexpected prediction ID')
            source = expected[rid]
            semantic = source['candidate_semantic_identities']
            require(item['candidate_semantic_ids'] == semantic and
                    item['candidate_instance_identities'] == source['candidate_instance_identities'] and
                    item['candidate_membership_sha256'] == source['candidate_membership_sha256'],
                    'candidate membership drift')
            if source['module'] == 'APISem':
                require(item['abstain'] and item['predicted_candidate_semantic_id'] is None,
                        'APISem without confirmed_hard must abstain')
            else:
                require(not item['abstain'] and item['predicted_candidate_semantic_id'] in semantic,
                        'invalid predicted candidate')
                require(set(item['ranked_candidate_semantic_ids']) == set(semantic),
                        'invalid ranked candidate set')
                require(set(item['variants']) == {'A_clean_symbolic_only', 'B_final_clean_stage2'},
                        'registered variant missing')
                for variant_value in item['variants'].values():
                    require(set(variant_value['per_seed_scores']) == {str(s) for s in SEEDS},
                            'missing seed scores')
                    require(set(variant_value['ensemble_scores']) == set(semantic),
                            'ensemble candidate coverage mismatch')
                    require(all(math.isfinite(float(v)) for v in variant_value['ensemble_scores'].values()),
                            'non-finite ensemble score')
                require(len(item['permutation_robustness']) == 5,
                        'not five deterministic permutations')
            got[rid] = item
    require(len(got) == len(expected) == 2605 and set(got) == set(expected),
            'incomplete prediction output')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'predict'))
    args = parser.parse_args()
    try:
        if args.action == 'preflight':
            verify_freeze()
        else:
            predict()
    except Exception as exc:
        print(json.dumps({'event': 'FINAL_TEST_PREDICTION_BLOCKED',
                          'reason': str(exc)}, ensure_ascii=False),
              file=sys.stderr, flush=True)
        raise
