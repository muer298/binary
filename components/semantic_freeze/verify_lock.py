#!/usr/bin/env python3
"""SERAPH v0_3_3b identity-only freeze. Never open sealed TEST Gold."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

R = Path('/data/chaoni/mjy/resym_exp/training_runs/seraph_local')
P = Path('/data/chaoni/mjy/resym_exp/packages/seraph_final_freeze_v0_3_3b')
O = R / 'final_freeze_v0_3_3b'
B = R / 'final_test_v2_build'
J = R / 'clean_joint_v0_2_10'
H = R / 'clean_hsym_v0_2_8'
A = R / 'stage1_clean_v0_1_5' / 'contrastive' / 'adapter'
MODEL = Path('/data/chaoni/mjy/DeepSeek-Coder-6.7B-Base')
G = R / 'joint_v0_2_7_global_split' / 'joint_manifest_global_clean_v0_2_7.jsonl'
TEST_MANIFEST = B / 'FINAL_TEST_V2_MANIFEST.jsonl'
GOLD = B / 'final_test_v2_gold_sealed' / 'FINAL_TEST_V2_GOLD_SEALED.jsonl'
EXPECTED_TEST = '88f1f3af035ef67d163181a1a9827fe4112bfa60667bbfb20b1375ba7b6a351b'
EXPECTED_GOLD = '78fe1593494f2f9bfd3e3afeb788f336705a10c3503af65c7e31332580bf0aed'
EXPECTED_GLOBAL = '769b0ae6d75a484f5071a96eec5279edf90c58eb033bff12ad91a79571b211b0'
SEEDS = [1337, 3407, 20260929]
MODULE_POLICY = {'TypeSem': 'unified_llm_gate_clean', 'InterfaceSem': 'module_aware_clean', 'LayoutSem': 'unified_llm_gate_clean'}


def demand(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact(path, expected=None):
    path = Path(path)
    demand(path.is_file(), f'missing artifact: {path}')
    actual = digest(path)
    if expected:
        demand(actual == expected, f'hash mismatch: {path}: {actual} != {expected}')
    return {'path': str(path), 'sha256': actual, 'size_bytes': path.stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')


def identities():
    projects, binaries, functions, candidates = set(), set(), set(), set()
    counts = Counter()
    with TEST_MANIFEST.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            p, b = row['project_id'], row['binary_id']
            demand(isinstance(p, str) and isinstance(b, str), 'missing TEST identity')
            projects.add(p); binaries.add(b)
            fid = row['function_identity']
            demand(fid['binary_id'] == b and fid['project_id'] == p, 'function identity not namespaced')
            functions.add((b, str(fid['function_entry'])))
            for c in row['candidate_instance_identities']:
                demand(c.startswith(b + '::'), 'candidate identity not binary namespaced')
                candidates.add(c)
            counts[row['module']] += 1
    demand(sum(counts.values()) == 2605, 'TEST record count changed')
    demand(dict(counts) == {'TypeSem': 1024, 'InterfaceSem': 812, 'LayoutSem': 768, 'APISem': 1}, 'TEST module counts changed')
    demand(len(projects) == 4 and len(binaries) == 4, 'TEST project/binary count changed')
    registry = read_json(B / 'ALL_USED_PROJECTS_AND_BINARIES.json')
    historical_projects = set(registry['projects'])
    historical_binaries = set(registry['binary_ids'])
    p_overlap = sorted(projects & historical_projects)
    b_overlap = sorted(binaries & historical_binaries)
    matrix = read_json(B / 'FINAL_TEST_V2_LEAKAGE_MATRIX.json')
    checks = {}
    for name, row in matrix['matrix'].items():
        zero = all(row['overlap_counts'][kind] == 0 and not row['overlaps'][kind]
                   for kind in ('project', 'binary', 'function', 'candidate_instance'))
        zero &= row.get('function_and_candidate_instance_identity_include_binary_id') is True
        checks[name] = bool(zero)
    demand(all(checks.values()), 'recorded historical overlap is nonzero')
    demand(not p_overlap and not b_overlap, 'fresh project/binary overlap with historical registry')
    demand(matrix['final_counts'] == {'project': 4, 'binary': 4, 'function': len(functions), 'candidate_instance': len(candidates)},
           'fresh identity counts differ from leakage matrix')
    return {
        'projects': sorted(projects), 'binaries': sorted(binaries),
        'fresh_identity_counts': dict(matrix['final_counts']),
        'historical_project_overlap': p_overlap, 'historical_binary_overlap': b_overlap,
        'historical_categories': checks,
        'function_candidate_zero_inference': 'Binary-namespaced identities; fresh binary overlap 0. Original category matrices rechecked for all 4 identity levels.',
        'evidence': {
            'leakage_matrix': artifact(B / 'FINAL_TEST_V2_LEAKAGE_MATRIX.json'),
            'historical_registry': artifact(B / 'ALL_USED_PROJECTS_AND_BINARIES.json'),
        },
    }


def disk_gate():
    data = shutil.disk_usage('/data')
    mounts = {}
    for mount in ('/data', '/', '/nasdata'):
        if Path(mount).is_dir():
            d = shutil.disk_usage(mount)
            mounts[mount] = {'free_bytes': d.free, 'total_bytes': d.total}
    # No checkpoint copies or Ghidra extraction during freeze or future inference.
    # Estimates deliberately include a 10x multiplier on the 17 MB input manifest.
    estimates = {
        'freeze_reports_and_lock_bytes': 20 * 1024**2,
        'prediction_bytes': 200 * 1024**2,
        'evaluation_bytes': 50 * 1024**2,
        'permutation_audit_bytes': 500 * 1024**2,
        'logs_and_runtime_scratch_bytes': 500 * 1024**2,
        'reserved_free_margin_bytes': 1024**3,
    }
    required = sum(estimates.values())
    return {'mounts': mounts, 'estimated_budget': estimates, 'required_bytes': required,
            'data_free_bytes': data.free, 'passed': data.free >= required,
            'caveat': 'Planning estimate, not a guarantee. Recheck disk before authorized TEST; stop if a later implementation needs larger caches or Ghidra extraction.'}


def preflight():
    status = read_json(B / 'FINAL_TEST_V2_FINAL_STATUS.json')
    demand(status['status'] == 'READY_FOR_FINAL_FREEZE_CLEAN_V2', 'TEST V2 build not ready')
    demand(status['performance_evaluation_count'] == 0 and status['stage2_prediction_count'] == 0, 'TEST already used')
    demand(status['gold_directory_mode_octal'] == '000' and status['gold_sha256'] == EXPECTED_GOLD, 'Gold seal record differs')
    demand((GOLD.parent.stat().st_mode & 0o777) == 0, 'Gold directory is not mode 000')
    # Directory is deliberately inaccessible; do not stat/open its child. Existence and hash rely on pre-seal status.
    test = artifact(TEST_MANIFEST, EXPECTED_TEST)
    global_manifest = artifact(G, EXPECTED_GLOBAL)
    selection_path = J / 'STAGE2_FINAL_SELECTION_CONTRACT_CLEAN_V2.json'
    selection = read_json(selection_path)
    demand(selection['status'] == 'FROZEN_CLEAN_V2' and selection['selection_source'] == 'v0_2_10_DEV_only', 'selection not frozen')
    demand(selection['seeds'] == SEEDS, 'seed list changed')
    demand(selection['manifest_sha256'] == global_manifest['sha256'], 'global manifest binding differs')
    demand(selection['ensemble_rule'] == 'per-seed softmax then arithmetic mean aligned by candidate semantic identity', 'ensemble rule changed')
    demand(selection['explicit_ssq_enabled'] is False and selection['calibration_mode'] == 'NOT_APPLICABLE_NO_VALID_S_LLM', 'unsafe SSQ/calibration')
    demand(selection['constraint_mode'] == 'NO_EXECUTABLE_CONSTRAINT_SUPERVISION', 'unsafe constraint mode')
    demand(selection['FINAL_TEST_V2_prediction_count'] == 0 and selection['FINAL_TEST_V2_evaluation_count'] == 0, 'selection includes TEST use')
    adapter = artifact(A / 'adapter_model.safetensors', selection['clean_stage1_adapter_sha256'])
    hsym = artifact(H / 'node_embeddings_clean.pt', selection['clean_hsym_sha256'])
    hstatus = read_json(H / 'v0_2_8_final_status.json')
    demand(hstatus['status'] == 'READY_FOR_V0_2_9_CLEAN_JOINT', 'h_sym status not ready')
    clean_report = read_json(H / 'clean_embedding_export_report.json')
    demand(clean_report['shape'] == [39793, 64] and clean_report['index_count'] == 39793, 'h_sym shape/index invalid')
    demand(clean_report['nan_count'] == 0 and clean_report['inf_count'] == 0, 'h_sym non-finite')
    stage1_status = read_json(R / 'stage1_clean_v0_1_5' / 'v0_1_5_final_status.json')
    demand('READY' in str(stage1_status.get('status')), 'Stage1 clean status not ready')
    stage2_status = read_json(J / 'v0_2_10_final_status.json')
    demand('READY' in str(stage2_status.get('status')), 'Stage2 clean status not ready')
    checkpoint_mapping = {}
    for module, arm in MODULE_POLICY.items():
        selected = selection['module_policy'][module]
        demand(selected['selected_arm'] == arm, f'{module} selected arm changed')
        rows = selected['checkpoint_artifacts']
        demand([r['seed'] for r in rows] == SEEDS, f'{module} seeds changed')
        checkpoint_mapping[module] = [{'seed': r['seed'], **artifact(r['path'], r['sha256'])} for r in rows]
    baseline = [{'seed': seed, **artifact(J / 'arms' / 'symbolic_only_clean' / f'seed_{seed}' / 'ranker.pt')} for seed in SEEDS]
    isolation = identities()
    disk = disk_gate()
    demand(disk['passed'], f"disk gate failed: {disk['data_free_bytes']} free < {disk['required_bytes']} required")
    base_weights = [artifact(path) for path in sorted(MODEL.glob('model-*.safetensors'))]
    demand(len(base_weights) == 2, 'expected two base-model safetensors shards')
    return {'status': 'PREFLIGHT_PASSED', 'test_manifest': test, 'gold_recorded_sha256': EXPECTED_GOLD,
            'gold_current_content_rehashed': False, 'gold_current_file_existence_independently_verified': False,
            'gold_seal_evidence': artifact(B / 'FINAL_TEST_V2_FINAL_STATUS.json'),
            'global_manifest': global_manifest, 'selection_contract': artifact(selection_path),
            'candidate_order_contract_sha256': selection['candidate_order_sha256'],
            'adapter': adapter, 'adapter_config': artifact(A / 'adapter_config.json'),
            'stage1_projection': artifact(A.parent / 'projection.pt'),
            'base_model_config': artifact(MODEL / 'config.json'),
            'base_model_index': artifact(MODEL / 'model.safetensors.index.json'),
            'base_model_tokenizer': artifact(MODEL / 'tokenizer.json'),
            'base_model_weights': base_weights,
            'hsym_embeddings': hsym, 'hsym_checkpoint': artifact(H / 'clean_alignment_checkpoint.pt'),
            'hsym_index': artifact(H / 'node_embedding_index_clean.json'),
            'hsym_feature_schema': artifact(H / 'clean_feature_schema.json'),
            'hsym_feature_vocab': artifact(H / 'clean_feature_vocab.json'),
            'stage1_status': artifact(R / 'stage1_clean_v0_1_5' / 'v0_1_5_final_status.json'),
            'stage2_status': artifact(J / 'v0_2_10_final_status.json'),
            'checkpoint_mapping': checkpoint_mapping, 'symbolic_baseline_checkpoints': baseline,
            'isolation': isolation, 'disk': disk,
            'provenance': artifact(B / 'SOURCE_AND_BINARY_PROVENANCE.json')}


def freeze(pf):
    demand(not O.exists(), f'output exists, refusing overwrite: {O}')
    O.mkdir(parents=True, exist_ok=False)
    metric = {'schema': 'seraph_final_metric_contract_clean_v2', 'modules': {
        x: {'primary': ['R@1', 'MRR', 'R@5'], 'candidate_identity_alignment': True}
        for x in MODULE_POLICY},
        'R@5_limitation': 'Usually 2-4 candidates; R@5 has limited discriminative value.',
        'APISem': {'primary': ['resolved_coverage', 'abstain_rate'], 'final_test_n': 1,
                   'interpretation': 'INSUFFICIENT_FINAL_TEST_SAMPLE_FOR_APISEM; no generalization claim'},
        'split': 'FINAL_TEST_V2', 'frozen_before_gold_access': True}
    apisem = {'schema': 'seraph_apisem_final_inference_contract_v2', 'mode': 'DETERMINISTIC_CONFIRMED_HARD',
              'if_confirmed_hard_candidate_exists': 'emit that candidate',
              'otherwise': 'ABSTAIN / UNRESOLVED', 'learned_head_fallback': False,
              'final_test_n': 1, 'generalization_claim_allowed': False}
    variant = {'schema': 'seraph_final_variant_registry_clean_v2', 'registered_before_gold_access': True,
               'A': {'name': 'clean symbolic-only baseline', 'arm': 'symbolic_only_clean',
                     'seeds': SEEDS, 'checkpoints': pf['symbolic_baseline_checkpoints']},
               'B': {'name': 'FINAL clean Stage2', 'module_policy': MODULE_POLICY,
                     'seeds': SEEDS, 'checkpoints': pf['checkpoint_mapping']},
               'failed_SSQ_registered': False, 'new_variants_after_gold_access_allowed': False}
    write_json(O / 'APISEM_FINAL_INFERENCE_CONTRACT_V2.json', apisem)
    write_json(O / 'FINAL_METRIC_CONTRACT_V2.json', metric)
    write_json(O / 'FINAL_VARIANT_REGISTRY_V2.json', variant)
    manifest = {'schema': 'seraph_final_model_manifest_clean_v2', 'pipeline_version': 'v0_3_3b',
                'status': 'FINAL_MODEL_FROZEN_READY_FOR_TEST_AUTHORIZATION', 'FINAL_TEST_AUTHORIZED': False,
                'source_binary_revision_match': 'NOT_PROVEN', 'test_performance_evaluation_count': 0,
                'source_snapshots': [{k: row[k] for k in ('project', 'binary_id', 'source_commit', 'source_tree')}
                                     for row in read_json(B / 'SOURCE_AND_BINARY_PROVENANCE.json')['records']],
                'artifacts': {k: v for k, v in pf.items() if k not in ('isolation', 'disk', 'checkpoint_mapping', 'symbolic_baseline_checkpoints', 'status', 'gold_recorded_sha256', 'gold_current_content_rehashed', 'gold_current_file_existence_independently_verified')},
                'gold_seal': {'path': str(GOLD), 'preseal_sha256': EXPECTED_GOLD,
                              'directory_mode': '000', 'current_content_rehashed': False,
                              'current_file_existence_independently_verified': False},
                'module_policy': MODULE_POLICY, 'checkpoint_mapping': pf['checkpoint_mapping'],
                'ensemble': {'seeds': SEEDS, 'normalization': 'per-seed softmax over candidate scores',
                             'candidate_alignment': 'candidate semantic identity',
                             'aggregation': 'arithmetic mean of aligned per-seed probabilities'},
                'APISem_mode': 'DETERMINISTIC_CONFIRMED_HARD', 'explicit_ssq_enabled': False,
                'SSQ_status': 'BLOCKED_SSQ_SCORING', 'calibration_enabled': False,
                'calibration_status': 'NOT_APPLICABLE_NO_VALID_S_LLM',
                'constraint_loss_enabled': False, 'constraint_mode': 'NO_EXECUTABLE_CONSTRAINT_SUPERVISION',
                'metric_contract': artifact(O / 'FINAL_METRIC_CONTRACT_V2.json'),
                'variant_registry': artifact(O / 'FINAL_VARIANT_REGISTRY_V2.json'),
                'apisem_contract': artifact(O / 'APISEM_FINAL_INFERENCE_CONTRACT_V2.json')}
    manifest_path = O / 'FINAL_MODEL_MANIFEST_CLEAN_V2.json'
    write_json(manifest_path, manifest)
    manifest_art = artifact(manifest_path)
    watched = [pf[x] for x in ('test_manifest', 'global_manifest', 'selection_contract', 'adapter', 'adapter_config',
                              'stage1_projection', 'base_model_config', 'base_model_index', 'base_model_tokenizer',
                              'hsym_embeddings', 'hsym_checkpoint', 'hsym_index', 'hsym_feature_schema',
                              'hsym_feature_vocab', 'stage1_status', 'stage2_status', 'provenance')]
    watched.extend(pf['isolation']['evidence'].values())
    watched.extend(pf['base_model_weights'])
    watched.append(artifact(P / 'verify_lock.py'))
    watched.extend([r for rows in pf['checkpoint_mapping'].values() for r in rows])
    watched.extend(pf['symbolic_baseline_checkpoints'])
    watched.extend([manifest_art, manifest['metric_contract'], manifest['variant_registry'], manifest['apisem_contract']])
    unique = {r['path']: {'path': r['path'], 'sha256': r['sha256'], 'size_bytes': r['size_bytes']} for r in watched}
    lock = {'schema': 'seraph_final_model_lock_clean_v2', 'freeze_timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'manifest_sha256': manifest_art['sha256'], 'watched_artifacts': list(unique.values()),
            'recorded_gold_sha256': EXPECTED_GOLD, 'gold_content_opened': False,
            'FINAL_TEST_AUTHORIZED': False, 'test_performance_evaluation_count': 0,
            'test_execution_guard': 'verify_lock.py verify --for-test requires a separate, explicit FINAL_TEST_AUTHORIZATION.json bound to this manifest; absent now'}
    write_json(O / 'FINAL_MODEL_FROZEN_CLEAN_V2.lock', lock)
    leakage = pf['isolation']
    lines = ['# FINAL_TEST_V2 leakage audit', '',
             'Identity-only audit; sealed Gold and TEST metrics not opened.', '',
             f"Fresh records: 2605; projects {leakage['fresh_identity_counts']['project']}; binaries {leakage['fresh_identity_counts']['binary']}; functions {leakage['fresh_identity_counts']['function']}; candidate instances {leakage['fresh_identity_counts']['candidate_instance']}.", '',
             'Fresh project/binary overlap with historical registry: 0 / 0.',
             'Function and candidate identities are binary-namespaced; fresh binary overlap is zero.', '',
             '| Historical category | Project/Binary/Function/Candidate overlap zero |', '|---|---|']
    lines.extend(f'| {k} | {"YES" if v else "NO"} |' for k, v in leakage['historical_categories'].items())
    lines.extend(['', f"Original leakage matrix SHA256: {leakage['evidence']['leakage_matrix']['sha256']}",
                  f"Historical registry SHA256: {leakage['evidence']['historical_registry']['sha256']}",
                  'Project/binary overlaps independently recomputed; function/candidate overlap follows binary namespace and recorded category matrices.', ''])
    (O / 'FINAL_LEAKAGE_AUDIT_V2.md').write_text('\n'.join(lines), encoding='utf-8')
    d = pf['disk']
    disk_lines = ['# Disk preflight', '', '| Mount | Free bytes |', '|---|---:|']
    disk_lines.extend(f'| {m} | {v["free_bytes"]} |' for m, v in d['mounts'].items())
    disk_lines.extend(['', '| Planned item | Budget bytes |', '|---|---:|'])
    disk_lines.extend(f'| {k} | {v} |' for k, v in d['estimated_budget'].items())
    disk_lines.extend(['', f"Required including 1 GiB free reserve: {d['required_bytes']} bytes.",
                       f"Observed /data free: {d['data_free_bytes']} bytes. Gate: {'PASS' if d['passed'] else 'BLOCKED'}.",
                       'This estimate assumes no new Ghidra export or checkpoint copy. Recheck before any later TEST run.',
                       'No historical cache/temp files were deleted.', ''])
    (O / 'FINAL_DISK_PREFLIGHT.md').write_text('\n'.join(disk_lines), encoding='utf-8')
    readiness = ['# FINAL_TEST_V2 readiness', '',
                 'Status: FINAL_MODEL_FROZEN_READY_FOR_TEST_AUTHORIZATION',
                 'FINAL_TEST_AUTHORIZED = false. No final-test-predict/evaluate was run.', '',
                 'TRAIN/DEV/CALIBRATION/compromised TEST/SSQ historical project and binary firewall: PASS.',
                 'Candidate-position balance: PASS in sealed build status.',
                 'Clean h_sym provenance and Stage-2 selection hashes: PASS.',
                 'SSQ disabled; APISem deterministic confirmed_hard or abstain; constraint disabled; calibration N/A.',
                 'Gold directory mode 000. Gold content was neither opened nor rehashed.',
                 'Gold child existence cannot be independently stat-verified under mode 000; pre-seal status supplies existence and SHA evidence.',
                 'SOURCE_BINARY_REVISION_MATCH = NOT_PROVEN.',
                 'Disk budget passes only for bounded prediction/evaluation implementation; recheck before authorization.',
                 'Any watched hash change must fail verify_lock.py.', '']
    (O / 'FINAL_TEST_READINESS_V2.md').write_text('\n'.join(readiness), encoding='utf-8')
    report = ['# v0_3_3b final model freeze', '',
              'Status: FINAL_MODEL_FROZEN_READY_FOR_TEST_AUTHORIZATION',
              'FINAL_TEST_AUTHORIZED = false. TEST prediction/evaluation count remains 0.', '',
              'Frozen clean stack: v0_1_5 Stage-1 adapter + v0_2_8 clean h_sym + v0_2_10 Stage-2.',
              'TypeSem: unified_llm_gate_clean; InterfaceSem: module_aware_clean; LayoutSem: unified_llm_gate_clean.',
              'All three modules use seeds 1337, 3407, 20260929 with per-seed softmax, semantic-identity alignment, arithmetic-mean ensemble.',
              'APISem: confirmed_hard deterministic evidence; otherwise abstain. Learned fallback disabled; n=1 is insufficient for generalization.',
              'Explicit SSQ disabled after development gate; calibration N/A; constraint disabled.',
              'Registered variants: A clean symbolic-only baseline; B final clean Stage-2. No failed SSQ variant.',
              'Metrics frozen before Gold access: per-module R@1, MRR, R@5; APISem resolved coverage and abstain rate.',
              'R@5 has limited discrimination when candidate counts are small.',
              'SOURCE_BINARY_REVISION_MATCH = NOT_PROVEN.',
              'Sealed Gold path was not opened, and its current content hash was not recomputed.',
              f'Manifest SHA256: {manifest_art["sha256"]}',
              'No training, calibration, TEST prediction, or TEST performance evaluation performed.', '']
    (O / 'V0_3_3B_FINAL_FREEZE_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    verify_lock(False)
    return {'status': 'FINAL_MODEL_FROZEN_READY_FOR_TEST_AUTHORIZATION', 'manifest_sha256': manifest_art['sha256'],
            'lock_path': str(O / 'FINAL_MODEL_FROZEN_CLEAN_V2.lock'), 'FINAL_TEST_AUTHORIZED': False,
            'report_path': str(O / 'V0_3_3B_FINAL_FREEZE_REPORT.md')}


def verify_lock(for_test):
    lock = read_json(O / 'FINAL_MODEL_FROZEN_CLEAN_V2.lock')
    demand(lock['manifest_sha256'] == digest(O / 'FINAL_MODEL_MANIFEST_CLEAN_V2.json'), 'manifest changed')
    for item in lock['watched_artifacts']:
        artifact(item['path'], item['sha256'])
    demand((GOLD.parent.stat().st_mode & 0o777) == 0, 'Gold directory mode changed')
    if for_test:
        authorization = O / 'FINAL_TEST_AUTHORIZATION.json'
        demand(authorization.is_file(), 'FINAL_TEST_AUTHORIZED=false; separate authorization file absent')
        approval = read_json(authorization)
        demand(approval.get('FINAL_TEST_AUTHORIZED') is True and
               approval.get('manifest_sha256') == lock['manifest_sha256'],
               'TEST authorization invalid or bound to another manifest')
    return {'status': 'LOCK_VERIFIED', 'watched_artifact_count': len(lock['watched_artifacts']),
            'FINAL_TEST_AUTHORIZED': lock['FINAL_TEST_AUTHORIZED']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'freeze', 'verify'))
    parser.add_argument('--for-test', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'preflight':
            result = preflight()
            result = {'status': result['status'], 'disk': result['disk'], 'isolation': result['isolation'],
                      'gold_current_content_rehashed': False, 'gold_current_file_existence_independently_verified': False,
                      'selected_checkpoint_hashes': {m: [x['sha256'] for x in rows] for m, rows in result['checkpoint_mapping'].items()}}
        elif args.action == 'freeze':
            result = freeze(preflight())
        else:
            result = verify_lock(args.for_test)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({'status': 'BLOCKED', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
