#!/usr/bin/env bash
set -euo pipefail

: "${TASK2_ROOT:?export TASK2_ROOT first}"
: "${EMBEDDINGS:?export EMBEDDINGS first}"
: "${INDEX:?export INDEX first}"
: "${NODES:?export NODES first}"
: "${EDGES:?export EDGES first}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT_222_V5_6:-$TASK2_ROOT/outputs/reasoning_222_v5_6}"
MAX_QUERIES="${MAX_QUERIES:-0}"
QUERY_TYPES="${QUERY_TYPES:-InstrSem}"
DEVICE="${DEVICE:-auto}"
MIN_HARD_API_SEEDS="${MIN_HARD_API_SEEDS:-1}"
ALLOW_ZERO_HARD_API_SEEDS="${ALLOW_ZERO_HARD_API_SEEDS:-0}"
API_PROVENANCE="${API_PROVENANCE:-}"
CONFIRM_POLICY_FILE="${CONFIRM_POLICY_FILE:-}"
SAMPLE_SPLIT_MANIFEST="${SAMPLE_SPLIT_MANIFEST:-}"
SAMPLE_SPLIT="${SAMPLE_SPLIT:-}"
QUERY_WHITELIST_JSONL="${QUERY_WHITELIST_JSONL:-}"
REPRESENTATION_MANIFEST="${REPRESENTATION_MANIFEST:-}"
RESOLUTION_MANIFEST="${RESOLUTION_MANIFEST:-}"
FORMAL_REPRESENTATION_INPUT="${FORMAL_REPRESENTATION_INPUT:-1}"
GOLD_EVAL_SPLIT="${GOLD_EVAL_SPLIT:-${SAMPLE_SPLIT:-test}}"

for path in "$EMBEDDINGS" "$INDEX" "$NODES" "$EDGES"; do
  test -s "$path" || { echo "ERROR missing or empty: $path"; exit 1; }
done

if [ "$FORMAL_REPRESENTATION_INPUT" = "1" ]; then
  test -s "$REPRESENTATION_MANIFEST" || {
    echo "ERROR formal v5.6 requires REPRESENTATION_MANIFEST"
    echo "Set it to node_embedding_export_manifest.json from 2.2.1.4."
    echo "For a historical-input ablation only: FORMAL_REPRESENTATION_INPUT=0"
    exit 1
  }
  if [ -n "$RESOLUTION_MANIFEST" ]; then
    test -s "$RESOLUTION_MANIFEST" || {
      echo "ERROR missing RESOLUTION_MANIFEST: $RESOLUTION_MANIFEST"
      exit 1
    }
    python3 -u "$SCRIPT_DIR/17_verify_resolved_representation_chain_v6_6.py" \
      --embeddings "$EMBEDDINGS" --index "$INDEX" \
      --resolution-manifest "$RESOLUTION_MANIFEST" \
      --base-export-manifest "$REPRESENTATION_MANIFEST" \
      --report "$TASK2_ROOT/reports/representation_input_v6_6.json"
    echo "Representation input: verified 2.2.1.4 -> 2.2.1.6 resolved chain"
  else
    python3 -u "$SCRIPT_DIR/11_verify_node_embedding_export_v6_3.py" \
      --embeddings "$EMBEDDINGS" \
      --index "$INDEX" \
      --manifest "$REPRESENTATION_MANIFEST" \
      --report "$TASK2_ROOT/reports/representation_input_v6_6.json" \
      --require-formal
    echo "Representation input: verified 2.2.1.4 base export"
  fi
else
  echo "WARNING: using unverified historical representation input (ablation)."
fi

PROVENANCE_ARGS=()
if [ -n "$API_PROVENANCE" ]; then
  test -s "$API_PROVENANCE" || {
    echo "ERROR missing or empty API_PROVENANCE: $API_PROVENANCE"
    exit 1
  }
  PROVENANCE_ARGS+=(--api-provenance "$API_PROVENANCE")
  echo "API provenance mode: compact sidecar $API_PROVENANCE"
else
  echo "API provenance mode: complete nodes file $NODES"
fi

POLICY_ARGS=()
if [ -n "$CONFIRM_POLICY_FILE" ]; then
  test -s "$CONFIRM_POLICY_FILE" || {
    echo "ERROR missing or empty CONFIRM_POLICY_FILE: $CONFIRM_POLICY_FILE"
    exit 1
  }
  POLICY_ARGS+=(--confirm-policy "$CONFIRM_POLICY_FILE")
  echo "Soft confirmation policy: $CONFIRM_POLICY_FILE"
else
  echo "Soft confirmation policy: built-in conservative defaults"
fi

SPLIT_ARGS=()
if [ -n "$SAMPLE_SPLIT_MANIFEST" ] || [ -n "$SAMPLE_SPLIT" ]; then
  if [ -z "$SAMPLE_SPLIT_MANIFEST" ] || [ -z "$SAMPLE_SPLIT" ]; then
    echo "ERROR SAMPLE_SPLIT_MANIFEST and SAMPLE_SPLIT must be set together"
    exit 1
  fi
  test -s "$SAMPLE_SPLIT_MANIFEST" || {
    echo "ERROR missing or empty SAMPLE_SPLIT_MANIFEST: $SAMPLE_SPLIT_MANIFEST"
    exit 1
  }
  SPLIT_ARGS+=(--sample-split-manifest "$SAMPLE_SPLIT_MANIFEST")
  SPLIT_ARGS+=(--sample-split "$SAMPLE_SPLIT")
  echo "Query sample split: $SAMPLE_SPLIT"
fi

QUERY_FILTER_ARGS=()
if [ -n "$QUERY_WHITELIST_JSONL" ]; then
  test -s "$QUERY_WHITELIST_JSONL" || {
    echo "ERROR missing or empty QUERY_WHITELIST_JSONL: $QUERY_WHITELIST_JSONL"
    exit 1
  }
  QUERY_FILTER_ARGS+=(--query-whitelist-jsonl "$QUERY_WHITELIST_JSONL")
  echo "Query whitelist: $QUERY_WHITELIST_JSONL"
fi

for output_file in \
  "$OUT/hypotheses_v5_3.jsonl" \
  "$OUT/validated_v5_3.jsonl" \
  "$OUT/validation_summary_v5_3.json" \
  "$OUT/node_embeddings_reasoned_v5_3.pt" \
  "$OUT/reasoning_final_v5_3.jsonl" \
  "$OUT/reasoning_summary_v5_3.json" \
  "$OUT/evaluation_v5_3.json" \
  "$OUT/reasoning_scope_for_223_v5_3.jsonl"
do
  test ! -e "$output_file" || {
    echo "ERROR refusing to overwrite: $output_file"
    echo "Choose a new OUT_222_V5_6 directory."
    exit 1
  }
done

mkdir -p "$OUT"

python3 -u "$SCRIPT_DIR/08_generate_hypotheses_v5_3.py" \
  --embeddings "$EMBEDDINGS" \
  --index "$INDEX" \
  --nodes "$NODES" \
  "${PROVENANCE_ARGS[@]}" \
  --edges "$EDGES" \
  --query-types "$QUERY_TYPES" \
  "${SPLIT_ARGS[@]}" \
  "${QUERY_FILTER_ARGS[@]}" \
  --top-k-per-type "${TOP_K_PER_TYPE:-3}" \
  --max-total-candidates "${MAX_TOTAL_CANDIDATES:-12}" \
  --max-queries "$MAX_QUERIES" \
  --query-batch-size "${QUERY_BATCH_SIZE:-256}" \
  --target-batch-size "${TARGET_BATCH_SIZE:-8192}" \
  --device "$DEVICE" \
  --out "$OUT/hypotheses_v5_3.jsonl"

python3 -u "$SCRIPT_DIR/09_validate_evidence_v5_3.py" \
  --hypotheses "$OUT/hypotheses_v5_3.jsonl" \
  --nodes "$NODES" \
  "${PROVENANCE_ARGS[@]}" \
  --edges "$EDGES" \
  "${POLICY_ARGS[@]}" \
  --out "$OUT/validated_v5_3.jsonl" \
  --summary-out "$OUT/validation_summary_v5_3.json"

READINESS_ARGS=()
if [ "$ALLOW_ZERO_HARD_API_SEEDS" = "1" ]; then
  READINESS_ARGS+=(--allow-zero)
fi

python3 -u "$SCRIPT_DIR/09b_check_hard_seed_readiness_v5_3.py" \
  --summary "$OUT/validation_summary_v5_3.json" \
  --min-hard-seeds "$MIN_HARD_API_SEEDS" \
  "${READINESS_ARGS[@]}"

RETRIEVAL_ARGS=()
if [ "${ITERATIVE_RETRIEVAL:-1}" = "1" ]; then
  RETRIEVAL_ARGS+=(--iterative-retrieval)
else
  RETRIEVAL_ARGS+=(--no-iterative-retrieval)
fi
if [ "${LEGACY_MONOTONIC_SCORE_UPDATES:-0}" = "1" ]; then
  RETRIEVAL_ARGS+=(--legacy-monotonic-score-updates)
fi
if [ "${LOCK_SOFT_CONFIRMATIONS:-1}" = "1" ]; then
  RETRIEVAL_ARGS+=(--lock-soft-confirmations)
else
  RETRIEVAL_ARGS+=(--allow-soft-confirmation-revisions)
fi
if [ "${ENFORCE_RAW_MONOTONIC:-1}" = "1" ]; then
  RETRIEVAL_ARGS+=(--enforce-raw-monotonic)
else
  RETRIEVAL_ARGS+=(--allow-raw-consistency-decrease)
fi

python3 -u "$SCRIPT_DIR/10_iterative_reasoning_v5_6.py" \
  --validated "$OUT/validated_v5_3.jsonl" \
  --edges "$EDGES" \
  --index "$INDEX" \
  --embeddings "$EMBEDDINGS" \
  --embedding-out "$OUT/node_embeddings_reasoned_v5_3.pt" \
  --max-iterations "${MAX_ITERATIONS:-10}" \
  --global-threshold "${GLOBAL_THRESHOLD:-0.85}" \
  --stable-rounds "${STABLE_ROUNDS:-2}" \
  --representation-step "${REPRESENTATION_STEP:-0.10}" \
  --embedding-rescore-blend "${EMBEDDING_RESCORE_BLEND:-0.25}" \
  --score-feedback-blend "${SCORE_FEEDBACK_BLEND:-0.35}" \
  --max-score-drop-per-round "${MAX_SCORE_DROP_PER_ROUND:-0.08}" \
  --max-score-rise-per-round "${MAX_SCORE_RISE_PER_ROUND:-0.08}" \
  --soft-switch-margin "${SOFT_SWITCH_MARGIN:-0.03}" \
  --soft-release-margin "${SOFT_RELEASE_MARGIN:-0.04}" \
  --soft-release-gap-margin "${SOFT_RELEASE_GAP_MARGIN:-0.03}" \
  --retrieval-interval "${RETRIEVAL_INTERVAL:-1}" \
  --retrieval-top-k-per-type "${RETRIEVAL_TOP_K_PER_TYPE:-5}" \
  --retrieval-max-candidates-per-type "${RETRIEVAL_MAX_CANDIDATES_PER_TYPE:-6}" \
  --retrieval-min-probability "${RETRIEVAL_MIN_PROBABILITY:-0.55}" \
  --retrieval-replacement-margin "${RETRIEVAL_REPLACEMENT_MARGIN:-0.02}" \
  --retrieval-query-batch-size "${RETRIEVAL_QUERY_BATCH_SIZE:-256}" \
  --retrieval-target-batch-size "${RETRIEVAL_TARGET_BATCH_SIZE:-8192}" \
  --retrieval-device "${RETRIEVAL_DEVICE:-$DEVICE}" \
  "${RETRIEVAL_ARGS[@]}" \
  --min-global-delta "${MIN_GLOBAL_DELTA:-0.000001}" \
  --max-state-change-rate-for-stable "${MAX_STATE_CHANGE_RATE_FOR_STABLE:-0.0001}" \
  "${POLICY_ARGS[@]}" \
  --out "$OUT/reasoning_final_v5_3.jsonl" \
  --summary-out "$OUT/reasoning_summary_v5_3.json"

GOLD_ARGS=()
if [ -n "${GOLD_CACHE_LIST:-}" ]; then
  IFS=':' read -r -a GOLD_CACHE_ARRAY <<< "$GOLD_CACHE_LIST"
  for cache in "${GOLD_CACHE_ARRAY[@]}"; do
    if [ -n "$cache" ]; then
      GOLD_ARGS+=(--gold-cache "$cache")
    fi
  done
fi
if [ -n "${GOLD_JSONL_LIST:-}" ]; then
  IFS=':' read -r -a GOLD_JSONL_ARRAY <<< "$GOLD_JSONL_LIST"
  for gold_jsonl in "${GOLD_JSONL_ARRAY[@]}"; do
    if [ -n "$gold_jsonl" ]; then
      test -s "$gold_jsonl" || {
        echo "ERROR missing or empty gold JSONL: $gold_jsonl"
        exit 1
      }
      GOLD_ARGS+=(--gold-jsonl "$gold_jsonl")
    fi
  done
  case "$GOLD_EVAL_SPLIT" in
    train) GOLD_PURPOSE=train ;;
    validation) GOLD_PURPOSE=calibration ;;
    test) GOLD_PURPOSE=final_test ;;
    *) echo "ERROR GOLD_EVAL_SPLIT must be train, validation or test"; exit 1 ;;
  esac
  SPLIT_GUARD_ARGS=()
  for gold_jsonl in "${GOLD_JSONL_ARRAY[@]}"; do
    [ -z "$gold_jsonl" ] && continue
    SPLIT_GUARD_ARGS+=(--gold "$gold_jsonl")
  done
  python3 -u "$SCRIPT_DIR/20_validate_gold_split_contract_v6_6.py" \
    "${SPLIT_GUARD_ARGS[@]}" \
    --expected-split "$GOLD_EVAL_SPLIT" \
    --purpose "$GOLD_PURPOSE" \
    --out "$OUT/gold_split_contract_v6_6.json"
fi

python3 -u "$SCRIPT_DIR/11_evaluate_reasoning_v5_5.py" \
  --input "$OUT/reasoning_final_v5_3.jsonl" \
  --summary "$OUT/reasoning_summary_v5_3.json" \
  "${POLICY_ARGS[@]}" \
  "${GOLD_ARGS[@]}" \
  --out "$OUT/evaluation_v5_3.json"

python3 -u "$SCRIPT_DIR/12_export_for_223_v5_3.py" \
  --input "$OUT/reasoning_final_v5_3.jsonl" \
  --out "$OUT/reasoning_scope_for_223_v5_3.jsonl"

VERIFY_MIN_HARD="$MIN_HARD_API_SEEDS"
if [ "$ALLOW_ZERO_HARD_API_SEEDS" = "1" ]; then
  VERIFY_MIN_HARD=0
fi
VERIFY_ARGS=(--out "$OUT" --min-hard-seeds "$VERIFY_MIN_HARD")
if [ -n "$API_PROVENANCE" ]; then
  VERIFY_ARGS+=(--require-sidecar)
fi
if [ "${EXPECTED_RECORDS:-0}" -gt 0 ]; then
  VERIFY_ARGS+=(--expected-records "$EXPECTED_RECORDS")
fi
if [ "${REQUIRE_RETRIEVED_CANDIDATE:-0}" = "1" ]; then
  VERIFY_ARGS+=(--require-retrieved-candidate)
fi
if [ "${REQUIRE_SCORE_DECREASE:-0}" = "1" ]; then
  VERIFY_ARGS+=(--require-score-decrease)
fi
if [ "${ITERATIVE_RETRIEVAL:-1}" != "1" ] \
  || [ "${LEGACY_MONOTONIC_SCORE_UPDATES:-0}" = "1" ] \
  || [ "${LOCK_SOFT_CONFIRMATIONS:-1}" != "1" ] \
  || [ "${ENFORCE_RAW_MONOTONIC:-1}" != "1" ]; then
  VERIFY_ARGS+=(--allow-closed-loop-ablation)
fi

python3 -u "$SCRIPT_DIR/13_verify_222_v5_6_run.py" \
  "${VERIFY_ARGS[@]}"

echo "DONE: $OUT"
