#!/usr/bin/env bash
set -euo pipefail

: "${TASK2_ROOT:?export TASK2_ROOT first}"
: "${SCOPE_222_V5_6:?export SCOPE_222_V5_6 first}"
: "${NODES:?export NODES first}"
: "${EDGES:?export EDGES first}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT_223_V6_6:-$TASK2_ROOT/outputs/reasoning_223_v6_6}"
ABLATION="${ABLATION:-full}"
GOLD_EVAL_SPLIT="${GOLD_EVAL_SPLIT:-test}"

for path in "$SCOPE_222_V5_6" "$NODES" "$EDGES"; do
  test -s "$path" || { echo "ERROR missing or empty: $path"; exit 1; }
done

for path in \
  "$OUT/four_module_summary_v6_6.json" \
  "$OUT/collaborative_result_v6_6.jsonl" \
  "$OUT/collaboration_summary_v6_6.json" \
  "$OUT/evaluation_v6_6.json"
do
  test ! -e "$path" || { echo "ERROR refusing to overwrite: $path"; exit 1; }
done
mkdir -p "$OUT/modules"

HEAD_ARGS=()
if [ -n "${MODULE_HEAD_CHECKPOINT:-}" ]; then
  test -s "$MODULE_HEAD_CHECKPOINT" || {
    echo "ERROR missing module head checkpoint: $MODULE_HEAD_CHECKPOINT"
    exit 1
  }
  HEAD_ARGS+=(--head-checkpoint "$MODULE_HEAD_CHECKPOINT")
fi

LAYOUT_ARGS=()
if [ -n "${LAYOUT_RERANK_CONFIG:-}" ]; then
  test -s "$LAYOUT_RERANK_CONFIG" || {
    echo "ERROR missing LayoutSem reranker config: $LAYOUT_RERANK_CONFIG"
    exit 1
  }
  LAYOUT_ARGS+=(--layout-config "$LAYOUT_RERANK_CONFIG")

  if [ -z "${MODULE_WEIGHTS:-}" ] && \
     [ "${LAYOUT_USE_RECOMMENDED_MODULE_WEIGHT:-1}" = "1" ]; then
    LAYOUT_SELECTED_WEIGHT="$(
      python3 - "$LAYOUT_RERANK_CONFIG" <<'PY'
import json
import sys

obj = json.load(open(sys.argv[1], encoding="utf-8"))
if obj.get("test_used_for_selection") is not False:
    raise SystemExit("ERROR: LayoutSem config used Test")
value = float(obj.get("recommended_module_weight", 0.25))
if not 0.0 <= value <= 1.0:
    raise SystemExit("ERROR: invalid recommended_module_weight")
print(value)
PY
    )"
    BASE_MODULE_WEIGHT="${MODULE_WEIGHT:-0.25}"
    MODULE_WEIGHTS="APISem=$BASE_MODULE_WEIGHT,TypeSem=$BASE_MODULE_WEIGHT,LayoutSem=$LAYOUT_SELECTED_WEIGHT,InterfaceSem=$BASE_MODULE_WEIGHT"
    export MODULE_WEIGHTS
  fi
elif [ "$GOLD_EVAL_SPLIT" = "test" ] && \
     [ "${REQUIRE_LAYOUT_CONFIG_FOR_TEST:-1}" = "1" ]; then
  echo "ERROR: final Test requires frozen LAYOUT_RERANK_CONFIG"
  exit 1
fi

python3 -u "$SCRIPT_DIR/19_run_four_module_inference_v6_6.py" \
  --reasoning "$SCOPE_222_V5_6" \
  --nodes "$NODES" \
  --edges "$EDGES" \
  "${HEAD_ARGS[@]}" \
  "${LAYOUT_ARGS[@]}" \
  --out-dir "$OUT/modules" \
  --summary-out "$OUT/four_module_summary_v6_6.json"

python3 - "$OUT/four_module_summary_v6_6.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("loaded_node_metadata", 0) <= 0:
    raise SystemExit("ERROR: no target node metadata loaded; NODES is incompatible")
if report.get("loaded_incident_edge_ids", 0) <= 0:
    raise SystemExit("ERROR: no incident graph context loaded; EDGES is incompatible")
print("specialised module graph context: PASSED")
PY

python3 - "$OUT/four_module_summary_v6_6.json" \
  "${MIN_LAYOUT_PAIRWISE_GROUP_COVERAGE:-0.10}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
minimum = float(sys.argv[2])
layout = (report.get("module_counts") or {}).get("LayoutInfer") or {}
groups = int(layout.get("records") or 0)
usable = int(layout.get("layout_query_groups_with_usable_pair") or 0)
coverage = usable / groups if groups else 0.0
print("LayoutSem pairwise-evidence groups =", usable, "/", groups)
print("LayoutSem pairwise-evidence coverage = %.2f%%" % (100.0 * coverage))
if groups <= 0:
    raise SystemExit("ERROR: no LayoutSem groups were produced")
if coverage + 1e-12 < minimum:
    raise SystemExit(
        "ERROR: LayoutSem pairwise evidence coverage %.4f < required %.4f; "
        "inspect NODES/query offset export before running full" % (coverage, minimum)
    )
print("LayoutSem v6.7.3 evidence preflight: PASSED")
PY

MODULE_ARGS=()
for module in apiinfer typeinfer layoutinfer interfaceinfer; do
  path="$OUT/modules/${module}_v6_6.jsonl"
  test -s "$path" || { echo "ERROR module produced no rows: $path"; exit 1; }
  MODULE_ARGS+=(--module-result "$path")
done

COLLAB_ARGS=()
[ -n "${MODULE_WEIGHTS:-}" ] && COLLAB_ARGS+=(
  --module-weights "$MODULE_WEIGHTS"
)
[ -n "${DIRECTION_WEIGHT_OVERRIDES:-}" ] && COLLAB_ARGS+=(
  --direction-weight-overrides "$DIRECTION_WEIGHT_OVERRIDES"
)
POLICY_ARGS=()
if [ -n "${CONFIRM_POLICY_FILE:-}" ]; then
  test -s "$CONFIRM_POLICY_FILE" || {
    echo "ERROR missing confirmation policy: $CONFIRM_POLICY_FILE"
    exit 1
  }
  POLICY_ARGS+=(--confirm-policy "$CONFIRM_POLICY_FILE")
fi

python3 -u "$SCRIPT_DIR/20_iterative_module_collaboration_v6_6.py" \
  --reasoning "$SCOPE_222_V5_6" \
  --edges "$EDGES" \
  "${MODULE_ARGS[@]}" \
  "${POLICY_ARGS[@]}" \
  --max-iterations "${MAX_MODULE_ITERATIONS:-10}" \
  --stable-rounds "${MODULE_STABLE_ROUNDS:-2}" \
  --global-threshold "${MODULE_GLOBAL_THRESHOLD:-0.85}" \
  --module-weight "${MODULE_WEIGHT:-0.25}" \
  "${COLLAB_ARGS[@]}" \
  --propagation-step "${MODULE_PROPAGATION_STEP:-0.25}" \
  --max-score-change-per-round "${MODULE_MAX_SCORE_CHANGE_PER_ROUND:-0.08}" \
  --ablation "$ABLATION" \
  --out "$OUT/collaborative_result_v6_6.jsonl" \
  --summary-out "$OUT/collaboration_summary_v6_6.json"

GOLD_ARGS=()
if [ -n "${GOLD_JSONL_LIST:-}" ]; then
  IFS=':' read -r -a GOLD_ARRAY <<< "$GOLD_JSONL_LIST"
  for path in "${GOLD_ARRAY[@]}"; do
    [ -z "$path" ] && continue
    test -s "$path" || { echo "ERROR missing gold: $path"; exit 1; }
    GOLD_ARGS+=(--gold-jsonl "$path")
  done
  case "$GOLD_EVAL_SPLIT" in
    train) GOLD_PURPOSE=train ;;
    validation) GOLD_PURPOSE=calibration ;;
    test) GOLD_PURPOSE=final_test ;;
    *) echo "ERROR GOLD_EVAL_SPLIT must be train, validation or test"; exit 1 ;;
  esac
  SPLIT_GUARD_ARGS=()
  for path in "${GOLD_ARRAY[@]}"; do
    [ -z "$path" ] && continue
    SPLIT_GUARD_ARGS+=(--gold "$path")
  done
  python3 -u "$SCRIPT_DIR/20_validate_gold_split_contract_v6_6.py" \
    "${SPLIT_GUARD_ARGS[@]}" \
    --expected-split "$GOLD_EVAL_SPLIT" \
    --purpose "$GOLD_PURPOSE" \
    --out "$OUT/gold_split_contract_v6_6.json"
fi
if [ -n "${API_PROVENANCE:-}" ]; then
  test -s "$API_PROVENANCE" || { echo "ERROR missing API_PROVENANCE"; exit 1; }
  GOLD_ARGS+=(--api-provenance-sidecar "$API_PROVENANCE")
fi

python3 -u "$SCRIPT_DIR/21_evaluate_223_v6_6.py" \
  --baseline "$SCOPE_222_V5_6" \
  --input "$OUT/collaborative_result_v6_6.jsonl" \
  --collaboration-summary "$OUT/collaboration_summary_v6_6.json" \
  --ablation "$ABLATION" \
  "${GOLD_ARGS[@]}" \
  --out "$OUT/evaluation_v6_6.json"

VERIFY_ARGS=()
[ "${EXPECTED_RECORDS:-0}" -gt 0 ] && VERIFY_ARGS+=(--expected-records "$EXPECTED_RECORDS")
[ "${MIN_GOLD_MATCHES:-0}" -gt 0 ] && VERIFY_ARGS+=(--min-gold-matches "$MIN_GOLD_MATCHES")
[ "${REQUIRE_GOLD_METRICS:-0}" = "1" ] && VERIFY_ARGS+=(--require-gold-metrics)
[ "${REQUIRE_ALL_MODULE_GOLD:-0}" = "1" ] && VERIFY_ARGS+=(--require-all-module-gold)
[ "${REQUIRE_RANKING_NONDEGRADING:-0}" = "1" ] && VERIFY_ARGS+=(
  --require-ranking-nondegrading
)
[ "${REQUIRE_MODULE_RANKING_NONDEGRADING:-0}" = "1" ] && VERIFY_ARGS+=(
  --require-module-ranking-nondegrading
)
VERIFY_ARGS+=(--ranking-tolerance "${RANKING_TOLERANCE:-0}")
VERIFY_ARGS+=(
  --module-ranking-tolerance "${MODULE_RANKING_TOLERANCE:-0}"
)
VERIFY_ARGS+=(
  --min-evaluable-confirmations "${MIN_EVALUABLE_CONFIRMATIONS:-20}"
)
[ -n "${CONFIRMATION_PRECISION_FLOORS:-}" ] && VERIFY_ARGS+=(
  --confirmation-precision-floors "$CONFIRMATION_PRECISION_FLOORS"
)
[ -n "${CONFIRMATION_COVERAGE_FLOORS:-}" ] && VERIFY_ARGS+=(
  --confirmation-coverage-floors "$CONFIRMATION_COVERAGE_FLOORS"
)

python3 -u "$SCRIPT_DIR/22_verify_223_v6_6.py" \
  --evaluation "$OUT/evaluation_v6_6.json" \
  --summary "$OUT/collaboration_summary_v6_6.json" \
  "${VERIFY_ARGS[@]}"

echo "DONE: $OUT"
