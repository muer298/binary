#!/usr/bin/env bash
# 2.2.1.1-2.2.1.3：把任意符合研究内容一契约的 semantic_dict JSON
# 转换为 USG、锚点邻域和分层节点特征。所有路径由环境变量提供。
set -euo pipefail

: "${SEMANTIC_JSON_ROOT:?export SEMANTIC_JSON_ROOT first}"
: "${WORK_ROOT:?export WORK_ROOT first}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TAG="${RUN_TAG:-final}"

GRAPH_DIR="${GRAPH_DIR:-$WORK_ROOT/graphs}"
ANCHOR_DIR="${ANCHOR_DIR:-$WORK_ROOT/anchors}"
FEATURE_DIR="${FEATURE_DIR:-$WORK_ROOT/features}"
REPORT_DIR="${REPORT_DIR:-$WORK_ROOT/reports}"

USG_INDEX="${USG_INDEX:-$GRAPH_DIR/usg_index_${RUN_TAG}.jsonl}"
NODES="${NODES:-$GRAPH_DIR/usg_nodes_${RUN_TAG}.jsonl}"
EDGES="${EDGES:-$GRAPH_DIR/usg_edges_${RUN_TAG}.jsonl}"
ANCHORS="${ANCHORS:-$ANCHOR_DIR/anchors_${RUN_TAG}.jsonl}"
ANCHORS_BALANCED="${ANCHORS_BALANCED:-$ANCHOR_DIR/anchors_balanced_${RUN_TAG}.jsonl}"
ANCHORS_FINAL="${ANCHORS_FINAL:-$ANCHOR_DIR/anchors_final_${RUN_TAG}.jsonl}"
NEIGH_BASE="${NEIGH_BASE:-$ANCHOR_DIR/neighborhoods_base_${RUN_TAG}.jsonl}"
NEIGH_CALLCTX="${NEIGH_CALLCTX:-$ANCHOR_DIR/neighborhoods_callctx_${RUN_TAG}.jsonl}"
NEIGH_PARAMFLOW="${NEIGH_PARAMFLOW:-$ANCHOR_DIR/neighborhoods_paramflow_${RUN_TAG}.jsonl}"
NEIGHBORHOODS="${NEIGHBORHOODS:-$ANCHOR_DIR/neighborhoods_final_${RUN_TAG}.jsonl}"
SELECTED_NODES="${SELECTED_NODES:-$GRAPH_DIR/usg_nodes_selected_${RUN_TAG}.jsonl}"
ENRICHED_NODES="${ENRICHED_NODES:-$GRAPH_DIR/usg_nodes_enriched_${RUN_TAG}.jsonl}"
FEATURES="${FEATURES:-$FEATURE_DIR/node_features_${RUN_TAG}.pt}"
FEATURE_VOCAB="${FEATURE_VOCAB:-$FEATURE_DIR/feature_vocab_${RUN_TAG}.json}"
FEATURE_JSONL="${FEATURE_JSONL:-$FEATURE_DIR/node_features_${RUN_TAG}.jsonl}"

mkdir -p "$GRAPH_DIR" "$ANCHOR_DIR" "$FEATURE_DIR" "$REPORT_DIR"

for output_path in \
  "$USG_INDEX" "$NODES" "$EDGES" "$ANCHORS" "$ANCHORS_BALANCED" \
  "$ANCHORS_FINAL" "$NEIGH_BASE" "$NEIGH_CALLCTX" "$NEIGH_PARAMFLOW" \
  "$NEIGHBORHOODS" "$SELECTED_NODES" "$ENRICHED_NODES" "$FEATURES" \
  "$FEATURE_VOCAB" "$FEATURE_JSONL"
do
  test ! -e "$output_path" || {
    echo "ERROR: refusing to overwrite $output_path" >&2
    echo "Choose a new WORK_ROOT or RUN_TAG." >&2
    exit 1
  }
done

"$PYTHON_BIN" -u "$SCRIPT_DIR/01_build_usg_index.py" \
  --json-root "$SEMANTIC_JSON_ROOT" \
  --out "$USG_INDEX" \
  --stats "$REPORT_DIR/usg_index_${RUN_TAG}.json"

"$PYTHON_BIN" -u "$SCRIPT_DIR/02_extract_nodes_edges.py" \
  --index "$USG_INDEX" \
  --nodes-out "$NODES" \
  --edges-out "$EDGES" \
  --stats "$REPORT_DIR/graph_schema_${RUN_TAG}.json" \
  --fail-on-error

ANCHOR_RESOLUTION_ARGS=()
if [ "${REQUIRE_RESOLVED_ANCHORS:-1}" = "1" ]; then
  ANCHOR_RESOLUTION_ARGS+=(--resolved-only)
fi
"$PYTHON_BIN" -u "$SCRIPT_DIR/04_build_anchors.py" \
  --nodes "$NODES" --edges "$EDGES" \
  --out "$ANCHORS" --stats "$REPORT_DIR/anchors_${RUN_TAG}.json" \
  --min-strength "${MIN_ANCHOR_STRENGTH:-0.0}" \
  "${ANCHOR_RESOLUTION_ARGS[@]}"

ANCHOR_LIMITS="${ANCHOR_LIMITS:-fixed_offset=10000,deref_chain=10000,def_use=10000,api_call=10000}"
"$PYTHON_BIN" -u "$SCRIPT_DIR/08_sample_anchors_balanced.py" \
  --anchors "$ANCHORS" --out "$ANCHORS_BALANCED" \
  --stats "$REPORT_DIR/anchors_balanced_${RUN_TAG}.json" \
  --limits "$ANCHOR_LIMITS" --seed "${RANDOM_SEED:-1337}"

"$PYTHON_BIN" -u "$SCRIPT_DIR/09_sample_anchors_api_filtered.py" \
  --anchors "$ANCHORS_BALANCED" --out "$ANCHORS_FINAL" \
  --stats "$REPORT_DIR/anchors_final_${RUN_TAG}.json" \
  --limits "$ANCHOR_LIMITS" --seed "${RANDOM_SEED:-1337}"

"$PYTHON_BIN" -u "$SCRIPT_DIR/05_sample_anchor_neighborhoods.py" \
  --nodes "$NODES" --edges "$EDGES" --anchors "$ANCHORS_FINAL" \
  --out "$NEIGH_BASE" --stats "$REPORT_DIR/neighborhoods_base_${RUN_TAG}.json" \
  --max-nodes-per-side "${MAX_NODES_PER_SIDE:-120}"

"$PYTHON_BIN" -u "$SCRIPT_DIR/05b_add_api_callctx_neighborhoods.py" \
  --nodes "$NODES" --in-neigh "$NEIGH_BASE" --out-neigh "$NEIGH_CALLCTX" \
  --stats "$REPORT_DIR/neighborhoods_callctx_${RUN_TAG}.json" \
  --pre-n "${CALLCTX_PRE:-15}" --post-n "${CALLCTX_POST:-8}" \
  --max-low-nodes "${MAX_LOW_NODES:-120}"

"$PYTHON_BIN" -u "$SCRIPT_DIR/05c_add_api_paramflow_neighborhoods.py" \
  --nodes "$NODES" --edges "$EDGES" \
  --neighborhoods-in "$NEIGH_CALLCTX" --neighborhoods-out "$NEIGH_PARAMFLOW" \
  --stats "$REPORT_DIR/neighborhoods_paramflow_${RUN_TAG}.json" \
  --arch "${TARGET_ARCH:-both}" --pre-scan "${PARAMFLOW_PRE:-15}" \
  --post-scan "${PARAMFLOW_POST:-8}" --max-hops "${PARAMFLOW_HOPS:-2}" \
  --max-flow-nodes "${MAX_FLOW_NODES:-40}" --max-low-nodes "${MAX_LOW_NODES:-120}"

"$PYTHON_BIN" -u "$SCRIPT_DIR/05e_repair_api_direct_only.py" \
  --nodes "$NODES" --edges "$EDGES" \
  --neighborhoods-in "$NEIGH_PARAMFLOW" --neighborhoods-out "$NEIGHBORHOODS" \
  --stats "$REPORT_DIR/neighborhoods_final_${RUN_TAG}.json" \
  --max-call-candidates "${MAX_CALL_CANDIDATES:-8}" \
  --max-flow-nodes "${MAX_FLOW_NODES:-40}" --max-low-nodes "${MAX_LOW_NODES:-120}"

"$PYTHON_BIN" -u "$SCRIPT_DIR/10_select_neighborhood_nodes.py" \
  --nodes "$NODES" --neighborhoods "$NEIGHBORHOODS" \
  --out "$SELECTED_NODES" --stats "$REPORT_DIR/selected_nodes_${RUN_TAG}.json"

"$PYTHON_BIN" -u "$SCRIPT_DIR/11_enrich_nodes_api_aware.py" \
  --nodes "$SELECTED_NODES" --out "$ENRICHED_NODES" \
  --stats "$REPORT_DIR/api_enrichment_${RUN_TAG}.json"

"$PYTHON_BIN" -u "$SCRIPT_DIR/06_featurize_objects_hier.py" \
  --nodes "$ENRICHED_NODES" --neighborhoods "$NEIGHBORHOODS" \
  --out-pt "$FEATURES" --vocab-out "$FEATURE_VOCAB" \
  --node-feature-jsonl "$FEATURE_JSONL" \
  --hash-dim-low "${HASH_DIM_LOW:-192}" --hash-dim-mid "${HASH_DIM_MID:-128}" \
  --hash-dim-high "${HASH_DIM_HIGH:-192}" --hash-dim-generic "${HASH_DIM_GENERIC:-64}"

if [ "${RUN_EVIDENCE_AUDIT:-1}" = "1" ]; then
  "$PYTHON_BIN" -u "$SCRIPT_DIR/08_audit_evidence_groups.py" \
    --neighborhoods "$NEIGHBORHOODS" \
    --report "$REPORT_DIR/evidence_groups_${RUN_TAG}.json" \
    --max-row-fallback-rate "${MAX_ROW_FALLBACK_RATE:-1.0}"
fi

echo "2.2.1 preparation: PASSED"
echo "NODES=$ENRICHED_NODES"
echo "EDGES=$EDGES"
echo "NEIGHBORHOODS=$NEIGHBORHOODS"
echo "FEATURES=$FEATURES"
