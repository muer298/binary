#!/usr/bin/env bash
# 重新生成锚点、邻域和节点特征；特征布局严格复用冻结 ReSym 合同。
set -euo pipefail

: "${PACKAGE_ROOT:?source external.env first}"
: "${EXT_DIR:?source external.env first}"
: "${RAW_NODES:?source external.env first}"
: "${EDGES:?source external.env first}"
: "${FROZEN_FEATURE_VOCAB:?source external.env first}"
: "${NODES:?source external.env first}"
: "${NODE_FEATURES:?source external.env first}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$ANCHOR_DIR" "$GRAPH_DIR" "$FEATURE_DIR" "$REPORT_DIR"
anchors="$ANCHOR_DIR/anchors_${RUN_TAG}.jsonl"
anchors_balanced="$ANCHOR_DIR/anchors_balanced_${RUN_TAG}.jsonl"
anchors_final="$ANCHOR_DIR/anchors_final_${RUN_TAG}.jsonl"
neigh_base="$ANCHOR_DIR/neighborhoods_base_${RUN_TAG}.jsonl"
neigh_callctx="$ANCHOR_DIR/neighborhoods_callctx_${RUN_TAG}.jsonl"
neigh_paramflow="$ANCHOR_DIR/neighborhoods_paramflow_${RUN_TAG}.jsonl"
feature_jsonl="$FEATURE_DIR/node_features_${RUN_TAG}.jsonl"
layout_report="$REPORT_DIR/layout_evidence_enrichment_${RUN_TAG}.json"
feature_contract_report="$REPORT_DIR/frozen_feature_contract_${RUN_TAG}.json"

for output in \
  "$anchors" "$anchors_balanced" "$anchors_final" "$neigh_base" \
  "$neigh_callctx" "$neigh_paramflow" "$NEIGHBORHOODS" \
  "$SELECTED_NODES" "$API_ENRICHED_NODES" "$NODES" \
  "$NODE_FEATURES" "$EXTERNAL_FEATURE_VOCAB" "$feature_jsonl" \
  "$layout_report" "$feature_contract_report"
do
  test ! -e "$output" || { echo "ERROR: refusing to overwrite $output" >&2; exit 1; }
done

read -r hash_low hash_mid hash_high hash_generic < <(
  "$PYTHON_BIN" - "$FROZEN_FEATURE_VOCAB" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
dims = value.get("hash_dims") or {}
required = [dims.get(name) for name in ("low", "mid", "high", "generic")]
if any(item is None for item in required):
    raise SystemExit("ERROR: frozen feature vocab lacks hash_dims")
print(*[int(item) for item in required])
PY
)

"$PYTHON_BIN" -u "$PACKAGE_ROOT/04_build_anchors.py" \
  --nodes "$RAW_NODES" --edges "$EDGES" --out "$anchors" \
  --stats "$REPORT_DIR/anchors_${RUN_TAG}.json" --min-strength 0 --resolved-only
limits="${ANCHOR_LIMITS:-fixed_offset=10000,deref_chain=10000,def_use=10000,api_call=10000}"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/08_sample_anchors_balanced.py" \
  --anchors "$anchors" --out "$anchors_balanced" \
  --stats "$REPORT_DIR/anchors_balanced_${RUN_TAG}.json" \
  --limits "$limits" --seed 1337
"$PYTHON_BIN" -u "$PACKAGE_ROOT/09_sample_anchors_api_filtered.py" \
  --anchors "$anchors_balanced" --out "$anchors_final" \
  --stats "$REPORT_DIR/anchors_final_${RUN_TAG}.json" \
  --limits "$limits" --seed 1337
"$PYTHON_BIN" -u "$PACKAGE_ROOT/05_sample_anchor_neighborhoods.py" \
  --nodes "$RAW_NODES" --edges "$EDGES" --anchors "$anchors_final" \
  --out "$neigh_base" --stats "$REPORT_DIR/neighborhoods_base_${RUN_TAG}.json" \
  --max-nodes-per-side "${MAX_NODES_PER_SIDE:-120}"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/05b_add_api_callctx_neighborhoods.py" \
  --nodes "$RAW_NODES" --in-neigh "$neigh_base" --out-neigh "$neigh_callctx" \
  --stats "$REPORT_DIR/neighborhoods_callctx_${RUN_TAG}.json" \
  --pre-n "${CALLCTX_PRE:-15}" --post-n "${CALLCTX_POST:-8}" \
  --max-low-nodes "${MAX_LOW_NODES:-120}"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/05c_add_api_paramflow_neighborhoods.py" \
  --nodes "$RAW_NODES" --edges "$EDGES" \
  --neighborhoods-in "$neigh_callctx" --neighborhoods-out "$neigh_paramflow" \
  --stats "$REPORT_DIR/neighborhoods_paramflow_${RUN_TAG}.json" \
  --arch x86_64 --pre-scan "${PARAMFLOW_PRE:-15}" --post-scan "${PARAMFLOW_POST:-8}" \
  --max-hops "${PARAMFLOW_HOPS:-2}" --max-flow-nodes "${MAX_FLOW_NODES:-40}" \
  --max-low-nodes "${MAX_LOW_NODES:-120}"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/05e_repair_api_direct_only.py" \
  --nodes "$RAW_NODES" --edges "$EDGES" \
  --neighborhoods-in "$neigh_paramflow" --neighborhoods-out "$NEIGHBORHOODS" \
  --stats "$REPORT_DIR/neighborhoods_final_${RUN_TAG}.json" \
  --max-call-candidates "${MAX_CALL_CANDIDATES:-8}" \
  --max-flow-nodes "${MAX_FLOW_NODES:-40}" --max-low-nodes "${MAX_LOW_NODES:-120}"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/10_select_neighborhood_nodes.py" \
  --nodes "$RAW_NODES" --neighborhoods "$NEIGHBORHOODS" \
  --out "$SELECTED_NODES" --stats "$REPORT_DIR/selected_nodes_${RUN_TAG}.json"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/11_enrich_nodes_api_aware.py" \
  --nodes "$SELECTED_NODES" --out "$API_ENRICHED_NODES" \
  --stats "$REPORT_DIR/api_enrichment_${RUN_TAG}.json"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/26_enrich_layout_evidence_v6_7_4.py" \
  --nodes-in "$API_ENRICHED_NODES" --nodes-out "$NODES" \
  --report-out "$layout_report" --fail-on-source-error
"$PYTHON_BIN" -u "$PACKAGE_ROOT/06_featurize_objects_hier.py" \
  --nodes "$NODES" --neighborhoods "$NEIGHBORHOODS" \
  --out-pt "$NODE_FEATURES" --vocab-out "$EXTERNAL_FEATURE_VOCAB" \
  --node-feature-jsonl "$feature_jsonl" \
  --hash-dim-low "$hash_low" --hash-dim-mid "$hash_mid" \
  --hash-dim-high "$hash_high" --hash-dim-generic "$hash_generic"
"$PYTHON_BIN" -u "$EXT_DIR/verify_feature_contract_v1.py" \
  --frozen-vocab "$FROZEN_FEATURE_VOCAB" \
  --external-vocab "$EXTERNAL_FEATURE_VOCAB" \
  --report "$feature_contract_report"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/08_audit_evidence_groups.py" \
  --neighborhoods "$NEIGHBORHOODS" \
  --report "$REPORT_DIR/evidence_groups_${RUN_TAG}.json" \
  --max-row-fallback-rate 1.0

echo "training invoked = False"
echo "PARTIAL DIAGNOSTIC NEIGHBORHOODS AND FROZEN-SCHEMA FEATURES: PASSED"
