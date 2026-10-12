#!/usr/bin/env bash
# 2.2.1.4：统一 Stage5 训练、Validation-best 冻结、正式节点嵌入导出。
set -euo pipefail

: "${WORK_ROOT:?export WORK_ROOT first}"
: "${SPLIT_MANIFEST:?export SPLIT_MANIFEST first}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TAG="${RUN_TAG:-final}"
MODE="${MODE:-smoke}"

FEATURES="${FEATURES:-$WORK_ROOT/features/node_features_${RUN_TAG}.pt}"
NEIGHBORHOODS="${NEIGHBORHOODS:-$WORK_ROOT/anchors/neighborhoods_final_${RUN_TAG}.jsonl}"
NODES="${NODES:-$WORK_ROOT/graphs/usg_nodes_enriched_${RUN_TAG}.jsonl}"
EDGES="${EDGES:-$WORK_ROOT/graphs/usg_edges_${RUN_TAG}.jsonl}"
STAGE5_OUT="${STAGE5_OUT:-$WORK_ROOT/outputs/stage5_${RUN_TAG}_${MODE}}"
EMBED_OUT="${EMBED_OUT:-$WORK_ROOT/outputs/embedding_${RUN_TAG}_${MODE}}"
REPORT_DIR="${REPORT_DIR:-$WORK_ROOT/reports}"

case "$MODE" in
  smoke)
    EPOCHS="${EPOCHS:-2}"
    MAX_ANCHORS="${MAX_ANCHORS:-5000}"
    MAX_EXPORT_NODES="${MAX_EXPORT_NODES:-5000}"
    ;;
  full|formal)
    EPOCHS="${EPOCHS:-30}"
    MAX_ANCHORS="${MAX_ANCHORS:-0}"
    MAX_EXPORT_NODES="${MAX_EXPORT_NODES:-0}"
    ;;
  *) echo "ERROR: MODE must be smoke, full, or formal" >&2; exit 2 ;;
esac

for input_path in "$FEATURES" "$NEIGHBORHOODS" "$NODES" "$EDGES" "$SPLIT_MANIFEST"; do
  test -s "$input_path" || { echo "ERROR: missing or empty $input_path" >&2; exit 1; }
done
for output_dir in "$STAGE5_OUT" "$EMBED_OUT"; do
  test ! -e "$output_dir" || { echo "ERROR: refusing to reuse $output_dir" >&2; exit 1; }
done
mkdir -p "$REPORT_DIR"

if [ -n "${CUDA_DEV:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEV"
fi
DEVICE="${DEVICE:-auto}"

"$PYTHON_BIN" -u "$SCRIPT_DIR/08_audit_evidence_groups.py" \
  --neighborhoods "$NEIGHBORHOODS" \
  --report "$REPORT_DIR/evidence_groups_pretrain_${RUN_TAG}_${MODE}.json" \
  --max-row-fallback-rate "${MAX_ROW_FALLBACK_RATE:-1.0}"

TRAIN_ARGS=(
  --node-features "$FEATURES"
  --neighborhoods "$NEIGHBORHOODS"
  --nodes "$NODES"
  --edges "$EDGES"
  --out-dir "$STAGE5_OUT"
  --split-mode sample
  --split-manifest "$SPLIT_MANIFEST"
  --require-split-manifest
  --missing-split-policy error
  --epochs "$EPOCHS"
  --batch-size "${BATCH_SIZE:-24}"
  --batching-mode target_grouped
  --max-target-group-size "${MAX_TARGET_GROUP_SIZE:-0}"
  --max-graph-edges "${MAX_GRAPH_EDGES:-500000}"
  --hidden-dim "${HIDDEN_DIM:-128}"
  --embed-dim "${EMBED_DIM:-64}"
  --etype-dim "${ETYPE_DIM:-16}"
  --rel-dim "${REL_DIM:-64}"
  --nhead "${NHEAD:-4}"
  --transformer-layers "${TRANSFORMER_LAYERS:-1}"
  --gnn-layers "${GNN_LAYERS:-2}"
  --lr "${LR:-1e-3}"
  --lambda-evidence "${LAMBDA_EVIDENCE:-1.0}"
  --lambda-constraint-margin "${LAMBDA_CONSTRAINT:-0.08}"
  --lambda-constraint-margin-start "${LAMBDA_CONSTRAINT_START:-0.0}"
  --constraint-warmup-epochs "${CONSTRAINT_WARMUP_EPOCHS:-2}"
  --constraint-ramp-epochs "${CONSTRAINT_RAMP_EPOCHS:-10}"
  --lambda-structure-recon "${LAMBDA_STRUCTURE:-0.04}"
  --row-multi-positive-weight "${ROW_MULTI_POSITIVE_WEIGHT:-1.0}"
  --target-attention-weight "${TARGET_ATTENTION_WEIGHT:-1.0}"
  --device "$DEVICE"
)
if [ "$MAX_ANCHORS" -gt 0 ]; then TRAIN_ARGS+=(--max-anchors "$MAX_ANCHORS"); fi
"$PYTHON_BIN" -u "$SCRIPT_DIR/07_train_alignment_stage5_full.py" "${TRAIN_ARGS[@]}"

"$PYTHON_BIN" -u "$SCRIPT_DIR/09_verify_stage3_split_isolation_v6_2.py" \
  --metrics "$STAGE5_OUT/alignment_metrics.json" \
  --report "$STAGE5_OUT/split_verification.json" --require-frozen-split

EXPORT_ARGS=(
  --checkpoint "$STAGE5_OUT/alignment_model_best.pt"
  --stage3-metrics "$STAGE5_OUT/alignment_metrics.json"
  --node-features "$FEATURES"
  --nodes "$NODES"
  --out-dir "$EMBED_OUT"
  --device "$DEVICE"
  --batch-size "${EXPORT_BATCH_SIZE:-8192}"
  --max-nodes "$MAX_EXPORT_NODES"
  --input-hash-policy sha256
)
if [ "$MODE" != "smoke" ]; then
  EXPORT_ARGS+=(--require-node-metadata-coverage --require-exact-node-set)
fi
"$PYTHON_BIN" -u "$SCRIPT_DIR/10_export_node_embeddings_v6_3.py" "${EXPORT_ARGS[@]}"

VERIFY_ARGS=(
  --embeddings "$EMBED_OUT/node_embeddings.pt"
  --index "$EMBED_OUT/node_embedding_index.json"
  --manifest "$EMBED_OUT/node_embedding_export_manifest.json"
  --report "$EMBED_OUT/node_embedding_export_verification.json"
)
if [ "$MODE" != "smoke" ]; then VERIFY_ARGS+=(--require-formal); fi
"$PYTHON_BIN" -u "$SCRIPT_DIR/11_verify_node_embedding_export_v6_3.py" "${VERIFY_ARGS[@]}"

echo "2.2.1 Stage5 and export: PASSED"
echo "STAGE5_OUT=$STAGE5_OUT"
echo "EMBED_OUT=$EMBED_OUT"
