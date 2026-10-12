#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$ROOT/config.json"
PYTHON="/home/cn/miniconda3/envs/seraph_llm/bin/python"
OUT="/data/chaoni/mjy/resym_exp/training_runs/seraph_local/clean_hsym_v0_2_8"
DATA="$OUT/data"
TRAIN="$OUT/training"
RAW_EDGES="/data/chaoni/mjy/resym_exp/validation_runs/blocksem_subset20_v7_1_0/steps/01_graph/usg_edges.jsonl"
ALL_ELIGIBLE_NODES="/data/chaoni/mjy/resym_exp/validation_runs/blocksem_subset20_v7_1_0/steps/02_features/graphs/model_input.jsonl"

prepare() {
  "$PYTHON" "$ROOT/src/prepare_clean_data.py" --config "$CONFIG"
  "$PYTHON" "$ROOT/src/build_clean_features.py" \
    --config "$CONFIG" \
    --nodes "$DATA/train_nodes.jsonl" \
    --edges "$DATA/train_edges.jsonl" \
    --output "$OUT/clean_node_features.pt" \
    --scope train
  "$PYTHON" "$ROOT/src/build_clean_features.py" \
    --config "$CONFIG" \
    --nodes "$ALL_ELIGIBLE_NODES" \
    --edges "$RAW_EDGES" \
    --output "$OUT/clean_node_features_coverage_only.pt" \
    --scope coverage_only
  cp "$CONFIG" "$OUT/training_config_snapshot.json"
}

train() {
  mkdir -p "$TRAIN"
  "$PYTHON" "$ROOT/src/train_clean_alignment.py" \
    --node-features "$OUT/clean_node_features.pt" \
    --neighborhoods "$DATA/train_neighborhoods_compact.jsonl" \
    --nodes "$DATA/train_nodes.jsonl" \
    --edges "$DATA/train_edges.jsonl" \
    --out-dir "$TRAIN" \
    --split-mode sample \
    --split-manifest "$DATA/encoder_inner_split_manifest.jsonl" \
    --require-split-manifest \
    --missing-split-policy error \
    --epochs 30 \
    --eval-every 1 \
    --batch-size 24 \
    --eval-batch-size 128 \
    --hidden-dim 128 \
    --embed-dim 64 \
    --lr 0.001 \
    --lambda-evidence 1.0 \
    --lambda-constraint-margin 0 \
    --lambda-constraint-margin-start 0 \
    --lambda-structure-recon 0.04 \
    --seed 1337 \
    --device cuda
  cp "$TRAIN/alignment_model_final.pt" "$OUT/clean_alignment_checkpoint.pt"
}

export_embeddings() {
  "$PYTHON" "$ROOT/src/export_clean_embeddings.py" \
    --features "$OUT/clean_node_features_coverage_only.pt" \
    --schema "$OUT/clean_feature_schema_coverage_only.json" \
    --vocab "$OUT/clean_feature_vocab_coverage_only.json" \
    --checkpoint "$OUT/clean_alignment_checkpoint.pt" \
    --output "$OUT/node_embeddings_clean.pt" \
    --index-output "$OUT/node_embedding_index_clean.json" \
    --report "$OUT/clean_embedding_export_report.json" \
    --device cuda
}

coverage() {
  "$PYTHON" "$ROOT/src/audit_embedding_coverage.py" \
    --config "$CONFIG" \
    --index "$OUT/node_embedding_index_clean.json"
}

finalize() {
  "$PYTHON" "$ROOT/src/finalize_reports.py" --config "$CONFIG"
}

case "${1:-all}" in
  prepare) prepare ;;
  train) train ;;
  export) export_embeddings ;;
  coverage) coverage ;;
  finalize) finalize ;;
  all) prepare; train; export_embeddings; coverage; finalize ;;
  *) echo "usage: $0 {prepare|train|export|coverage|finalize|all}" >&2; exit 2 ;;
esac
