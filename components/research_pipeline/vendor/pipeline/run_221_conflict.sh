#!/usr/bin/env bash
# 2.2.1.6：Level-3 mask / Level-2 project / Level-1 blend 分级消解。
set -euo pipefail

: "${WORK_ROOT:?export WORK_ROOT first}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TAG="${RUN_TAG:-final}"
MODE="${MODE:-smoke}"

EMBED_DIR="${EMBED_DIR:-$WORK_ROOT/outputs/embedding_${RUN_TAG}_${MODE}}"
NODES="${NODES:-$WORK_ROOT/graphs/usg_nodes_enriched_${RUN_TAG}.jsonl}"
EDGES="${EDGES:-$WORK_ROOT/graphs/usg_edges_${RUN_TAG}.jsonl}"
EMBEDDINGS="${EMBEDDINGS:-$EMBED_DIR/node_embeddings.pt}"
INDEX="${INDEX:-$EMBED_DIR/node_embedding_index.json}"
BINDING_STORE="${BINDING_STORE:-$WORK_ROOT/outputs/binding_${RUN_TAG}_${MODE}/node_binding_store.sqlite3}"
API_PROVENANCE="${API_PROVENANCE:-}"
OUT_2216="${OUT_2216:-$WORK_ROOT/outputs/conflict_${RUN_TAG}_${MODE}}"

case "$MODE" in
  smoke) MAX_NODES="${MAX_NODES:-100}" ;;
  full|formal) MAX_NODES="${MAX_NODES:-0}" ;;
  *) echo "ERROR: MODE must be smoke, full, or formal" >&2; exit 2 ;;
esac
for input_path in "$NODES" "$EDGES" "$EMBEDDINGS" "$INDEX"; do
  test -s "$input_path" || { echo "ERROR: missing or empty $input_path" >&2; exit 1; }
done
test ! -e "$OUT_2216" || { echo "ERROR: refusing to reuse $OUT_2216" >&2; exit 1; }

RUN_ARGS=(
  --nodes "$NODES" --edges "$EDGES" --embeddings "$EMBEDDINGS" --index "$INDEX"
  --out-dir "$OUT_2216" --max-nodes "$MAX_NODES"
  --max-iterations "${MAX_ITERATIONS:-10}" --progress-every "${PROGRESS_EVERY:-1000000}"
)
if [ -n "$API_PROVENANCE" ]; then
  test -s "$API_PROVENANCE" || { echo "ERROR: missing API_PROVENANCE" >&2; exit 1; }
  RUN_ARGS+=(--api-provenance "$API_PROVENANCE")
fi
if [ -s "$BINDING_STORE" ]; then RUN_ARGS+=(--binding-store "$BINDING_STORE"); fi
if [ "${APPLY_EVENTS:-0}" = "1" ]; then
  test -s "$BINDING_STORE" || { echo "ERROR: APPLY_EVENTS requires BINDING_STORE" >&2; exit 1; }
  RUN_ARGS+=(--apply-events)
fi
"$PYTHON_BIN" -u "$SCRIPT_DIR/13_resolve_hierarchical_constraints.py" "${RUN_ARGS[@]}"

VERIFY_ARGS=(--out-dir "$OUT_2216" --min-results "${MIN_RESULTS:-1}")
if [ -s "$BINDING_STORE" ]; then VERIFY_ARGS+=(--binding-store "$BINDING_STORE"); fi
if [ "${APPLY_EVENTS:-0}" = "1" ]; then VERIFY_ARGS+=(--require-events-applied); fi
"$PYTHON_BIN" -u "$SCRIPT_DIR/15_verify_conflict_resolution_v6_5.py" "${VERIFY_ARGS[@]}"

echo "2.2.1 conflict resolution: PASSED"
echo "OUT_2216=$OUT_2216"
