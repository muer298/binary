#!/usr/bin/env bash
# 2.2.1.5：构建不可变节点绑定、事件链和可追溯查询接口。
set -euo pipefail

: "${WORK_ROOT:?export WORK_ROOT first}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TAG="${RUN_TAG:-final}"
MODE="${MODE:-smoke}"

EMBED_DIR="${EMBED_DIR:-$WORK_ROOT/outputs/embedding_${RUN_TAG}_${MODE}}"
EMBEDDINGS="${EMBEDDINGS:-$EMBED_DIR/node_embeddings.pt}"
INDEX="${INDEX:-$EMBED_DIR/node_embedding_index.json}"
EMBED_MANIFEST="${EMBED_MANIFEST:-$EMBED_DIR/node_embedding_export_manifest.json}"
NODES="${NODES:-$WORK_ROOT/graphs/usg_nodes_enriched_${RUN_TAG}.jsonl}"
EDGES="${EDGES:-$WORK_ROOT/graphs/usg_edges_${RUN_TAG}.jsonl}"
BINDING_OUT="${BINDING_OUT:-$WORK_ROOT/outputs/binding_${RUN_TAG}_${MODE}}"
BINDING_STORE="$BINDING_OUT/node_binding_store.sqlite3"
BINDING_MANIFEST="$BINDING_OUT/binding_store_manifest.json"
BINDING_VERIFY="$BINDING_OUT/binding_store_verification.json"
BINDING_TRACE="$BINDING_OUT/binding_store_trace.json"

case "$MODE" in
  smoke) MAX_NODES="${MAX_NODES:-5000}"; MAX_EDGES="${MAX_EDGES:-200000}"; REQUIRE_FORMAL=0 ;;
  full|formal) MAX_NODES=0; MAX_EDGES=0; REQUIRE_FORMAL=1 ;;
  *) echo "ERROR: MODE must be smoke, full, or formal" >&2; exit 2 ;;
esac

for input_path in "$EMBEDDINGS" "$INDEX" "$EMBED_MANIFEST" "$NODES" "$EDGES"; do
  test -s "$input_path" || { echo "ERROR: missing or empty $input_path" >&2; exit 1; }
done
test ! -e "$BINDING_OUT" || { echo "ERROR: refusing to reuse $BINDING_OUT" >&2; exit 1; }
mkdir -p "$BINDING_OUT"

BUILD_ARGS=(
  --embedding-index "$INDEX" --embedding-manifest "$EMBED_MANIFEST"
  --embeddings "$EMBEDDINGS" --nodes "$NODES" --edges "$EDGES"
  --store "$BINDING_STORE" --manifest "$BINDING_MANIFEST"
  --max-nodes "$MAX_NODES" --max-edges "$MAX_EDGES"
  --commit-every "${COMMIT_EVERY:-50000}"
)
if [ "$REQUIRE_FORMAL" -eq 1 ]; then BUILD_ARGS+=(--require-formal); fi
"$PYTHON_BIN" -u "$SCRIPT_DIR/12_build_embedding_binding_records.py" "${BUILD_ARGS[@]}"

TRACE_NODE_ID="${TRACE_NODE_ID:-}"
if [ -z "$TRACE_NODE_ID" ]; then
  TRACE_NODE_ID="$($PYTHON_BIN - "$INDEX" "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from binding_store_v6_4 import load_index, resolve_node_metadata
rows = load_index(Path(sys.argv[1]))
print(resolve_node_metadata(rows[0])["node_id"] if rows else "")
PY
)"
fi

VERIFY_ARGS=(--store "$BINDING_STORE" --manifest "$BINDING_MANIFEST" --report "$BINDING_VERIFY")
if [ -n "$TRACE_NODE_ID" ]; then VERIFY_ARGS+=(--trace-node-id "$TRACE_NODE_ID"); fi
if [ "$REQUIRE_FORMAL" -eq 1 ]; then VERIFY_ARGS+=(--require-formal); fi
"$PYTHON_BIN" -u "$SCRIPT_DIR/15_verify_binding_store_v6_4.py" "${VERIFY_ARGS[@]}"

if [ -n "$TRACE_NODE_ID" ]; then
  "$PYTHON_BIN" -u "$SCRIPT_DIR/14_trace_binding_store_v6_4.py" \
    --store "$BINDING_STORE" --node-id "$TRACE_NODE_ID" --max-depth 1 --out "$BINDING_TRACE"
fi

echo "2.2.1 BindingStore: PASSED"
echo "BINDING_STORE=$BINDING_STORE"
