#!/usr/bin/env bash
set -Eeuo pipefail
: "${JSON_ROOT:?set JSON_ROOT to blocksem_v7 semantic outputs}"
: "${TASK2:?set TASK2 output directory}"
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BUILD_INDEX="${BUILD_INDEX:-$PKG/01_modified_upstream/01_build_usg_index_v7.py}"
mkdir -p "$TASK2"
python3 "$BUILD_INDEX" --json-root "$JSON_ROOT" --out "$TASK2/usg_index.jsonl" --stats "$TASK2/usg_index_stats.json"
python3 "$PKG/01_modified_upstream/02_extract_nodes_edges_blocksem_v7.py" \
  --index "$TASK2/usg_index.jsonl" \
  --nodes-out "$TASK2/usg_nodes.jsonl" \
  --edges-out "$TASK2/usg_edges.jsonl" \
  --stats "$TASK2/graph_schema_stats.json"
python3 - <<'PY'
import json,os
for name in ('usg_index_stats.json','graph_schema_stats.json'):
 p=os.path.join(os.environ['TASK2'],name)
 print('\n===',p,'===')
 print(json.dumps(json.load(open(p)),ensure_ascii=False,indent=2))
PY
