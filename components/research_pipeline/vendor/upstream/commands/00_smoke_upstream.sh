#!/usr/bin/env bash
set -Eeuo pipefail
: "${P:?set project root P}"
: "${MANIFEST:?set smoke manifest MANIFEST}"
: "${GHIDRA_SUPPORT:?set Ghidra support directory}"
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
OUT="${OUT:-$P/usg_outputs/blocksem_v7_smoke}"
WORK="${GHIDRA_WORK:-$P/ghidra_work/blocksem_v7_smoke}"
python3 "$PKG/01_modified_upstream/run_usg_headless_stream_v2.py" \
  --manifest "$MANIFEST" \
  --ghidra-support "$GHIDRA_SUPPORT" \
  --jython-script "$PKG/01_modified_upstream/over_v11_typesem_plus_paramflow_callfix_blocksem_v7.py" \
  --out-root "$OUT" --work-root "$WORK" --max-samples 3 --timeout-sec 1800 --analysis-timeout-per-file 600
find "$OUT" -type f -name '*_semantic_dict.json' | head -3 | while read -r f; do
  echo "=== $f ==="
  python3 "$PKG/01_modified_upstream/inspect_blocksem_v7_contract.py" "$f"
done
