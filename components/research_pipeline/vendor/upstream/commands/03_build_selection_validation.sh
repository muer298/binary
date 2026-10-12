#!/usr/bin/env bash
set -Eeuo pipefail
: "${GOLD:?set GOLD to independently rebuilt object Gold JSONL/JSON/CSV}"
: "${BASELINE:?set BASELINE to frozen pre-repair baseline predictions}"
: "${SEL_OUT:?set SEL_OUT to a NEW output directory}"
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
test ! -e "$SEL_OUT" || { echo "Refuse to overwrite Selection directory: $SEL_OUT" >&2; exit 2; }
python3 "$PKG/03_selection_validation/build_selection_validation_v7.py" \
  --gold "$GOLD" --baseline "$BASELINE" --out-dir "$SEL_OUT" \
  --group-by project --min-strong 100 --min-baseline-errors 10
python3 "$PKG/03_selection_validation/verify_selection_validation_v7.py" --selection-dir "$SEL_OUT"
cat "$SEL_OUT/selection_gate.json"
