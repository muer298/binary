#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SERAPH_V022_MAPPING_CONFIG:-$ROOT/config.json}"
PY="$(python3 "$ROOT/src/config_value.py" "$CFG" python)"

case "${1:-}" in
  export)
    exec "$PY" "$ROOT/src/run_full_export.py" --config "$CFG"
    ;;
  audit)
    exec "$PY" "$ROOT/src/audit_mapping.py" --config "$CFG"
    ;;
  all)
    "$PY" "$ROOT/src/run_full_export.py" --config "$CFG"
    exec "$PY" "$ROOT/src/audit_mapping.py" --config "$CFG"
    ;;
  *)
    echo "usage: bash run.sh {export|audit|all}" >&2
    exit 2
    ;;
esac
