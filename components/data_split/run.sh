#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$ROOT/config.json")"
case "${1:-}" in
  build) exec "$PY" "$ROOT/src/build_global_split.py" --config "$ROOT/config.json" ;;
  test) exec "$PY" -m pytest -q "$ROOT/tests" ;;
  *) echo "usage: bash run.sh {build|test}" >&2; exit 2 ;;
esac
