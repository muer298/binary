#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$ROOT/config.json"
PYTHON="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$CONFIG")"

usage() {
  echo "Usage: bash run.sh {test|preflight|prepare-contract|train-one|aggregate|select-final} [arguments]"
  echo "  train-one requires: --arm NAME --seed INTEGER --gpu INDEX"
}

case "${1:-}" in
  test)
    exec "$PYTHON" -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v
    ;;
  preflight)
    exec "$PYTHON" "$ROOT/src/preflight.py" --config "$CONFIG" "${@:2}"
    ;;
  prepare-contract)
    exec "$PYTHON" "$ROOT/src/prepare_contract.py" --config "$CONFIG"
    ;;
  train-one)
    if [[ $# -lt 7 ]]; then usage; exit 2; fi
    exec "$PYTHON" "$ROOT/src/train_clean_joint.py" --config "$CONFIG" "${@:2}"
    ;;
  aggregate)
    exec "$PYTHON" "$ROOT/src/aggregate_reports.py" --config "$CONFIG"
    ;;
  select-final)
    exec "$PYTHON" "$ROOT/src/select_final.py" --config "$CONFIG"
    ;;
  *)
    usage
    exit 2
    ;;
esac
