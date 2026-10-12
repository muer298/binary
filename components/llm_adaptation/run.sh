#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/cn/miniconda3/envs/seraph_llm/bin/python
case "${1:-}" in
  prepare) exec "$PY" -u "$ROOT/src/prepare.py" --config "$ROOT/config.json" ;;
  train-clean)
    GPU="${2:?usage: bash run.sh train-clean <GPU_ID>}"
    exec "$PY" -u "$ROOT/src/train_clean.py" --config "$ROOT/config.json" --gpu "$GPU" ;;
  verify) exec "$PY" -u "$ROOT/src/verify.py" --config "$ROOT/config.json" ;;
  *) echo "Usage: bash run.sh {prepare|train-clean <GPU_ID>|verify}" >&2; exit 2 ;;
esac
