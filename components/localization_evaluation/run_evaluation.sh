#!/usr/bin/env bash
set -euo pipefail

PKG=/data/chaoni/mjy/resym_exp/packages/seraph_task2_localization_eval_v0_1
OUT=/data/chaoni/mjy/resym_exp/training_runs/seraph_task2/localization_eval_v0_1
PY=/home/cn/miniconda3/envs/codet/bin/python

test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 2; }
cd "$PKG"
"$PY" -m py_compile evaluate_localization.py
/usr/bin/time -v "$PY" evaluate_localization.py --out "$OUT" 2>&1 | tee /tmp/seraph_localization_eval_v01.log
