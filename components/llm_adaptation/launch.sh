#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=/data/chaoni/mjy/resym_exp/training_runs/seraph_local/stage1_clean_v0_1_5
test ! -e "$OUT/formal_training.log"
test ! -e "$OUT/formal_training.exit"
test ! -e "$OUT/formal_training.pid"
test ! -e "$OUT/contrastive"
nohup bash -c '
  bash "$1/run.sh" train-clean 0 > "$2/formal_training.log" 2>&1
  status=$?
  printf "%s\n" "$status" > "$2/formal_training.exit"
' _ "$ROOT" "$OUT" </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$OUT/formal_training.pid"
printf 'FORMAL_TRAINING_PID=%s\n' "$pid"
