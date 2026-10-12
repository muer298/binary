#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=/data/chaoni/mjy/resym_exp/training_runs/seraph_local/clean_joint_v0_2_10
test -f "$OUT/v0_2_10_contract.json"
test ! -e "$OUT/formal_9runs.log"
test ! -e "$OUT/formal_9runs.exit"
test ! -e "$OUT/formal_9runs.pid"
test ! -e "$OUT/arms"
mkdir -p "$OUT/logs"
nohup bash -c '
  ROOT="$1"
  OUT="$2"
  status=0
  for arm in symbolic_only_clean unified_llm_gate_clean module_aware_clean; do
    for seed in 1337 3407 20260929; do
      log="$OUT/logs/${arm}__${seed}.log"
      exitfile="$OUT/logs/${arm}__${seed}.exit"
      if test -e "$log" || test -e "$exitfile"; then
        printf "REFUSING_OVERWRITE %s %s\n" "$arm" "$seed"
        status=90
        break 2
      fi
      printf "RUN_START %s %s %s\n" "$arm" "$seed" "$(date -u +%FT%TZ)"
      if bash "$ROOT/run.sh" train-one --arm "$arm" --seed "$seed" --gpu 0 > "$log" 2>&1; then
        printf "0\n" > "$exitfile"
        printf "RUN_PASSED %s %s %s\n" "$arm" "$seed" "$(date -u +%FT%TZ)"
      else
        status=$?
        printf "%s\n" "$status" > "$exitfile"
        printf "RUN_FAILED %s %s exit=%s\n" "$arm" "$seed" "$status"
        break 2
      fi
    done
  done
  if test "$status" -eq 0; then
    printf "AGGREGATE_START %s\n" "$(date -u +%FT%TZ)"
    if bash "$ROOT/run.sh" aggregate > "$OUT/logs/aggregate.log" 2>&1; then
      printf "AGGREGATE_PASSED\n"
    else
      status=$?
      printf "AGGREGATE_FAILED exit=%s\n" "$status"
    fi
  fi
  if test "$status" -eq 0; then
    printf "SELECT_FINAL_START %s\n" "$(date -u +%FT%TZ)"
    if bash "$ROOT/run.sh" select-final > "$OUT/logs/select_final.log" 2>&1; then
      printf "SELECT_FINAL_PASSED\n"
    else
      status=$?
      printf "SELECT_FINAL_FAILED exit=%s\n" "$status"
    fi
  fi
  printf "%s\n" "$status" > "$OUT/formal_9runs.exit"
' _ "$ROOT" "$OUT" > "$OUT/formal_9runs.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$OUT/formal_9runs.pid"
printf 'FORMAL_9RUNS_PID=%s\n' "$pid"
