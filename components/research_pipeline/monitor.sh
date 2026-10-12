#!/usr/bin/env bash
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/CONFIG.env"

echo "============================================================"
echo "BlockSem subset100 v7.1.0"
echo "RUN=$RUN_DIR"
echo "============================================================"

"$PYTHON_BIN" "$HERE/control.py" status || true

echo
echo "========== latest pipeline progress =========="
if [[ -f "$RUN_DIR/pipeline_progress.json" ]]; then
  cat "$RUN_DIR/pipeline_progress.json"
else
  echo "NOT_STARTED"
fi

echo
echo "========== latest background job =========="
LATEST=$(
  find "$RUN_DIR/background_jobs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
  | sort | tail -1
)
if [[ -n "${LATEST:-}" ]]; then
  echo "JOB=$LATEST"
  if [[ -f "$LATEST/run.log" ]]; then
    tail -40 "$LATEST/run.log"
  fi
else
  echo "NO_JOB"
fi

echo
echo "========== rankings =========="
for f in candidate_rankings.csv candidate_rankings.jsonl rank_summary.json; do
  if [[ -s "$RUN_DIR/$f" ]]; then
    ls -lh "$RUN_DIR/$f"
  else
    echo "NOT_YET $f"
  fi
done
