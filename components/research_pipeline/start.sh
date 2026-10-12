#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$HERE/CONFIG.env"

MODE="${1:-status}"
if [[ "$#" -gt 0 ]]; then shift; fi

case "$MODE" in
  status|models|select)
    exec "$PYTHON_BIN" "$HERE/control.py" "$MODE" "$@"
    ;;
  prepare|all|continue)
    JOB="$(date -u +%Y%m%dT%H%M%SZ)_${MODE}_$$"
    mkdir -p "$RUN_DIR/background_jobs/$JOB"
    JD="$RUN_DIR/background_jobs/$JOB"
    if command -v nice >/dev/null 2>&1; then
      nohup nice -n "$JOB_NICE" "$PYTHON_BIN" -u "$HERE/control.py" "$MODE" --job-dir "$JD" "$@" \
        </dev/null >"$JD/run.log" 2>&1 &
    else
      nohup "$PYTHON_BIN" -u "$HERE/control.py" "$MODE" --job-dir "$JD" "$@" \
        </dev/null >"$JD/run.log" 2>&1 &
    fi
    PID=$!
    printf '%s\n' "$PID" > "$JD/launcher.pid"
    printf '%s\n' "$JD" > "$RUN_DIR/background_jobs/latest_${MODE}.txt"
    printf '已提交后台任务 PID=%s\nRUN=%s\nJOB=%s\nLOG=%s/run.log\n' "$PID" "$RUN_DIR" "$JD" "$JD"
    printf '查看：tail -f %q\n' "$JD/run.log"
    sleep 1
    if ! kill -0 "$PID" 2>/dev/null; then
      echo '任务已很快结束；最后日志：'
      tail -60 "$JD/run.log" || true
    fi
    ;;
  *)
    echo '用法：bash start.sh status|models|select|prepare|all|continue' >&2
    exit 2
    ;;
esac
