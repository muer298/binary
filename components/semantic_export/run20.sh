#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN=/data/chaoni/mjy/resym_exp/validation_runs/blocksem_subset20_v7_1_0
PREVIOUS="$RUN/downstream_readiness_v7_2_1/all_scopefix"
for ARG in "$@"; do
  case "$ARG" in
    --run-dir|--run-dir=*|--previous-dir|--previous-dir=*)
      echo 'run20.sh fixes the existing 20-sample input; use start.sh for another run.' >&2; exit 2;;
  esac
done
exec python3 -S "$HERE/launch.py" "$@" --run-dir "$RUN" --previous-dir "$PREVIOUS"
