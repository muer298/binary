#!/usr/bin/env bash
# 可选前置：从原始 semantic_dict 来源补全 LayoutSem 固定偏移/访问宽度证据。
# 不读 Gold、不覆盖输入；缺失来源默认失败关闭。
set -euo pipefail

: "${NODES_IN:?export NODES_IN first}"
: "${NODES_OUT:?export NODES_OUT first}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPORT_OUT="${REPORT_OUT:-${NODES_OUT%.jsonl}_layout_evidence_report.json}"

test -s "$NODES_IN" || { echo "ERROR: missing or empty $NODES_IN" >&2; exit 1; }
test ! -e "$NODES_OUT" || { echo "ERROR: refusing to overwrite $NODES_OUT" >&2; exit 1; }
test ! -e "$REPORT_OUT" || { echo "ERROR: refusing to overwrite $REPORT_OUT" >&2; exit 1; }

ARGS=(
  --nodes-in "$NODES_IN" --nodes-out "$NODES_OUT" --report-out "$REPORT_OUT"
  --fail-on-source-error
)
if [ -n "${ALLOW_MISSING_SOURCE_FILE:-}" ]; then
  test -s "$ALLOW_MISSING_SOURCE_FILE" || { echo "ERROR: invalid allow-list" >&2; exit 1; }
  ARGS+=(--allow-missing-source-file "$ALLOW_MISSING_SOURCE_FILE")
  ARGS+=(--max-allowed-missing-sources "${MAX_ALLOWED_MISSING_SOURCES:-1}")
fi
"$PYTHON_BIN" -u "$SCRIPT_DIR/26_enrich_layout_evidence_v6_7_4.py" "${ARGS[@]}"
echo "LayoutSem evidence enrichment: PASSED"
echo "NODES_OUT=$NODES_OUT"
echo "REPORT_OUT=$REPORT_OUT"
