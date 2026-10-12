#!/usr/bin/env bash
# 无需真实数据的包级检查：Python/Shell 语法、历史文件禁入、绝对路径扫描和测试。
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" - "$PACKAGE_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in sorted(root.glob("*.py")):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("Python syntax: PASSED (%d files)" % len(list(root.glob("*.py"))))
PY

while IFS= read -r -d '' file_path; do
  bash -n "$file_path"
done < <(find "$PACKAGE_ROOT" -maxdepth 1 -type f -name '*.sh' -print0)

if find "$PACKAGE_ROOT" -type f \( \
  -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.pt' \
  -o -name '*.pth' -o -name '*.ckpt' -o -name '*.sqlite3' -o -name '*.jsonl' \
\) -print -quit | grep -q .; then
  echo "ERROR: package contains data/model/result artifacts" >&2
  exit 1
fi

if find "$PACKAGE_ROOT" -type f \( \
  -iname '*legacy*' -o -iname '*original*' -o -iname '*backup*' \
  -o -iname 'old_*' -o -iname '*_old.*' -o -iname '*-old.*' \
\) -print -quit | grep -q .; then
  echo "ERROR: package contains a historical or backup filename" >&2
  exit 1
fi

if rg -n "(^|[[:space:]\"'])/(data|Users|home)/" "$PACKAGE_ROOT" \
  -g '*.py' -g '*.sh' -g '*.env' -g '!run_static_checks.sh' >/dev/null; then
  echo "ERROR: package contains a hard-coded machine path" >&2
  exit 1
fi

export PYTHONPATH="$PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m unittest discover -s "$PACKAGE_ROOT/tests" -p 'test_*.py'

echo "PURE SCRIPT BUNDLE CHECK: PASSED"
