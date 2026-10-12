#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/chaoni/mjy/resym_exp"
LIST="$ROOT/subset/resym_small80_paths.txt"
FULL="$ROOT/samples_s80/full"
STRIPPED="$ROOT/samples_s80/stripped"

mkdir -p "$FULL" "$STRIPPED"

total=$(wc -l < "$LIST")
i=0

while IFS= read -r src; do
  [ -z "$src" ] && continue
  i=$((i + 1))

  rel="${src#$ROOT/raw/bin/}"
  if [ "$rel" = "$src" ]; then
    rel="$(basename "$src")"
  fi

  full_out="$FULL/$rel"
  strip_out="$STRIPPED/$rel"

  mkdir -p "$(dirname "$full_out")"
  mkdir -p "$(dirname "$strip_out")"

  # full 保留 DWARF。优先硬链接，省空间；不行再复制。
  if [ ! -e "$full_out" ]; then
    ln "$src" "$full_out" 2>/dev/null || cp -p "$src" "$full_out"
  fi

  # stripped 去掉 debug 信息。
  if [ ! -s "$strip_out" ]; then
    cp -p "$src" "$strip_out.tmp"
    objcopy --strip-debug "$strip_out.tmp"
    mv "$strip_out.tmp" "$strip_out"
  fi

  if [ $((i % 500)) -eq 0 ]; then
    echo "prepared $i / $total"
  fi
done < "$LIST"

echo "done: prepared $i samples"
