#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build manifest with explicit full_path + stripped_path.

Example:
python3 scripts/build_manifest_with_full.py \
  --full-root samples_c80/full \
  --stripped-root samples_c80/stripped \
  --out samples_c80/manifest_c80.csv
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full-root', required=True)
    ap.add_argument('--stripped-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--status', default='ok')
    args = ap.parse_args()

    full_root = Path(args.full_root).resolve()
    stripped_root = Path(args.stripped_root).resolve()
    out = Path(args.out).resolve()

    rows = []
    missing_full = []
    for sp in sorted(stripped_root.rglob('*')):
        if not sp.is_file():
            continue
        rel = sp.relative_to(stripped_root)
        fp = full_root / rel
        if not fp.exists():
            missing_full.append(str(rel))
            continue
        rows.append({
            'sample_id': str(rel).replace('/', '__'),
            'full_path': str(fp.resolve()),
            'stripped_path': str(sp.resolve()),
            'status': args.status,
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'full_path', 'stripped_path', 'status'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'wrote {out} with {len(rows)} rows')
    if missing_full:
        print(f'WARNING: missing full binaries for {len(missing_full)} stripped files')
        for x in missing_full[:10]:
            print('  missing full:', x)


if __name__ == '__main__':
    main()
