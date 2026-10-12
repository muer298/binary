#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query complete node provenance and UpdateEvent history."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping

from binding_store_v6_4 import connect_store, get_metadata, trace_node


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent),
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def read_ids(path: str) -> List[str]:
    if not path:
        return []
    return [
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--node-id", action="append", default=[])
    parser.add_argument("--node-ids-file", default="")
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--node-type", default="")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--out", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    store = Path(args.store).resolve()
    if not store.is_file():
        raise FileNotFoundError(store)
    connection = connect_store(store, read_only=True)
    node_ids = list(args.node_id) + read_ids(args.node_ids_file)
    if args.sample_id or args.node_type:
        conditions = []
        parameters: List[Any] = []
        if args.sample_id:
            conditions.append("sample_id=?")
            parameters.append(args.sample_id)
        if args.node_type:
            conditions.append("node_type=?")
            parameters.append(args.node_type)
        query = "SELECT node_id FROM bindings WHERE " + " AND ".join(conditions)
        query += " ORDER BY source_index_row LIMIT ?"
        parameters.append(max(1, int(args.limit)))
        node_ids.extend(row["node_id"] for row in connection.execute(query, parameters))
    node_ids = list(dict.fromkeys(str(node_id).strip() for node_id in node_ids if str(node_id).strip()))
    if not node_ids:
        raise SystemExit(
            "ERROR: select at least one node with --node-id/--node-ids-file/--sample-id"
        )
    traces: List[Dict[str, Any]] = []
    errors = []
    for node_id in node_ids[:max(1, int(args.limit))]:
        try:
            traces.append(trace_node(connection, node_id, max(0, int(args.max_depth))))
        except Exception as exc:
            errors.append({"node_id": node_id, "error": str(exc)})
    result = {
        "schema": "task2_binding_trace_query_v6_4",
        "store": str(store),
        "metadata": get_metadata(connection),
        "requested": len(node_ids),
        "returned": len(traces),
        "errors": errors,
        "traces": traces,
    }
    connection.close()
    if args.out:
        out = Path(args.out).resolve()
        if out.exists() and not args.overwrite:
            raise FileExistsError("Refusing to overwrite trace: %s" % out)
        atomic_json(out, result)
        print("wrote", out)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit("ERROR: some requested nodes could not be traced")


if __name__ == "__main__":
    main()

