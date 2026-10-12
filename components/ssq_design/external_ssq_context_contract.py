"""Freeze the DESIGN context policy before any External SSQ API request.

Only DESIGN source records are JSON-decoded. The frozen request objects are read,
validated and hashed, never rebuilt or rewritten. HOLDOUT has a rule, not rows.
"""

from __future__ import annotations

import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import external_ssq_design_client as design


SOURCE_FILES = (
    design.BASE / "clean_joint_v0_2_10/dev_manifest.jsonl",
    design.BASE / "ssq_layout_dev_supplement_v0_3_1a/layout_records/mapped_layout_records.jsonl",
)
ID_RE = re.compile(r'"record_id"\s*:\s*"([^"]+)"')
LOCAL = "VERIFIED_LOCAL"
FUNCTION = "FUNCTION_LEVEL_FALLBACK"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_design_rows(ids: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for path in SOURCE_FILES:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                match = ID_RE.search(line)
                if match is None or match.group(1) not in ids:
                    continue
                row = json.loads(line)  # A DESIGN record only; never decode HOLDOUT Gold.
                rid = row["record_id"]
                if rid in found:
                    raise RuntimeError(f"duplicate DESIGN source record: {rid}")
                found[rid] = row
    if set(found) != ids:
        raise RuntimeError(f"DESIGN source coverage {len(found)}/{len(ids)}")
    return found


def classify(request_row: dict, source_row: dict) -> dict:
    rid = request_row["request_id"]
    module = request_row["module"]
    request = request_row["request"]
    code = request["code"]
    mapping = source_row.get("instruction_mapping") or {}
    status = mapping.get("mapping_status") or "MISSING"
    usable = mapping.get("instruction_usable") is True
    verified = status == "VERIFIED" and usable
    scope = code["scope"]
    if source_row["record_id"] != rid or request["request_id"] != rid:
        raise RuntimeError(f"identity drift: {rid}")
    if source_row["module"] != module:
        raise RuntimeError(f"module drift: {rid}")

    if module == "InterfaceSem":
        if scope != "function" or code["pseudo_c"] != source_row["pseudo_c"][:7000]:
            raise RuntimeError(f"InterfaceSem function context drift: {rid}")
        mode = FUNCTION
        role = "PRIMARY_INTERFACE_FUNCTION_CONTEXT"
    elif module in ("TypeSem", "LayoutSem") and verified:
        if scope != "verified_instruction_local":
            raise RuntimeError(f"verified scope drift: {rid}")
        if code.get("mapping_level") != mapping.get("mapping_level"):
            raise RuntimeError(f"mapping level drift: {rid}")
        if code.get("instruction_address") != mapping.get("instruction_address"):
            raise RuntimeError(f"instruction identity drift: {rid}")
        if "<FOCUS>" not in code["pseudo_c"] or "</FOCUS>" not in code["pseudo_c"]:
            raise RuntimeError(f"local focus marker missing: {rid}")
        mode = LOCAL
        role = "VERIFIED_INSTRUCTION_LOCAL"
    elif module in ("TypeSem", "LayoutSem"):
        if scope != "function_fallback_unverified_instruction":
            raise RuntimeError(f"fallback scope drift: {rid}")
        if code["pseudo_c"] != source_row["pseudo_c"][:7000]:
            raise RuntimeError(f"fallback pseudo-C drift: {rid}")
        if code.get("instruction_address") is not None or code.get("mapping_level") is not None:
            raise RuntimeError(f"unverified instruction metadata in fallback: {rid}")
        mode = FUNCTION
        role = "UNVERIFIED_LOCAL_FALLBACK"
    else:
        raise RuntimeError(f"unsupported module: {module}")

    return {
        "record_id": rid,
        "module": module,
        "mapping_status": status,
        "instruction_usable": usable,
        "mapping_level": mapping.get("mapping_level") if mode == LOCAL else None,
        "context_mode": mode,
        "function_context_role": role,
        "frozen_request_scope": scope,
        "request_sha256": sha_bytes(design.canonical(request)),
    }


def write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)


def main() -> None:
    rows, fresh_preflight = design.preflight()
    preflight_path = design.OUT / "DESIGN_PREFLIGHT.json"
    if json.loads(preflight_path.read_text(encoding="utf-8")) != fresh_preflight:
        raise RuntimeError("frozen DESIGN preflight differs from fresh read-only audit")
    if (design.OUT / "attempts").exists():
        raise RuntimeError("API attempts already exist; cannot freeze context after API")
    ids = {row["request_id"] for row in rows}
    if len(ids) != 628:
        raise RuntimeError("DESIGN request count or uniqueness drift")
    source = source_design_rows(ids)
    entries = [classify(row, source[row["request_id"]]) for row in rows]
    counts = collections.Counter((r["module"], r["context_mode"]) for r in entries)
    expected = {
        ("TypeSem", LOCAL): 197,
        ("TypeSem", FUNCTION): 18,
        ("InterfaceSem", FUNCTION): 28,
        ("LayoutSem", LOCAL): 356,
        ("LayoutSem", FUNCTION): 29,
    }
    if counts != expected:
        raise RuntimeError(f"frozen context counts drift: {counts}")
    by_id = {row["record_id"]: row for row in fresh_preflight["rows"]}
    for entry in entries:
        prior = by_id[entry["record_id"]]
        if (entry["request_sha256"] != prior["request_sha256"] or
                entry["frozen_request_scope"] != prior["context_scope"]):
            raise RuntimeError(f"preflight request/context drift: {entry['record_id']}")

    contract = {
        "schema": "seraph_external_ssq_context_fallback_contract_v0_4_0c",
        "status": "FROZEN_BEFORE_API",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_scope": "EXTERNAL_SSQ_DESIGN_ONLY",
        "design_records": 628,
        "type_fallback": 18,
        "layout_fallback": 29,
        "exceptional_fallback_total": 47,
        "interface_function_context": 28,
        "context_modes": [LOCAL, FUNCTION],
        "counts": {f"{m}|{mode}": n for (m, mode), n in sorted(counts.items())},
        "request_file_sha256": fresh_preflight["request_file_sha256"],
        "design_preflight_sha256": design.sha(preflight_path),
        "split_sha256": fresh_preflight["split_sha256"],
        "smoke_api_config_sha256": fresh_preflight["smoke_config_sha256"],
        "source_manifest_sha256": {str(path): design.sha(path) for path in SOURCE_FILES},
        "rules": {
            "TypeSem": "verified instruction-local when mapping_status=VERIFIED and instruction_usable=true; otherwise frozen function-level pseudo-C fallback",
            "InterfaceSem": "frozen function-level pseudo-C primary context",
            "LayoutSem": "verified local/instruction context when mapping_status=VERIFIED and instruction_usable=true; otherwise frozen function-level pseudo-C fallback",
            "forbidden_context_methods": ["nearest-address", "nearest-line", "string similarity", "unverified local span"],
            "request_content_mutation": "FORBIDDEN",
            "HOLDOUT_future_rule": "Apply these same three module rules to the already frozen HOLDOUT requests without changing mapping thresholds, request content, candidate order, prompt or API configuration; no HOLDOUT records or Gold are opened here.",
        },
        "pre_registered_design_gates": {
            "ALL_net_correction": ">0 for a module to be enabled",
            "VERIFIED_LOCAL_ONLY": "For TypeSem/LayoutSem: net_correction >=0, API R@1 >= Stage2 R@1, API MRR >= Stage2 MRR; conservative no-harm operationalization",
            "engineering": "Module valid rate >=0.95; every parseable attempt has exact candidate key set and no invented candidate",
            "candidate_legality": "100% among parseable attempts; an all-invalid module fails",
            "InterfaceSem": "ALL function-level net_correction >0 plus engineering and candidate legality",
            "READY_status": "At least one module enabled, 628/628 DESIGN requests completed, no global candidate-legality violation; otherwise BLOCKED_EXTERNAL_SSQ_PERFORMANCE",
        },
        "HOLDOUT_api_requests": 0,
        "HOLDOUT_evaluations": 0,
        "CALIBRATION_opened": 0,
        "FINAL_TEST_opened": 0,
        "records": entries,
    }
    target = design.OUT / "EXTERNAL_SSQ_CONTEXT_FALLBACK_CONTRACT.json"
    digest_path = design.OUT / "EXTERNAL_SSQ_CONTEXT_FALLBACK_CONTRACT.sha256"
    if target.exists() or digest_path.exists():
        raise FileExistsError("context fallback contract already frozen")
    data = (json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    digest = sha_bytes(data)
    write_exclusive(target, data)
    write_exclusive(digest_path, f"{digest}  {target.name}\n".encode())
    print(json.dumps({"contract": str(target), "sha256": digest,
                      "records": len(entries), "counts": contract["counts"],
                      "api_requests_sent": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
