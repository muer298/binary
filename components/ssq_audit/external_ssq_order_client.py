"""Gold-free, frozen candidate-order audit of External SSQ DESIGN requests.

The original request is reused verbatim except for candidate list order. No
HOLDOUT request, Gold, CALIBRATION, or FINAL TEST artifact is decoded here.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time


BASE = Path("/data/chaoni/mjy/resym_exp/training_runs/seraph_local")
PKG_C = Path("/data/chaoni/mjy/resym_exp/packages/seraph_external_ssq_v0_4_0c_design")
sys.path.insert(0, str(PKG_C))
import external_ssq_design_client as design  # noqa: E402

fixed = design.fixed
DESIGN = design.OUT
HOLDOUT = BASE / "external_ssq_v0_4_0d_holdout"
OUT = BASE / "external_ssq_v0_4_0e_order_audit"
OLD = design.OLD
SEED = "20261007"
CONTEXT_SHA = "354c9aa631cb337f76c4c23ffdef9edcee737c8ccf027bb970cbbca1320466ce"
DESIGN_POLICY_SHA = "5566e59cd1a2408b5db3f5c6ab36b035ec12627e4fcc18bce1b6f1bc95f00023"
HOLDOUT_DECISION_SHA = "fd80579ac3e9b9470399d2203bf9ecceb33269a2a85060c873b6c4c32bdc84b1"
HOLDOUT_METRICS_SHA = "e1427f85cc158e5061568095654e900df8778bf3d50a190315708d1fd5c38b00"
TARGETS = {"InterfaceSem": 28, "LayoutSem_5_order": 60, "LayoutSem_2_order": 40}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def ranking_key(module: str, rid: str) -> str:
    return hashlib.sha256(f"seraph-v0_4_0e|{SEED}|{module}|{rid}".encode()).hexdigest()


def order_indices(rid: str, count: int) -> list[tuple[int, ...]]:
    original = tuple(range(count))
    other = [p for p in itertools.permutations(range(count)) if p != original]
    other.sort(key=lambda p: (hashlib.sha256(f"{SEED}|{rid}|{','.join(map(str, p))}".encode()).hexdigest(), p))
    if count >= 3:
        if len(other) < 4:
            raise RuntimeError("not enough distinct candidate permutations")
        return [original, *other[:4]]
    if count == 2:
        return [original, (1, 0)]
    raise RuntimeError("candidate count below two")


def permuted_request(source: dict, ordered_ids: list[str]) -> dict:
    original = source["request"]
    by_id = {c["candidate_id"]: c for c in original["candidates"]}
    if len(by_id) != len(original["candidates"]) or set(ordered_ids) != set(by_id):
        raise RuntimeError("candidate ID membership drift")
    result = {k: v for k, v in original.items() if k != "candidates"}
    result["candidates"] = [by_id[cid] for cid in ordered_ids]
    if {k: v for k, v in result.items() if k != "candidates"} != {
        k: v for k, v in original.items() if k != "candidates"
    }:
        raise RuntimeError("non-candidate request content changed")
    if {c["candidate_id"]: c for c in result["candidates"]} != by_id:
        raise RuntimeError("candidate text or ID changed")
    return result


def select_rows() -> tuple[list[dict], dict]:
    # Read-only upstream contract checks; no DESIGN or HOLDOUT API is rerun.
    design.preflight()
    if sha(DESIGN / "EXTERNAL_SSQ_CONTEXT_FALLBACK_CONTRACT.json") != CONTEXT_SHA:
        raise RuntimeError("context fallback contract changed")
    if sha(DESIGN / "EXTERNAL_SSQ_MODULE_POLICY.json") != DESIGN_POLICY_SHA:
        raise RuntimeError("DESIGN module policy changed")
    if sha(HOLDOUT / "EXTERNAL_SSQ_HOLDOUT_DECISION.json") != HOLDOUT_DECISION_SHA:
        raise RuntimeError("HOLDOUT module decision changed")
    if sha(HOLDOUT / "EXTERNAL_SSQ_HOLDOUT_METRICS.json") != HOLDOUT_METRICS_SHA:
        raise RuntimeError("HOLDOUT historical metrics changed")
    decision = json.loads((HOLDOUT / "EXTERNAL_SSQ_HOLDOUT_DECISION.json").read_text(encoding="utf-8"))
    if (decision["status"] != "READY_FOR_EXTERNAL_SSQ_CALIBRATION" or
            {m: decision["modules"][m]["External_SSQ"] for m in
             ("TypeSem", "InterfaceSem", "LayoutSem")} != {
                "TypeSem": "DISABLED", "InterfaceSem": "ENABLED", "LayoutSem": "ENABLED"}):
        raise RuntimeError("frozen post-HOLDOUT policy changed")
    rows = design.design_rows()  # DESIGN requests only; no Gold or correctness.
    interface = [r for r in rows if r["module"] == "InterfaceSem"]
    layout_local = [r for r in rows if r["module"] == "LayoutSem" and
                    r["request"]["code"]["scope"] == "verified_instruction_local"]
    two = [r for r in layout_local if len(r["request"]["candidates"]) == 2]
    five = [r for r in layout_local if len(r["request"]["candidates"]) >= 3]
    if (len(interface), len(layout_local), len(two), len(five)) != (28, 356, 176, 180):
        raise RuntimeError("DESIGN eligibility distribution changed")
    if any(len(r["request"]["candidates"]) != 3 for r in interface):
        raise RuntimeError("InterfaceSem candidate count changed")
    interface.sort(key=lambda r: (ranking_key(r["module"], r["request_id"]), r["request_id"]))
    five.sort(key=lambda r: (ranking_key(r["module"], r["request_id"]), r["request_id"]))
    two.sort(key=lambda r: (ranking_key(r["module"], r["request_id"]), r["request_id"]))
    selected = [(r, "FIVE_DISTINCT_ORDERS") for r in interface]
    selected += [(r, "FIVE_DISTINCT_ORDERS") for r in five[:60]]
    selected += [(r, "TWO_ORDER_EXHAUSTIVE_DIAGNOSTIC") for r in two[:40]]
    if len({r["request_id"] for r, _ in selected}) != 128:
        raise RuntimeError("sample contains duplicate record IDs")
    return selected, {"InterfaceSem": len(interface), "LayoutSem_verified_local": len(layout_local),
                      "LayoutSem_two_candidate_eligible": len(two),
                      "LayoutSem_three_plus_candidate_eligible": len(five)}


def expected_manifest() -> tuple[list[dict], dict]:
    selected, pools = select_rows()
    result = []
    for row, cohort in selected:
        rid = row["request_id"]
        candidates = row["request"]["candidates"]
        ids = [c["candidate_id"] for c in candidates]
        orders = []
        for index, permutation in enumerate(order_indices(rid, len(ids))):
            ordered = [ids[i] for i in permutation]
            request = permuted_request(row, ordered)
            orders.append({"order_index": index, "variant_id": f"{rid}__order_{index}",
                           "candidate_ids": ordered,
                           "request_sha256": hashlib.sha256(canonical(request)).hexdigest()})
        if len({tuple(o["candidate_ids"]) for o in orders}) != len(orders):
            raise RuntimeError("duplicate candidate order in manifest")
        result.append({"record_id": rid, "module": row["module"], "cohort": cohort,
                       "candidate_count": len(ids), "original_candidate_ids": ids,
                       "source_request_sha256": hashlib.sha256(canonical(row["request"])).hexdigest(),
                       "context_scope": row["request"]["code"]["scope"], "orders": orders})
    result.sort(key=lambda r: (r["module"], r["record_id"]))
    if len(result) != 128 or sum(len(r["orders"]) for r in result) != 520:
        raise RuntimeError("manifest record/order count drift")
    counts = collections.Counter((r["module"], r["cohort"]) for r in result)
    if counts != {("InterfaceSem", "FIVE_DISTINCT_ORDERS"): 28,
                  ("LayoutSem", "FIVE_DISTINCT_ORDERS"): 60,
                  ("LayoutSem", "TWO_ORDER_EXHAUSTIVE_DIAGNOSTIC"): 40}:
        raise RuntimeError("manifest cohort count drift")
    return result, pools


def manifest_bytes(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode() for r in rows)


def prepare() -> None:
    rows, pools = expected_manifest()
    manifest_data = manifest_bytes(rows)
    digest = hashlib.sha256(manifest_data).hexdigest()
    OUT.mkdir(parents=False, exist_ok=False)
    manifest = OUT / "EXTERNAL_SSQ_ORDER_AUDIT_MANIFEST.jsonl"
    with manifest.open("xb") as handle:
        handle.write(manifest_data)
    with (OUT / "EXTERNAL_SSQ_ORDER_AUDIT_MANIFEST.sha256").open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {manifest.name}\n")
    write_new(OUT / "ORDER_AUDIT_PREFLIGHT.json", {
        "schema": "seraph_external_ssq_candidate_order_preflight_v0_4_0e",
        "status": "FROZEN_BEFORE_API",
        "seed": SEED,
        "selection_rule": "SHA256 of fixed seed, module and record_id; no Gold or correctness",
        "eligibility_rule": "InterfaceSem all 28 three-candidate DESIGN records; LayoutSem verified-local DESIGN only, 60 three-plus-candidate five-order primary plus 40 two-candidate exhaustive orientation diagnostic",
        "pool_counts": pools,
        "selected_records": 128,
        "variant_requests": 520,
        "interface_records": 28,
        "layout_five_order_records": 60,
        "layout_two_order_records": 40,
        "five_order_limitation": "Two-candidate records have only two mathematically distinct orders; they are separate exhaustive diagnostics, not claimed to have five orders",
        "manifest_sha256": digest,
        "design_context_contract_sha256": CONTEXT_SHA,
        "design_module_policy_sha256": DESIGN_POLICY_SHA,
        "holdout_decision_sha256": HOLDOUT_DECISION_SHA,
        "holdout_metrics_sha256": HOLDOUT_METRICS_SHA,
        "smoke_api_config_sha256": design.EXPECTED_CONTRACT_SHA,
        "prompt_sha256": fixed.PROMPT_SHA,
        "frozen_gates": {
            "candidate_legality": "100% among parseable attempts; no invented candidate",
            "valid_structured_rate": ">=0.95 for module and each LayoutSem cohort",
            "top1_identity_agreement": ">=0.90 for module and each LayoutSem cohort",
            "winner_flip_rate": "<=0.10 for module and each LayoutSem cohort",
            "mean_pairwise_spearman_rank_consistency": ">=0.90 for module and each LayoutSem cohort",
            "aggregation": "For each record, all unordered pairs of its distinct orders; missing/abstain top1 counts as disagreement and missing ranks as zero correlation. Module mean weights records equally.",
            "module_policy": "TypeSem stays DISABLED; each Interface/Layout module passes only if all its applicable module/cohort gates pass; READY if at least one module passes",
        },
        "HOLDOUT_Gold_read": False,
        "CALIBRATION_opened": False,
        "FINAL_TEST_V2_opened": False,
        "FINAL_TEST_V3_opened": False,
    })
    print(json.dumps({"prepared": True, "manifest_sha256": digest,
                      "records": 128, "variant_requests": 520, "pools": pools,
                      "TypeSem_API_calls": 0}, sort_keys=True))


def frozen_manifest() -> list[dict]:
    expected, _ = expected_manifest()
    path = OUT / "EXTERNAL_SSQ_ORDER_AUDIT_MANIFEST.jsonl"
    actual = path.read_bytes()
    digest = hashlib.sha256(actual).hexdigest()
    sidecar = (OUT / "EXTERNAL_SSQ_ORDER_AUDIT_MANIFEST.sha256").read_text(encoding="utf-8").strip()
    preflight = json.loads((OUT / "ORDER_AUDIT_PREFLIGHT.json").read_text(encoding="utf-8"))
    if sidecar != f"{digest}  {path.name}" or digest != preflight["manifest_sha256"]:
        raise RuntimeError("order audit manifest SHA256 changed")
    if actual != manifest_bytes(expected):
        raise RuntimeError("order audit manifest/request contract changed")
    return expected


def run() -> None:
    rows = frozen_manifest()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    print("API_KEY_AVAILABLE=" + ("true" if key else "false"), flush=True)
    if not key:
        raise SystemExit(2)
    source_by_id = {r["request_id"]: r for r in design.design_rows()}
    attempts_dir = OUT / "attempts"
    attempts_dir.mkdir(exist_ok=True)
    completed = 0
    index = 0
    for record in rows:
        source = source_by_id[record["record_id"]]
        for order in record["orders"]:
            index += 1
            target = attempts_dir / (order["variant_id"] + ".json")
            request = permuted_request(source, order["candidate_ids"])
            if hashlib.sha256(canonical(request)).hexdigest() != order["request_sha256"]:
                raise RuntimeError("permutation request hash mismatch")
            if target.exists():
                prior = json.loads(target.read_text(encoding="utf-8"))
                if prior["request_sha256"] != order["request_sha256"] or prior["prompt_sha256"] != fixed.PROMPT_SHA:
                    raise RuntimeError("resume checkpoint hash mismatch")
                completed += 1
                continue
            started = time.monotonic()
            attempts = []
            parsed = None
            for attempt_number in range(1, fixed.MAX_ATTEMPTS + 1):
                response, error = fixed.api_call(key, request)
                if error is not None:
                    attempt = {"attempt_number": attempt_number, **error,
                               "api_valid_response": False, "parse_valid": False,
                               "candidate_completeness": False, "candidate_legality": False,
                               "invented_candidate": False, "empty_response": False,
                               "usage": {}, "raw_content": None}
                else:
                    parsed, flags, invalid = fixed.validate(response["content"], request)
                    attempt = {"attempt_number": attempt_number,
                               "http_status": response["http_status"],
                               "api_valid_response": True,
                               "parse_valid": flags["parse_valid"],
                               "candidate_completeness": flags["candidate_complete"],
                               "candidate_legality": flags["candidate_legal"],
                               "invented_candidate": flags["invented_candidate"],
                               "empty_response": not flags["nonempty"],
                               "score_numeric_finite": flags["score_numeric_finite"],
                               "abstain_valid": flags["abstain_valid"],
                               "usage": response["usage"],
                               "finish_reason": response["finish_reason"],
                               "response_model": response["response_model"],
                               "latency_seconds": response["latency_seconds"],
                               "timeout": False,
                               "error_kind": invalid,
                               "raw_content": response["content"]}
                if key in json.dumps(attempt, ensure_ascii=False):
                    raise RuntimeError("secret appeared in response; refusing to save")
                attempts.append(attempt)
                if parsed is not None:
                    break
                if error is not None and not error["retryable"]:
                    break
                if attempt_number < fixed.MAX_ATTEMPTS:
                    time.sleep(2 ** attempt_number)
            final = attempts[-1]
            result = {"variant_id": order["variant_id"], "record_id": record["record_id"],
                      "module": record["module"], "cohort": record["cohort"],
                      "order_index": order["order_index"], "candidate_ids": order["candidate_ids"],
                      "request_sha256": order["request_sha256"],
                      "prompt_sha256": fixed.PROMPT_SHA,
                      "model": fixed.MODEL,
                      "valid": parsed is not None, "parsed": parsed,
                      "retry_count": len(attempts) - 1, "attempts": attempts,
                      "elapsed_seconds": round(time.monotonic() - started, 3),
                      "final_http_status": final.get("http_status"),
                      "final_error_kind": final.get("error_kind")}
            write_new(target, result)
            completed += 1
            print(json.dumps({"completed": completed, "total": 520,
                              "module": record["module"], "cohort": record["cohort"],
                              "valid": result["valid"], "retry_count": result["retry_count"],
                              "error_kind": result["final_error_kind"]}), flush=True)
            if final.get("http_status") in (400, 401, 403):
                print("STOP_UNRETRYABLE_API_STATUS", flush=True)
                print(json.dumps({"completed_ids": completed, "expected": 520}), flush=True)
                return
    print(json.dumps({"completed_ids": completed, "expected": 520,
                      "TypeSem_API_calls": 0}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
