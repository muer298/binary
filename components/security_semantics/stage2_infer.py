"""Gold-blind, frozen Task1 Stage-2 forward on fixed Juliet USG objects."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch


ROOT = Path("/data/chaoni/mjy/resym_exp")
OLD = Path(os.environ.get("SERAPH_SAMPLE_ROOT", str(ROOT / "training_runs/seraph_task2/v0_1_0")))
OUT = Path(os.environ.get("SERAPH_SEMANTIC_ROOT", str(ROOT / "training_runs/seraph_task2/v0_1_1")))
PACKAGE = ROOT / "packages/seraph_clean_joint_v0_2_10"
FREEZE = ROOT / "training_runs/seraph_local/final_freeze_v0_3_3c/FINAL_MODEL_MANIFEST_CLEAN_V2.json"
GLOBAL_SPLIT = ROOT / "packages/seraph_local_joint_v0_2_7_global_split/src"
SEEDS = (1337, 3407, 20260929)
MODULES = ("TypeSem", "InterfaceSem", "LayoutSem")
POLICY = {"TypeSem": "unified_llm_gate_clean", "InterfaceSem": "module_aware_clean",
          "LayoutSem": "unified_llm_gate_clean"}

sys.path.insert(0, str(GLOBAL_SPLIT))
from build_global_split import canonical_candidate  # noqa: E402
sys.path.remove(str(GLOBAL_SPLIT))
sys.modules.pop("common", None)
sys.path.insert(0, str(PACKAGE / "src"))
from model import CleanJointRanker  # noqa: E402
from train_clean_joint import FrozenLLMRepresentations, _tensor_payload  # noqa: E402
from common import sha256_json  # noqa: E402


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blind(sample):
    return "JULIET_" + hashlib.sha256(sample.encode()).hexdigest()[:20]


def blind_ref(sample, reference):
    if reference is None:
        return None
    prefix = sample + "::"
    if not str(reference).startswith(prefix):
        raise RuntimeError("unexpected USG identity")
    return blind(sample) + str(reference)[len(sample):]


def norm(value):
    return format(int(str(value or "0"), 16), "x")


def stable(*values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def load_checkpoint(item, arm, seed):
    if sha(Path(item["path"])) != item["sha256"]:
        raise RuntimeError("frozen Stage-2 checkpoint hash changed")
    payload = torch.load(item["path"], map_location="cpu", weights_only=False)
    if payload["arm"] != arm or payload["seed"] != seed:
        raise RuntimeError("frozen Stage-2 checkpoint identity changed")
    model = CleanJointRanker(payload["arm"], payload["seed"],
                             symbolic_dim=payload["symbolic_dim"],
                             llm_dim=payload["llm_dim"], bridge_dim=payload["bridge_dim"],
                             modules=payload["modules"],
                             apisem_enabled=payload["apisem_enabled"],
                             temperature=payload["temperature"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def value(module, node):
    attrs = node.get("attrs") or {}
    def bits(item):
        size = item.get("type_size")
        return size * 8 if isinstance(size, int) and size > 0 else "UNKNOWN"
    if module == "TypeSem":
        return {"type_category": attrs.get("return_type_category") or "UNKNOWN",
                "return_type": attrs.get("return_type") or "UNKNOWN",
                "parameters": [{"index": i, "type_category": p.get("type_category", "UNKNOWN"),
                                "bit_width": bits(p),
                                "signedness": p.get("signedness", "UNKNOWN"),
                                "is_pointer": p.get("is_pointer", "UNKNOWN")}
                               for i, p in enumerate(attrs.get("parameter_type_sem") or [])],
                "locals": [{"type_category": p.get("type_category", "UNKNOWN"),
                            "bit_width": bits(p),
                            "signedness": p.get("signedness", "UNKNOWN"),
                            "is_pointer": p.get("is_pointer", "UNKNOWN")}
                           for p in attrs.get("local_var_type_sem") or []]}
    if module == "InterfaceSem":
        return {"parameter_count": attrs.get("param_count", "UNKNOWN"),
                "parameters": [{"index": i, "role": p.get("role", "UNKNOWN"),
                                "type_category": p.get("type_category", "UNKNOWN")}
                               for i, p in enumerate(attrs.get("params") or [])],
                "return_role": (attrs.get("ret") or {}).get("role", "UNKNOWN"),
                "buffer_role": "UNKNOWN", "size_role": "UNKNOWN",
                "input_role": "UNKNOWN", "output_role": "UNKNOWN"}
    if module == "LayoutSem":
        return {"base_object": attrs.get("base_object") or "UNKNOWN",
                "offsets": attrs.get("observed_field_offsets") or [],
                "field_size": "UNKNOWN", "object_size": "UNKNOWN",
                "layout_confidence": attrs.get("confidence", "UNKNOWN")}
    raise ValueError(module)


def main():
    export = json.loads((OUT / "PSEUDOC_EXPORT_REPORT.json").read_text())
    expected = int(os.environ.get("SERAPH_EXPECTED_SAMPLES", str(export["sample_count"])))
    if export["sample_count"] != expected:
        raise RuntimeError("Juliet pseudo-C export sample count changed")
    failed = [r for r in export["reports"] if r["status"] != "EXPORTED"]
    if failed:
        raise RuntimeError(f"Juliet pseudo-C export failures: {len(failed)}")
    manifest = json.loads(FREEZE.read_text())
    if manifest["module_policy"] != POLICY or manifest["ensemble"]["seeds"] != list(SEEDS):
        raise RuntimeError("frozen model policy drift")
    if manifest["APISem_mode"] != "DETERMINISTIC_CONFIRMED_HARD":
        raise RuntimeError("frozen APISem mode drift")
    cfg = json.loads((PACKAGE / "config.json").read_text())
    if sha(Path(cfg["llm"]["stage1_adapter"]) / "adapter_model.safetensors") != manifest["artifacts"]["adapter"]["sha256"]:
        raise RuntimeError("frozen Stage-1 adapter hash changed")
    emb_root = OLD / "juliet_clean_hsym_forward"
    index = json.loads((emb_root / "node_embedding_index_clean_juliet.json").read_text())
    symbolic = _tensor_payload(emb_root / "node_embeddings_clean_juliet.pt")
    if len(symbolic.shape) != 2 or symbolic.shape[1] != 64 or symbolic.shape[0] != len(index) or not torch.isfinite(symbolic).all():
        raise RuntimeError("Juliet clean h_sym coverage/finite mismatch")
    node_file = OLD / "juliet_usg_final/usg_nodes.jsonl"
    sem_nodes = {}
    query_nodes = {}
    local_pool = collections.defaultdict(lambda: collections.defaultdict(list))
    sample_pool = collections.defaultdict(lambda: collections.defaultdict(list))
    for node in rows(node_file):
        kind = node["node_type"]
        nid = node["node_id"]
        if kind in ("InstrSem", "BlockSem"):
            query_nodes[nid] = kind
        elif kind in MODULES:
            sem_nodes[nid] = node
            key = (node["sample_id"], norm(node.get("function_entry_norm") or node.get("function_entry")), kind)
            canon = canonical_candidate(node)["canonical_semantic_id"]
            local_pool[key][canon].append(node)
            sample_pool[(node["sample_id"], kind)][canon].append(node)
    evidence = collections.defaultdict(list)
    for edge in rows(OLD / "juliet_usg_final/usg_edges.jsonl"):
        if edge.get("edge_kind") != "evidence" or not edge.get("source_resolved") or not edge.get("target_resolved"):
            continue
        src, dst = edge.get("source_node"), edge.get("target_node")
        if src in query_nodes and dst in sem_nodes:
            if query_nodes[src] == "InstrSem" or sem_nodes[dst]["node_type"] == "InterfaceSem":
                evidence[dst].append((src, edge["edge_id"]))
        elif dst in query_nodes and src in sem_nodes:
            if query_nodes[dst] == "InstrSem" or sem_nodes[src]["node_type"] == "InterfaceSem":
                evidence[src].append((dst, edge["edge_id"]))
    text_by_function = {}
    decomp = collections.Counter()
    for report in export["reports"]:
        payload = json.loads(Path(report["output"]).read_text(encoding="utf-8"))
        sample = report["sample_id"]
        for function in payload["functions"]:
            decomp[function["status"]] += 1
            if function["status"] == "OK" and function["function_text"].strip():
                text_by_function[(sample, norm(function["function_entry"]))] = function["function_text"]
    # Exactly one deterministic evidence-backed query per semantic object.
    prepared = []
    skipped = collections.Counter()
    for nid, node in sorted(sem_nodes.items()):
        sample = node["sample_id"]
        module = node["node_type"]
        entry = norm(node.get("function_entry_norm") or node.get("function_entry"))
        available = sorted({(src, eid) for src, eid in evidence.get(nid, []) if src in index})
        text = text_by_function.get((blind(sample), entry))
        if not available or not text or nid not in index:
            skipped[(module, "NO_VERIFIED_QUERY" if not available else "NO_PSEUDOC_OR_EMBEDDING")] += 1
            continue
        query, evidence_ref = min(available, key=lambda x: (query_nodes[x[0]] != "InstrSem", stable("v011-query", nid, *x)))
        local = [(min(nodes, key=lambda x: x["node_id"]), canon, "SAME_FUNCTION")
                 for canon, nodes in local_pool[(sample, entry, module)].items()]
        local.sort(key=lambda pair: stable("v011-local", query, pair[1]))
        local = local[:4]
        if len(local) < 2:
            negatives = [(min(nodes, key=lambda x: x["node_id"]), canon, "OTHER_FUNCTION_TEMPLATE")
                         for canon, nodes in sample_pool[(sample, module)].items()
                         if canon not in {x[1] for x in local}]
            negatives.sort(key=lambda pair: stable("v011-negative", query, pair[1]))
            local.extend(negatives[:2])
        local = [pair for pair in local if pair[0]["node_id"] in index]
        if len(local) < 2 or len({x[1] for x in local}) != len(local):
            skipped[(module, "NO_LEGAL_CANDIDATE_SET")] += 1
            continue
        local.sort(key=lambda pair: stable("v011-order", query, pair[1]))
        prepared.append({"sample": sample, "module": module, "entry": entry,
                         "object": node, "query": query, "evidence_ref": evidence_ref,
                         "candidates": local, "pseudo_c": text,
                         "record_id": stable("task2-v011-stage2", query, nid),
                         "instruction_mapping": {"instruction_usable": False}})
    if not prepared:
        raise RuntimeError("no legal Juliet Stage-2 forward queries")
    if not torch.cuda.is_available():
        raise RuntimeError("frozen Stage-1 forward requires available CUDA")
    free, _ = torch.cuda.mem_get_info(0)
    if free < 40 * 1024**3:
        raise RuntimeError("GPU 0 does not have 40 GiB free for frozen Stage-1 forward")
    llm = FrozenLLMRepresentations(cfg, "unified_llm_gate_clean", prepared, [],
                                   torch.device("cuda:0"), OUT / "representation_cache_v2",
                                   {"manifest_sha256": sha(FREEZE),
                                    "training_config_sha256": sha(PACKAGE / "config.json")})
    if llm.adapter_hash_before != llm.adapter_hash_after:
        raise RuntimeError("adapter changed during frozen forward")
    llm.release()
    models = {}
    for module in MODULES:
        models[module] = {}
        items = manifest["checkpoint_mapping"][module]
        for item in items:
            models[module][item["seed"]] = load_checkpoint(item, POLICY[module], item["seed"])
        if set(models[module]) != set(SEEDS):
            raise RuntimeError("missing frozen seed")
    dest = OUT / "juliet_stage2_semantics_v2.jsonl"
    if dest.exists():
        raise FileExistsError(dest)
    pending = dest.with_suffix(".pending")
    if pending.exists():
        raise FileExistsError(pending)
    counts = collections.Counter()
    with torch.no_grad(), pending.open("x", encoding="utf-8") as handle:
        for i, item in enumerate(prepared, 1):
            module = item["module"]
            candidates = item["candidates"]
            semantic_ids = [x[1] for x in candidates]
            query_h = symbolic[index[item["query"]]].unsqueeze(0)
            candidate_h = symbolic[[index[x[0]["node_id"]] for x in candidates]]
            llm_h = llm.function_cache[sha256_json(item["pseudo_c"])].unsqueeze(0)
            per_seed = {}
            probs = []
            for seed in SEEDS:
                model = models[module][seed]
                fused, _, _ = model.fuse(module, query_h, llm_h)
                logits, _ = model.score(module, fused, candidate_h)
                score = torch.softmax(logits[0].double(), dim=0).tolist()
                if not all(math.isfinite(v) for v in score):
                    raise RuntimeError("nonfinite Stage-2 score")
                per_seed[str(seed)] = dict(zip(semantic_ids, score))
                probs.append(score)
            mean = [sum(row[j] for row in probs) / len(SEEDS) for j in range(len(candidates))]
            winner = max(range(len(mean)), key=lambda j: (mean[j], semantic_ids[j]))
            winner_node, winner_semantic, winner_scope = candidates[winner]
            sample = item["sample"]
            applied = winner_scope == "SAME_FUNCTION"
            row = {"schema": "task2_v011_frozen_stage2_semantic_object",
                   "sample_id": blind(sample), "function_entry": item["entry"],
                   "object_id": blind_ref(sample, item["object"]["node_id"]),
                   "module": module, "query_id": blind_ref(sample, item["query"]),
                   "query_node_type": query_nodes[item["query"]],
                   "candidate_list": [{"object_id": blind_ref(x[0]["sample_id"], x[0]["node_id"]),
                                       "semantic_id": x[1], "scope": x[2]}
                                      for x in candidates],
                   "ensemble_scores": dict(zip(semantic_ids, mean)),
                   "per_seed_scores": per_seed,
                   "top1_prediction": winner_semantic, "confidence": mean[winner],
                   "semantic_value": value(module, winner_node) if applied else "UNKNOWN",
                   "semantic_value_status": "FROZEN_MODEL_SELECTED_SAME_FUNCTION_RAW_CANDIDATE" if applied else "UNKNOWN_OTHER_FUNCTION_TEMPLATE",
                   "evidence_refs": [blind_ref(sample, item["evidence_ref"])],
                   "mapping_level": "FUNCTION_LEVEL_PSEUDOC",
                   "model_arm": POLICY[module], "model_seeds": list(SEEDS)}
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            counts[(module, row["semantic_value_status"])] += 1
            if i % 1000 == 0:
                print(json.dumps({"event": "STAGE2_FORWARD", "completed": i,
                                  "total": len(prepared)}), flush=True)
    os.replace(pending, dest)
    # APISem remains a separate deterministic hard-evidence channel.
    api_count = 0
    with dest.open("a", encoding="utf-8") as handle:
        for summary in rows(OLD / "security_input_manifest.jsonl"):
            for api in summary.get("APISem_confirmed_hard") or []:
                name = str(api.get("api_name") or "UNKNOWN")
                row = {"schema": "task2_v011_frozen_stage2_semantic_object",
                       "sample_id": summary["sample_id"], "function_entry": summary["function_entry"],
                       "object_id": api["node_id"], "module": "APISem",
                       "candidate_list": [name], "ensemble_scores": "NOT_APPLICABLE_HARD_EVIDENCE",
                       "top1_prediction": name, "confidence": "CONFIRMED_HARD",
                       "semantic_value": {"api_identity": name, "semantic_action": "UNKNOWN",
                                          "source_effect": "UNKNOWN", "sink_effect": "UNKNOWN",
                                          "resource_effect": "UNKNOWN", "confirmed_hard": True},
                       "evidence_refs": api.get("call_instruction_nodes") or [],
                       "mapping_level": "EXACT_CALLSITE_OR_UNIQUE_SYMBOL_MATCH",
                       "model_arm": "DETERMINISTIC_CONFIRMED_HARD"}
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                api_count += 1
    report = {"schema": "task2_v011_frozen_stage2_forward_report", "status": "PASSED",
              "sample_count": expected, "semantic_objects_scored": len(prepared),
              "APISem_hard_objects": api_count,
              "query_rule": "one deterministic exact USG evidence edge per semantic object; no nearest/root query",
              "candidate_rule": "same-function canonical instances plus same-binary other-function templates if needed; no Gold",
              "other_function_template_is_not_object_truth": True,
              "mapping_level": "function-level pseudo-C only; no invented instruction mapping",
              "skipped": {str(k): v for k, v in skipped.items()},
              "decompile_status": dict(decomp),
              "status_counts": {str(k): v for k, v in counts.items()},
              "stage2_semantics_sha256": sha(dest),
              "stage1_adapter_sha256": llm.adapter_hash_before,
              "final_manifest_sha256": sha(FREEZE),
              "Gold_read": False, "backward_steps": 0, "optimizer_steps": 0}
    with (OUT / "JULIET_STAGE2_FORWARD_REPORT_V2.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"event": "STAGE2_FORWARD_COMPLETE", "scored": len(prepared),
                      "api_hard": api_count, "sha256": report["stage2_semantics_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
