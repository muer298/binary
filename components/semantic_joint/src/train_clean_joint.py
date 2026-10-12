from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from checkpointing import reload_checkpoint, save_checkpoint
from common import dump_json, jsonl_rows, load_json, sha256_file, sha256_json
from model import CleanJointRanker, masked_module_loss
from permutation_eval import evaluate_permutation_robustness


def _tensor_payload(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        return payload.float()
    if isinstance(payload, dict) and torch.is_tensor(payload.get("embeddings")):
        return payload["embeddings"].float()
    raise RuntimeError("unsupported clean embedding payload")


def _parameter_snapshot(model):
    return {name: value.detach().float().cpu().clone() for name, value in model.named_parameters()}


def _parameter_delta(before, model):
    return {name: float((value.detach().float().cpu() - before[name]).abs().max())
            for name, value in model.named_parameters()}


def _metrics(details):
    result = {}
    for module in sorted({row["module"] for row in details}):
        ranks = [row["rank"] for row in details if row["module"] == module]
        result[module] = {
            "n": len(ranks),
            "R@1": sum(rank <= 1 for rank in ranks) / len(ranks),
            "R@5": sum(rank <= 5 for rank in ranks) / len(ranks),
            "MRR": sum(1.0 / rank for rank in ranks) / len(ranks),
            "R@5_note": "low_discriminative_power_when_candidate_count_is_3_to_4",
        }
    return result


def _stat(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    ordered = sorted(values)

    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {"n": len(values), "mean": mean,
            "std_population": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
            "min": min(values), "max": max(values),
            "p10": percentile(0.10), "p50": percentile(0.50), "p90": percentile(0.90)}


class FrozenLLMRepresentations:
    """Precompute the audited fixed representation contract without gradients."""

    def __init__(self, cfg, arm, train, dev, device, cache_root, contract):
        self.cfg = cfg
        self.arm = arm
        self.device = device
        self.function_cache = {}
        self.instruction_cache = {}
        self.target_coverage_errors = 0
        self.model = None
        self.base = None
        self.tokenizer = None
        self.adapter_hash_before = None
        self.adapter_hash_after = None
        self.cache_root = Path(cache_root)
        self.contract = contract
        if arm == "symbolic_only_clean":
            return
        adapter = Path(cfg["llm"]["stage1_adapter"]) / "adapter_model.safetensors"
        self.adapter_hash_before = sha256_file(adapter)
        if ("first_batch_v0_1_3" in str(adapter)
                or self.adapter_hash_before != cfg["llm"]["expected_stage1_adapter_sha256"]):
            raise RuntimeError("contaminated Stage-1 adapter path/hash forbidden")
        self.adapter_hash_after = self.adapter_hash_before
        function_rows, instruction_rows = self._required_rows(train + dev)
        function_keys = {sha256_json(row["pseudo_c"]) for row in function_rows}
        instruction_keys = {row["record_id"] for row in instruction_rows}
        meta = {
            "schema": "seraph_v0_2_9_frozen_llm_representation_cache",
            "manifest_sha256": contract["manifest_sha256"],
            "training_config_sha256": contract["training_config_sha256"],
            "adapter_sha256": self.adapter_hash_before,
            "model_dir": cfg["llm"]["model_dir"],
            "hidden_dim": cfg["llm"]["hidden_dim"],
            "max_length": cfg["llm"]["max_length"],
        }
        function_path = self.cache_root / "function_level.pt"
        instruction_path = self.cache_root / "instruction_level.pt"
        function_ready = self._load_cache(function_path, meta, function_keys, self.function_cache)
        instruction_ready = (not instruction_rows) or self._load_cache(
            instruction_path, meta, instruction_keys, self.instruction_cache)
        if function_ready and instruction_ready:
            return
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        llm = cfg["llm"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm["model_dir"], local_files_only=True, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(
            llm["model_dir"], torch_dtype=torch.bfloat16, device_map={"": 0},
            local_files_only=True, low_cpu_mem_usage=True, use_safetensors=True,
            trust_remote_code=False)
        self.base.config.use_cache = False
        self.model = PeftModel.from_pretrained(
            self.base, llm["stage1_adapter"], is_trainable=False)
        self.model.eval()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("base model or Stage-1 adapter is not frozen")
        if not function_ready:
            self._precompute_functions(function_rows)
            self._save_cache(function_path, meta, self.function_cache)
        if not instruction_ready:
            self._precompute_instructions(instruction_rows)
            self._save_cache(instruction_path, meta, self.instruction_cache)
        self.adapter_hash_after = sha256_file(adapter)
        if self.adapter_hash_before != self.adapter_hash_after:
            raise RuntimeError("frozen Stage-1 adapter changed during representation extraction")

    def _required_rows(self, records):
        # The shared function cache deliberately covers every TRAIN/DEV record
        # so unified and module-aware arms have an identical immutable key set.
        function_rows = list(records)
        instruction_rows = []
        for row in records:
            if self.arm == "module_aware_clean" and row["module"] in ("TypeSem", "LayoutSem"):
                instruction_rows.append(row)
        return function_rows, instruction_rows

    def _load_cache(self, path, expected_meta, expected_keys, destination):
        if not path.is_file():
            return False
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("metadata") != expected_meta:
            raise RuntimeError("frozen LLM cache metadata mismatch: %s" % path)
        vectors = payload.get("vectors", {})
        if set(vectors) != set(expected_keys):
            raise RuntimeError("frozen LLM cache identity set mismatch: %s" % path)
        for key, value in vectors.items():
            if value is not None and (list(value.shape) != [self.cfg["llm"]["hidden_dim"]]
                                      or not torch.isfinite(value).all()):
                raise RuntimeError("invalid frozen LLM cache vector: %s" % key)
        destination.update(vectors)
        return True

    def _save_cache(self, path, metadata, vectors):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError("refusing to overwrite frozen LLM cache: %s" % path)
        pending = path.with_name(path.name + ".pending.%d" % os.getpid())
        torch.save({"metadata": metadata, "vectors": vectors}, pending)
        os.replace(pending, path)

    def _function_batch(self, texts):
        llm = self.cfg["llm"]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=llm["max_length"])
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.no_grad():
            output = self.model(**batch, output_hidden_states=True, use_cache=False,
                                return_dict=True)
            hidden = output.hidden_states[-1].float()
            mask = batch["attention_mask"].unsqueeze(-1)
            return ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu()

    def _precompute_functions(self, function_rows):
        unique_text = {}
        for row in function_rows:
            unique_text.setdefault(sha256_json(row["pseudo_c"]), row["pseudo_c"])
        keys = sorted(unique_text)
        for offset in range(0, len(keys), 8):
            batch_keys = keys[offset:offset + 8]
            vectors = self._function_batch([unique_text[key] for key in batch_keys])
            for key, vector in zip(batch_keys, vectors):
                self.function_cache[key] = vector

    def _precompute_instructions(self, instruction_rows):
        prepared = []
        ids_cache = {}
        for row in instruction_rows:
            key = row["record_id"]
            side = row["instruction_mapping"]
            if not side.get("instruction_usable", False):
                self.instruction_cache[key] = None
                continue
            text_key = sha256_json(row["pseudo_c"])
            if text_key not in ids_cache:
                ids_cache[text_key] = self.tokenizer(
                    row["pseudo_c"], add_special_tokens=False)["input_ids"]
            window = side["window"]
            values = ids_cache[text_key][int(window["window_start"]):int(window["window_end"])]
            targets = [int(value) for value in window["target_local_indices"]]
            if (not values or not targets or max(targets) >= len(values)
                    or len(values) > int(self.cfg["llm"]["max_length"])):
                self.target_coverage_errors += 1
                raise RuntimeError("audited instruction target is outside its fixed window")
            prepared.append((key, values, targets))
        pad_id = int(self.tokenizer.pad_token_id)
        for offset in range(0, len(prepared), 8):
            batch = prepared[offset:offset + 8]
            width = max(len(values) for _, values, _ in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=self.device)
            attention = torch.zeros((len(batch), width), dtype=torch.long, device=self.device)
            for index, (_, values, _) in enumerate(batch):
                input_ids[index, :len(values)] = torch.tensor(values, dtype=torch.long, device=self.device)
                attention[index, :len(values)] = 1
            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention,
                                    output_hidden_states=True, use_cache=False, return_dict=True)
                hidden = output.hidden_states[-1].float()
                for index, (key, _, targets) in enumerate(batch):
                    target = torch.tensor(targets, dtype=torch.long, device=self.device)
                    self.instruction_cache[key] = hidden[index, target].mean(0).cpu()

    def vector(self, row):
        if self.arm == "symbolic_only_clean":
            return None
        if self.arm == "unified_llm_gate_clean" or row["module"] in ("InterfaceSem", "APISem"):
            return self.function_cache[sha256_json(row["pseudo_c"])].to(self.device).unsqueeze(0)
        value = self.instruction_cache.get(row["record_id"])
        return None if value is None else value.to(self.device).unsqueeze(0)

    def release(self):
        self.model = None
        self.base = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _forward_record(model, symbolic, row, llm_vectors, device, candidate_ids=None):
    candidate_ids = list(row["candidate_ids"] if candidate_ids is None else candidate_ids)
    lookup = dict(zip(row["candidate_ids"], row["symbolic_candidate_indices"]))
    candidate_indices = [lookup[candidate] for candidate in candidate_ids]
    query = symbolic[int(row["symbolic_query_index"])].to(device).unsqueeze(0)
    candidates = symbolic[candidate_indices].to(device)
    fused, gate, gate_semantics = model.fuse(row["module"], query, llm_vectors.vector(row))
    logits, projected = model.score(row["module"], fused, candidates)
    gold_index = candidate_ids.index(row["gold_candidate_id"])
    return {
        "module": row["module"], "logits": logits[0], "gold_index": gold_index,
        "query": fused, "gold_candidate": projected[gold_index].unsqueeze(0),
        "gate": gate, "gate_semantics": gate_semantics,
        "candidate_ids": candidate_ids,
    }


def _validate(model, symbolic, rows, llm_vectors, device):
    details = []
    gates = collections.defaultdict(list)
    model.eval()
    with torch.no_grad():
        for row in rows:
            item = _forward_record(model, symbolic, row, llm_vectors, device)
            order = torch.argsort(item["logits"], descending=True).tolist()
            rank = 1 + order.index(item["gold_index"])
            gate = item["gate"]
            if gate is not None:
                gates[row["module"]].append(float(gate.mean()))
            details.append({
                "record_id": row["record_id"], "module": row["module"],
                "rank": rank,
                "top1_candidate_id": item["candidate_ids"][order[0]],
                "gold_candidate_id": row["gold_candidate_id"],
                "candidate_count": len(item["candidate_ids"]),
                "gate": None if gate is None else float(gate.mean()),
                "gate_semantics": item["gate_semantics"],
            })
    return details, {module: _stat(values) for module, values in gates.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    cfg = load_json(args.config)
    if not cfg["execution"]["authorized_after_v028_handoff"]:
        raise SystemExit("STOP: v0_2_9 execution is not authorized")
    if args.arm not in cfg["arms"] or not cfg["arms"][args.arm]["enabled"]:
        raise SystemExit("STOP: arm is not enabled")
    if args.seed not in cfg["fixed_seeds"]:
        raise SystemExit("STOP: seed is not one of the three frozen seeds")

    root = Path(cfg["output_root"])
    contract = load_json(root / "v0_2_10_contract.json")
    if contract["status"] != "READY_FOR_EXPLICITLY_AUTHORIZED_TRAINING":
        raise SystemExit("STOP: contract is not ready")
    if contract["split_sealing"]["test_evaluation_count"] != 0:
        raise RuntimeError("TEST evaluation counter is nonzero")
    if contract["split_sealing"]["calibration_evaluation_count"] != 0:
        raise RuntimeError("CALIBRATION evaluation counter is nonzero")
    if sha256_file(cfg["global_manifest"]) != contract["manifest_sha256"]:
        raise RuntimeError("global manifest changed after contract creation")
    if sha256_file(cfg["clean_embedding_path"]) != contract["clean_embedding_sha256"]:
        raise RuntimeError("clean embedding changed after contract creation")
    adapter_path = Path(cfg["llm"]["stage1_adapter"]) / "adapter_model.safetensors"
    adapter_sha = sha256_file(adapter_path)
    if ("first_batch_v0_1_3" in str(adapter_path)
            or adapter_sha != cfg["llm"]["expected_stage1_adapter_sha256"]
            or adapter_sha != contract["clean_stage1_adapter_sha256"]):
        raise RuntimeError("contaminated or changed Stage-1 adapter forbidden")
    if sha256_file(args.config) != contract["training_config_sha256"]:
        raise RuntimeError("training config changed after contract creation")
    artifact_paths = {
        "checkpoint": cfg["clean_alignment_checkpoint"],
        "embedding": cfg["clean_embedding_path"],
        "index": cfg["clean_embedding_index"],
        "schema": cfg["clean_feature_schema"],
        "vocab": cfg["clean_feature_vocab"],
    }
    if {name: sha256_file(path) for name, path in artifact_paths.items()} != contract["clean_artifact_sha256"]:
        raise RuntimeError("bound v0_2_8 clean artifact hash changed after contract creation")

    train = list(jsonl_rows(root / "train_manifest.jsonl"))
    dev = list(jsonl_rows(root / "dev_manifest.jsonl"))
    if any(row["global_split"] != "TRAIN" for row in train):
        raise RuntimeError("training manifest contains a non-TRAIN record")
    if any(row["global_split"] != "DEV" for row in dev):
        raise RuntimeError("development manifest contains a non-DEV record")
    if len(train) != contract["train_records"] or len(dev) != contract["dev_records"]:
        raise RuntimeError("contract record count changed")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("formal v0_2_9 training requires an explicitly selected free GPU")
    device = torch.device("cuda:0")
    free_before, total_vram = torch.cuda.mem_get_info(0)
    torch.cuda.reset_peak_memory_stats(0)

    symbolic = _tensor_payload(cfg["clean_embedding_path"])
    if symbolic.ndim != 2 or symbolic.shape[1] != 64 or not torch.isfinite(symbolic).all():
        raise RuntimeError("clean symbolic tensor contract failed at training entry")
    modules = list(cfg["enabled_modules"])
    if cfg["apisem_learned"]["enabled"]:
        modules.append("APISem")
    budget = cfg["training_budget"]
    model = CleanJointRanker(
        args.arm, args.seed, symbolic_dim=64, llm_dim=cfg["llm"]["hidden_dim"],
        bridge_dim=budget["bridge_dim"], modules=cfg["enabled_modules"],
        apisem_enabled=cfg["apisem_learned"]["enabled"],
        temperature=budget["candidate_temperature"]).to(device)
    expected_init = contract["initialization_hashes"]["%s/seed_%d" % (args.arm, args.seed)]
    if model.initialization_hashes() != expected_init:
        raise RuntimeError("deterministic parameter initialization hash mismatch")
    before = _parameter_snapshot(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget["learning_rate"],
                                  weight_decay=budget["weight_decay"])
    llm_vectors = FrozenLLMRepresentations(
        cfg, args.arm, train, dev, device, root / "representation_cache", contract)
    started = time.monotonic()
    history = []
    optimizer_steps = 0
    max_gradient = 0.0
    batch_size = int(budget["batch_size"])
    for epoch in range(int(budget["epochs"])):
        order = list(range(len(train)))
        random.Random(args.seed + epoch * 1000003).shuffle(order)
        for offset in range(0, len(order), batch_size):
            batch = [train[index] for index in order[offset:offset + batch_size]]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            items = [_forward_record(model, symbolic, row, llm_vectors, device) for row in batch]
            loss, per_module = masked_module_loss(items, modules, budget["align_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            gradients = [float(parameter.grad.detach().float().abs().max())
                         for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or max(gradients) <= 0 or not math.isfinite(max(gradients)):
                raise RuntimeError("missing or non-finite gradient")
            max_gradient = max(max_gradient, max(gradients))
            torch.nn.utils.clip_grad_norm_(model.parameters(), budget["gradient_clip"])
            optimizer.step()
            optimizer_steps += 1
            history.append({
                "epoch": epoch + 1, "optimizer_step": optimizer_steps,
                "loss": float(loss.detach()),
                "module_losses": {module: {key: (value if key == "n" else float(value.detach()))
                                            for key, value in values.items()}
                                  for module, values in per_module.items()},
            })

    deltas = _parameter_delta(before, model)
    if max(deltas.values(), default=0.0) <= 0:
        raise RuntimeError("ranker parameters did not update")
    details, gate_statistics = _validate(model, symbolic, dev, llm_vectors, device)
    validation = _metrics(details)

    def score_fn(row, ordered_ids):
        with torch.no_grad():
            item = _forward_record(model, symbolic, row, llm_vectors, device, ordered_ids)
        return item["logits"].detach().cpu().tolist()

    permutation = evaluate_permutation_robustness(
        dev, score_fn, args.seed,
        permutations=int(cfg["candidate_protocol"]["permutation_robustness_per_query"]))
    metadata = {
        "manifest_sha256": contract["manifest_sha256"],
        "candidate_order_sha256": contract["candidate_order_sha256"],
        "clean_embedding_sha256": contract["clean_embedding_sha256"],
        "training_config_sha256": contract["training_config_sha256"],
        "trainer_code_sha256": sha256_file(__file__),
        "parameter_initialization_hashes": expected_init,
        "test_evaluation_count": 0, "calibration_evaluation_count": 0,
    }
    final, checkpoint_manifest = save_checkpoint(
        root, args.arm, args.seed, model, optimizer, metadata)

    smoke_row = dev[0]
    smoke_llm = llm_vectors.vector(smoke_row)
    smoke_query = symbolic[smoke_row["symbolic_query_index"]].unsqueeze(0)
    smoke_candidates = symbolic[smoke_row["symbolic_candidate_indices"]]

    def factory(payload):
        return CleanJointRanker(
            payload["arm"], payload["seed"], payload["symbolic_dim"], payload["llm_dim"],
            payload["bridge_dim"], tuple(payload["modules"]), payload["apisem_enabled"],
            payload["temperature"])

    def sample_forward(reloaded):
        value = None if smoke_llm is None else smoke_llm.cpu()
        fused, gate, _ = reloaded.fuse(smoke_row["module"], smoke_query, value)
        logits, _ = reloaded.score(smoke_row["module"], fused, smoke_candidates)
        return (logits,) if gate is None else (logits, gate)

    reload_smoke = reload_checkpoint(final, factory, sample_forward)
    elapsed = time.monotonic() - started
    report = {
        "schema": "seraph_clean_joint_arm_seed_v0_2_10", "status": "DONE",
        "arm": args.arm, "seed": args.seed, "training_budget": budget,
        "training_config_sha256": contract["training_config_sha256"],
        "trainer_code_sha256": sha256_file(__file__),
        "train_records": len(train), "dev_records": len(dev),
        "optimizer_steps": optimizer_steps, "history": history,
        "module_loss_summary": {
            module: {
                "n_steps_present": sum(module in row["module_losses"] for row in history),
                "initial": next(row["module_losses"][module]["total"] for row in history
                                if module in row["module_losses"]),
                "final": next(row["module_losses"][module]["total"] for row in reversed(history)
                              if module in row["module_losses"]),
            } for module in modules
        },
        "validation": validation, "validation_details": details,
        "gate_statistics": gate_statistics, "permutation_robustness": permutation,
        "parameter_initialization_hashes": expected_init,
        "parameter_max_abs_delta": deltas, "max_gradient": max_gradient,
        "trainable_params": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "base_model_trainable_params": 0, "stage1_adapter_trainable_params": 0,
        "adapter_sha256_before": llm_vectors.adapter_hash_before,
        "adapter_sha256_after": llm_vectors.adapter_hash_after,
        "instruction_target_coverage_errors": llm_vectors.target_coverage_errors,
        "gpu_index_physical": args.gpu, "gpu_name": torch.cuda.get_device_name(0),
        "free_vram_before_bytes": int(free_before), "total_vram_bytes": int(total_vram),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "training_seconds": elapsed, "checkpoint_manifest": checkpoint_manifest,
        "reload_smoke": reload_smoke, "test_evaluation_count": 0,
        "calibration_evaluation_count": 0, "constraint_loss": cfg["constraint_loss"],
        "compromised_test_v1_opened": 0,
        "compromised_test_v1_evaluated": 0,
        "final_test_v2_opened": 0,
        "final_test_v2_predicted": 0,
        "final_test_v2_evaluated": 0,
        "apisem_learned": cfg["apisem_learned"],
        "nan_or_inf": False,
    }
    dump_json(final / "training_report.json", report)
    dump_json(final / "training_config.json", cfg)
    llm_vectors.release()
    print(json.dumps({"status": "DONE", "arm": args.arm, "seed": args.seed,
                      "validation": validation, "checkpoint": str(final)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
