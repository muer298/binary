from __future__ import annotations

import os
from pathlib import Path

import torch

from common import dump_json, sha256_file


def checkpoint_dir(output_root, arm, seed):
    return Path(output_root) / "arms" / arm / ("seed_%d" % int(seed))


def save_checkpoint(output_root, arm, seed, model, optimizer, metadata):
    final = checkpoint_dir(output_root, arm, seed)
    if final.exists():
        raise RuntimeError("refusing to overwrite checkpoint: %s" % final)
    pending = final.with_name(final.name + ".pending.%d" % os.getpid())
    if pending.exists():
        raise RuntimeError("pending checkpoint path already exists: %s" % pending)
    pending.mkdir(parents=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "arm": arm,
        "seed": int(seed),
        "symbolic_dim": model.symbolic_dim,
        "llm_dim": model.llm_dim,
        "bridge_dim": model.bridge_dim,
        "modules": list(model.modules_enabled),
        "apisem_enabled": model.apisem_enabled,
        "temperature": model.temperature,
        "metadata": metadata,
    }
    torch.save(payload, pending / "ranker.pt")
    digest = sha256_file(pending / "ranker.pt")
    manifest = {
        "schema": "seraph_clean_joint_checkpoint_v0_2_9",
        "arm": arm,
        "seed": int(seed),
        "checkpoint_sha256": digest,
        "parameter_initialization_hashes": metadata["parameter_initialization_hashes"],
        "manifest_sha256": metadata["manifest_sha256"],
        "candidate_order_sha256": metadata["candidate_order_sha256"],
        "clean_embedding_sha256": metadata["clean_embedding_sha256"],
        "training_config_sha256": metadata["training_config_sha256"],
        "trainer_code_sha256": metadata["trainer_code_sha256"],
        "test_evaluation_count": metadata.get("test_evaluation_count", 0),
        "calibration_evaluation_count": metadata.get("calibration_evaluation_count", 0),
    }
    dump_json(pending / "checkpoint_manifest.json", manifest)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(pending, final)
    return final, manifest


def reload_checkpoint(path, model_factory, sample_forward):
    path = Path(path)
    try:
        payload = torch.load(path / "ranker.pt", map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path / "ranker.pt", map_location="cpu")
    model = model_factory(payload)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        output = sample_forward(model)
    tensors = []
    if torch.is_tensor(output):
        tensors = [output]
    elif isinstance(output, (list, tuple)):
        tensors = [value for value in output if torch.is_tensor(value)]
    if not tensors or not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("checkpoint reload forward failed")
    return {
        "passed": True,
        "checkpoint_sha256": sha256_file(path / "ranker.pt"),
        "output_shapes": [list(value.shape) for value in tensors],
    }
