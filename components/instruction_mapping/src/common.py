from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".pending.%d" % os.getpid())
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(pending, target)


def write_rows(path, values):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".pending.%d" % os.getpid())
    with pending.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(pending, target)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_entry(value):
    text = str(value or "").strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    return text.lstrip("0") or "0"


def address_int(value):
    if value in (None, ""):
        return None
    try:
        return int(str(value).strip().lower(), 16)
    except ValueError:
        return None


def quantiles(values):
    values = sorted(float(item) for item in values)
    if not values:
        return None
    def at(fraction):
        return values[round((len(values) - 1) * fraction)]
    mean = sum(values) / len(values)
    return {
        "n": len(values), "mean": mean,
        "std": (sum((item - mean) ** 2 for item in values) / len(values)) ** 0.5,
        "min": values[0], "max": values[-1],
        "p10": at(0.1), "p50": at(0.5), "p90": at(0.9),
    }


def centered_window(total_tokens, target_indices, max_length):
    indices = sorted(set(int(index) for index in target_indices))
    if not indices:
        return None
    target_start, target_end = indices[0], indices[-1] + 1
    if target_end - target_start > max_length:
        return None
    window_length = min(max_length, total_tokens)
    desired = (target_start + target_end - window_length) // 2
    window_start = max(0, min(desired, max(0, total_tokens - window_length)))
    if window_start > target_start:
        window_start = target_start
    if window_start + window_length < target_end:
        window_start = target_end - window_length
    window_start = max(0, min(window_start, max(0, total_tokens - window_length)))
    window_end = min(total_tokens, window_start + window_length)
    if not (window_start <= target_start and target_end <= window_end):
        return None
    return {
        "window_start": window_start,
        "window_end": window_end,
        "target_global_start": target_start,
        "target_global_end": target_end,
        "target_global_indices": indices,
        "target_local_start": target_start - window_start,
        "target_local_end": target_end - window_start,
        "target_local_indices": [index - window_start for index in indices],
    }
