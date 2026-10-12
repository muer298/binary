#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production conflict-resolution core for Research Content 2, section 2.2.1.6.

This module implements the three operations required by the design guide:

* level 3 ``mask``: deterministic constraints intersect a persistent legal
  candidate mask.  Lower levels can never unmask a level-3 rejection;
* level 2 ``project``: high-confidence vector constraints make the smallest
  line-segment/half-space update available from the registered projector;
* level 1 ``blend``: statistical constraints make a bounded soft update.

It deliberately does not execute predicate or projector names from input.
Names are resolved through controlled registries.  Compatible USG schema aliases
are adapted at the boundary, and unsupported inputs become explicit audit
records rather than executable strings.

The resolver is pure with respect to the BindingStore.  The command-line
orchestrator first creates and verifies an output tensor, then converts each
changed result into an append-only UpdateEvent.  This separation prevents a
failed tensor write from leaving an audit event that points to no checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F


SCHEMA = "task2_mask_project_blend_resolver_v6_5"
CONSTRAINT_SCHEMA = "task2_compiled_constraint_v6_5"
RESULT_SCHEMA = "task2_node_resolution_v6_5"

LEVEL_STATISTICAL = 1
LEVEL_HIGH_CONFIDENCE = 2
LEVEL_DETERMINISTIC = 3

HARD_API_SOURCES = frozenset(("import_table", "symbol_table", "exact_signature"))
HARD_RULE_SOURCES = frozenset(
    (
        "deterministic_rule",
        "abi_rule",
        "calling_convention",
        "dwarf_exact",
        "validated_manual_rule",
    )
)
HIGH_NODE_TYPES = frozenset(("TypeSem", "InterfaceSem", "APISem", "LayoutSem"))

NESTED_KEYS = (
    "attrs",
    "attributes",
    "metadata",
    "meta",
    "context",
    "payload",
    "node",
    "edge",
    "record",
    "data",
    "constraint",
    "provenance",
)


class ResolutionError(RuntimeError):
    """Raised when a resolution contract is violated."""


class UnsupportedConstraint(ResolutionError):
    """Raised in strict mode for an unregistered constraint operation."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, finite_float(value, default)))


def _scalar(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("node_id", "id", "ref", "value", "name", "label"):
            if key in value and not isinstance(value.get(key), (Mapping, list, tuple, set)):
                text = str(value.get(key) or "").strip()
                if text:
                    return text
        return ""
    if isinstance(value, (list, tuple, set)):
        return ""
    return str(value or "").strip()


def mapping_layers(record: Mapping[str, Any], max_depth: int = 4) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    """Yield bounded nested mappings in deterministic order."""

    visited: Set[int] = set()

    def visit(path: str, value: Mapping[str, Any], depth: int) -> Iterable[Tuple[str, Mapping[str, Any]]]:
        marker = id(value)
        if marker in visited:
            return
        visited.add(marker)
        yield path, value
        if depth >= max_depth:
            return
        for key in NESTED_KEYS:
            nested = value.get(key)
            if isinstance(nested, Mapping):
                for item in visit("%s.%s" % (path, key), nested, depth + 1):
                    yield item

    for item in visit("record", record, 0):
        yield item


def first_value(record: Mapping[str, Any], aliases: Sequence[str], default: Any = None) -> Tuple[Any, str]:
    layers = list(mapping_layers(record))
    # Alias priority must beat wrapper depth.  An explicit nested api_role is
    # safer than a root-level generic role used for another purpose.
    for alias in aliases:
        for path, layer in layers:
            if alias in layer and layer.get(alias) not in (None, "", [], {}):
                return layer.get(alias), "%s.%s" % (path, alias)
    return default, ""


def list_value(record: Mapping[str, Any], aliases: Sequence[str]) -> Tuple[List[str], str]:
    value, path = first_value(record, aliases, [])
    if isinstance(value, str):
        parts = [item.strip() for item in re.split(r"[,|]", value) if item.strip()]
        return parts, path
    if isinstance(value, Mapping):
        return [str(key) for key, enabled in value.items() if bool(enabled)], path
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            text = _scalar(item)
            if text:
                result.append(text)
        return result, path
    text = _scalar(value)
    return ([text] if text else []), path


def normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def normalize_node_type(value: Any) -> str:
    token = normalize_token(value).replace("_", "")
    return {
        "typesem": "TypeSem",
        "type": "TypeSem",
        "interfacesem": "InterfaceSem",
        "interface": "InterfaceSem",
        "apisem": "APISem",
        "api": "APISem",
        "layoutsem": "LayoutSem",
        "layout": "LayoutSem",
        "instrsem": "InstrSem",
        "instruction": "InstrSem",
        "blocksem": "BlockSem",
        "cfgsem": "CFGSem",
        "dfgsem": "DFGSem",
    }.get(token, str(value or "unknown"))


def infer_node_type(node_id: Any) -> str:
    parts = str(node_id or "").split("::")
    return normalize_node_type(parts[-2]) if len(parts) >= 3 else "unknown"


def infer_sample_id(node_id: Any) -> str:
    text = str(node_id or "")
    return text.split("::", 1)[0] if "::" in text else ""


def normalize_source(value: Any) -> str:
    token = normalize_token(value)
    aliases = {
        "import": "import_table",
        "imports": "import_table",
        "importtable": "import_table",
        "dynsym": "import_table",
        "dynamic_symbol": "import_table",
        "plt": "import_table",
        "symbol": "symbol_table",
        "symbols": "symbol_table",
        "symboltable": "symbol_table",
        "symtab": "symbol_table",
        "external": "symbol_table",
        "signature": "exact_signature",
        "exact": "exact_signature",
        "exact_sig": "exact_signature",
        "abi": "abi_rule",
        "dwarf": "dwarf_exact",
    }
    return aliases.get(token, token)


def normalize_role(value: Any, record: Optional[Mapping[str, Any]] = None) -> str:
    token = normalize_token(value)
    aliases = {
        "ret": "return",
        "retval": "return",
        "return_value": "return",
        "result": "return",
        "output": "return",
        "parameter": "arg",
        "param": "arg",
        "argument": "arg",
    }
    token = aliases.get(token, token)
    match = re.search(r"(?:arg|param|parameter)_?(\d+)", token)
    if match:
        return "arg%d" % int(match.group(1))
    if token in ("return", "arg"):
        return token
    if record is not None:
        index, _ = first_value(record, ("param_idx", "parameter_index", "arg_index", "argument_index"), None)
        if index is not None:
            try:
                return "arg%d" % int(index)
            except (TypeError, ValueError):
                pass
        flag, _ = first_value(record, ("is_return", "return_value"), None)
        if flag is True or normalize_token(flag) in ("true", "yes", "return"):
            return "return"
        source_path, _ = first_value(record, ("source_path", "semantic_path", "path"), "")
        path_token = normalize_token(source_path)
        match = re.search(r"(?:arg|param|parameter)_?(\d+)", path_token)
        if match:
            return "arg%d" % int(match.group(1))
        if "return" in path_token:
            return "return"
    return token


def canonical_candidate(value: Any) -> str:
    text = str(value or "").strip().lower()
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return "unknown"
    if any(item in compact for item in ("pointer", "ptr", "void*", "char*", "buffer", "buf", "address")):
        if any(item in compact for item in ("struct", "record", "object")):
            return "struct_ptr"
        return "ptr"
    if any(item in compact for item in ("struct", "record", "layout", "aggregate")):
        return "struct"
    if any(item in compact for item in ("float", "double", "real")):
        return "float"
    if any(item in compact for item in ("bool", "boolean", "flag")):
        return "bool"
    if any(item in compact for item in ("int", "uint", "long", "short", "size_t", "ssize_t", "fd", "count", "length", "offset")):
        return "int"
    return normalize_token(text) or "unknown"


def candidate_matches(candidate: str, allowed: Sequence[str]) -> bool:
    if not allowed:
        return True
    raw = normalize_token(candidate)
    category = canonical_candidate(candidate)
    for item in allowed:
        if raw == normalize_token(item) or category == canonical_candidate(item):
            return True
    return False


def normalize_level(value: Any, strength: float = 0.0) -> int:
    token = normalize_token(value)
    if token in ("3", "level_3", "hard", "deterministic", "confirmed_hard"):
        return LEVEL_DETERMINISTIC
    if token in ("2", "level_2", "soft", "high_confidence", "confirmed_soft", "projection"):
        return LEVEL_HIGH_CONFIDENCE
    if token in ("1", "level_1", "statistical", "weak", "blend"):
        return LEVEL_STATISTICAL
    try:
        number = int(value)
        if number in (1, 2, 3):
            return number
    except (TypeError, ValueError):
        pass
    return LEVEL_HIGH_CONFIDENCE if strength >= 0.8 else LEVEL_STATISTICAL


def scores_to_logits(scores: Mapping[str, Any]) -> Tuple[List[str], torch.Tensor]:
    labels = [str(label) for label in scores]
    if not labels:
        labels = ["unknown"]
        return labels, torch.zeros(1, dtype=torch.float32)
    values = torch.tensor([finite_float(scores[label], 0.0) for label in scores], dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ResolutionError("Candidate scores contain NaN/Inf")
    if torch.all(values >= 0):
        total = float(values.sum().item())
        if total > 0:
            probabilities = values / values.sum()
            return labels, torch.log(probabilities.clamp_min(1e-12))
    return labels, values


def finite_softmax(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 1 or logits.numel() == 0:
        raise ResolutionError("Candidate logits must be a non-empty vector")
    finite = torch.isfinite(logits)
    if not bool(finite.any()):
        # The caller should normally take the explicit unresolved-hard path.
        # This finite fallback ensures no audit output contains NaN.
        return torch.full_like(logits, 1.0 / float(logits.numel()))
    safe = logits.clone()
    safe[~finite] = -1e30
    probabilities = F.softmax(safe, dim=0)
    if not torch.isfinite(probabilities).all():
        raise ResolutionError("Softmax produced NaN/Inf")
    return probabilities


def extract_candidate_scores(record: Mapping[str, Any]) -> Dict[str, float]:
    for aliases in (
        ("candidates", "candidate_scores", "candidate_distribution", "probabilities", "prediction_scores"),
        ("type_candidates", "pred_type_candidates_extended", "type_category_candidates"),
    ):
        value, _ = first_value(record, aliases, None)
        if isinstance(value, Mapping):
            result = {str(key): finite_float(score, 0.0) for key, score in value.items()}
            if result:
                return result
        if isinstance(value, list) and value:
            if all(isinstance(item, Mapping) for item in value):
                result = {}
                for item in value:
                    label, _ = first_value(item, ("label", "candidate", "name", "value", "type"), "")
                    score, _ = first_value(item, ("score", "confidence", "probability", "logit"), 0.0)
                    if _scalar(label):
                        result[_scalar(label)] = finite_float(score, 0.0)
                if result:
                    return result
            labels = [_scalar(item) for item in value if _scalar(item)]
            confidence, _ = first_value(
                record,
                ("type_confidence", "candidate_confidence", "candidate_probabilities", "scores"),
                [],
            )
            if labels:
                if isinstance(confidence, (list, tuple)):
                    return {
                        label: finite_float(confidence[index], 0.0) if index < len(confidence) else 0.0
                        for index, label in enumerate(labels)
                    }
                return {label: 1.0 / len(labels) for label in labels}
    selected, _ = first_value(
        record,
        ("selected_candidate", "pred_type", "predicted_type", "category", "type_category", "api_name"),
        "",
    )
    if _scalar(selected):
        confidence, _ = first_value(record, ("confidence", "prediction_confidence", "score"), 1.0)
        return {_scalar(selected): clamp01(confidence, 1.0)}
    return {}


def extract_state(record: Mapping[str, Any]) -> str:
    value, _ = first_value(record, ("status", "state", "candidate_status", "semantic_status"), "hypothesis")
    token = normalize_token(value)
    aliases = {
        "confirmed": "confirmed_soft",
        "soft": "confirmed_soft",
        "hard": "confirmed_hard",
        "unresolved": "hypothesis",
    }
    return aliases.get(token, token or "hypothesis")


def extract_confidence(record: Mapping[str, Any], scores: Mapping[str, float]) -> float:
    value, _ = first_value(record, ("confidence", "semantic_confidence", "prediction_confidence"), None)
    if value is not None:
        return clamp01(value)
    if not scores:
        return 0.0
    _, logits = scores_to_logits(scores)
    return float(finite_softmax(logits).max().item())


def rank_gate(record: Mapping[str, Any], require_rank1: bool = False, min_rr: Optional[float] = None) -> bool:
    """Strict rank filter: missing data fails whenever a filter is requested."""

    rank, _ = first_value(record, ("rank", "candidate_rank"), None)
    rr, _ = first_value(record, ("reciprocal_rank", "rr", "mrr_contribution"), None)
    if require_rank1:
        if rank is None:
            return False
        try:
            if int(rank) != 1:
                return False
        except (TypeError, ValueError):
            return False
    if min_rr is not None:
        if rr is None:
            return False
        try:
            if float(rr) < float(min_rr):
                return False
        except (TypeError, ValueError):
            return False
    return True


def api_signature_expectations(api_name: Any) -> Dict[str, str]:
    name = normalize_token(str(api_name or "").replace("@plt", "").replace(".plt", ""))
    exact: Dict[str, Dict[str, str]] = {
        "malloc": {"arg0": "int", "return": "ptr"},
        "calloc": {"arg0": "int", "arg1": "int", "return": "ptr"},
        "realloc": {"arg0": "ptr", "arg1": "int", "return": "ptr"},
        "free": {"arg0": "ptr"},
        "memcpy": {"arg0": "ptr", "arg1": "ptr", "arg2": "int", "return": "ptr"},
        "memmove": {"arg0": "ptr", "arg1": "ptr", "arg2": "int", "return": "ptr"},
        "memset": {"arg0": "ptr", "arg1": "int", "arg2": "int", "return": "ptr"},
        "strlen": {"arg0": "ptr", "return": "int"},
        "strcmp": {"arg0": "ptr", "arg1": "ptr", "return": "int"},
        "strncmp": {"arg0": "ptr", "arg1": "ptr", "arg2": "int", "return": "int"},
        "read": {"arg0": "int", "arg1": "ptr", "arg2": "int", "return": "int"},
        "write": {"arg0": "int", "arg1": "ptr", "arg2": "int", "return": "int"},
        "close": {"arg0": "int", "return": "int"},
    }
    for key, value in exact.items():
        if name == key or name.endswith("_" + key):
            return dict(value)
    return {}


@dataclass
class ConstraintSpec:
    constraint_id: str
    node_id: str
    level: int
    weight: float
    predicate_name: str = "always_true"
    projection_name: str = "towards_constraint"
    allowed_candidates: Tuple[str, ...] = ()
    denied_candidates: Tuple[str, ...] = ()
    candidate_bias: Dict[str, float] = field(default_factory=dict)
    constraint_vector: Optional[torch.Tensor] = None
    vector_threshold: float = 0.0
    source_node: str = ""
    target_node: str = ""
    source_edge_id: str = ""
    source_edge_class: str = "constraint"
    origin: str = "constraint_edge"
    hard_source: str = ""
    hard_source_verified: bool = False
    role: str = ""
    raw_hash: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def semantic_key(self) -> Tuple[Any, ...]:
        vector_hash = ""
        if self.constraint_vector is not None:
            vector_hash = hashlib.sha256(
                self.constraint_vector.detach().cpu().float().contiguous().numpy().tobytes()
            ).hexdigest()
        return (
            self.node_id,
            self.predicate_name,
            self.projection_name,
            tuple(sorted(self.allowed_candidates)),
            tuple(sorted(self.denied_candidates)),
            tuple(sorted((str(k), float(v)) for k, v in self.candidate_bias.items())),
            self.role,
            vector_hash,
        )

    def as_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        result = {
            "schema": CONSTRAINT_SCHEMA,
            "constraint_id": self.constraint_id,
            "node_id": self.node_id,
            "level": self.level,
            "weight": self.weight,
            "predicate_name": self.predicate_name,
            "projection_name": self.projection_name,
            "allowed_candidates": list(self.allowed_candidates),
            "denied_candidates": list(self.denied_candidates),
            "candidate_bias": dict(self.candidate_bias),
            "has_constraint_vector": self.constraint_vector is not None,
            "vector_threshold": self.vector_threshold,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "source_edge_id": self.source_edge_id,
            "source_edge_class": self.source_edge_class,
            "origin": self.origin,
            "hard_source": self.hard_source,
            "hard_source_verified": self.hard_source_verified,
            "role": self.role,
            "raw_hash": self.raw_hash,
        }
        if include_raw:
            result["raw"] = dict(self.raw)
        return result


Predicate = Callable[[Mapping[str, Any]], bool]
Projector = Callable[[torch.Tensor, ConstraintSpec], torch.Tensor]


class PredicateRegistry:
    def __init__(self) -> None:
        self._predicates: Dict[str, Predicate] = {}

    def register(self, name: str, predicate: Predicate) -> None:
        key = normalize_token(name)
        if not key or key in self._predicates:
            raise ValueError("Predicate already registered or empty: %s" % name)
        self._predicates[key] = predicate

    def contains(self, name: str) -> bool:
        return normalize_token(name) in self._predicates

    def evaluate(self, name: str, context: Mapping[str, Any]) -> bool:
        key = normalize_token(name)
        if key not in self._predicates:
            raise UnsupportedConstraint("Unregistered predicate: %s" % name)
        return bool(self._predicates[key](context))


class ProjectionRegistry:
    def __init__(self) -> None:
        self._projectors: Dict[str, Projector] = {}

    def register(self, name: str, projector: Projector) -> None:
        key = normalize_token(name)
        if not key or key in self._projectors:
            raise ValueError("Projector already registered or empty: %s" % name)
        self._projectors[key] = projector

    def contains(self, name: str) -> bool:
        return normalize_token(name) in self._projectors

    def project(self, name: str, embedding: torch.Tensor, constraint: ConstraintSpec) -> torch.Tensor:
        key = normalize_token(name)
        if key not in self._projectors:
            raise UnsupportedConstraint("Unregistered projector: %s" % name)
        result = self._projectors[key](embedding, constraint)
        if result.shape != embedding.shape or not torch.isfinite(result).all():
            raise ResolutionError("Projector %s produced invalid embedding" % name)
        return result


def cosine_score(embedding: torch.Tensor, target: Optional[torch.Tensor]) -> float:
    if target is None:
        return 1.0
    left = embedding.detach().float().reshape(-1)
    right = target.detach().to(left.device).float().reshape(-1)
    if left.numel() != right.numel() or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        return -1.0
    if float(left.norm().item()) <= 1e-12 or float(right.norm().item()) <= 1e-12:
        return -1.0
    return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=-1).item())


def project_towards_constraint(embedding: torch.Tensor, constraint: ConstraintSpec) -> torch.Tensor:
    target = constraint.constraint_vector
    if target is None:
        return embedding.clone()
    target = target.to(embedding.device, embedding.dtype).reshape_as(embedding)
    threshold = max(-1.0, min(1.0, float(constraint.vector_threshold)))
    if cosine_score(embedding, target) >= threshold:
        return embedding.clone()
    upper = clamp01(constraint.weight, 1.0)
    candidate = (1.0 - upper) * embedding + upper * target
    if cosine_score(candidate, target) < threshold:
        return candidate
    # Find the smallest change on the z->target line that reaches the
    # feasible cosine cap.  Bisection is bounded and deterministic.
    low = 0.0
    high = upper
    for _ in range(40):
        middle = 0.5 * (low + high)
        probe = (1.0 - middle) * embedding + middle * target
        if cosine_score(probe, target) >= threshold:
            high = middle
        else:
            low = middle
    return (1.0 - high) * embedding + high * target


def project_affine_halfspace(embedding: torch.Tensor, constraint: ConstraintSpec) -> torch.Tensor:
    normal = constraint.constraint_vector
    if normal is None:
        return embedding.clone()
    normal = normal.to(embedding.device, embedding.dtype).reshape_as(embedding)
    norm_sq = torch.dot(normal.reshape(-1), normal.reshape(-1))
    if float(norm_sq.item()) <= 1e-12:
        return embedding.clone()
    boundary = float(constraint.vector_threshold)
    current = torch.dot(normal.reshape(-1), embedding.reshape(-1))
    if float(current.item()) >= boundary:
        return embedding.clone()
    return embedding + ((boundary - current) / norm_sq) * normal


def project_nonnegative(embedding: torch.Tensor, constraint: ConstraintSpec) -> torch.Tensor:
    del constraint
    return embedding.clamp_min(0.0)


def project_unit_norm(embedding: torch.Tensor, constraint: ConstraintSpec) -> torch.Tensor:
    del constraint
    norm = embedding.norm()
    return embedding / norm.clamp_min(1e-12)


def build_default_predicates() -> PredicateRegistry:
    registry = PredicateRegistry()
    registry.register("always_true", lambda context: True)
    registry.register("finite_embedding", lambda context: bool(torch.isfinite(context["embedding"]).all()))
    registry.register(
        "selected_in_allowed",
        lambda context: candidate_matches(
            str(context.get("selected_candidate") or ""),
            context["constraint"].allowed_candidates,
        ),
    )
    registry.register(
        "selected_not_denied",
        lambda context: not candidate_matches(
            str(context.get("selected_candidate") or ""),
            context["constraint"].denied_candidates,
        ),
    )
    # These names are controlled aliases; the concrete allowed/denied values
    # remain data, while no arbitrary code is evaluated.
    for alias in (
        "type_compatible",
        "layout_type_compatible",
        "api_role_compatible",
        "malloc_returns_pointer",
        "candidate_allowed",
    ):
        registry.register(
            alias,
            lambda context: candidate_matches(
                str(context.get("selected_candidate") or ""),
                context["constraint"].allowed_candidates,
            ),
        )
    return registry


def build_default_projections() -> ProjectionRegistry:
    registry = ProjectionRegistry()
    registry.register("towards_constraint", project_towards_constraint)
    registry.register("nearest_feasible", project_towards_constraint)
    registry.register("affine_halfspace", project_affine_halfspace)
    registry.register("nonnegative", project_nonnegative)
    registry.register("unit_norm", project_unit_norm)
    registry.register("identity", lambda embedding, constraint: embedding.clone())
    return registry


def _predicate_context(
    labels: Sequence[str],
    probabilities: torch.Tensor,
    embedding: torch.Tensor,
    constraint: ConstraintSpec,
) -> Dict[str, Any]:
    selected_index = int(torch.argmax(probabilities).item())
    return {
        "labels": list(labels),
        "probabilities": probabilities,
        "selected_candidate": labels[selected_index],
        "embedding": embedding,
        "constraint": constraint,
    }


def make_reverse_vector_constraints(
    target_node: str,
    source_nodes: Sequence[str],
    target_vector: torch.Tensor,
    level: int = LEVEL_HIGH_CONFIDENCE,
    threshold: float = 0.7,
) -> List[ConstraintSpec]:
    """Create one high->low check for every source supporting a target.

    This helper makes the non-symmetric direction explicit: evidence is first
    aggregated by target, whereas the high-level target creates one reverse
    constraint check per supporting low-level object.
    """

    result = []
    for source_node in source_nodes:
        material = {
            "target_node": target_node,
            "source_node": source_node,
            "level": int(level),
            "threshold": float(threshold),
        }
        result.append(ConstraintSpec(
            constraint_id="reverse::" + sha256_json(material)[:24],
            node_id=str(source_node),
            source_node=str(target_node),
            target_node=str(source_node),
            level=int(level),
            weight=1.0,
            predicate_name="always_true",
            projection_name="towards_constraint",
            constraint_vector=target_vector.detach().clone(),
            vector_threshold=float(threshold),
            origin="reverse_high_to_low",
            raw_hash=sha256_json(material),
            raw=material,
        ))
    return result


def detect_projection_incompatibilities(constraints: Sequence[ConstraintSpec]) -> List[Dict[str, Any]]:
    vector_constraints = [
        item for item in constraints
        if item.level == LEVEL_HIGH_CONFIDENCE and item.constraint_vector is not None
    ]
    conflicts = []
    for left_index, left in enumerate(vector_constraints):
        for right in vector_constraints[left_index + 1:]:
            cosine = cosine_score(left.constraint_vector, right.constraint_vector)
            left_cap = math.acos(max(-1.0, min(1.0, left.vector_threshold)))
            right_cap = math.acos(max(-1.0, min(1.0, right.vector_threshold)))
            angle = math.acos(max(-1.0, min(1.0, cosine)))
            if left_cap + right_cap + 1e-8 < angle:
                conflicts.append({
                    "kind": "incompatible_high_confidence_projections",
                    "left_constraint_id": left.constraint_id,
                    "right_constraint_id": right.constraint_id,
                    "constraint_vector_cosine": cosine,
                    "required_cap_sum": left_cap + right_cap,
                    "actual_angle": angle,
                })
    return conflicts


def deduplicate_constraints(constraints: Sequence[ConstraintSpec]) -> Tuple[List[ConstraintSpec], Dict[str, Any]]:
    """Deduplicate by immutable ID and semantic key, keeping highest level."""

    by_id: Dict[str, ConstraintSpec] = {}
    identical_id_duplicates = 0
    for item in constraints:
        previous = by_id.get(item.constraint_id)
        if previous is None:
            by_id[item.constraint_id] = item
            continue
        if previous.raw_hash != item.raw_hash or previous.semantic_key() != item.semantic_key():
            raise ResolutionError("Conflicting duplicate constraint_id: %s" % item.constraint_id)
        identical_id_duplicates += 1

    by_semantic: Dict[Tuple[Any, ...], ConstraintSpec] = {}
    suppressed = []
    for item in by_id.values():
        key = item.semantic_key()
        previous = by_semantic.get(key)
        if previous is None:
            by_semantic[key] = item
            continue
        winner = max((previous, item), key=lambda value: (value.level, value.weight, value.constraint_id))
        loser = item if winner is previous else previous
        by_semantic[key] = winner
        suppressed.append({
            "constraint_id": loser.constraint_id,
            "kept_constraint_id": winner.constraint_id,
            "reason": "semantic_duplicate_lower_priority",
        })
    output = sorted(
        by_semantic.values(),
        key=lambda item: (-item.level, -item.weight, item.constraint_id),
    )
    return output, {
        "input_constraints": len(constraints),
        "output_constraints": len(output),
        "identical_id_duplicates": identical_id_duplicates,
        "semantic_duplicates_suppressed": len(suppressed),
        "suppressed": suppressed[:100],
    }


class ConstraintCompiler:
    """Compile observed USG records into controlled ConstraintSpecs."""

    def __init__(
        self,
        node_records: Mapping[str, Mapping[str, Any]],
        embedding_by_node: Mapping[str, torch.Tensor],
        high_strength_threshold: float = 0.8,
        default_vector_threshold: float = 0.7,
        require_model_rank1: bool = False,
        min_model_rr: Optional[float] = None,
    ) -> None:
        self.nodes = node_records
        self.embedding_by_node = embedding_by_node
        self.high_strength_threshold = float(high_strength_threshold)
        self.default_vector_threshold = float(default_vector_threshold)
        self.require_model_rank1 = bool(require_model_rank1)
        self.min_model_rr = min_model_rr
        self.audit: Counter = Counter()
        self.skipped: List[Dict[str, Any]] = []

    def node_type(self, node_id: str) -> str:
        record = self.nodes.get(node_id, {})
        value, _ = first_value(record, ("node_type", "semantic_type", "node_kind", "kind", "type"), "")
        return normalize_node_type(value or infer_node_type(node_id))

    def node_role(self, node_id: str) -> str:
        record = self.nodes.get(node_id, {})
        value, _ = first_value(record, ("api_role", "parameter_role", "argument_role", "role", "io_role"), "")
        return normalize_role(value, record)

    def api_source(self, node_id: str, edge: Mapping[str, Any]) -> str:
        node = self.nodes.get(node_id, {})
        value, _ = first_value(
            edge,
            ("api_source", "confirmation_source", "match_type", "source_kind", "provenance_source"),
            "",
        )
        if not _scalar(value):
            value, _ = first_value(
                node,
                ("api_source", "confirmation_source", "match_type", "source_kind", "provenance_source"),
                "",
            )
        return normalize_source(value)

    def _constraint_vector(self, raw: Mapping[str, Any], other_node: str) -> Optional[torch.Tensor]:
        value, _ = first_value(raw, ("constraint_vector", "projection_vector", "target_vector", "feasible_vector"), None)
        if isinstance(value, (list, tuple)) and value:
            vector = torch.tensor([finite_float(item, 0.0) for item in value], dtype=torch.float32)
            return vector if torch.isfinite(vector).all() else None
        vector = self.embedding_by_node.get(other_node)
        return vector.detach().clone() if vector is not None else None

    def _candidate_bias(self, raw: Mapping[str, Any]) -> Dict[str, float]:
        value, _ = first_value(raw, ("candidate_bias", "candidate_weights", "candidate_scores", "preferred_candidates"), {})
        if isinstance(value, Mapping):
            return {str(key): finite_float(score, 0.0) for key, score in value.items()}
        labels, _ = list_value(raw, ("preferred_candidates", "preferred_labels"))
        return {label: 1.0 for label in labels}

    def _make_spec(
        self,
        raw: Mapping[str, Any],
        edge_id: str,
        edge_class: str,
        source: str,
        target: str,
        subject: str,
        origin: str,
        forced_expected: str = "",
        forced_role: str = "",
        forced_level: Optional[int] = None,
        hard_source: str = "",
        source_edge_id: Optional[str] = None,
    ) -> ConstraintSpec:
        strength_value, _ = first_value(raw, ("strength", "weight", "confidence", "score", "probability"), 1.0)
        strength = clamp01(strength_value, 1.0)
        level_value, _ = first_value(raw, ("constraint_level", "level", "confirmation_level", "priority", "severity"), "")
        level = int(forced_level or normalize_level(level_value, strength))
        source_normalized = normalize_source(hard_source)
        source_type = self.node_type(source)
        hard_verified = source_normalized in HARD_RULE_SOURCES
        if source_type == "APISem" or origin.startswith("api_"):
            hard_verified = source_normalized in HARD_API_SOURCES
        if level == LEVEL_DETERMINISTIC and not hard_verified:
            level = LEVEL_HIGH_CONFIDENCE
            self.audit["unverified_hard_downgraded"] += 1

        allowed, _ = list_value(
            raw,
            ("allowed_candidates", "allowed_labels", "legal_candidates", "required_candidates"),
        )
        expected, _ = first_value(
            raw,
            ("expected_candidate", "expected_category", "required_type", "expected_type", "required_value"),
            "",
        )
        if forced_expected:
            expected = forced_expected
        if _scalar(expected):
            allowed.append(_scalar(expected))
        denied, _ = list_value(raw, ("denied_candidates", "forbidden_candidates", "illegal_candidates"))
        allowed = sorted(set(item for item in allowed if item))
        denied = sorted(set(item for item in denied if item))

        predicate, _ = first_value(raw, ("predicate_name", "predicate", "rule_name", "constraint_type"), "")
        predicate_name = normalize_token(predicate)
        if not predicate_name:
            if allowed:
                predicate_name = "selected_in_allowed"
            elif denied:
                predicate_name = "selected_not_denied"
            else:
                predicate_name = "always_true"

        projection, _ = first_value(raw, ("projection_name", "projector", "projection", "operator"), "")
        projection_name = normalize_token(projection) or "towards_constraint"
        threshold_value, _ = first_value(
            raw,
            ("vector_threshold", "satisfaction_threshold", "min_similarity", "margin"),
            self.default_vector_threshold,
        )
        role_value, _ = first_value(raw, ("api_role", "parameter_role", "argument_role", "role"), "")
        role = normalize_role(forced_role or role_value, raw)
        other = source if subject == target else target
        vector = self._constraint_vector(raw, other)
        material = {
            "edge_id": edge_id,
            "edge_class": edge_class,
            "source": source,
            "target": target,
            "subject": subject,
            "origin": origin,
            "level": level,
            "allowed": allowed,
            "denied": denied,
            "role": role,
            "predicate": predicate_name,
            "projection": projection_name,
            "raw_hash": sha256_json(raw),
        }
        constraint_id = str(edge_id or "constraint::" + sha256_json(material)[:28])
        return ConstraintSpec(
            constraint_id=constraint_id,
            node_id=subject,
            level=level,
            weight=strength,
            predicate_name=predicate_name,
            projection_name=projection_name,
            allowed_candidates=tuple(allowed),
            denied_candidates=tuple(denied),
            candidate_bias=self._candidate_bias(raw),
            constraint_vector=vector,
            vector_threshold=max(-1.0, min(1.0, finite_float(threshold_value, self.default_vector_threshold))),
            source_node=source,
            target_node=target,
            # Compiled constraint IDs may be synthetic. BindingStore lineage
            # must cite only the real backing graph edge in edge_catalog.
            source_edge_id=edge_id if source_edge_id is None else source_edge_id,
            source_edge_class=edge_class,
            origin=origin,
            hard_source=source_normalized,
            hard_source_verified=hard_verified,
            role=role,
            raw_hash=sha256_json(raw),
            raw=dict(raw),
        )

    def compile_edge(self, raw: Mapping[str, Any], resolved: Mapping[str, Any]) -> List[ConstraintSpec]:
        edge_id = str(resolved.get("edge_id") or "")
        edge_class = str(resolved.get("edge_class") or "relation")
        source = str(resolved.get("source_node") or "")
        target = str(resolved.get("target_node") or "")
        if not source or not target:
            self.audit["unresolved_endpoint"] += 1
            return []
        source_type = self.node_type(source)
        target_type = self.node_type(target)

        # Deterministic API constraints require both an audited source and an
        # exact role binding.  Missing roles are skipped instead of applying
        # every signature position to every adjacent node.
        api_node = source if source_type == "APISem" else (target if target_type == "APISem" else "")
        subject = target if api_node == source else source
        if api_node:
            api_record = self.nodes.get(api_node, {})
            api_name, _ = first_value(api_record, ("api_name", "api_target", "function_name", "name"), "")
            expectations = api_signature_expectations(api_name)
            edge_role, _ = first_value(raw, ("api_role", "parameter_role", "argument_role", "role"), "")
            role = normalize_role(edge_role, raw)
            if role != "return" and not re.fullmatch(r"arg\d+", role or ""):
                role = self.node_role(subject)
            if role != "return" and not re.fullmatch(r"arg\d+", role or ""):
                role = ""
            hard_source = self.api_source(api_node, raw)
            if expectations:
                if not role or role == "arg":
                    self.audit["api_role_missing"] += 1
                    self.skipped.append({
                        "edge_id": edge_id,
                        "reason": "api_role_missing",
                        "api_node": api_node,
                        "subject_node": subject,
                        "api_name": _scalar(api_name),
                    })
                    return []
                expected = expectations.get(role)
                if expected is None:
                    self.audit["api_role_not_in_signature"] += 1
                    return []
                spec = self._make_spec(
                    raw,
                    edge_id,
                    edge_class,
                    api_node,
                    subject,
                    subject,
                    origin="api_signature_edge",
                    forced_expected=expected,
                    forced_role=role,
                    forced_level=LEVEL_DETERMINISTIC,
                    hard_source=hard_source,
                )
                self.audit["api_role_bound"] += 1
                if spec.level == LEVEL_DETERMINISTIC:
                    self.audit["api_hard_verified"] += 1
                return [spec]

        if edge_class == "constraint":
            applies, _ = list_value(raw, ("applies_to", "constrained_nodes", "subject_nodes"))
            subjects = [item for item in applies if item in self.nodes or item in self.embedding_by_node]
            if not subjects:
                subject_value, _ = first_value(raw, ("subject_node", "constrained_node", "node_id"), "")
                subjects = [_scalar(subject_value)] if _scalar(subject_value) else [target]
            bidirectional, _ = first_value(raw, ("bidirectional", "symmetric", "both_directions"), False)
            if bool(bidirectional):
                subjects = sorted(set(subjects + [source, target]))
            hard_source, _ = first_value(raw, ("hard_source", "constraint_source", "source_kind", "provenance_source"), "")
            specs = [
                self._make_spec(
                    raw,
                    "%s::%s" % (edge_id, subject) if len(subjects) > 1 else edge_id,
                    edge_class,
                    source,
                    target,
                    subject,
                    origin="constraint_edge",
                    hard_source=_scalar(hard_source),
                    source_edge_id=edge_id,
                )
                for subject in subjects if subject
            ]
            self.audit["constraint_edges_compiled"] += len(specs)
            return specs

        # Evidence edges become level-2 projection or level-1 blend hints only
        # when their type has a defensible type/layout interpretation.
        evidence_type, _ = first_value(raw, ("evidence_type", "edge_type", "relation_type", "type"), "")
        evidence = normalize_token(evidence_type)
        expected_map = {
            "deref_chain": "ptr",
            "dereference_chain": "ptr",
            "fixed_offset": "struct",
            "field_offset": "struct",
        }
        expected = expected_map.get(evidence, "")
        if expected:
            strength = clamp01(resolved.get("strength"), 0.0)
            level = LEVEL_HIGH_CONFIDENCE if strength >= self.high_strength_threshold else LEVEL_STATISTICAL
            # A model-derived high-confidence item must pass requested rank
            # filters; missing rank/rr never passes silently.
            model_marker, _ = first_value(raw, ("model_alignment", "rank", "reciprocal_rank"), None)
            rank_filter_active = self.require_model_rank1 or self.min_model_rr is not None
            if (rank_filter_active or model_marker is not None) and not rank_gate(
                raw, self.require_model_rank1, self.min_model_rr
            ):
                level = LEVEL_STATISTICAL
                self.audit["model_rank_gate_downgraded"] += 1
            spec = self._make_spec(
                raw,
                "derived::%s::%s" % (edge_id, expected),
                edge_class,
                source,
                target,
                target,
                origin="evidence_derived_constraint",
                forced_expected=expected if level == LEVEL_HIGH_CONFIDENCE else "",
                forced_level=level,
                source_edge_id=edge_id,
            )
            if level == LEVEL_STATISTICAL and not spec.candidate_bias:
                spec.candidate_bias = {expected: max(strength, 0.05)}
            self.audit["evidence_constraints_compiled"] += 1
            return [spec]
        self.audit["non_constraint_edge_skipped"] += 1
        return []

    def compile_api_self_constraints(self, selected_node_ids: Iterable[str]) -> List[ConstraintSpec]:
        result = []
        for node_id in selected_node_ids:
            if self.node_type(node_id) != "APISem":
                continue
            node = self.nodes.get(node_id, {})
            api_name, _ = first_value(node, ("api_name", "api_target", "function_name", "name"), "")
            source = self.api_source(node_id, node)
            if not _scalar(api_name) or source not in HARD_API_SOURCES:
                continue
            raw = {
                "edge_kind": "constraint",
                "constraint_type": "exact_api_identity",
                "source_node": node_id,
                "target_node": node_id,
                "allowed_candidates": [_scalar(api_name)],
                "predicate_name": "selected_in_allowed",
                "constraint_level": 3,
                "hard_source": source,
            }
            result.append(self._make_spec(
                raw,
                "api-self::" + sha256_json(raw)[:24],
                "constraint",
                node_id,
                node_id,
                node_id,
                origin="api_self_identity",
                forced_expected=_scalar(api_name),
                forced_level=LEVEL_DETERMINISTIC,
                hard_source=source,
                source_edge_id="",
            ))
            self.audit["api_self_constraints"] += 1
        return result


class ConflictResolver:
    def __init__(
        self,
        predicates: Optional[PredicateRegistry] = None,
        projections: Optional[ProjectionRegistry] = None,
        max_iterations: int = 10,
        epsilon: float = 1e-5,
        confirmation_threshold: float = 0.85,
        max_statistical_alpha: float = 0.35,
        stable_rounds: int = 2,
        fail_on_unsupported: bool = False,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.predicates = predicates or build_default_predicates()
        self.projections = projections or build_default_projections()
        self.max_iterations = int(max_iterations)
        self.epsilon = max(0.0, float(epsilon))
        self.confirmation_threshold = clamp01(confirmation_threshold, 0.85)
        self.max_statistical_alpha = clamp01(max_statistical_alpha, 0.35)
        self.stable_rounds = max(1, int(stable_rounds))
        self.fail_on_unsupported = bool(fail_on_unsupported)

    def _allowed_mask(self, labels: Sequence[str], constraint: ConstraintSpec) -> torch.Tensor:
        mask = torch.ones(len(labels), dtype=torch.bool)
        if constraint.allowed_candidates:
            mask = torch.tensor(
                [candidate_matches(label, constraint.allowed_candidates) for label in labels],
                dtype=torch.bool,
            )
        if constraint.denied_candidates:
            denied = torch.tensor(
                [candidate_matches(label, constraint.denied_candidates) for label in labels],
                dtype=torch.bool,
            )
            mask &= ~denied
        return mask

    def _symbolic_satisfied(
        self,
        labels: Sequence[str],
        probabilities: torch.Tensor,
        embedding: torch.Tensor,
        constraint: ConstraintSpec,
    ) -> Tuple[bool, str]:
        if not self.predicates.contains(constraint.predicate_name):
            if self.fail_on_unsupported:
                raise UnsupportedConstraint("Unregistered predicate: %s" % constraint.predicate_name)
            return False, "unsupported_predicate"
        context = _predicate_context(labels, probabilities, embedding, constraint)
        return self.predicates.evaluate(constraint.predicate_name, context), ""

    def _project_candidate_distribution(
        self,
        labels: Sequence[str],
        logits: torch.Tensor,
        legal_mask: torch.Tensor,
        constraint: ConstraintSpec,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Project candidate mass toward the level-2 symbolic feasible set.

        Vector projection alone cannot repair a symbolic candidate violation
        when no classifier is re-run inside the resolver.  This bounded
        simplex projection therefore moves the minimum group mass needed by
        the constraint while preserving relative probability inside and
        outside the feasible group.  It is not a level-3 mask: alternatives
        retain non-zero probability and later evidence can still revise them.
        """

        if not constraint.allowed_candidates and not constraint.denied_candidates:
            return logits, {"applied": False, "reason": "no_symbolic_candidate_set"}
        feasible = legal_mask.clone()
        if constraint.allowed_candidates:
            feasible &= torch.tensor(
                [candidate_matches(label, constraint.allowed_candidates) for label in labels],
                dtype=torch.bool,
            )
        if constraint.denied_candidates:
            feasible &= ~torch.tensor(
                [candidate_matches(label, constraint.denied_candidates) for label in labels],
                dtype=torch.bool,
            )
        if not bool(feasible.any()):
            return logits, {"applied": False, "reason": "no_feasible_existing_candidate"}
        probabilities = finite_softmax(logits.masked_fill(~legal_mask, float("-inf")))
        before_mass = float(probabilities[feasible].sum().item())
        # Level 2 is high-confidence but non-deterministic.  More than half of
        # the mass makes a feasible candidate preferred, while the cap below
        # one preserves uncertainty.
        desired_mass = min(0.999999, max(0.500001, clamp01(constraint.weight, 0.8)))
        if before_mass + 1e-9 >= desired_mass:
            return logits, {
                "applied": False,
                "reason": "candidate_mass_already_feasible",
                "before_mass": before_mass,
                "after_mass": before_mass,
            }
        outside = legal_mask & ~feasible
        projected = probabilities.clone()
        projected[feasible] *= desired_mass / max(before_mass, 1e-12)
        outside_mass = float(projected[outside].sum().item())
        if bool(outside.any()) and outside_mass > 0:
            projected[outside] *= (1.0 - desired_mass) / outside_mass
        elif bool(outside.any()):
            projected[outside] = (1.0 - desired_mass) / float(outside.sum().item())
        projected[~legal_mask] = 0.0
        projected /= projected.sum().clamp_min(1e-12)
        updated = torch.log(projected.clamp_min(1e-12))
        updated[~legal_mask] = logits[~legal_mask]
        return updated, {
            "applied": True,
            "reason": "bounded_simplex_projection",
            "before_mass": before_mass,
            "after_mass": float(projected[feasible].sum().item()),
            "desired_mass": desired_mass,
            "feasible_candidates": [
                labels[index] for index, allowed in enumerate(feasible) if bool(allowed)
            ],
        }

    def _detect(
        self,
        labels: Sequence[str],
        logits: torch.Tensor,
        embedding: torch.Tensor,
        constraints: Sequence[ConstraintSpec],
        hard_mask: torch.Tensor,
        unresolved_hard: bool,
    ) -> List[Dict[str, Any]]:
        probabilities = finite_softmax(logits.masked_fill(~hard_mask, float("-inf")))
        conflicts = []
        for constraint in constraints:
            symbolic_ok, unsupported = self._symbolic_satisfied(
                labels, probabilities, embedding, constraint
            )
            vector_score = cosine_score(embedding, constraint.constraint_vector)
            vector_ok = (
                constraint.constraint_vector is None
                or vector_score + 1e-8 >= constraint.vector_threshold
            )
            hard_empty = constraint.level == LEVEL_DETERMINISTIC and unresolved_hard
            if not symbolic_ok or not vector_ok or hard_empty:
                conflicts.append({
                    "node_id": constraint.node_id,
                    "constraint_id": constraint.constraint_id,
                    "level": constraint.level,
                    "weight": constraint.weight,
                    "symbolic_violation": not symbolic_ok,
                    "vector_violation": max(0.0, constraint.vector_threshold - vector_score),
                    "vector_score": vector_score,
                    "unsupported": unsupported,
                    "hard_candidate_set_empty": hard_empty,
                    "origin": constraint.origin,
                })
        return conflicts

    def resolve(
        self,
        node_id: str,
        candidate_scores: Mapping[str, Any],
        embedding: torch.Tensor,
        constraints: Sequence[ConstraintSpec],
        initial_state: str = "hypothesis",
        initial_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        if embedding.ndim != 1 or embedding.numel() == 0:
            raise ResolutionError("Embedding for %s must be a non-empty vector" % node_id)
        if not torch.isfinite(embedding).all():
            raise ResolutionError("Embedding for %s contains NaN/Inf" % node_id)
        labels, base_logits = scores_to_logits(candidate_scores)
        logits = base_logits.clone()
        current_embedding = embedding.detach().clone().float()
        constraints, dedupe_audit = deduplicate_constraints(constraints)
        for item in constraints:
            if item.node_id != node_id:
                raise ResolutionError(
                    "Constraint %s belongs to %s, not %s"
                    % (item.constraint_id, item.node_id, node_id)
                )

        hard_mask = torch.ones(len(labels), dtype=torch.bool)
        hard_constraints = [item for item in constraints if item.level == LEVEL_DETERMINISTIC]
        high_constraints = [item for item in constraints if item.level == LEVEL_HIGH_CONFIDENCE]
        soft_constraints = [item for item in constraints if item.level == LEVEL_STATISTICAL]

        # A previously hard-confirmed decision becomes an additional level-3
        # lock.  It can conflict with a new deterministic rule, but no lower
        # level can silently overwrite it.
        initial_probabilities = finite_softmax(logits)
        initial_selected = labels[int(torch.argmax(initial_probabilities).item())]
        level_lock = 0
        if normalize_token(initial_state) == "confirmed_hard":
            hard_mask &= torch.tensor([label == initial_selected for label in labels], dtype=torch.bool)
            level_lock = LEVEL_DETERMINISTIC

        hard_trace = []
        for constraint in hard_constraints:
            local_mask = self._allowed_mask(labels, constraint)
            hard_mask &= local_mask
            level_lock = LEVEL_DETERMINISTIC
            hard_trace.append({
                "constraint_id": constraint.constraint_id,
                "allowed_before_intersection": [labels[i] for i, value in enumerate(local_mask) if bool(value)],
                "remaining_after_intersection": [labels[i] for i, value in enumerate(hard_mask) if bool(value)],
            })

        unresolved_hard = not bool(hard_mask.any())
        # Reporting must remain finite even when the legal level-3
        # intersection is empty.  ``probability_mask`` is only a numerical
        # fallback; ``hard_mask`` stays empty in the audit and lower levels are
        # skipped, so this never pretends an illegal candidate became legal.
        probability_mask = hard_mask.clone()
        new_hypotheses = []
        if unresolved_hard:
            expected = sorted(set(
                candidate
                for constraint in hard_constraints
                for candidate in constraint.allowed_candidates
            ))
            hypothesis_material = {
                "node_id": node_id,
                "kind": "unresolved_hard_constraint",
                "expected_candidates": expected,
                "constraint_ids": [item.constraint_id for item in hard_constraints],
            }
            new_hypotheses.append({
                "hypothesis_id": "hypothesis::" + sha256_json(hypothesis_material)[:28],
                **hypothesis_material,
                "status": "hypothesis",
            })
            # Preserve the original finite distribution.  Never softmax an
            # all--inf vector.
            probability_mask = torch.ones(len(labels), dtype=torch.bool)

        projection_incompatibilities = detect_projection_incompatibilities(high_constraints)
        force_max_iterations = bool(projection_incompatibilities)
        history = []
        historical_conflicts = []
        previous_active: Optional[Set[str]] = None
        previous_probabilities = initial_probabilities.clone()
        stable_count = 0
        stop_reason = "max_iterations"

        for iteration in range(1, self.max_iterations + 1):
            before_embedding = current_embedding.clone()
            before_probabilities = finite_softmax(logits.masked_fill(~probability_mask, float("-inf")))
            operations = []

            # Level 3: the immutable mask is reapplied every round.
            masked_logits = logits.masked_fill(~probability_mask, float("-inf"))
            operations.append({
                "operation": "mask",
                "level": 3,
                "remaining_candidates": [labels[i] for i, value in enumerate(hard_mask) if bool(value)],
                "unresolved_hard": unresolved_hard,
            })

            # Level 2: controlled nearest-feasible projections.
            for constraint in ([] if unresolved_hard else high_constraints):
                if not self.projections.contains(constraint.projection_name):
                    if self.fail_on_unsupported:
                        raise UnsupportedConstraint(
                            "Unregistered projector: %s" % constraint.projection_name
                        )
                    operations.append({
                        "operation": "project_skipped",
                        "level": 2,
                        "constraint_id": constraint.constraint_id,
                        "reason": "unsupported_projector",
                    })
                    continue
                old = current_embedding.clone()
                current_embedding = self.projections.project(
                    constraint.projection_name, current_embedding, constraint
                )
                logits, symbolic_projection = self._project_candidate_distribution(
                    labels, logits, probability_mask, constraint
                )
                operations.append({
                    "operation": "project",
                    "level": 2,
                    "constraint_id": constraint.constraint_id,
                    "projector": constraint.projection_name,
                    "delta": float(torch.norm(current_embedding - old).item()),
                    "satisfaction": cosine_score(current_embedding, constraint.constraint_vector),
                    "candidate_projection": symbolic_projection,
                })

            # Level 1: bounded soft fusion.  Candidate biases are allowed to
            # reorder only candidates that survive the immutable hard mask.
            for constraint in ([] if unresolved_hard else soft_constraints):
                alpha = min(self.max_statistical_alpha, clamp01(constraint.weight))
                old = current_embedding.clone()
                if constraint.constraint_vector is not None and alpha > 0:
                    vector = constraint.constraint_vector.to(current_embedding.device, current_embedding.dtype)
                    if vector.shape == current_embedding.shape and torch.isfinite(vector).all():
                        current_embedding = (1.0 - alpha) * current_embedding + alpha * vector
                for index, label in enumerate(labels):
                    if not bool(probability_mask[index]):
                        continue
                    bias = 0.0
                    for candidate, score in constraint.candidate_bias.items():
                        if candidate_matches(label, (candidate,)):
                            bias += finite_float(score, 0.0)
                    logits[index] += alpha * bias
                operations.append({
                    "operation": "blend",
                    "level": 1,
                    "constraint_id": constraint.constraint_id,
                    "alpha": alpha,
                    "delta": float(torch.norm(current_embedding - old).item()),
                })

            # A lower-level operation can never unmask a hard rejection.
            masked_logits = logits.masked_fill(~probability_mask, float("-inf"))
            probabilities = finite_softmax(masked_logits)
            active = self._detect(
                labels,
                logits,
                current_embedding,
                constraints,
                probability_mask,
                unresolved_hard,
            )
            for conflict in active:
                historical_conflicts.append({"iteration": iteration, **conflict})
            active_set = set(item["constraint_id"] for item in active)
            embedding_delta = float(torch.norm(current_embedding - before_embedding).item())
            probability_delta = float(torch.max(torch.abs(probabilities - before_probabilities)).item())
            combined_delta = max(embedding_delta, probability_delta)

            if previous_active is not None and active_set == previous_active and combined_delta < self.epsilon:
                stable_count += 1
            else:
                stable_count = 0
            history.append({
                "iteration": iteration,
                "operations": operations,
                "active_conflict_ids": sorted(active_set),
                "active_conflict_count": len(active_set),
                "embedding_delta": embedding_delta,
                "probability_delta": probability_delta,
                "combined_delta": combined_delta,
                "selected_candidate": labels[int(torch.argmax(probabilities).item())],
                "confidence": float(probabilities.max().item()),
                "hard_mask_preserved": unresolved_hard or all(
                    float(probabilities[index].item()) == 0.0
                    for index, allowed in enumerate(hard_mask)
                    if not bool(allowed)
                ),
            })

            if not active_set and not unresolved_hard:
                stop_reason = "no_active_conflicts"
                previous_probabilities = probabilities
                break
            if (
                not force_max_iterations
                and stable_count >= self.stable_rounds
                and not any(item["level"] == LEVEL_DETERMINISTIC for item in active)
            ):
                stop_reason = "stable_unresolved"
                previous_probabilities = probabilities
                break
            previous_active = active_set
            previous_probabilities = probabilities

        final_probabilities = previous_probabilities
        selected_index = int(torch.argmax(final_probabilities).item())
        selected = labels[selected_index]
        confidence = float(final_probabilities[selected_index].item())
        final_active = history[-1]["active_conflict_ids"] if history else []
        if unresolved_hard:
            state = "hypothesis"
            confidence = min(confidence, max(0.0, float(initial_confidence)))
        elif level_lock == LEVEL_DETERMINISTIC and bool(hard_mask.sum().item() == 1):
            state = "confirmed_hard"
            confidence = 1.0
            final_probabilities = torch.zeros_like(final_probabilities)
            final_probabilities[int(torch.nonzero(hard_mask, as_tuple=False)[0].item())] = 1.0
            selected_index = int(torch.argmax(final_probabilities).item())
            selected = labels[selected_index]
        elif confidence >= self.confirmation_threshold and not any(
            conflict["level"] == LEVEL_DETERMINISTIC
            for conflict in self._detect(
                labels, logits, current_embedding, constraints, probability_mask, unresolved_hard
            )
        ):
            state = "confirmed_soft"
        else:
            state = "hypothesis"

        return {
            "schema": RESULT_SCHEMA,
            "node_id": node_id,
            "initial_state": initial_state,
            "initial_confidence": clamp01(initial_confidence),
            "initial_candidates": {
                label: float(initial_probabilities[index].item())
                for index, label in enumerate(labels)
            },
            "final_candidates": {
                label: float(final_probabilities[index].item())
                for index, label in enumerate(labels)
            },
            "selected_candidate": selected,
            "state": state,
            "confidence": confidence,
            "embedding": current_embedding,
            "embedding_delta": float(torch.norm(current_embedding - embedding.float()).item()),
            "iterations_run": len(history),
            "stop_reason": stop_reason,
            "active_conflict_ids": list(final_active),
            "active_conflict_count": len(final_active),
            "historical_conflicts": historical_conflicts,
            "historical_conflict_count": len(historical_conflicts),
            "new_hypotheses": new_hypotheses,
            "unresolved_hard_conflict": unresolved_hard,
            "level_lock": level_lock,
            "hard_mask": {label: bool(hard_mask[index]) for index, label in enumerate(labels)},
            "hard_trace": hard_trace,
            "projection_incompatibilities": projection_incompatibilities,
            "constraint_ids": [item.constraint_id for item in constraints],
            "constraint_levels": dict(Counter(str(item.level) for item in constraints)),
            "deduplication": dedupe_audit,
            "iteration_trace": history,
            "checks": {
                "embedding_finite": bool(torch.isfinite(current_embedding).all()),
                "probabilities_finite": bool(torch.isfinite(final_probabilities).all()),
                "probabilities_sum_to_one": math.isclose(
                    float(final_probabilities.sum().item()), 1.0, rel_tol=0.0, abs_tol=1e-6
                ),
                "lower_levels_did_not_unmask_hard_candidates": unresolved_hard or all(
                    float(final_probabilities[index].item()) == 0.0
                    for index, allowed in enumerate(hard_mask)
                    if not bool(allowed)
                ),
                "unresolved_hard_has_hypothesis": (not unresolved_hard) or bool(new_hypotheses),
                "max_iteration_respected": len(history) <= self.max_iterations,
            },
        }


def serializable_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    output = dict(result)
    embedding = output.pop("embedding", None)
    if torch.is_tensor(embedding):
        output["embedding_norm"] = float(embedding.norm().item())
        output["embedding_dimension"] = int(embedding.numel())
    return output
