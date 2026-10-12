from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import stable_seed


ARMS = ("symbolic_only_clean", "unified_llm_gate_clean", "module_aware_clean")
CORE_MODULES = ("TypeSem", "InterfaceSem", "LayoutSem")
OPTIONAL_MODULE = "APISem"


def _init_linear(linear, master_seed, component_name):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(master_seed, component_name))
    fan_in = max(1, linear.weight.shape[1])
    bound = math.sqrt(6.0 / fan_in)
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound, generator=generator)
        if linear.bias is not None:
            bias_bound = 1.0 / math.sqrt(fan_in)
            linear.bias.uniform_(-bias_bound, bias_bound, generator=generator)


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


class CleanJointRanker(nn.Module):
    """Stage-2 ranker with order-independent deterministic initialization.

    Every component is initialized from ``hash(master_seed, component_name)``.
    Shared semantic heads and shared W_s therefore have identical initial
    values for comparable arms, independent of Python module creation order.
    """

    def __init__(self, arm, seed, symbolic_dim=64, llm_dim=4096, bridge_dim=256,
                 modules=CORE_MODULES, apisem_enabled=False, temperature=0.2,
                 type_residual_alpha=0.05):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unknown arm: %s" % arm)
        self.arm = arm
        self.seed = int(seed)
        self.symbolic_dim = int(symbolic_dim)
        self.llm_dim = int(llm_dim)
        self.bridge_dim = int(bridge_dim)
        self.temperature = float(temperature)
        self.apisem_enabled = bool(apisem_enabled)
        active = list(modules)
        if self.apisem_enabled and OPTIONAL_MODULE not in active:
            active.append(OPTIONAL_MODULE)
        self.modules_enabled = tuple(active)

        self.heads = nn.ModuleDict({
            module: nn.Linear(self.symbolic_dim, self.bridge_dim, bias=False)
            for module in self.modules_enabled
        })
        if arm in ("symbolic_only_clean", "unified_llm_gate_clean"):
            self.W_s = nn.Linear(self.symbolic_dim, self.bridge_dim)
        if arm == "unified_llm_gate_clean":
            self.W_l = nn.Linear(self.llm_dim, self.bridge_dim)
            self.gate = nn.Linear(self.bridge_dim * 2, 1)
        if arm == "module_aware_clean":
            self.W_s_by = nn.ModuleDict({m: nn.Linear(self.symbolic_dim, self.bridge_dim) for m in self.modules_enabled})
            self.W_l_by = nn.ModuleDict({m: nn.Linear(self.llm_dim, self.bridge_dim) for m in self.modules_enabled})
            self.gate_by = nn.ModuleDict({m: nn.Linear(self.bridge_dim * 2, 1) for m in self.modules_enabled})

        self._initialize(type_residual_alpha)

    def _initialize(self, type_residual_alpha):
        for module, head in self.heads.items():
            _init_linear(head, self.seed, "shared.semantic_head.%s" % module)
        if hasattr(self, "W_s"):
            _init_linear(self.W_s, self.seed, "shared.W_s")
        if hasattr(self, "W_l"):
            _init_linear(self.W_l, self.seed, "unified.W_l")
            _init_linear(self.gate, self.seed, "unified.gate")
        if hasattr(self, "W_s_by"):
            for module in self.modules_enabled:
                _init_linear(self.W_s_by[module], self.seed, "module.%s.W_s" % module)
                _init_linear(self.W_l_by[module], self.seed, "module.%s.W_l" % module)
                _init_linear(self.gate_by[module], self.seed, "module.%s.gate" % module)
            if "TypeSem" in self.gate_by:
                alpha = float(type_residual_alpha)
                if not 0 < alpha < 1:
                    raise ValueError("type_residual_alpha must be in (0,1)")
                with torch.no_grad():
                    self.gate_by["TypeSem"].weight.zero_()
                    self.gate_by["TypeSem"].bias.fill_(math.log(alpha / (1.0 - alpha)))

    def initialization_hashes(self):
        return {name: tensor_sha256(parameter) for name, parameter in self.named_parameters()}

    def fuse(self, module, h_sym, h_llm=None):
        if module not in self.modules_enabled:
            raise RuntimeError("module is disabled: %s" % module)
        if self.arm == "symbolic_only_clean":
            return self.W_s(h_sym), None, "symbolic_only"
        if self.arm == "unified_llm_gate_clean":
            if h_llm is None:
                raise RuntimeError("unified_llm_gate_clean requires function-level LLM representation")
            symbolic = self.W_s(h_sym)
            language = self.W_l(h_llm)
            gate = torch.sigmoid(self.gate(torch.cat([symbolic, language], dim=-1)))
            return gate * symbolic + (1.0 - gate) * language, gate, "symbolic_weight"
        symbolic = self.W_s_by[module](h_sym)
        if h_llm is None:
            return symbolic, None, "contract_fixed_symbolic_fallback_for_unusable_instruction_mapping"
        language = self.W_l_by[module](h_llm)
        gate = torch.sigmoid(self.gate_by[module](torch.cat([symbolic, language], dim=-1)))
        if module == "TypeSem":
            return symbolic + gate * language, gate, "llm_residual_alpha"
        return gate * symbolic + (1.0 - gate) * language, gate, "symbolic_weight"

    def score(self, module, fused_query, candidate_hsym):
        projected = self.heads[module](candidate_hsym)
        logits = (
            F.normalize(fused_query, dim=-1).unsqueeze(1)
            * F.normalize(projected, dim=-1)
        ).sum(-1) / self.temperature
        return logits, projected


def masked_module_loss(items, enabled_modules, align_weight=0.1):
    """Average only modules that have enabled, non-empty supervision.

    Each item has ``module``, ``logits`` (1-D), ``gold_index``, ``query`` and
    ``gold_candidate``. APISem contributes nothing when it is disabled; it is
    not inserted as a numerical zero into the module mean.
    """
    enabled = set(enabled_modules)
    per_module = {}
    for module in sorted(enabled):
        rows = [item for item in items if item["module"] == module]
        if not rows:
            continue
        semantic = []
        alignment = []
        for item in rows:
            target = torch.tensor([int(item["gold_index"])], device=item["logits"].device)
            semantic.append(F.cross_entropy(item["logits"].unsqueeze(0), target))
            alignment.append(1.0 - F.cosine_similarity(
                F.normalize(item["query"], dim=-1),
                F.normalize(item["gold_candidate"], dim=-1), dim=-1,
            ).mean())
        l_sem = torch.stack(semantic).mean()
        l_align = torch.stack(alignment).mean()
        per_module[module] = {
            "semantic": l_sem,
            "alignment": l_align,
            "total": l_sem + float(align_weight) * l_align,
            "n": len(rows),
        }
    if not per_module:
        raise RuntimeError("no enabled module has supervision in this batch")
    total = torch.stack([value["total"] for value in per_module.values()]).mean()
    return total, per_module
