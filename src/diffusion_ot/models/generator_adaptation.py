"""Selective Stage 1B generator adaptation and paired raw/EMA restoration.

Only the added conditioning modules are exposed. Pretrained SiT weights are
never included, even though semantic LoRA modules live inside its attention.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def generator_adaptation_enabled(config: dict[str, Any]) -> bool:
    return bool((config.get("generator_adaptation") or {}).get("enabled", False))


def generator_modules(branch: Any) -> tuple[nn.ModuleDict, nn.ModuleDict]:
    semantic = branch.semantic_transformer
    adapters = nn.ModuleDict({
        "z_proj": semantic.z_proj,
        "adapters": semantic.adapters,
        "final_adapter": semantic.final_adapter,
    })
    lora = nn.ModuleDict()
    for layer, targets in semantic._attention_lora_modules.items():
        for target, module in targets.items():
            lora[f"layer_{layer}_{target}_down"] = module.lora_down
            lora[f"layer_{layer}_{target}_up"] = module.lora_up
    return adapters, lora


def generator_parameter_view(branch: Any) -> nn.ModuleDict:
    adapters, lora = generator_modules(branch)
    return nn.ModuleDict({"adapters": adapters, "lora": lora})


def configure_generator_adaptation(branch: Any) -> nn.ModuleDict:
    branch.semantic_transformer.eval().requires_grad_(False)
    # Keep the learned null vector fixed. The shared G(null) path is still
    # trained by semantic dropout and protected by a frozen Stage 1A teacher.
    if branch.semantic_conditioner is None:
        raise ValueError("Generator adaptation requires learned-null semantic CFG.")
    branch.semantic_conditioner.eval().requires_grad_(False)
    view = generator_parameter_view(branch)
    if not len(view["lora"]):
        raise ValueError("Experiment D requires existing attention LoRA.")
    view.requires_grad_(True)
    # Modules have no batch statistics. Preserve eval mode to avoid introducing
    # stochastic LoRA dropout into ODE/checkpoint recomputation.
    view.eval()
    branch.encoder.train().requires_grad_(True)
    return view


def baseline_parameter_snapshot(branch: Any, view: nn.Module) -> dict[str, torch.Tensor]:
    identifiers = {id(p) for p in view.parameters()}
    return {name: p.detach().cpu().clone()
            for name, p in branch.semantic_transformer.named_parameters() if id(p) in identifiers}


def predict_with_parameters(branch: Any, parameters: dict[str, torch.Tensor],
                            x_t: torch.Tensor, timestep: torch.Tensor,
                            z: torch.Tensor, class_labels: torch.Tensor):
    """Read a fixed teacher without copying a backbone or mutating live weights."""
    overrides = {name: value.to(device=x_t.device, dtype=x_t.dtype)
                 for name, value in parameters.items()}
    return torch.func.functional_call(
        branch.semantic_transformer, overrides, (),
        {"hidden_states": x_t, "timestep": timestep, "z": z,
         "class_labels": class_labels}, strict=False,
    )


def load_joint_generator(branch: Any, checkpoint: dict[str, Any], domain: str, *, weights: str) -> None:
    if weights not in {"raw", "ema"}:
        raise ValueError("Generator weights must be raw or ema.")
    adapted = generator_adaptation_enabled(checkpoint.get("config") or {})
    key = "generators" if adapted else "fixed_generators"
    state = (checkpoint.get(key) or {}).get(domain)
    if state is None:
        raise ValueError(f"Joint checkpoint has no {key}.{domain} state.")
    if adapted and int(checkpoint.get("format_version", 0)) != 4:
        raise ValueError("Adapted generators require checkpoint format 4.")
    if not adapted and checkpoint.get("generators"):
        raise ValueError("Checkpoint contains adapted generators but its config disables them.")
    branch.load_generator_state_dict(state)
    if adapted and weights == "ema":
        state = (checkpoint.get("generator_ema") or {}).get(domain)
        if state is None:
            raise ValueError(f"Joint checkpoint has no generator_ema.{domain}; refusing raw/EMA mixing.")
        generator_parameter_view(branch).load_state_dict(state, strict=True)
