"""Protect a primary encoder objective from large or conflicting auxiliary gradients.

This asymmetric projection plus norm cap is inspired by PCGrad; it is not
PCGrad's randomized, symmetric multi-task algorithm. The guarantee concerns
the supplied raw gradients, not Adam steps or held-out reconstruction error.
"""
from __future__ import annotations

import math

import torch


def guarded_backward(
    primary: torch.Tensor,
    auxiliary: torch.Tensor,
    parameter_groups: dict[str, list[torch.nn.Parameter]],
    *,
    max_auxiliary_ratio: float = 0.25,
    project_conflicts: bool = True,
) -> dict[str, dict[str, float | bool]]:
    if not math.isfinite(max_auxiliary_ratio) or max_auxiliary_ratio < 0:
        raise ValueError("max_auxiliary_ratio must be finite and nonnegative.")
    parameters = [p for group in parameter_groups.values() for p in group]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Gradient guard parameter groups must not overlap.")
    primary_gradients = torch.autograd.grad(primary, parameters, retain_graph=True, allow_unused=True)
    auxiliary_gradients = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
    metrics, offset = {}, 0
    for name, group in parameter_groups.items():
        if not group:
            raise ValueError("Gradient guard needs nonempty parameter groups.")
        if len({p.device for p in group}) != 1:
            raise ValueError("Each gradient guard group must be on a single device.")
        primary_parts, auxiliary_parts = [], []
        for p, gp, ga in zip(group, primary_gradients[offset:offset+len(group)], auxiliary_gradients[offset:offset+len(group)]):
            primary_parts.append(torch.zeros_like(p) if gp is None else gp.detach())
            auxiliary_parts.append(torch.zeros_like(p) if ga is None else ga.detach())
        offset += len(group)
        squared_primary = sum(g.float().square().sum() for g in primary_parts)
        squared_auxiliary = sum(g.float().square().sum() for g in auxiliary_parts)
        dot = sum((gp.float() * ga.float()).sum() for gp, ga in zip(primary_parts, auxiliary_parts))
        scalars = torch.stack((squared_primary, squared_auxiliary, dot)).cpu().tolist()
        if not all(math.isfinite(x) for x in scalars):
            raise FloatingPointError(f"Non-finite {name} encoder gradients.")
        primary_squared, auxiliary_squared, dot_value = scalars
        primary_norm, auxiliary_norm = math.sqrt(primary_squared), math.sqrt(auxiliary_squared)
        conflict = project_conflicts and dot_value < 0 and primary_squared > 0
        coefficient = dot_value / primary_squared if conflict else 0.0
        adjusted = [ga - coefficient * gp for gp, ga in zip(primary_parts, auxiliary_parts)]
        adjusted_norm = math.sqrt(float(sum(g.float().square().sum() for g in adjusted)))
        if not math.isfinite(adjusted_norm):
            raise FloatingPointError(f"Non-finite {name} projected auxiliary gradients.")
        scale = min(1.0, max_auxiliary_ratio * primary_norm / max(adjusted_norm, 1e-30))
        for p, gp, ga in zip(group, primary_parts, adjusted):
            p.grad = gp + scale * ga
        metrics[name] = {
            "primary_gradient_norm": primary_norm,
            "auxiliary_gradient_norm_before": auxiliary_norm,
            "auxiliary_gradient_norm_after": adjusted_norm * scale,
            "auxiliary_ratio_before": auxiliary_norm / max(primary_norm, 1e-30),
            "auxiliary_ratio_after": adjusted_norm * scale / max(primary_norm, 1e-30),
            "cosine_before": dot_value / max(primary_norm * auxiliary_norm, 1e-30),
            "conflict_projected": bool(conflict),
            "auxiliary_scale": scale,
        }
    return metrics
