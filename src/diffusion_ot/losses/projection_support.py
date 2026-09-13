"""Regularize Eq. (7) means against fixed target decoder-code distributions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from diffusion_ot.losses.infoot import sinkhorn_divergence


@dataclass
class ProjectionSupportResult:
    loss: torch.Tensor
    metrics: dict[str, dict[str, float | bool]]


def projection_support_loss(
    weights: dict[str, torch.Tensor],
    references: dict[str, torch.Tensor],
    target_anchors: dict[str, torch.Tensor],
    anchor_variances: dict[str, float],
    *,
    teacher_weights: dict[str, torch.Tensor] | None = None,
    regularization: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 1e-5,
) -> ProjectionSupportResult:
    """Same-domain costs only; teach weights without inflating target values.

    Gradients pass through full conditional probabilities. The current raw
    value bank and Stage 1A target anchors are detached for this loss so direct
    prototype inflation cannot reduce it. Both KDE encoders still get gradients
    through the weights. No output rescaling or top-k truncation is performed.
    """
    values, metrics = [], {}
    for source, target in (("cat", "dog"), ("dog", "cat")):
        direction = f"{source}_to_{target}"
        target_codes = target_anchors[target].detach().float()
        projected = weights[direction] @ references[target].detach().float()
        masses = torch.ones(len(target_codes), device=target_codes.device) / len(target_codes)
        if teacher_weights is not None:
            # Match the target structure mixture relevant to these queries,
            # rather than forcing a small batch to cover every target mode.
            masses = teacher_weights[direction].detach().to(target_codes).mean(0).clamp_min(1e-8)
            masses = masses / masses.sum()
        scale = projected.shape[1] * float(anchor_variances[target])
        result = sinkhorn_divergence(
            projected, target_codes, target_masses=masses, cost_scale=scale, regularization=regularization,
            max_iterations=max_iterations, tolerance=tolerance,
        )
        values.append(result.loss)
        with torch.no_grad():
            target_mean = (masses[:, None] * target_codes).sum(0)
            target_variance = (masses[:, None] * (target_codes-target_mean).square()).sum(0).mean().clamp_min(1e-12)
            metrics[direction] = {
                "sinkhorn_divergence": float(result.loss.detach()),
                "max_marginal_residual": result.max_marginal_residual,
                "converged": result.converged,
                "projected_variance_to_anchor_ratio": float(projected.var(0, unbiased=False).mean() / target_variance),
                "projected_mean_shift_squared": float((projected.mean(0)-target_mean).square().mean() / target_variance),
                "target_mass_effective_count": float((-(masses * masses.log()).sum()).exp()),
            }
    return ProjectionSupportResult(torch.stack(values).mean(), metrics)


def support_options(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "regularization": float(config.get("regularization", 0.1)),
        "max_iterations": int(config.get("max_iterations", 1000)),
        "tolerance": float(config.get("tolerance", 1e-5)),
    }
