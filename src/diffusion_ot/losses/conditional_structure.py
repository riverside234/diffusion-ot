"""Train the out-of-reference InfoOT readout using a frozen structural teacher.

This is a project extension, not an objective from the official InfoOT repo.
Teacher probabilities describe descriptor similarity, not ground-truth pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from diffusion_ot.losses.infoot import (
    conditional_reference_log_weights,
    infoot_cross_distance_scale,
    infoot_distance_scale,
)


@dataclass
class ConditionalStructureResult:
    loss: torch.Tensor
    metrics: dict[str, dict[str, float]]
    weights: dict[str, torch.Tensor]


def conditional_structure_loss(
    references: dict[str, torch.Tensor],
    queries: dict[str, torch.Tensor],
    reference_structure: dict[str, torch.Tensor],
    query_structure: dict[str, torch.Tensor],
    coupling: torch.Tensor,
    *,
    bandwidth: float,
    cost_scale: float,
    teacher_temperature: float = 0.1,
) -> ConditionalStructureResult:
    """Average bidirectional KL(teacher || full Eq. 7 conditional readout).

    Callers must fit the plan on references only. Queries come from a disjoint
    portion of the training batch (or validation split during fixed probes).
    Only encoder features receive gradients; the plan and teacher are detached.
    Each target bank is its complete reference set, without top-k truncation.
    """
    if not math.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("Structure cost scale must be finite and positive.")
    if not math.isfinite(teacher_temperature) or teacher_temperature <= 0:
        raise ValueError("Projection teacher temperature must be finite and positive.")
    losses, metrics, weights = [], {}, {}
    for source, target in (("cat", "dog"), ("dog", "cat")):
        if len(queries[source]) < 1 or len(references[target]) < 2:
            raise ValueError("Conditional structure training needs queries and at least two target references.")
        if len(query_structure[source]) != len(queries[source]) or len(reference_structure[target]) != len(references[target]):
            raise ValueError("Structure descriptors must match query/reference sample counts.")
        source_features = F.normalize(queries[source].float(), dim=-1)
        source_references = F.normalize(references[source].float(), dim=-1)
        target_references = F.normalize(references[target].float(), dim=-1)
        log_weights = conditional_reference_log_weights(
            source_features, source_references, target_references,
            coupling.detach() if source == "cat" else coupling.detach().T,
            bandwidth=bandwidth,
            distance_scale_x=infoot_cross_distance_scale(source_features, source_references),
            distance_scale_y=infoot_distance_scale(target_references),
        )
        costs = torch.cdist(
            query_structure[source].detach().to(source_features),
            reference_structure[target].detach().to(source_features),
        ) / cost_scale
        teacher_log_weights = F.log_softmax(-costs / teacher_temperature, dim=1)
        teacher_weights = teacher_log_weights.exp()
        loss = F.kl_div(log_weights, teacher_weights, reduction="batchmean")
        losses.append(loss)
        direction = f"{source}_to_{target}"
        weights[direction] = log_weights.exp()
        with torch.no_grad():
            probability = weights[direction]
            projected = probability @ references[target].float()
            metrics[direction] = {
                "kl": float(loss.detach()),
                "expected_structure_cost": float((probability * costs).sum(1).mean()),
                "teacher_structure_cost": float((teacher_weights * costs).sum(1).mean()),
                "uniform_structure_cost": float(costs.mean()),
                "effective_targets": float((-(probability * log_weights).sum(1)).exp().mean()),
                "teacher_effective_targets": float((-(teacher_weights * teacher_log_weights).sum(1)).exp().mean()),
                "projected_to_target_norm_ratio": float(
                    projected.norm(dim=1).mean() / references[target].float().norm(dim=1).mean().clamp_min(1e-8)
                ),
            }
    return ConditionalStructureResult(torch.stack(losses).mean(), metrics, weights)
