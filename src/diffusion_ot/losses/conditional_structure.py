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
    conditional_variance_decomposition,
    conditional_reference_log_weights,
    infoot_cross_distance_scale,
    infoot_distance_scale,
)


@dataclass
class ConditionalStructureResult:
    loss: torch.Tensor
    metrics: dict[str, dict[str, float | None]]
    weights: dict[str, torch.Tensor]
    teacher_weights: dict[str, torch.Tensor]
    log_weights: dict[str, torch.Tensor]


@torch.no_grad()
def conditional_query_diagnostics(log_weights, teacher_weights, costs):
    """Compare query-specific matching with the same bank's average readout.

    The baseline preserves the student's aggregate target preferences but
    removes its source-query correspondence. This is a diagnostic comparison,
    not a model trained without KL and not independent semantic validation.
    Log-space averaging remains finite with narrow, underflowing kernels.
    """
    queries, targets = log_weights.shape
    probability = log_weights.exp()
    average_log = torch.logsumexp(log_weights, dim=0, keepdim=True) - math.log(queries)
    independent_kl = F.kl_div(average_log.expand_as(log_weights), teacher_weights, reduction="batchmean")
    current_kl = F.kl_div(log_weights, teacher_weights, reduction="batchmean")
    independent_cost = (average_log.exp() * costs).sum(1).mean()
    current_cost = (probability * costs).sum(1).mean()
    return {
        "query_count": queries,
        "reference_targets": targets,
        "uniform_kl": float(F.kl_div(torch.full_like(log_weights, -math.log(targets)),
                                      teacher_weights, reduction="batchmean")),
        "query_independent_kl": float(independent_kl),
        "kl_gain_over_query_independent": float(independent_kl - current_kl),
        "query_independent_structure_cost": float(independent_cost),
        "structure_cost_gain_over_query_independent": float(independent_cost - current_cost),
        "query_information_nats": float((probability * (log_weights - average_log)).sum(1).mean()),
    }


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
    reference_matching: dict[str, torch.Tensor] | None = None,
    query_matching: dict[str, torch.Tensor] | None = None,
    differentiate_distance_scale: bool = False,
) -> ConditionalStructureResult:
    """Average bidirectional KL(teacher || full Eq. 7 conditional readout).

    Callers must fit the plan on references only. Queries come from a disjoint
    portion of the training batch (or validation split during fixed probes).
    Only encoder/matching features receive gradients; the plan and teacher are detached.
    Each target bank is its complete reference set, without top-k truncation.
    Optional matching features change the KDE geometry, while references and
    queries remain raw decoder codes. They must retain the same row ordering.
    """
    if not math.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("Structure cost scale must be finite and positive.")
    if not math.isfinite(teacher_temperature) or teacher_temperature <= 0:
        raise ValueError("Projection teacher temperature must be finite and positive.")
    if (reference_matching is None) != (query_matching is None):
        raise ValueError("Supply both reference and query matching features, or neither.")
    losses, metrics, weights, teacher_distributions, log_distributions = [], {}, {}, {}, {}
    for source, target in (("cat", "dog"), ("dog", "cat")):
        if len(queries[source]) < 1 or len(references[target]) < 2:
            raise ValueError("Conditional structure training needs queries and at least two target references.")
        if len(query_structure[source]) != len(queries[source]) or len(reference_structure[target]) != len(references[target]):
            raise ValueError("Structure descriptors must match query/reference sample counts.")
        match_ref = references if reference_matching is None else reference_matching
        match_query = queries if query_matching is None else query_matching
        if any(len(match_ref[d]) != len(references[d]) or len(match_query[d]) != len(queries[d])
               for d in (source, target)):
            raise ValueError("Matching feature rows must correspond to raw code rows.")
        source_features = F.normalize(match_query[source].float(), dim=-1)
        source_references = F.normalize(match_ref[source].float(), dim=-1)
        target_references = F.normalize(match_ref[target].float(), dim=-1)
        log_weights = conditional_reference_log_weights(
            source_features, source_references, target_references,
            coupling.detach() if source == "cat" else coupling.detach().T,
            bandwidth=bandwidth,
            distance_scale_x=infoot_cross_distance_scale(
                source_features, source_references, detach=not differentiate_distance_scale),
            distance_scale_y=infoot_distance_scale(
                target_references, detach=not differentiate_distance_scale),
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
        # Preserve the stable log-space readout for auxiliary contrastive
        # supervision; taking log(exp(log_weights)) can underflow in fp32.
        log_distributions[direction] = log_weights
        teacher_distributions[direction] = teacher_weights
        with torch.no_grad():
            probability = weights[direction]
            projected = probability @ references[target].float()
            teacher_projected = teacher_weights @ references[target].float()
            target_variance = references[target].float().var(0, unbiased=False).mean().clamp_min(1e-12)
            uniform_cost = costs.mean()
            teacher_cost = (teacher_weights * costs).sum(1).mean()
            student_cost = (probability * costs).sum(1).mean()
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
                "projected_to_target_variance_ratio": float(
                    projected.var(0, unbiased=False).mean() / target_variance
                ),
                "teacher_projected_to_target_variance_ratio": float(
                    teacher_projected.var(0, unbiased=False).mean() / target_variance
                ),
                "teacher_projected_to_target_norm_ratio": float(
                    teacher_projected.norm(dim=1).mean()
                    / references[target].float().norm(dim=1).mean().clamp_min(1e-8)
                ),
                "fraction_of_teacher_cost_gain": float(
                    (uniform_cost - student_cost) / (uniform_cost - teacher_cost).clamp_min(1e-12)
                ),
            }
            metrics[direction].update({
                f"teacher_{key}": value for key, value in
                conditional_variance_decomposition(teacher_weights, references[target]).items()
            })
            metrics[direction].update(conditional_query_diagnostics(log_weights, teacher_weights, costs))
    return ConditionalStructureResult(
        torch.stack(losses).mean(), metrics, weights, teacher_distributions, log_distributions)
