"""Encoder-only transport helpers for the no-external-teacher experiment.

Direct cross-domain cosine cost assumes the two learned matching spaces can
develop compatible coordinates. Independent Stage 1A encoders do not ensure
this: the cost breaks the independent-plan symmetry but provides no semantic
ground truth. Keep this experiment separate from frozen-teacher controls.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from diffusion_ot.losses.infoot import (
    conditional_reference_log_weights,
    conditional_variance_decomposition,
    infoot_cross_distance_scale,
    infoot_distance_scale,
)


@dataclass
class EncoderConditionalReadoutResult:
    metrics: dict[str, dict[str, float | int | None]]
    weights: dict[str, torch.Tensor]
    log_weights: dict[str, torch.Tensor]


def _validate_features(features: torch.Tensor, name: str) -> None:
    if features.ndim != 2 or not features.shape[0] or not features.shape[1]:
        raise ValueError(f"{name} must be a nonempty [samples, dimensions] feature matrix.")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError(f"{name} must contain finite floating-point features.")


@torch.no_grad()
def encoder_transport_cost(cat_matching: torch.Tensor, dog_matching: torch.Tensor) -> torch.Tensor:
    """Detached cosine cost, fixed for each OT solve, with no external features.

    Normalize each input row, then return 1 - cosine similarity in [0, 2].
    The normalization gives the cost a bounded scale, not semantic alignment.
    A nonseparable cross cost avoids the stationary independent product plan
    of pure MI initialized with independent marginals; it is not a guarantee
    that the resulting assignments are meaningful.
    """
    _validate_features(cat_matching, "cat_matching")
    _validate_features(dog_matching, "dog_matching")
    if cat_matching.device != dog_matching.device or cat_matching.shape[1] != dog_matching.shape[1]:
        raise ValueError("Encoder cross cost requires matching dimensions and the same device.")
    dtype = torch.float64 if torch.float64 in (cat_matching.dtype, dog_matching.dtype) else torch.float32
    with torch.autocast(device_type=cat_matching.device.type, enabled=False):
        cat, dog = cat_matching.to(dtype=dtype), dog_matching.to(dtype=dtype)
        if (cat.norm(dim=1) < 1e-8).any() or (dog.norm(dim=1) < 1e-8).any():
            raise ValueError("Encoder cross cost requires nonzero matching features.")
        return (1.0 - F.normalize(cat, dim=1) @ F.normalize(dog, dim=1).T).clamp(0.0, 2.0)


def encoder_conditional_readout(
    references: dict[str, torch.Tensor],
    queries: dict[str, torch.Tensor],
    coupling: torch.Tensor,
    *,
    bandwidth: float,
    reference_matching: dict[str, torch.Tensor] | None = None,
    query_matching: dict[str, torch.Tensor] | None = None,
    differentiate_distance_scale: bool = False,
) -> EncoderConditionalReadoutResult:
    """Bidirectional Eq. 7 probabilities without a teacher or a KL objective.

    Fit the coupling on references only; callers supply disjoint queries.
    Gradients through readout weights reach learned matching features, never
    the detached coupling. Raw target codes stay unchanged for the caller's
    weighted decoder condition. Diagnostics measure query dependence and code
    spread; none is an independent estimate of translation quality.
    """
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Projection bandwidth must be finite and positive.")
    if (reference_matching is None) != (query_matching is None):
        raise ValueError("Supply both reference and query matching features, or neither.")
    match_ref = references if reference_matching is None else reference_matching
    match_query = queries if query_matching is None else query_matching
    for domain in ("cat", "dog"):
        for collection, name in ((references, "references"), (queries, "queries"),
                                 (match_ref, "reference_matching"), (match_query, "query_matching")):
            if domain not in collection:
                raise ValueError(f"{name} must include cat and dog features.")
            _validate_features(collection[domain], f"{name}.{domain}")
        if len(references[domain]) < 2:
            raise ValueError("Encoder conditional readout needs at least two references per domain.")
        if (len(match_ref[domain]) != len(references[domain])
                or len(match_query[domain]) != len(queries[domain])):
            raise ValueError("Matching feature rows must correspond to raw code rows.")
        if (match_ref[domain].shape[1] != match_query[domain].shape[1]
                or references[domain].shape[1] != queries[domain].shape[1]):
            raise ValueError("Query and reference dimensions must match within each domain.")
    matrices = [collection[d] for collection in (references, queries, match_ref, match_query)
                for d in ("cat", "dog")]
    device = references["cat"].device
    if coupling.device != device or any(value.device != device for value in matrices):
        raise ValueError("Encoder conditional readout requires all tensors on the same device.")
    if coupling.shape != (len(references["cat"]), len(references["dog"])):
        raise ValueError("Reference banks do not match the coupling dimensions.")
    dtype = torch.float64 if any(value.dtype == torch.float64 for value in matrices) else torch.float32
    metrics, weights, log_distributions = {}, {}, {}
    with torch.autocast(device_type=device.type, enabled=False):
        for source, target in (("cat", "dog"), ("dog", "cat")):
            source_features = F.normalize(match_query[source].to(dtype=dtype), dim=1)
            source_references = F.normalize(match_ref[source].to(dtype=dtype), dim=1)
            target_references = F.normalize(match_ref[target].to(dtype=dtype), dim=1)
            plan = coupling.detach().to(dtype=dtype)
            log_weights = conditional_reference_log_weights(
                source_features, source_references, target_references,
                plan if source == "cat" else plan.T,
                bandwidth=bandwidth,
                distance_scale_x=infoot_cross_distance_scale(
                    source_features, source_references, detach=not differentiate_distance_scale),
                distance_scale_y=infoot_distance_scale(
                    target_references, detach=not differentiate_distance_scale),
            )
            direction = f"{source}_to_{target}"
            log_distributions[direction] = log_weights
            weights[direction] = log_weights.exp()
            with torch.no_grad():
                probability = weights[direction]
                target_codes = references[target].to(dtype=dtype)
                projected = probability @ target_codes
                average_log = torch.logsumexp(log_weights, dim=0, keepdim=True) - math.log(len(log_weights))
                query_information = (probability * (log_weights - average_log)).sum(1).mean().clamp_min(0)
                information_bound = math.log(min(len(log_weights), len(target_codes)))
                metrics[direction] = {
                    "query_count": len(log_weights),
                    "reference_targets": len(target_codes),
                    "effective_targets": float((-(probability * log_weights).sum(1)).exp().mean()),
                    "query_information_nats": float(query_information),
                    "query_information_fraction_of_bound": float(query_information / information_bound)
                        if information_bound > 0 else None,
                    "mean_query_total_variation_from_marginal": float(
                        .5 * (probability - average_log.exp()).abs().sum(1).mean()),
                    "mean_max_target_probability": float(probability.max(1).values.mean()),
                    "projected_to_target_norm_ratio": float(
                        projected.norm(dim=1).mean() / target_codes.norm(dim=1).mean().clamp_min(1e-8)),
                    "projected_to_target_variance_ratio": float(
                        projected.var(0, unbiased=False).mean()
                        / target_codes.var(0, unbiased=False).mean().clamp_min(1e-12)),
                }
                metrics[direction].update(conditional_variance_decomposition(probability, target_codes))
    return EncoderConditionalReadoutResult(metrics, weights, log_distributions)
