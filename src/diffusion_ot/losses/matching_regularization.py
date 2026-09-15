"""Variance/covariance protection for unit-normalized matching features.

Inspired by VICReg (https://arxiv.org/abs/2105.04906), with dimension-aware
scaling for this project's unit vectors. The loss uses matching outputs;
raw decoder codes and conditional means are not regularization targets.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass
class MatchingRegularizationResult:
    loss: torch.Tensor
    metrics: dict[str, Any]


def _validate_options(std_target: float, variance_weight: float, covariance_weight: float, eps: float) -> None:
    if not math.isfinite(std_target) or not 0 < std_target <= 1:
        raise ValueError("matching_regularization.std_target must be in (0, 1] for scaled unit features.")
    if not math.isfinite(variance_weight) or variance_weight <= 0:
        raise ValueError("matching_regularization.variance_weight must be finite and positive.")
    if not math.isfinite(covariance_weight) or covariance_weight < 0:
        raise ValueError("matching_regularization.covariance_weight must be finite and nonnegative.")
    if not math.isfinite(eps) or not 0 < eps < std_target ** 2:
        raise ValueError("matching_regularization.eps must be positive and smaller than std_target squared.")


def matching_regularization_options(config: dict[str, Any]) -> dict[str, float] | None:
    if not config.get("enabled", False):
        return None
    allowed = {"enabled", "std_target", "variance_weight", "covariance_weight", "eps"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown matching_regularization settings: {sorted(unknown)}")
    options = {key: float(config.get(key, default)) for key, default in {
        "std_target": .7, "variance_weight": .02, "covariance_weight": .001, "eps": 1e-4,
    }.items()}
    _validate_options(**options)
    return options


def matching_regularization_loss(
    features: dict[str, torch.Tensor], *, std_target: float = .7,
    variance_weight: float = .02, covariance_weight: float = .001, eps: float = 1e-4,
) -> MatchingRegularizationResult:
    """Average independent domain penalties on post-L2 OT reference features.

    For M in R^[N,D], use U=sqrt(D)*M, population covariance C=Uc.T@Uc/N,
    V=mean(relu(std_target-sqrt(diag(C)+eps))) and
    R=sum_{i!=j}(C_ij^2)/(D*(D-1)). Unlike VICReg's 1/D covariance reduction,
    this averages off-diagonal entries so its scale does not grow with D.
    No empirical batch-variance normalization is applied: it would cancel V.

    Inputs come from matching_features and are already unit-normalized; they
    are not modified. Gradients reach E and matching heads through those inputs.
    A constant bank has a finite loss/gradient, but at exact collapse its
    variance gradient is zero. This is preventive regularization, not a reset.
    """
    _validate_options(std_target, variance_weight, covariance_weight, eps)
    if not features:
        raise ValueError("Matching regularization needs at least one feature bank.")
    variances, covariances, domains = [], [], {}
    for domain, values in features.items():
        if values.ndim != 2 or min(values.shape) < 2:
            raise ValueError("Matching regularization needs [N, D] features with N >= 2 and D >= 2.")
        # Preserve float64 for numerical checks; training accumulation is fp32.
        values = values if values.dtype == torch.float64 else values.float()
        count, dimension = values.shape
        with torch.autocast(device_type=values.device.type, enabled=False):
            centered = (values - values.mean(0)) * math.sqrt(dimension)
            per_dimension_variance = centered.square().mean(0)
            std = (per_dimension_variance + eps).sqrt()
            variance = (std_target - std).clamp_min(0).mean()
            covariance = centered.T @ centered / count
            off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
            decorrelation = off_diagonal.square().sum() / (dimension * (dimension - 1))
        variances.append(variance)
        covariances.append(decorrelation)
        with torch.no_grad():
            domains[domain] = {
                "samples": count, "feature_dim": dimension,
                "variance_loss": float(variance), "covariance_loss": float(decorrelation),
                "matching_variance": float(per_dimension_variance.mean()),
                "scaled_std_mean": float(std.mean()), "scaled_std_min": float(std.min()),
                "fraction_below_std_target": float((std < std_target).float().mean()),
                "raw_std_floor": math.sqrt((std_target ** 2 - eps) / dimension),
            }
    variance = torch.stack(variances).mean()
    covariance = torch.stack(covariances).mean()
    weighted_variance, weighted_covariance = variance_weight * variance, covariance_weight * covariance
    loss = weighted_variance + weighted_covariance
    metrics = {
        "sample_scope": "ot_references", "std_target": std_target,
        "feature_scale": "sqrt_dimension", "variance_estimator": "population",
        "covariance_reduction": "mean_squared_offdiagonal",
        "variance_weight": variance_weight, "covariance_weight": covariance_weight,
        "variance_loss": float(variance.detach()), "covariance_loss": float(covariance.detach()),
        "weighted_variance_loss": float(weighted_variance.detach()),
        "weighted_covariance_loss": float(weighted_covariance.detach()), "weighted_loss": float(loss.detach()),
        **domains,
    }
    return MatchingRegularizationResult(loss, metrics)
