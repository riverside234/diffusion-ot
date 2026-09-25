"""Neural InfoOT objectives evaluated at a detached fitted transport plan.

The full objective differentiates the encoder cost and MI kernels (including
live RMS scales supplied by the caller). Entropy affects its value, but not
neural gradients. The optional relative term is separately weighted outside
the full objective; it never replaces or reweights the inner solver's cost.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from diffusion_ot.losses.encoder_transport import encoder_transport_cost, relative_encoder_transport_loss
from diffusion_ot.losses.infoot import coupling_entropy, plain_infoot_feature_loss


@dataclass(frozen=True)
class InfoOTAlignmentOptions:
    feature_objective: str = "mi"
    relative_weight: float = 0.0

    @property
    def extended(self) -> bool:
        return self.feature_objective == "full" or self.relative_weight > 0


def infoot_alignment_options(config: dict) -> InfoOTAlignmentOptions:
    infoot = config.get("infoot") or {}
    weights = config.get("loss_weights") or {}
    matching = config.get("matching") or {}
    objective = infoot.get("feature_objective", "mi")
    if objective not in {"mi", "full"}:
        raise ValueError("infoot.feature_objective must be mi or full.")
    relative = float(weights.get("infoot_relative", 0.0))
    if not math.isfinite(relative) or relative < 0:
        raise ValueError("loss_weights.infoot_relative must be finite and nonnegative.")
    options = InfoOTAlignmentOptions(objective, relative)
    if options.extended:
        if infoot.get("variant", "plain") != "fused" or infoot.get("cross_cost_source", "dino") != "encoder":
            raise ValueError("Full/relative neural InfoOT requires fused, encoder-derived cross cost.")
        if (matching.get("distance_scale", "infoot_rms") != "infoot_rms"
                or matching.get("distance_scale_gradient", "detached") != "full"):
            raise ValueError("Full/relative neural InfoOT requires full live infoot_rms gradients.")
        for name, value in (("infoot_alignment", weights.get("infoot_alignment", .02)),
                            ("cross_cost_weight", infoot.get("cross_cost_weight", 1.0)),
                            ("mi_weight", infoot.get("mi_weight", 1.0)),
                            ("entropy_epsilon", infoot.get("entropy_epsilon", .05))):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
    return options


@dataclass
class InfoOTAlignmentResult:
    loss: torch.Tensor
    mi_loss: torch.Tensor
    cost_loss: torch.Tensor
    entropy_loss: torch.Tensor
    relative_loss: torch.Tensor
    options: InfoOTAlignmentOptions

    def metrics(self, *, alignment_weight: float, relative_weight: float) -> dict[str, float]:
        """Raw terms include their inner coefficients; weighted terms include ramps."""
        values = {}
        if self.options.feature_objective == "full":
            for name in ("cost", "mi", "entropy"):
                value = float(getattr(self, f"{name}_loss").detach())
                values[f"infoot_{name}_loss"] = value
                values[f"weighted_infoot_{name}_loss"] = alignment_weight * value
        if self.options.relative_weight > 0:
            value = float(self.relative_loss.detach())
            values.update(infoot_relative_loss=value, infoot_relative_weight=relative_weight,
                          weighted_infoot_relative_loss=relative_weight * value)
        return values


def infoot_alignment_loss(
    cat: torch.Tensor, dog: torch.Tensor, coupling: torch.Tensor, *,
    options: InfoOTAlignmentOptions,
    bandwidth: float,
    distance_scale_x: float | torch.Tensor,
    distance_scale_y: float | torch.Tensor,
    mi_weight: float,
    entropy_epsilon: float,
    cross_cost_weight: float = 1.0,
    eps: float = 1e-8,
) -> InfoOTAlignmentResult:
    # Stop gradients through ALL occurrences of the fitted plan, including MI
    # and entropy. No unrolling or implicit differentiation of the solver.
    plan = coupling.detach()
    mi_loss = plain_infoot_feature_loss(
        cat, dog, plan, bandwidth=bandwidth,
        distance_scale_x=distance_scale_x, distance_scale_y=distance_scale_y,
        mi_weight=mi_weight, eps=eps,
    )
    cost_loss = entropy_loss = relative_loss = mi_loss.new_zeros(())
    if options.feature_objective == "full":
        cost = encoder_transport_cost(cat, dog, detach=False)
        cost_loss = float(cross_cost_weight) * (plan.to(cost) * cost).sum()
        entropy_loss = -float(entropy_epsilon) * coupling_entropy(plan, eps=eps)
    if options.relative_weight > 0:
        relative_loss = relative_encoder_transport_loss(cat, dog, plan)
    return InfoOTAlignmentResult(mi_loss + cost_loss + entropy_loss,
                                 mi_loss, cost_loss, entropy_loss, relative_loss, options)
