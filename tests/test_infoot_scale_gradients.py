"""RMS-normalized neural kernels must differentiate their normalization.

The plan is fixed in these tests: they check the neural update, not gradients
through the alternating OT solver. Historical detached updates stay available.
"""
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.losses.infoot import (
    conditional_reference_log_weights, gaussian_kernel, infoot_cross_distance_scale,
    infoot_distance_scale, plain_infoot_feature_loss,
)
from diffusion_ot.losses.semantic_prior import validate_prior_resume
from diffusion_ot.models.matching_head import matching_geometry_diagnostics


@pytest.mark.parametrize("objective", ["conditional", "mi"])
def test_full_rms_neural_gradients_match_finite_differences(objective):
    torch.manual_seed(20260915)
    q, x, y = [torch.randn(*shape, dtype=torch.float64, requires_grad=True)
               for shape in ((2, 3), (4, 3), (5, 3))]
    plan = torch.rand(4, 5, dtype=torch.float64)
    plan /= plan.sum()

    def loss(q, x, y, detach=False):
        if objective == "conditional":
            return conditional_reference_log_weights(
                q, x, y, plan, bandwidth=.7,
                distance_scale_x=infoot_cross_distance_scale(q, x, detach=detach),
                distance_scale_y=infoot_distance_scale(y, detach=detach),
            )
        return plain_infoot_feature_loss(
            x, y, plan, bandwidth=.7,
            distance_scale_x=infoot_distance_scale(x, detach=detach),
            distance_scale_y=infoot_distance_scale(y, detach=detach),
        )

    torch.testing.assert_close(loss(q, x, y), loss(q, x, y, detach=True))
    assert torch.autograd.gradcheck(loss, (q, x, y), atol=1e-5, rtol=1e-4)


def test_normalized_cone_has_no_spurious_rms_shrinkage_gradient():
    torch.manual_seed(20260915)
    directions = [F.normalize(torch.randn(*shape, dtype=torch.float64), dim=1)
                  for shape in ((4, 5), (6, 5), (7, 5))]
    plan = torch.rand(6, 7, dtype=torch.float64)
    plan /= plan.sum()

    def loss(concentration, detach):
        # Unit vectors whose pairwise distances all scale by concentration.
        q, x, y = [torch.cat((concentration * u,
                              (1-concentration.square()).sqrt().expand(len(u), 1)), 1)
                   for u in directions]
        return -conditional_reference_log_weights(
            q, x, y, plan, bandwidth=.3,
            distance_scale_x=infoot_cross_distance_scale(q, x, detach=detach),
            distance_scale_y=infoot_distance_scale(y, detach=detach),
        ).mean()

    concentration = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    historical = loss(concentration, True)
    corrected = loss(concentration, False)
    full_gradient, = torch.autograd.grad(corrected, concentration)
    historical_gradient, = torch.autograd.grad(historical, concentration)
    torch.testing.assert_close(corrected, loss(concentration.detach() * 2, False))
    numerical = (loss(concentration.detach()+1e-5, False)
                 - loss(concentration.detach()-1e-5, False)) / 2e-5
    assert abs(float(full_gradient)) < 1e-10
    assert abs(float(numerical)) < 1e-9
    assert float(historical_gradient) > .5  # The reproduced wrong shrinkage direction.


def test_collapsed_bank_has_finite_rms_gradients():
    features = torch.ones(5, 3, dtype=torch.float64, requires_grad=True)
    scale = infoot_distance_scale(features, detach=False)
    assert scale.item() == pytest.approx(1e-8)
    gaussian_kernel(features, distance_scale=scale).sum().backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.count_nonzero() == 0
    assert not infoot_distance_scale(features).requires_grad


def test_geometry_distinguishes_head_concentration_from_decoder_collapse():
    codes = torch.eye(4)
    identity = matching_geometry_diagnostics(codes, codes)
    concentrated = matching_geometry_diagnostics(codes, codes[:1].expand(4, -1))
    assert identity["matching_variance"] == pytest.approx(.75)
    assert identity["matching_covariance_effective_rank"] == pytest.approx(3)
    assert concentrated["raw_normalized_variance"] == pytest.approx(.75)
    assert concentrated["matching_variance"] == 0
    assert concentrated["matching_covariance_effective_rank"] == 0
    assert concentrated["matching_mean_norm"] == 1


def test_resume_cannot_silently_change_rms_gradient_rule():
    old = {"matching_head": {"enabled": True}, "matching": {"distance_scale": "infoot_rms"}}
    validate_prior_resume(old, deepcopy(old))
    new = deepcopy(old)
    new["matching"]["distance_scale_gradient"] = "full"
    with pytest.raises(ValueError, match="Resume cannot change matching"):
        validate_prior_resume(old, new)
