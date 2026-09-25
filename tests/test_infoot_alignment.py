"""Mathematical and gradient checks for detached-plan neural InfoOT losses."""
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.losses.encoder_transport import encoder_transport_cost, relative_encoder_transport_loss
from diffusion_ot.losses.infoot import (
    coupling_entropy, gaussian_kernel, infoot_distance_scale,
    infoot_mutual_information, plain_infoot_feature_loss, solve_infoot,
)
from diffusion_ot.losses.infoot_alignment import (
    InfoOTAlignmentOptions, infoot_alignment_loss, infoot_alignment_options,
)
from diffusion_ot.losses.semantic_prior import validate_prior_resume
from diffusion_ot.training.stage1b_logging import Stage1BLogFormatter


def banks():
    rng = torch.Generator().manual_seed(19)
    x = torch.randn(4, 6, generator=rng, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, 6, generator=rng, dtype=torch.float64, requires_grad=True)
    plan = torch.full((4, 5), .05, dtype=torch.float64)
    plan[:2, :2] += torch.tensor([[.025, -.025], [-.025, .025]], dtype=torch.float64)
    return x, y, plan.requires_grad_()


def loss(x, y, plan, *, objective="full", relative=.01, entropy=.02, cost_weight=1.):
    return infoot_alignment_loss(x, y, plan,
        options=InfoOTAlignmentOptions(objective, relative), bandwidth=.55,
        distance_scale_x=infoot_distance_scale(x, detach=False),
        distance_scale_y=infoot_distance_scale(y, detach=False),
        mi_weight=.1, entropy_epsilon=entropy, cross_cost_weight=cost_weight)


def test_full_objective_matches_formula_with_live_cost_and_detached_plan():
    x, y, plan = banks()
    result = loss(x, y, plan, cost_weight=.7)
    gamma = plan.detach()
    cost = 1 - F.normalize(x, dim=1) @ F.normalize(y, dim=1).T
    kernels = [gaussian_kernel(v, bandwidth=.55, distance_scale=infoot_distance_scale(v, detach=False))
               for v in (x, y)]
    expected = .7 * (gamma * cost).sum() - .1 * infoot_mutual_information(gamma, *kernels) - .02 * coupling_entropy(gamma)
    torch.testing.assert_close(result.loss, expected)
    actual_grads = torch.autograd.grad(result.loss, (x, y, plan), retain_graph=True, allow_unused=True)
    expected_grads = torch.autograd.grad(expected, (x, y))
    assert actual_grads[2] is None
    for actual, reference in zip(actual_grads, expected_grads):
        assert actual.norm() > 0
        torch.testing.assert_close(actual, reference)
    assert not result.entropy_loss.requires_grad


def test_relative_term_matches_centered_formula_and_live_independent_baseline():
    x, y, plan = banks()
    relative = relative_encoder_transport_loss(x, y, plan)
    nx, ny = F.normalize(x, dim=1), F.normalize(y, dim=1)
    centered = -(plan.detach() * ((nx - nx.mean(0)) @ (ny - ny.mean(0)).T)).sum()
    cost = 1 - nx @ ny.T
    gap = (plan.detach() * cost).sum() - cost.mean()
    torch.testing.assert_close(relative, centered, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(relative, gap, atol=1e-14, rtol=1e-12)
    gradients = torch.autograd.grad(relative, (x, y, plan), retain_graph=True, allow_unused=True)
    expected = torch.autograd.grad(gap, (x, y), retain_graph=True)
    wrong = torch.autograd.grad((plan.detach() * cost).sum() - cost.mean().detach(), (x, y))
    assert gradients[2] is None
    for actual, correct, detached_baseline in zip(gradients, expected, wrong):
        torch.testing.assert_close(actual, correct, atol=1e-14, rtol=1e-12)
        assert not torch.allclose(actual, detached_baseline)


def test_combined_full_and_relative_gradients_pass_finite_differences():
    x, y, plan = banks()
    def evaluate(left, right):
        result = loss(left, right, plan)
        return .2 * result.loss + .01 * result.relative_loss
    assert torch.autograd.gradcheck(evaluate, (x, y), fast_mode=True)


def test_entropy_changes_scalar_but_not_feature_gradients():
    x, y, plan = banks()
    first, second = loss(x, y, plan, entropy=.02), loss(x, y, plan, entropy=.2)
    torch.testing.assert_close(second.loss - first.loss, -.18 * coupling_entropy(plan.detach()))
    for a, b in zip(torch.autograd.grad(first.loss, (x, y)), torch.autograd.grad(second.loss, (x, y))):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_full_neural_value_matches_fitted_solver_objective():
    x, y, _ = banks()
    sx, sy = [infoot_distance_scale(v) for v in (x, y)]
    solved = solve_infoot(x, y, cross_cost=encoder_transport_cost(x, y), cross_cost_weight=.7,
        bandwidth=.55, distance_scale_x=sx, distance_scale_y=sy, mi_weight=.1, entropy_epsilon=.2,
        inner_iterations=300, projection_iterations=2000, outer_tolerance=1e-5,
        strict_convergence=True, require_outer_convergence=True)
    result = loss(x, y, solved.coupling, entropy=.2, cost_weight=.7)
    assert float(result.loss.detach()) == pytest.approx(solved.objective, abs=1e-10)


def test_old_mi_objective_and_detached_cost_default_are_unchanged():
    x, y, plan = banks()
    options = infoot_alignment_options({})
    assert options == InfoOTAlignmentOptions() and not options.extended
    result = loss(x, y, plan, objective="mi", relative=0)
    old = plain_infoot_feature_loss(x, y, plan.detach(), mi_weight=.1, bandwidth=.55,
        distance_scale_x=infoot_distance_scale(x, detach=False), distance_scale_y=infoot_distance_scale(y, detach=False))
    torch.testing.assert_close(result.loss, old, rtol=0, atol=0)
    for a, b in zip(torch.autograd.grad(result.loss, (x, y)), torch.autograd.grad(old, (x, y))):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert result.metrics(alignment_weight=.2, relative_weight=0) == {}
    detached = encoder_transport_cost(x, y)
    live = encoder_transport_cost(x, y, detach=False)
    assert not detached.requires_grad and live.requires_grad
    torch.testing.assert_close(detached, live, rtol=0, atol=0)
    with torch.no_grad():
        assert not encoder_transport_cost(x, y, detach=False).requires_grad


def test_relative_term_has_no_signal_for_independent_plan():
    x, y, _ = banks()
    result = relative_encoder_transport_loss(x, y, torch.full((4, 5), .05, dtype=x.dtype))
    assert abs(float(result.detach())) < 1e-14
    for grad in torch.autograd.grad(result, (x, y)):
        assert grad.norm() < 1e-14


def test_relative_term_opposes_common_cone_contraction_for_informative_plan():
    rng = torch.Generator().manual_seed(19)
    u = F.normalize(torch.randn(8, 5, generator=rng, dtype=torch.float64), dim=1)
    v = F.normalize(u + .3 * torch.randn(8, 5, generator=rng, dtype=torch.float64), dim=1)
    plan = torch.eye(8, dtype=torch.float64) / 8
    raw, relative = [], []
    for spread in (.8, .4, .2):
        t = torch.tensor(spread, dtype=torch.float64, requires_grad=True)
        x, y = [torch.cat([(1-t*t).sqrt().expand(8, 1), t * value], dim=1) for value in (u, v)]
        cost = (plan * encoder_transport_cost(x, y, detach=False)).sum()
        contrast = relative_encoder_transport_loss(x, y, plan)
        raw.append(float(cost.detach()))
        relative.append(float(contrast.detach()))
        assert torch.autograd.grad(contrast, t)[0] < 0
    assert raw[0] > raw[1] > raw[2]
    assert relative[0] < relative[1] < relative[2] < 0


def config():
    return {"infoot": {"variant": "fused", "cross_cost_source": "encoder", "feature_objective": "full"},
            "matching": {"distance_scale_gradient": "full"},
            "loss_weights": {"infoot_alignment": .2, "infoot_relative": .01}}


@pytest.mark.parametrize("section,key,value", [
    ("infoot", "feature_objective", "typo"), ("infoot", "cross_cost_source", "dino"),
    ("infoot", "variant", "plain"), ("matching", "distance_scale_gradient", "detached"),
    ("loss_weights", "infoot_relative", -1), ("loss_weights", "infoot_relative", float("nan")),
    ("infoot", "cross_cost_weight", float("inf")),
])
def test_reject_invalid_objectives_before_training(section, key, value):
    cfg = config()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        infoot_alignment_options(cfg)


def test_resume_cannot_silently_change_objective_or_relative_weight():
    cfg = config()
    validate_prior_resume(cfg, deepcopy(cfg))
    for section, key, value in (("infoot", "feature_objective", "mi"), ("loss_weights", "infoot_relative", .02)):
        other = deepcopy(cfg)
        other[section][key] = value
        with pytest.raises(ValueError, match="InfoOT neural objectives"):
            validate_prior_resume(cfg, other)


def test_logs_retain_full_cost_with_zero_mi_and_relative_at_zero_warmup():
    cfg = config()
    cfg["infoot"]["mi_weight"] = 0
    fmt = Stage1BLogFormatter(cfg)
    logged = fmt.format({"infoot_feature_loss": .4, "infoot_relative_loss": -.2,
                         "weighted_infoot_relative_loss": 0., "infoot_relative_weight": 0.})
    assert {"infoot_alignment", "infoot_relative"} <= set(logged["enabled_losses"])
    assert logged["weighted_infoot_relative_loss"] == 0
    cfg["loss_weights"]["infoot_relative"] = 0
    without = Stage1BLogFormatter(cfg).format(logged)
    assert not any("relative" in key for key in without)
