"""Numerical regressions for bounded Sinkhorn continuation during InfoOT recovery."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from diffusion_ot.losses import infoot


def _triangular_problem():
    features = torch.tensor([[0.0], [1.0]])
    cost = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    options = dict(
        cross_cost=cost,
        mi_weight=0.0,
        entropy_epsilon=0.1,
        inner_iterations=1,
        projection_iterations=100,
        projection_tolerance=1e-3,
        outer_tolerance=1e-5,
        min_inner_iterations=5,
        outer_patience=3,
        strict_convergence=True,
        require_outer_convergence=True,
        recovery_iterations=100,
    )
    return features, options


def test_discarding_recovery_duals_repeats_a_stable_but_inaccurate_plan(monkeypatch):
    original = infoot._sinkhorn_transport_with_duals

    def cold_reset(*args, **kwargs):
        kwargs.pop("initial_log_v", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(infoot, "_sinkhorn_transport_with_duals", cold_reset)
    features, options = _triangular_problem()
    with pytest.raises(RuntimeError) as error:
        infoot.solve_infoot(features, features, **options)
    message = str(error.value)
    assert "after 101 iterations" in message
    assert "plan_delta_l1=0," in message
    assert "sinkhorn_converged=True" in message  # loose output feasibility passes
    assert "effective_projection_tolerance=5e-07" in message
    assert "last_inner_converged=False" in message  # stricter recovery accuracy fails
    assert "last_20_delta_range=[0, 0]" in message


def test_warm_recovery_converges_to_analytic_plan_without_rng_or_dtype_changes(monkeypatch):
    original = infoot._sinkhorn_transport_with_duals
    recovery_calls = []

    def observed(cost, a, b, **kwargs):
        plan, log_v = original(cost, a, b, **kwargs)
        if cost.dtype == torch.float64:
            residual = max(float((plan.sum(1) - a).abs().max()),
                           float((plan.sum(0) - b).abs().max()))
            recovery_calls.append((residual, kwargs["tolerance"], kwargs.get("initial_log_v")))
        return plan, log_v

    monkeypatch.setattr(infoot, "_sinkhorn_transport_with_duals", observed)
    features, options = _triangular_problem()
    rng_before = torch.get_rng_state().clone()
    result = infoot.solve_infoot(features, features, **options)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    assert result.coupling.dtype == features.dtype == torch.float32
    assert result.outer_converged and result.sinkhorn_converged
    assert result.plan_delta_l1 <= options["outer_tolerance"]
    assert max(result.row_residual, result.column_residual) <= options["projection_tolerance"]
    assert result.effective_projection_tolerance == pytest.approx(5e-7)
    assert 1 < result.recovery_iterations < 10
    assert recovery_calls[-1][0] <= recovery_calls[-1][1] == pytest.approx(5e-7)
    assert any(log_v is not None for _, _, log_v in recovery_calls)
    # With equal marginals, diagonal/off-diagonal odds are exp(-1 / (2 epsilon)).
    q = 0.5 / (1.0 + math.exp(5.0))
    exact = torch.tensor([[q, 0.5 - q], [0.5 - q, q]], dtype=torch.float64)
    torch.testing.assert_close(result.coupling.double(), exact, rtol=0, atol=3e-7)
    repeat = infoot.solve_infoot(features, features, **options)
    torch.testing.assert_close(repeat.coupling, result.coupling, rtol=0, atol=0)
    assert repeat.iterations == result.iterations


def test_two_bounded_dual_continuations_equal_one_uninterrupted_sinkhorn_solve():
    cost = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    marginal = torch.full((2,), 0.5, dtype=torch.float64)
    kwargs = dict(regularization=0.1, tolerance=1e-30)
    first, log_v = infoot._sinkhorn_transport_with_duals(
        cost, marginal, marginal, max_iterations=100, **kwargs,
    )
    saved_log_v = log_v.clone()
    continued, continued_log_v = infoot._sinkhorn_transport_with_duals(
        cost, marginal, marginal, initial_log_v=log_v, max_iterations=100, **kwargs,
    )
    uninterrupted, uninterrupted_log_v = infoot._sinkhorn_transport_with_duals(
        cost, marginal, marginal, max_iterations=200, **kwargs,
    )
    torch.testing.assert_close(log_v, saved_log_v, rtol=0, atol=0)
    torch.testing.assert_close(continued, uninterrupted, rtol=0, atol=0)
    torch.testing.assert_close(continued_log_v, uninterrupted_log_v, rtol=0, atol=0)
    assert (continued.sum(1) - marginal).abs().max() < (first.sum(1) - marginal).abs().max()


def test_warm_start_replaces_cost_instead_of_accumulating_old_cost():
    a = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    b = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64)
    old_cost = torch.tensor([[0.0, 0.2, 0.6], [0.5, 0.0, 0.3], [0.8, 0.1, 0.4]], dtype=torch.float64)
    new_cost = torch.tensor([[0.6, 0.0, 0.1], [0.1, 0.7, 0.2], [0.4, 0.2, 0.0]], dtype=torch.float64)
    kwargs = dict(regularization=0.15, tolerance=1e-12, max_iterations=500)
    _, old_log_v = infoot._sinkhorn_transport_with_duals(old_cost, a, b, **kwargs)
    warm, _ = infoot._sinkhorn_transport_with_duals(new_cost, a, b, initial_log_v=old_log_v, **kwargs)
    cold = infoot.sinkhorn_transport_from_cost(new_cost, a, b, **kwargs)
    accumulated = infoot.sinkhorn_transport_from_cost(old_cost + new_cost, a, b, **kwargs)
    torch.testing.assert_close(warm, cold, atol=2e-12, rtol=0)
    torch.testing.assert_close(warm.sum(1), a, atol=1e-12, rtol=0)
    torch.testing.assert_close(warm.sum(0), b, atol=1e-12, rtol=0)
    assert (warm - accumulated).abs().max() > 0.01


def test_warm_solution_matches_pot_log_sinkhorn():
    ot = pytest.importorskip("ot")
    cost = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    marginal = torch.full((2,), 0.5, dtype=torch.float64)
    _, log_v = infoot._sinkhorn_transport_with_duals(
        cost, marginal, marginal, regularization=0.1, tolerance=1e-12, max_iterations=100,
    )
    warm, _ = infoot._sinkhorn_transport_with_duals(
        cost, marginal, marginal, initial_log_v=log_v,
        regularization=0.1, tolerance=1e-12, max_iterations=2000,
    )
    reference = ot.bregman.sinkhorn_log(
        marginal.numpy(), marginal.numpy(), cost.numpy(), 0.1,
        numItermax=3000, stopThr=1e-13,
    )
    torch.testing.assert_close(warm, torch.from_numpy(reference), atol=2e-12, rtol=0)


@pytest.mark.parametrize("bad_log_v", [
    torch.zeros(3), torch.zeros(1, 2),
    torch.tensor([0.0, float("nan")]), torch.tensor([float("inf"), 0.0]),
])
def test_invalid_warm_scaling_is_rejected(bad_log_v):
    with pytest.raises(ValueError, match="warm-start log_v must be finite and match"):
        infoot._sinkhorn_transport_with_duals(
            torch.zeros(2, 2), torch.full((2,), 0.5), torch.full((2,), 0.5),
            initial_log_v=bad_log_v, regularization=0.1, max_iterations=10, tolerance=1e-6,
        )
