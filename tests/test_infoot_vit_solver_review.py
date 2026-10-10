"""Numerical failures and diagnostics found in the offline solver review."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infoot_vit.infoot_helper import infoot as legacy
from infoot_vit.infoot_helper import partial
from infoot_vit.infoot_helper import conditional
from infoot_vit.infoot_helper.conditional import BalancedModel, _balanced_mi_gradient


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_balanced_gradient_matches_existing_objective_and_healthy_legacy():
    x = torch.tensor([[0.], [.4], [2.]], dtype=torch.float64)
    kx, _ = partial.kernel_state(x, .8)
    ky, _ = partial.kernel_state(x + .2, .5)
    plan = torch.tensor([[.1, .03, .02], [.01, .13, .04], [.03, .06, .18]], dtype=torch.float64)
    actual, floored = _balanced_mi_gradient(plan, kx, ky, 1e-300)
    torch.testing.assert_close(actual, legacy.migrad(plan, kx, ky), atol=1e-12, rtol=1e-12)
    assert floored == 0
    variable = plan.clone().requires_grad_(True)
    loss = legacy.fitting_loss(variable, kx, ky, reg=0., eps=1e-300)
    expected, = torch.autograd.grad(loss, variable)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_balanced_gradient_handles_zero_density_and_respects_log_floor():
    kernel = torch.eye(2, dtype=torch.float64)
    plan = torch.tensor([[.5, 0.], [1e-6, .499999]], dtype=torch.float64)
    assert not torch.isfinite(legacy.migrad(plan, kernel, kernel)).all()
    actual, floored = _balanced_mi_gradient(plan, kernel, kernel, 1e-4)
    variable = plan.clone().requires_grad_(True)
    # Compare the exact negative MI derivative independently of the entropy
    # term, whose gradient at a zero plan entry is not needed by this solver.
    loss = -(variable * legacy.ratio(variable, kernel, kernel).clamp_min(1e-4).log()).sum()
    expected, = torch.autograd.grad(loss, variable)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert floored == 2 and torch.isfinite(actual).all()


def test_balanced_narrow_kernel_low_entropy_fit_remains_finite():
    x = torch.tensor([[0.], [1.]], dtype=torch.float64)
    observed = []
    model = BalancedModel.fit(x, x, dict(h=.01, lam=.1, reg=.001, cost_scale=1.),
                              on_step=lambda plan, report: observed.append(report))
    assert model.state["status"] == "converged"
    assert observed[-1]["gradient_log_floor_entries"] == 2
    torch.testing.assert_close(model.plan, torch.eye(2, dtype=torch.float64) * .5)
    torch.testing.assert_close(model.project(x), x)


def test_balanced_backtracking_checks_full_objective_and_keeps_marginals(monkeypatch):
    x = torch.tensor([[0.], [1.]], dtype=torch.float64)
    candidate = torch.eye(2, dtype=torch.float64) * .5
    monkeypatch.setattr(conditional, "entropy_subproblem", lambda *a: (candidate, {"error": 0.}))
    observed = []
    options = dict(h=.7, lam=.01, reg=2., max_outer_steps=1)
    model = BalancedModel.fit(x, x, options, on_step=lambda p, r: observed.append((p.clone(), r)))
    # The full step is uphill, while a half step decreases the actual InfoOT
    # objective. This exercises damping independently of the inner solver.
    row = model.state["history"][-1]
    assert row["status"] == "max_outer_steps" and row["step_size"] == .5
    assert row["backtracks"] == 1 and row["objective_delta"] < 0
    assert row["accepted_plan_delta_l1"] == pytest.approx(.5 * row["plan_delta_l1"])
    expected = .5 * (torch.full_like(candidate, .25) + candidate)
    torch.testing.assert_close(model.plan, expected, atol=0, rtol=0)
    torch.testing.assert_close(observed[0][0], expected, atol=0, rtol=0)
    torch.testing.assert_close(model.plan.sum(0), x.new_full((2,), .5))
    torch.testing.assert_close(model.plan.sum(1), x.new_full((2,), .5))
    solver = legacy.FusedInfoOT(x, x, h=.7)
    loss = legacy.fitting_loss(expected, solver.Ks, solver.Kt, reg=2.,
                               C=solver.C / solver.C.mean(), mi_weight=.01, eps=1e-300)
    assert row["objective"] == pytest.approx(float(loss), abs=1e-12)


@pytest.mark.parametrize("options", [dict(max_backtracks=1), dict(outer_tolerance=.6)])
def test_balanced_stalled_update_is_not_reported_as_convergence(monkeypatch, options):
    x = torch.tensor([[0.], [1.]], dtype=torch.float64)
    candidate = torch.eye(2, dtype=torch.float64) * .5
    monkeypatch.setattr(conditional, "entropy_subproblem", lambda *a: (candidate, {"error": 0.}))
    observed = []
    model = BalancedModel.fit(x, x, dict(h=.7, lam=.01, reg=2., **options),
        on_step=lambda p, r: observed.append((p.clone(), r)))
    row = model.state["history"][-1]
    assert model.state["status"] == row["status"] == "line_search_stalled"
    assert row["plan_delta_l1"] > model.state["config"]["outer_tolerance"]
    assert row["accepted_plan_delta_l1"] == row["objective_delta"] == row["step_size"] == 0.
    torch.testing.assert_close(model.plan, torch.full_like(candidate, .25), atol=0, rtol=0)
    torch.testing.assert_close(observed[-1][0], model.plan, atol=0, rtol=0)


def test_balanced_converged_plan_is_a_checked_fixed_point():
    x = torch.tensor([[0.], [.4], [2.]], dtype=torch.float64)
    y = torch.tensor([[.1], [.6], [1.8], [2.5]], dtype=torch.float64)
    model = BalancedModel.fit(x, y, dict(h=.7, lam=.075, reg=.075, max_outer_steps=1200))
    cfg = model.state["config"]
    assert model.state["status"] == "converged"
    assert model.state["history"][-1]["plan_delta_l1"] <= cfg["outer_tolerance"]
    reference = legacy.FusedInfoOT(x, y, h=cfg["h"])
    grad, _ = _balanced_mi_gradient(model.plan, reference.Ks, reference.Kt, cfg["log_floor"])
    candidate, _ = partial.entropy_subproblem(x.new_full((len(x),), 1 / len(x)),
        y.new_full((len(y),), 1 / len(y)), reference.C / reference.C.mean() + cfg["lam"] * grad, 1., cfg)
    assert float((candidate - model.plan).abs().sum()) <= cfg["outer_tolerance"]
    assert all(row["objective_delta"] <= cfg["objective_tolerance"] for row in model.state["history"])
    for direction in ("source", "target"):
        assert model.state["kernel_diagnostics"][direction]["mean_effective_neighbors"] >= 1.


def test_partial_stall_is_emitted_with_last_accepted_plan(monkeypatch):
    cost = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    kernel = torch.eye(2, dtype=torch.float64)
    candidate = torch.tensor([[0., .4], [.4, 0.]], dtype=torch.float64)
    monkeypatch.setattr(partial, "entropy_subproblem", lambda *a: (candidate, {"method": "test_uphill_candidate"}))
    observed = []
    plan, report = partial.solve_partial(cost, kernel, kernel, config=dict(
        lam=.1, reg=.05, cost_scale=1., max_backtracks=1),
        on_step=lambda saved, row: observed.append((saved.clone(), row.copy())))
    assert report["status"] == "line_search_stalled"
    assert len(observed) == 1
    torch.testing.assert_close(observed[0][0], plan)
    torch.testing.assert_close(plan, torch.full((2, 2), .2, dtype=torch.float64))
    row = observed[0][1]
    assert row == report["history"][-1]
    assert row["status"] == "line_search_stalled" and row["backtracks"] == 1
    assert row["accepted_plan_delta_l1"] == row["objective_delta"] == 0.
    assert row["plan_delta_l1"] > 0.


def test_partial_trace_distinguishes_accepted_steps_and_outer_budget():
    x = torch.tensor([[0.], [1.], [2.]], dtype=torch.float64)
    kernel, _ = partial.kernel_state(x, .7)
    observed = []
    _, report = partial.solve_partial(partial.distance(x, x), kernel, kernel,
        config=dict(lam=.1, reg=.2, max_outer_steps=1), on_step=lambda p, r: observed.append(r))
    assert report["status"] == observed[-1]["status"] == "max_outer_steps"
    row = observed[-1]
    assert row["accepted_plan_delta_l1"] == pytest.approx(row["step_size"] * row["plan_delta_l1"])
    assert row["objective_delta"] <= 1e-12 and row["backtracks"] >= 0
