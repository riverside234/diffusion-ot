"""Numerical failures and diagnostics found in the offline solver review."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infoot_vit.infoot_helper import infoot as legacy
from infoot_vit.infoot_helper import partial
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
