"""Budget-boundary optimality certificates; never accept feasibility alone."""
import math
import json
from pathlib import Path

import pytest
import torch
import yaml

from test_infoot_vit_mapping import cpu_threads
from infoot_vit.infoot_helper.partial import solver_config
from infoot_vit.infoot_helper.partial_batch import (
    _partial_certificate, entropy_subproblem_batch, solve_partial_batch,
)


@pytest.mark.parametrize("cheap,budget", [(145, 990), (146, 690)])
def test_log_residual_budget_stop_can_match_analytic_optimum(cheap, budget):
    # An exact, symmetric 196-patch problem. Tiny tail entries keep the raw
    # log-plan L2 above 1e-10 even after the mass-carrying plan has stabilized.
    n, mass, reg = 196, .75, .075
    cost = torch.ones(1, n, n, dtype=torch.float64)
    cost[:, :cheap, :cheap] = .005
    a = cost.new_full((1, n), 1/n)
    c = solver_config(dict(reg=reg, max_inner_steps=budget))
    plan, report = entropy_subproblem_batch(a, a, cost, mass, c)
    assert report["ok"].all() and report["iterations"].item() == budget
    assert report["error"].item() > c["inner_tolerance"]
    assert report["stop_reason"].item() == 2
    for key in ("relative_plan_delta_l1", "kkt_error", "relative_duality_gap"):
        assert report[key].item() <= c["inner_tolerance"]
    rest, k = n-cheap, math.exp(-.995/reg)
    deficit = mass-cheap/n
    aa, bb, cc = n*deficit*cheap, rest*k*(n*deficit-cheap), -rest**2*k
    exp_u = (-bb+math.sqrt(bb*bb-4*aa*cc))/(2*aa)
    exp_w = (1/n)/(cheap*exp_u**2+rest*k*exp_u)
    expected = torch.full((n, n), k*exp_w, dtype=torch.float64)
    expected[:cheap, :] *= exp_u
    expected[:, :cheap] *= exp_u
    expected[:cheap, :cheap] /= k
    torch.testing.assert_close(plan[0], expected, atol=2e-13, rtol=1e-9)
    if cheap == 145:
        # The same successful inner certificate propagates into persisted outer
        # diagnostics, and mixing a finished neighbor cannot change the plan.
        costs = torch.cat((cost, torch.ones_like(cost)))
        kernels = torch.eye(n, dtype=torch.float64)[None].expand(2, -1, -1)
        plans, reports = solve_partial_batch(costs, kernels, kernels, keep_mass=mass,
            config=dict(c, lam=0., cost_scale=1.))
        assert all(r["status"] == "converged" for r in reports)
        assert reports[0]["history"][-1]["inner"]["convergence_reason"] == "primal_dual_certificate"
        assert reports[1]["history"][-1]["inner"]["iterations"] == 1
        torch.testing.assert_close(plans[0], plan[0], atol=1e-13, rtol=1e-11)
        json.dumps(reports, allow_nan=False)


def test_stable_feasible_but_nonoptimal_plan_has_no_kkt_certificate():
    p = torch.full((1, 2, 2), .2, dtype=torch.float64)
    logp = p.log()
    a = p.new_full((1, 2), .5)
    u = v = p.new_zeros((1, 2))
    base = p.new_tensor([[[0., -4.], [-4., 0.]]])
    delta, kkt = _partial_certificate(p, logp, logp, base, a, a, .8,
                                      u, v, p.new_tensor([math.log(.2)]))
    assert delta.item() == 0. and kkt.item() > 1.


def test_sharper_active_recipe_has_budget_for_saturated_point_eight_mass():
    root = Path(__file__).resolve().parents[1]
    recipe = yaml.safe_load((root / "infoot_vit/configs/grouped_partial.yaml").read_text())["partial"]
    # Preserve the old unaccelerated budget regression as a reference. The
    # accelerated recipe is covered by test_infoot_vit_partial_newton.py.
    c = solver_config(dict(recipe["solver"], inner_acceleration="none"))
    n, cheap, mass = 196, 156, recipe["keep_mass"]
    cost = torch.ones(1, n, n, dtype=torch.float64)
    cost[:, :cheap, :cheap] = .005
    a = cost.new_full((1, n), 1/n)
    _, short = entropy_subproblem_batch(a, a, cost, mass, dict(c, max_inner_steps=10000))
    assert not short["ok"].item() and short["kkt_error"].item() > c["inner_tolerance"]
    plan, report = entropy_subproblem_batch(a, a, cost, mass, c)
    assert report["ok"].item() and 10000 < report["iterations"].item() <= c["max_inner_steps"]
    rest, k = n-cheap, math.exp(-.995/c["reg"])
    deficit = mass-cheap/n
    aa, bb, cc = n*deficit*cheap, rest*k*(n*deficit-cheap), -rest**2*k
    exp_u = (-bb+math.sqrt(bb*bb-4*aa*cc))/(2*aa)
    exp_w = (1/n)/(cheap*exp_u**2+rest*k*exp_u)
    expected = torch.full((n, n), k*exp_w, dtype=torch.float64)
    expected[:cheap, :] *= exp_u
    expected[:, :cheap] *= exp_u
    expected[:cheap, :cheap] /= k
    torch.testing.assert_close(plan[0], expected, atol=2e-13, rtol=1e-9)


def test_capacity_complementarity_is_required_even_with_stationarity():
    p = torch.full((1, 2, 2), .2, dtype=torch.float64)
    a = p.new_full((1, 2), .5)
    u, v = -torch.ones_like(a), torch.zeros_like(a)
    w = p.new_tensor([math.log(.2)])
    # This base makes stationarity exact, but u<0 at rows with unused capacity.
    base = p.log() - u[:, :, None] - v[:, None, :] - w[:, None, None]
    delta, kkt = _partial_certificate(p, p.log(), p.log(), base, a, a, .8, u, v, w)
    assert delta.item() == 0. and kkt.item() == pytest.approx(.25)


def test_budget_check_does_not_promote_inner_solution_to_outer_convergence():
    cost = torch.tensor([[[0., 1.], [1., 0.]]], dtype=torch.float64)
    kernel = torch.eye(2, dtype=torch.float64)[None]
    _, reports = solve_partial_batch(cost, kernel, kernel,
        config=dict(reg=.4, lam=.1, max_outer_steps=1, max_inner_steps=100))
    assert reports[0]["history"][-1]["inner"]["failure_code"] == 0
    assert reports[0]["status"] == "max_outer_steps"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_budget_failure_and_healthy_neighbor_remain_independent(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cost = torch.ones(2, 196, 196, dtype=torch.float64, device=device)
    cost[1, :156, :156] = .005
    a = cost.new_full((2, 196), 1/196)
    c = solver_config(dict(reg=.05, max_inner_steps=30))
    plan, r = entropy_subproblem_batch(a, a, cost, .8, c)
    assert r["ok"].tolist() == [True, False]
    assert r["iterations"].tolist() == [1, 30]
    assert r["stop_reason"].tolist() == [1, 0]
    assert r["kkt_error"][1] > c["inner_tolerance"]
    assert torch.isfinite(plan).all() and plan.device == cost.device
    assert all(r[k].device == cost.device for k in ("kkt_error", "relative_plan_delta_l1", "stop_reason"))
