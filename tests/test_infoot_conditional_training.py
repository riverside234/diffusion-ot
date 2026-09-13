from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.losses.conditional_structure import conditional_structure_loss
from diffusion_ot.losses.infoot import (
    conditional_reference_log_weights, conditional_reference_weights,
    infoot_distance_scale, solve_infoot, transport_diagnostics,
)
from diffusion_ot.losses.semantic_prior import validate_prior_resume


@pytest.mark.parametrize("bandwidth", [0.1, 0.7])
def test_log_projection_matches_full_eq7_with_nonuniform_unequal_marginals(bandwidth):
    torch.manual_seed(140)
    x, y, queries = [torch.randn(*shape, dtype=torch.float64) for shape in ((5, 3), (7, 4), (3, 3))]
    a = torch.tensor([.1, .15, .2, .25, .3], dtype=torch.float64)
    b = torch.arange(1, 8, dtype=torch.float64) / 28
    solution = solve_infoot(x, y, a=a, b=b, cross_cost=torch.rand(5, 7, dtype=torch.float64),
                            entropy_epsilon=.2, inner_iterations=3, projection_iterations=1000,
                            projection_tolerance=1e-12)
    settings = dict(a=a, b=b, bandwidth=bandwidth)
    log_weights = conditional_reference_log_weights(queries, x, y, solution.coupling, **settings)
    expected = conditional_reference_weights(queries, x, y, solution.coupling, **settings)
    torch.testing.assert_close(log_weights.exp(), expected, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(torch.logsumexp(log_weights, dim=1), torch.zeros(3, dtype=x.dtype))


def test_log_projection_numerical_gradients_and_small_bandwidth_do_not_clip():
    torch.manual_seed(142)
    q, x, y = [torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in ((2, 2), (3, 2), (4, 3))]
    plan = torch.rand(3, 4, dtype=torch.float64)
    plan /= plan.sum()
    assert torch.autograd.gradcheck(
        lambda q, x, y: conditional_reference_log_weights(q, x, y, plan, bandwidth=.7),
        (q, x, y), atol=1e-4,
    )
    # A deliberately sparse plan and very narrow kernels underflow in ordinary
    # probability space; wrong, low-probability predictions still need gradients.
    sparse_plan = torch.eye(3, dtype=torch.float64) / 3
    log_prob = conditional_reference_log_weights(q, x, y[:3], sparse_plan, bandwidth=.01)
    loss = -log_prob.mean()
    loss.backward()
    assert torch.isfinite(log_prob).all()
    assert (log_prob.exp() == 0).any()
    for features in (q, x, y):
        assert torch.isfinite(features.grad).all()
        assert features.grad.norm() > 0


def test_conditional_structure_updates_both_domains_but_not_teacher_or_plan():
    torch.manual_seed(144)
    references = {d: torch.randn(6, 5, requires_grad=True) for d in ("cat", "dog")}
    queries = {d: torch.randn(3, 5, requires_grad=True) for d in references}
    structures = {d: torch.randn(6, 4, requires_grad=True) for d in references}
    query_structures = {d: torch.randn(3, 4, requires_grad=True) for d in references}
    plan = (torch.eye(6) * .9 / 6 + torch.ones(6, 6) * .1 / 36).requires_grad_()
    result = conditional_structure_loss(references, queries, structures, query_structures, plan,
                                        bandwidth=.1, cost_scale=2, teacher_temperature=.1)
    result.loss.backward()
    for d in references:
        for value in (references[d], queries[d]):
            assert torch.isfinite(value.grad).all() and value.grad.norm() > 0
        assert structures[d].grad is None and query_structures[d].grad is None
    assert plan.grad is None
    for direction in result.weights:
        torch.testing.assert_close(result.weights[direction].sum(1), torch.ones(3))
        assert result.metrics[direction]["expected_structure_cost"] > 0


def test_small_conditional_training_problem_learns_queries_not_used_in_plan():
    torch.manual_seed(146)
    reference = F.normalize(torch.randn(10, 8), dim=1)
    structures = {d: reference.clone() for d in ("cat", "dog")}
    references = {d: reference.clone() for d in structures}
    # Queries are new, slightly perturbed structure samples, initially encoded
    # near the wrong reference. The fixed plan cannot repair that permutation.
    query_structures = {d: F.normalize(reference[:5] + .02 * torch.randn(5, 8), dim=1) for d in structures}
    queries = {d: torch.nn.Parameter(reference[5:].clone()) for d in structures}
    plan = torch.eye(10) * .98 / 10 + torch.ones(10, 10) * .02 / 100
    optimizer = torch.optim.Adam(list(queries.values()), lr=.03)
    def evaluate():
        return conditional_structure_loss(references, queries, structures, query_structures, plan,
                                          bandwidth=.25, cost_scale=1, teacher_temperature=.1)
    first = evaluate()
    for _ in range(60):
        optimizer.zero_grad()
        result = evaluate()
        result.loss.backward()
        optimizer.step()
    last = evaluate()
    assert last.loss < first.loss * .3
    for direction in last.metrics:
        assert last.metrics[direction]["expected_structure_cost"] < first.metrics[direction]["expected_structure_cost"] * .5


def test_solver_reports_failure_and_only_stops_on_feasible_stable_plans():
    torch.manual_seed(148)
    x, y = torch.randn(5, 3, dtype=torch.float64), torch.randn(7, 4, dtype=torch.float64)
    settings = dict(cross_cost=torch.rand(5, 7, dtype=torch.float64), entropy_epsilon=.02,
                    projection_iterations=1, projection_tolerance=1e-12, inner_iterations=3,
                    outer_tolerance=1.0, min_inner_iterations=1, outer_patience=1)
    failed = solve_infoot(x, y, **settings)
    assert not failed.sinkhorn_converged
    assert not failed.outer_converged
    assert failed.unconverged_inner_steps == 3
    with pytest.raises(RuntimeError, match="marginals did not converge"):
        solve_infoot(x, y, strict_convergence=True, **settings)
    flat = torch.zeros(6, 4)
    stable = solve_infoot(flat, flat, inner_iterations=100, outer_tolerance=1e-6,
                          min_inner_iterations=5, outer_patience=3, strict_convergence=True)
    assert stable.sinkhorn_converged and stable.outer_converged
    assert stable.iterations == 5
    diagnostics = transport_diagnostics(stable.coupling)
    assert diagnostics["normalized_row_entropy"] == pytest.approx(1.0)
    assert diagnostics["mean_row_effective_targets"] == pytest.approx(6.0)


def test_reducing_inner_mi_weight_avoids_assignment_saturation_on_high_dimensional_features():
    torch.manual_seed(150)
    features = F.normalize(torch.randn(64, 128), dim=1)
    teacher = F.normalize(torch.randn(64, 8), dim=1)
    target_teacher = F.normalize(teacher + .4 * torch.randn_like(teacher), dim=1)
    cost = torch.cdist(teacher, target_teacher)
    cost /= cost.median()
    settings = dict(cross_cost=cost, bandwidth=.7, distance_scale_x=infoot_distance_scale(features),
                    distance_scale_y=infoot_distance_scale(features), entropy_epsilon=.05,
                    inner_iterations=30, projection_iterations=1000)
    old = solve_infoot(features, features, mi_weight=1.0, **settings)
    new = solve_infoot(features, features, mi_weight=.1, **settings)
    assert transport_diagnostics(old.coupling)["mean_row_effective_targets"] < 1.1
    assert transport_diagnostics(new.coupling)["mean_row_effective_targets"] > 2
    assert new.sinkhorn_converged


def test_projection_objective_cannot_be_silently_changed_on_resume():
    saved = {"infoot": {"variant": "fused", "mi_weight": .1},
             "conditional_structure": {"enabled": True, "bandwidth_multiplier": .1}}
    validate_prior_resume(saved, saved)
    for key, value in (("conditional_structure", {}), ("infoot", {"variant": "fused", "mi_weight": 1.0})):
        changed = deepcopy(saved)
        changed[key] = value
        with pytest.raises(ValueError, match=f"Resume cannot change {key}"):
            validate_prior_resume(saved, changed)
