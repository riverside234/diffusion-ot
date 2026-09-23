from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from diffusion_ot.losses import infoot
from diffusion_ot.losses.semantic_prior import validate_prior_resume
from test_stage1b_extensions import experiment, minimal_pcgrad_recipe


def slow_problem():
    features = torch.tensor([[1., 0.], [-1., 0.]])
    options = dict(
        cross_cost=torch.tensor([[0., .00027], [.00027, 0.]]),
        bandwidth=.1, mi_weight=.05, entropy_epsilon=.05, inner_iterations=1200,
        projection_iterations=200, projection_tolerance=1e-5, outer_tolerance=1e-5,
        strict_convergence=True, require_outer_convergence=True,
    )
    return features, options


def test_near_threshold_outer_failure_recovers_from_same_plan_in_fp64(monkeypatch):
    features, options = slow_problem()
    with pytest.raises(RuntimeError, match="outer updates did not converge after 1200 iterations"):
        infoot.solve_infoot(features, features, **options)
    partial = infoot.solve_infoot(features, features, **{**options, "require_outer_convergence": False})
    assert partial.sinkhorn_converged and not partial.outer_converged
    assert 1e-5 < partial.plan_delta_l1 < 2e-5

    original_gradient = infoot.infoot_plan_gradient
    first_recovery_plan = []

    def capture(plan, *args, **kwargs):
        if plan.dtype == torch.float64 and not first_recovery_plan:
            first_recovery_plan.append(plan.clone())
        return original_gradient(plan, *args, **kwargs)

    monkeypatch.setattr(infoot, "infoot_plan_gradient", capture)
    rng = torch.get_rng_state().clone()
    result = infoot.solve_infoot(features, features, recovery_iterations=4800, **options)
    # Recovery continues the exhausted plan rather than restarting the solve.
    torch.testing.assert_close(first_recovery_plan[0], partial.coupling.double(), rtol=0, atol=0)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    assert 1200 < result.iterations < 6000
    assert result.recovery_iterations == result.iterations - 1200
    assert result.effective_projection_tolerance == pytest.approx(5e-7)
    assert result.sinkhorn_converged and result.outer_converged
    assert result.plan_delta_l1 <= 1e-5
    assert result.row_residual <= 1e-5 and result.column_residual <= 1e-5
    assert result.coupling.dtype == features.dtype
    assert result.coupling.grad_fn is None
    assert torch.isfinite(result.coupling).all() and (result.coupling >= 0).all()


def test_recovery_enabled_leaves_fast_solve_bitwise_unchanged(monkeypatch):
    features = torch.zeros(4, 2)
    options = dict(cross_cost=1 - torch.eye(4), entropy_epsilon=.5,
                   inner_iterations=1200, outer_tolerance=1e-5,
                   strict_convergence=True, require_outer_convergence=True)
    original_gradient = infoot.infoot_plan_gradient
    calls = []

    def counted(plan, *args, **kwargs):
        calls.append(plan.dtype)
        return original_gradient(plan, *args, **kwargs)

    monkeypatch.setattr(infoot, "infoot_plan_gradient", counted)
    before = infoot.solve_infoot(features, features, **options)
    original_calls = calls[:]
    calls.clear()
    after = infoot.solve_infoot(features, features, recovery_iterations=4800, **options)
    assert calls == original_calls == [torch.float32] * 5
    assert before.iterations == after.iterations == 5
    assert before.objective == after.objective
    assert after.recovery_iterations == 0
    torch.testing.assert_close(before.coupling, after.coupling, rtol=0, atol=0)


def test_exhausted_recovery_still_raises_and_reports_precision_and_residuals():
    features, options = slow_problem()
    options.update(inner_iterations=10, recovery_iterations=4)
    with pytest.raises(RuntimeError) as error:
        infoot.solve_infoot(features, features, **options)
    message = str(error.value)
    assert "outer updates did not converge after 14 iterations" in message
    assert "sinkhorn_converged=True" in message
    assert "recovery_updates=4/4" in message
    assert "solve_dtype=torch.float64" in message
    assert "last_4_delta_range=" in message


def test_recovered_rectangular_plan_preserves_nonuniform_marginals_and_dtype():
    generator = torch.Generator().manual_seed(7)
    x, y = torch.randn(4, 3, generator=generator), torch.randn(6, 3, generator=generator)
    a = torch.tensor([.1, .2, .3, .4])
    b = torch.tensor([.1, .1, .2, .1, .2, .3])
    result = infoot.solve_infoot(x, y, a=a, b=b,
        cross_cost=torch.rand(4, 6, generator=generator), mi_weight=.1, entropy_epsilon=.3,
        inner_iterations=1, recovery_iterations=300, projection_iterations=1000,
        outer_tolerance=1e-5, strict_convergence=True, require_outer_convergence=True)
    assert result.recovery_iterations > 0 and result.outer_converged
    assert result.coupling.dtype == torch.float32
    torch.testing.assert_close(result.coupling.sum(1), a, rtol=0, atol=1e-5)
    torch.testing.assert_close(result.coupling.sum(0), b, rtol=0, atol=1e-5)


def test_recovery_still_rejects_infeasible_plan_even_when_outer_change_is_zero():
    features = torch.zeros(8, 2)
    cost = torch.zeros(8, 8)
    cost[4:, 4:] = 1.
    with pytest.raises(RuntimeError, match="Sinkhorn marginals did not converge"):
        infoot.solve_infoot(features, features, cross_cost=cost, mi_weight=0., entropy_epsilon=.02,
            inner_iterations=1, recovery_iterations=4, projection_iterations=1,
            outer_tolerance=1e-5, strict_convergence=True, require_outer_convergence=True)


@pytest.mark.parametrize("changes", [
    {"recovery_iterations": -1}, {"inner_iterations": 0},
    {"strict_convergence": False}, {"require_outer_convergence": False},
])
def test_recovery_requires_valid_budget_and_strict_acceptance(changes):
    features, options = slow_problem()
    options.update(recovery_iterations=100)
    options.update(changes)
    with pytest.raises(ValueError, match="recovery"):
        infoot.solve_infoot(features, features, **options)


def strict_config():
    return {"conditional_structure": {"enabled": True}, "infoot": {
        "inner_iterations": 1200, "strict_convergence": True,
        "require_outer_convergence": True, "outer_tolerance": 1e-5,
    }}


@pytest.mark.parametrize("old_budget", [None, 0, 100])
def test_resume_can_enable_or_increase_recovery_after_strict_old_solves(old_budget):
    saved = strict_config()
    if old_budget is not None:
        saved["infoot"]["recovery_iterations"] = old_budget
    current = deepcopy(saved)
    current["infoot"]["recovery_iterations"] = 4800
    originals = deepcopy((saved, current))
    validate_prior_resume(saved, current)
    assert (saved, current) == originals


@pytest.mark.parametrize("changes", [
    {"recovery_iterations": 10}, {"outer_tolerance": 2e-5},
    {"strict_convergence": False}, {"require_outer_convergence": False},
    {"mi_weight": .2},
])
def test_recovery_resume_does_not_allow_weaker_policy_or_objective_changes(changes):
    saved = strict_config()
    saved["infoot"]["recovery_iterations"] = 100
    current = deepcopy(saved)
    current["infoot"].update(recovery_iterations=4800)
    current["infoot"].update(changes)
    with pytest.raises(ValueError, match="Resume cannot change infoot"):
        validate_prior_resume(saved, current)


def test_recovery_cannot_be_added_to_marginal_nonstrict_checkpoint():
    saved = strict_config()
    saved["infoot"]["strict_convergence"] = False
    current = deepcopy(saved)
    current["infoot"]["recovery_iterations"] = 100
    with pytest.raises(ValueError, match="Resume cannot change infoot"):
        validate_prior_resume(saved, current)


def test_active_controls_and_evaluation_share_recovery_policy():
    root = Path(__file__).resolve().parents[1] / "configs"
    paths = [root / "stage1b_infoot" / name for name in (
        "structure_decoder_sit_b2.yaml", "structure_decoder_patch_sit_b2.yaml",
        "structure_decoder_dino_control_sit_b2.yaml")]
    paths.append(root / "stage1b_eval/structure_decoder_sit_b2.yaml")
    for path in paths:
        options = infoot.solver_kwargs(yaml.safe_load(path.read_text())["infoot"])
        assert options["inner_iterations"] == 1200
        assert options["recovery_iterations"] == 4800
        assert options["projection_iterations"] == 10000
        assert options["outer_tolerance"] == options["projection_tolerance"] == 1e-5
        assert options["strict_convergence"] and options["require_outer_convergence"]


def test_trainer_backward_validation_and_resume_with_recovered_plans(experiment):
    run, _, _ = experiment

    def recovery_recipe(config):
        minimal_pcgrad_recipe(config)
        # A one-update normal cap cannot pass the three-update patience check.
        config["infoot"].update(inner_iterations=1, recovery_iterations=300)

    checkpoint, logs = run("recovered", steps=1, modify=recovery_recipe)
    assert checkpoint["step"] == 1

    def increased_recipe(config):
        recovery_recipe(config)
        config["infoot"]["recovery_iterations"] = 600

    checkpoint, logs = run("recovered", resume=True, modify=increased_recipe)
    assert checkpoint["step"] == 2
    for row in logs["train"]:
        assert row["infoot_recovery_iterations"] > 0
        assert row["infoot_outer_converged"] and row["infoot_sinkhorn_converged"]
        assert row["window_mean"]["infoot_recovery_used"] == 1.
        assert row["infoot_effective_projection_tolerance"] < row["infoot_projection_tolerance"]
    assert logs["train"][-1]["infoot_recovery_iteration_budget"] == 600
    for name in ("projection_probe", "decoded_translation"):
        probe = logs["validation"][-1][name]
        if name == "decoded_translation":
            probe = probe["solver"]
        assert probe["recovery_iteration_budget"] == 600
        assert probe["recovery_iterations"] > 0
        assert probe["outer_converged"] and probe["sinkhorn_converged"]
