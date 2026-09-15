from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from diffusion_ot.losses.infoot import solve_infoot, solver_kwargs
from diffusion_ot.losses.semantic_prior import validate_prior_resume


def test_slow_feasible_plan_converges_with_larger_cap_at_same_tolerance():
    # Deterministic slow fixed-point problem. Marginals converge immediately,
    # but a small fused cost and equal MI/entropy weights need >300 outer steps.
    features = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    cost = torch.tensor([[0., .0005], [.0005, 0.]], dtype=torch.float64)
    options = dict(cross_cost=cost, bandwidth=.1, mi_weight=.05, entropy_epsilon=.05,
                   projection_iterations=200, projection_tolerance=1e-5,
                   outer_tolerance=1e-5, strict_convergence=True)
    unfinished = solve_infoot(features, features, inner_iterations=300, **options)
    assert unfinished.sinkhorn_converged and not unfinished.outer_converged
    with pytest.raises(RuntimeError, match="outer updates did not converge after 300 iterations"):
        solve_infoot(features, features, inner_iterations=300, require_outer_convergence=True, **options)
    converged = solve_infoot(features, features, inner_iterations=1200, require_outer_convergence=True, **options)
    assert converged.sinkhorn_converged and converged.outer_converged
    assert 300 < converged.iterations < 1200
    assert converged.plan_delta_l1 <= options["outer_tolerance"]
    torch.testing.assert_close(converged.coupling.sum(0), torch.full((2,), .5, dtype=torch.float64))
    torch.testing.assert_close(converged.coupling.sum(1), torch.full((2,), .5, dtype=torch.float64))


def test_larger_cap_keeps_fast_solve_result_and_work_identical():
    features = torch.zeros(4, 2, dtype=torch.float64)
    options = dict(cross_cost=1 - torch.eye(4, dtype=torch.float64), entropy_epsilon=.5,
                   outer_tolerance=1e-5, strict_convergence=True, require_outer_convergence=True)
    before = solve_infoot(features, features, inner_iterations=300, **options)
    after = solve_infoot(features, features, inner_iterations=1200, **options)
    assert before.iterations == after.iterations == 5
    torch.testing.assert_close(before.coupling, after.coupling, rtol=0, atol=0)


def _strict_config():
    return {"conditional_structure": {"enabled": True},
            "infoot": {"variant": "fused", "inner_iterations": 300, "mi_weight": .1,
                       "outer_tolerance": 1e-5, "projection_tolerance": 1e-5,
                       "strict_convergence": True, "require_outer_convergence": True}}


@pytest.mark.parametrize("explicit_old_budget", [True, False])
def test_resume_allows_only_budget_increase_for_required_converged_plans(explicit_old_budget):
    saved = _strict_config()
    if not explicit_old_budget:
        saved["infoot"].pop("inner_iterations")
    current = deepcopy(saved)
    current["infoot"]["inner_iterations"] = 1200
    original_saved, original_current = deepcopy(saved), deepcopy(current)
    validate_prior_resume(saved, current)
    assert saved == original_saved and current == original_current


@pytest.mark.parametrize("field,value", [
    ("inner_iterations", 200), ("mi_weight", .2), ("outer_tolerance", 2e-5),
    ("projection_tolerance", 1e-4), ("require_outer_convergence", False),
    ("outer_patience", 1), ("strict_convergence", False), ("entropy_epsilon", .1),
])
def test_resume_still_rejects_objective_tolerance_or_policy_changes(field, value):
    saved = _strict_config()
    current = deepcopy(saved)
    current["infoot"]["inner_iterations"] = 1200
    current["infoot"][field] = value
    with pytest.raises(ValueError, match="Resume cannot change infoot"):
        validate_prior_resume(saved, current)


def test_truncated_outer_plans_cannot_change_iteration_budget_on_resume():
    saved = _strict_config()
    saved["infoot"]["require_outer_convergence"] = False
    current = deepcopy(saved)
    current["infoot"]["inner_iterations"] = 1200
    with pytest.raises(ValueError, match="Resume cannot change infoot"):
        validate_prior_resume(saved, current)


def test_active_and_rms_only_train_eval_use_same_larger_cap_and_strict_tolerance():
    root = Path(__file__).resolve().parents[1]
    for stage in ("stage1b_infoot", "stage1b_eval"):
        for variant in ("structure_decoder_sit_b2.yaml", "structure_decoder_rmsgrad_sit_b2.yaml",
                        "structure_decoder_vicreg_cosine_sit_b2.yaml"):
            config = yaml.safe_load((root / "configs" / stage / variant).read_text())
            options = solver_kwargs(config["infoot"])
            assert options["inner_iterations"] == 1200
            assert options["outer_tolerance"] == options["projection_tolerance"] == 1e-5
            assert options["outer_patience"] == 3
            assert options["strict_convergence"] and options["require_outer_convergence"]
