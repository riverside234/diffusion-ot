import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "infoot"))
    from infoot_helper import infoot
    from infoot_helper.transport import multistart, optimization, plan_io, transport_utils
    return SimpleNamespace(infoot=infoot, multi=multistart, opt=optimization,
                           io=plan_io, utils=transport_utils)


@pytest.fixture
def solver(modules):
    source = torch.tensor([[0., 1.], [1., 0.], [2., 1.]], dtype=torch.float64)
    target = torch.tensor([[0.1, 0.2], [0.6, -0.2], [1.3, 0.7], [2., -0.1]],
                          dtype=torch.float64)
    return modules.infoot.FusedInfoOT(source, target, h=0.5, lam=0.1, reg=0.3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("clipped", [False, True])
def test_manual_mi_gradient_matches_scalar_loss(modules, dtype, clipped):
    P = torch.tensor([[0., 0.2], [0.3, 0.5]], dtype=dtype)
    if clipped:
        P = P * 1e-12
    P.requires_grad_()
    Kx = torch.eye(2, dtype=dtype)
    Ky = torch.tensor([[1., 0.1], [0.1, 1.]], dtype=dtype)
    value = modules.infoot.fitting_loss(P, Kx, Ky, reg=0)
    expected, = torch.autograd.grad(value, P)
    actual = modules.infoot.migrad(P, Kx, Ky)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


def test_full_objective_gradient_counts_entropy_once(modules):
    P = torch.tensor([[1e-12, 0.2], [0.3, 0.5]], dtype=torch.float64,
                     requires_grad=True)
    K = torch.eye(2, dtype=P.dtype)
    C = torch.tensor([[0., 1.], [2., 0.]], dtype=P.dtype)
    loss = modules.infoot.fitting_loss(P, K, K, 0.02, C, mi_weight=0.1)
    actual, = torch.autograd.grad(loss, P)
    expected = C + 0.1 * modules.infoot.migrad(P, K, K) + 0.02 * (P.log() + 1)
    torch.testing.assert_close(actual, expected)
    entropy = -(P * P.log()).sum()
    mi_only = modules.infoot.fitting_loss(P, K, K, 0, C, mi_weight=0.1)
    torch.testing.assert_close(loss, mi_only - 0.02 * entropy)


def test_initial_plans_are_feasible_distinct_and_reproducible(modules, solver):
    rng = torch.random.get_rng_state().clone()
    plans = []
    for index in range(6):
        name, P = modules.multi.initial_plan(solver, index, 13, marginal_tol=1e-7)
        repeat_name, repeated = modules.multi.initial_plan(solver, index, 13, marginal_tol=1e-7)
        assert name == repeat_name and torch.equal(P, repeated)
        assert modules.utils.is_feasible(P, 1e-7)
        plans.append(P)
    assert not torch.equal(plans[0], plans[1])
    assert torch.equal(rng, torch.random.get_rng_state())


def test_single_run_decreases_loss_and_does_not_mutate_initial_plan(modules, solver):
    _, P0 = modules.multi.initial_plan(solver, 1, 1)
    P0.requires_grad_()
    original = P0.detach().clone()
    P = solver.solve(numIter=12, verbose=False, P0=P0)
    values = [row["loss"] for row in solver.diagnostics_["history"]]
    assert all(right <= left for left, right in zip(values, values[1:]))
    assert modules.utils.is_feasible(P)
    assert torch.equal(P0, original) and not P.requires_grad
    assert solver.diagnostics_["loss"] <= values[0]


def test_backtracking_accepts_fractional_step_when_full_step_increases_loss(modules):
    P = torch.full((2, 2), 0.25, dtype=torch.float64)
    Q = torch.eye(2, dtype=P.dtype) / 2
    target = 0.75 * P + 0.25 * Q
    objective = lambda plan: (plan - target).square().sum()
    initial_loss = objective(P).item()
    candidate, value, alpha = modules.utils.backtrack(P, Q, objective, initial_loss, 1e-7)
    assert objective(Q).item() > initial_loss
    assert alpha == 0.25 and value < initial_loss
    assert modules.utils.is_feasible(candidate, 1e-7)
    torch.testing.assert_close(candidate, target)


def test_multistart_rejects_objective_override(modules, solver):
    with pytest.raises(ValueError, match="Set reg on the solver"):
        modules.multi.solve_multistart(solver, reg=1.0, verbose=False)


@pytest.mark.parametrize("kind", ["shape", "negative", "marginal", "nan"])
def test_invalid_initial_plan_is_rejected(solver, kind):
    P = solver.Xs.new_full((3, 4), 1 / 12)
    if kind == "shape":
        P = P[:1]
    elif kind == "negative":
        P[0, 0] = -0.1
    elif kind == "marginal":
        P[0] *= 2
    else:
        P[0, 0] = float("nan")
    with pytest.raises(ValueError, match="P0"):
        solver.solve(numIter=2, verbose=False, P0=P)


def test_multiple_starts_escape_uniform_solution(modules):
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]],
                     dtype=torch.float64)
    solver = modules.infoot.InfoOT(x, x, h=0.4, reg=0.1)
    P = modules.multi.solve_multistart(solver, numIter=20, restarts=2,
                                      seed=7, verbose=False)
    report = solver.diagnostics_
    baseline = report["runs"][0]["loss"]
    assert report["loss"] < baseline - 0.1
    assert modules.utils.is_feasible(P)
    torch.testing.assert_close(
        modules.infoot.fitting_loss(P, solver.Ks, solver.Kt, solver.reg),
        P.new_tensor(report["loss"]),
    )


def test_continuation_ranks_final_target_objective_only(modules, solver):
    kernels = (solver.Ks.clone(), solver.Kt.clone())
    P = modules.multi.solve_multistart(solver, numIter=5, restarts=2,
                                      seed=9, continuation=True, verbose=False)
    report = solver.diagnostics_
    candidate = report["runs"][-1]
    assert [stage["reg"] for stage in candidate["stages"]] == [1.5, 0.6, 0.3]
    assert solver.reg == 0.3 and solver.h == 0.5
    torch.testing.assert_close((solver.Ks, solver.Kt), kernels)
    assert report["loss"] == min(run["loss"] for run in report["runs"])
    torch.testing.assert_close(
        modules.infoot.fitting_loss(P, solver.Ks, solver.Kt, 0.3, solver.C, 0.1),
        P.new_tensor(report["loss"]),
    )


def test_restart_seed_reproduces_plan_without_changing_global_rng(modules, solver):
    rng = torch.random.get_rng_state().clone()
    first = modules.multi.solve_multistart(solver, numIter=4, restarts=3,
                                          seed=5, verbose=False).clone()
    second = modules.multi.solve_multistart(solver, numIter=4, restarts=3,
                                           seed=5, verbose=False)
    assert torch.equal(first, second)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_sinkhorn_rejects_bad_marginals_after_retry(modules, monkeypatch):
    calls = []

    def invalid(p, q, cost, **kwargs):
        calls.append(kwargs["numItermax"])
        return torch.zeros_like(cost), {"log_u": torch.zeros_like(p),
                                       "log_v": torch.zeros_like(q)}

    monkeypatch.setattr(modules.utils.ot.bregman, "sinkhorn", invalid)
    with pytest.raises(FloatingPointError, match="marginal"):
        modules.utils.sinkhorn_plan(torch.zeros(3, 4), 0.02, sinkhorn_iter=10)
    assert calls == [10, 20]


def test_inner_failure_preserves_feasible_result_and_reports_failure(modules, solver, monkeypatch):
    def fail(*args, **kwargs):
        raise FloatingPointError("probe failure")

    monkeypatch.setattr(modules.opt, "sinkhorn_plan", fail)
    P = solver.solve(numIter=3, verbose=False)
    assert modules.utils.is_feasible(P)
    assert solver.diagnostics_["status"] == "inner_failure"
    assert solver.diagnostics_["error"] == "probe failure"
    assert not solver.converged_


def test_failed_line_search_is_not_convergence(modules, solver, monkeypatch):
    _, Q = modules.multi.initial_plan(solver, 1, 10)
    calls = []

    def proposal(*args):
        calls.append(args[2])
        return Q, None

    monkeypatch.setattr(modules.opt, "sinkhorn_plan", proposal)
    monkeypatch.setattr(modules.opt, "backtrack",
                        lambda P, Q, fn, value, tol: (P, value, 0.0))
    P = solver.solve(numIter=4, verbose=False)
    assert modules.utils.is_feasible(P)
    assert calls == [1e-4, 1e-5]
    assert solver.diagnostics_["status"] == "stalled" and not solver.converged_


def test_save_load_preserves_optimized_mapping_and_diagnostics(modules, solver):
    P = modules.multi.solve_multistart(solver, numIter=3, restarts=2, verbose=False)
    query = solver.Xs + 0.05
    before = modules.infoot.projection(solver.conditional_score(query), solver.Xt)
    buffer = io.BytesIO()
    modules.infoot.save_plan(buffer, P, solver.h, solver.reg, solver.lam,
                            optimization=solver.diagnostics_)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    assert torch.equal(saved["P"], P)
    assert saved["optimization"] == solver.diagnostics_
    assert saved["solver"] == "FusedInfoOT" and saved["feature_space"] == "raw"
    solver.P = saved["P"]
    after = modules.infoot.projection(solver.conditional_score(query), solver.Xt)
    assert torch.equal(before, after)


def test_metadata_identifies_bank_contents_and_current_checkpoint(modules, tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"model one")
    bank_path = tmp_path / "bank.pt"
    bank = {"v_bank": torch.arange(6).reshape(3, 2), "checkpoint_path": str(checkpoint)}
    torch.save(bank, bank_path)
    first = modules.io.bank_metadata(bank_path, bank)
    checkpoint.write_bytes(b"model two")
    second = modules.io.bank_metadata(bank_path, bank)
    assert first["sha256"] == second["sha256"]
    assert first["checkpoint_at_fit"]["sha256"] != second["checkpoint_at_fit"]["sha256"]
    bank["v_bank"] = bank["v_bank"].flip(0)
    torch.save(bank, bank_path)
    assert modules.io.bank_metadata(bank_path, bank)["sha256"] != first["sha256"]
