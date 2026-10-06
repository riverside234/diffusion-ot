import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "infoot"))
    from infoot_helper import infoot
    from infoot_helper.transport import optimization, plan_io, transport_utils
    return SimpleNamespace(infoot=infoot, opt=optimization,
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


def test_baseline_uses_uniform_start_and_full_updates_then_returns_best(modules, solver, monkeypatch):
    uniform = solver.Xs.new_full((3, 4), 1 / 12)
    delta = torch.zeros_like(uniform)
    delta[:2, :2] = uniform.new_tensor([[0.04, -0.04], [-0.04, 0.04]])
    higher, lower = uniform + delta, uniform + 2 * delta
    proposals = iter((higher, lower, higher))
    inputs = []

    def objective(P, **kwargs):
        value = 3.0 if torch.equal(P, higher) else 1.0 if torch.equal(P, lower) else 2.0
        return P.new_tensor(value)

    def gradient(P, *args):
        inputs.append(P.clone())
        return torch.zeros_like(P)

    monkeypatch.setattr(modules.opt, "fitting_loss", objective)
    monkeypatch.setattr(modules.opt, "migrad", gradient)
    monkeypatch.setattr(modules.opt, "sinkhorn_plan", lambda *args: (next(proposals), None))
    P = solver.solve(numIter=3, verbose=False)
    torch.testing.assert_close(inputs, [uniform, higher, lower])
    torch.testing.assert_close(P, lower)
    assert modules.utils.is_feasible(P) and not P.requires_grad
    report = solver.diagnostics_
    assert [row["loss"] for row in report["history"]] == [2.0, 3.0, 1.0, 3.0]
    assert report["best_iteration"] == 2 and report["initialization"] == "uniform"
    assert report["status"] == "iteration_limit" and not solver.converged_


def test_convergence_requires_repeated_small_changes(modules, solver, monkeypatch):
    uniform = solver.Xs.new_full((3, 4), 1 / 12)
    calls = []

    def proposal(*args):
        calls.append(args)
        return uniform, None

    monkeypatch.setattr(modules.opt, "sinkhorn_plan", proposal)
    P = solver.solve(numIter=10, verbose=False)
    assert len(calls) == 3
    assert solver.converged_ and solver.diagnostics_["status"] == "converged"
    torch.testing.assert_close(P, uniform)


def test_baseline_reproduces_plan_without_changing_global_rng(modules, solver):
    rng = torch.random.get_rng_state().clone()
    first = solver.solve(numIter=4, verbose=False).clone()
    second = solver.solve(numIter=4, verbose=False)
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


def test_save_load_preserves_optimized_mapping_and_diagnostics(modules, solver):
    P = solver.solve(numIter=3, verbose=False)
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
