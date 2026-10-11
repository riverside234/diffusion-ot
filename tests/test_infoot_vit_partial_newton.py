"""Sharp, mean-scaled partial OT: exact optima and independent batch failures."""
import importlib
import json
import math

import pytest
import torch

from test_infoot_vit_mapping import banks, cpu_threads
from test_infoot_vit_partial_batch import batched_config, problem
from infoot_vit.infoot_helper.partial import solver_config, entropy_subproblem
from infoot_vit.infoot_helper.partial_batch import entropy_subproblem_batch, solve_partial_batch
from infoot_vit.infoot_helper.partial_newton import dual_newton_step
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, validate_config


def sharp_blocks():
    cost = torch.ones(2, 196, 196, dtype=torch.float64)
    for i, cheap in enumerate((155, 156)):
        cost[i, :cheap, :cheap] = .005
    return cost / cost.mean((-2, -1), keepdim=True)


def analytic_block(cost, cheap, mass, reg):
    n = len(cost)
    rest, k = n-cheap, math.exp(-float(cost[-1, -1]-cost[0, 0])/reg)
    deficit = mass-cheap/n
    aa, bb, cc = n*deficit*cheap, rest*k*(n*deficit-cheap), -rest**2*k
    eu = (-bb+math.sqrt(bb*bb-4*aa*cc))/(2*aa)
    ew = (1/n)/(cheap*eu**2+rest*k*eu)
    expected = torch.full_like(cost, k*ew)
    expected[:cheap, :] *= eu
    expected[:, :cheap] *= eu
    expected[:cheap, :cheap] /= k
    return expected


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mean_scaled_196_patch_optimum_and_mixed_batch(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    hard = sharp_blocks().to(device)
    # Include an already-converged neighbor and a quarantined invalid member.
    cost = torch.cat((torch.ones_like(hard[:1]), hard, torch.full_like(hard[:1], float("nan"))))
    a = cost.new_full((4, 196), 1/196)
    c = solver_config(dict(reg=.05, inner_acceleration="newton", max_inner_steps=1000))
    plan, info = entropy_subproblem_batch(a, a, cost, .8, c)
    assert info["ok"].tolist() == [True, True, True, False]
    assert info["iterations"][0].item() == 1 and info["newton_steps"][0].item() == 0
    assert (info["newton_steps"][1:3] > 0).all()
    assert (info["iterations"][1:3] < 500).all()
    assert plan.device == cost.device and plan.dtype == torch.float64
    for i, cheap in enumerate((155, 156), 1):
        torch.testing.assert_close(plan[i], analytic_block(cost[i], cheap, .8, .05), atol=2e-12, rtol=1e-8)
        for key in ("kkt_error", "relative_duality_gap", "relative_plan_delta_l1"):
            assert info[key][i] <= c["inner_tolerance"]
        solo, _ = entropy_subproblem_batch(a[i:i+1], a[i:i+1], cost[i:i+1], .8, c)
        torch.testing.assert_close(solo[0], plan[i], atol=2e-13, rtol=1e-9)
    torch.testing.assert_close(plan[0], torch.full_like(plan[0], .8/196**2), atol=1e-15, rtol=1e-12)


def test_sharp_unaccelerated_plan_really_fails_optimality():
    # Reproduce the v2 pattern at the full 20k budget, with production cost scaling.
    cost = sharp_blocks()[:1]
    a = cost.new_full((1, 196), 1/196)
    _, r = entropy_subproblem_batch(a, a, cost, .8, solver_config(dict(reg=.05, max_inner_steps=20000)))
    assert not r["ok"].item() and r["kkt_error"].item() > 1e-10
    assert r["relative_duality_gap"].item() > 1e-10


@pytest.mark.parametrize("mass", [.3, .8, 1.])
def test_accelerated_path_agrees_with_pot_and_nonuniform_capacities(mass):
    cost, _, _ = problem()
    a = cost.new_tensor([.1, .2, .3, .4]).expand(3, -1)
    b = cost.new_tensor([.1, .2, .15, .25, .3]).expand(3, -1)
    c = solver_config(dict(reg=.4, inner_acceleration="newton", inner_tolerance=1e-12))
    actual, info = entropy_subproblem_batch(a, b, cost, mass, c)
    assert info["ok"].all()
    for i in range(3):
        expected, _ = entropy_subproblem(a[i], b[i], cost[i], mass, c)
        torch.testing.assert_close(actual[i], expected, atol=1e-12, rtol=1e-9)


def test_newton_step_is_dual_descent_and_failed_linear_solve_is_rejected(monkeypatch):
    base = -sharp_blocks()/ .05
    a = base.new_full((2, 196), 1/196)
    u = v = torch.zeros_like(a)
    w = math.log(.8)-base.logsumexp((-2,-1))
    nu, nv, nw, accepted = dual_newton_step(base, a, a, .8, u, v, w)
    def value(uu, vv, ww):
        return (base+uu[:,:,None]+vv[:,None,:]+ww[:,None,None]).exp().sum((-2,-1))-(a*uu).sum(-1)-(a*vv).sum(-1)-.8*ww
    assert accepted.all() and (nu <= 0).all() and (nv <= 0).all()
    assert (value(nu,nv,nw) < value(u,v,w)).all()
    monkeypatch.setattr(torch.linalg, "solve_ex", lambda h, r, **kw: (torch.full_like(r, float("nan")), torch.ones(len(h), dtype=torch.int32)))
    bad = dual_newton_step(base, a, a, .8, u, v, w)
    assert not bad[-1].any()
    for actual, old in zip(bad, (u,v,w)):
        assert torch.equal(actual, old)


def asymmetric_problem(count=1):
    rng = torch.Generator().manual_seed(42)
    costs = 1 + .2*torch.rand(count, 12, 15, generator=rng, dtype=torch.float64)
    costs[:, :9, :11] *= .01
    return costs / costs.mean((-2, -1), keepdim=True)


def test_actual_newton_updates_match_pot_on_asymmetric_problem():
    cost = asymmetric_problem()
    a, b = cost.new_full((1, 12), 1/12), cost.new_full((1, 15), 1/15)
    c = solver_config(dict(reg=.075, inner_acceleration="newton", max_inner_steps=2000))
    plan, info = entropy_subproblem_batch(a, b, cost, .8, c)
    assert info["ok"].all() and info["newton_steps"].item() > 0
    reference, _ = entropy_subproblem(a[0], b[0], cost[0], .8, dict(c, max_inner_steps=40000))
    torch.testing.assert_close(plan[0], reference, atol=5e-12, rtol=1e-8)
    def objective(p):
        return (p*cost[0]).sum() + c["reg"]*(torch.special.xlogy(p,p)-p).sum()
    assert float((objective(plan[0])-objective(reference)).abs()) < 1e-11


def test_newton_chunks_include_final_member_and_are_invariant():
    cost = asymmetric_problem(65)
    cost[5].fill_(1.)
    a, b = cost.new_full((65,12), 1/12), cost.new_full((65,15), 1/15)
    c = solver_config(dict(reg=.075, inner_acceleration="newton", max_inner_steps=1000))
    full, info = entropy_subproblem_batch(a, b, cost, .8, c)
    assert info["ok"].all() and info["newton_steps"][-1] > 0
    pieces = [entropy_subproblem_batch(a[i:i+7], b[i:i+7], cost[i:i+7], .8, c)[0] for i in range(0,65,7)]
    torch.testing.assert_close(full, torch.cat(pieces), atol=2e-13, rtol=1e-9)


def test_acceleration_preserves_outer_objective_and_budget_failure():
    cost, kx, ky = problem()
    cfg = dict(reg=.4, inner_acceleration="newton", max_outer_steps=100)
    _, reports = solve_partial_batch(cost, kx, ky, config=cfg)
    assert all(r["status"] == "converged" for r in reports)
    for r in reports:
        values = [step["objective"] for step in r["history"]]
        assert all(b <= a+1e-12 for a,b in zip(values,values[1:]))
    _, short = solve_partial_batch(cost, kx, ky, config=dict(cfg, max_inner_steps=1))
    assert any(r["status"] == "inner_failed" for r in short)
    json.dumps(reports, allow_nan=False)


def test_fail_fast_preserves_diagnostics_and_allows_exact_resume(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.pair_batch_fit")
    real = module.solve_partial_batch
    calls = []
    def fail_batch(cost, *args, **kw):
        calls.append(len(cost))
        return real(torch.full_like(cost, float("nan")), *args, **kw)
    monkeypatch.setattr(module, "solve_partial_batch", fail_batch)
    config = dict(batched_config(2), pair_failure_abort_batches=1)
    config["partial"]["solver"]["inner_acceleration"] = "newton"
    with pytest.raises(RuntimeError, match="2 pairs unattempted"):
        fit_mapping(config, root=tmp_path)
    assert calls == [2]
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    report = json.loads((directory / "pair_batch_report.json").read_text())
    assert report["status"] == "aborted_all_failed_batches"
    assert report["failed_pairs"] == 2 and report["unattempted_pairs"] == 2
    assert report["unfinished_attempted_pairs"] == 0
    assert len(list((directory / "plans/pairs").rglob("latest.pt"))) == 2
    assert next((directory / "logs").glob("*/first_failed_pair.pt")).exists()
    monkeypatch.setattr(module, "solve_partial_batch", real)
    fit_mapping(config, root=tmp_path, resume=directory)
    assert len((directory / "pairs.jsonl").read_text().splitlines()) == 4
    assert not list((directory / "plans/pairs").rglob("latest.pt"))
    assert json.loads((directory/"manifest.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_abort_setting_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="pair_failure_abort_batches"):
        validate_config(dict(batched_config(), pair_failure_abort_batches=value))


def test_invalid_acceleration_rejected():
    with pytest.raises(ValueError, match="inner_acceleration"):
        solver_config(dict(inner_acceleration="balanced_fallback"))


def test_replay_keeps_objective_and_records_success_or_failure(tmp_path):
    from infoot_vit.replay_partial_pair import main
    from infoot_vit.infoot_helper.feature_bank import file_hash
    cost, kx, ky = problem(1)
    example = tmp_path / "first_failed_pair.pt"
    cfg = solver_config(dict(reg=.4, lam=.1))
    torch.save(dict(cost=cost[0], kx=kx[0], ky=ky[0], a=cost.new_full((4,), .25),
        b=cost.new_full((5,), .2), config=cfg, keep_mass=.8), example)
    before = file_hash(example)
    output = tmp_path / "replay.json"
    assert main([str(example), "--output", str(output), "--device", "cpu", "--copies", "2", "--threads", "1"]) == 0
    r = json.loads(output.read_text())
    assert r["status"] == "converged" and r["replica_max_abs_difference"] == 0
    assert r["config"] == dict(cfg, inner_acceleration="newton")
    assert r["peak_gpu_allocated_bytes"] is None
    assert file_hash(example) == before and output.with_suffix(".pt").exists()
    assert main([str(example), "--output", str(output), "--device", "cpu", "--max-inner-steps", "1", "--threads", "1"]) == 2
    assert json.loads(output.read_text())["status"] == "failed"


def test_serial_cli_clears_batch_only_recipe_options(monkeypatch, tmp_path):
    from infoot_vit import infoot_fit
    seen = []
    monkeypatch.setattr(infoot_fit, "fit_mapping", lambda config, **kw: seen.append(validate_config(config)) or tmp_path)
    assert infoot_fit.main(["--config", str(infoot_fit.ROOT/"infoot_vit/configs/grouped_partial.yaml"),
                           "--device", "cpu", "--serial-pairs"]) == 0
    assert seen[0]["pair_batch_size"] is None and seen[0]["pair_failure_abort_batches"] is None
    assert seen[0]["partial"]["solver"]["inner_acceleration"] == "none"
