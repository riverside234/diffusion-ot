"""Execution optimizations preserve exact partial OT and recovery contracts."""
import importlib
import json

import pytest
import torch

from test_infoot_vit_mapping import banks, cpu_threads
from test_infoot_vit_partial_batch import problem, batched_config
from test_infoot_vit_partial_newton import asymmetric_problem, sharp_blocks
from infoot_vit.infoot_helper.partial import solver_config, entropy_subproblem, solve_partial
from infoot_vit.infoot_helper.partial_batch import entropy_subproblem_batch, solve_partial_batch
from infoot_vit.infoot_helper.partial_newton import dual_newton_step
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, resource_estimate
from infoot_vit.infoot_helper.pair_timing import PairTimer


OPTIONS = dict(inner_acceleration="newton", inner_compact=True, inner_warm_start=True,
               newton_batch_size=2, newton_check_every=5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compaction_reduces_work_and_preserves_retired_states(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cost = asymmetric_problem(5).to(device)
    cost[0].fill_(1.)
    cost[2].fill_(float("nan"))
    a, b = cost.new_full((5,12), 1/12), cost.new_full((5,15), 1/15)
    cfg = solver_config(dict(reg=.075, max_inner_steps=1000, **OPTIONS))
    compact, info = entropy_subproblem_batch(a, b, cost, .8, cfg)
    fixed, ref = entropy_subproblem_batch(a, b, cost, .8, dict(cfg,
        inner_compact=False, newton_batch_size=64, newton_check_every=0))
    assert info["ok"].tolist() == [True, True, False, True, True]
    assert compact.device == cost.device and compact.dtype == torch.float64
    torch.testing.assert_close(compact, fixed, atol=2e-13, rtol=1e-9)
    for key in ("iterations", "failure", "stop_reason"):
        assert torch.equal(info[key], ref[key])
    assert info["iterations"][0] == info["work_iterations"][0] == 1
    assert info["work_iterations"].sum() < ref["work_iterations"].sum()
    assert info["iterations"][2] == 0
    # Last incomplete Newton chunk and original ordering survive compaction.
    for i in (1,3,4):
        solo, _ = entropy_subproblem_batch(a[i:i+1], b[i:i+1], cost[i:i+1], .8, cfg)
        torch.testing.assert_close(compact[i], solo[0], atol=2e-13, rtol=1e-9)
        for key in ("kkt_error", "relative_duality_gap", "relative_plan_delta_l1"):
            assert info[key][i] <= cfg["inner_tolerance"]


@pytest.mark.parametrize("mass", [.3, .8, 1.])
def test_warm_start_rebuilds_new_cost_and_matches_pot(mass):
    costs, _, _ = problem()
    a = costs.new_tensor([.1,.2,.3,.4]).expand(3,-1)
    b = costs.new_tensor([.1,.2,.15,.25,.3]).expand(3,-1)
    cfg = solver_config(dict(reg=.4, inner_tolerance=1e-12, **OPTIONS))
    _, previous = entropy_subproblem_batch(a, b, costs, mass, cfg)
    before = [v.clone() for v in previous["dual_state"]]
    changed = costs + .003 * torch.arange(20).reshape(1,4,5) - 20
    result, info = entropy_subproblem_batch(a, b, changed, mass, cfg, initial_duals=previous["dual_state"])
    assert info["ok"].all()
    for i in range(3):
        ref, _ = entropy_subproblem(a[i], b[i], changed[i], mass, cfg)
        torch.testing.assert_close(result[i], ref, atol=3e-12, rtol=1e-9)
    for actual, old in zip(previous["dual_state"], before):
        assert torch.equal(actual, old)
    # Nonfinite warm state is quarantined by falling back to cold initialization.
    bad = [v.clone() for v in before]
    bad[0][1].fill_(float("nan"))
    fallback, r = entropy_subproblem_batch(a, b, changed, mass, cfg, initial_duals=bad)
    assert r["ok"].all()
    torch.testing.assert_close(result, fallback, atol=3e-12, rtol=1e-9)


def test_certified_warm_start_reduces_inner_iterations():
    cost = asymmetric_problem()
    a, b = cost.new_full((1,12),1/12), cost.new_full((1,15),1/15)
    cfg = solver_config(dict(reg=.075, max_inner_steps=1000, **OPTIONS))
    cold, r = entropy_subproblem_batch(a, b, cost, .8, cfg)
    warm, w = entropy_subproblem_batch(a, b, cost, .8, cfg, initial_duals=r["dual_state"])
    assert r["ok"].all() and w["ok"].all()
    assert w["iterations"].sum() < r["iterations"].sum()
    torch.testing.assert_close(cold, warm, atol=3e-12, rtol=1e-9)


@pytest.mark.parametrize("mass", [.8, 1.])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_infoot_matches_serial_and_reports_timing(mass, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cost, kx, ky = problem(device=device)
    # Easy neighbor retires before other pairs; warm caches must retain IDs.
    cost[0].fill_(1.); kx[0].fill_(1.); ky[0].fill_(1.)
    cfg = solver_config(dict(reg=.4, max_outer_steps=100, **OPTIONS))
    timer = PairTimer(cost.device)
    actual, reports = solve_partial_batch(cost, kx, ky, config=cfg, keep_mass=mass, timer=timer)
    for i, report in enumerate(reports):
        expected, _ = solve_partial(cost[i], kx[i], ky[i], config=cfg, keep_mass=mass)
        assert report["status"] == "converged"
        torch.testing.assert_close(actual[i], expected, atol=1e-9, rtol=1e-8)
        values = [r["objective"] for r in report["history"]]
        assert all(b <= a + cfg["objective_tolerance"] for a,b in zip(values, values[1:]))
    timing = timer.report()
    assert set(timing["host_wall_seconds"]) == {"mi_and_gradient", "inner_solver", "outer_update_and_diagnostics"}
    assert all(t > 0 for t in timing["host_wall_seconds"].values())
    if device == "cuda":
        assert set(timing["cuda_stream_seconds"]) == set(timing["host_wall_seconds"])
        assert all(t >= 0 for t in timing["cuda_stream_seconds"].values())
    else:
        assert timing["cuda_stream_seconds"] == {}
    json.dumps(reports, allow_nan=False)


@pytest.mark.parametrize("easy", [True, False])
def test_newton_early_exit_matches_fixed_work(monkeypatch, easy):
    base = torch.zeros(2,4,5,dtype=torch.float64) if easy else -sharp_blocks()/.05
    a = base.new_full((len(base),base.shape[1]),1/base.shape[1])
    b = base.new_full((len(base),base.shape[2]),1/base.shape[2])
    u,v = torch.zeros_like(a), torch.zeros_like(b)
    w = base.new_tensor(.8).log()-base.logsumexp((-2,-1))
    fixed = dual_newton_step(base,a,b,.8,u,v,w)
    counts = []
    real = torch.Tensor.exp
    def count_exp(tensor):
        counts.append(1)
        return real(tensor)
    monkeypatch.setattr(torch.Tensor, "exp", count_exp)
    early = dual_newton_step(base,a,b,.8,u,v,w,check_every=5)
    assert early[-1].all() and len(counts) <= 31
    if easy:
        assert len(counts) == 6  # Initial plan + five trials, skipping 25 trials.
    for actual, expected in zip(early, fixed):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_optimized_fit_save_resume_and_performance_logs(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.pair_batch_fit")
    real = module.solve_partial_batch
    config = batched_config(3)
    config["partial"]["solver"].update(OPTIONS)
    first = True
    def fail_one(cost, *args, **kwargs):
        nonlocal first
        if first:
            cost = cost.clone(); cost[1].fill_(float("nan")); first = False
        return real(cost, *args, **kwargs)
    monkeypatch.setattr(module, "solve_partial_batch", fail_one)
    with pytest.raises(RuntimeError, match="1 partial pairs failed"):
        fit_mapping(config, root=tmp_path)
    directory = next((tmp_path/"outputs/infoot_vit").iterdir())
    original = [json.loads(line) for line in (directory/"pairs.jsonl").read_text().splitlines()]
    assert len(original) == 3
    report = json.loads((directory/"pair_batch_report.json").read_text())
    assert len(report["batch_performance"]) == 2
    for row in report["batch_performance"]:
        assert row["peak_gpu_allocated_bytes"] is None
        assert row["timing"]["host_wall_seconds"]["checkpoint_and_logging"] > 0
    monkeypatch.setattr(module, "solve_partial_batch", real)
    fit_mapping(config, root=tmp_path, resume=directory)
    entries = [json.loads(line) for line in (directory/"pairs.jsonl").read_text().splitlines()]
    assert entries[:3] == original and len(entries) == 4
    assert not list((directory/"plans").rglob("latest.pt"))
    assert json.loads((directory/"manifest.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("key,value", [("newton_batch_size",True),("newton_batch_size",0),
    ("newton_check_every",-1),("newton_check_every",31),("inner_compact",1),("inner_warm_start","true")])
def test_invalid_execution_settings(key,value):
    with pytest.raises(ValueError, match=key):
        solver_config({key:value})


def test_memory_estimate_accounts_for_newton_subbatch():
    shape = (2000,196,768)
    small = resource_estimate(shape,shape,"grouped_partial",8,2048,64)
    large = resource_estimate(shape,shape,"grouped_partial",8,2048,128)
    assert small["estimated_working_gib"] < large["estimated_working_gib"] < 64


def test_benchmark_optimized_vs_reference_on_same_pairs(tmp_path):
    import yaml
    from infoot_vit.benchmark_partial_batch import main
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(dict(partial=dict(keep_mass=.8,
        solver=dict(reg=.4, **OPTIONS)))))
    output = tmp_path/"benchmark.json"
    assert main(["--config", str(path), "--device", "cpu", "--reference", "batched-reference",
        "--pairs", "3", "--pair-batch-size", "2", "--patches", "4", "--feature-dim", "3",
        "--repeats", "1", "--threads", "1", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "measured" and report["valid_speed_comparison"]
    assert set(report["runs"]) == {"batched", "batched-reference"}
    assert report["reference_solver"]["inner_warm_start"] is False
    assert report["solver"]["inner_warm_start"] is True
