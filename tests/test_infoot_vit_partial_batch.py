"""Exact batched dense partial InfoOT against the unchanged POT serial path."""
import json
import importlib

import pytest
import torch

from test_infoot_vit_mapping import banks, config, cpu_threads
from infoot_vit.infoot_helper.partial import (
    distance, kernel_state, information, information_gradient, entropy_subproblem,
    solve_partial, solver_config, feasibility, objective,
)
from infoot_vit.infoot_helper.partial_batch import (
    entropy_subproblem_batch, solve_partial_batch, objective_values,
)
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, validate_config, resource_estimate
from infoot_vit.infoot_helper.feature_bank import file_hash
from infoot_vit.infoot_helper.mapping import FeatureMapper


def batched_config(size=3):
    return dict(config("grouped_partial"), fit_pair_top_k=2, pair_batch_size=size,
                pair_checkpoint_every=2, device="cpu")


def problem(count=3, device="cpu"):
    rng = torch.Generator().manual_seed(260)
    xs = torch.randn(count, 4, 3, generator=rng, dtype=torch.float64).to(device)
    ys = torch.randn(count, 5, 3, generator=rng, dtype=torch.float64).to(device)
    kx = torch.stack([kernel_state(x, .7)[0] for x in xs])
    ky = torch.stack([kernel_state(y, .7)[0] for y in ys])
    return distance(xs, ys), kx, ky


@pytest.mark.parametrize("mass", [.3, .8, 1.])
def test_inner_matches_pot_with_nonuniform_caps_and_independent_counts(mass):
    costs, _, _ = problem()
    a = torch.tensor([.1, .2, .3, .4], dtype=torch.float64).expand(3, -1)
    b = torch.tensor([.1, .2, .15, .25, .3], dtype=torch.float64).expand(3, -1)
    c = solver_config(dict(reg=.4))
    actual, diagnostics = entropy_subproblem_batch(a, b, costs, mass, c)
    assert diagnostics["ok"].all()
    for i in range(3):
        expected, log = entropy_subproblem(a[i], b[i], costs[i], mass, c)
        torch.testing.assert_close(actual[i], expected, atol=1e-12, rtol=1e-11)
        assert float(diagnostics["error"][i]) == pytest.approx(log["error"], abs=1e-12)
        feasibility(actual[i], a[i], b[i], mass, c)


def test_information_and_gradient_keep_members_independent():
    costs, kx, ky = problem()
    plan = costs.softmax(-1) / costs.shape[1] * .8
    gradient = information_gradient(plan, kx, ky)
    for i in range(len(costs)):
        torch.testing.assert_close(information(plan, kx, ky)[i], information(plan[i], kx[i], ky[i]), atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(gradient[i], information_gradient(plan[i], kx[i], ky[i]), atol=1e-12, rtol=1e-12)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("mass,lam", [(.8, 0.), (.8, .1), (1., .1)])
def test_outer_agrees_with_serial_plans_objectives_and_stopping(count, mass, lam):
    cost, kx, ky = problem(count)
    c = solver_config(dict(reg=.4, lam=lam, max_outer_steps=100))
    actual, reports = solve_partial_batch(cost, kx, ky, keep_mass=mass, config=c)
    for i in range(count):
        expected, report = solve_partial(cost[i], kx[i], ky[i], keep_mass=mass, config=c)
        torch.testing.assert_close(actual[i], expected, atol=1e-10, rtol=1e-9)
        assert reports[i]["status"] == report["status"] == "converged"
        assert len(reports[i]["history"]) == len(report["history"])
        assert reports[i]["mass"] == pytest.approx(mass, abs=c["mass_tolerance"])
        for key in ("cost", "information", "entropy", "objective"):
            assert reports[i]["history"][-1][key] == pytest.approx(report["history"][-1][key], abs=1e-11)
        totals = [r["objective"] for r in reports[i]["history"]]
        assert all(b <= a + c["objective_tolerance"] for a, b in zip(totals, totals[1:]))


def test_invalid_member_and_inner_budget_do_not_break_neighbors():
    cost, kx, ky = problem()
    cost[0].fill_(1.)  # Converges after one inner iteration.
    cost[2, 0, 0] = float("nan")
    plans, reports = solve_partial_batch(cost, kx, ky, config=dict(lam=0., reg=.01, max_inner_steps=1))
    assert [r["status"] for r in reports] == ["converged", "inner_failed", "invalid_input"]
    assert torch.isfinite(plans).all()
    assert reports[1]["history"][-1]["inner"]["failure_code"] == 2
    json.dumps(reports, allow_nan=False)


def test_full_siglip_patch_shape_and_finite_output():
    costs = torch.ones(2, 196, 196, dtype=torch.float64)
    kernels = torch.eye(196, dtype=torch.float64).expand(2, -1, -1)
    plans, reports = solve_partial_batch(costs, kernels, kernels, config=dict(lam=0.))
    assert plans.shape == (2, 196, 196) and torch.isfinite(plans).all()
    assert all(r["status"] == "converged" for r in reports)
    torch.testing.assert_close(plans, torch.full_like(plans, .8 / 196**2), atol=1e-15, rtol=1e-12)


def test_mixed_inner_convergence_counts_stop_at_their_own_iteration():
    costs, _, _ = problem()
    costs[0].fill_(1.)
    c = solver_config(dict(reg=.05))
    a = costs.new_full((3, 4), 1/4)
    b = costs.new_full((3, 5), 1/5)
    plans, info = entropy_subproblem_batch(a, b, costs, .8, c)
    single, solo = entropy_subproblem_batch(a[:1], b[:1], costs[:1], .8, c)
    assert info["iterations"][0] == solo["iterations"][0] == 1
    assert (info["iterations"][1:] > 1).all()
    assert info["ok"].all()
    torch.testing.assert_close(plans[0], single[0], atol=0, rtol=0)


def test_outer_budget_keeps_last_accepted_iterate_and_matches_serial():
    cost, kx, ky = problem()
    c = solver_config(dict(reg=.4, max_outer_steps=1, cost_scale=2.))
    plans, reports = solve_partial_batch(cost, kx, ky, config=c)
    for i, report in enumerate(reports):
        expected, reference = solve_partial(cost[i], kx[i], ky[i], config=c)
        torch.testing.assert_close(plans[i], expected, atol=1e-12, rtol=1e-11)
        assert report["status"] == reference["status"] == "max_outer_steps"
        assert report["history"][-1]["objective_delta"] <= c["objective_tolerance"]


def test_line_search_stall_is_local_and_never_accepts_uphill_plan(monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.partial_batch")
    costs = torch.tensor([[[0., 1.], [1., 0.]], [[1., 1.], [1., 1.]]], dtype=torch.float64)
    kernels = torch.stack([torch.eye(2), torch.ones(2, 2)]).double()
    def uphill(a, b, cost, mass, config):
        candidate = torch.tensor([[[0., .4], [.4, 0.]], [[.2, .2], [.2, .2]]], dtype=torch.float64)
        return candidate, dict(ok=torch.ones(2, dtype=torch.bool), error=torch.zeros(2),
            iterations=torch.ones(2, dtype=torch.long), failure=torch.zeros(2, dtype=torch.long),
            cost_shift=torch.zeros(2), residuals=module.residual_values(candidate, a, b, mass), method="test")
    monkeypatch.setattr(module, "entropy_subproblem_batch", uphill)
    plans, reports = module.solve_partial_batch(costs, kernels, kernels, config=dict(cost_scale=1., max_backtracks=1))
    assert [r["status"] for r in reports] == ["line_search_stalled", "converged"]
    torch.testing.assert_close(plans, torch.full_like(plans, .2), atol=0, rtol=0)
    assert reports[0]["history"][-1]["accepted_plan_delta_l1"] == 0.


def test_cli_batch_overrides_and_config_defaults(tmp_path, monkeypatch):
    import yaml
    from infoot_vit import infoot_fit
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(batched_config()))
    seen = []
    monkeypatch.setattr(infoot_fit, "fit_mapping", lambda config, **kwargs: seen.append(config) or tmp_path)
    for option, expected in ((["--pair-batch-size", "1"], 1), (["--serial-pairs"], None)):
        assert infoot_fit.main(["--config", str(path), "--device", "cpu", *option]) == 0
        assert seen[-1]["pair_batch_size"] == expected and seen[-1]["device"] == "cpu"
    actual = yaml.safe_load((infoot_fit.ROOT / "infoot_vit/configs/grouped_partial.yaml").read_text())
    assert actual["device"] == "cuda" and actual["pair_batch_size"] == 256
    assert actual["fit_pair_top_k"] == 8 and actual["partial"]["keep_mass"] == .8


def test_converged_outer_members_are_never_updated_again():
    cost, kx, ky = problem()
    cost[0].fill_(1.)
    kx[0].fill_(1.); ky[0].fill_(1.)
    snapshots = []
    def observe(plans, reports):
        snapshots.append((plans.clone(), [None if r is None else r["status"] for r in reports]))
    plans, reports = solve_partial_batch(cost, kx, ky, config=dict(reg=.4), on_step=observe)
    assert snapshots[0][1][0] == "converged" and len(snapshots) > 1
    assert all(row[1][0] is None for row in snapshots[1:])
    assert all(torch.equal(row[0][0], snapshots[0][0][0]) for row in snapshots)
    assert reports[0]["iterations"] == 1 < reports[1]["iterations"]


def test_fit_incomplete_batch_save_load_and_serial_equivalence(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.pair_batch_fit")
    real = module.solve_partial_batch
    sizes = []
    def observe(cost, kx, ky, **kwargs):
        sizes.append(len(cost))
        assert cost.dtype == kx.dtype == ky.dtype == torch.float64
        assert cost.device == kx.device == ky.device == torch.device("cpu")
        return real(cost, kx, ky, **kwargs)
    monkeypatch.setattr(module, "solve_partial_batch", observe)
    batched = FeatureMapper.load(fit_mapping(batched_config(), root=tmp_path))
    serial = FeatureMapper.load(fit_mapping(batched_config(None), root=tmp_path))
    assert sizes == [3, 1]  # Four selected pairs; no dropped or padded member.
    assert len(batched.pairs) == len(serial.pairs) == 4
    for key, entry in batched.pairs.items():
        state = torch.load(batched.directory / entry["file"], weights_only=True)
        expected = torch.load(serial.directory / serial.pairs[key]["file"], weights_only=True)
        assert state["plan"].dtype == torch.float32
        assert torch.equal(state["r"], state["plan"].double().sum(1))
        torch.testing.assert_close(state["plan"], expected["plan"], atol=1e-8, rtol=1e-7)
    assert not list((batched.directory / "plans").rglob("latest.pt"))
    assert len(list((batched.directory / "plans/pairs").rglob("iterations.jsonl"))) == 4
    query = banks[2]
    args = dict(valid_mask=torch.ones(query.features.shape[:2], dtype=torch.bool), return_metadata=True)
    left = batched.map_features(query.features, query.ids, **args)
    right = serial.map_features(query.features, query.ids, **args)
    torch.testing.assert_close(left.mapped_features, right.mapped_features, atol=1e-8, rtol=1e-7)


def test_failure_isolated_successes_saved_and_resume_skips_them(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.pair_batch_fit")
    real = module.solve_partial_batch
    calls = 0
    def fail_one(cost, kx, ky, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            cost = cost.clone(); cost[0, 0, 0] = float("nan")
        return real(cost, kx, ky, **kwargs)
    monkeypatch.setattr(module, "solve_partial_batch", fail_one)
    with pytest.raises(RuntimeError, match="1 partial pairs failed; 3 successful pairs saved"):
        fit_mapping(batched_config(), root=tmp_path)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    entries = [json.loads(s) for s in (directory / "pairs.jsonl").read_text().splitlines()]
    hashes = {e["file"]: file_hash(directory / e["file"]) for e in entries}
    failure = json.loads((directory / "pair_failures.jsonl").read_text())
    assert failure["status"] == "invalid_input"
    assert (directory / failure["checkpoint"]).exists()
    assert len(list((directory / "plans").rglob("latest.pt"))) == 1
    resumed_sizes = []
    def resume(cost, kx, ky, **kwargs):
        resumed_sizes.append(len(cost))
        # Rebuilt kernels retain f64 arithmetic, not a promoted f32 snapshot.
        assert not torch.equal(kx, kx.float().double())
        return real(cost, kx, ky, **kwargs)
    monkeypatch.setattr(module, "solve_partial_batch", resume)
    fit_mapping(batched_config(), root=tmp_path, resume=directory)
    assert resumed_sizes == [1]
    assert all(file_hash(directory / path) == sha for path, sha in hashes.items())
    assert not list((directory / "plans").rglob("latest.pt"))
    assert len(FeatureMapper.load(directory).pairs) == 4
    assert json.loads((directory / "manifest.json").read_text())["failed_pair_count"] == 0
    assert (directory / "pair_failures.jsonl").exists()  # Historical evidence remains.


def test_interrupt_after_registration_preserves_neighbors_and_resumes(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.fit_mapping")
    cleanup = module._cleanup_latest
    def interrupt(directory, entry, log):
        if entry["file"].startswith("plans/pairs/"):
            raise KeyboardInterrupt()
        cleanup(directory, entry, log)
    monkeypatch.setattr(module, "_cleanup_latest", interrupt)
    with pytest.raises(KeyboardInterrupt):
        fit_mapping(batched_config(), root=tmp_path)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    entry = json.loads((directory / "pairs.jsonl").read_text())
    original_hash = file_hash(directory / entry["file"])
    assert (directory / entry["file"]).with_suffix("").joinpath("latest.pt").exists()
    monkeypatch.setattr(module, "_cleanup_latest", cleanup)
    fit_mapping(batched_config(), root=tmp_path, resume=directory)
    assert file_hash(directory / entry["file"]) == original_hash
    assert len(FeatureMapper.load(directory).pairs) == 4
    assert not list((directory / "plans").rglob("latest.pt"))


def test_failed_verification_does_not_discard_successful_neighbors(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.fit_mapping")
    verify = module._verified_save
    calls = 0
    def fail(directory, path, state, validator):
        nonlocal calls
        result = verify(directory, path, state, validator)
        if "pairs" in path.parts:
            calls += 1
            if calls == 1:
                raise ValueError("one simulated verification failure")
        return result
    monkeypatch.setattr(module, "_verified_save", fail)
    with pytest.raises(RuntimeError, match="3 successful pairs saved"):
        fit_mapping(batched_config(), root=tmp_path)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    assert len((directory / "pairs.jsonl").read_text().splitlines()) == 3
    failed = json.loads((directory / "pair_failures.jsonl").read_text())
    assert failed["status"] == "save_or_validation_failed"
    assert (directory / failed["checkpoint"]).exists()
    monkeypatch.setattr(module, "_verified_save", verify)
    fit_mapping(batched_config(), root=tmp_path, resume=directory)
    assert len(FeatureMapper.load(directory).pairs) == 4


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_batch_size_rejected(value):
    with pytest.raises(ValueError, match="pair_batch_size"):
        validate_config(batched_config(value))


def test_batch_memory_and_checkpoint_estimates_scale():
    shape = (2000, 196, 768)
    serial = resource_estimate(shape, shape, "grouped_partial", 8)
    batch = resource_estimate(shape, shape, "grouped_partial", 8, 256)
    assert batch["estimated_working_gib"] > serial["estimated_working_gib"]
    assert batch["latest_plan_storage_bytes"] == 2 * 256 * 196 * 196 * 4
    assert batch["plan_storage_bytes"] == serial["plan_storage_bytes"]


def test_benchmark_cpu_smoke_and_cuda_unavailable_report(tmp_path, monkeypatch):
    import yaml
    from infoot_vit.benchmark_partial_batch import main
    yaml_path = tmp_path / "benchmark.yaml"
    yaml_path.write_text(yaml.safe_dump(dict(partial=dict(keep_mass=.8, solver=dict(reg=.4, lam=.1, h=.7)))))
    output = tmp_path / "benchmark.json"
    assert main(["--config", str(yaml_path), "--device", "cpu", "--pairs", "3", "--patches", "4",
        "--feature-dim", "3", "--pair-batch-size", "2", "--threads", "1", "--repeats", "1", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "measured" and report["valid_speed_comparison"]
    assert report["agreement"]["max_plan_abs_error"] < 1e-10
    assert report["runs"]["batched"][0]["peak_allocated_bytes"] is None
    assert report["summary"]["serial"]["median_pairs_per_second"] > 0
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "unverified"


def test_benchmark_records_solver_failure_instead_of_claiming_valid_speed(tmp_path):
    import yaml
    from infoot_vit.benchmark_partial_batch import main
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(dict(partial=dict(keep_mass=.8, solver=dict(reg=.01, max_inner_steps=1)))))
    output = tmp_path / "failed.json"
    assert main(["--config", str(path), "--device", "cpu", "--pairs", "2", "--patches", "4",
                 "--feature-dim", "3", "--threads", "1", "--repeats", "1", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "solver_failed" and not report["valid_speed_comparison"]
    assert report["agreement"]["failures"]["serial"]
    assert report["agreement"]["failures"]["batched"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_stays_on_device_and_matches_serial_cuda():
    cost, kx, ky = problem(device="cuda")
    c = solver_config(dict(reg=.4))
    def observe(plans, reports):
        assert plans.device.type == "cuda" and plans.dtype == torch.float64
    actual, reports = solve_partial_batch(cost, kx, ky, config=c, on_step=observe)
    for i in range(len(cost)):
        expected, report = solve_partial(cost[i], kx[i], ky[i], config=c)
        torch.testing.assert_close(actual[i], expected, atol=1e-9, rtol=1e-8)
        assert reports[i]["status"] == report["status"]
