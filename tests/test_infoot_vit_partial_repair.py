"""Near-stationary precision recovery and immutable failed-pair migration."""
import importlib
import json

import pytest
import torch

from test_infoot_vit_mapping import banks, cpu_threads
from test_infoot_vit_partial_batch import batched_config
from infoot_vit.infoot_helper.feature_bank import digest, file_hash, write_json, save_tensor
from infoot_vit.infoot_helper.fit_mapping import fit_mapping
from infoot_vit.infoot_helper.mapping import FeatureMapper
from infoot_vit.infoot_helper.partial import solver_config, kernel_state, distance
from infoot_vit.infoot_helper.partial_repair import REPAIR_PREDECESSOR, prepare_partial_repair, repair_partial_fit
from infoot_vit.infoot_helper.storage import VERSION as STORAGE_VERSION


def test_real_196_patch_nonzero_mi_solve_preserves_strict_certificates():
    from infoot_vit.infoot_helper.partial_batch import solve_partial_batch
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(196, 6, generator=generator, dtype=torch.float64)
    y = torch.randn(196, 6, generator=generator, dtype=torch.float64)
    kx, ky = (kernel_state(features, .35)[0] for features in (x, y))
    config = solver_config(dict(h=.35, reg=.04, lam=.035, inner_acceleration="newton",
                                inner_warm_start=True, inner_compact=True))
    plans, reports = solve_partial_batch(distance(x, y)[None], kx[None], ky[None], config=config)
    report = reports[0]
    assert report["status"] == "converged" and plans.dtype == torch.float64
    assert report["history"][-1]["plan_delta_l1"] <= config["outer_tolerance"]
    assert report["mass_error"] <= config["mass_tolerance"]
    assert max(report["row_cap_error"], report["column_cap_error"]) <= config["feasibility_tolerance"]
    for step in report["history"]:
        assert step["objective_delta"] <= config["objective_tolerance"]
        for key in ("kkt_error", "relative_duality_gap", "relative_plan_delta_l1"):
            assert step["inner"][key] <= step["inner"]["certificate_tolerance"]


def test_certified_inner_precision_can_stall_then_refine_without_relaxing_outer(monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.partial_batch")
    cost = torch.tensor([[[1.-1e-8, 1.+1e-8], [1.+1e-8, 1.-1e-8]],
                         [[1., 1.], [1., 1.]]], dtype=torch.float64)
    kernels = torch.ones_like(cost)
    config = solver_config(dict(reg=.04, lam=.035, cost_scale=1., inner_acceleration="newton"))
    real = module.entropy_subproblem_batch
    calls = []
    def approximate(a, b, effective, mass, cfg, **kwargs):
        plan, info = real(a, b, effective, mass, cfg, **kwargs)
        calls.append((len(effective), cfg["inner_tolerance"]))
        if len(calls) == 1:
            # A tiny mass-normalization error is legal under ALL original
            # certificates but dominates the near-fixed-point objective gain.
            previous = plan.log()
            plan[0] *= (mass + 4e-12) / mass
            log_plan = plan.log()
            base = -(effective - effective.amin((-2, -1))[:, None, None]) / cfg["reg"]
            u, v = info["dual_state"]
            w = (log_plan - base - u[:, :, None] - v[:, None, :]).mean((-2, -1))
            info["relative_plan_delta_l1"], info["kkt_error"] = module._partial_certificate(
                plan, log_plan, previous, base, a, b, mass, u, v, w)
            info["duality_gap"], info["relative_duality_gap"] = module._duality_gap(
                plan, log_plan, base, a, b, mass, u, v, w, cfg["reg"])
            info["residuals"] = module.residual_values(plan, a, b, mass)
            assert module.feasible(info["residuals"], cfg).all()
            for key in ("relative_plan_delta_l1", "kkt_error", "relative_duality_gap"):
                assert (info[key] <= cfg["inner_tolerance"]).all()
            initial = torch.full_like(plan, .2)
            old = module.objective_values(initial, cost, kernels, kernels, cfg)
            status = module._outer_step(initial, plan, old, cost, kernels, kernels, cfg,
                                        info["ok"], info["ok"])[-1]
            assert status.tolist() == [module.STALLED, module.CONVERGED]
        return plan, info
    monkeypatch.setattr(module, "entropy_subproblem_batch", approximate)
    plans, reports = module.solve_partial_batch(cost, kernels, kernels, config=config)
    assert calls == [(2, 1e-10), (1, 1e-12), (1, 1e-10)]
    assert all(r["status"] == "converged" for r in reports)
    assert plans.dtype == torch.float64
    first = reports[0]["history"][0]
    assert first["plan_delta_l1"] > config["outer_tolerance"]
    assert first["status"] == "running" and first["step_size"] == 1.
    assert first["inner"]["precision_refinements"] == 1
    assert first["inner"]["certificate_tolerance"] == 1e-12
    assert first["objective_delta"] <= config["objective_tolerance"]
    assert reports[0]["history"][-1]["plan_delta_l1"] <= config["outer_tolerance"]
    assert reports[1]["history"][0]["inner"]["precision_refinements"] == 0


@pytest.fixture
def failed_fit(tmp_path, banks, monkeypatch):
    orchestration = importlib.import_module("infoot_vit.infoot_helper.fit_mapping")
    batch = importlib.import_module("infoot_vit.infoot_helper.pair_batch_fit")
    real_identity, real_solve = orchestration.fit_identity, batch.solve_partial_batch
    def old_identity(config, source, target, runtime):
        _, implementation = real_identity(config, source, target, runtime)
        implementation.update(REPAIR_PREDECESSOR)
        return digest(dict(config=config, source=source.artifact_id, target=target.artifact_id,
            source_ids=source.ids, target_ids=target.ids, storage=STORAGE_VERSION,
            implementation=implementation, runtime=runtime)), implementation
    calls = 0
    def fail_one(cost, kx, ky, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            cost = cost.clone()
            cost[0, 0, 0] = float("nan")
        return real_solve(cost, kx, ky, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(orchestration, "fit_identity", old_identity)
        patch.setattr(batch, "solve_partial_batch", fail_one)
        with pytest.raises(RuntimeError, match="1 partial pairs failed; 3 successful pairs saved"):
            fit_mapping(batched_config(), root=tmp_path)
    return next((tmp_path / "outputs/infoot_vit").iterdir())


def test_repair_migrates_known_solver_and_retries_only_failure(tmp_path, failed_fit, monkeypatch):
    original = {p.relative_to(failed_fit).as_posix(): file_hash(p) for p in failed_fit.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Resume fingerprint mismatch"):
        fit_mapping(batched_config(), root=tmp_path, resume=failed_fit)
    # A rejected normal resume writes a diagnostic log, but never changes the
    # manifest or plans; the repair itself must leave every source file intact.
    original = {p.relative_to(failed_fit).as_posix(): file_hash(p) for p in failed_fit.rglob("*") if p.is_file()}
    module = importlib.import_module("infoot_vit.infoot_helper.pair_batch_fit")
    real, sizes = module.solve_partial_batch, []
    def observe(cost, kx, ky, **kwargs):
        sizes.append(len(cost))
        assert cost.dtype == kx.dtype == ky.dtype == torch.float64
        assert not torch.equal(kx, kx.float().double())
        return real(cost, kx, ky, **kwargs)
    monkeypatch.setattr(module, "solve_partial_batch", observe)
    output = repair_partial_fit(failed_fit, tmp_path / "repaired", root=tmp_path)
    assert sizes == [1]
    assert original == {p.relative_to(failed_fit).as_posix(): file_hash(p) for p in failed_fit.rglob("*") if p.is_file()}
    mapper = FeatureMapper.load(output)
    assert len(mapper.pairs) == 4
    manifest = json.loads((output / "manifest.json").read_text())
    old_manifest = json.loads((failed_fit / "manifest.json").read_text())
    assert old_manifest["status"] == "failed" and manifest["status"] == "complete"
    assert manifest["fit_fingerprint"] != old_manifest["fit_fingerprint"]
    assert manifest["config"] == old_manifest["config"]
    assert manifest["repair"]["reused_successful_pairs"] == 3
    assert manifest["repair"]["pairs_to_retry"] == 1
    assert set(manifest["repair"]["changed_implementation_files"]) == set(REPAIR_PREDECESSOR)
    assert manifest["models"] == old_manifest["models"]
    assert manifest["pair_selection"] == old_manifest["pair_selection"]
    assert manifest["pair_kernels"] == old_manifest["pair_kernels"]
    for line in (failed_fit / "pairs.jsonl").read_text().splitlines():
        entry = json.loads(line)
        before = torch.load(failed_fit / entry["file"], weights_only=True)
        after = torch.load(output / entry["file"], weights_only=True)
        assert before["report"] == after["report"]
        for name in ("plan", "a", "b", "r", "c"):
            assert torch.equal(before[name], after[name]) and before[name].dtype == after[name].dtype
        assert after["plan"].dtype == torch.float32
        assert after["reused_from"]["sha256"] == entry["sha256"]


@pytest.mark.parametrize("damage", ["outer", "inner", "capacity", "mass"])
def test_repair_rejects_unsubstantiated_success(tmp_path, failed_fit, damage):
    manifest = json.loads((failed_fit / "manifest.json").read_text())
    rows = [json.loads(line) for line in (failed_fit / "pairs.jsonl").read_text().splitlines()]
    entry = rows[0]
    path = failed_fit / entry["file"]
    state = torch.load(path, weights_only=True)
    if damage == "outer":
        state["report"]["history"][-1]["status"] = "line_search_stalled"
    elif damage == "inner":
        state["report"]["history"][-1]["inner"]["kkt_error"] = 1.
    elif damage == "capacity":
        state["a"] = state["a"] * 1.1
    else:
        state["report"]["keep_mass"] = .7
    save_tensor(path, state)
    entry["sha256"] = file_hash(path)
    index = failed_fit / "pairs.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest["pair_inventory"]["sha256"] = file_hash(index)
    write_json(failed_fit / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Repair"):
        prepare_partial_repair(failed_fit, tmp_path / "repaired", root=tmp_path)
    assert not (tmp_path / "repaired/manifest.json").exists()


def test_repair_refuses_overwrite_and_unknown_solver_lineage(tmp_path, failed_fit, monkeypatch):
    with pytest.raises(ValueError, match="separate new directory"):
        prepare_partial_repair(failed_fit, failed_fit / "repair", root=tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_partial_repair(failed_fit, existing, root=tmp_path)
    module = importlib.import_module("infoot_vit.infoot_helper.partial_repair")
    monkeypatch.setattr(module, "REPAIR_PREDECESSOR", dict(REPAIR_PREDECESSOR, **{"partial_batch.py": "unknown"}))
    with pytest.raises(ValueError, match="unrelated implementation"):
        prepare_partial_repair(failed_fit, tmp_path / "repaired", root=tmp_path)
    assert not (tmp_path / "repaired").exists()
