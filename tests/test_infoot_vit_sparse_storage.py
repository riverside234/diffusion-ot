"""Storage-efficient grouped_partial experiment; other mapping modes are controls."""
from copy import deepcopy
import importlib
import json

import pytest
import torch

from test_infoot_vit_mapping import banks, config, bank, mapped, cpu_threads
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, inspect_fit, resource_estimate, validate_config
from infoot_vit.infoot_helper.feature_bank import file_hash, save_tensor
from infoot_vit.infoot_helper.mapping import FeatureMapper
from infoot_vit.infoot_helper.pair_selection import select_pairs
from infoot_vit.infoot_helper.sampling import sample_ids
from infoot_vit.infoot_helper.storage import store_plan_state, validate_plan, store_kernels, load_kernels
from infoot_vit.infoot_helper.partial import solver_config, feasibility, kernel_state
from infoot_vit.infoot_helper.conditional import partial_projection


def sparse_config(k=1, count=None):
    c = config("grouped_partial")
    c.update(fit_pair_top_k=k, sampling=dict(images_per_domain=count, seed=42))
    return c


def test_sampling_2000_fixed_seed_order_independence_and_no_replacement():
    ids = [f"cat_train_{i:04d}" for i in range(2500)]
    selected = sample_ids(ids, 2000, 42)
    assert selected == sample_ids(list(reversed(ids)), 2000, 42)
    assert selected == sorted(selected) and len(set(selected)) == 2000
    assert selected != sample_ids(ids, 2000, 43)
    with pytest.raises(ValueError, match="no replacement"):
        sample_ids(ids, 2501, 42)


def test_pair_selection_uses_conditional_probabilities_with_stable_ties():
    class Router:
        source = torch.arange(2).reshape(2, 1)
        def conditional_weights(self, q):
            return torch.tensor([[.4, .4, .2], [.1, .3, .6]], dtype=torch.float64)[q[:, 0]]
    selection = select_pairs(Router(), ["s0", "s1"], ["z", "a", "b"], 1, chunk_size=1)
    assert [row["target_ids"] for row in selection["rows"]] == [["a"], ["b"]]
    assert selection["rows"][0]["retained_probability"] == .4
    assert selection["rows"][0]["discarded_probability"] == .6


def test_fit_sampling_router_shape_identity_and_unsampled_split_leakage(tmp_path, banks):
    cfg = sparse_config(count=2)
    report = inspect_fit(cfg, tmp_path)
    target_ids = sample_ids(banks[1].ids, 2, 42)
    assert report["resources"]["source_shape"][0] == 2
    assert report["resources"]["target_shape"][0] == 2
    mapper = FeatureMapper.load(fit_mapping(cfg, root=tmp_path))
    assert mapper.image.plan.shape == (2, 2)
    assert mapper.target.ids == target_ids
    assert mapper.manifest["sampling"]["target"]["seed"] == 42
    assert mapper.manifest["sampling"]["target"]["ordered_ids"] == target_ids
    excluded = next(i for i in banks[1].ids if i not in target_ids)
    bad = bank(tmp_path, "leaking_query", banks[2].features[:1], "cat", "val", ids=[excluded])
    with pytest.raises(ValueError, match="held-out"):
        mapper.project_bank(bad, tmp_path / "leak_result")
    from infoot_vit.infoot_test import main as test_main
    with pytest.raises(ValueError, match="held-out"):
        test_main(["--mapping", str(mapper.directory), "--query-bank", str(bad.path), "--dry-run", "--threads", "1"])


def test_float32_round_trip_marginals_and_storage_budget_are_not_solver_tolerances():
    c = solver_config(dict(feasibility_tolerance=1e-12, mass_tolerance=1e-12))
    a = b = torch.full((3,), 1/3, dtype=torch.float64)
    plan = torch.outer(a, b)
    state = store_plan_state(dict(plan=plan, a=a, b=b))
    assert state["plan"].dtype == torch.float32
    assert state["r"].dtype == torch.float64
    assert torch.equal(state["r"], state["plan"].double().sum(1))
    errors = state["storage"]["quantization"]
    assert errors["l1_error"] > 0 and errors["row_sum_error_max"] > 0
    report = validate_plan(state, a, b, 1., c, balanced=True)
    assert 1e-12 < report["mass_error"] < report["mass_tolerance"] < 1e-7
    with pytest.raises(ValueError, match="Infeasible"):
        feasibility(state["plan"].double(), a, b, 1., c)  # Strict live solver still rejects this!
    corrupt = deepcopy(state)
    corrupt["plan"] *= 1.001
    corrupt["r"], corrupt["c"] = corrupt["plan"].double().sum(1), corrupt["plan"].double().sum(0)
    with pytest.raises(ValueError, match="storage tolerance"):
        validate_plan(corrupt, a, b, 1., c, balanced=True)
    wrong_dtype = deepcopy(state); wrong_dtype["plan"] = wrong_dtype["plan"].double()
    with pytest.raises(ValueError, match="dtype/version"):
        validate_plan(wrong_dtype, a, b, 1., c)


def test_kernel_quantization_reports_underflow_and_loads_as_double():
    k = torch.tensor([[[1., 1e-60], [1e-60, 1.]]], dtype=torch.float64)
    stored = store_kernels(dict(kx=k, ky=k))
    assert stored["kx"].dtype == torch.float32
    assert stored["storage"]["quantization"]["kx"]["underflow_entries"] == 2
    loaded = load_kernels(stored)
    assert loaded["ky"].dtype == torch.float64 and loaded["ky"][0, 0, 1] == 0


def test_sparse_projection_normalizes_only_saved_routes_and_separates_rejection(tmp_path, banks, monkeypatch):
    directory = fit_mapping(sparse_config(), root=tmp_path)
    mapper = FeatureMapper.load(directory)
    assert len(mapper.pairs) == 2
    monkeypatch.setattr(importlib.import_module("infoot_vit.infoot_helper.fit_mapping"), "solve_partial",
                        lambda *a, **k: pytest.fail("No inference fitting"))
    actual = mapped(mapper, banks[2], return_metadata=True)
    for qi, q in enumerate(banks[2].features):
        theta = mapper.image.pair_weights(q.reshape(1, -1))[0]
        retained = theta[mapper.pair_mask].sum()
        theta = theta.masked_fill(~mapper.pair_mask, 0.) / retained
        numerator, denominator = torch.zeros_like(q), torch.zeros(len(q), dtype=torch.float64)
        for (sid, tid) in mapper.pairs:
            i, j = mapper.source_index[sid], mapper.target_index[tid]
            state = mapper._pair(sid, tid)
            candidate, g, _ = partial_projection(q, mapper.x[i], mapper.y[j], state["plan"], state["a"], state["b"],
                mapper.shared["sx"][i], mapper.shared["sy"][j], mapper.shared["h_projection"], mapper.shared["support"][i],
                target_kernel=mapper.projection_ky[j])
            numerator += theta[i, j] * g[:, None] * candidate
            denominator += theta[i, j] * g
        torch.testing.assert_close(actual.mapped_features[qi], numerator / denominator[:, None], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual.match_confidence[qi], denominator, atol=1e-12, rtol=1e-12)
        detail = actual.diagnostics["queries"][qi]
        assert 0 < detail["fit_pair_discarded_routing_mass"] < 1
        assert detail["fit_pair_retained_routing_mass"] == pytest.approx(float(retained))
        accounted = actual.match_confidence[qi] + torch.tensor(detail["ot_rejected_mass"], dtype=torch.float64) + torch.tensor(detail["support_invalid_retained_mass"], dtype=torch.float64)
        torch.testing.assert_close(accounted, torch.ones_like(accounted), atol=1e-12, rtol=0)
        assert detail["matched_rejected_group_mass_residual"] < 1e-12


@pytest.mark.parametrize("k", [3, 99])
def test_sparse_full_pair_equivalence_when_k_covers_every_target(tmp_path, banks, k):
    full = FeatureMapper.load(fit_mapping(sparse_config(None), root=tmp_path))
    all_k = FeatureMapper.load(fit_mapping(sparse_config(k), root=tmp_path))
    a, b = [mapped(m, banks[2], return_metadata=True) for m in (full, all_k)]
    assert full.pairs.keys() == all_k.pairs.keys()
    torch.testing.assert_close(a.mapped_features, b.mapped_features, atol=0, rtol=0)
    torch.testing.assert_close(a.match_confidence, b.match_confidence, atol=0, rtol=0)


def test_missing_selected_pair_is_not_intentionally_excluded(tmp_path, banks):
    directory = fit_mapping(sparse_config(), root=tmp_path)
    mapper = FeatureMapper.load(directory)
    excluded = next((s, t) for s in mapper.source.ids for t in mapper.target.ids if (s, t) not in mapper.pairs)
    with pytest.raises(KeyError, match="intentionally excluded"):
        mapper._pair(*excluded)
    entry = next(iter(mapper.pairs.values()))
    path = directory / entry["file"]
    with path.open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="Missing/changed"):
        FeatureMapper.load(directory)
    path.unlink()
    with pytest.raises(ValueError, match="Missing/changed"):
        FeatureMapper.load(directory)


def test_checkpoint_cleanup_resume_and_exact_double_kernel_rebuild(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.fit_mapping")
    original = module.solve_partial
    calls = 0
    def interrupted(cost, kx, ky, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            kwargs["on_step"](cost.new_full(cost.shape, .8 / cost.numel()), dict(iteration=1))
            raise RuntimeError("interrupted pair")
        return original(cost, kx, ky, **kwargs)
    monkeypatch.setattr(module, "solve_partial", interrupted)
    with pytest.raises(RuntimeError, match="interrupted pair"):
        fit_mapping(sparse_config(), root=tmp_path)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    entries = [json.loads(line) for line in (directory / "pairs.jsonl").read_text().splitlines()]
    assert len(entries) == 1
    final = directory / entries[0]["file"]
    assert not (final.with_suffix("") / "latest.pt").exists()
    before = file_hash(final)
    assert len(list((directory / "plans").rglob("latest.pt"))) == 1  # Failed pair retained.
    latest = torch.load(next((directory / "plans").rglob("latest.pt")), weights_only=True)
    assert latest["plan"].dtype == torch.float32
    def resumed(cost, kx, ky, **kwargs):
        assert kx.dtype == ky.dtype == torch.float64
        assert any(torch.equal(kx, kernel_state(x.double(), .7)[0]) for x in banks[0].features)
        assert not torch.equal(kx, kx.float().double())
        return original(cost, kx, ky, **kwargs)
    monkeypatch.setattr(module, "solve_partial", resumed)
    monkeypatch.setattr(module, "select_pairs", lambda *a, **k: pytest.fail("Must use saved selection"))
    fit_mapping(sparse_config(), root=tmp_path, resume=directory)
    assert file_hash(final) == before
    assert not list((directory / "plans").rglob("latest.pt"))
    assert list((directory / "plans").rglob("iterations.jsonl"))
    assert len(FeatureMapper.load(directory).pairs) == 2
    router = torch.load(directory / "plans/image.pt", weights_only=True)
    assert router["plan"].dtype == torch.float32 and router["r"].dtype == torch.float64
    kernels = torch.load(directory / "plans/pair_kernels.pt", weights_only=True)
    assert kernels["kx"].dtype == kernels["ky"].dtype == torch.float32


def test_failed_final_verification_keeps_latest_and_does_not_register(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.fit_mapping")
    verify = module._verified_save
    def failed(*args, **kwargs):
        verify(*args, **kwargs)
        raise ValueError("simulated final verification failure")
    monkeypatch.setattr(module, "_verified_save", failed)
    with pytest.raises(ValueError, match="verification failure"):
        fit_mapping(sparse_config(), root=tmp_path)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["models"] == {}
    assert (directory / "plans/image/latest.pt").exists()
    monkeypatch.setattr(module, "_verified_save", verify)
    fit_mapping(sparse_config(), root=tmp_path, resume=directory)
    assert not list((directory / "plans").rglob("latest.pt"))


def test_compact_fit_and_map_cli_and_resume_selection_fingerprint(tmp_path, banks):
    import yaml
    from infoot_vit.infoot_fit import main as fit_main
    from infoot_vit.infoot_test import main as map_main
    cfg_path = tmp_path / "sparse.yaml"
    cfg_path.write_text(yaml.safe_dump(config("grouped_partial")))
    args = ["--config", str(cfg_path), "--project-root", str(tmp_path), "--threads", "1", "--sample-images", "2", "--fit-pair-top-k", "1"]
    assert fit_main(args) == 0
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    assert map_main(["--mapping", str(directory), "--query-bank", str(banks[2].path), "--dry-run", "--threads", "1"]) == 0
    assert map_main(["--mapping", str(directory), "--query-bank", str(banks[2].path), "--threads", "1",
                     "--output-dir", str(tmp_path / "mapped")]) == 0
    changed = sparse_config(count=2); changed["sampling"]["seed"] = 43
    with pytest.raises(ValueError, match="fingerprint"):
        fit_mapping(changed, root=tmp_path, resume=directory)


def test_resume_cleans_registered_plan_after_interruption_before_cleanup(tmp_path, banks, monkeypatch):
    module = importlib.import_module("infoot_vit.infoot_helper.fit_mapping")
    cleanup = module._cleanup_latest
    def interrupted(directory, entry, log):
        if entry["file"].startswith("plans/pairs/"):
            raise RuntimeError("registered before cleanup")
        return cleanup(directory, entry, log)
    monkeypatch.setattr(module, "_cleanup_latest", interrupted)
    with pytest.raises(RuntimeError, match="registered before cleanup"):
        fit_mapping(sparse_config(), root=tmp_path)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    entry = json.loads((directory / "pairs.jsonl").read_text())
    assert (directory / entry["file"]).with_suffix("").joinpath("latest.pt").exists()
    before = file_hash(directory / entry["file"])
    monkeypatch.setattr(module, "_cleanup_latest", cleanup)
    fit_mapping(sparse_config(), root=tmp_path, resume=directory)
    assert file_hash(directory / entry["file"]) == before
    assert not list((directory / "plans").rglob("latest.pt"))


def test_resource_estimates_and_scope():
    shape = (2000, 196, 768)
    sparse = resource_estimate(shape, shape, "grouped_partial", 8)
    full = resource_estimate(shape, shape, "grouped_partial")
    assert sparse["pairs"] == 16000 and full["pairs"] == 4000000
    assert sparse["plan_storage_bytes"] == 16000 * 196 * 196 * 4
    assert sparse["plan_storage_bytes"] * 250 == full["plan_storage_bytes"]
    for mode in ("patch_global", "grouped_patch", "whole_map"):
        c = config(mode); c["fit_pair_top_k"] = 8
        with pytest.raises(ValueError, match="only to grouped_partial"):
            validate_config(c)
        assert resource_estimate(shape, shape, mode)["storage_dtype"] == "float64"
