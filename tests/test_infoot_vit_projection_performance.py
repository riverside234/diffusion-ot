"""Exact saved-pair projection, cache integrity and batching regression checks."""
from copy import deepcopy
import json

import pytest
import torch

from test_infoot_vit_mapping import banks, config, cpu_threads, mapped
from infoot_vit.infoot_helper.conditional import partial_projection
from infoot_vit.infoot_helper.dense_partial_projection import project_pairs, runtime_settings
from infoot_vit.infoot_helper.fit_mapping import fit_mapping
from infoot_vit.infoot_helper.mapping import FeatureMapper
from infoot_vit.infoot_helper.partial import distance


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("batch_size", [1, 3, 5])
def test_batched_projection_matches_serial_nonuniform_proposals(device, batch_size):
    rng = torch.Generator().manual_seed(109)
    source = torch.randn(batch_size, 4, 3, dtype=torch.float64, generator=rng).to(device)
    target = torch.randn(batch_size, 5, 3, dtype=torch.float64, generator=rng).to(device)
    query = torch.randn(6, 3, dtype=torch.float64, generator=rng).to(device)
    a = torch.tensor([.1, .2, .3, .4], dtype=torch.float64, device=device).expand(batch_size, -1)
    b = torch.tensor([.05, .15, .2, .25, .35], dtype=torch.float64, device=device).expand(batch_size, -1)
    retention = torch.tensor([0., .4, .6, 1.], dtype=torch.float64, device=device)
    plans = a[..., None] * b[:, None] * retention[None, :, None]
    logs = -.5 * distance(query.expand(batch_size, -1, -1), source).square()
    ky = (-.5 * distance(target, target).square()).exp()
    thresholds = torch.tensor([float("-inf"), -1., 1., -2., float("-inf")],
        dtype=torch.float64, device=device)[:batch_size]
    candidate, raw, valid, diag = project_pairs(logs, target, plans, a, b, ky, thresholds,
        torch.full((batch_size,), 1e-7, dtype=torch.float64, device=device))
    for i in range(batch_size):
        expected, g, detail = partial_projection(query, source[i], target[i], plans[i], a[i], b[i],
            1., 1., 1., dict(threshold=None if i == 0 else float(thresholds[i])),
            target_kernel=ky[i], query_logs=logs[i])
        torch.testing.assert_close(candidate[i], expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(raw[i], detail["raw_confidence"], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(raw[i] * valid[i], g, atol=1e-12, rtol=1e-12)
        assert torch.equal(valid[i], detail["support_valid"])
        for key in ("confidence_underflow", "score_log_retry"):
            assert torch.equal(diag[key][i], detail[key])
        assert candidate.device.type == diag["patch_entropy"].device.type == device


def test_tiny_score_retry_zero_retention_and_underflow_are_preserved():
    x = torch.tensor([[0.], [1.]], dtype=torch.float64)
    a = b = torch.full((2,), .5, dtype=torch.float64)
    plan = torch.tensor([[.4, 0.], [0., 0.]], dtype=torch.float64)
    logs = torch.tensor([[[-1e4, 0.]], [[-1e4, 0.]], [[0., -1e4]]], dtype=torch.float64)
    plans = torch.stack((plan, plan, torch.zeros_like(plan)))
    candidate, raw, valid, diag = project_pairs(logs, x.expand(3, -1, -1), plans,
        a.expand(3, -1), b.expand(3, -1), torch.eye(2, dtype=torch.float64).expand(3, -1, -1),
        torch.ones(3, dtype=torch.float64), torch.full((3,), 1e-7, dtype=torch.float64))
    assert torch.isfinite(candidate).all() and not raw.any() and not valid.any()
    assert diag["confidence_underflow"][:, 0].tolist() == [True, True, False]
    assert diag["score_log_retry"][:, 0].tolist() == [True, True, False]
    with pytest.raises(ValueError, match="underflow.*not genuine"):
        project_pairs(logs[:1], x[None], plans[:1], a[None], b[None],
            torch.eye(2, dtype=torch.float64)[None], torch.tensor([float("-inf")]), torch.tensor([1e-7]))


@pytest.fixture
def fitted(tmp_path, banks):
    cfg = config("grouped_partial")
    cfg.update(fit_pair_top_k=2, sampling=dict(images_per_domain=None, seed=42))
    return fit_mapping(cfg, root=tmp_path)


def serial_aggregate(mapper, patches, theta, alpha):
    """Independent copy of the original aggregation, including its diagnostics."""
    numerator, confidence = torch.zeros_like(patches), patches.new_zeros(len(patches))
    rejected, invalid, invalid_routes = (torch.zeros_like(confidence) for _ in range(3))
    budgets = patches.new_zeros(len(patches), len(alpha))
    effective = error = roundoff = 0.
    underflow = retry = 0
    geometry = {}
    for i, sid in enumerate(mapper.source.ids):
        logs = mapper._partial_query(patches, i)
        for tid in mapper.neighbors[sid]:
            j = mapper.target_index[tid]
            route = theta[i, j]
            if route == 0:
                continue
            candidate, _, diag, _ = mapper._partial_projection(patches, i, j, sid, tid, logs)
            raw = diag["raw_confidence"].clamp(0, 1)
            roundoff = max(roundoff, float((raw - diag["raw_confidence"]).abs().max()))
            g = raw * diag["support_valid"]
            confidence += route * g
            numerator += route * g[:, None] * candidate
            rejected += route * (1 - raw)
            invalid += route * raw * (~diag["support_valid"])
            invalid_routes += route * (~diag["support_valid"])
            budget = route * g * diag["weight_row_sums"] + route * (1 - g)
            budgets[:, j] += budget
            error = max(error, float((budget - route).abs().max()))
            effective += float((route * g * diag["patch_entropy"].exp()).sum())
            underflow += int(diag["confidence_underflow"].sum())
            retry += int(diag["score_log_retry"].sum())
            for name, value in dict(diag["plan_geometry"],
                    within_image_top1_patch_probability=diag["projected_patch_top1_probability"],
                    query_source_effective_patches=diag["query_source_effective_patches"]).items():
                geometry[name] = geometry.get(name, 0.) + (route * g * value).sum()
    positive = confidence > mapper.config["projection"]["confidence"]["mass_floor"]
    result = torch.zeros_like(patches)
    result[positive] = numerator[positive] / confidence[positive, None]
    denominator = float(confidence.sum())
    detail = dict(matched_rejected_accounting_error=error,
        matched_rejected_group_mass_residual=float((budgets - alpha[None]).abs().max()),
        ot_rejected_mass=rejected.tolist(), support_invalid_retained_mass=invalid.tolist(),
        support_invalid_route_mass=invalid_routes.tolist(), retention_roundoff_clipped_max=roundoff,
        within_image_effective_patches=effective / denominator if denominator else 0.,
        confidence_min=float(confidence.min()), confidence_mean=float(confidence.mean()),
        support_invalid_confidence_underflow_rows=underflow, score_log_retry_rows=retry)
    detail.update({key: float(value / denominator) if denominator else 0. for key, value in geometry.items()})
    return result, confidence, detail


@pytest.mark.parametrize("pair_batch_size", [1, 3, 32])
@pytest.mark.parametrize("support_gate", [False, True])
def test_complete_mapping_matches_serial_incomplete_blocks_and_masks(fitted, banks, pair_batch_size, support_gate):
    base = FeatureMapper.load(fitted)
    projection = deepcopy(base.config["projection"])
    projection["confidence"].update(all_invalid_policy="bypass")
    if support_gate:
        projection["confidence"].update(support_calibration="fixed", support_log_threshold=-1.)
    mapper = FeatureMapper.load(fitted, projection=projection, pair_projection_batch_size=pair_batch_size)
    actual = mapped(mapper, banks[2], return_metadata=True, chunk_size=1)
    for index, patches in enumerate(banks[2].features):
        theta = mapper.image.pair_weights(patches.flatten()[None])[0]
        theta = theta.masked_fill(~mapper.pair_mask, 0.) / theta[mapper.pair_mask].sum()
        expected, confidence, detail = serial_aggregate(mapper, patches, theta, theta.sum(0))
        torch.testing.assert_close(actual.mapped_features[index], expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual.match_confidence[index], confidence, atol=1e-12, rtol=1e-12)
        cf = mapper.config["projection"]["confidence"]
        assert torch.equal(actual.valid_mask[index], (confidence > cf["mass_floor"]) & (confidence >= cf["threshold"]))
        for key, value in detail.items():
            assert actual.diagnostics["queries"][index][key] == pytest.approx(value, abs=1e-12, rel=1e-12)
    again = mapped(mapper, banks[2], return_metadata=True, chunk_size=2)
    torch.testing.assert_close(again.mapped_features, actual.mapped_features, atol=0, rtol=0)
    assert again.diagnostics["queries"] == actual.diagnostics["queries"]


def test_cache_reuses_validated_plans_and_never_calls_serial_path(fitted, banks, monkeypatch):
    mapper = FeatureMapper.load(fitted, pair_cache_bytes=10**6)
    calls = []
    original = mapper._read_pair
    def read(*key):
        calls.append(key)
        return original(*key)
    monkeypatch.setattr(mapper, "_read_pair", read)
    monkeypatch.setattr(mapper, "_partial_projection", lambda *a: pytest.fail("Per-pair serial projection"))
    first = mapped(mapper, banks[2], return_metadata=True)
    second = mapped(mapper, banks[2], return_metadata=True)
    assert len(calls) == len(set(calls)) == len(mapper.pairs)
    report = second.diagnostics["performance"]
    assert report["pair_file_loads"] == len(mapper.pairs) and report["pair_cache_hits"] >= len(mapper.pairs)
    assert report["cache_resident_bytes"] <= report["pair_cache_bytes"]
    allocated = sum(value[name].untyped_storage().nbytes() for value, _, _ in mapper._dense_projector.cache.values()
        for name in ("plan", "a", "b"))
    assert allocated == report["cache_resident_bytes"]
    torch.testing.assert_close(first.mapped_features, second.mapped_features, atol=0, rtol=0)


@pytest.mark.parametrize("cache_bytes", [0, 128])
def test_disabled_and_bounded_cache_preserves_results(fitted, banks, cache_bytes):
    mapper = FeatureMapper.load(fitted, pair_cache_bytes=cache_bytes)
    first = mapped(mapper, banks[2], return_metadata=True)
    second = mapped(mapper, banks[2], return_metadata=True)
    stats = second.diagnostics["performance"]
    assert stats["cache_resident_bytes"] <= cache_bytes and stats["cache_peak_bytes"] <= cache_bytes
    assert stats["pair_file_loads"] > len(mapper.pairs)
    if cache_bytes == 0:
        assert stats["cached_pairs"] == stats["pair_cache_hits"] == 0
    torch.testing.assert_close(first.mapped_features, second.mapped_features, atol=0, rtol=0)


@pytest.mark.parametrize("change", ["corrupt", "delete"])
def test_cached_plan_changes_are_detected(fitted, banks, change):
    mapper = FeatureMapper.load(fitted)
    mapped(mapper, banks[2], return_metadata=True)
    entry = next(iter(mapper.pairs.values()))
    path = fitted / entry["file"]
    if change == "corrupt":
        with path.open("ab") as handle:
            handle.write(b"corrupt")
    else:
        path.unlink()
    with pytest.raises(ValueError, match="Missing/changed"):
        mapped(mapper, banks[2], return_metadata=True)


def test_runtime_settings_cli_and_saved_metadata(fitted, banks, tmp_path):
    from infoot_vit.infoot_test import main
    for cache, batch in ((-1, 1), (True, 1), (0, 0), (0, True)):
        with pytest.raises(ValueError):
            runtime_settings(cache, batch)
    output = tmp_path / "projected"
    assert main(["--mapping", str(fitted), "--query-bank", str(banks[2].path), "--device", "cpu",
        "--threads", "1", "--pair-cache-mib", "1", "--projection-pair-batch-size", "3",
        "--output-dir", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    perf = manifest["diagnostics"]["performance"]
    assert perf["pair_cache_bytes"] == 1024**2 and perf["pair_projection_batch_size"] == 3
    assert "dense_partial_projection.py" in manifest["projection_dependency_sha256"]


def test_performance_overrides_do_not_apply_to_other_modes(tmp_path, banks):
    directory = fit_mapping(config("whole_map"), root=tmp_path)
    with pytest.raises(ValueError, match="only to dense grouped_partial"):
        FeatureMapper.load(directory, pair_cache_bytes=0)
    with pytest.raises(ValueError, match="only to dense grouped_partial"):
        FeatureMapper.load(directory, pair_projection_batch_size=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_mapper_cache_device_and_serial_agreement(fitted, banks):
    cpu = FeatureMapper.load(fitted, device="cpu")
    gpu = FeatureMapper.load(fitted, device="cuda", pair_projection_batch_size=3)
    expected = mapped(cpu, banks[2], return_metadata=True)
    actual = mapped(gpu, banks[2], return_metadata=True)
    torch.testing.assert_close(actual.mapped_features, expected.mapped_features, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(actual.match_confidence, expected.match_confidence, atol=1e-11, rtol=1e-11)
    assert torch.equal(actual.valid_mask, expected.valid_mask)
    for value, _, _ in gpu._dense_projector.cache.values():
        assert value["plan"].dtype == torch.float32 and value["plan"].device.type == "cuda"
        assert value["a"].dtype == torch.float64 and value["a"].device.type == "cuda"
    first_loads = actual.diagnostics["performance"]["pair_file_loads"]
    again = mapped(gpu, banks[2], return_metadata=True)
    assert again.diagnostics["performance"]["pair_file_loads"] == first_loads
