"""Gaussian fidelity and acceptance tests, independent of low-rank algebra tests."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest
import torch

from test_infoot_vit_lowrank import tiny_config
from test_infoot_vit_mapping import banks, cpu_threads, mapped
from infoot_vit.lowrank import experiment, kernels, kernel_audit, solver
from infoot_vit.lowrank.config import canonical
from infoot_vit.lowrank.mapping import LowRankMapper
from infoot_vit.infoot_helper.feature_bank import digest, write_json


@pytest.mark.parametrize("method", [kernels.PRF, kernels.OPRF])
def test_full_scale_features_integrate_to_exact_gaussian(method):
    # Deterministic Gaussian quadrature, not a noisy Monte Carlo assertion.
    nodes, weights = np.polynomial.hermite.hermgauss(100)
    omega = torch.tensor(nodes, dtype=torch.float64)[None, :] * math.sqrt(2)
    x = torch.tensor([[-.45], [.15], [.65]], dtype=torch.float64)
    state = dict(method=method, mean=torch.zeros(1, dtype=x.dtype), sigma=1.,
                 omega=omega, rank=100, a=0. if method == kernels.PRF else kernels.oprf_coefficient(1, 1.2))
    f = kernels.features(x, state)
    expected_kernel = kernel_audit.gaussian(x, x, 1.)
    integrated = (f * torch.tensor(weights / math.sqrt(math.pi))) @ f.T * 100
    torch.testing.assert_close(integrated, expected_kernel, rtol=1e-10, atol=1e-10)
    # Finite-rank rows are deliberately not normalized to one.
    assert not torch.allclose(f.square().sum(1), torch.ones(3, dtype=x.dtype))


@pytest.mark.parametrize("method", [kernels.PRF, kernels.OPRF])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_features_preserve_raw_inputs_device_gradient_and_chunks(method, device):
    x = torch.randn(29, 5, dtype=torch.float64, device=device, generator=torch.Generator(device=device).manual_seed(97))
    original = x.clone()
    f, state = kernels.fit_features(x, 17, 1.2, 18, chunk_size=4, method=method, orthogonal=True)
    torch.testing.assert_close(x, original, atol=0, rtol=0)
    torch.testing.assert_close(f, kernels.features(x, state, chunk_size=11), atol=2e-14, rtol=2e-14)
    assert f.device == x.device and state["omega"].device == x.device and (f > 0).all()
    query = x[:3].clone().requires_grad_(True)
    gradient = torch.autograd.grad(kernels.features(query, state).square().sum(), query)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    # Orthogonality within each block, including the incomplete final block.
    for block in state["omega"].split(5, dim=1):
        unit = block / block.norm(dim=0)
        torch.testing.assert_close(unit.T @ unit, torch.eye(block.shape[1], device=device, dtype=x.dtype), atol=1e-12, rtol=0)
    report = kernels.error_report(x, f, state, seed=24, count=100, density_queries=3, chunk_size=7)
    disk_report = kernels.error_report(x, f, state, seed=24, count=100, density_queries=3, chunk_size=7, storage_roundtrip=True)
    assert disk_report["relative_rmse"] == pytest.approx(report["relative_rmse"], rel=1e-6, abs=1e-7)


def test_oprf_fidelity_against_gaussian_on_fixed_synthetic_features():
    x = torch.randn(80, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(21))
    f, state = kernels.fit_features(x, 2048, 1.5, 22, method=kernels.OPRF, orthogonal=True)
    check = kernels.error_report(x, f, state, seed=23, count=512, density_queries=8)
    assert check["relative_rmse"] < .15
    assert check["density_relative_error_max"] < .15


def test_exact_reference_detects_mi_gradient_and_grouped_mapping_error():
    x = torch.tensor([[-1.], [-.2], [.3], [1.]], dtype=torch.float64)
    y = x.flip(0) + .1
    plan = .8 * torch.eye(4, dtype=x.dtype) / 4 + .2 / 16
    cost = torch.cdist(x, y)
    kx, ky = kernel_audit.gaussian(x, x, .7), kernel_audit.gaussian(y, y, .9)
    kq = kernel_audit.gaussian(x[:2] + .05, x, .7)
    args = (plan, cost, kx, ky, kq)
    options = dict(target=y, target_group=torch.tensor([0, 0, 1, 1]), lam=.3, reg=.05)
    same = kernel_audit.compare(*args, kx, ky, kq, **options)
    for name in ("mi_absolute_error", "mi_gradient_relative_error", "objective_gradient_relative_error", "mean_within_image_weight_tv", "mapped_feature_relative_error"):
        assert same[name] == pytest.approx(0., abs=1e-14)
    assert same["mi_gradient_cosine"] == pytest.approx(1.)
    different = kernel_audit.compare(*args, torch.ones_like(kx), torch.ones_like(ky), torch.ones_like(kq), **options)
    assert different["mi_absolute_error"] > .01
    assert different["mi_gradient_relative_error"] > .01
    assert different["mean_within_image_weight_tv"] > .01
    assert different["mapped_feature_relative_error"] > .01
    assert different["exact"]["cost"] == different["approximate"]["cost"]
    assert different["exact"]["entropy"] == different["approximate"]["entropy"]


def test_acceptance_checks_both_densities_and_storage_precision():
    c = canonical(tiny_config())["kernel"]
    ok = dict(relative_rmse=.1, density_relative_error_mean=.1, density_relative_error_max=.2)
    checks = {name: dict(ok, float32_roundtrip=dict(ok)) for name in ("source", "target")}
    assert kernel_audit.acceptance(checks, c)["accepted"]
    checks["target"]["float32_roundtrip"]["density_relative_error_max"] = 1.1
    result = kernel_audit.acceptance(checks, c)
    assert not result["accepted"] and len(result["failures"]) == 1
    assert result["failures"][0]["precision"] == "float32_roundtrip"
    checks["source"]["relative_rmse"] = float("nan")
    assert len(kernel_audit.acceptance(checks, c)["failures"]) == 2


def audit_config():
    c = tiny_config()
    c["kernel_rank"] = 256
    c["kernel"].update(method=kernels.OPRF, orthogonal=True, h=1.5, error_policy="error",
        max_relative_rmse=.5, max_density_relative_error_mean=.25, max_density_relative_error_max=1.,
        reference_images=1, reference_patches_per_image=2, reference_queries=2)
    return c


def test_failed_acceptance_persists_diagnostics_before_any_transport(tmp_path, banks, monkeypatch):
    c = audit_config()
    c["kernel_rank"] = 1
    c["kernel"]["max_relative_rmse"] = 1e-12
    monkeypatch.setattr(experiment.BalancedModel, "fit", lambda *a, **kw: pytest.fail("Router started before kernel acceptance"))
    monkeypatch.setattr(experiment, "fit_unit", lambda *a, **kw: pytest.fail("Patch fit started before kernel acceptance"))
    with pytest.raises(RuntimeError, match="Kernel accuracy acceptance failed"):
        experiment.fit(c, root=tmp_path)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "failed" and not manifest["files"]
    quality = json.loads((directory / "kernel_quality.json").read_text())
    assert not quality["accepted"]
    for name in ("kernel_quality.json", "kernel_approximation.json", "kernel_reference.json"):
        assert (directory / name).is_file() and list((directory / "logs").glob("*/" + name))
    assert not (directory / "factors.pt").exists()
    with pytest.raises(RuntimeError, match="Kernel accuracy acceptance failed"):
        experiment.fit(c, root=tmp_path, resume=directory)
    assert len(list((directory / "logs").glob("*/kernel_quality.json"))) == 2


def test_kernel_only_resume_storage_and_grouped_projection(tmp_path, banks, monkeypatch):
    c = audit_config()
    with monkeypatch.context() as patch:
        patch.setattr(experiment.BalancedModel, "fit", lambda *a, **kw: pytest.fail("Kernel-only command started a router"))
        directory = experiment.fit(c, root=tmp_path, kernel_check_only=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "kernel_checked" and set(manifest["files"]) == {"kernels"}
    reference = json.loads((directory / "kernel_reference.json").read_text())
    assert reference["status"] == "completed"
    assert reference["plan_row_residual"] < 1e-14 and reference["plan_column_residual"] < 1e-14
    assert not set(reference["source_flat_indices"]) & set(reference["query_flat_indices"])
    original_bytes = (directory / "kernels.pt").read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        LowRankMapper.load(directory)
    assert experiment.fit(c, root=tmp_path, resume=directory) == directory
    assert (directory / "kernels.pt").read_bytes() == original_bytes
    assert not list(directory.rglob("latest.pt"))
    state = torch.load(directory / "kernels.pt", weights_only=True)
    assert state["fx"].dtype == state["fy"].dtype == torch.float32
    assert state["source"]["omega"].dtype == torch.float64
    bad = deepcopy(state); bad["source"]["a"] += .1
    with pytest.raises(ValueError, match="scale compensation"):
        experiment.validate_kernels(bad, (8, 8, 256), 3, canonical(c)["kernel"])
    bad = deepcopy(state); bad["approximation"]["target"]["relative_rmse"] = 99.
    with pytest.raises(ValueError, match="accuracy limits"):
        experiment.validate_kernels(bad, (8, 8, 256), 3, canonical(c)["kernel"])
    monkeypatch.setattr(kernels, "fit_features", lambda *a, **kw: pytest.fail("Inference kernel refit"))
    monkeypatch.setattr(solver, "solve", lambda *a, **kw: pytest.fail("Inference OT refit"))
    mapper = LowRankMapper.load(directory)
    result = mapped(mapper, banks[2], chunk_size=1)
    mapper.config["projection"].update(target_chunk_size=1, patch_chunk_size=1)
    torch.testing.assert_close(result, mapped(mapper, banks[2], chunk_size=2), atol=2e-12, rtol=0)
    # Full saved-factor projection versus a small dense expansion, independent
    # of the exact-Gaussian comparison above.
    factors = torch.load(directory / "factors.pt", weights_only=True)
    q, r, g = [factors[k].double() for k in ("q", "r", "g")]
    fx, fy = state["fx"].double(), state["fy"].double()
    smoothing = (q / g) @ r.T @ (fy @ fy.T) / (fy @ fy.T).mean(0)
    for i, image in enumerate(banks[2].features):
        scores = kernels.features(image, state["source"]) @ fx.T @ smoothing
        alpha = mapper.image.conditional_weights(image.reshape(1, -1))[0]
        expected = sum(alpha[j] * (scores[:, j*4:(j+1)*4] / scores[:, j*4:(j+1)*4].sum(1, keepdim=True)) @ mapper.y[j] for j in range(len(mapper.y)))
        torch.testing.assert_close(result[i], expected, atol=2e-12, rtol=0)


def test_legacy_artifact_without_new_kernel_config_still_loads(tmp_path, banks):
    directory = experiment.fit(tiny_config(), root=tmp_path)
    before = mapped(LowRankMapper.load(directory), banks[2])
    manifest = json.loads((directory / "manifest.json").read_text())
    old_keys = {"h", "seed", "check_pairs", "density_queries", "error_warn_relative_rmse"}
    manifest["config"]["kernel"] = {k: v for k, v in manifest["config"]["kernel"].items() if k in old_keys}
    manifest["artifact_id"] = digest({k: v for k, v in manifest.items() if k != "artifact_id"})
    write_json(directory / "manifest.json", manifest)
    torch.testing.assert_close(before, mapped(LowRankMapper.load(directory), banks[2]), rtol=0, atol=0)


def test_cli_overrides_and_partial_scope(tmp_path, monkeypatch):
    import yaml
    from infoot_vit import infoot_fit_lowrank as cli
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(audit_config()))
    captured = {}
    def fake_fit(c, **kwargs):
        captured.update(config=c, options=kwargs)
        return tmp_path
    monkeypatch.setattr(cli, "fit", fake_fit)
    cli.main(["--config", str(path), "--kernel-check-only", "--kernel-method", kernels.LEGACY, "--kernel-rank", "512"])
    assert captured["options"]["kernel_check_only"]
    assert captured["config"]["kernel_rank"] == 512
    assert not captured["config"]["kernel"]["orthogonal"]
    with pytest.raises(ValueError, match="only to grouped_patch_lowrank"):
        canonical(dict(audit_config(), mode="grouped_partial_lowrank"))
