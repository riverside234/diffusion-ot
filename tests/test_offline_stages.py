from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_ot.evaluation.full_infoot import (
    FullFitSettings, feasibility, fit_global, project_full, reference_rms, tiled_sinkhorn,
)
from diffusion_ot.evaluation.offline_artifacts import atomic_torch, bind_run, read_torch
from diffusion_ot.losses.infoot import (
    conditional_projection_weights, infoot_distance_scale, solve_infoot, uniform_marginals,
)


def fixture_features():
    generator = torch.Generator().manual_seed(51)
    return (torch.randn(7, 4, generator=generator, dtype=torch.float64),
            torch.randn(9, 4, generator=generator, dtype=torch.float64))


def settings():
    return FullFitSettings(mi_weight=.03, entropy_epsilon=.5, outer_iterations=50,
                           sinkhorn_iterations=200, absolute_tolerance=1e-10,
                           relative_tolerance=1e-8, outer_tolerance=1e-9,
                           block_size=3, checkpoint_every=2)


def test_rms_matches_complete_pairwise_formula_and_rejects_collapse():
    x, _ = fixture_features()
    assert reference_rms(x) == pytest.approx(float(infoot_distance_scale(x)), rel=1e-12)
    with pytest.raises(ValueError, match="collapsed"):
        reference_rms(torch.ones(8, 3))


def test_full_fit_dense_parity_resume_and_identity(tmp_path):
    x, y = fixture_features()
    s = settings()
    cost = torch.cdist(x, y) * .2
    scales = (reference_rms(x), reference_rms(y))
    plan, report = fit_global(x, y, cost, scales=scales, settings=s, output=tmp_path / "full", identity="test")
    assert report["converged"] and report["feasible"]
    dense = solve_infoot(x, y, bandwidth=s.bandwidth, distance_scale_x=scales[0], distance_scale_y=scales[1],
                         mi_weight=s.mi_weight, entropy_epsilon=s.entropy_epsilon,
                         inner_iterations=s.outer_iterations, projection_iterations=s.sinkhorn_iterations,
                         projection_tolerance=s.absolute_tolerance, cross_cost=cost,
                         outer_tolerance=s.outer_tolerance, min_inner_iterations=s.min_outer_iterations,
                         outer_patience=s.patience, strict_convergence=True, require_outer_convergence=True)
    torch.testing.assert_close(plan, dense.coupling, atol=1e-9, rtol=1e-8)
    partial, _ = fit_global(x, y, cost, scales=scales, settings=s, output=tmp_path / "resume",
                            identity="test", max_new_iterations=2)
    assert partial is None
    with pytest.raises(ValueError, match="fingerprint"):
        fit_global(x, y, cost, scales=scales, settings=s, output=tmp_path / "resume", identity="wrong", resume=True)
    resumed, _ = fit_global(x, y, cost, scales=scales, settings=s, output=tmp_path / "resume", identity="test", resume=True)
    torch.testing.assert_close(plan, resumed, atol=0, rtol=0)


def test_relative_feasibility_and_unconverged_export(tmp_path):
    a, b = uniform_marginals(5, 7, device="cpu", dtype=torch.float64)
    plan = a[:, None] * b
    plan[0] += 1e-5
    checks = feasibility(plan, a, b, replace(settings(), absolute_tolerance=1e-3))
    assert checks["row_absolute"] < 1e-3 and not checks["feasible"]
    x, y = fixture_features()
    plan, report = fit_global(x, y, torch.cdist(x, y), scales=(reference_rms(x), reference_rms(y)),
                              settings=replace(settings(), outer_iterations=1), output=tmp_path, identity="test")
    assert plan is None and not report["converged"]
    assert (tmp_path / "solver_progress.pt").exists()


@pytest.mark.parametrize("reverse", [False, True])
def test_projection_dense_parity_and_chunk_invariance(reverse):
    x, y = fixture_features()
    if reverse:
        x, y = y, x
    s = settings()
    a, b = uniform_marginals(len(x), len(y), device=x.device, dtype=x.dtype)
    plan, _, _ = tiled_sinkhorn(torch.cdist(x, y) * .1, a, b, s)
    query = x[:5] + .1
    raw = torch.arange(len(y) * 6, dtype=torch.float64).view(len(y), 6)
    sx, sy = reference_rms(x), reference_rms(y)
    expected = conditional_projection_weights(query, y, x, y, plan, bandwidth=.10,
                                               distance_scale_x=sx, distance_scale_y=sy) @ raw
    for qb, tb in ((1, 1), (2, 3), (20, 20)):
        actual = project_full(query, x, y, plan, raw, source_scale=sx, target_scale=sy,
                              query_batch_size=qb, target_block_size=tb)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_noise_gallery_identity_and_output_guard(tmp_path):
    from diffusion_ot.evaluation.final_eval import fixed_noise, gallery_ids
    ids = [f"image_{i}" for i in range(24)]
    all_noise = fixed_noise(ids, (4, 8, 8), 42, "cat_to_dog")
    pieces = torch.cat([fixed_noise(ids[:7], (4, 8, 8), 42, "cat_to_dog"),
                        fixed_noise(ids[7:], (4, 8, 8), 42, "cat_to_dog")])
    assert torch.equal(all_noise, pieces)
    assert gallery_ids(ids, 42, "cat_to_dog") == gallery_ids(ids[::-1], 42, "cat_to_dog")
    assert len(gallery_ids(ids, 42, "cat_to_dog")) == 16
    bind_run(tmp_path, "one", resume=False)
    bind_run(tmp_path, "one", resume=True)
    with pytest.raises(ValueError, match="mismatch"):
        bind_run(tmp_path, "two", resume=True)


def test_real_ssim_and_gallery_exclusion(tmp_path):
    pytest.importorskip("skimage")
    from diffusion_ot.evaluation.final_image_metrics import save_rgb, source_ssim, image_name, exact_image_files
    pixels = torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(2))
    source, other = tmp_path / image_name("source"), tmp_path / image_name("other")
    save_rgb(source, pixels)
    save_rgb(other, 1-pixels)
    assert source_ssim(source, source) == pytest.approx(1)
    assert source_ssim(source, other) < .5
    assert len(exact_image_files(tmp_path, ["source", "other"])) == 2
    save_rgb(tmp_path / "grid.png", pixels)
    with pytest.raises(ValueError, match="contaminated"):
        exact_image_files(tmp_path, ["source", "other"])


def test_fid_official_statistics_parity_and_cache(monkeypatch, tmp_path):
    fid = pytest.importorskip("cleanfid.fid")
    from diffusion_ot.evaluation.final_image_metrics import CleanFID
    paths = []
    for i in range(6):
        p = tmp_path / f"{i}.png"
        p.write_bytes(bytes([i]))
        paths.append(p)
    features = np.random.default_rng(3).normal(size=(6, 3))
    calls = []
    def extract(selected, **kwargs):
        calls.append(selected)
        return features[[int(Path(p).stem) for p in selected]]
    monkeypatch.setattr(fid, "get_files_features", extract)
    metric = CleanFID.__new__(CleanFID)
    metric.cache_dir, metric.versions, metric.weight_hash = tmp_path, {"test": "1"}, "fake"
    metric.model, metric.device, metric.batch_size, metric.num_workers = None, torch.device("cpu"), 2, 0
    expected = float(fid.fid_from_feats(features[:3], features[3:]))
    assert metric.compute(paths[:3], paths[3:]) == pytest.approx(expected)
    assert metric.compute(paths[:3], paths[3:]) == pytest.approx(expected)
    assert len(calls) == 2
    paths[0].write_bytes(b"changed")
    metric.compute(paths[:3], paths[3:])
    assert len(calls) == 3


def test_complete_dataset_detects_partial_latent_manifest(tmp_path):
    from diffusion_ot.evaluation.offline_pipeline import complete_dataset
    (tmp_path / "manifests").mkdir()
    (tmp_path / "latents/cat_train").mkdir(parents=True)
    (tmp_path / "train.yaml").write_text("data_config: data.yaml\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text("manifest_dir: manifests\nlatent_dir: latents\n", encoding="utf-8")
    (tmp_path / "manifests/cat_train.jsonl").write_text(
        '\n'.join(json.dumps({"sample_id": k, "domain": "cat"}) for k in ["a", "b", "c"]), encoding="utf-8")
    rows = []
    for key in ["a", "b"]:
        p = tmp_path / f"latents/cat_train/{key}.pt"
        torch.save(torch.zeros(4, 8, 8), p)
        rows.append({"sample_id": key, "domain": "cat", "latent_path": str(p)})
    (tmp_path / "latents/latent_manifest.jsonl").write_text('\n'.join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        complete_dataset({"stage1a": {"cat": {"config": "train.yaml"}}}, tmp_path, "cat", "train")


def test_model_fingerprints_exclude_mutable_loader_receipts(tmp_path):
    import yaml
    from diffusion_ot.evaluation.offline_pipeline import dependency_hashes
    (tmp_path / "model").mkdir()
    for name, config in {
        "train.yaml": {"model_config": "model.yaml", "data_config": "data.yaml"},
        "model.yaml": {"pretrained": "pretrained.yaml"},
        "pretrained.yaml": {"local_dir": "model", "metadata_file": "model/snapshot_report.json"},
        "data.yaml": {"manifest_dir": "manifests", "latent_dir": "latents"},
    }.items():
        (tmp_path / name).write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "manifests").mkdir()
    for d in ["cat", "dog"]:
        for s in ["train", "val"]:
            (tmp_path / f"manifests/{d}_{s}.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "stage1a.pt").write_bytes(b"state")
    (tmp_path / "model/weights.pt").write_bytes(b"weights")
    receipt = tmp_path / "model/snapshot_report.json"
    receipt.write_text("{}", encoding="utf-8")
    alignment = {"stage1a": {d: {"config": "train.yaml", "checkpoint": "stage1a.pt"} for d in ["cat", "dog"]}}
    original = dependency_hashes(alignment, tmp_path)
    receipt.write_text('{"downloaded": false}', encoding="utf-8")
    assert dependency_hashes(alignment, tmp_path) == original
    (tmp_path / "model/weights.pt").write_bytes(b"different")
    assert dependency_hashes(alignment, tmp_path) != original


@pytest.mark.parametrize("rms_mode,weights", [("full_bank", "ema"), ("reference_ema", "raw"), ("reference_ema", "ema")])
def test_toy_stage23_to_stage4_and_resume(monkeypatch, tmp_path, rms_mode, weights):
    """Exercise both orchestrators; replace expensive models/dataset/FID only."""
    import yaml
    import diffusion_ot.evaluation.offline_pipeline as pipeline
    import diffusion_ot.evaluation.final_eval as final
    import diffusion_ot.evaluation.stage1a_eval as sampling
    import diffusion_ot.data.afhq as afhq
    import diffusion_ot.data.ground_truth as ground_truth
    import diffusion_ot.losses.semantic_prior as prior_module
    from diffusion_ot.losses.projection_rms import checkpoint_projection_rms
    from diffusion_ot.models.patch_sampler import PatchSampleMLP, load_patch_projector
    from diffusion_ot.training.decoded_translation import self_supervised_translation_options

    class Dataset:
        def __init__(self, domain, split):
            self.records = []
            for i in range(6 if split == "train" else 18):
                key = f"{domain}_{split}_{i}"
                path = tmp_path / f"{key}.pt"
                latent = torch.rand(3, 16, 16, generator=torch.Generator().manual_seed(i + (100 if domain == "dog" else 0)))
                torch.save(latent, path)
                self.records.append({"sample_id": key, "domain": domain, "latent_path": str(path)})
        def __len__(self):
            return len(self.records)
        def __getitem__(self, index):
            r = self.records[index]
            return {"x0_latent": read_torch(r["latent_path"]), "sample_id": r["sample_id"], "metadata": r}
    datasets = {(d, s): Dataset(d, s) for d in ["cat", "dog"] for s in ["train", "val"]}
    monkeypatch.setattr(pipeline, "complete_dataset", lambda a, r, d, s: datasets[d, s])
    monkeypatch.setattr(final, "complete_dataset", pipeline.complete_dataset)
    monkeypatch.setattr(pipeline, "dependency_hashes", lambda *a: {})
    data_path = tmp_path / "data.yaml"
    data_path.write_text("image_size: 16\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "domain_data_path", lambda *a: data_path)
    monkeypatch.setattr(final, "domain_data_path", pipeline.domain_data_path)
    class Encoder(torch.nn.Module):
        spatial_feature_channels = (3,)
        def forward(self, x):
            return x.mean((2, 3))
    loads = []
    alignment = {"stage": "stage1b_fused_infoot"}
    checkpoint = {"stage": "stage1b_fused_infoot", "format_version": 4, "step": 10, "config": alignment}
    if rms_mode == "reference_ema":
        alignment.update({"projection_rms": {"mode": "reference_ema", "decay": .99, "eps": 1e-8},
                          "infoot": {"variant": "fused", "cross_cost_source": "encoder"},
                          "decoded_translation": {"supervision": "self_supervised", "objective": "patchnce",
                              "patchnce": {"sampler": "mlp_sample", "layers": [0], "projection_dim": 8}}})
        checkpoint["projection_rms_state"] = {
            family: {"version": 1, "decay": .99, "eps": 1e-8, "num_updates": 10,
                     "variances": values, "batch_variances": values}
            for family, values in (("raw", {"cat": .25, "dog": .36}), ("ema", {"cat": .49, "dog": .64}))}
        for seed, key in enumerate(("patch_projectors", "patch_projector_ema")):
            checkpoint[key] = {d: PatchSampleMLP((3,), 8, seed=seed).state_dict() for d in ("cat", "dog")}

    def context(*args, **kwargs):
        loads.append(args[2])
        branch = torch.nn.Module()
        branch.encoder = Encoder()
        paired = read_torch(kwargs["checkpoint_path"])
        tracker = checkpoint_projection_rms(paired, args[0], weights=kwargs["joint_weights"])
        if rms_mode == "reference_ema":
            assert tracker.state_dict() == checkpoint["projection_rms_state"][weights]
            head = load_patch_projector(branch.encoder, self_supervised_translation_options(alignment["decoded_translation"]),
                                        paired, args[2], weights=weights, device="cpu")
            expected_head = checkpoint["patch_projector_ema" if weights == "ema" else "patch_projectors"][args[2]]
            for key, value in head.state_dict().items():
                torch.testing.assert_close(value, expected_head[key], atol=0, rtol=0)
        return SimpleNamespace(branch=branch, vae=torch.nn.Identity(), matching_head=None, device="cpu",
                               dtype=torch.float32, transformer=None, training_config={})
    monkeypatch.setattr(pipeline.legacy, "_load_domain_context", context)
    fake_prior = SimpleNamespace(fingerprint="prior", lookup=lambda ids, *a: torch.arange(len(ids)*3).view(-1, 3).float(),
                                 cost=lambda a, b: torch.cdist(a, b) / 100)
    def load_prior(*args):
        assert rms_mode == "full_bank", "Self-supervised offline alignment must not load DINO"
        return fake_prior
    monkeypatch.setattr(prior_module, "load_semantic_prior", load_prior)
    torch.save(checkpoint, tmp_path / "joint.pt")
    (tmp_path / "alignment.yaml").write_text(yaml.safe_dump(alignment), encoding="utf-8")
    config = {"alignment_config": "alignment.yaml", "output_dir": "s23", "reference_split": "train",
              "reference_samples_per_domain": "all", "encoding_batch_size": 3, "solver_device": "cpu",
              "require_projection_rms_mode": rms_mode,
              "projection_bandwidth": .1, "fit": {"mi_weight": 0., "block_size": 3, "outer_iterations": 10}}
    (tmp_path / "s23.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    fit_scales_seen = []
    original_fit = pipeline.fit_global
    def checked_fit(x, y, cost, **kwargs):
        scales = (reference_rms(x), reference_rms(y))
        assert kwargs["scales"] == scales
        fit_scales_seen.append(scales)
        if rms_mode == "reference_ema":
            from diffusion_ot.losses.encoder_transport import encoder_transport_cost
            torch.testing.assert_close(cost, encoder_transport_cost(x, y))
        return original_fit(x, y, cost, **kwargs)
    monkeypatch.setattr(pipeline, "fit_global", checked_fit)
    result = pipeline.run_stage23(tmp_path / "s23.yaml", "joint.pt", root=tmp_path, device="cpu", weights=weights, max_new_iterations=2)
    assert result["status"] == "preflight_complete" and not (tmp_path / "s23/bundle.json").exists()
    result = pipeline.run_stage23(tmp_path / "s23.yaml", "joint.pt", root=tmp_path, device="cpu", weights=weights, resume=True)
    assert result["status"] == "complete"
    manifest, banks, plan = pipeline.load_bundle(tmp_path / "s23")
    assert plan.shape == (6, 6) and manifest["projection_bandwidth"] == .1
    assert fit_scales_seen == [tuple(manifest["fit_scales"].values())] * 2
    if rms_mode == "reference_ema":
        assert manifest["projection_rms"]["weights"] == weights
        assert manifest["projection_rms"]["num_updates"] == 10
        assert manifest["projection_scales"] == {d: value ** .5 for d, value in checkpoint["projection_rms_state"][weights]["variances"].items()}
        assert manifest["projection_scales"] != manifest["fit_scales"]
        assert manifest["prior_fingerprint"] is None
        exported = read_torch(tmp_path / "s23/models/paired_state.pt")
        assert exported["projection_rms_state"] == checkpoint["projection_rms_state"]
    else:
        assert manifest["projection_scales"] == manifest["fit_scales"]

    with monkeypatch.context() as scope:
        scope.setattr(pipeline, "reference_rms", lambda features: reference_rms(features) * (1 + 1e-14))
        pipeline.load_bundle(tmp_path / "s23")  # FP64 reduction roundoff across machines is harmless.
        scope.setattr(pipeline, "reference_rms", lambda features: reference_rms(features) * 1.01)
        with pytest.raises(ValueError, match="fit RMS"):
            pipeline.load_bundle(tmp_path / "s23")

    projection_calls = []
    def checked_projection(*args, **kwargs):
        direction = ("cat", "dog") if len(projection_calls) == 0 else ("dog", "cat")
        assert (kwargs["source_scale"], kwargs["target_scale"]) == tuple(manifest["projection_scales"][d] for d in direction)
        projection_calls.append(direction)
        return project_full(*args, **kwargs)
    monkeypatch.setattr(final, "project_full", checked_projection)
    monkeypatch.setattr(final, "metric_versions", lambda: {"test": "toy"})
    monkeypatch.setattr(afhq, "load_afhq_dataset", lambda *a: SimpleNamespace(_fingerprint="toy"))
    monkeypatch.setattr(ground_truth, "load_ground_truth_images", lambda path, records, **kw:
                        torch.stack([read_torch(r["latent_path"]) for r in records]))
    generation_calls = []
    def integrate(branch, transformer, noise, codes, **kwargs):
        generation_calls.append(len(codes))
        return (codes[:, :, None, None] + .05*noise).clamp(0, 1)
    monkeypatch.setattr(sampling, "integrate_pdae_flow", integrate)
    monkeypatch.setattr(sampling, "decode_vae_latents", lambda vae, x: x)
    class FakeFID:
        weight_hash = "test"
        def __init__(self, *a, **kw):
            pass
        def compute(self, fake, real):
            assert len(fake) == len(real) == 18
            return 1.25
    monkeypatch.setattr(final, "CleanFID", FakeFID)
    config4 = {"output_dir": "s4", "evaluation_split": "val", "source_samples_per_domain": "all",
               "require_projection_rms_mode": rms_mode,
               "real_samples_per_domain": "all", "metrics": ["fid", "source_ssim"], "seed": 1,
               "image_size": 16, "num_steps": 2, "guidance_scale": 1, "gallery_pairs_per_direction": 16,
               "encoding_batch_size": 3, "projection_batch_size": 4, "projection_target_block_size": 3,
               "generation_batch_size": 4, "fid_batch_size": 4, "fid_cache_dir": "fid", "num_workers": 0}
    (tmp_path / "s4.yaml").write_text(yaml.safe_dump(config4), encoding="utf-8")
    report = final.run_stage4(tmp_path / "s4.yaml", "s23", root=tmp_path, device="cpu")
    assert report["status"] == "complete" and sum(generation_calls) == 36
    assert len(projection_calls) == 2
    evaluation_protocol = json.loads((tmp_path / "s4/protocol.json").read_text())
    assert evaluation_protocol["projection_rms"] == manifest["projection_rms"]
    assert evaluation_protocol["projection_scales"] == manifest["projection_scales"]
    for direction in ["cat_to_dog", "dog_to_cat"]:
        assert report["metrics"][direction]["source_ssim"]["count"] == 18
        assert len(list((tmp_path / f"s4/galleries/{direction}").glob("*.png"))) == 49
        assert report["metrics"][direction]["gallery_pairs"] == 16
    loaded_before_resume = len(loads)
    final.run_stage4(tmp_path / "s4.yaml", "s23", root=tmp_path, device="cpu", resume=True)
    assert sum(generation_calls) == 36
    assert len(loads) == loaded_before_resume
    # Damaged images are regenerated, not silently included in metrics.
    damaged = next((tmp_path / "s4/images/cat_to_dog").glob("*.png"))
    damaged.write_bytes(b"damaged")
    final.run_stage4(tmp_path / "s4.yaml", "s23", root=tmp_path, device="cpu", resume=True)
    assert sum(generation_calls) == 37
    # No history is updated while encoding held-out queries or resuming.
    if rms_mode == "reference_ema":
        assert read_torch(tmp_path / "s23/models/paired_state.pt")["projection_rms_state"] == checkpoint["projection_rms_state"]
    # A recipe cannot silently override a bundle's calibration policy.
    config4["require_projection_rms_mode"] = "reference_ema" if rms_mode == "full_bank" else "full_bank"
    (tmp_path / "s4.yaml").write_text(yaml.safe_dump(config4), encoding="utf-8")
    with pytest.raises(ValueError, match="requires projection RMS"):
        final.run_stage4(tmp_path / "s4.yaml", "s23", root=tmp_path, device="cpu", resume=True)
    # Coupling tampering must be rejected even if its shape still matches.
    atomic_torch(tmp_path / "s23/transport.pt", {"coupling": plan * 2})
    with pytest.raises(ValueError, match="changed"):
        pipeline.load_bundle(tmp_path / "s23")
