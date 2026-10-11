from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infoot_vit import diagnose_blur
from infoot_vit.infoot_helper.conditioning_spread import (
    calibrate_native_spread, conditioning_variants, restore_token_spread,
    scale_token_deviations, token_moments,
)
from infoot_vit.infoot_helper.feature_bank import BankWriter, FeatureBank, digest, file_hash, save_tensor, write_json


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def native_example(dtype=torch.float64):
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(3, 12, 8, generator=generator, dtype=dtype)
    x += torch.arange(8, dtype=dtype)[None, None] * .7
    mask = torch.ones(3, 12, dtype=torch.bool)
    mask[0, :3] = False
    mask[1, 4:8] = False
    return x, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_restoration_preserves_mean_mask_dtype_and_recovers_known_contraction(dtype):
    native, mask = native_example(dtype)
    # Normalize native spreads so this exact recovery test has one population target.
    variance = token_moments(native, mask)[1]
    native = scale_token_deviations(native, mask, variance.rsqrt())
    contracted = scale_token_deviations(native, mask, .625)
    original = contracted.clone()
    corrected, report = restore_token_spread(contracted, mask, 1.)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(corrected, native, atol=tolerance, rtol=tolerance)
    assert torch.equal(contracted, original)
    assert torch.equal(corrected[~mask], original[~mask])
    assert corrected.dtype == dtype and not corrected.requires_grad
    assert report["gains"] == pytest.approx([1.6] * 3, rel=tolerance)
    before_mean = token_moments(contracted, mask)[0]
    after_mean, after_var = token_moments(corrected, mask)
    torch.testing.assert_close(after_mean, before_mean, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(after_var, torch.ones(3, dtype=torch.float64), atol=tolerance, rtol=tolerance)


def test_cap_no_shrinking_and_query_order_chunk_independence():
    native, mask = native_example()
    native = scale_token_deviations(native, mask, token_moments(native, mask)[1].rsqrt())
    inputs = scale_token_deviations(native, mask, torch.tensor([.1, .7, 1.2]))
    together, report = restore_token_spread(inputs, mask, 1.)
    assert report["gains"] == pytest.approx([2., 1 / .7, 1.], rel=1e-7)
    assert report["capped"] == [True, False, False]
    assert report["variance_after"] == pytest.approx([.04, 1., 1.44], rel=1e-7)
    separate = torch.cat([restore_token_spread(x[None], m[None], 1.)[0] for x, m in zip(inputs, mask)])
    torch.testing.assert_close(together, separate, atol=0, rtol=0)
    order = torch.tensor([2, 0, 1])
    permuted = restore_token_spread(inputs[order], mask[order], 1.)[0]
    torch.testing.assert_close(permuted, together[order], atol=0, rtol=0)
    torch.testing.assert_close(together[2], inputs[2], atol=1e-15, rtol=1e-15)


def test_centered_correction_survives_decoder_layernorm_while_norm_scaling_does_not():
    native, mask = native_example()
    norm = torch.nn.LayerNorm(8, eps=1e-6, elementwise_affine=False).double()
    contracted = scale_token_deviations(native, mask, .5)
    target = float(token_moments(native, mask)[1].mean())
    corrected, _ = restore_token_spread(contracted, mask, target)
    # Ordinary norm inflation is almost removed by per-token LayerNorm.
    torch.testing.assert_close(norm(contracted * 2), norm(contracted), atol=2e-6, rtol=2e-6)
    baseline_error = (norm(contracted)[mask] - norm(native)[mask]).square().mean()
    corrected_error = (norm(corrected)[mask] - norm(native)[mask]).square().mean()
    assert corrected_error < baseline_error * .02


def test_calibration_is_population_mean_not_pooled_image_mean_variation():
    native, _ = native_example()
    calibration = calibrate_native_spread(native, chunk_size=1)
    expected = native.var(1, unbiased=False).mean()
    assert calibration["target_within_image_variance"] == pytest.approx(float(expected))
    translated = native + torch.tensor([0., 100., -20.])[:, None, None]
    assert calibrate_native_spread(translated)["target_within_image_variance"] == pytest.approx(float(expected))


@pytest.mark.parametrize("failure", ["empty", "one", "constant", "nan", "bad_mask", "bad_target"])
def test_invalid_or_collapsed_conditions_fail_explicitly(failure):
    features, mask = native_example()
    target = 1.
    if failure == "empty":
        mask[0] = False
    elif failure == "one":
        mask[0] = False
        mask[0, 0] = True
    elif failure == "constant":
        features[0] = 1
    elif failure == "nan":
        features[0, 0, 0] = float("nan")
    elif failure == "bad_mask":
        mask = mask.float()
    else:
        target = float("inf")
    with pytest.raises(ValueError):
        restore_token_spread(features, mask, target)


def test_native_counterfactual_controls_only_spread_and_mask():
    native, mask = native_example()
    mapped = scale_token_deviations(native, mask, .65)
    variants, report = conditioning_variants(mapped, mask, native, 1.)
    assert variants["native_reference_full"][1].all()
    for name in set(variants) - {"native_reference_full"}:
        assert torch.equal(variants[name][1], mask)
    torch.testing.assert_close(variants["native_reference_contracted"][0], mapped)
    assert report["native_contraction_gains"] == pytest.approx([.65] * 3)
    assert report["native_contraction_reached_mapped_variance"] == [True] * 3


@pytest.fixture
def saved_experiment(tmp_path):
    representation = dict(grid=[2, 2], dim=8, patch_order="row_major_no_special_tokens",
        encoder={"revision": "synthetic"}, layer="synthetic", preprocessing={}, normalization="none")
    generator = torch.Generator().manual_seed(501)
    banks = []
    for name, domain, split, count in [("target", "dog", "train", 8), ("query", "cat", "val", 2)]:
        writer = BankWriter(tmp_path / name, domain=domain, split=split, representation=representation, provenance={})
        writer.append(torch.randn(count, 4, 8, generator=generator),
            [dict(sample_id=f"{domain}_{split}_{i}", domain=domain) for i in range(count)])
        writer.finish()
        banks.append(FeatureBank.load(tmp_path / name))
    target, query = banks
    fit = dict(schema="siglip_infoot_mapping_v3", status="complete", source_domain="cat", target_domain="dog",
        config=dict(mode="grouped_partial", fit_pair_top_k=8, solver=dict(h=.36)),
        target_bank=dict(path="../target", artifact_id=target.artifact_id),
        source_ids=["cat_train_0"], target_ids=target.ids, representation=representation)
    fit["artifact_id"] = digest(fit)
    write_json(tmp_path / "fit/manifest.json", fit)
    valid = torch.tensor([[True, True, True, False], [True, False, True, True]])
    # Repeated top-1 dog reference still gets distinct per-query starting noise.
    features = scale_token_deviations(target.features[[0, 0]], valid, .6)
    state = dict(mapped_features=features, valid_mask=valid, match_confidence=valid.float() * .8,
                 ids=query.ids, records=query.records)
    save_tensor(tmp_path / "mapped/mapped.pt", state)
    projection = dict(bandwidth_multiplier=.20 / .36, patch_bandwidth=.20,
                      confidence=dict(all_invalid_policy="error", threshold=.05))
    manifest = dict(schema="infoot_mapped_features_v2", mapper_id=fit["artifact_id"], ids=query.ids,
        query_bank_id=query.artifact_id, query_domain="cat", representation=representation,
        projection=projection, output=dict(file="mapped.pt", sha256=file_hash(tmp_path / "mapped/mapped.pt")),
        diagnostics=dict(queries=[dict(query_id=sid, image_weights=[1.] + [0.] * 7) for sid in query.ids]))
    manifest["artifact_id"] = digest(manifest)
    write_json(tmp_path / "mapped/manifest.json", manifest)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    generation = dict(mapper_id=fit["artifact_id"], query_ids=query.ids,
        target_reference_ids=[target.ids[0]] * 2, projection=projection, confidence_policy=projection["confidence"],
        target_domain="dog", checkpoint_sha256=file_hash(checkpoint), weights="raw", num_steps=40,
        guidance_scale=1.5, solver="euler", seed=20260903)
    generation["per_image_noise_seeds"] = [int.from_bytes(
        hashlib.sha256(f"20260903:{sid}".encode()).digest()[:8], "big") % (2**63 - 1) for sid in query.ids]
    write_json(tmp_path / "mapped/generation_report.json", generation)
    return fit, manifest, generation, query, target, state


@pytest.mark.parametrize("change", ["checkpoint", "noise", "order", "reference", "projection", "bank", "failed_snapshot"])
def test_comparison_rejects_unmatched_controls(saved_experiment, tmp_path, change):
    fit, manifest, generation, query, target, state = deepcopy(saved_experiment)
    if change == "checkpoint":
        args = SimpleNamespace(output_dir=tmp_path / "new", mapped=tmp_path / "mapped", checkpoint=tmp_path / "checkpoint.pt")
        args.checkpoint.write_bytes(b"changed")
        with pytest.raises(ValueError, match="Checkpoint hash"):
            diagnose_blur.run_comparison(args)
        assert not args.output_dir.exists()
        return
    if change == "noise":
        generation["per_image_noise_seeds"][0] += 1
    elif change == "order":
        generation["query_ids"] = list(reversed(generation["query_ids"]))
    elif change == "reference":
        generation["target_reference_ids"][0] = target.ids[-1]
    elif change == "projection":
        manifest["projection"]["patch_bandwidth"] = .15
    elif change == "bank":
        target.manifest["split"] = "val"
    else:
        generation["incomplete_fit"] = {"failed_pair_count": 425}
    with pytest.raises(ValueError):
        diagnose_blur.validate_comparison(fit, manifest, generation, query, target, state)


def test_full_runner_uses_original_cache_frozen_weights_equal_noise_and_saves_separate_output(
        saved_experiment, tmp_path, monkeypatch):
    from diffusion_ot.evaluation import stage1a_eval
    from diffusion_ot.data import ground_truth
    fit, manifest, generation, query, target, state = saved_experiment
    before_hash = file_hash(tmp_path / "mapped/mapped.pt")
    branch = torch.nn.Module()
    branch.encoder = SimpleNamespace(snapshot_identity=target.representation["encoder"],
                                    architecture_spec=dict(features=target.representation["layer"]))
    branch.semantic_transformer = torch.nn.Module()
    branch.semantic_transformer.feature_projector = torch.nn.Sequential(
        torch.nn.LayerNorm(8), torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 8))
    evaluator = SimpleNamespace(branch=branch, transformer=torch.nn.Linear(2, 2), vae=torch.nn.Identity(),
        device="cpu", model_dtype=torch.float32, checkpoint_path=tmp_path / "checkpoint.pt")
    loaded, sampled = [], []
    def load(*args, **kwargs):
        loaded.append(kwargs)
        return evaluator
    def integrate(branch, transformer, noise, features, **kwargs):
        sampled.append((noise.clone(), features.clone(), kwargs))
        noise.add_(123)  # Even a mutating sampler must not contaminate other rows.
        return noise
    monkeypatch.setattr(stage1a_eval, "load_stage1a_evaluator", load)
    monkeypatch.setattr(stage1a_eval, "integrate_pdae_flow", integrate)
    monkeypatch.setattr(stage1a_eval, "decode_vae_latents", lambda vae, z, **kwargs: torch.zeros(len(z), 3, 8, 8))
    monkeypatch.setattr(ground_truth, "load_ground_truth_images", lambda path, records: torch.ones(len(records), 3, 8, 8))
    train = tmp_path / "dog.yaml"
    train.write_text("domain: dog\ndata_config: unused.yaml\nclass_conditioning:\n  null_label: 1000\n")
    evaluation = tmp_path / "eval.yaml"
    evaluation.write_text("{}\n")
    args = SimpleNamespace(mapping=tmp_path / "fit", mapped=tmp_path / "mapped", target_bank=None,
        query_bank=query.path, checkpoint=evaluator.checkpoint_path, output_dir=tmp_path / "comparison",
        allow_failed_pairs=False, train_config=train, eval_config=evaluation, device="cpu", batch_size=1)
    report = diagnose_blur.run_comparison(args)
    assert report["status"] == "complete" and len(sampled) == 5 and len(loaded) == 1
    assert loaded[0]["weights"] == "raw" and report["generation"] == {
        key: generation[key] for key in report["generation"]}
    assert not torch.equal(sampled[0][0][0], sampled[0][0][1])
    for noise, features, options in sampled:
        assert torch.equal(noise, sampled[0][0])
        assert options["num_steps"] == 40 and options["guidance_scale"] == 1.5 and options["solver"] == "euler"
    assert not sampled[0][2]["condition_padding_mask"].any()
    for _, _, options in sampled[1:]:
        assert torch.equal(options["condition_padding_mask"], ~state["valid_mask"])
    torch.testing.assert_close(sampled[3][1], state["mapped_features"], atol=0, rtol=0)
    assert all(not module.training for module in (evaluator.branch, evaluator.transformer, evaluator.vae))
    assert all(not p.requires_grad for module in (evaluator.branch, evaluator.transformer) for p in module.parameters())
    assert file_hash(tmp_path / "mapped/mapped.pt") == before_hash
    saved = torch.load(args.output_dir / "conditioning_variants.pt", weights_only=True)
    assert torch.equal(saved["initial_noise"], sampled[0][0])
    assert torch.equal(saved["match_confidence"], state["match_confidence"])
    assert torch.equal(saved["variants"]["mapped_spread_restored"]["valid_mask"], state["valid_mask"])
    assert report["calibration"]["selected_ids"] == fit["target_ids"]
    assert (args.output_dir / "comparison_grid.png").is_file()
    disk_report = json.loads((args.output_dir / "blur_report.json").read_text())
    assert disk_report["status"] == "complete" and len(disk_report["row_order"]) == 7
    assert "after_decoder_layernorm" in disk_report["conditioning_diagnostics"]["mapped_spread_restored"]
    with pytest.raises(FileExistsError):
        diagnose_blur.run_comparison(args)
