from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from diffusion_ot.evaluation.checkpoint_cache import CheckpointEvaluationCache, cached
from diffusion_ot.evaluation.checkpoint_screen import (
    bandwidth_comparability, report_rows, screen_variants, summarize_reports,
)
from diffusion_ot.evaluation import stage1b_eval as evaluation


def config():
    return {"seed": 31, "data": {"query_samples_per_domain": 256},
            "matching": {"bandwidth_multiplier": .55, "projection_bandwidth_multiplier": .25},
            "translation": {"temperature": 1.}, "infoot": {"entropy_epsilon": .02}}


def test_screen_preserves_cohort_and_fit_but_separates_private_noise_draws():
    original = config()
    variants = screen_variants(original)
    assert len(variants) == 12
    assert {v["axes"]["bandwidth"] for v in variants} == {.15, .2, .25, .35}
    assert {v["config"]["seed"] for v in variants} == {31}
    assert {v["config"]["translation"]["seed"] for v in variants} == {31, 100031, 200031}
    assert original == config()
    for variant in variants:
        cfg = variant["config"]
        assert cfg["matching"]["bandwidth_multiplier"] == .55
        assert cfg["infoot"] == original["infoot"]
        assert cfg["translation"]["temperature"] == 1.
        assert cfg["translation"]["readouts"] == ["conditional_mean", "conditional_map", "conditional_sample"]
        assert cfg["projection_audit"]["enabled"] and not cfg["visualization"]["enabled"]
        assert cfg["translation"]["save_generation_inputs"]


def test_finalists_only_run_explicit_pairs_at_20_40_with_64_images():
    variants = screen_variants(config(), finalists=[(.25, "conditional_mean"), (.2, "conditional_map")])
    assert len(variants) == 12
    assert {v["axes"]["steps"] for v in variants} == {20, 40}
    assert all(v["axes"]["samples"] == 64 for v in variants)
    assert all(v["axes"]["readouts"] == (["conditional_mean"] if v["axes"]["bandwidth"] == .25 else ["conditional_map"]) for v in variants)
    robustness = screen_variants(config(), bandwidths=[.25], draws=1, bank_seeds=[41, 51])
    assert {v["config"]["data"]["bank_seed"] for v in robustness} == {41, 51}
    assert all(v["config"]["seed"] == v["config"]["translation"]["seed"] == 31 for v in robustness)


@pytest.mark.parametrize("options", [dict(draws=0), dict(bandwidths=[0.]), dict(bandwidths=[float("nan")]),
    dict(bandwidths=[.25, .25]), dict(finalists=[(.25, "unknown")]), dict(samples=257), dict(steps=[0])])
def test_invalid_screen_rejected_before_loading_models(options):
    with pytest.raises(ValueError):
        screen_variants(config(), **options)


def test_cache_reuses_factories_and_drops_stale_checkpoint_or_weight_state():
    cache = CheckpointEvaluationCache()
    cache.bind({"checkpoint": "a", "weights": "ema"})
    value = cached(cache, "fit", {"fit": .55}, object)
    assert cached(cache, "fit", {"fit": .55}, object) is value
    assert cached(cache, "fit", {"fit": .6}, object) is not value
    cache.bind({"checkpoint": "a", "weights": "raw"})
    assert cached(cache, "fit", {"fit": .55}, object) is not value
    cache.bind({"checkpoint": "b", "weights": "raw"})
    assert not cache.values


def test_cached_banks_encode_once_and_invalidate_selection_and_representation():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, x):
            self.calls += 1
            return x.flatten(1)
    encoder = Encoder()
    data = [dict(x0_latent=torch.arange(4.).reshape(1, 2, 2) + i, sample_id=str(i), metadata={}) for i in range(12)]
    cache = CheckpointEvaluationCache()
    cache.bind("checkpoint1")
    options = dict(domain="cat", split="train", count=6, seed=7, batch_size=3, device="cpu",
                   dtype=torch.float32, checkpoint_id="a", cache=cache)
    first = evaluation.build_latent_bank(encoder, data, **options)
    assert evaluation.build_latent_bank(encoder, data, **options) is first
    assert encoder.calls == 2
    changed = evaluation.build_latent_bank(encoder, data, **{**options, "seed": 8})
    assert encoder.calls == 4 and first.sample_ids != changed.sample_ids
    evaluation.build_latent_bank(encoder, data, **{**options, "checkpoint_id": "b"})
    assert encoder.calls == 6


def test_readouts_use_actual_corrected_weights_and_do_not_consume_global_rng():
    weights = torch.tensor([[.8, .1, .1], [.1, .2, .7]])
    targets = torch.tensor([[1., 0.], [0., 1.], [2., 3.]])
    state = torch.get_rng_state().clone()
    codes, indices = evaluation.translation_readout_codes(weights, targets,
        ["conditional_mean", "conditional_map", "conditional_sample"], temperature=1., seed=7)
    torch.testing.assert_close(codes["conditional_mean"], weights @ targets)
    torch.testing.assert_close(codes["conditional_map"], targets[[0, 2]])
    expected = torch.multinomial(weights, 1, generator=torch.Generator().manual_seed(8)).squeeze(1)
    torch.testing.assert_close(indices["conditional_sample"], expected)
    again, _ = evaluation.translation_readout_codes(weights, targets, ["conditional_sample"], temperature=1., seed=7)
    torch.testing.assert_close(again["conditional_sample"], codes["conditional_sample"])
    assert torch.equal(state, torch.get_rng_state())


def test_saved_noise_codes_and_ids_match_every_readout_and_finalist_integration(tmp_path, monkeypatch):
    import diffusion_ot.evaluation.stage1a_eval as stage1a
    import torchvision.utils
    from test_stage1b_eval import _bank
    query = _bank("cat", "val", ["q0", "q1"])
    gallery = _bank("dog", "train", ["r0", "r1", "r2"])
    context = SimpleNamespace(device="cpu", dtype=torch.float32, vae=None, branch=None, transformer=None, training_config={})
    monkeypatch.setattr(evaluation, "_load_latents_from_bank", lambda bank, count: torch.zeros(count, 3, 2, 2))
    monkeypatch.setattr(stage1a, "decode_vae_latents", lambda vae, x: x)
    monkeypatch.setattr(torchvision.utils, "save_image", lambda *a, **kw: None)
    generated = []
    def integrate(branch, transformer, noise, codes, **kw):
        assert not torch.is_grad_enabled()
        generated.append((noise.clone(), codes.clone(), kw["num_steps"]))
        return noise
    monkeypatch.setattr(stage1a, "integrate_pdae_flow", integrate)
    for steps in (20, 40):
        path = tmp_path / f"grid_{steps}.png"
        metrics = evaluation._save_translation_grid(context, context, query, gallery,
            torch.tensor([[.8, .1, .1], [.1, .2, .7]]), count=2, num_steps=steps,
            guidance_scale=1., temperature=1., seed=7, output_path=path,
            include_source=False, save_generation_inputs=True)
        payload = torch.load(metrics["generation_inputs"]["path"], weights_only=True)
        assert payload["protocol"]["query_ids"] == ["q0", "q1"]
        assert payload["protocol"]["target_ids"] == gallery.sample_ids
        for offset, name in enumerate(payload["protocol"]["readouts"]):
            noise, codes, actual_steps = generated[-3 + offset]
            torch.testing.assert_close(noise, payload["noise"], rtol=0, atol=0)
            torch.testing.assert_close(codes, payload["codes"][name], rtol=0, atol=0)
            assert actual_steps == steps
        assert metrics["readout_selection"]["conditional_map"]["unique_targets"] == 2
    assert all(torch.equal(x[0], generated[0][0]) for x in generated)
    for i in range(3):
        torch.testing.assert_close(generated[i][1], generated[i + 3][1], rtol=0, atol=0)


def report_fixture(h):
    cfg = config()
    cfg["matching"]["projection_bandwidth_multiplier"] = h
    cfg["translation"].update(readouts=["conditional_mean", "conditional_map"], num_steps=20, samples_per_direction=2)
    decoded = {"source_query_ids": ["q0", "q1"], "image_loss_seed": 30}
    for name in ("infoot_conditional_mean", "selected_target"):
        decoded[name] = {"samples": 2, "image_losses": {"coarse_rgb": {"loss": h, "per_image_loss": [h, h]}},
                         "diagnostics": {"target_patch_swd": {"distance": 2 * h}}}
    return {"seed": 31, "generation_protocol": {"projection_bandwidth_multiplier": h},
            "evaluation_protocol": {"identifier": str(h), "checkpoint": "step_008000.pt", "checkpoint_step": 8000,
                "weights": "ema", "seed": 31, "effective_config": cfg, "ordered_sample_ids": {"cat": ["q0", "q1"]}},
            "projections": {"cat_to_dog": {"decoded_image_diagnostics": decoded}}}


def test_summary_checks_protocol_before_paired_claims_and_uses_existing_map_metric_name(tmp_path):
    a, b = report_fixture(.25), report_fixture(.35)
    assert bandwidth_comparability(a, b)["matched"]
    for key, value in (("weights", "raw"), ("checkpoint", "other.pt"), ("ordered_sample_ids", {})):
        changed = deepcopy(b)
        changed["evaluation_protocol"][key] = value
        assert not bandwidth_comparability(a, changed)["matched"]
    assert report_rows(a)[1]["coarse_rgb"] == .25  # MAP is saved as selected_target.
    paths = []
    for name, report in (("a", a), ("b", b)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    summary = summarize_reports(paths, tmp_path / "summary")
    comparison = summary["comparisons"][0]
    assert comparison["matched"]
    assert comparison["paired_image_losses"]["cat_to_dog.conditional_map.coarse_rgb"]["delta"] == pytest.approx(.1)
    b["evaluation_protocol"]["effective_config"]["translation"]["num_steps"] = 40
    paths[1].write_text(json.dumps(b))
    summary = summarize_reports(paths, tmp_path / "summary")
    assert not summary["comparisons"][0]["matched"]
    assert not summary["comparisons"][0]["paired_image_losses"]
    b["generation_protocol"]["projection_bandwidth_multiplier"] = .25
    b["evaluation_protocol"]["effective_config"]["matching"]["projection_bandwidth_multiplier"] = .25
    paths[1].write_text(json.dumps(b))
    summary = summarize_reports(paths, tmp_path / "summary")
    integration = [x for x in summary["comparisons"] if x["axis"] == "integration_steps"]
    assert len(integration) == 1 and integration[0]["matched"]
    assert integration[0]["paired_image_losses"]
