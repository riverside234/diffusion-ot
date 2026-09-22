from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from diffusion_ot.losses.semantic_prior import validate_prior_resume
from diffusion_ot.models.patch_discriminator import RGBPatchDiscriminator
from diffusion_ot.training.decoded_translation import DecoderTraining, frozen_discriminator, validate_decoder_config
from test_decoded_translation import config, domains, tiny_features
from test_diffaugment import inputs
from test_stage1b_extensions import experiment, minimal_pcgrad_recipe


def test_patchgan_returns_patch_logits_and_freezing_preserves_image_derivatives():
    torch.manual_seed(3)
    discriminator = RGBPatchDiscriminator(base_channels=4)
    image = torch.rand(2, 3, 256, 256, requires_grad=True)
    with frozen_discriminator(discriminator):
        score = discriminator(image)
        assert score.shape == (2, 1, 30, 30)
        loss = -score.mean()
    saved = deepcopy(discriminator.state_dict())
    loss.backward()
    assert image.grad.norm() > 0 and torch.isfinite(image.grad).all()
    assert all(p.grad is None for p in discriminator.parameters())
    for key, value in saved.items():
        torch.testing.assert_close(value, discriminator.state_dict()[key], rtol=0, atol=0)


@pytest.fixture
def runtime_factory(monkeypatch, tmp_path):
    import diffusion_ot.training.decoded_translation as module
    monkeypatch.setattr(module, "load_image_features", tiny_features)

    def originals(self, domain, records):
        return torch.stack([torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(
            int(record["sample_id"]) + 700)) for record in records])
    monkeypatch.setattr(DecoderTraining, "original_source_images", originals)

    def make(kind="rgb_patchgan", weight=.02, augment=True):
        torch.manual_seed(29)
        cfg = config()
        cfg["decoded_translation"].update(
            discriminator_kind=kind, discriminator_base_channels=4,
            perceptual_weight=.1, perceptual_mode="contrastive", structure_weight=0.,
            color_histogram={"weight": weight, "bins": 16, "input_size": 16, "sigma": .05},
            diffaugment={"enabled": augment, "translation_ratio": .125})
        validate_decoder_config(cfg)
        ctx = domains()
        for value in ctx.values():
            decode = value.vae.decode
            value.vae.decode = lambda x, decode=decode: SimpleNamespace(
                sample=F.interpolate(decode(x).sample, (32, 32), mode="bilinear", align_corners=False))
        return DecoderTraining(cfg, ctx, None, tmp_path, seed=18)
    return make


def call_inputs():
    args, kwargs = inputs()
    kwargs["source_query_metadata"] = {d: [{"sample_id": str(i)} for i in range(4)] for d in ("cat", "dog")}
    return args, kwargs


def test_patch_color_routes_gradients_and_augments_only_adversarial_inputs(runtime_factory):
    runtime = runtime_factory()
    control = runtime_factory(weight=0., augment=False)
    args, kwargs = call_inputs()
    seen = []
    hook = runtime.discriminators["dog"].register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach()))
    loss, metrics = runtime.loss(*args, step=1, **kwargs)
    hook.remove()
    _, baseline = control.loss(*args, step=1, **kwargs)
    assert len(seen) == 4 and all(x.shape == (2, 3, 32, 32) for x in seen)
    assert (seen[0] == 0).any() and (seen[1] == 0).any()
    torch.testing.assert_close(seen[1], seen[2], rtol=0, atol=0)  # same fake shifts for D/G
    assert metrics["perceptual_loss"] == pytest.approx(baseline["perceptual_loss"])
    assert metrics["structure_loss"] == pytest.approx(baseline["structure_loss"])
    assert metrics["ramp"] == .5
    assert metrics["color_histogram_effective_weighted_loss"] == pytest.approx(.01 * metrics["color_histogram_loss"])
    assert float(loss.detach()) == pytest.approx(.5 * (.1 * metrics["perceptual_loss"] +
        .01 * metrics["adversarial_loss"] + .02 * metrics["color_histogram_loss"]))
    assert set(runtime.image_objectives) == {"perceptual", "adversarial", "structure", "color"}
    color_grads = torch.autograd.grad(runtime.image_objectives["color"], tuple(args[1].values()), retain_graph=True)
    assert all(g.norm() > 0 for g in color_grads)
    loss.backward()
    assert all(ref.grad is not None and ref.grad.norm() > 0 for ref in args[1].values())
    assert all(p.grad is None for p in runtime.features.parameters())
    assert all(p.grad is None for p in runtime.discriminators.parameters())
    assert all(p.grad is None for value in runtime.domains.values() for p in value.vae.parameters())


def test_fixed_images_histograms_rng_and_checkpoint_resume(runtime_factory, tmp_path):
    runtime = runtime_factory()
    args, kwargs = call_inputs()
    runtime.loss(*args, step=2, **kwargs)
    saved = deepcopy(runtime.checkpoint_state())
    saved.update(config=deepcopy(runtime.config), format_version=4)
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        first = runtime.loss(*args, step=2, validation_seed=21, validation_image_dir=tmp_path / "panel", **kwargs)[1]
        second = runtime.loss(*args, step=2, validation_seed=21, validation_image_dir=tmp_path / "panel", **kwargs)[1]
    assert first == second and torch.equal(rng, torch.get_rng_state())
    assert not first["adversarial_augmentation"]["applied"]
    for direction in ("cat_to_dog", "dog_to_cat"):
        assert Path(first[direction]["validation_grid"]).is_file()
        assert len(first[direction]["per_image_color_histogram_loss"]) == 2
    for key in ("decoded_discriminators", "decoded_augmentation_states", "decoded_noise_states"):
        for name, value in saved[key].items():
            torch.testing.assert_close(value, runtime.checkpoint_state()[key][name], rtol=0, atol=0)
    expected, expected_metrics = runtime.loss(*args, step=3, **kwargs)
    resumed = runtime_factory()
    resumed.load_checkpoint(saved)
    actual, actual_metrics = resumed.loss(*args, step=3, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_metrics == expected_metrics
    with pytest.raises(ValueError, match="different discriminator kind"):
        runtime_factory(kind="dino_feature").load_checkpoint(saved)


def test_zero_weight_control_measures_same_validation_color_without_training_term(runtime_factory):
    runtime, control = runtime_factory(), runtime_factory(weight=0.)
    args, kwargs = call_inputs()
    with torch.no_grad():
        first = runtime.loss(*args, step=0, validation_seed=21, **kwargs)[1]
        second = control.loss(*args, step=0, validation_seed=21, **kwargs)[1]
    assert first["color_histogram_loss"] == second["color_histogram_loss"]
    assert "color" not in control.image_objectives
    assert second["color_histogram_effective_weighted_loss"] == 0.
    _, metrics = control.loss(*args, step=2, **kwargs)
    assert "color_histogram_loss" not in metrics


def test_active_controls_only_change_requested_training_objectives():
    root = Path(__file__).resolve().parents[1] / "configs/stage1b_infoot"
    configs = [yaml.safe_load((root / name).read_text()) for name in (
        "structure_decoder_sit_b2.yaml", "structure_decoder_patch_sit_b2.yaml", "structure_decoder_dino_control_sit_b2.yaml")]
    evaluation = yaml.safe_load((root.parent / "stage1b_eval/structure_decoder_sit_b2.yaml").read_text())
    assert evaluation["matching"]["bandwidth_multiplier"] == configs[0]["matching"]["bandwidth_multiplier"] == .55
    assert evaluation["matching"]["projection_bandwidth_multiplier"] == configs[0]["conditional_structure"]["bandwidth_multiplier"] == .10
    assert evaluation["infoot"] == configs[0]["infoot"]
    assert evaluation["translation"]["color_histogram"] == {k: v for k, v in configs[0]["decoded_translation"]["color_histogram"].items() if k != "weight"}
    for cfg in configs:
        validate_decoder_config(cfg)
        validate_prior_resume(cfg, cfg)
        assert cfg["decoded_translation"]["save_validation_images"]
        cfg.pop("output_dir")
        image = cfg["decoded_translation"]
        image.pop("discriminator_kind", None)
        image.pop("discriminator_base_channels", None)
        image.pop("discriminator_hidden_dim", None)
        image["color_histogram"].pop("weight")
    assert configs[0] == configs[1] == configs[2]


@pytest.mark.parametrize("key,value", [("discriminator_kind", "typo"), ("discriminator_base_channels", 0),
    ("discriminator_base_channels", True), ("save_validation_images", "yes"), ("color_histogram", {"sigma": 0})])
def test_invalid_patch_color_config_is_rejected(key, value):
    cfg = config()
    cfg["decoded_translation"][key] = value
    with pytest.raises(ValueError):
        validate_decoder_config(cfg)


def test_pcgrad_patch_color_trains_validates_and_resumes(experiment, monkeypatch):
    from test_decoded_translation import TinyVAE
    decode = TinyVAE.decode
    monkeypatch.setattr(TinyVAE, "decode", lambda self, x: SimpleNamespace(
        sample=F.interpolate(decode(self, x).sample, (32, 32), mode="bilinear", align_corners=False)))
    original = DecoderTraining.original_source_images
    monkeypatch.setattr(DecoderTraining, "original_source_images", lambda self, domain, records:
        F.interpolate(original(self, domain, records), (32, 32), mode="bilinear", align_corners=False))

    def recipe(cfg):
        minimal_pcgrad_recipe(cfg)
        cfg["decoded_translation"].update(discriminator_kind="rgb_patchgan", discriminator_base_channels=4,
            color_histogram={"weight": .02, "bins": 8, "input_size": 16, "sigma": .1},
            save_validation_images=True, diffaugment={"enabled": True, "translation_ratio": .0625})

    run, _, _ = experiment
    saved, logs = run("patch_color", steps=1, modify=recipe)
    assert saved["decoded_discriminator_kind"] == "rgb_patchgan"
    row = logs["train"][0]
    for group in ("encoder", "matching_head", "generator"):
        assert "color" in row["pcgrad"]["groups"][group]["active_tasks"]
        assert row["gradient_conflicts"]["groups"][group + ".all"]["color_vs_adversarial"]["valid"]
    for row in logs["validation"]:
        metrics = row["decoded_translation"]
        assert 0 <= metrics["color_histogram_loss"] <= 1.
        for direction in ("cat_to_dog", "dog_to_cat"):
            path = Path(metrics[direction]["validation_grid"])
            assert path.is_file() and path.parent.name == f"step_{row['step']:06d}"
    assert logs["validation"][0]["decoded_translation"]["query_ids"] == logs["validation"][-1]["decoded_translation"]["query_ids"]
    resumed, resumed_logs = run("patch_color", resume=True, modify=recipe)
    assert resumed["step"] == resumed["decoded_discriminator_updates"] == 2
    assert "color" in resumed_logs["train"][-1]["pcgrad"]["groups"]["encoder"]["active_tasks"]
    def changed(cfg):
        recipe(cfg)
        cfg["decoded_translation"]["color_histogram"]["weight"] = .04
    with pytest.raises(ValueError, match="Resume cannot change decoded_translation"):
        run("patch_color", resume=True, modify=changed)


def test_standalone_evaluation_uses_original_rgb_and_preserves_generated_panel(runtime_factory, monkeypatch, tmp_path):
    import diffusion_ot.data.ground_truth as gt
    import diffusion_ot.evaluation.stage1b_eval as evaluation
    runtime = runtime_factory()
    source, target = runtime.domains["cat"], runtime.domains["dog"]
    source.data_config_path = tmp_path / "unused.yaml"
    latents = torch.randn(2, 4, 8, 8)
    metadata = [{"sample_id": str(i)} for i in range(2)]
    originals = runtime.original_source_images("cat", metadata)
    monkeypatch.setattr(evaluation, "_load_latents_from_bank", lambda bank, count: latents[:count])
    def load(path, records):
        assert path == source.data_config_path and records == metadata
        return originals
    monkeypatch.setattr(gt, "load_ground_truth_images", load)
    bank = SimpleNamespace(sample_ids=["0", "1"], metadata=metadata)
    target_bank = SimpleNamespace(raw_codes=torch.randn(5, 6))
    weights = torch.softmax(torch.randn(2, 5), dim=1)
    kwargs = dict(count=2, num_steps=3, guidance_scale=1., temperature=1., seed=23,
                  readouts=["conditional_mean"], include_source=False)
    state = torch.get_rng_state().clone()
    with torch.no_grad():
        baseline = evaluation._save_translation_grid(source, target, bank, target_bank, weights,
            output_path=tmp_path / "baseline.png", **kwargs)
        measured = evaluation._save_translation_grid(source, target, bank, target_bank, weights,
            output_path=tmp_path / "measured.png", color_histogram={"bins": 16, "input_size": 16}, **kwargs)
    assert not baseline
    assert (tmp_path / "baseline.png").read_bytes() == (tmp_path / "measured.png").read_bytes()
    assert Path(measured["source_pair_grid"]).is_file()
    assert measured["source_query_ids"] == ["0", "1"]
    assert measured["color_histogram_target"] == "original_source_rgb_whole_image"
    row_name = evaluation._TRANSLATION_ROW_NAMES["conditional_mean"]
    assert len(measured[row_name]["per_image_color_histogram_loss"]) == 2
    assert torch.equal(state, torch.get_rng_state())
