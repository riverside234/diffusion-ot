"""Runtime, logging, validation and resume coverage for the four RGB additions."""
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from diffusion_ot.evaluation.stage1b_eval import _validate_self_supervised_checkpoint
from diffusion_ot.losses.translation_image import IMAGE_LOSS_PROTOCOL
from diffusion_ot.training.self_supervised_translation import SelfSupervisedDecoderTraining
from test_self_supervised_translation import config, domains, batch
from test_translation_image import image_options
from test_stage1b_extensions import experiment, assert_finite_numbers, assert_tensor_tree_equal
from test_stage1b_self_supervised import self_supervised_recipe
from test_decoded_translation import TinyVAE


def image_recipe(cfg):
    self_supervised_recipe(cfg)
    cfg["decoded_translation"].update(image_options())
    cfg["decoded_translation"]["source_contrastive_projection_gradient_scale"] = .1


def test_actual_pcgrad_training_logs_all_terms_and_resume_restores_private_image_rng(experiment, monkeypatch):
    import diffusion_ot.training.decoded_translation as external
    def forbidden(*args, **kwargs):
        pytest.fail("Loaded an external image teacher or discriminator")
    for name in ("load_image_features", "RGBPatchDiscriminator", "FeatureDiscriminator"):
        monkeypatch.setattr(external, name, forbidden)
    monkeypatch.setattr(TinyVAE, "encode", lambda self, rgb: SimpleNamespace(
        latent_dist=SimpleNamespace(mean=F.conv2d(rgb, self.conv.weight.transpose(0, 1)))), raising=False)
    run, _, _ = experiment
    complete, logs = run("rgb_complete", modify=image_recipe)
    for row in logs["train"] + logs["validation"]:
        assert_finite_numbers(row)
        assert row["log_schema_version"] == 4
        assert set(image_options()) <= set(row["enabled_losses"])
        decoded = row["decoded_translation"]
        assert decoded["image_loss_protocol"] == IMAGE_LOSS_PROTOCOL
        assert set(decoded["image_losses"]) == set(image_options())
        assert decoded["weighted_loss"] == pytest.approx(.1 * decoded["source_contrastive_loss"] + sum(
            value["weight"] * value["loss"] for value in decoded["image_losses"].values()))
        assert decoded["effective_weighted_loss"] == pytest.approx(decoded["ramp"] * decoded["weighted_loss"])
        for source, target in (("cat", "dog"), ("dog", "cat")):
            direction = decoded[f"{source}_to_{target}"]
            assert all(key.startswith(f"{target}_train") for key in direction["target_rgb_reference_ids"])
            assert all(value["loss"] >= 0 for value in direction["image_losses"].values())
            if row in logs["validation"]:
                assert not set(direction["target_rgb_baseline_ids"]) & set(direction["target_rgb_reference_ids"])
                assert direction["image_losses"]["target_patch_swd"]["real_to_real_distance"] >= 0
    for row in logs["train"]:
        for name in image_options():
            assert row[f"weighted_{name}_generator_gradient_norm"] > 0
            assert row["window_mean"][f"{name}_loss"] >= 0
            assert f"{name}_vs_source_contrastive" in row["gradient_conflicts"]["groups"]["generator.all"]
        assert set(row["pcgrad"]["groups"]["generator"]["active_tasks"]) == {"native", "decoded_images"}
    run("rgb_resumed", steps=1, modify=image_recipe)
    resumed, _ = run("rgb_resumed", resume=True, modify=image_recipe)
    for field in ("decoded_image_sampling_states", "decoded_noise_states", "rng_state"):
        assert_tensor_tree_equal(complete[field], resumed[field])
    _validate_self_supervised_checkpoint(complete["config"], complete)


def test_fixed_image_validation_is_repeatable_without_consuming_training_rng(tmp_path, monkeypatch):
    cfg = config()
    cfg["decoded_translation"].update(image_options())
    runtime = SelfSupervisedDecoderTraining(cfg, domains(), None, tmp_path, seed=15)
    seen = []
    def originals(domain, records):
        seen.append((domain, tuple(r["sample_id"] for r in records)))
        return torch.rand(len(records), 3, 8, 8, generator=torch.Generator().manual_seed(len(records) + 19))
    monkeypatch.setattr(runtime, "original_source_images", originals)
    weights, refs, keys, _, latents = batch()
    metadata = {d: [{"sample_id": f"{d}_{i}"} for i in range(5)] for d in domains()}
    before = deepcopy(runtime.checkpoint_state())
    global_state = torch.get_rng_state().clone()
    with torch.no_grad():
        first = runtime.loss(weights, refs, latents, latents, step=0, validation_seed=10,
            source_query_codes=keys, source_query_metadata=metadata, source_reference_metadata=metadata)
        second = runtime.loss(weights, refs, latents, latents, step=0, validation_seed=10,
            source_query_codes=keys, source_query_metadata=metadata, source_reference_metadata=metadata)
    assert_tensor_tree_equal(first, second)
    assert_tensor_tree_equal(before, runtime.checkpoint_state())
    torch.testing.assert_close(global_state, torch.get_rng_state())
    assert seen[:4] == seen[4:]
    restored = SelfSupervisedDecoderTraining(cfg, domains(), None, tmp_path, seed=99)
    restored.load_checkpoint({**before, "config": cfg, "format_version": 4})
    assert_tensor_tree_equal(before, restored.checkpoint_state())
    bad = deepcopy(before)
    bad["decoded_image_loss_options"]["coarse_rgb"]["weight"] = .05
    with pytest.raises(ValueError, match="image-loss"):
        restored.load_checkpoint(bad)
    bad = deepcopy(before)
    bad.pop("decoded_image_sampling_states")
    with pytest.raises(ValueError, match="sampling state"):
        restored.load_checkpoint(bad)


def test_external_image_guard_and_image_protocol_cannot_be_bypassed(tmp_path):
    cfg = config()
    cfg["decoded_translation"].update(image_options())
    runtime = SelfSupervisedDecoderTraining(cfg, domains(), None, tmp_path, seed=3)
    payload = {**runtime.checkpoint_state(), "config": deepcopy(cfg)}
    _validate_self_supervised_checkpoint(cfg, payload)
    changed = deepcopy(cfg)
    changed["decoded_translation"]["local_layout"]["weight"] = 0
    with pytest.raises(ValueError, match="image-loss"):
        _validate_self_supervised_checkpoint(changed, payload)
    legacy = SelfSupervisedDecoderTraining(config(), domains(), None, tmp_path, seed=3)
    with pytest.raises(ValueError, match="image-loss"):
        runtime.load_checkpoint(legacy.checkpoint_state())


def test_standalone_reports_all_terms_on_same_images_without_extra_rollouts(tmp_path, monkeypatch):
    import diffusion_ot.data.ground_truth as gt
    import diffusion_ot.evaluation.stage1a_eval as stage1a
    import diffusion_ot.evaluation.stage1b_eval as evaluation
    import torchvision.utils
    from diffusion_ot.losses.translation_image import translation_image_options

    runtime = SelfSupervisedDecoderTraining(config(), domains(), None, tmp_path, seed=5)
    source, target = runtime.domains["cat"], runtime.domains["dog"]
    source.data_config_path, target.data_config_path = tmp_path / "cat.yaml", tmp_path / "dog.yaml"
    latents = torch.randn(2, 4, 8, 8)
    source_records = [{"sample_id": f"cat_val_{i}"} for i in range(2)]
    target_records = [{"sample_id": f"dog_train_{i}"} for i in range(5)]
    bank = SimpleNamespace(metadata=source_records, sample_ids=[r["sample_id"] for r in source_records])
    target_bank = SimpleNamespace(raw_codes=torch.randn(5, 6), metadata=target_records)
    monkeypatch.setattr(evaluation, "_load_latents_from_bank", lambda bank, count: latents[:count])
    records_seen = []
    def originals(path, records):
        domain = "cat" if path == source.data_config_path else "dog"
        assert all(r["sample_id"].startswith(domain) for r in records)
        records_seen.extend(r["sample_id"] for r in records)
        return torch.stack([torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(
            int(r["sample_id"].rsplit("_", 1)[1]) + 30)) for r in records])
    monkeypatch.setattr(gt, "load_ground_truth_images", originals)
    saved_images, conditions = [], []
    monkeypatch.setattr(torchvision.utils, "save_image", lambda images, *a, **kw: saved_images.append(images.clone()))
    integrate = stage1a.integrate_pdae_flow
    def capture(*args, **kwargs):
        conditions.append(args[3].clone())
        return integrate(*args, **kwargs)
    monkeypatch.setattr(stage1a, "integrate_pdae_flow", capture)
    weights = torch.softmax(torch.randn(2, 5), -1)
    args = dict(count=2, num_steps=3, guidance_scale=1., temperature=1., seed=5,
                readouts=["conditional_mean", "conditional_map"], output_path=tmp_path / "grid.png")
    state = torch.get_rng_state().clone()
    baseline = evaluation._save_translation_grid(source, target, bank, target_bank, weights, **args)
    measured = evaluation._save_translation_grid(source, target, bank, target_bank, weights,
        **args, teacher_free_image_options=translation_image_options(image_options()))
    assert baseline == {} and len(conditions) == 4
    assert_tensor_tree_equal(conditions[:2], conditions[2:])
    torch.testing.assert_close(saved_images[0], saved_images[1], rtol=0, atol=0)
    torch.testing.assert_close(state, torch.get_rng_state())
    assert measured["image_loss_protocol"] == IMAGE_LOSS_PROTOCOL
    assert set(measured["target_rgb_reference_ids"]).isdisjoint(measured["target_rgb_baseline_ids"])
    for readout in ("infoot_conditional_mean", "selected_target"):
        assert set(measured[readout]["image_losses"]) == set(image_options())
        assert measured[readout]["image_losses"]["target_patch_swd"]["real_to_real_distance"] >= 0
    assert set(records_seen[:2]) == {"cat_val_0", "cat_val_1"}
