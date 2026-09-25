"""CPU training coverage for the isolated PDAE-only Stage 1B experiment."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from diffusion_ot.models.generator_adaptation import generator_parameter_view
from test_decoded_translation import TinyVAE
from test_stage1b_extensions import (
    assert_finite_numbers,
    assert_tensor_tree_equal,
    experiment,
    minimal_pcgrad_recipe,
)


def self_supervised_recipe(cfg):
    minimal_pcgrad_recipe(cfg)
    cfg.pop("semantic_prior", None)
    projection = dict(cfg["conditional_structure"])
    projection.pop("teacher_temperature", None)
    cfg["conditional_projection"] = projection
    cfg["conditional_structure"] = {"enabled": False}
    cfg["infoot"].update(cross_cost_source="encoder", cross_cost_weight=1.)
    cfg["loss_weights"].update(conditional_structure=0., projection_support=0.,
                               infoot_alignment=.2, alignment_warmup_steps=1)
    cfg["decoded_translation"].update(
        supervision="self_supervised", perceptual_weight=0., adversarial_weight=0.,
        source_contrastive_weight=.1, source_contrastive_temperature=.2,
        source_contrastive_negative_similarity_threshold=1., source_contrastive_readout="target",
        color_histogram={"weight": 0.}, diffaugment={"enabled": False},
    )
    cfg["self_supervised_diagnostics"] = {"enabled": True, "image_samples": 2, "pooled_size": 4}


@pytest.fixture
def own_encoder_run(experiment, monkeypatch):
    import diffusion_ot.losses.color_histogram as color
    import diffusion_ot.losses.conditional_structure as conditional
    import diffusion_ot.losses.semantic_prior as semantic
    import diffusion_ot.training.decoded_translation as decoded

    def forbidden(*args, **kwargs):
        raise AssertionError("The PDAE-only experiment executed external supervision.")

    monkeypatch.setattr(decoded, "load_image_features", forbidden)
    monkeypatch.setattr(semantic, "SemanticPriorBank", forbidden)
    monkeypatch.setattr(semantic, "neighborhood_distillation_loss", forbidden)
    monkeypatch.setattr(conditional, "conditional_structure_loss", forbidden)
    monkeypatch.setattr(decoded, "RGBPatchDiscriminator", forbidden)
    monkeypatch.setattr(decoded, "FeatureDiscriminator", forbidden)
    monkeypatch.setattr(decoded, "histogan_color_distance", forbidden)
    monkeypatch.setattr(color, "histogan_color_distance", forbidden)
    calls = []

    def encode(self, rgb):
        # Deterministic tiny RGB -> latent transform, preserving image gradients.
        calls.append(tuple(rgb.shape))
        latent = F.conv2d(rgb, self.conv.weight.transpose(0, 1))
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=latent))

    monkeypatch.setattr(TinyVAE, "encode", encode, raising=False)
    return (*experiment, calls)


def assert_no_external_metrics(record):
    forbidden = {"conditional_structure", "conditional_structure_loss", "semantic_neighborhood_loss",
                 "perceptual_loss", "structure_loss", "adversarial_loss", "color_histogram_loss",
                 "matching_contrastive", "code_consistency_loss", "cat_anchor_loss", "dog_anchor_loss"}
    if isinstance(record, dict):
        assert not forbidden.intersection(record)
        for value in record.values():
            assert_no_external_metrics(value)
    elif isinstance(record, list):
        for value in record:
            assert_no_external_metrics(value)


def test_own_encoder_training_validates_updates_and_resumes_without_external_models(own_encoder_run):
    run, originals, latest, encode_calls = own_encoder_run
    complete, logs = run("self_complete", modify=self_supervised_recipe)
    assert complete["step"] == 2
    assert complete["decoded_supervision"] == "self_supervised"
    assert complete["decoded_source_contrastive_readout"] == "target"
    assert not any("discriminator" in key or "augmentation" in key for key in complete)
    assert [row["step"] for row in logs["validation"]] == [0, 1, 2]
    assert encode_calls and all(shape == (2, 3, 8, 8) for shape in encode_calls)
    expected_tasks = {
        "encoder": {"native", "transport", "protection", "source_contrastive"},
        "matching_head": {"transport", "protection", "source_contrastive"},
        "generator": {"native", "source_contrastive"},
    }
    for row in logs["train"]:
        assert row["optimizer_gradient_mode"] == "pcgrad"
        assert row["weighted_decoded_encoder_gradient_norm"] > 0
        assert row["weighted_decoded_generator_gradient_norm"] > 0
        assert row["weighted_decoded_matching_head_gradient_norm"] > 0
        assert "source_contrastive" in row["enabled_losses"]
        assert row["log_schema_version"] == 3
        assert "primary_objective" not in row["window_mean"]
        assert "auxiliary_objective" not in row["window_mean"]
        assert "infoot_restart" not in row
        assert "solver_health" in row and "solver_window_max" in row
        assert "decoded_to_reconstruction_generator_gradient_ratio" in row
        assert "infoot_vs_translation" in row["gradient_conflicts"]["groups"]["encoder.all"]
        for group, tasks in expected_tasks.items():
            assert set(row["pcgrad"]["groups"][group]["active_tasks"]) == tasks
    for row in logs["train"] + logs["validation"]:
        assert_finite_numbers(row)
        assert_no_external_metrics(row)
        assert "conditional_projection" in row
        decoded = row["decoded_translation"]
        assert decoded["supervision"] == "self_supervised"
        assert decoded["source_contrastive_loss"] > 0
        assert decoded["weighted_loss"] == pytest.approx(.1 * decoded["source_contrastive_loss"])
        for source, target in (("cat", "dog"), ("dog", "cat")):
            direction = f"{source}_to_{target}"
            assert decoded[direction]["readout_domain"] == target
            retrieval = decoded[direction]["source_contrastive"]
            assert retrieval["positive_target"] == "detached_original_source_encoder_code"
            assert retrieval["negative_bank_size"] == 8
            assert retrieval["usable_samples"] == 2
            assert row["conditional_projection"][direction]["reference_targets"] == 6
    for row in logs["validation"]:
        assert not any("stage1a" in key for key in row)
        assert row["loss"] == pytest.approx(row["cat_raw_reconstruction"] + row["dog_raw_reconstruction"]
            + row["weighted_infoot_alignment_loss"] + row["matching_regularization_loss"]
            + row["decoded_translation"]["weighted_loss"])
        for domain, target in (("cat", "dog"), ("dog", "cat")):
            decoded = row["decoded_translation"]
            native = decoded["native_reconstruction"][domain]
            assert native["target"] == "original_rgb"
            assert native["samples"] == 2 and native["pixel_mse"] > 0
            assert native["query_ids"] == decoded["query_ids"][domain]
            assert "source_correspondence" in decoded[f"{domain}_to_{target}"]
    for domain, original in originals.items():
        current = latest[domain].branch
        assert any(not torch.equal(value, original.branch.encoder.state_dict()[key])
                   for key, value in current.encoder.state_dict().items())
        for group in ("adapters", "lora"):
            before = generator_parameter_view(original.branch)[group].state_dict()
            assert any(not torch.equal(value, before[key])
                       for key, value in generator_parameter_view(current)[group].state_dict().items())
    run("self_resumed", steps=1, modify=self_supervised_recipe)
    resumed, resumed_logs = run("self_resumed", resume=True, modify=self_supervised_recipe)
    assert resumed["step"] == 2
    assert [row["step"] for row in resumed_logs["train"]] == [1, 2]
    # The existing shuffled DataLoader does not restore its current permutation
    # and cursor. Verify the supported private-noise/global-RNG guarantees;
    # do not promise bitwise continuation of model weights or training batches.
    for key in ("decoded_noise_states", "rng_state"):
        assert_tensor_tree_equal(complete[key], resumed[key])


def test_own_encoder_resume_rejects_changed_source_objective_and_projection(own_encoder_run):
    run, _, _, _ = own_encoder_run
    run("self_resume_guard", steps=1, modify=self_supervised_recipe)
    for key, update in (("decoded_translation", {"source_contrastive_weight": .2}),
                        ("decoded_translation", {"source_contrastive_readout": "source"}),
                        ("conditional_projection", {"bandwidth_multiplier": .4})):
        def modify(cfg):
            self_supervised_recipe(cfg)
            cfg[key].update(update)
        with pytest.raises(ValueError, match=f"Resume cannot change {key}"):
            run("self_resume_guard", resume=True, modify=modify)


def test_own_encoder_diagnostics_preserve_updates_and_noise(own_encoder_run):
    run, _, _, _ = own_encoder_run
    measured, _ = run("self_measured", modify=self_supervised_recipe)
    def quiet(cfg):
        self_supervised_recipe(cfg)
        cfg["train"].update(gradient_conflicts=False, gradient_diagnostics_every=0)
        cfg["self_supervised_diagnostics"]["enabled"] = False
    control, _ = run("self_quiet", modify=quiet)
    for key in ("encoders", "matching_heads", "generators", "optimizer", "rng_state", "decoded_noise_states"):
        assert_tensor_tree_equal(measured[key], control[key])


def test_self_validation_writes_native_and_translation_panels_on_matching_originals(own_encoder_run):
    run, _, _, _ = own_encoder_run
    def panels(cfg):
        self_supervised_recipe(cfg)
        cfg["decoded_translation"]["save_validation_images"] = True
    _, logs = run("self_panels", steps=1, modify=panels)
    first, last = [row["decoded_translation"] for row in logs["validation"]]
    for domain, target in (("cat", "dog"), ("dog", "cat")):
        native = last["native_reconstruction"][domain]
        translated = last[f"{domain}_to_{target}"]
        assert Path(native["validation_grid"]).is_file()
        assert Path(translated["validation_grid"]).is_file()
        assert native["query_ids"] == first["native_reconstruction"][domain]["query_ids"]
        assert native["seed"] == first["native_reconstruction"][domain]["seed"]
        assert native["samples"] == translated["source_correspondence"]["samples"] == 2


def test_unscheduled_translation_does_not_dilute_window_raw_infonce(own_encoder_run):
    run, _, _, _ = own_encoder_run
    def intermittent(cfg):
        self_supervised_recipe(cfg)
        cfg["decoded_translation"]["every_steps"] = 2
    _, logs = run("self_intermittent", modify=intermittent)
    skipped, active = logs["train"]
    assert skipped["decoded_translation"] == {"active": False}
    assert skipped["window_mean"]["source_contrastive_loss"] is None
    assert active["window_mean"]["source_contrastive_active"] == .5
    assert active["window_mean"]["source_contrastive_loss"] == active["decoded_translation"]["source_contrastive_loss"]
    assert active["window_mean"]["decoded_translation_loss"] == pytest.approx(.5 * active["decoded_translation"]["effective_weighted_loss"])


def test_self_supervised_configs_match_and_link_original_flow_only_checkpoints():
    root = Path(__file__).resolve().parents[1]
    train = yaml.safe_load((root / "configs/stage1b_infoot/self_supervised_sit_b2.yaml").read_text())
    evaluate = yaml.safe_load((root / train["quick_evaluation"]["config"]).read_text())
    assert train["output_dir"] == "outputs/stage1b_self"
    assert train["decoded_translation"]["source_contrastive_weight"] == .15
    assert train["self_supervised_diagnostics"]["enabled"]
    assert train["infoot"] == evaluate["infoot"]
    assert train["matching"]["bandwidth_multiplier"] == evaluate["matching"]["bandwidth_multiplier"] == .55
    assert train["conditional_projection"]["bandwidth_multiplier"] == evaluate["matching"]["projection_bandwidth_multiplier"] == .1
    assert not train.get("semantic_prior") and not evaluate.get("semantic_prior")
    assert not evaluate["translation"]["decoded_structure_metrics"]
    assert not evaluate["translation"]["include_structure_teacher_mean"]
    assert "lpips" not in evaluate["reconstruction"]["metrics"]
    for domain in ("cat", "dog"):
        stage1a = train["stage1a"][domain]
        assert stage1a["checkpoint"] == f"outputs/stage1a_{domain}_sit_b2_cfg_adaln_all_lora_r64/checkpoints/latest.pt"
        original = yaml.safe_load((root / stage1a["config"]).read_text())
        assert not (original.get("refinement") or {}).get("enabled", False)
    # Validate the actual recipe's disabled teacher terms, not just the fixture.
    from diffusion_ot.training.decoded_translation import validate_decoder_config
    validate_decoder_config(train)


@pytest.mark.parametrize("rms_enabled", [False, True])
@pytest.mark.parametrize("weights", ["raw", "ema"])
def test_standalone_self_evaluation_uses_encoder_cost_without_external_features(
        own_encoder_run, monkeypatch, tmp_path, rms_enabled, weights):
    import diffusion_ot.data.ground_truth as ground_truth
    import diffusion_ot.evaluation.stage1b_eval as evaluation
    from diffusion_ot.models.matching_head import make_matching_head, matching_head_spec
    from diffusion_ot.models.generator_adaptation import load_joint_generator
    from diffusion_ot.losses.projection_rms import checkpoint_projection_rms

    run, _, latest, _ = own_encoder_run
    def recipe(cfg):
        self_supervised_recipe(cfg)
        if rms_enabled:
            cfg["projection_rms"] = {"mode": "reference_ema", "decay": .99, "eps": 1e-8}
    checkpoint, _ = run("self_eval_training", steps=1, modify=recipe)
    config = checkpoint["config"]
    checkpoint_path = tmp_path / "self_eval_training/checkpoints/latest.pt"
    root = Path(__file__).resolve().parents[1]
    eval_config = yaml.safe_load((root / "configs/stage1b_eval/self_supervised_sit_b2.yaml").read_text())
    eval_config.update(project_root=str(tmp_path), output_dir="eval", alignment_device="cpu")
    eval_config["infoot"] = dict(config["infoot"])
    if rms_enabled:
        eval_config["projection_rms"] = dict(config["projection_rms"])
    eval_config["matching"]["projection_bandwidth_multiplier"] = config["conditional_projection"]["bandwidth_multiplier"]
    eval_config["data"].update(reference_samples_per_domain=6, projection_samples_per_domain=8,
                               query_samples_per_domain=2, batch_size=4, num_workers=0)
    eval_config["translation"].update(samples_per_direction=2, num_steps=3)
    eval_config["reconstruction"].update(samples_per_domain=2, num_steps=3)
    eval_config["visualization"]["enabled"] = False
    eval_path = tmp_path / "self_evaluation.yaml"
    eval_path.write_text(yaml.safe_dump(eval_config))

    def latent(sample_id):
        domain, split, index = sample_id.split("_")
        generator = torch.Generator().manual_seed(int(index) + (100 if domain == "dog" else 0)
                                                 + (20 if split == "val" else 0))
        return torch.randn(4, 8, 8, generator=generator)

    class Dataset:
        def __init__(self, domain, split):
            self.domain, self.split = domain, split
        def __len__(self):
            return 8
        def __getitem__(self, index):
            sample_id = f"{self.domain}_{self.split}_{index}"
            return {"x0_latent": latent(sample_id), "sample_id": sample_id,
                    "domain": self.domain, "split": self.split,
                    "metadata": {"sample_id": sample_id, "latent_path": "unused"}}

    def load_context(alignment, project_root, domain, **kwargs):
        value = latest[domain]
        value.domain = domain
        value.stage1a_checkpoint_path = tmp_path / f"{domain}.pt"
        value.stage1a_checkpoint_step = 100
        value.stage1a_weights = "ema"
        value.checkpoint_path = checkpoint_path
        value.checkpoint_step = 1
        value.weights = weights
        value.branch.encoder.load_state_dict(checkpoint["encoder_ema" if weights == "ema" else "encoders"][domain])
        load_joint_generator(value.branch, checkpoint, domain, weights=weights)
        value.matching_head = make_matching_head(matching_head_spec(config), device="cpu")
        value.matching_head.load_state_dict(checkpoint["matching_head_ema" if weights == "ema" else "matching_heads"][domain])
        value.projection_rms = checkpoint_projection_rms(checkpoint, config, weights=weights)
        value.branch.eval()
        value.matching_head.eval()
        return value

    monkeypatch.setattr(evaluation, "_load_domain_context", load_context)
    monkeypatch.setattr(evaluation, "_dataset", lambda context, split, root: Dataset(context.domain, split))
    monkeypatch.setattr(evaluation, "_load_latents_from_bank",
                        lambda bank, count: torch.stack([latent(s) for s in bank.sample_ids[:count]]))
    monkeypatch.setattr(ground_truth, "load_ground_truth_images",
                        lambda path, rows: torch.stack([latent(row["sample_id"])[:3].sigmoid() for row in rows]))
    def forbidden(*args, **kwargs):
        raise AssertionError("The PDAE-only evaluator loaded external image features.")
    monkeypatch.setattr(evaluation, "_lpips", forbidden)
    report = evaluation.run_stage1b_evaluation(
        tmp_path / "self_eval_training.yaml", eval_path, checkpoint_path=checkpoint_path, weights=weights)
    assert report.solver["cross_cost_source"] == "encoder"
    assert report.solver["semantic_prior_fingerprint"] is None
    assert report.solver["sinkhorn_converged"] and report.solver["outer_converged"]
    assert "transport_encoder_cost" in report.solver and "transport_structure_cost" not in report.solver
    assert report.reference_sizes == {"cat": 6, "dog": 6}
    assert report.projection_sizes == {"cat": 8, "dog": 8}
    assert report.query_sizes == {"cat": 2, "dog": 2}
    for domain in ("cat", "dog"):
        assert report.reconstruction[domain]["image_reference"] == "original_dataset_rgb"
        assert "lpips" not in report.reconstruction[domain]
    for direction in ("cat_to_dog", "dog_to_cat"):
        assert Path(report.translation_grids[direction]).is_file()
        assert report.projections[direction]["projection_target_count"] == 8
        assert "structure_prior_diagnostics" not in report.projections[direction]
    assert (Path(report.output_dir) / "evaluation_report.json").is_file()
    if rms_enabled:
        rms = report.generation_protocol["projection_rms"]
        tracker = checkpoint_projection_rms(checkpoint, config, weights=weights)
        assert rms["weights"] == weights and rms["frozen"] and rms["source"] == "checkpoint"
        assert rms["scales"] == tracker.scales()
        for source, target in (("cat", "dog"), ("dog", "cat")):
            direction = report.projections[f"{source}_to_{target}"]
            assert direction["conditional_distance_scales"] == {
                "query_source": rms["scales"][source], "projection_target": rms["scales"][target]}
        # An old standalone YAML must not silently reinstate query-batch RMS.
        eval_config.pop("projection_rms")
        eval_path.write_text(yaml.safe_dump(eval_config))
        with pytest.raises(ValueError, match="match the training projection_rms"):
            evaluation.run_stage1b_evaluation(tmp_path / "self_eval_training.yaml", eval_path,
                                              checkpoint_path=checkpoint_path, weights=weights)
        eval_config["projection_rms"] = dict(config["projection_rms"])
    eval_config["infoot"]["cross_cost_source"] = "dino"
    eval_path.write_text(yaml.safe_dump(eval_config))
    with pytest.raises(ValueError, match="match the training infoot.cross_cost_source"):
        evaluation.run_stage1b_evaluation(tmp_path / "self_eval_training.yaml", eval_path,
                                          checkpoint_path=checkpoint_path, weights="raw")
