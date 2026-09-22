from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from diffusion_ot.training.stage1b_logging import Stage1BLogFormatter
from test_stage1b_extensions import experiment, minimal_pcgrad_recipe, assert_tensor_tree_equal


@pytest.fixture
def active_config():
    return yaml.safe_load((Path(__file__).resolve().parents[1] /
        "configs/stage1b_infoot/structure_decoder_sit_b2.yaml").read_text())


def test_active_recipe_omits_disabled_losses_but_preserves_real_zeros_and_diagnostics(active_config):
    formatter = Stage1BLogFormatter(active_config)
    disabled = dict.fromkeys(("cat_anchor_loss", "dog_anchor_loss", "latent_anchor_weight",
        "semantic_neighborhood_loss", "semantic_neighborhood_weight", "semantic_neighborhood_geometry",
        "projection_support_loss", "projection_support_weight", "null_preservation_loss",
        "conditioned_preservation_loss", "conditioned_preservation_samples", "code_consistency_loss",
        "weighted_null_preservation_generator_gradient_norm", "weighted_neighborhood_gradient_norm"), 1.23)
    # Nonzero stale values are still removed; real zeros and negative GAN scores stay.
    active = {"loss": 0., "conditional_structure_loss": 0., "conditional_structure_weight": 0.,
              "matching_regularization_loss": 0., "cat_reconstruction_loss": 0., "alignment_weight": 0.}
    image = {"structure_loss": .25, "structure_weight": 0., "structure_effective_weighted_loss": 0.,
        "perceptual_loss": 0., "perceptual_cosine_distance": .2, "adversarial_loss": -.1,
        "color_histogram_loss": 0., "seed": 123, "reference_ids": {"cat": ["a"]},
        "query_ids": {"cat": ["b"]}, "solver": {"outer_converged": False},
        "cat_to_dog": {"structure_loss": .3, "perceptual_loss": 0., "perceptual_cosine_distance": .2}}
    row = {**disabled, **active, "gradient_guard": {}, "window_mean": {**disabled, **active},
        "decoded_translation": image, "cat_stage1a_reconstruction": .5,
        "cat_current_encoder_stage1a_generator_reconstruction": .6,
        "cat_raw_reconstruction": .4, "cat_null_reconstruction": .45,
        "projection_probe": {"sample_ids": {"cat": {"query": ["b"]}}, "outer_converged": False},
        "feature_geometry": {"cat": {"matching_covariance_effective_rank": 28.}}}
    original = deepcopy(row)
    cleaned = formatter.format(row)
    assert row == original
    assert formatter.format(cleaned) == cleaned
    assert not set(disabled).intersection(cleaned)
    assert not set(disabled).intersection(cleaned["window_mean"])
    assert all(cleaned[k] == v for k, v in active.items())
    assert "gradient_guard" not in cleaned and not any("stage1a" in k for k in cleaned)
    for key in ("cat_raw_reconstruction", "cat_null_reconstruction", "feature_geometry", "projection_probe"):
        assert cleaned[key] == row[key]
    decoded = cleaned["decoded_translation"]
    assert decoded["adversarial_loss"] == -.1 and decoded["perceptual_loss"] == 0.
    assert decoded["diagnostics"] == {"structure_cosine_distance": .25, "perceptual_cosine_distance": .2}
    assert decoded["cat_to_dog"]["diagnostics"]["structure_cosine_distance"] == .3
    for key in ("seed", "query_ids", "reference_ids", "solver"):
        assert decoded[key] == image[key]
    assert set(cleaned["enabled_losses"]) == {"cat_reconstruction", "dog_reconstruction", "infoot_alignment",
        "conditional_structure", "matching_variance", "matching_covariance", "perceptual", "adversarial", "color_histogram"}


def test_legacy_enabled_losses_and_guard_remain_visible(active_config):
    active_config["loss_weights"].update(latent_anchor="0.01", semantic_neighborhood=.1, projection_support=.1)
    active_config["projection_support"]["enabled"] = True
    active_config["generator_adaptation"].update(null_preservation_weight=.1, conditioned_preservation_weight=.1)
    active_config["decoded_translation"].update(structure_weight=.1, structure_contrastive_weight=.1, code_consistency_weight=.1)
    active_config["matching_contrastive"] = {"enabled": True, "neighborhood_weight": .1, "conditional_weight": .1}
    active_config["gradient_guard"]["enabled"] = True
    row = {"cat_anchor_loss": 0., "null_preservation_loss": 0., "conditioned_preservation_loss": 0.,
        "semantic_neighborhood_loss": 0., "projection_support_loss": 0., "code_consistency_loss": 0.,
        "cat_stage1a_reconstruction": .4, "gradient_guard": {"cat": {"cosine": 0.}},
        "matching_contrastive": {"neighborhood_loss": 0., "conditional_loss": 0.},
        "decoded_translation": {"structure_loss": .3, "structure_weight": .1, "code_consistency_loss": 0.}}
    result = Stage1BLogFormatter(active_config).format(row)
    assert all(result[k] == v for k, v in row.items())


def test_weight_zero_color_and_covariance_are_diagnostics(active_config):
    active_config["decoded_translation"]["color_histogram"]["weight"] = 0.
    active_config["matching_regularization"]["covariance_weight"] = 0.
    active_config["matching_contrastive"] = {"enabled": True, "neighborhood_weight": 0., "conditional_weight": .01}
    row = {"matching_covariance_loss": .8, "weighted_matching_covariance_encoder_gradient_norm": 0.,
        "matching_regularization": {"covariance_loss": .8, "weighted_covariance_loss": 0., "covariance_weight": 0.,
            "cat": {"covariance_loss": .7, "relative_covariance_energy": .6}},
        "matching_contrastive": {"neighborhood_loss": 0., "weighted_neighborhood_loss": 0., "neighborhood": {},
            "neighborhood_weight": 0., "conditional_loss": .3},
        "decoded_translation": {"color_histogram_loss": .4, "color_histogram_weight": 0.,
            "color_histogram_effective_weighted_loss": 0., "color_histogram_protocol": "test",
            "cat_to_dog": {"per_image_color_histogram_loss": [.4]}}}
    result = Stage1BLogFormatter(active_config).format(row)
    assert "matching_covariance_loss" not in result
    assert result["matching_regularization"]["diagnostics"]["mean_squared_offdiagonal_covariance"] == .8
    assert result["matching_regularization"]["cat"]["relative_covariance_energy"] == .6
    assert result["matching_contrastive"] == {"conditional_loss": .3}
    decoded = result["decoded_translation"]
    assert "color_histogram_loss" not in decoded and "color_histogram_weight" not in decoded
    assert decoded["diagnostics"]["color_histogram_distance"] == .4
    assert decoded["cat_to_dog"]["diagnostics"]["per_image_color_histogram_distance"] == [.4]
    assert decoded["color_histogram_protocol"] == "test"


def test_disabled_conflicts_removed_without_hiding_valid_zero_or_invalid_measurements(active_config):
    pairs = {name: {"cosine": None, "valid": False} for name in (
        "perceptual_vs_structure", "code_vs_decoded", "code_vs_reconstruction", "dino_vs_adversarial",
        "perceptual_vs_adversarial", "color_vs_perceptual", "translation_vs_reconstruction")}
    pairs["color_vs_perceptual"] = {"cosine": 0., "valid": True}
    row = {"gradient_conflicts": {"groups": {"generator.all": pairs}, "code_routing": "generator_only"}}
    result = Stage1BLogFormatter(active_config).format(row)["gradient_conflicts"]
    assert "code_routing" not in result
    assert set(result["groups"]["generator.all"]) == {"perceptual_vs_adversarial", "color_vs_perceptual", "translation_vs_reconstruction"}
    assert result["groups"]["generator.all"]["color_vs_perceptual"]["cosine"] == 0.
    assert result["groups"]["generator.all"]["perceptual_vs_adversarial"]["cosine"] is None


def test_log_cleanup_does_not_change_training_or_validation_numbers(experiment, monkeypatch):
    run, _, _ = experiment
    cleaned, logs = run("clean_logs", modify=minimal_pcgrad_recipe)
    with monkeypatch.context() as patch:
        patch.setattr(Stage1BLogFormatter, "format", lambda self, row: row)
        control, original_logs = run("original_logs", modify=minimal_pcgrad_recipe)
    for key in ("encoders", "matching_heads", "generators", "optimizer", "decoded_discriminators",
                "decoded_discriminator_optimizer", "decoded_noise_states", "rng_state"):
        assert_tensor_tree_equal(cleaned[key], control[key])
    for measured, original in zip(logs["validation"], original_logs["validation"]):
        assert measured["cat_raw_reconstruction"] == original["cat_raw_reconstruction"]
        assert measured["decoded_translation"]["query_ids"] == original["decoded_translation"]["query_ids"]
        assert measured["decoded_translation"]["perceptual_loss"] == original["decoded_translation"]["perceptual_loss"]
        assert measured["decoded_translation"]["diagnostics"]["structure_cosine_distance"] == original["decoded_translation"]["structure_loss"]
