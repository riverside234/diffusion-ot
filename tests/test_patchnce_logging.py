"""Patch-level retrieval must not be labeled as global source-code retrieval."""
from copy import deepcopy
from pathlib import Path

import yaml

from diffusion_ot.training.stage1b_logging import Stage1BLogFormatter


def _recipe():
    path = Path(__file__).resolve().parents[1] / "configs/stage1b_infoot/self_supervised_sit_b2.yaml"
    config = yaml.safe_load(path.read_text())
    config["decoded_translation"].update(
        objective="patchnce", source_contrastive_weight=0.,
        patchnce={"weight": .15, "temperature": .2, "num_patches": 64, "layers": [0, 1, 2]})
    return config


def test_patchnce_logs_keep_spatial_diagnostics_and_remove_stale_global_objective():
    formatter = Stage1BLogFormatter(_recipe())
    stale = {"source_contrastive_loss": 3., "source_contrastive_weight": .15,
             "source_contrastive_effective_weight": .1, "source_contrastive_readout": "target",
             "source_contrastive_gradient_routing": "old", "source_negative_bank_ids": {"cat": ["a"]}}
    spatial = {"patchnce_loss": 0., "patchnce_weight": .15, "patchnce_effective_weight": 0.,
               "patchnce_readout": "target", "patchnce_gradient_routing": "live_target_detached_source",
               "patchnce": {"retrieval_top1": .5, "num_patches": 64},
               "source_correspondence": {"edge_correlation": None}, "effective_weighted_loss": 0.}
    row = {"decoded_translation": {**stale, **spatial, "cat_to_dog": {**stale, **spatial}},
           "window_mean": {"source_contrastive_loss": 3., "source_contrastive_active": 1.,
                           "patchnce_loss": None, "patchnce_active": 0.},
           "native_reconstruction": {"cat": {"pixel_mse": .1}},
           "solver_health": {"outer_delta_to_tolerance": .2},
           "conditional_projection": {"cat_to_dog": {"query_information_nats": .3}},
           "gradient_conflicts": {"groups": {"encoder.all": {
               "infoot_vs_translation": {"cosine": -.3, "valid": True}}}},
           "pcgrad": {"tasks": ["reconstruction", "transport", "protection", "patchnce"]}}
    original = deepcopy(row)
    result = formatter.format(row)
    assert row == original and formatter.format(result) == result
    assert result["decoded_translation"] == {**spatial, "cat_to_dog": spatial}
    assert result["window_mean"] == {"patchnce_loss": None, "patchnce_active": 0.}
    assert set(result["enabled_losses"]) == {
        "cat_reconstruction", "dog_reconstruction", "infoot_alignment", "matching_variance",
        "matching_covariance", "patchnce"}
    for key in ("native_reconstruction", "solver_health", "conditional_projection", "gradient_conflicts", "pcgrad"):
        assert result[key] == row[key]


def test_global_source_recipe_does_not_keep_stale_patchnce_metrics():
    config = _recipe()
    config["decoded_translation"].pop("objective")  # Existing configs retain their original semantics.
    config["decoded_translation"].pop("patchnce")
    config["decoded_translation"]["source_contrastive_weight"] = .15
    result = Stage1BLogFormatter(config).format({
        "decoded_translation": {"source_contrastive_loss": 0., "patchnce_loss": 2.,
                                "cat_to_dog": {"patchnce": {"retrieval_top1": 1.}}},
        "window_mean": {"source_contrastive_loss": 0., "patchnce_loss": 2., "patchnce_active": 1.}})
    assert result["decoded_translation"] == {"source_contrastive_loss": 0.}
    assert result["window_mean"] == {"source_contrastive_loss": 0.}
    assert "source_contrastive" in result["enabled_losses"] and "patchnce" not in result["enabled_losses"]


def test_zero_patch_weight_is_not_reported_as_an_enabled_loss():
    config = _recipe()
    config["decoded_translation"]["patchnce"]["weight"] = 0.
    assert "patchnce" not in Stage1BLogFormatter(config).format({})["enabled_losses"]
