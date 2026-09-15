import json
from pathlib import Path
import runpy

import pytest


summary_tools = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/summarize_infoot_training.py"))


def test_training_only_paste_keeps_missing_validation_and_missing_metrics_explicit(tmp_path):
    log = tmp_path / "paste.txt"
    log.write_text(json.dumps({"step": 200, "event": "train", "window_mean": {
        "cat_reconstruction_loss": .4, "dog_reconstruction_loss": .5,
    }}) + "\n", encoding="utf-8")
    result = summary_tools["summarize"](None, 1000, train_log=log)
    assert result["validation_status"] == "not_supplied"
    assert result["validation"] == []
    assert result["sources"]["validation"] is None
    assert result["training"][0]["window_reconstruction"] == pytest.approx(.9)
    assert result["training"][0]["cat_matching_variance"] is None
    assert result["training"][0]["matching_regularization_loss"] is None
    assert "cannot be inferred" in summary_tools["table"](result)
    assert "n/a" in summary_tools["training_table"](result)
    with pytest.raises(FileNotFoundError):
        summary_tools["summarize"](None, 1000, train_log=log, validation_log=tmp_path / "missing.jsonl")


def test_resume_duplicates_and_separate_fixed_validation_are_not_extra_replicates(tmp_path):
    (tmp_path / "train.jsonl").write_text('\n'.join(json.dumps(row) for row in [
        {"step": 1}, {"step": 200, "window_mean": {"loss": 5}},
        {"step": 200, "window_mean": {"loss": 7}},
    ]), encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text(json.dumps({"step": 200,
        "cat_raw_reconstruction": 1.1, "cat_stage1a_reconstruction": 1.0}), encoding="utf-8")
    result = summary_tools["summarize"](tmp_path, 1000)
    assert result["validation_status"] == "available"
    assert result["sources"]["training"]["duplicate_steps"] == 1
    assert len(result["training"]) == 2
    assert result["training"][-1]["window_loss"] == 7
    assert result["validation"][0]["cat_rec_drift_pct"] == pytest.approx(10)


def test_decoded_validation_preserves_failed_and_missing_solver_status(tmp_path):
    (tmp_path / "train.jsonl").write_text(json.dumps({"step": 200}), encoding="utf-8")
    rows = [{"step": 0, "projection_probe": {"outer_converged": False, "sinkhorn_converged": True,
              "iterations": 100, "matching_feature_variance": {"cat": .15}},
             "cat_null_reconstruction": .6, "cat_stage1a_null_reconstruction": .5,
             "decoded_translation": {"structure_loss": .2, "cat_to_dog": {"structure_loss": .1}}},
            {"step": 200}]
    (tmp_path / "validation.jsonl").write_text('\n'.join(map(json.dumps, rows)), encoding="utf-8")
    result = summary_tools["summarize"](tmp_path, 1000)
    assert result["validation_steps_failed_outer_convergence"] == [0]
    assert result["validation_steps_missing_outer_convergence"] == [200]
    assert result["validation_uses_same_ids"] is None
    assert result["validation"][0]["cat_null_rec_drift_pct"] == pytest.approx(20)
    assert result["validation"][0]["cat_matching_variance"] == .15
    assert result["validation"][0]["cat_to_dog.decoded_structure"] == .1
    assert result["validation"][0]["decoded_outer_converged"] is None
    assert "False" in summary_tools["table"](result)
    assert "n/a" in summary_tools["table"](result)


def test_protection_diagnostics_survive_training_and_validation_summary():
    row = {"step": 1, "matching_regularization_loss": .007,
           "matching_regularization": {"variance_loss": .3, "covariance_loss": 1,
               "weighted_variance_loss": .006, "weighted_covariance_loss": .001,
               "cat": {"fraction_below_std_target": .5, "scaled_std_min": .2}},
           "window_mean": {"matching_regularization_loss": .008},
           "matching_regularization_to_reconstruction_encoder_gradient_ratio": .4}
    for kind in ("training_row", "validation_row"):
        result = summary_tools[kind](row)
        assert result["matching_regularization_loss"] == .007
        assert result["matching_regularization.weighted_covariance_loss"] == .001
        assert result["matching_regularization.cat.fraction_below_std_target"] == .5
        assert result["matching_regularization.dog.scaled_std_min"] is None
    training = summary_tools["training_row"](row)
    assert training["window_matching_regularization_loss"] == .008
    assert training["matching_regularization_to_reconstruction_encoder_gradient_ratio"] == .4


def test_geometry_and_component_gradients_preserve_missing_values():
    geometry = {"cat": {"matching_covariance_effective_rank": 12, "raw_normalized_variance": .95,
                       "raw_normalized_covariance_participation_rank": 40}}
    for kind, field in (("training_row", {"feature_geometry": geometry}),
                        ("validation_row", {"projection_probe": {"feature_geometry": geometry}})):
        result = summary_tools[kind]({"step": 500, **field})
        assert result["cat_matching_covariance_effective_rank"] == 12
        assert result["cat_raw_normalized_covariance_participation_rank"] == 40
        assert result["dog_matching_covariance_effective_rank"] is None
    row = summary_tools["training_row"]({"step": 500, "semantic_neighborhood_geometry": "rms_distance",
                                        "weighted_matching_covariance_matching_head_gradient_norm": .0001})
    assert row["semantic_neighborhood_geometry"] == "rms_distance"
    assert row["weighted_matching_covariance_matching_head_gradient_norm"] == .0001
    assert row["weighted_matching_variance_matching_head_gradient_norm"] is None
