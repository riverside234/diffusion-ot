from pathlib import Path
import runpy

import pytest

from test_stage1b_extensions import experiment, minimal_pcgrad_recipe, assert_tensor_tree_equal


summary_tools = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/summarize_infoot_training.py"))


def test_kl_diagnostics_integrate_without_changing_training(experiment, monkeypatch):
    run, _, _ = experiment
    measured, logs = run("kl_diagnostics", modify=minimal_pcgrad_recipe)
    for row in logs["train"]:
        assert row["weighted_conditional_structure_loss"] == pytest.approx(
            row["conditional_structure_weight"] * row["conditional_structure_loss"])
        assert "weighted_conditional_structure_loss" in row["window_mean"]
        pairs = row["gradient_conflicts"]["groups"]["encoder.all"]
        assert pairs["conditional_vs_translation"]["valid"]
        assert pairs["conditional_vs_reconstruction"]["valid"]
        assert pairs["conditional_vs_infoot"]["valid"]
        assert "conditional_vs_protection" in pairs
    for row in logs["validation"]:
        assert row["weighted_conditional_structure_loss"] == pytest.approx(.05 * row["conditional_structure_loss"])
        for direction in ("cat_to_dog", "dog_to_cat"):
            metrics = row["conditional_structure"][direction]
            assert metrics["query_count"] == 2 and metrics["reference_targets"] == 6
            assert metrics["kl_gain_over_query_independent"] == pytest.approx(metrics["query_independent_kl"] - metrics["kl"])
        assert row["projection_probe"]["teacher_temperature"] > 0
        assert row["projection_probe"]["structure_cost_scale"] > 0

    def no_new_diagnostics(config):
        minimal_pcgrad_recipe(config)
        config["train"]["gradient_conflicts"] = False

    with monkeypatch.context() as patch:
        patch.setattr("diffusion_ot.losses.conditional_structure.conditional_query_diagnostics", lambda *a: {})
        control, _ = run("without_kl_diagnostics", modify=no_new_diagnostics)
    for key in ("encoders", "matching_heads", "generators", "optimizer", "decoded_discriminators",
                "decoded_discriminator_optimizer", "decoded_noise_states", "rng_state"):
        assert_tensor_tree_equal(measured[key], control[key])


def test_kl_report_preserves_missing_data_and_separates_gradient_phases():
    report = summary_tools["conditional_kl_report"]({"validation": [], "training": []})
    assert "No isolated KL gradient pairs recorded" in report
    assert "No fixed validation supplied" in report
    rows = []
    for phase, cosine in (("warmup", 1.), ("full_weight", -.5), ("full_weight", None)):
        rows.append(summary_tools["training_row"]({"step": len(rows), "gradient_conflicts": {
            "phase": phase, "groups": {"encoder.all": {"conditional_vs_translation": {
                "cosine": cosine, "valid": cosine is not None, "first_to_second_norm_ratio": 2.}}}}}))
    validation = summary_tools["validation_row"]({"step": 3, "conditional_structure": {
        "cat_to_dog": {"kl": 1., "kl_gain_over_query_independent": -.25}},
        "projection_probe": {"outer_converged": False}})
    report = summary_tools["conditional_kl_report"]({"validation": [validation], "training": rows})
    assert "-0.25000" in report and "False" in report and "n/a" in report
    assert "full_weight | encoder.all | conditional_vs_translation | 1 / 2 | -0.50000 | 1.00000" in report
    assert "warmup | encoder.all | conditional_vs_translation | 1 / 1 | 1.00000 | 0.00000" in report
    assert "not a model trained without KL" in report
