"""v6 integration: training, validation-only texture, private RNG resume, configs."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from test_stage1b_extensions import experiment, assert_finite_numbers, assert_tensor_tree_equal
from test_stage1b_spatial_correlative import recipe, spatial_run

ROOT = Path(__file__).resolve().parents[1]


def v6_recipe(cfg):
    recipe(cfg)
    cfg["source_aware_selection"] = dict(enabled=True, spatial_weight=.25, appearance_weight=.25, appearance_size=4)
    cfg["conditional_projection"]["bandwidth_multiplier"] = .25
    cfg["decoded_translation"].update(color_histogram=dict(weight=0.), target_patch_swd=dict(weight=0.),
        source_lab_swd=dict(weight=.04, sizes=[8, 4], patch_size=3, directions=8))
    cfg["self_supervised_diagnostics"]["target_patch_swd"] = dict(enabled=True, sizes=[8, 4],
        patch_size=3, patches_per_image=4, directions=8, scale_floor=.01)


def test_v6_training_logs_gradient_routing_and_private_rng_resume(spatial_run, tmp_path, monkeypatch):
    import diffusion_ot.losses.translation_image as images
    import diffusion_ot.training.train_joint_infoot as trainer
    from diffusion_ot.evaluation.stage1b_eval import _validate_self_supervised_checkpoint
    run, _, _ = spatial_run
    real_swd = images.target_patch_swd
    texture_calls = []
    def diagnostic_only(*args, **kwargs):
        assert not torch.is_grad_enabled(), "Disabled texture loss entered the training graph"
        texture_calls.append(1)
        return real_swd(*args, **kwargs)
    monkeypatch.setattr(images, "target_patch_swd", diagnostic_only)
    full, logs = run("v6_full", modify=v6_recipe)
    assert texture_calls
    _validate_self_supervised_checkpoint(full["config"], full)
    initial = trainer._load_checkpoint(tmp_path / "v6_full/checkpoints/step_000000.pt")
    assert initial["source_aware_selection_state"] == full["source_aware_selection_state"]
    for domain, ids in full["source_aware_selection_state"]["calibration"]["sample_ids"].items():
        assert all(i.startswith(domain + "_train_") for i in ids)
    for row in logs["train"] + logs["validation"]:
        assert_finite_numbers(row)
        assert row["log_schema_version"] == 5
        assert "source_lab_swd" in row["enabled_losses"]
        assert not {"target_patch_swd", "color_histogram"} & set(row["enabled_losses"])
        decoded = row["decoded_translation"]
        assert set(decoded["image_losses"]) == {"coarse_rgb", "local_layout", "source_lab_swd"}
        assert decoded["source_contrastive_projection_gradient_scale"] == .1
    for row in logs["train"]:
        assert row["weighted_source_lab_swd_generator_gradient_norm"] > 0
        assert row["weighted_source_lab_swd_matching_head_gradient_norm"] > 0
        assert "source_lab_swd_vs_source_contrastive" in row["gradient_conflicts"]["groups"]["generator.all"]
        assert "target_patch_swd_loss" not in row
    for row in logs["validation"]:
        for direction in ("cat_to_dog", "dog_to_cat"):
            diagnostic = row["decoded_translation"][direction]["diagnostics"]["target_patch_swd"]
            assert diagnostic["training_weight"] == 0 and diagnostic["real_to_real_distance"] > 0
            assert not set(diagnostic["reference_ids"]) & set(diagnostic["baseline_ids"])
            selection = row["decoded_translation"]["source_aware_selection"][direction]
            assert selection["mean_weight_l1_change"] > 0
    run("v6_resume", steps=1, modify=v6_recipe)
    resumed, _ = run("v6_resume", resume=True, modify=v6_recipe)
    assert resumed["step"] == 2
    # Existing shuffled DataLoader does not restore its permutation/cursor.
    # Check exact private RNG and calibration continuity, not model equality.
    for key in ("decoded_noise_states", "decoded_image_sampling_states", "rng_state", "source_aware_selection_state"):
        assert_tensor_tree_equal(full[key], resumed[key])
    def changed(cfg):
        v6_recipe(cfg)
        cfg["source_aware_selection"]["appearance_weight"] = .5
    with pytest.raises(ValueError, match="disagree"):
        run("v6_resume", steps=3, resume=True, modify=changed)
    def changed_bandwidth(cfg):
        v6_recipe(cfg)
        cfg["conditional_projection"]["bandwidth_multiplier"] = .1
    with pytest.raises(ValueError, match="projection bandwidth"):
        run("v6_resume", steps=3, resume=True, modify=changed_bandwidth)


def test_v6_and_bandwidth_control_keep_other_v45_settings_fixed():
    for stage in ("stage1b_infoot", "stage1b_eval"):
        def load(name):
            return yaml.safe_load((ROOT / "configs" / stage / f"self_supervised_infonce_{name}_sit_b2.yaml").read_text())
        baseline, control, v6 = (load(name) for name in ("v4_5", "v6_bandwidth_control", "v6"))
        expected = deepcopy(baseline)
        expected["output_dir"] = control["output_dir"]
        if stage == "stage1b_infoot":
            expected["quick_evaluation"] = control["quick_evaluation"]
            expected["conditional_projection"]["bandwidth_multiplier"] = .25
        else:
            expected["matching"]["projection_bandwidth_multiplier"] = .25
            expected["projection_audit"] = {"enabled": True}
            expected["visualization"].update(n_neighbors=15, min_dist=.1)
        assert control == expected
        expected["output_dir"] = v6["output_dir"]
        expected["source_aware_selection"] = v6["source_aware_selection"]
        if stage == "stage1b_infoot":
            expected["quick_evaluation"] = v6["quick_evaluation"]
            expected["train"]["max_steps"] = 8000  # The supplied v6 run extends the earlier 2,500-step recipe.
            expected["decoded_translation"]["color_histogram"]["weight"] = 0.
            expected["decoded_translation"]["target_patch_swd"]["weight"] = 0.
            expected["decoded_translation"]["source_lab_swd"] = v6["decoded_translation"]["source_lab_swd"]
            expected["self_supervised_diagnostics"]["target_patch_swd"] = v6["self_supervised_diagnostics"]["target_patch_swd"]
            assert v6["decoded_translation"]["source_lab_swd"]["weight"] == .04
        assert v6 == expected
