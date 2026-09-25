"""Full InfoOT + relative loss in the real Stage 1B loop, with tiny CPU models."""
from pathlib import Path

import pytest
import yaml

from diffusion_ot.losses.infoot_alignment import infoot_alignment_options
from test_stage1b_extensions import experiment, assert_finite_numbers
from test_stage1b_self_supervised import own_encoder_run, assert_no_external_metrics
from test_stage1b_patchnce import patchnce_training, patchnce_recipe


def full_recipe(config):
    patchnce_recipe(config)
    config["decoded_translation"]["patchnce"].update(sampler="mlp_sample", projection_dim=16)
    config["projection_rms"] = {"mode": "reference_ema", "decay": .99, "eps": 1e-8}
    config["infoot"]["feature_objective"] = "full"
    config["loss_weights"].update(infoot_relative=.01, alignment_warmup_steps=4)


@pytest.mark.parametrize("pcgrad", [False, True])
def test_full_training_validation_gradients_and_resume(patchnce_training, pcgrad):
    run, _, _, _ = patchnce_training
    def recipe(cfg):
        full_recipe(cfg)
        cfg["pcgrad"]["enabled"] = pcgrad
    first, logs = run("full", steps=1, modify=recipe)
    resumed, resumed_logs = run("full", resume=True, modify=recipe)
    assert first["step"] == 1 and resumed["step"] == 2
    assert resumed["config"]["loss_weights"]["infoot_relative"] == .01
    assert all(s["num_updates"] == 2 for s in resumed["projection_rms_state"].values())
    assert [r["step"] for r in resumed_logs["train"]] == [1, 2]
    for row in resumed_logs["train"] + logs["validation"]:
        assert_finite_numbers(row)
        assert_no_external_metrics(row)
        assert row["infoot_feature_objective"] == "full"
        assert {"infoot_alignment", "infoot_relative", "matching_variance", "matching_covariance", "patchnce"} <= set(row["enabled_losses"])
        assert row["infoot_feature_loss"] == pytest.approx(sum(row[f"infoot_{k}_loss"] for k in ("cost", "mi", "entropy")))
        assert row["weighted_infoot_alignment_loss"] == pytest.approx(sum(row[f"weighted_infoot_{k}_loss"] for k in ("cost", "mi", "entropy")))
        assert row["weighted_infoot_relative_loss"] == pytest.approx(row["infoot_relative_weight"] * row["infoot_relative_loss"])
        is_train = row.get("event") == "train"
        suffix = "reconstruction_loss" if is_train else "raw_reconstruction"
        decoded = row["decoded_translation"]
        total = sum(row[f"{d}_{suffix}"] for d in ("cat", "dog"))
        total += row["weighted_infoot_alignment_loss"] + row["weighted_infoot_relative_loss"]
        total += row["matching_regularization_loss"] + decoded.get("effective_weighted_loss", decoded["weighted_loss"])
        assert row["loss"] == pytest.approx(total, abs=2e-6)
        assert row["infoot_relative_weight"] == pytest.approx(.01 * min(row["step"] / 4, 1) if is_train else .01)
        if is_train:
            assert row["infoot_feature_loss"] == pytest.approx(row["infoot_objective"], abs=2e-6)
            assert row["infoot_relative_loss"] == pytest.approx(-row["encoder_cost_gain_over_independent"], abs=1e-5)
            for part in ("cost", "mi", "relative"):
                assert row[f"weighted_infoot_{part}_gradient_norm"] > 0
                assert row[f"weighted_infoot_{part}_matching_head_gradient_norm"] > 0
            if pcgrad:
                for group in ("encoder", "matching_head"):
                    assert "transport" in row["pcgrad"]["groups"][group]["active_tasks"]
                assert "transport" not in row["pcgrad"]["groups"]["generator"]["active_tasks"]
    def changed(cfg):
        recipe(cfg)
        cfg["loss_weights"]["infoot_relative"] = .02
    with pytest.raises(ValueError, match="InfoOT neural objectives"):
        run("full", resume=True, modify=changed)


def test_full_recipe_keeps_original_protection_and_a_separate_output():
    root = Path(__file__).resolve().parents[1]
    folder = root / "configs" / "stage1b_infoot"
    load = lambda path: yaml.safe_load(path.read_text(encoding="utf-8"))
    old = load(folder / "self_supervised_patchnce_mlp_rms_ema_sit_b2.yaml")
    new = load(folder / "self_supervised_patchnce_mlp_full_sit_b2.yaml")
    assert infoot_alignment_options(old).feature_objective == "mi"
    assert infoot_alignment_options(new).feature_objective == "full"
    assert new["loss_weights"]["infoot_relative"] == .01
    for key in ("stage1a", "matching", "matching_regularization", "decoded_translation", "pcgrad", "projection_rms", "train", "data", "generator_adaptation"):
        assert new[key] == old[key]
    assert new["output_dir"] != old["output_dir"]
    evaluation = load(root / new["quick_evaluation"]["config"])
    assert evaluation["projection_rms"] == new["projection_rms"]
    assert evaluation["matching"]["bandwidth_multiplier"] == new["matching"]["bandwidth_multiplier"] == .55
    for key, value in evaluation["infoot"].items():
        assert value == new["infoot"][key]
