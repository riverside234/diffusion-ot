"""Full InfoOT + relative loss in the real Stage 1B loop, with tiny CPU models."""
from pathlib import Path

import pytest
import torch
import yaml

from diffusion_ot.losses.infoot_alignment import infoot_alignment_options
from test_stage1b_extensions import experiment, assert_finite_numbers
from test_stage1b_self_supervised import own_encoder_run, assert_no_external_metrics
from test_stage1b_patchnce import patchnce_training, patchnce_recipe


def full_recipe(config, objective="patchnce", code_mlp=False):
    patchnce_recipe(config)
    if objective == "patchnce":
        config["decoded_translation"]["patchnce"].update(sampler="mlp_sample", projection_dim=16)
    else:
        config["decoded_translation"].pop("patchnce")
        config["decoded_translation"].update(objective="source_infonce", source_contrastive_weight=.05)
        if code_mlp:
            config["decoded_translation"]["source_contrastive_projector"] = {
                "kind": "mlp", "projection_dim": 16, "lr": 2e-4, "grad_clip_norm": 1.}
        config["matching_regularization"].update(std_target=.8, variance_weight=.1, covariance_weight=.03)
    config["projection_rms"] = {"mode": "reference_ema", "decay": .99, "eps": 1e-8}
    config["infoot"]["feature_objective"] = "full"
    config["loss_weights"].update(infoot_relative=.01, alignment_warmup_steps=4)


@pytest.mark.parametrize("pcgrad", [False, True])
@pytest.mark.parametrize("objective,code_mlp", [("patchnce", False), ("source_infonce", False), ("source_infonce", True)])
def test_full_training_validation_gradients_and_resume(patchnce_training, pcgrad, objective, code_mlp):
    run, _, _, _ = patchnce_training
    translation_task = "patchnce" if objective == "patchnce" else "source_contrastive"
    def recipe(cfg):
        full_recipe(cfg, objective, code_mlp)
        cfg["pcgrad"]["enabled"] = pcgrad
    first, logs = run("full", steps=1, modify=recipe)
    resumed, resumed_logs = run("full", resume=True, modify=recipe)
    assert first["step"] == 1 and resumed["step"] == 2
    assert resumed["decoded_objective"] == objective
    assert ("patch_projectors" in resumed) == (objective == "patchnce")
    assert ("code_projectors" in resumed) == code_mlp
    if code_mlp:
        assert resumed["code_projector_ema_state"]["num_updates"] == 2
        assert set(resumed["code_projector_ema"]) == {"cat", "dog"}
        group = next(g for g in resumed["optimizer"]["param_groups"] if g.get("name") == "code_projectors")
        assert group["lr"] == 2e-4
        for domain in ("cat", "dog"):
            assert any(not torch.equal(value, first["code_projectors"][domain][key])
                       for key, value in resumed["code_projectors"][domain].items())
    assert resumed["config"]["loss_weights"]["infoot_relative"] == .01
    assert all(s["num_updates"] == 2 for s in resumed["projection_rms_state"].values())
    assert [r["step"] for r in resumed_logs["train"]] == [1, 2]
    for row in resumed_logs["train"] + logs["validation"]:
        assert_finite_numbers(row)
        assert_no_external_metrics(row)
        assert row["infoot_feature_objective"] == "full"
        assert {"infoot_alignment", "infoot_relative", "matching_variance", "matching_covariance", translation_task} <= set(row["enabled_losses"])
        assert ("source_contrastive" if objective == "patchnce" else "patchnce") not in row["enabled_losses"]
        assert row["infoot_feature_loss"] == pytest.approx(sum(row[f"infoot_{k}_loss"] for k in ("cost", "mi", "entropy")))
        assert row["weighted_infoot_alignment_loss"] == pytest.approx(sum(row[f"weighted_infoot_{k}_loss"] for k in ("cost", "mi", "entropy")))
        assert row["weighted_infoot_relative_loss"] == pytest.approx(row["infoot_relative_weight"] * row["infoot_relative_loss"])
        is_train = row.get("event") == "train"
        suffix = "reconstruction_loss" if is_train else "raw_reconstruction"
        decoded = row["decoded_translation"]
        assert decoded["objective"] == objective
        assert decoded[f"{translation_task}_loss"] > 0
        for source, target in (("cat", "dog"), ("dog", "cat")):
            assert decoded[f"{source}_to_{target}"]["readout_domain"] == target
            assert translation_task in decoded[f"{source}_to_{target}"]
            if code_mlp:
                retrieval = decoded[f"{source}_to_{target}"][translation_task]
                assert retrieval["comparison_space"] == "global_mlp"
                assert retrieval["query_projector_domain"] == target
                assert retrieval["key_projector_domain"] == source
                assert retrieval["key_projector_detached"]
        total = sum(row[f"{d}_{suffix}"] for d in ("cat", "dog"))
        total += row["weighted_infoot_alignment_loss"] + row["weighted_infoot_relative_loss"]
        total += row["matching_regularization_loss"] + decoded.get("effective_weighted_loss", decoded["weighted_loss"])
        assert row["loss"] == pytest.approx(total, abs=2e-6)
        assert row["infoot_relative_weight"] == pytest.approx(.01 * min(row["step"] / 4, 1) if is_train else .01)
        if is_train:
            if code_mlp:
                assert row["weighted_decoded_code_projector_gradient_norm"] > 0
                assert row["code_projector_gradient_norm_pre_clip"] > 0
                assert row["code_projector_learning_rate"] == 2e-4
            for group in ("encoder", "generator", "matching_head"):
                assert row[f"weighted_decoded_{group}_gradient_norm"] > 0
            assert row["infoot_feature_loss"] == pytest.approx(row["infoot_objective"], abs=2e-6)
            assert row["infoot_relative_loss"] == pytest.approx(-row["encoder_cost_gain_over_independent"], abs=1e-5)
            for part in ("cost", "mi", "relative"):
                assert row[f"weighted_infoot_{part}_gradient_norm"] > 0
                assert row[f"weighted_infoot_{part}_matching_head_gradient_norm"] > 0
            if pcgrad:
                assert ("patch_projector" in row["pcgrad"]["groups"]) == (objective == "patchnce")
                assert ("code_projector" in row["pcgrad"]["groups"]) == code_mlp
                if code_mlp:
                    assert row["pcgrad"]["groups"]["code_projector"]["active_tasks"] == ["source_contrastive"]
                for group in ("encoder", "matching_head", "generator"):
                    assert translation_task in row["pcgrad"]["groups"][group]["active_tasks"]
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
