"""Recipe/provenance and standalone evaluation checks for encoder-only Stage 1B."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.integrations.hf_snapshot import load_yaml_config
from diffusion_ot.training.decoded_translation import (
    validate_decoder_config,
    validate_flow_only_stage1a,
)


ROOT = Path(__file__).resolve().parents[1]


def _recipe():
    return load_yaml_config(ROOT / "configs/stage1b_infoot/self_supervised_sit_b2.yaml")


def _evaluation():
    return load_yaml_config(ROOT / "configs/stage1b_eval/self_supervised_sit_b2.yaml")


def test_self_supervised_recipe_is_valid_and_has_matching_teacher_free_evaluation():
    config, evaluation = _recipe(), _evaluation()
    validate_decoder_config(config)
    assert config["infoot"] == evaluation["infoot"]
    assert config["matching"]["bandwidth_multiplier"] == evaluation["matching"]["bandwidth_multiplier"]
    assert config["conditional_projection"]["bandwidth_multiplier"] == evaluation["matching"]["projection_bandwidth_multiplier"]
    assert config["decoded_translation"]["source_contrastive_readout"] == "target"
    assert config["decoded_translation"]["source_contrastive_weight"] > 0
    assert config["conditional_projection"]["query_samples_per_domain"] == config["decoded_translation"]["batch_size"]
    assert config["data"]["transport_batch_size"] > config["decoded_translation"]["batch_size"] + 1
    assert "semantic_prior" not in config and "semantic_prior" not in evaluation
    assert not evaluation["translation"]["decoded_structure_metrics"]
    assert "color_histogram" not in evaluation["translation"]
    assert "lpips" not in evaluation["reconstruction"]["metrics"]
    assert not evaluation.get("proxy_labels", {}).get("path")
    assert not evaluation.get("proxy_labels", {}).get("attributes")


def test_self_supervised_initializers_point_to_original_pdae_not_dino_recipes():
    config = _recipe()
    for domain in ("cat", "dog"):
        initialization = config["stage1a"][domain]
        recipe = load_yaml_config(ROOT / initialization["config"])
        validate_flow_only_stage1a(config, recipe)
        assert initialization["config"] == f"configs/stage1a_pdae/{domain}_sit_b2_lora.yaml"
        assert initialization["checkpoint"] == f"{recipe['output_dir']}/checkpoints/latest.pt"
        assert not recipe.get("refinement", {}).get("enabled", False)
        assert recipe["adapter"]["lora_rank"] == config["stage1a"]["require_attention_lora_rank"]
        assert recipe["adapter"]["lora_alpha"] == config["stage1a"]["require_attention_lora_alpha"]
        assert recipe["semantic_cfg"]["enabled"]


@pytest.mark.parametrize("section,key,value", [
    ("decoded_translation", "adversarial_weight", .01),
    ("decoded_translation", "perceptual_weight", .1),
    ("decoded_translation", "structure_weight", .1),
    ("decoded_translation", "structure_contrastive_weight", .01),
    ("decoded_translation", "code_consistency_weight", .01),
    ("decoded_translation", "color_histogram", {"weight": .02}),
    ("decoded_translation", "diffaugment", {"enabled": True, "policy": "translation"}),
    ("conditional_structure", "enabled", True),
    ("matching_contrastive", "enabled", True),
    ("loss_weights", "semantic_neighborhood", .01),
    ("loss_weights", "conditional_structure", .01),
    ("loss_weights", "latent_anchor", .01),
    ("generator_adaptation", "null_preservation_weight", .1),
    ("generator_adaptation", "conditioned_preservation_weight", .1),
    ("infoot", "cross_cost_source", "dino"),
    ("infoot", "cross_cost_weight", 0.),
    ("infoot", "variant", "plain"),
])
def test_self_supervised_recipe_rejects_external_or_incompatible_objectives(section, key, value):
    config = _recipe()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_decoder_config(config)


@pytest.mark.parametrize("origin", ["configured", "checkpoint_config", "checkpoint_training_state"])
def test_flow_only_provenance_guard_rejects_refined_initializer_metadata(origin):
    config = _recipe()
    recipe = {"refinement": {"enabled": False}}
    checkpoint = {"config": {"refinement": {"enabled": False}}}
    if origin == "configured":
        recipe["refinement"]["enabled"] = True
    elif origin == "checkpoint_config":
        checkpoint["config"]["refinement"]["enabled"] = True
    else:
        checkpoint["train_state"] = {"refinement": {"enabled": True}}
    with pytest.raises(ValueError, match="refine|flow-only"):
        validate_flow_only_stage1a(config, recipe, checkpoint)


def test_flow_only_provenance_guard_accepts_legacy_flow_checkpoint_and_preserves_other_recipes():
    config = _recipe()
    validate_flow_only_stage1a(config, {"stage": "stage1a_pdae"}, {"step": 50000})
    validate_flow_only_stage1a(config, {"refinement": {"enabled": False}},
                              {"config": {"refinement": {"enabled": False}}})
    external = deepcopy(config)
    external["decoded_translation"]["supervision"] = "external"
    validate_flow_only_stage1a(external, {"refinement": {"enabled": True}},
                              {"train_state": {"refinement": {"enabled": True}}})


@pytest.mark.parametrize("checkpointed", [False, True])
def test_standalone_evaluation_uses_encoder_cost_and_no_external_features(monkeypatch, tmp_path, checkpointed):
    """Real bank projection/solver/report with lightweight model/image boundaries."""
    import yaml
    import diffusion_ot.evaluation.stage1b_eval as evaluation_module
    import diffusion_ot.losses.semantic_prior as prior_module
    import diffusion_ot.training.decoded_translation as decoder_module

    config, evaluation = _recipe(), _evaluation()
    for value in (config, evaluation):
        value["project_root"] = str(tmp_path)
    evaluation["alignment_device"] = "cpu"
    evaluation["visualization"]["enabled"] = False
    evaluation["data"].update(reference_samples_per_domain=4,
                              projection_samples_per_domain=6,
                              query_samples_per_domain=3)
    # Test the real fused-cost wiring deterministically without testing MI iteration convergence here.
    evaluation["infoot"].update(mi_weight=0., entropy_epsilon=.2, inner_iterations=20,
                                recovery_iterations=0)
    config["infoot"] = deepcopy(evaluation["infoot"])
    config_path, evaluation_path = tmp_path / "alignment.yaml", tmp_path / "evaluation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    evaluation_path.write_text(yaml.safe_dump(evaluation), encoding="utf-8")
    checkpoint_path = tmp_path / "joint.pt" if checkpointed else None
    if checkpoint_path is not None:
        checkpoint_path.write_bytes(b"mocked-model-boundary")

    def forbidden(*args, **kwargs):
        raise AssertionError("No external feature model/prior may be loaded")

    monkeypatch.setattr(prior_module, "SemanticPriorBank", forbidden)
    monkeypatch.setattr(decoder_module, "load_image_features", forbidden)
    monkeypatch.setattr(evaluation_module, "_lpips", forbidden)

    contexts = {}
    for domain in ("cat", "dog"):
        contexts[domain] = SimpleNamespace(
            domain=domain, branch=SimpleNamespace(encoder=None),
            checkpoint_path=tmp_path / f"{domain}.pt", checkpoint_step=50000,
            weights="ema", stage1a_checkpoint_path=tmp_path / f"{domain}.pt",
            stage1a_checkpoint_step=50000, stage1a_weights="ema",
            device="cpu", dtype=torch.float32,
            stage1a_architecture={"semantic_cfg_enabled": True, "attention_lora": {"enabled": True}},
        )
    monkeypatch.setattr(evaluation_module, "_load_domain_context",
                        lambda alignment, root, domain, **kwargs: contexts[domain])
    monkeypatch.setattr(evaluation_module, "_dataset", lambda *args: None)

    def build_bank(encoder, dataset, *, domain, split, count, checkpoint_id, **kwargs):
        seed = (7 if domain == "cat" else 11) + (0 if split == "train" else 100)
        raw = torch.randn(count, 5, generator=torch.Generator().manual_seed(seed))
        ids = [f"{domain}_{split}_{index}" for index in range(count)]
        return evaluation_module.LatentBank(domain, split, raw, F.normalize(raw, dim=1), ids,
                                           [{"sample_id": value} for value in ids], checkpoint_id)

    monkeypatch.setattr(evaluation_module, "build_latent_bank", build_bank)
    solved = []
    original_solve = evaluation_module.solve_infoot

    def solve(cat, dog, **kwargs):
        expected = (1 - F.normalize(cat, dim=1) @ F.normalize(dog, dim=1).T).clamp(0, 2)
        torch.testing.assert_close(kwargs["cross_cost"], expected)
        assert not kwargs["cross_cost"].requires_grad
        result = original_solve(cat, dog, **kwargs)
        solved.append((expected, result.coupling))
        return result

    monkeypatch.setattr(evaluation_module, "solve_infoot", solve)
    reconstruction_calls, translation_calls = [], []

    def reconstruction(context, bank, **kwargs):
        assert kwargs["include_lpips"] is False
        reconstruction_calls.append(context.domain)
        return {"pixel_mse": .1, "pixel_psnr": 10., "latent_mse": .2}

    def translation(source, target, query, projection, probabilities, **kwargs):
        assert kwargs["teacher_codes"] is None
        assert "image_features" not in kwargs
        assert "color_histogram" not in kwargs
        assert probabilities.shape == (3, 6)
        translation_calls.append((source.domain, target.domain))
        return {}

    monkeypatch.setattr(evaluation_module, "_evaluate_reconstruction", reconstruction)
    monkeypatch.setattr(evaluation_module, "_save_translation_grid", translation)
    report = evaluation_module.run_stage1b_evaluation(
        config_path, evaluation_path, checkpoint_path=checkpoint_path)
    assert reconstruction_calls == ["cat", "dog"]
    assert translation_calls == [("cat", "dog"), ("dog", "cat")]
    assert len(solved) == 1
    assert report.solver["cross_cost_source"] == "encoder"
    assert report.solver["semantic_prior_fingerprint"] is None
    assert "transport_structure_cost" not in report.solver
    assert report.solver["transport_encoder_cost"] == pytest.approx(float((solved[0][0] * solved[0][1]).sum()))
    assert all("structure_prior_diagnostics" not in direction for direction in report.projections.values())
    assert Path(report.output_dir, "evaluation_report.json").is_file()
    payload = torch.load(Path(report.output_dir, "coupling.pt"), weights_only=True)
    assert payload["cross_cost_source"] == "encoder"
    assert payload["semantic_prior_fingerprint"] is None


def test_evaluation_rejects_encoder_to_dino_cost_switch_before_loading_models(tmp_path, monkeypatch):
    import yaml
    import diffusion_ot.evaluation.stage1b_eval as module

    config, evaluation = _recipe(), _evaluation()
    for value in (config, evaluation):
        value["project_root"] = str(tmp_path)
    evaluation["infoot"]["cross_cost_source"] = "dino"
    config_path, eval_path = tmp_path / "train.yaml", tmp_path / "eval.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    eval_path.write_text(yaml.safe_dump(evaluation), encoding="utf-8")
    monkeypatch.setattr(module, "_load_domain_context", lambda *args, **kwargs: pytest.fail("Loaded models before validating cost provenance"))
    with pytest.raises(ValueError, match="cross_cost_source"):
        module.run_stage1b_evaluation(config_path, eval_path)


def test_evaluation_checkpoint_guard_accepts_same_self_supervision_protocol():
    from diffusion_ot.evaluation.stage1b_eval import _validate_self_supervised_checkpoint

    config = _recipe()
    _validate_self_supervised_checkpoint(config, {"config": deepcopy(config)})


@pytest.mark.parametrize("section,key,value", [
    ("decoded_translation", "supervision", "external"),
    ("decoded_translation", "source_contrastive_readout", "source"),
    ("infoot", "cross_cost_source", "dino"),
    ("infoot", "variant", "plain"),
])
def test_evaluation_checkpoint_guard_rejects_mislabeled_self_supervision(section, key, value):
    from diffusion_ot.evaluation.stage1b_eval import _validate_self_supervised_checkpoint

    config = _recipe()
    saved = deepcopy(config)
    saved[section][key] = value
    with pytest.raises(ValueError):
        _validate_self_supervised_checkpoint(config, {"config": saved})


def test_evaluation_checkpoint_guard_rejects_self_checkpoint_as_external_recipe():
    from diffusion_ot.evaluation.stage1b_eval import _validate_self_supervised_checkpoint

    config = _recipe()
    saved = deepcopy(config)
    config["decoded_translation"]["supervision"] = "external"
    config["infoot"]["cross_cost_source"] = "dino"
    with pytest.raises(ValueError):
        _validate_self_supervised_checkpoint(config, {"config": saved})


def test_evaluation_checkpoint_guard_requires_self_recipe_metadata_but_preserves_legacy_external():
    from diffusion_ot.evaluation.stage1b_eval import _validate_self_supervised_checkpoint

    config = _recipe()
    with pytest.raises(ValueError):
        _validate_self_supervised_checkpoint(config, {"step": 500})
    _validate_self_supervised_checkpoint({}, {"step": 500})
    _validate_self_supervised_checkpoint({"decoded_translation": {"supervision": "external"}},
                                        {"config": {"stage": "stage1b_fused_infoot"}})
