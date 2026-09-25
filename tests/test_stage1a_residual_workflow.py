"""Real Stage 1A train/resume/evaluation paths with a tiny residual CNN and SiT."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from diffusion_ot.models.pdae_sit import build_pdae_sit_branch as real_builder
from diffusion_ot.evaluation.stage1a_eval import stage1a_architecture_metadata, validate_stage1a_architecture
from test_decoded_translation import TinyVAE
from test_pdae_sit_shapes import FakeSiT
from test_residual_encoder import tiny_config, small_cpu_thread_pool
from test_stage1a_refinement import setup as training_setup


@pytest.fixture
def residual_training_setup(training_setup, monkeypatch):
    import diffusion_ot.models.pdae_sit as models
    import diffusion_ot.integrations.sit_diffusers as sit
    import diffusion_ot.data.ground_truth as ground_truth

    run, config, latest, root, _, _, source = training_setup
    config.update(tiny_config())
    config["train"]["initialize_from"] = None
    config["refinement"] = {"enabled": False}

    def components(*args, **kwargs):
        with torch.random.fork_rng():
            torch.manual_seed(948)
            transformer, vae = FakeSiT(), TinyVAE()
        return SimpleNamespace(transformer=transformer, vae=vae)

    def build(*args, **kwargs):
        value = real_builder(*args, **kwargs)
        latest["branch"] = value
        latest["initial_encoder"] = deepcopy(value.encoder.state_dict())
        return value

    monkeypatch.setattr(sit, "load_sit_components", components)
    monkeypatch.setattr(models, "build_pdae_sit_branch", build)
    monkeypatch.setattr(ground_truth, "load_afhq_dataset", lambda *args, **kwargs: source)
    return run, config, latest, root


@pytest.mark.parametrize("weight_type", ["pdae_flow_snr", "uniform", "cosmap"])
def test_residual_flow_training_updates_encoder_and_resumes_optimizer_ema_and_step(
    residual_training_setup, weight_type,
):
    run, config, latest, _ = residual_training_setup
    config["loss_weighting"] = {"type": weight_type}
    checkpoint, logs, report = run("rescnn", steps=2)
    assert report.initial_step == 0 and report.final_step == 2
    assert checkpoint["model"]["format_version"] == 5
    assert checkpoint["train_state"]["initialization"] is None
    assert checkpoint["train_state"]["refinement"] is None
    assert checkpoint["train_state"]["loss_weighting"]["type"] == weight_type
    assert logs["refinement"] == []
    assert logs["train"][1]["encoder_grad_norm_pre_clip"] > 0
    assert any(not torch.equal(p, latest["initial_encoder"][name])
               for name, p in checkpoint["model"]["encoder"].items())
    for row in logs["train"] + logs["validation"]:
        assert row["encoder_kind"] == "residual_cnn_v1"
        assert "refinement" not in row
        assert row["timestep_sampling"] == "uniform"
    for row in logs["train"]:
        assert row["loss_weighting"] == weight_type
        assert row["unweighted_flow_mse"] > 0
        if weight_type == "uniform":
            assert row["weight_mean"] == 1.
            assert row["loss"] == pytest.approx(row["unweighted_flow_mse"])
        elif weight_type == "cosmap":
            assert 2 / torch.pi <= row["weight_mean"] <= 4 / torch.pi
    for row in logs["validation"]:
        assert row["training_loss_weighting"] == weight_type
        assert row["mse_weighting"] == "uniform"
    assert logs["validation"][0]["encoder_input_space"] == "latent"
    resumed, logs, report = run("rescnn", steps=3, resume=True)
    assert report.initial_step == 2 and report.final_step == 3
    assert resumed["ema"]["num_updates"] == 3
    assert [row["step"] for row in logs["train"]] == [1, 2, 3]
    assert [row["step"] for row in logs["validation"]] == [0, 1, 2, 3]
    assert all(float(state["step"]) == 3 for state in resumed["optimizer"]["state"].values())
    with pytest.raises(ValueError, match="architecture mismatch"):
        run("rescnn", steps=4, resume=True,
            modify=lambda c: c["encoder"].update(attention_heads=2))
    other_weighting = "uniform" if weight_type != "uniform" else "cosmap"
    with pytest.raises(ValueError, match="flow weighting objective changed on resume"):
        run("rescnn", steps=4, resume=True,
            modify=lambda c: c.update(loss_weighting={"type": other_weighting}))


@pytest.mark.parametrize("weights", ["raw", "ema"])
@pytest.mark.parametrize("weight_type", ["pdae_flow_snr", "uniform", "cosmap"])
def test_residual_evaluation_loads_matching_weights_and_scores_original_rgb(
    residual_training_setup, weights, weight_type,
):
    from diffusion_ot.evaluation.stage1a_eval import load_stage1a_evaluator, run_stage1a_smoke_test

    run, config, _, root = residual_training_setup
    config["loss_weighting"] = {"type": weight_type}
    checkpoint, _, _ = run("rescnn_eval", steps=2)
    train_path = root / "rescnn_eval.yaml"
    eval_path = root / "eval.yaml"
    eval_path.write_text(yaml.safe_dump({
        "stage": "stage1a_eval", "project_root": str(root), "weights": weights, "seed": 949,
        "architecture": {"require_encoder_kind": "residual_cnn_v1", "require_encoder_input_space": "latent"},
        "dataset": {"smoke_samples": 2},
        "sampling": {"smoke_num_steps": 2, "variants": ["correct_z", "shuffled_z"],
                     "guidance_scales": [1.0], "inferred_noise": {"enabled": True, "num_steps": 2}},
        "metrics": {"image_reference": "original_rgb"},
    }))
    evaluator = load_stage1a_evaluator(train_path, eval_path, device="cpu")
    for name, parameter in evaluator.branch.encoder.named_parameters():
        expected = (checkpoint["model"]["encoder"][name] if weights == "raw"
                    else checkpoint["ema"]["shadow"]["encoder." + name])
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert stage1a_architecture_metadata(evaluator.branch)["encoder_architecture"] == checkpoint["model"]["encoder_architecture"]
    report = run_stage1a_smoke_test(train_path, eval_path, device="cpu")
    assert report.checkpoint_step == 2 and report.weights == weights
    assert report.encoder_input_space == "latent" and report.image_reference == "original_rgb"
    assert report.encoder_architecture == checkpoint["model"]["encoder_architecture"]
    assert report.row_order[:2] == ["original", "vae_reconstruction"]
    assert Path(report.grid_path).is_file()
    roundtrip = json.loads(Path(report.extra_reports["inferred_noise_roundtrip"]).read_text())
    assert roundtrip["encoder_architecture"] == report.encoder_architecture
    assert roundtrip["image_reference"] == "original_rgb"


def test_evaluation_protocol_requires_the_residual_latent_architecture():
    root = Path(__file__).resolve().parents[1]
    recipe = yaml.safe_load((root / "configs/stage1a_eval/residual_sit_b2_256.yaml").read_text())
    assert recipe["metrics"]["image_reference"] == "original_rgb"
    assert recipe["sampling"]["fixed_starting_noise"]
    assert recipe["sampling"]["inferred_noise"]["report_separately"]
    config = yaml.safe_load((root / "configs/stage1a_pdae/cat_sit_b2_lora_residual.yaml").read_text())
    config["adapter"].update(injection_layers=[0, 1], lora_layers=[0, 1])
    branch = real_builder(FakeSiT(), stage_config=config)
    validate_stage1a_architecture(branch, recipe["architecture"])
    plain = real_builder(FakeSiT())
    assert "encoder_architecture" not in stage1a_architecture_metadata(plain)
    with pytest.raises(ValueError, match="encoder kind"):
        validate_stage1a_architecture(plain, recipe["architecture"])


@pytest.mark.parametrize("override", [None, "pinned_backbone.yaml"])
def test_training_and_evaluation_resolve_the_same_pretrained_config(
    residual_training_setup, monkeypatch, override,
):
    import diffusion_ot.integrations.sit_diffusers as sit
    from diffusion_ot.evaluation.stage1a_eval import load_stage1a_evaluator

    run, config, _, root = residual_training_setup
    if override is not None:
        config["pretrained_config"] = override
    requested = []
    load_components = sit.load_sit_components

    def record_components(path, **kwargs):
        requested.append(Path(path))
        return load_components(path, **kwargs)

    monkeypatch.setattr(sit, "load_sit_components", record_components)
    run("backbone_parity", steps=1)
    eval_path = root / "eval.yaml"
    eval_path.write_text(yaml.safe_dump({
        "project_root": str(root), "weights": "ema",
        "sampling": {"variants": ["correct_z"]},
    }))
    load_stage1a_evaluator(root / "backbone_parity.yaml", eval_path, device="cpu")
    expected = root / (override or "pretrained.yaml")
    assert requested == [expected, expected]


@pytest.mark.parametrize("weight_type", ["logit_normal", "pdae_noise_snr"])
def test_unimplemented_loss_weighting_cannot_silently_run_pdae_flow_snr(
    residual_training_setup, monkeypatch, weight_type,
):
    import diffusion_ot.integrations.sit_diffusers as sit

    run, _, _, _ = residual_training_setup

    def unexpected_model_load(*args, **kwargs):
        raise AssertionError("Invalid objective must fail before loading the backbone.")

    monkeypatch.setattr(sit, "load_sit_components", unexpected_model_load)
    with pytest.raises(ValueError, match="Unsupported Stage 1A loss_weighting.type"):
        run("invalid_weighting", steps=1,
            modify=lambda c: c.update(loss_weighting={"type": weight_type}))


def test_time_weighting_runs_share_initialization_timesteps_and_validation_protocol(
    residual_training_setup, monkeypatch,
):
    import diffusion_ot.losses.pdae_flow as flow

    run, config, latest, _ = residual_training_setup
    actual_weight = flow.flow_loss_weight
    sampled_times = []
    def record_weight(t, config=None, **kwargs):
        sampled_times.append(t.detach().clone())
        return actual_weight(t, config, **kwargs)
    monkeypatch.setattr(flow, "flow_loss_weight", record_weight)
    initial_encoders, times, probes = [], [], []
    for mode in ("pdae_flow_snr", "uniform", "cosmap"):
        config["loss_weighting"] = {"type": mode}
        sampled_times.clear()
        _, logs, _ = run(mode, steps=2)
        initial_encoders.append(deepcopy(latest["initial_encoder"]))
        times.append(list(sampled_times))
        probes.append(logs["validation"][0])
    assert len(times[0]) == 4  # Two updates, two microbatches each.
    for index in (1, 2):
        for name, value in initial_encoders[0].items():
            torch.testing.assert_close(value, initial_encoders[index][name], rtol=0, atol=0)
        for left, right in zip(times[0], times[index]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert probes[index]["correct_z"] == probes[0]["correct_z"]
        assert probes[index]["time_bins"] == probes[0]["time_bins"]


def test_legacy_checkpoint_with_implicit_weighting_defaults_can_resume(residual_training_setup):
    run, config, _, root = residual_training_setup
    checkpoint, _, _ = run("legacy_weighting", steps=1)
    checkpoint["train_state"].pop("loss_weighting")
    assert "loss_weighting" not in checkpoint["config"]
    torch.save(checkpoint, root / "legacy_weighting/checkpoints/latest.pt")
    config["loss_weighting"] = {"type": "pdae_flow_snr", "gamma": .1}
    _, _, report = run("legacy_weighting", steps=2, resume=True)
    assert report.initial_step == 1 and report.final_step == 2


@pytest.mark.parametrize("domain", ["cat", "dog"])
def test_time_weighting_recipes_change_only_objective_and_output(domain):
    root = Path(__file__).resolve().parents[1]
    stem = root / "configs/stage1a_pdae"
    base = yaml.safe_load((stem / f"{domain}_sit_b2_lora_residual.yaml").read_text())
    outputs = {base["output_dir"]}
    for mode in ("uniform", "cosmap"):
        candidate = yaml.safe_load((stem / f"{domain}_sit_b2_lora_residual_{mode}.yaml").read_text())
        assert candidate["loss_weighting"] == {"type": mode}
        assert candidate["output_dir"] not in outputs
        outputs.add(candidate["output_dir"])
        candidate["loss_weighting"] = base["loss_weighting"]
        candidate["output_dir"] = base["output_dir"]
        assert candidate == base
