"""P1a native objective and fresh residual Stage 1A -> Stage 1B contract."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from diffusion_ot.config_defaults import STAGE1A_EVAL, STAGE1B_TRAIN, STAGE1B_EVAL, stage1a_training_config
from diffusion_ot.losses.pdae_flow import flow_loss_weight, pdae_flow_snr_weight
from diffusion_ot.training.native_flow import native_flow_objective, validate_joint_objective, validate_stage1a_objective
from test_stage1a_residual_workflow import residual_training_setup, training_setup
from test_residual_encoder import small_cpu_thread_pool
from test_decoded_translation import TinyVAE

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["cosmap", "uniform", "pdae_flow_snr"])
def test_stage1b_dispatches_exact_native_weights_with_live_gradients(monkeypatch, kind):
    import diffusion_ot.losses.pdae_flow as flow
    import diffusion_ot.models.pdae_sit as models
    from diffusion_ot.training.train_joint_infoot import _reconstruction_loss
    t = torch.tensor([.05, .3, .5, .9])
    target = torch.arange(1., 5.).reshape(4, 1, 1, 1).expand(4, 4, 2, 2)
    monkeypatch.setattr(flow, "make_linear_flow_target", lambda x, **kw: SimpleNamespace(t=t, target_v=target, x_t=x))
    monkeypatch.setattr(models, "make_null_class_labels", lambda *a, **kw: None)
    prediction = torch.zeros_like(target, requires_grad=True)
    branch = SimpleNamespace(predict_with_z=lambda *a, **kw: SimpleNamespace(delta_sample=prediction, base_sample=torch.zeros_like(target)))
    cfg = dict(loss_weighting={"type": kind})
    domain = SimpleNamespace(training_config=cfg, branch=branch, transformer=None)
    diagnostics = {}
    actual = _reconstruction_loss(domain, torch.zeros_like(target), torch.zeros(4, 2), diagnostics=diagnostics)
    expected_weights = flow_loss_weight(t, cfg["loss_weighting"])
    expected = (target.square().flatten(1).mean(1) * expected_weights).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(prediction.grad, -2 * target * expected_weights[:, None, None, None] / target.numel())
    assert diagnostics["unweighted_flow_mse"] == float(target.square().mean())
    if kind == "cosmap":
        torch.testing.assert_close(expected_weights, 2 / (torch.pi * (t.square() + (1 - t).square())))
    if kind == "pdae_flow_snr":
        torch.testing.assert_close(expected_weights, pdae_flow_snr_weight(t=t, gamma=.1,
            normalization_mode="fixed_uniform", clamp_min=.001, clamp_max=None), rtol=0, atol=0)


def test_objective_provenance_rejects_changed_and_unverifiable_cosmap():
    cfg = dict(loss_weighting={"type": "cosmap"})
    objective = native_flow_objective(cfg)
    validate_stage1a_objective(cfg, {"config": cfg}, required_type="cosmap")
    validate_joint_objective({"native_flow_objectives": {"cat": objective}}, "cat", cfg)
    for changed in ({"loss_weighting": {"type": "uniform"}},
                    {**cfg, "flow": {"direction": "data_to_noise"}},
                    {**cfg, "flow": {"time_eps": .001}}):
        with pytest.raises(ValueError, match="objective changed"):
            validate_joint_objective({"native_flow_objectives": {"cat": objective}}, "cat", changed)
    with pytest.raises(ValueError, match="fresh cosmap"):
        validate_joint_objective({}, "cat", cfg)
    with pytest.raises(ValueError, match="provenance"):
        validate_stage1a_objective(cfg, {})
    validate_joint_objective({}, "cat", {})  # Historical SNR checkpoints stay loadable.
    with pytest.raises(ValueError, match="accepts only"):
        native_flow_objective({"loss_weighting": {"type": "cosmap", "gamma": .1}})


def test_default_pair_routes_exact_recipes_and_retains_controlled_objectives():
    cfg = yaml.safe_load((ROOT / STAGE1B_TRAIN).read_text())
    eva = yaml.safe_load((ROOT / STAGE1B_EVAL).read_text())
    assert cfg["stage1a"]["require_encoder_kind"] == "residual_cnn_v1"
    assert cfg["stage1a"]["require_loss_weighting"] == "cosmap"
    for d in ("cat", "dog"):
        assert cfg["stage1a"][d]["config"] == stage1a_training_config(d)
        a = yaml.safe_load((ROOT / stage1a_training_config(d)).read_text())
        assert a["loss_weighting"] == {"type": "cosmap"}
        assert a["train"]["resume_from"] is None and not a["train"].get("initialize_from")
        assert a["output_dir"] in cfg["stage1a"][d]["checkpoint"]
    assert cfg["train"]["resume_from"] is None
    assert cfg["matching"]["bandwidth_multiplier"] == .55 and cfg["infoot"]["entropy_epsilon"] == .02
    assert cfg["conditional_projection"]["bandwidth_multiplier"] == .25
    assert cfg["quick_evaluation"]["config"] == STAGE1B_EVAL
    assert cfg["output_dir"] == "outputs/stage1b_nce_v7_residual_cosmap"
    assert eva["output_dir"] == "outputs/stage1b_eval_nce_v7_residual_cosmap"
    assert eva["translation"]["num_steps"] == 20
    assert eva["translation"]["readouts"] == ["conditional_mean", "conditional_map"]
    assert eva["input_statistics"]["enabled"]
    assert yaml.safe_load((ROOT / STAGE1A_EVAL).read_text())["input_statistics"]["enabled"]


@pytest.mark.parametrize("script,argv,field,expected", [
    ("train_joint_infoot", [], "config", STAGE1B_TRAIN),
    ("evaluate_infoot_alignment", [], "eval_config", STAGE1B_EVAL),
    ("evaluate_pdae_domain", ["--domain", "dog"], "eval_config", STAGE1A_EVAL),
    ("train_pdae_domain", ["--domain", "cat"], "domain", "cat"),
])
def test_cli_defaults(script, argv, field, expected, monkeypatch):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / (script + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", [script, *argv])
    assert getattr(module.parse_args(), field) == expected


@pytest.mark.parametrize("weights", ["raw", "ema"])
def test_fresh_both_domain_pipeline_and_input_probes(residual_training_setup, monkeypatch, weights):
    from diffusion_ot.evaluation.stage1a_eval import run_stage1a_smoke_test
    from diffusion_ot.training.train_joint_infoot import train_joint_infoot
    from diffusion_ot.evaluation.stage1b_eval import run_stage1b_evaluation, _load_domain_context
    run, stage1a, latest, root = residual_training_setup
    stage1a["loss_weighting"] = {"type": "cosmap"}
    import diffusion_ot.data.ground_truth as ground_truth
    original_images = ground_truth.load_afhq_dataset(root / "data.yaml")
    original_images.extend([{**row, "label": "dog"} for row in list(original_images)])
    # Build a second disjoint synthetic domain with real cached-latent manifests.
    for split in ("train", "val"):
        records = []
        for line in (root / f"manifests/cat_{split}.jsonl").read_text().splitlines():
            record = json.loads(line)
            old_id = record["sample_id"]
            new_index = int(record["hf_index"]) + 12
            record.update(domain="dog", sample_id=f"afhq_dog_{new_index:06d}", hf_index=new_index)
            records.append(record)
            out = root / "latents" / f"dog_{split}" / (record["sample_id"] + ".pt")
            out.parent.mkdir(parents=True, exist_ok=True)
            x = torch.load(root / "latents" / f"cat_{split}" / (old_id + ".pt"), weights_only=True)
            torch.save(x * 1.1 + .2, out)
        (root / f"manifests/dog_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    input_options = dict(enabled=True, train_samples=8, development_samples=4, batch_size=4, ridge_alphas=[.01, 1.])
    a_eval = dict(stage="stage1a_eval", project_root=str(root), weights=weights, seed=9,
                  architecture=dict(require_encoder_kind="residual_cnn_v1", require_encoder_input_space="latent"),
                  dataset=dict(smoke_samples=2), sampling=dict(smoke_num_steps=2, variants=["correct_z", "shuffled_z"], guidance_scales=[1.]),
                  metrics=dict(image_reference="original_rgb"), input_statistics=input_options)
    eval_path = root / "a_eval.yaml"
    eval_path.write_text(yaml.safe_dump(a_eval))
    for d in ("cat", "dog"):
        stage1a["domain"] = d
        ckpt, _, report = run(f"{d}_fresh", steps=1)
        assert report.initial_step == 0 and ckpt["train_state"]["initialization"] is None
        assert ckpt["native_flow_objective"]["loss_weighting"] == {"type": "cosmap"}
        eva = run_stage1a_smoke_test(root / f"{d}_fresh.yaml", eval_path, device="cpu")
        probe = json.loads(Path(eva.extra_reports["input_statistics"]).read_text())
        assert probe["counts"] == dict(train=8, development=4)
        assert probe["provenance"]["weights"] == weights
        assert eva.num_samples == 2 and eva.image_reference == "original_rgb"
        with pytest.raises(ValueError, match="already has a checkpoint"):
            run(f"{d}_fresh", steps=1)
    monkeypatch.setattr(TinyVAE, "encode", lambda self, rgb: SimpleNamespace(
        latent_dist=SimpleNamespace(mean=F.conv2d(rgb, self.conv.weight.transpose(0, 1)))), raising=False)
    cfg = yaml.safe_load((ROOT / STAGE1B_TRAIN).read_text())
    cfg.update(project_root=str(root), output_dir="joint", seed=42)
    cfg["stage1a"].update(weights="ema", require_attention_lora_rank=2, require_attention_lora_alpha=2)
    for d in ("cat", "dog"):
        cfg["stage1a"][d] = dict(config=f"{d}_fresh.yaml", checkpoint=f"{d}_fresh/checkpoints/step_000001.pt", device="cpu")
    cfg["data"].update(transport_batch_size=6, reconstruction_batch_size=2, num_workers=0, pin_memory=False)
    cfg["train"].update(max_steps=1, save_every=1, log_every=1, validation_every=1, validation_samples=2, gradient_diagnostics_every=0)
    cfg["matching"].update(calibration_samples=8)
    cfg["matching_head"].update(input_dim=12, hidden_dim=8)
    cfg["conditional_projection"].update(query_samples_per_domain=2, validation_reference_samples=6, validation_query_samples=2)
    cfg["spatial_correlative_cost"].update(layers=[0, 1], grid_size=2)
    cfg["source_aware_selection"].update(appearance_size=4)
    cfg["infoot"].update(entropy_epsilon=.2, inner_iterations=400, recovery_iterations=800)
    cfg["decoded_translation"].update(batch_size=2, num_steps=2, validation_num_steps=2)
    cfg["decoded_translation"]["source_contrastive_projector"].update(projection_dim=8)
    from test_translation_image import image_options
    cfg["decoded_translation"].update(image_options())
    cfg["decoded_translation"].update(target_patch_swd=dict(weight=0.), source_lab_swd=dict(weight=.04, sizes=[8, 4], patch_size=3, directions=8))
    cfg["self_supervised_diagnostics"]["target_patch_swd"] = dict(enabled=True, sizes=[8, 4], patch_size=3, patches_per_image=4, directions=8, scale_floor=.01)
    config_path = root / "joint.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    result = train_joint_infoot(config_path, max_steps=1)
    joint_path = Path(result.checkpoint_path)
    trained = torch.load(joint_path, weights_only=False)
    initial = torch.load(joint_path.with_name("step_000000.pt"), weights_only=False)
    assert initial["step"] == 0 and initial["ema_state"]["num_updates"] == 0
    assert initial["projection_rms_state"]["raw"]["num_updates"] == 0
    for d in ("cat", "dog"):
        assert initial["native_flow_objectives"][d]["loss_weighting"] == {"type": "cosmap"}
        architecture = trained["stage1a_provenance"][d]["architecture"]
        assert architecture["spatial_features"]["resolutions"][:2] == [8, 4]
        assert architecture["encoder_architecture"]["normalize_input"] is False
        initial_a = torch.load(root / f"{d}_fresh/checkpoints/step_000001.pt", weights_only=False)
        for name, value in initial["encoders"][d].items():
            torch.testing.assert_close(value, initial_a["ema"]["shadow"]["encoder." + name], rtol=0, atol=0)
        context = _load_domain_context(cfg, root, d, checkpoint_path=joint_path, joint_weights=weights, device_override="cpu")
        state = trained["encoder_ema" if weights == "ema" else "encoders"][d]
        for name, value in context.branch.encoder.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        assert initial["source_aware_selection_state"]["calibration"] == trained["source_aware_selection_state"]["calibration"]
        ids = initial["source_aware_selection_state"]["calibration"]["sample_ids"][d]
        development_ids = {json.loads(line)["sample_id"] for line in (root / f"manifests/{d}_val.jsonl").read_text().splitlines()}
        assert set(ids).isdisjoint(development_ids)
    eva = yaml.safe_load((ROOT / STAGE1B_EVAL).read_text())
    eva.update(project_root=str(root), output_dir="joint_eval", alignment_device="cpu", input_statistics=input_options,
               infoot=cfg["infoot"], spatial_correlative_cost=cfg["spatial_correlative_cost"], source_aware_selection=cfg["source_aware_selection"])
    eva["data"].update(reference_samples_per_domain=6, projection_samples_per_domain=8, query_samples_per_domain=4, batch_size=4)
    eva["translation"].update(samples_per_direction=2, num_steps=2)
    eva["reconstruction"].update(samples_per_domain=2, num_steps=2)
    eva["visualization"]["enabled"] = False
    path = root / "joint_eval.yaml"
    path.write_text(yaml.safe_dump(eva))
    report = run_stage1b_evaluation(config_path, path, checkpoint_path=joint_path, weights=weights, evaluate_step0=True)
    assert report.baseline_comparison["status"] == "compared"
    for d in ("cat", "dog"):
        probe = json.loads(Path(report.input_statistics[d]).read_text())
        assert probe["provenance"]["weights"] == weights
        assert probe["provenance"]["checkpoint_step"] == 1
        assert report.evaluation_protocol["effective_config"]["native_flow_objectives"][d]["loss_weighting"] == {"type": "cosmap"}
    resumed = train_joint_infoot(config_path, max_steps=2, resume_from="latest")
    assert resumed.initial_step == 1 and resumed.final_step == 2
    validation = [json.loads(line) for line in (root / "joint/logs/validation.jsonl").read_text().splitlines()]
    assert all(row["cat_raw_unweighted_flow_mse"] > 0 for row in validation)
    assert all(row["native_flow_objectives"]["dog"]["loss_weighting"] == {"type": "cosmap"} for row in validation)
    # Exercise the actual evaluator's provenance guard, not only its helper.
    corrupted = deepcopy(trained)
    corrupted["native_flow_objectives"]["cat"]["loss_weighting"] = {"type": "uniform"}
    bad_path = root / "bad_native.pt"
    torch.save(corrupted, bad_path)
    with pytest.raises(ValueError, match="native flow objective changed"):
        _load_domain_context(cfg, root, "cat", checkpoint_path=bad_path, joint_weights=weights, device_override="cpu")
