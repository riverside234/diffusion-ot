"""Backward-only conditional-weight scaling for global, MLP-projected InfoNCE."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from diffusion_ot.evaluation.stage1b_eval import _validate_self_supervised_checkpoint
from diffusion_ot.training.decoded_translation import (
    self_supervised_translation_options, validate_decoder_config,
)
from diffusion_ot.training.self_supervised_translation import (
    SelfSupervisedDecoderTraining, scale_conditional_weight_gradient,
)
from test_code_projector import mlp_config
from test_self_supervised_translation import batch, domains
from test_stage1b_extensions import assert_tensor_tree_equal, experiment
from test_stage1b_self_supervised import own_encoder_run, self_supervised_recipe


SCALE = "source_contrastive_projection_gradient_scale"
SAVED_SCALE = "decoded_source_contrastive_projection_gradient_scale"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scale", [0., .1, 1.])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_weight_gate_preserves_forward_values_and_scales_backward(scale, dtype):
    torch.manual_seed(14)
    weights = torch.randn(3, 7, dtype=dtype).softmax(-1).requires_grad_()
    output = scale_conditional_weight_gradient(weights, scale)
    assert torch.equal(output, weights) and output.dtype == dtype
    output.sum().backward()
    torch.testing.assert_close(weights.grad, torch.full_like(weights, scale), rtol=0, atol=0)
    with torch.no_grad():
        assert scale_conditional_weight_gradient(weights, scale) is weights


def test_decoded_mlp_infonce_scales_only_weight_path_preserving_conditions_and_generator(tmp_path, monkeypatch):
    import diffusion_ot.training.self_supervised_translation as translation

    integrate = translation.integrate_training_flow
    conditions = []

    def capture(*args, **kwargs):
        conditions.append(args[3].detach().clone())
        return integrate(*args, **kwargs)

    monkeypatch.setattr(translation, "integrate_training_flow", capture)

    def measure(scale):
        torch.manual_seed(53)
        cfg = mlp_config()
        cfg["decoded_translation"][SCALE] = scale
        ctx = domains()
        runtime = SelfSupervisedDecoderTraining(cfg, ctx, None, tmp_path, seed=53)
        weights, refs, keys, logits, latents = batch()
        loss, metrics = runtime.loss(weights, refs, latents, latents, step=2, source_query_codes=keys)
        loss.backward()
        grads = {
            "weight_logits": {d: x.grad.clone() for d, x in logits.items()},
            "target_codes": {d: x.grad.clone() for d, x in refs.items()},
        }
        for domain in ctx:
            for name, module in {"readout": ctx[domain].branch.encoder,
                                 "adapters": runtime.views[domain]["adapters"],
                                 "lora": runtime.views[domain]["lora"],
                                 "mlp": runtime.code_projectors[domain]}.items():
                grads[f"{domain}.{name}"] = {n: p.grad.clone() for n, p in module.named_parameters()
                                            if p.grad is not None}
                assert any(g.norm() > 0 for g in grads[f"{domain}.{name}"].values())
            assert all(p.grad is None for p in ctx[domain].vae.parameters())
        assert all(x.grad is None for x in keys.values())
        assert metrics[SCALE] == scale
        assert metrics["source_contrastive_projection_gradient_scope"] == "conditional_weights_only"
        return loss.detach(), grads, {d: c.vae.encoded_rgb for d, c in ctx.items()}

    full_loss, full_grads, full_rgb = measure(1.)
    scaled_loss, scaled_grads, scaled_rgb = measure(.1)
    assert torch.equal(full_loss, scaled_loss)
    assert_tensor_tree_equal(conditions[:2], conditions[2:])
    assert_tensor_tree_equal(full_rgb, scaled_rgb)
    for direction, grad in full_grads.pop("weight_logits").items():
        assert grad.norm() > 0
        torch.testing.assert_close(scaled_grads["weight_logits"][direction], .1 * grad, rtol=2e-5, atol=1e-10)
    scaled_grads.pop("weight_logits")
    # Direct reference-code, readout, adapter/LoRA and MLP gradients are identical.
    # In a real shared encoder the *additional* path through W is still scaled.
    assert_tensor_tree_equal(full_grads, scaled_grads)


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf"), True, None, "bad"])
def test_invalid_gradient_scales_fail_option_validation(value):
    cfg = mlp_config()
    cfg["decoded_translation"][SCALE] = value
    with pytest.raises(ValueError, match="finite and in"):
        self_supervised_translation_options(cfg["decoded_translation"])


def test_gradient_scale_is_not_silently_applied_to_patchnce_or_external_losses():
    cfg = mlp_config()
    cfg["decoded_translation"][SCALE] = .1
    cfg["decoded_translation"]["objective"] = "patchnce"
    with pytest.raises(ValueError, match="requires objective: source_infonce"):
        self_supervised_translation_options(cfg["decoded_translation"])
    cfg["decoded_translation"]["supervision"] = "external"
    with pytest.raises(ValueError, match="requires supervision: self_supervised"):
        validate_decoder_config(cfg)


def test_resume_and_evaluation_enforce_scale_provenance_but_accept_legacy_unit_scale(tmp_path):
    cfg = mlp_config()
    cfg["decoded_translation"][SCALE] = .1
    runtime = SelfSupervisedDecoderTraining(cfg, domains(), None, tmp_path, seed=11)
    payload = runtime.checkpoint_state()
    assert payload[SAVED_SCALE] == .1
    payload.update(config=deepcopy(cfg), format_version=4)
    restored = SelfSupervisedDecoderTraining(cfg, domains(), None, tmp_path, seed=12)
    restored.load_checkpoint(payload)
    for key, value in restored.checkpoint_state().items():
        assert_tensor_tree_equal(payload[key], value)
    _validate_self_supervised_checkpoint(cfg, payload)
    full = deepcopy(cfg)
    full["decoded_translation"].pop(SCALE)
    legacy = SelfSupervisedDecoderTraining(full, domains(), None, tmp_path, seed=13)
    with pytest.raises(ValueError, match="projection gradient scale"):
        legacy.load_checkpoint(payload)
    with pytest.raises(ValueError, match="projection gradient scale"):
        _validate_self_supervised_checkpoint(full, payload)
    broken = deepcopy(payload)
    broken.pop(SAVED_SCALE)
    with pytest.raises(ValueError, match="metadata disagree"):
        _validate_self_supervised_checkpoint(cfg, broken)
    old = legacy.checkpoint_state()
    old.pop(SAVED_SCALE)
    old.update(config=full, format_version=4)
    legacy.load_checkpoint(old)
    _validate_self_supervised_checkpoint(full, {**old, "config": full})
    with pytest.raises(ValueError, match="projection gradient scale"):
        runtime.load_checkpoint(old)


def v3_tiny_recipe(cfg, scale=.1):
    self_supervised_recipe(cfg)
    cfg["infoot"]["feature_objective"] = "mi"
    cfg["loss_weights"]["infoot_relative"] = .01
    cfg["matching_regularization"].update(std_target=.8, variance_weight=.1, covariance_weight=.3)
    cfg["projection_rms"] = {"mode": "reference_ema", "decay": .99, "eps": 1e-8}
    cfg["matching_head"]["lr"] = 1e-5
    cfg["decoded_translation"][SCALE] = scale
    cfg["decoded_translation"]["source_contrastive_projector"] = {
        "kind": "mlp", "projection_dim": 16, "lr": 2e-4, "grad_clip_norm": 1.}


def test_trainer_keeps_generator_gradients_and_independent_losses_while_scaling_matching_gradients(own_encoder_run):
    run, _, _, _ = own_encoder_run
    control, control_logs = run("v3_full_gradient", steps=1, modify=lambda c: v3_tiny_recipe(c, 1.))
    gated, gated_logs = run("v3_gated", steps=1, modify=v3_tiny_recipe)
    full, scaled = (logs["train"][0] for logs in (control_logs, gated_logs))
    key = "weighted_decoded_matching_head_gradient_norm"
    assert full[key] > 0
    assert scaled[key] == pytest.approx(.1 * full[key], rel=2e-5)
    key = "weighted_decoded_generator_gradient_norm"
    assert scaled[key] == pytest.approx(full[key], rel=1e-7)
    for component in ("infoot_mi", "infoot_relative", "matching_variance", "matching_covariance"):
        key = f"weighted_{component}_matching_head_gradient_norm"
        assert scaled[key] == full[key]
    assert_tensor_tree_equal(control["generators"], gated["generators"])
    assert full["decoded_translation"]["source_contrastive_loss"] == scaled["decoded_translation"]["source_contrastive_loss"]
    for row in gated_logs["train"] + gated_logs["validation"]:
        assert row["decoded_translation"][SCALE] == .1
    assert gated[SAVED_SCALE] == .1
    resumed, logs = run("v3_gated", steps=2, resume=True, modify=v3_tiny_recipe)
    assert resumed["step"] == 2 and resumed[SAVED_SCALE] == .1
    assert [row["step"] for row in logs["train"]] == [1, 2]
    with pytest.raises(ValueError, match="Resume cannot change decoded_translation"):
        run("v3_gated", steps=3, resume=True, modify=lambda c: v3_tiny_recipe(c, 1.))


def test_v3_configs_preserve_mlps_and_supply_an_isolated_gradient_control():
    def load(kind, name):
        return yaml.safe_load((ROOT / "configs" / kind / f"{name}_sit_b2.yaml").read_text())
    cfg = load("stage1b_infoot", "self_supervised_infonce_v3")
    control = load("stage1b_infoot", "self_supervised_infonce_v3_control")
    evaluation = load("stage1b_eval", "self_supervised_infonce_v3")
    validate_decoder_config(cfg)
    validate_decoder_config(control)
    assert cfg["decoded_translation"][SCALE] == .1
    assert control["decoded_translation"][SCALE] == 1.
    assert control["output_dir"] != cfg["output_dir"]
    control["output_dir"] = cfg["output_dir"]
    control["decoded_translation"][SCALE] = .1
    assert control == cfg
    assert cfg["decoded_translation"]["source_contrastive_projector"]["kind"] == "mlp"
    assert cfg["infoot"]["feature_objective"] == "mi"
    assert cfg["loss_weights"]["infoot_relative"] == .01
    assert cfg["matching_regularization"]["covariance_weight"] == .3
    assert cfg["matching_head"]["lr"] == 1e-5
    assert cfg["projection_rms"] == evaluation["projection_rms"]
    assert {k: v for k, v in cfg["infoot"].items() if k != "feature_objective"} == evaluation["infoot"]
    refs = cfg["data"]["transport_batch_size"] - cfg["conditional_projection"]["query_samples_per_domain"]
    assert refs == evaluation["data"]["reference_samples_per_domain"] == evaluation["data"]["projection_samples_per_domain"] == 224
    assert cfg["matching"]["bandwidth_multiplier"] == evaluation["matching"]["bandwidth_multiplier"] == .55
    assert cfg["conditional_projection"]["bandwidth_multiplier"] == evaluation["matching"]["projection_bandwidth_multiplier"] == .1
    for domain in ("cat", "dog"):
        assert cfg["stage1a"][domain]["config"] == f"configs/stage1a_pdae/{domain}_sit_b2_lora.yaml"
