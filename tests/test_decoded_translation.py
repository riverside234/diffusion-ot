from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from test_pdae_sit_shapes import FakeSiT
from test_infoot_semantic_prior import save_prior
from diffusion_ot.models.pdae_sit import PDAESiTBranch, SemanticSiTWrapper
from diffusion_ot.models.generator_adaptation import (
    baseline_parameter_snapshot, configure_generator_adaptation, generator_parameter_view,
    load_joint_generator, predict_with_parameters,
)
from diffusion_ot.training.decoded_translation import (
    DecoderTraining, FrozenImageFeatures, integrate_training_flow, validate_decoder_config,
)


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, 1)
        self.config = SimpleNamespace(scaling_factor=1.0)

    def decode(self, x):
        return SimpleNamespace(sample=self.conv(x).tanh())


class TinyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 2, stride=2)
        self.config = SimpleNamespace(hidden_size=8)

    def forward(self, pixel_values):
        patches = self.conv(pixel_values).flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=torch.cat([patches.mean(1, keepdim=True), patches], 1))


def tiny_features(*args, **kwargs):
    return FrozenImageFeatures(TinyDINO(), {"do_resize": False, "do_center_crop": False,
                                          "image_mean": [0, 0, 0], "image_std": [1, 1, 1]}, 2)


def branch():
    base = FakeSiT()
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(4 * 8 * 8, 6), nn.LayerNorm(6))
    semantic = SemanticSiTWrapper(base, z_dim=6, injection_layers=[0, 1], bottleneck_dim=4,
                                 attention_lora=True, lora_rank=2, lora_alpha=2, lora_layers=[0, 1])
    result = PDAESiTBranch(encoder, semantic, semantic_cfg_enabled=True, z_dim=6,
                          semantic_dropout_probability=.1)
    # Mimic a trained Stage 1A state; zero initial gates otherwise obscure
    # encoder/code gradients on the first synthetic translation.
    with torch.no_grad():
        for adapter in [*semantic.adapters.values(), semantic.final_adapter]:
            adapter.cond_proj[-1].weight.normal_(0, .1)
        for layer in semantic._attention_lora_modules.values():
            for module in layer.values():
                module.lora_up.weight.normal_(0, .05)
    return result


def config():
    return {"stage": "stage1b_fused_infoot", "trainable": {"encoders": True, "adapters": True, "attention_lora": True},
            "matching_head": {"enabled": True, "input_dim": 6, "hidden_dim": 8, "lr": .002},
            "conditional_structure": {"enabled": True, "query_samples_per_domain": 2,
                                      "bandwidth_multiplier": .3, "teacher_temperature": .2,
                                      "validation_reference_samples": 6, "validation_query_samples": 2},
            "generator_adaptation": {"enabled": True, "lr_adapter": .001, "lr_lora": .001,
                                     "null_preservation_weight": .1},
            "decoded_translation": {"enabled": True, "batch_size": 2, "num_steps": 3,
                                    "validation_num_steps": 4, "warmup_steps": 2,
                                    "structure_weight": .1, "adversarial_weight": .01,
                                    "discriminator_hidden_dim": 8, "discriminator_lr": .001},
            "ema": {"enabled": True, "decay": .9, "warmup_steps": 1}}


def domains():
    from diffusion_ot.evaluation.stage1a_eval import validate_stage1a_architecture
    values = {}
    for d in ("cat", "dog"):
        model = branch()
        values[d] = SimpleNamespace(branch=model, transformer=model.semantic_transformer.base,
                                    device="cpu", dtype=torch.float32, vae=TinyVAE(),
                                    training_config={"flow": {"direction": "noise_to_data"}},
                                    stage1a_architecture=validate_stage1a_architecture(model, {}))
    return values


def test_adaptation_only_exposes_added_modules_and_preserves_base_and_null():
    value = branch()
    view = configure_generator_adaptation(value)
    ids = {id(p) for p in view.parameters()}
    assert ids and len(ids) == len(list(view.parameters()))
    for p in value.semantic_transformer.parameters():
        assert p.requires_grad == (id(p) in ids)
    assert not value.semantic_conditioner.null_token.requires_grad
    assert all(p.requires_grad for p in value.encoder.parameters())
    assert not any(p.requires_grad for p in value.semantic_transformer.base.x_embedder.parameters())


def test_checkpointed_sampler_matches_evaluation_values_and_gradients():
    from diffusion_ot.evaluation.stage1a_eval import integrate_pdae_flow
    torch.manual_seed(11)
    value = branch()
    configure_generator_adaptation(value)
    noise, z = torch.randn(2, 4, 8, 8), torch.randn(2, 6, requires_grad=True)
    kwargs = {"num_steps": 4, "guidance_scale": 1.7}
    expected = integrate_pdae_flow(value, value.semantic_transformer.base, noise, z.detach(), **kwargs)
    direct = integrate_training_flow(value, value.semantic_transformer.base, noise, z, checkpoint_steps=False, **kwargs)
    params = [z, *generator_parameter_view(value).parameters()]
    direct_grads = torch.autograd.grad(direct.square().mean(), params)
    checked = integrate_training_flow(value, value.semantic_transformer.base, noise, z, checkpoint_steps=True, **kwargs)
    checked_grads = torch.autograd.grad(checked.square().mean(), params)
    torch.testing.assert_close(checked, expected)
    for actual, target in zip(checked_grads, direct_grads):
        torch.testing.assert_close(actual, target)
    assert checked_grads[0].norm() > 0


def test_fixed_generator_teacher_does_not_mutate_live_graph():
    torch.manual_seed(12)
    value = branch()
    view = configure_generator_adaptation(value)
    baseline = baseline_parameter_snapshot(value, view)
    x, t, z = torch.randn(2, 4, 8, 8), torch.rand(2), torch.randn(2, 6, requires_grad=True)
    labels = torch.full((2,), 10)
    old = value.predict_with_z(x, t, z, class_labels=labels).sample.detach()
    with torch.no_grad():
        for p in view.parameters():
            p.add_(.01)
    live = value.predict_with_z(x, t, z, class_labels=labels).sample
    with torch.no_grad():
        teacher = predict_with_parameters(value, baseline, x, t, z, labels).sample
    torch.testing.assert_close(teacher, old)
    assert not torch.allclose(live, teacher)
    live.square().mean().backward()
    assert z.grad.norm() > 0


def test_image_loss_updates_decoder_weights_and_codes_without_teacher_or_d_gradients(monkeypatch, tmp_path):
    import diffusion_ot.training.decoded_translation as module
    monkeypatch.setattr(module, "load_image_features", tiny_features)
    torch.manual_seed(5)
    ctx = domains()
    runtime = DecoderTraining(config(), ctx, None, tmp_path, seed=5)
    references = {d: torch.randn(5, 6, requires_grad=True) for d in ctx}
    logits = {f"{s}_to_{t}": torch.randn(2, 5, requires_grad=True) for s, t in (("cat", "dog"), ("dog", "cat"))}
    weights = {d: F.softmax(v, -1) for d, v in logits.items()}
    images = {d: torch.randn(2, 4, 8, 8) for d in ctx}
    d_before = deepcopy(runtime.discriminators.state_dict())
    loss, metrics = runtime.loss(weights, references, images, images, step=2)
    loss.backward()
    assert all(v.grad.norm() > 0 for v in references.values())
    assert all(v.grad.norm() > 0 for v in logits.values())
    for view in runtime.views.values():
        assert sum(float(p.grad.norm()) for p in view["adapters"].parameters() if p.grad is not None) > 0
        assert sum(float(p.grad.norm()) for p in view["lora"].parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in runtime.features.parameters())
    assert all(p.grad is None for p in runtime.discriminators.parameters())
    assert all(p.grad is None for v in ctx.values() for p in v.vae.parameters())
    assert any(not torch.equal(v, d_before[k]) for k, v in runtime.discriminators.state_dict().items())
    assert metrics["cat_to_dog"]["reference_targets"] == 5
    assert runtime.discriminator_updates == 1
    saved = deepcopy(runtime.discriminators.state_dict())
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        first = runtime.loss(weights, references, images, images, step=2, validation_seed=20)[1]
        second = runtime.loss(weights, references, images, images, step=2, validation_seed=20)[1]
    assert first == second
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for k, v in runtime.discriminators.state_dict().items():
        torch.testing.assert_close(v, saved[k])
    assert runtime.discriminator_updates == 1


@pytest.mark.parametrize("guarded", [False, True])
def test_experiment_d_training_resume_fixed_baseline_and_ema(monkeypatch, tmp_path, guarded):
    import yaml
    import diffusion_ot.data.latent_dataset as data
    import diffusion_ot.training.train_joint_infoot as train
    import diffusion_ot.training.decoded_translation as decoded
    from diffusion_ot.models.matching_head import load_matching_head

    torch.manual_seed(40)
    originals = domains()
    features = tiny_features()
    monkeypatch.setattr(decoded, "load_image_features", lambda *a, **kw: deepcopy(features))
    latest = {}
    class Dataset:
        def __init__(self, path, domain, split="train", **kwargs):
            self.domain, self.split = domain, split
            self.records = [{"sample_id": f"{domain}_{split}_{i}", "latent_path": "unused"} for i in range(8)]
        def __len__(self):
            return 8
        def __getitem__(self, index):
            gen = torch.Generator().manual_seed(index + (100 if self.domain == "dog" else 0) + (20 if self.split == "val" else 0))
            return {"x0_latent": torch.randn(4, 8, 8, generator=gen), "sample_id": self.records[index]["sample_id"],
                    "domain": self.domain, "split": self.split, "latent_path": "unused", "metadata": self.records[index]}
    def load_domain(config, root, domain, **kwargs):
        value = deepcopy(originals[domain])
        value.checkpoint_path, value.checkpoint_step = tmp_path / f"{domain}.pt", 100
        value.data_config_path = tmp_path / "data.yaml"
        train.freeze_generator_train_encoder(value.branch)
        latest[domain] = value
        return value
    monkeypatch.setattr(data, "CachedLatentDataset", Dataset)
    monkeypatch.setattr(train, "_load_training_domain", load_domain)
    save_prior(tmp_path / "prior.pt")
    for d in originals:
        (tmp_path / f"{d}.yaml").write_text(yaml.safe_dump({"data_config": "data.yaml"}))
    cfg = config()
    cfg["generator_adaptation"].update(conditioned_preservation_weight=.05,
                                      conditioned_preservation_samples=2)
    cfg.update(project_root=str(tmp_path), output_dir="out",
               stage1a={d: {"config": f"{d}.yaml"} for d in originals},
               data={"transport_batch_size": 8, "reconstruction_batch_size": 4, "num_workers": 0,
                     "random_horizontal_flip": 0, "pin_memory": False},
               matching={"bandwidth_multiplier": .7, "calibration_samples": 8,
                         "distance_scale_gradient": "full"},
               matching_regularization={"enabled": True, "std_target": .7,
                                        "variance_weight": .02, "covariance_weight": .001},
               infoot={"variant": "fused", "inner_iterations": 100, "entropy_epsilon": .2,
                       "mi_weight": .1, "outer_tolerance": 1e-5,
                       "strict_convergence": True, "require_outer_convergence": True},
               semantic_prior={"path": "prior.pt", "neighborhood_geometry": "rms_distance"},
               loss_weights={"semantic_neighborhood": .05, "conditional_structure": .05},
               train={"max_steps": 2, "log_every": 1, "save_every": 1, "validation_every": 1,
                      "validation_samples": 2, "gradient_diagnostics_every": 1, "lr_encoder": .001},
               gradient_guard={"enabled": guarded, "max_auxiliary_ratio": .25})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    validate_decoder_config(cfg)
    report = train.train_joint_infoot(path)
    payload = train._load_checkpoint(Path(report.checkpoint_path))
    assert payload["format_version"] == 4
    assert len(payload["optimizer"]["param_groups"]) == 4
    assert payload["decoded_discriminator_updates"] == 2
    initial_discriminator = deepcopy(payload["decoded_discriminators"])
    validation = [json.loads(row) for row in (tmp_path / "out/logs/validation.jsonl").read_text().splitlines()]
    training = [json.loads(row) for row in (tmp_path / "out/logs/train.jsonl").read_text().splitlines()]
    for row in training:
        assert row["infoot_distance_scale_gradient"] == "full"
        assert row["semantic_neighborhood_geometry"] == "rms_distance"
        assert row["infoot_iteration_budget"] == 100
        assert row["infoot_outer_converged"] and row["infoot_iterations"] < 100
        protection = row["matching_regularization"]
        assert row["matching_regularization_loss"] == pytest.approx(
            .02 * protection["variance_loss"] + .001 * protection["covariance_loss"])
        for domain in ("cat", "dog"):
            assert protection[domain]["samples"] == row["infoot_reference_counts"][domain] == 6
            assert protection[domain]["matching_variance"] == pytest.approx(row["matching_feature_variance"][domain])
        for group in ("encoder", "matching_head"):
            assert row[f"weighted_matching_regularization_{group}_gradient_norm"] > 0
            for term in ("variance", "covariance"):
                assert row[f"weighted_matching_{term}_{group}_gradient_norm"] >= 0
        for term in ("alignment", "neighborhood", "conditional_structure"):
            assert row[f"weighted_{term}_matching_head_gradient_norm"] > 0
        for group in ("encoder", "matching_head", "generator"):
            assert row[f"weighted_decoded_{group}_gradient_norm"] > 0
        assert row["reconstruction_generator_gradient_norm"] > 0
        assert row["weighted_null_preservation_generator_gradient_norm"] >= 0
        assert row["weighted_conditioned_preservation_generator_gradient_norm"] >= 0
        assert row["conditioned_preservation_samples"] == 2
        assert row["conditioned_preservation_weight"] == .05
        assert row["conditioned_preservation_loss"] >= 0
        assert row["window_mean"]["weighted_conditioned_preservation_loss"] == pytest.approx(
            .05 * row["window_mean"]["conditioned_preservation_loss"])
        decoded = row["decoded_translation"]
        assert decoded["effective_weighted_loss"] == pytest.approx(decoded["weighted_loss"] * decoded["ramp"])
        assert 0 < decoded["discriminator_clip_scale"] <= 1
        assert row["feature_geometry"]["cat"]["raw_normalized_variance"] > 0
    first = training[0]
    # The protection is included at full configured strength from update 1,
    # even though the other alignment/image objectives are still warming up.
    assert first["window_mean"]["auxiliary_objective"] == pytest.approx(
        first["alignment_weight"] * first["infoot_feature_loss"]
        + first["semantic_neighborhood_weight"] * first["semantic_neighborhood_loss"]
        + first["conditional_structure_weight"] * first["conditional_structure_loss"]
        + first["decoded_translation"]["effective_weighted_loss"]
        + first["matching_regularization_loss"], abs=1e-6)
    assert first["window_mean"]["primary_objective"] == pytest.approx(
        first["cat_reconstruction_loss"] + first["dog_reconstruction_loss"]
        + .1 * first["null_preservation_loss"] + .05 * first["conditioned_preservation_loss"], abs=1e-6)
    assert all("feature_geometry" in row["projection_probe"] for row in validation)
    for row in validation:
        assert row["projection_probe"]["iteration_budget"] == 100
        assert row["decoded_translation"]["solver"]["iteration_budget"] == 100
        assert row["matching_regularization_loss"] > 0
        for domain in ("cat", "dog"):
            assert row["matching_regularization"][domain]["matching_variance"] == pytest.approx(
                row["projection_probe"]["matching_feature_variance"][domain])
        assert "plan_delta_l1" in row["projection_probe"]
        solver = row["decoded_translation"]["solver"]
        assert solver["sinkhorn_converged"]
        assert solver["iterations"] > 0
        assert solver["plan_delta_l1"] >= 0
    initial_probe = validation[0]
    for domain in ("cat", "dog"):
        for key in ("stage1a_encoder_current_generator_reconstruction",
                    "current_encoder_stage1a_generator_reconstruction"):
            assert initial_probe[f"{domain}_{key}"] == pytest.approx(initial_probe[f"{domain}_stage1a_reconstruction"])
            assert all(math.isfinite(row[f"{domain}_{key}"]) for row in validation)
    for domain in originals:
        current, original = latest[domain].branch, originals[domain].branch
        initial = original.state_dict()
        adapted = {id(p) for p in generator_parameter_view(current).parameters()}
        encoder_ids = {id(p) for p in current.encoder.parameters()}
        for name, p in current.named_parameters():
            if id(p) not in adapted | encoder_ids:
                torch.testing.assert_close(p, initial[name], rtol=0, atol=0)
        assert any(not torch.equal(p, dict(original.encoder.named_parameters())[n]) for n, p in current.encoder.named_parameters())
        for group in ("adapters", "lora"):
            after, before = generator_parameter_view(current)[group].state_dict(), generator_parameter_view(original)[group].state_dict()
            assert any(not torch.equal(v, before[k]) for k, v in after.items())
        for key in (f"{domain}_stage1a_reconstruction", f"{domain}_stage1a_null_reconstruction"):
            assert all(v[key] == pytest.approx(validation[0][key], abs=1e-7) for v in validation)
        raw, ema = deepcopy(original), deepcopy(original)
        load_joint_generator(raw, payload, domain, weights="raw")
        load_joint_generator(ema, payload, domain, weights="ema")
        assert any(not torch.equal(v, generator_parameter_view(raw).state_dict()[k]) for k, v in generator_parameter_view(ema).state_dict().items())
        assert load_matching_head(payload, domain, weights="ema", device="cpu") is not None
        assert payload["matching_heads"][domain]["residual.3.weight"].norm() > 0
    assert all(v["decoded_translation"]["discriminator_updates"] == v["step"] for v in validation)
    # Resume a complete optimizer/D/EMA state and continue, not a new G run.
    # A larger strict outer cap is numerical headroom, not a new objective.
    cfg["infoot"]["inner_iterations"] = 1200
    path.write_text(yaml.safe_dump(cfg))
    resumed = train.train_joint_infoot(path, max_steps=3, resume_from="latest")
    final = train._load_checkpoint(Path(resumed.checkpoint_path))
    assert final["config"]["infoot"]["inner_iterations"] == 1200
    assert resumed.initial_step == 2
    assert final["decoded_discriminator_updates"] == 3
    assert final["generator_ema_state"]["num_updates"] == final["ema_state"]["num_updates"] == 3
    assert final["matching_head_ema_state"]["num_updates"] == 3
    assert any(not torch.equal(v, initial_discriminator[k]) for k, v in final["decoded_discriminators"].items())
    assert "coupling" not in final and "transport_plan" not in final
    changed_preservation = deepcopy(cfg)
    changed_preservation["generator_adaptation"]["conditioned_preservation_weight"] = .1
    path.write_text(yaml.safe_dump(changed_preservation))
    with pytest.raises(ValueError, match="Resume cannot change generator_adaptation"):
        train.train_joint_infoot(path, max_steps=4, resume_from="latest")
    path.write_text(yaml.safe_dump(cfg))
    broken = deepcopy(final)
    del broken["generator_ema"]["cat"]
    with pytest.raises(ValueError, match="generator_ema.cat"):
        load_joint_generator(branch(), broken, "cat", weights="ema")
    cfg["matching_regularization"]["covariance_weight"] = .002
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Resume cannot change matching_regularization"):
        train.train_joint_infoot(path, max_steps=4, resume_from="latest")
    cfg["matching_regularization"]["covariance_weight"] = .001
    cfg["decoded_translation"]["structure_weight"] = .2
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Resume cannot change decoded_translation"):
        train.train_joint_infoot(path, max_steps=4, resume_from="latest")

    # Exercise the actual evaluator loader, including the Stage 1A initialization
    # followed by paired joint E/G/head restoration (without downloading SiT).
    import diffusion_ot.integrations.sit_diffusers as integration
    import diffusion_ot.models.pdae_sit as pdae
    import diffusion_ot.evaluation.stage1b_eval as evaluation
    for domain, original in originals.items():
        torch.save({"model": original.branch.pdae_state_dict(), "step": 100}, tmp_path / f"{domain}.pt")
        (tmp_path / f"{domain}.yaml").write_text(yaml.safe_dump({"data_config": "data.yaml", "model_config": "model.yaml", "device": "cpu"}))
    (tmp_path / "model.yaml").write_text(yaml.safe_dump({"pretrained": "pretrained.yaml"}))
    cfg = deepcopy(final["config"])
    cfg["stage1a"]["weights"] = "raw"
    for domain in originals:
        cfg["stage1a"][domain]["checkpoint"] = f"{domain}.pt"
        final["stage1a_provenance"][domain]["weights"] = "raw"
    torch.save(final, tmp_path / "joint.pt")
    monkeypatch.setattr(integration, "load_sit_components", lambda *a, **kw: SimpleNamespace(transformer=FakeSiT(), vae=TinyVAE()))
    monkeypatch.setattr(integration, "validate_transformer_config", lambda *a, **kw: [])
    monkeypatch.setattr(pdae, "build_pdae_sit_branch", lambda *a, **kw: branch())
    for weights in ("raw", "ema"):
        ctx = evaluation._load_domain_context(cfg, tmp_path, "cat", checkpoint_path=tmp_path / "joint.pt", joint_weights=weights, device_override="cpu")
        key = "encoders" if weights == "raw" else "encoder_ema"
        for name, value in ctx.branch.encoder.state_dict().items():
            torch.testing.assert_close(value, final[key]["cat"][name])
        expected = branch()
        load_joint_generator(expected, final, "cat", weights=weights)
        for name, value in generator_parameter_view(ctx.branch).state_dict().items():
            torch.testing.assert_close(value, generator_parameter_view(expected).state_dict()[name])


def test_decoded_grid_reports_structure_for_each_readout(monkeypatch, tmp_path):
    import diffusion_ot.evaluation.stage1b_eval as evaluation
    from test_stage1b_eval import _bank
    torch.manual_seed(8)
    ctx = domains()
    for value in ctx.values():
        configure_generator_adaptation(value.branch)
    source, target = _bank("cat", "val", ["c0", "c1"]), _bank("dog", "train", ["d0", "d1", "d2"])
    target.raw_codes = torch.randn(3, 6)
    monkeypatch.setattr(evaluation, "_load_latents_from_bank", lambda b, n: torch.randn(n, 4, 8, 8))
    metrics = evaluation._save_translation_grid(ctx["cat"], ctx["dog"], source, target,
                                               torch.full((2, 3), 1/3), count=2, num_steps=3,
                                               guidance_scale=1, temperature=1, seed=2,
                                               output_path=tmp_path / "grid.png", image_features=tiny_features(),
                                               teacher_codes=torch.randn(2, 6))
    assert (tmp_path / "grid.png").is_file()
    for name in ("infoot_conditional_mean", "selected_target", "sampled_target", "structure_teacher_mean"):
        assert metrics[name]["samples"] == 2
        assert 0 <= metrics[name]["structure_loss"] <= 2


def test_resume_restores_dedicated_translation_noise_and_discriminator_optimizer(monkeypatch, tmp_path):
    import diffusion_ot.training.decoded_translation as module
    monkeypatch.setattr(module, "load_image_features", tiny_features)
    value = DecoderTraining(config(), domains(), None, tmp_path, seed=6)
    for gen in value.noise_generators.values():
        torch.randn(13, generator=gen)
    state = value.checkpoint_state()
    state.update(config=config(), format_version=4)
    restored = DecoderTraining(config(), domains(), None, tmp_path, seed=10)
    restored.load_checkpoint(state)
    for domain in value.domains:
        torch.testing.assert_close(torch.randn(17, generator=value.noise_generators[domain]),
                                   torch.randn(17, generator=restored.noise_generators[domain]))


def test_frozen_features_keep_gradients_through_resize_crop_and_normalization():
    model = FrozenImageFeatures(TinyDINO(), {"size": {"shortest_edge": 12},
                                           "crop_size": {"height": 8, "width": 8}, "resample": 3}, 2)
    pixels = torch.rand(2, 3, 10, 16, requires_grad=True)
    structure, tokens = model(pixels)
    (structure[:, 0].mean() + tokens.square().mean()).backward()
    assert pixels.grad.norm() > 0
    assert all(p.grad is None for p in model.parameters())


def test_reconstruction_semantic_dropout_and_null_code_are_used():
    from diffusion_ot.training.train_joint_infoot import _reconstruction_loss
    torch.manual_seed(4)
    ctx = domains()["cat"]
    configure_generator_adaptation(ctx.branch)
    ctx.branch.semantic_conditioner.dropout_probability = 1.0
    x = torch.randn(3, 4, 8, 8)
    z = torch.randn(3, 6, requires_grad=True)
    diagnostics = {}
    _reconstruction_loss(ctx, x, z, semantic_dropout=True, diagnostics=diagnostics).backward()
    assert diagnostics["semantic_dropout_fraction"] == 1
    assert z.grad is None or z.grad.count_nonzero() == 0
    assert ctx.branch.semantic_conditioner.null_token.grad is None
    assert sum(float(p.grad.norm()) for p in generator_parameter_view(ctx.branch).parameters() if p.grad is not None) > 0


def test_decoder_config_rejects_silent_partial_experiment():
    cfg = config()
    validate_decoder_config(cfg)
    for field, value in (("structure_weight", 0), ("num_steps", 0), ("batch_size", 5)):
        changed = deepcopy(cfg)
        changed["decoded_translation"][field] = value
        with pytest.raises(ValueError):
            validate_decoder_config(changed)
    changed = deepcopy(cfg)
    changed["trainable"]["base_transformers"] = True
    with pytest.raises(ValueError, match="backbone"):
        validate_decoder_config(changed)


@pytest.mark.parametrize("key,value", [("conditioned_preservation_weight", -1),
    ("conditioned_preservation_weight", float("nan")), ("conditioned_preservation_weight", float("inf")),
    ("conditioned_preservation_samples", 0), ("conditioned_preservation_samples", 1.5),
    ("conditioned_preservation_samples", True)])
def test_conditioned_preservation_options_are_validated(key, value):
    cfg = config()
    cfg["generator_adaptation"][key] = value
    with pytest.raises(ValueError, match="generator_adaptation.conditioned_preservation"):
        validate_decoder_config(cfg)


def test_conditioned_preservation_is_g_only_and_restores_a_perturbed_native_function(monkeypatch, tmp_path):
    import diffusion_ot.training.decoded_translation as decoded
    torch.manual_seed(81)
    ctx, cfg = domains(), config()
    cfg["generator_adaptation"].update(conditioned_preservation_weight=.05, conditioned_preservation_samples=2)
    monkeypatch.setattr(decoded, "load_image_features", tiny_features)
    trainer = DecoderTraining(cfg, ctx, None, tmp_path, seed=4)
    latents = {d: torch.randn(4, 4, 8, 8, requires_grad=True) for d in ctx}
    codes = {d: v.branch.encoder(latents[d]) for d,v in ctx.items()}
    torch.manual_seed(18)
    assert trainer.conditioned_preservation_loss(latents, codes).item() == pytest.approx(0, abs=1e-12)
    frozen_teacher = deepcopy(trainer.baselines)
    # A native conditioning-path change should be observable and correctable.
    with torch.no_grad():
        for view in trainer.views.values():
            for p in view["adapters"].parameters():
                p.add_(.03 * torch.randn_like(p))
    before = {d: deepcopy(v.branch.semantic_transformer.state_dict()) for d,v in ctx.items()}
    torch.manual_seed(18)
    loss = trainer.conditioned_preservation_loss(latents, codes)
    assert loss > 0
    loss.backward()
    for d,v in ctx.items():
        assert latents[d].grad is None
        assert all(p.grad is None for p in v.branch.encoder.parameters())
        assert v.branch.semantic_conditioner.null_token.grad is None
        assert all(p.grad is None for p in v.branch.semantic_transformer.base.x_embedder.parameters())
        assert all(torch.equal(x, trainer.baselines[d][k]) for k,x in frozen_teacher[d].items())
        # Teacher evaluation never changes the live parameters, even during backward.
        assert all(torch.equal(x, v.branch.semantic_transformer.state_dict()[k]) for k,x in before[d].items())
    norm = sum(p.grad.square().sum() for p in trainer.parameters if p.grad is not None).sqrt()
    assert norm > 0
    with torch.no_grad():
        for p in trainer.parameters:
            if p.grad is not None:
                p.add_(p.grad, alpha=-1e-3)
    torch.manual_seed(18)
    assert trainer.conditioned_preservation_loss(latents, codes) < loss


def test_disabled_conditioned_preservation_skips_teacher_and_keeps_rng(monkeypatch, tmp_path):
    import diffusion_ot.training.decoded_translation as decoded
    monkeypatch.setattr(decoded, "load_image_features", tiny_features)
    trainer = DecoderTraining(config(), domains(), None, tmp_path, seed=4)
    def unexpected(*args, **kwargs):
        raise AssertionError("Disabled preservation must not run a model forward.")
    monkeypatch.setattr(trainer, "_prediction_preservation_loss", unexpected)
    rng = torch.get_rng_state().clone()
    assert trainer.conditioned_preservation_loss({}, {}).item() == 0
    assert torch.equal(torch.get_rng_state(), rng)
