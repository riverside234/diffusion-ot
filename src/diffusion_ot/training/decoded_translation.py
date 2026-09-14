"""Experiment D: differentiable full-mean decoding and image supervision.

The frozen DINOv2 network supplies spatial structure and patch/global features
for two small trainable discriminators. These are project extensions, not
objectives from the official InfoOT implementation. No paired target is assumed.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from diffusion_ot.losses.semantic_prior import patch_structure_descriptor
from diffusion_ot.models.generator_adaptation import (
    baseline_parameter_snapshot, configure_generator_adaptation,
    generator_adaptation_enabled, load_joint_generator,
    predict_with_parameters,
)
from diffusion_ot.models.pdae_sit import make_null_class_labels


def validate_decoder_config(config: dict[str, Any]) -> None:
    adapted = generator_adaptation_enabled(config)
    image = config.get("decoded_translation") or {}
    enabled = bool(image.get("enabled", False))
    trainable = config.get("trainable") or {}
    if trainable.get("base_transformers", False):
        raise ValueError("The pretrained SiT backbone must stay frozen.")
    if not adapted:
        if enabled or trainable.get("adapters", False) or trainable.get("attention_lora", False):
            raise ValueError("Enable generator_adaptation for trainable G / decoded translation.")
        return
    if not all(trainable.get(k, False) for k in ("encoders", "adapters", "attention_lora")):
        raise ValueError("Experiment D requires trainable encoders, adapters, and attention_lora.")
    if not enabled or not (config.get("conditional_structure") or {}).get("enabled", False):
        raise ValueError("Experiment D requires decoded_translation and conditional_structure.")
    if not (config.get("matching_head") or {}).get("enabled", False):
        raise ValueError("Experiment D requires matching heads.")
    options = config["generator_adaptation"]
    for name, default in (("lr_adapter", 1e-5), ("lr_lora", 5e-6), ("grad_clip_norm", 1.0),
                          ("null_preservation_weight", .1)):
        value = float(options.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"generator_adaptation.{name} must be finite and positive.")
    for name, default in (("structure_weight", 0.1), ("adversarial_weight", 0.01),
                          ("discriminator_lr", 1e-4), ("discriminator_grad_clip", 1.0)):
        value = float(image.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"decoded_translation.{name} must be finite and positive.")
    for name, default in (("batch_size", 4), ("num_steps", 50), ("validation_num_steps", 50),
                          ("every_steps", 1), ("warmup_steps", 2000), ("discriminator_hidden_dim", 256)):
        if int(image.get(name, default)) < 1:
            raise ValueError(f"decoded_translation.{name} must be positive.")
    if int(image.get("start_step", 0)) < 0:
        raise ValueError("decoded_translation.start_step must be nonnegative.")
    if not math.isfinite(float(image.get("guidance_scale", 1.0))):
        raise ValueError("decoded_translation.guidance_scale must be finite.")
    count = int((config.get("conditional_structure") or {}).get("query_samples_per_domain", 32))
    if int(image.get("batch_size", 4)) > count:
        raise ValueError("Decoded batch_size cannot exceed the disjoint conditional query count.")
    if (config.get("projection_support") or {}).get("enabled", False):
        raise ValueError("Experiment D disables the conditional-mean support loss.")


def integrate_training_flow(branch, transformer, initial_state, z, *, num_steps,
                            guidance_scale=1.0, null_label=None, checkpoint_steps=True):
    """Same midpoint-time Euler rule as evaluation, with an intact autograd graph."""
    if int(num_steps) < 1 or not math.isfinite(float(guidance_scale)):
        raise ValueError("Invalid differentiable flow integration settings.")
    labels = make_null_class_labels(transformer, len(z), initial_state.device, null_label)
    state = initial_state.clone()
    dt = 1.0 / int(num_steps)
    for index in range(int(num_steps)):
        timestep = torch.full((len(z),), (index + .5) * dt,
                              device=state.device, dtype=torch.float32)
        # Bind time at each step: checkpoint recomputation must not capture the
        # final value of the loop variable. The whole prediction reenters the
        # semantic-LoRA context, including during backward.
        def velocity(current, code, time=timestep):
            return branch.predict_cfg_with_z(current, time, code,
                                             guidance_scale=guidance_scale,
                                             class_labels=labels).sample
        if checkpoint_steps and torch.is_grad_enabled():
            value = checkpoint(velocity, state, z, use_reentrant=False)
        else:
            value = velocity(state, z)
        state = state + dt * value
    return state


def decode_training_images(vae, latents, *, checkpoint_decode=True):
    dtype = next(vae.parameters()).dtype
    scale = float(getattr(vae.config, "scaling_factor", 1.0))
    def decode(value):
        result = vae.decode((value / scale).to(dtype=dtype))
        return result.sample if hasattr(result, "sample") else result[0]
    pixels = (checkpoint(decode, latents, use_reentrant=False)
              if checkpoint_decode and torch.is_grad_enabled() else decode(latents))
    return ((pixels.float() + 1) / 2).clamp(0, 1)


class FrozenImageFeatures(nn.Module):
    def __init__(self, model, processor: dict, grid_size: int, *, checkpoint_features=True):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.processor = deepcopy(processor)
        self.grid_size = int(grid_size)
        self.feature_dim = int(model.config.hidden_size)
        self.checkpoint_features = bool(checkpoint_features)
        self.register_buffer("mean", torch.tensor(processor.get("image_mean", [.485, .456, .406])).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(processor.get("image_std", [.229, .224, .225])).view(1, 3, 1, 1))

    def preprocess(self, images):
        # A torch-only counterpart to the cached processor. Never send generated
        # images through PIL/NumPy/AutoImageProcessor: those detach gradients.
        value = images.float()
        if self.processor.get("do_resize", True):
            size = self.processor.get("size", {"shortest_edge": 256})
            if "shortest_edge" in size:
                h, w = value.shape[-2:]
                short = int(size["shortest_edge"])
                target = (short, int(w * short / h)) if h <= w else (int(h * short / w), short)
            else:
                target = (int(size["height"]), int(size["width"]))
            mode = {2: "bilinear", 3: "bicubic"}.get(int(self.processor.get("resample", 3)))
            if mode is None:
                raise ValueError("Decoded DINO preprocessing supports bilinear/bicubic resizing.")
            value = F.interpolate(value, size=target, mode=mode, align_corners=False, antialias=True)
        if self.processor.get("do_center_crop", True):
            crop = self.processor.get("crop_size", {"height": 224, "width": 224})
            h, w = int(crop["height"]), int(crop["width"])
            if min(value.shape[-2] - h, value.shape[-1] - w) < 0:
                raise ValueError("DINO crop exceeds resized image dimensions.")
            top, left = (value.shape[-2] - h) // 2, (value.shape[-1] - w) // 2
            value = value[..., top:top+h, left:left+w]
        return (value - self.mean) / self.std if self.processor.get("do_normalize", True) else value

    def forward(self, images):
        value = self.preprocess(images)
        def extract(pixels):
            return self.model(pixel_values=pixels).last_hidden_state
        tokens = (checkpoint(extract, value, use_reentrant=False)
                  if self.checkpoint_features and torch.is_grad_enabled() else extract(value))
        return patch_structure_descriptor(tokens[:, 1:], self.grid_size), tokens.float()


def load_image_features(prior, root: Path, device: str, *, checkpoint_features=True):
    from transformers import AutoModel
    metadata = prior.metadata
    if metadata.get("descriptor") != "pooled_patch_self_similarity_v1":
        raise ValueError("Decoded training requires the pooled DINOv2 structure cache.")
    revision = metadata.get("resolved_revision")
    if not revision:
        raise ValueError("Rebuild the structure cache with a resolved model revision for experiment D.")
    model = AutoModel.from_pretrained(metadata["model"], revision=revision,
                                     cache_dir=root / "artifacts" / "pretrained" / "dinov2",
                                     attn_implementation="eager")
    return FrozenImageFeatures(model, metadata["processor"], metadata["grid_size"],
                               checkpoint_features=checkpoint_features).to(device).eval()


class FeatureDiscriminator(nn.Module):
    """Hinge discriminator on frozen DINO CLS + patch tokens (not pixels)."""
    def __init__(self, width, hidden=256):
        super().__init__()
        from torch.nn.utils.parametrizations import spectral_norm
        self.layers = nn.Sequential(nn.LayerNorm(width),
                                    spectral_norm(nn.Linear(width, hidden)), nn.LeakyReLU(.2),
                                    spectral_norm(nn.Linear(hidden, 1)))

    def forward(self, tokens):
        scores = self.layers(tokens).squeeze(-1)
        return .5 * (scores[:, 0] + scores[:, 1:].mean(1))


@contextmanager
def frozen_discriminator(module):
    mode = module.training
    flags = [p.requires_grad for p in module.parameters()]
    module.eval().requires_grad_(False)
    try:
        yield
    finally:
        module.train(mode)
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)


class DecoderTraining:
    def __init__(self, config, domains, prior, root, *, seed):
        from diffusion_ot.training.train_joint_infoot import EncoderEMA
        self.config, self.options, self.domains = config, config["decoded_translation"], domains
        self.views = {d: configure_generator_adaptation(v.branch) for d, v in domains.items()}
        self.parameters = [p for view in self.views.values() for p in view.parameters()]
        self.baselines = {d: baseline_parameter_snapshot(v.branch, self.views[d]) for d, v in domains.items()}
        self.fixed_generators = {d: self._cpu_copy(v.branch.generator_state_dict()) for d, v in domains.items()}
        self.features = load_image_features(prior, root, domains["cat"].device,
                                            checkpoint_features=self.options.get("checkpoint_features", True))
        for value in domains.values():
            if value.vae is None:
                raise ValueError("Experiment D requires the frozen VAE decoder.")
            value.vae.eval().requires_grad_(False)
            if (value.training_config.get("flow") or {}).get("direction", "noise_to_data") != "noise_to_data":
                raise ValueError("Decoded training currently requires noise_to_data flow.")
            if abs(value.branch.semantic_conditioner.dropout_probability - .1) > 1e-8:
                raise ValueError("Experiment D preserves semantic dropout_probability=0.1.")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 801)
            self.discriminators = nn.ModuleDict({d: FeatureDiscriminator(
                self.features.feature_dim, int(self.options.get("discriminator_hidden_dim", 256))) for d in domains})
        self.discriminators.to(domains["cat"].device)
        self.optimizer = torch.optim.AdamW(self.discriminators.parameters(),
                                           lr=float(self.options.get("discriminator_lr", 1e-4)),
                                           betas=(0.0, .99), weight_decay=0)
        ema = config.get("ema") or {}
        self.ema = EncoderEMA(self.views, decay=float(ema.get("decay", .995)),
                              warmup_steps=int(ema.get("warmup_steps", 500))) if ema.get("enabled", True) else None
        # Dedicated per-domain streams keep validation/resume independent of
        # reconstruction noise and prevent fixed-noise translation overfitting.
        self.noise_generators = {d: torch.Generator(device=v.device).manual_seed(seed + 901 + i)
                                 for i, (d, v) in enumerate(domains.items())}
        self.discriminator_updates = 0

    @staticmethod
    def _cpu_copy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: DecoderTraining._cpu_copy(v) for k, v in value.items()}
        return deepcopy(value)

    def parameter_groups(self):
        options = self.config["generator_adaptation"]
        return [{"params": [p for view in self.views.values() for p in view[key].parameters()],
                 "lr": float(options.get(rate, default)), "name": key}
                for key, rate, default in (("adapters", "lr_adapter", 1e-5), ("lora", "lr_lora", 5e-6))]

    def ramp(self, step):
        return min(max(step - int(self.options.get("start_step", 0)), 0)
                   / int(self.options.get("warmup_steps", 2000)), 1.0)

    def active(self, step):
        return self.ramp(step) > 0 and step % int(self.options.get("every_steps", 1)) == 0

    def null_preservation_loss(self, latents, codes):
        from diffusion_ot.losses.pdae_flow import make_linear_flow_target
        losses = []
        for domain, value in self.domains.items():
            flow = value.training_config.get("flow") or {}
            target = make_linear_flow_target(latents[domain], eps=float(flow.get("time_eps", 1e-5)),
                                             direction="noise_to_data")
            null_z = value.branch.semantic_null_like(codes[domain].detach())
            labels = make_null_class_labels(value.transformer, len(null_z), value.device,
                                            (value.training_config.get("class_conditioning") or {}).get("null_label"))
            with torch.no_grad():
                teacher = predict_with_parameters(value.branch, self.baselines[domain],
                                                   target.x_t, target.t, null_z, labels).sample
            student = value.branch.predict_with_z(target.x_t, target.t, null_z, class_labels=labels).sample
            losses.append((student.float() - teacher.float()).square().mean().to(self.domains["cat"].device))
        return torch.stack(losses).mean()

    def loss(self, weights, references, source_latents, real_latents, *, step, validation_seed=None):
        evaluation = validation_seed is not None
        feature_device = next(self.features.parameters()).device
        structure_losses, adversarial_losses, metrics = [], [], {}
        pairs = {}
        for offset, (source, target) in enumerate((("cat", "dog"), ("dog", "cat"))):
            context = self.domains[target]
            count = min(int(self.options.get("batch_size", 4)), len(weights[f"{source}_to_{target}"]))
            probability = weights[f"{source}_to_{target}"][:count]
            # Average every raw target reference. No top-k, sampling, or detach.
            codes = (probability @ references[target].float()).to(context.device, dtype=context.dtype)
            generator = (torch.Generator(device=context.device).manual_seed(validation_seed + offset)
                         if evaluation else self.noise_generators[target])
            noise = torch.randn((count, *source_latents[source].shape[1:]), generator=generator,
                                device=context.device, dtype=context.dtype)
            generated = integrate_training_flow(
                context.branch, context.transformer, noise, codes,
                num_steps=int(self.options.get("validation_num_steps", 50) if evaluation else self.options.get("num_steps", 50)),
                guidance_scale=float(self.options.get("guidance_scale", 1.0)),
                null_label=(context.training_config.get("class_conditioning") or {}).get("null_label"),
                checkpoint_steps=bool(self.options.get("checkpoint_sampler", True)),
            )
            images = decode_training_images(context.vae, generated).to(feature_device)
            structure, fake_tokens = self.features(images)
            with torch.no_grad():
                source_context = self.domains[source]
                source_images = decode_training_images(source_context.vae, source_latents[source][:count].to(
                    source_context.device, dtype=source_context.dtype)).to(feature_device)
                source_structure, _ = self.features(source_images)
                real_images = decode_training_images(context.vae, real_latents[target][:count].to(
                    context.device, dtype=context.dtype)).to(feature_device)
                _, real_tokens = self.features(real_images)
            structure_loss = (1 - F.cosine_similarity(structure, source_structure.detach(), dim=-1)).mean()
            structure_losses.append(structure_loss)
            pairs[target] = (fake_tokens, real_tokens)
            metrics[f"{source}_to_{target}"] = {"structure_loss": float(structure_loss.detach()),
                                               "samples": count, "reference_targets": references[target].shape[0]}
        if not evaluation:
            self.optimizer.zero_grad(set_to_none=True)
            self.discriminators.train().requires_grad_(True)
            discriminator_loss = sum((F.relu(1 - self.discriminators[d](real.detach())).mean()
                                      + F.relu(1 + self.discriminators[d](fake.detach())).mean()) / 2
                                     for d, (fake, real) in pairs.items()) / len(pairs)
            if not torch.isfinite(discriminator_loss):
                raise FloatingPointError("Non-finite decoded discriminator loss.")
            discriminator_loss.backward()
            norm = nn.utils.clip_grad_norm_(self.discriminators.parameters(),
                                            float(self.options.get("discriminator_grad_clip", 1.0)), error_if_nonfinite=True)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.discriminator_updates += 1
            metrics["discriminator_loss"] = float(discriminator_loss.detach())
            metrics["discriminator_gradient_norm"] = float(norm)
        with frozen_discriminator(self.discriminators):
            for target, (fake, real) in pairs.items():
                fake_score, real_score = self.discriminators[target](fake), self.discriminators[target](real.detach())
                adversarial_losses.append(-fake_score.mean())
                metrics[f"{target}_real_score"] = float(real_score.detach().mean())
                metrics[f"{target}_fake_score"] = float(fake_score.detach().mean())
        structure = torch.stack(structure_losses).mean()
        adversarial = torch.stack(adversarial_losses).mean()
        total = (float(self.options.get("structure_weight", .1)) * structure
                 + float(self.options.get("adversarial_weight", .01)) * adversarial)
        metrics.update(structure_loss=float(structure.detach()), adversarial_loss=float(adversarial.detach()),
                       weighted_loss=float(total.detach()), ramp=1.0 if evaluation else self.ramp(step),
                       discriminator_updates=self.discriminator_updates,
                       num_steps=int(self.options.get("validation_num_steps", 50) if evaluation else self.options.get("num_steps", 50)))
        return total * (1.0 if evaluation else self.ramp(step)), metrics

    def checkpoint_state(self):
        return {"generators": {d: self._cpu_copy(v.branch.generator_state_dict()) for d, v in self.domains.items()},
                "fixed_generators": self.fixed_generators,
                "generator_baseline_parameters": self.baselines,
                "generator_ema": self.ema.export() if self.ema else None,
                "generator_ema_state": self.ema.state_dict() if self.ema else None,
                "decoded_discriminators": self._cpu_copy(self.discriminators.state_dict()),
                "decoded_discriminator_optimizer": self.optimizer.state_dict(),
                "decoded_discriminator_updates": self.discriminator_updates,
                "decoded_noise_states": {d: gen.get_state().cpu() for d, gen in self.noise_generators.items()}}

    def load_checkpoint(self, payload):
        for domain, value in self.domains.items():
            load_joint_generator(value.branch, payload, domain, weights="raw")
        self.fixed_generators = payload["fixed_generators"]
        baselines = payload["generator_baseline_parameters"]
        for d, current in self.baselines.items():
            if set(current) != set(baselines[d]):
                raise ValueError("Frozen generator baseline parameter mismatch.")
        self.baselines = baselines
        if self.ema is not None:
            if payload.get("generator_ema_state") is None:
                raise ValueError("Resume checkpoint has no generator EMA state.")
            self.ema.load_state_dict(payload["generator_ema_state"], self.views)
        self.discriminators.load_state_dict(payload["decoded_discriminators"], strict=True)
        self.optimizer.load_state_dict(payload["decoded_discriminator_optimizer"])
        self.discriminator_updates = int(payload["decoded_discriminator_updates"])
        for d, gen in self.noise_generators.items():
            gen.set_state(payload["decoded_noise_states"][d].cpu())
