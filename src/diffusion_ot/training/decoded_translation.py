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
from torch.func import functional_call
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from diffusion_ot.losses.contrastive import detached_key_contrastive_loss
from diffusion_ot.losses.semantic_prior import patch_structure_descriptor
from diffusion_ot.models.generator_adaptation import (
    baseline_parameter_snapshot, configure_generator_adaptation,
    generator_adaptation_enabled, load_joint_generator,
    predict_with_parameters,
)
from diffusion_ot.models.pdae_sit import make_null_class_labels
from diffusion_ot.training.diffaugment import diffaugment_options, random_translation


def validate_decoder_config(config: dict[str, Any]) -> None:
    adapted = generator_adaptation_enabled(config)
    image = config.get("decoded_translation") or {}
    diffaugment_options(image.get("diffaugment"))
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
    conditioned_weight = float(options.get("conditioned_preservation_weight", 0.0))
    if not math.isfinite(conditioned_weight) or conditioned_weight < 0:
        raise ValueError("generator_adaptation.conditioned_preservation_weight must be finite and nonnegative.")
    count = options.get("conditioned_preservation_samples", 4)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("generator_adaptation.conditioned_preservation_samples must be a positive integer.")
    for name, default in (("structure_weight", 0.1), ("adversarial_weight", 0.01),
                          ("discriminator_lr", 1e-4), ("discriminator_grad_clip", 1.0)):
        value = float(image.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"decoded_translation.{name} must be finite and positive.")
    for name in ("structure_contrastive_weight", "code_consistency_weight"):
        value = float(image.get(name, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"decoded_translation.{name} must be finite and nonnegative.")
    temperature = float(image.get("structure_contrastive_temperature", .2))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("decoded_translation.structure_contrastive_temperature must be finite and positive.")
    similarity = float(image.get("structure_contrastive_negative_similarity_threshold", .95))
    if not math.isfinite(similarity) or not -1 <= similarity <= 1:
        raise ValueError("decoded_translation.structure_contrastive_negative_similarity_threshold must be in [-1, 1].")
    if image.get("code_consistency_mode", "cosine") not in ("cosine", "contrastive"):
        raise ValueError("decoded_translation.code_consistency_mode must be cosine or contrastive.")
    code_temperature = float(image.get("code_consistency_temperature", .2))
    if not math.isfinite(code_temperature) or code_temperature <= 0:
        raise ValueError("decoded_translation.code_consistency_temperature must be finite and positive.")
    code_similarity = float(image.get("code_consistency_negative_similarity_threshold", .95))
    if not math.isfinite(code_similarity) or not -1 <= code_similarity <= 1:
        raise ValueError("decoded_translation.code_consistency_negative_similarity_threshold must be in [-1, 1].")
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


def decoded_structure_contrastive_loss(generated, source, negatives, *, temperature=.2,
                                        negative_similarity_threshold=.95):
    """Retrieve the source structure among detached, disjoint reference images.

    The positive is the generated image's own source. Near-duplicate source
    structures are masked rather than treated as negatives. This deliberately
    uses the full reference bank, not only the few images decoded this step.
    No usable negative means no contrastive supervision for that row.
    """
    return detached_key_contrastive_loss(
        generated, source, negatives, temperature=temperature,
        negative_similarity_threshold=negative_similarity_threshold)


def generated_code_consistency_loss(encoder, generated_latents, condition, *, mode="cosine",
                                    condition_bank=None, temperature=.2,
                                    negative_similarity_threshold=.95):
    """Recover a detached full-mean condition through a fixed live readout.

    Detached functional parameters preserve derivatives with respect to the
    generated latent without training the semantic readout to agree with G.
    Buffer copies also prevent a stateful readout from mutating training state.
    The current PDAE encoder uses GroupNorm/LayerNorm and has no stochastic
    layers. The trainer routes this objective to generator parameters only;
    callers must not backpropagate it into the conditioning encoder/head path.
    Contrastive recovery retrieves the supplied condition against other fresh
    projected query conditions, including queries not decoded this update.
    The default cosine mode is retained for old configs/checkpoints only.
    This is latent recovery, not an RGB decode/re-encode or realism metric.
    """
    if mode not in ("cosine", "contrastive"):
        raise ValueError("Code consistency mode must be cosine or contrastive.")
    parameters = {name: value.detach() for name, value in encoder.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in encoder.named_buffers()}
    recovered = functional_call(encoder, (parameters, buffers), (generated_latents,), strict=True).float()
    target = condition.detach().to(recovered)
    if recovered.ndim != 2 or recovered.shape != target.shape:
        raise ValueError("Recovered and conditioning semantic codes must have matching [batch, features] shapes.")
    if not torch.isfinite(recovered).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Non-finite generated semantic-code recovery.")
    target_norm, recovered_norm = target.norm(dim=-1), recovered.norm(dim=-1)
    valid = target_norm > 1e-6
    cosine = F.cosine_similarity(recovered, target, dim=-1, eps=1e-6)
    contrastive_metrics = None
    if mode == "contrastive":
        if condition_bank is None:
            raise ValueError("Contrastive code consistency requires the full projected condition bank.")
        loss, contrastive_metrics = detached_key_contrastive_loss(
            recovered, target, condition_bank, temperature=temperature,
            negative_similarity_threshold=negative_similarity_threshold)
    else:
        loss = (1 - cosine[valid]).mean() if valid.any() else recovered.sum() * 0
    metrics = {"loss": float(loss.detach()), "mode": mode, "samples": len(recovered),
               "valid_conditions": int(valid.sum()), "readout": "final_diffusion_latent",
               "matched_cosine": float(cosine[valid].detach().mean()) if valid.any() else None,
               "recovered_to_condition_norm_ratio": float((recovered_norm[valid] / target_norm[valid]).detach().mean()) if valid.any() else None,
               "condition_norm": float(target_norm.mean()),
               "recovered_norm": float(recovered_norm.detach().mean()),
               "condition_batch_variance": float(target.var(dim=0, unbiased=False).mean()),
               "recovered_batch_variance": float(recovered.detach().var(dim=0, unbiased=False).mean()),
               "shuffled_cosine": None, "matched_minus_shuffled_cosine": None,
               "retrieval_top1": None}
    # Compare every mismatched condition deterministically. No RNG consumption,
    # and no retrieval/gap claim for a singleton or invalid-condition batch.
    with torch.no_grad():
        if int(valid.sum()) >= 2:
            similarities = F.normalize(recovered[valid], dim=-1, eps=1e-6) @ F.normalize(target[valid], dim=-1, eps=1e-6).T
            identity = torch.eye(len(similarities), device=similarities.device, dtype=torch.bool)
            shuffled = similarities[~identity].mean()
            metrics.update(shuffled_cosine=float(shuffled),
                           matched_minus_shuffled_cosine=float(similarities.diag().mean() - shuffled),
                           retrieval_top1=float((similarities.argmax(-1) == torch.arange(len(similarities), device=similarities.device)).float().mean()))
    if contrastive_metrics is not None:
        # Keep the previous small decoded-batch retrieval diagnostic separate.
        # The primary metric uses the larger masked bank and tie-aware accuracy.
        metrics["decoded_batch_retrieval_top1"] = metrics["retrieval_top1"]
        metrics.update(contrastive_metrics)
        metrics["condition_bank_size"] = len(condition_bank)
        metrics["negative_source"] = "current_projected_query_conditions"
    return loss, metrics


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
        self.diffaugment = diffaugment_options(self.options.get("diffaugment"))
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
        # Augmentation runs where DINO lives, including for a remote domain's
        # decoded images. It must not consume diffusion/global RNG streams.
        self.augmentation_generators = {
            d: torch.Generator(device=domains["cat"].device).manual_seed(seed + 1001 + i)
            for i, d in enumerate(domains)
        }
        self.discriminator_updates = 0
        # This component is returned separately because only G may optimize it.
        self.code_consistency_objective = torch.zeros((), device=domains["cat"].device)

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
        return self._prediction_preservation_loss(latents, codes, force_null=True)

    def conditioned_preservation_loss(self, latents, baseline_codes):
        """Protect native G(E_1A(x)) with a fixed teacher; gradients reach G only.

        Same-domain Stage 1A codes keep the reference function fixed while E
        adapts. Do not apply this teacher to cross-domain conditional means:
        those are precisely the inputs whose decoding Stage 1B should improve.
        Zero weight preserves the previous compute cost and RNG sequence.
        """
        options = self.config["generator_adaptation"]
        if float(options.get("conditioned_preservation_weight", 0.0)) == 0:
            return torch.zeros((), device=self.domains["cat"].device)
        count = int(options.get("conditioned_preservation_samples", 4))
        return self._prediction_preservation_loss(
            {d: x[:count] for d, x in latents.items()},
            {d: z[:count] for d, z in baseline_codes.items()}, force_null=False,
        )

    def _prediction_preservation_loss(self, latents, codes, *, force_null):
        from diffusion_ot.losses.pdae_flow import make_linear_flow_target
        losses = []
        for domain, value in self.domains.items():
            flow = value.training_config.get("flow") or {}
            target = make_linear_flow_target(latents[domain].detach(), eps=float(flow.get("time_eps", 1e-5)),
                                             direction="noise_to_data")
            condition = codes[domain].detach()
            if force_null:
                condition = value.branch.semantic_null_like(condition)
            labels = make_null_class_labels(value.transformer, len(condition), value.device,
                                            (value.training_config.get("class_conditioning") or {}).get("null_label"))
            with torch.no_grad():
                teacher = predict_with_parameters(value.branch, self.baselines[domain],
                                                   target.x_t, target.t, condition, labels).sample
            student = value.branch.predict_with_z(target.x_t, target.t, condition, class_labels=labels).sample
            losses.append((student.float() - teacher.float()).square().mean().to(self.domains["cat"].device))
        return torch.stack(losses).mean()

    def loss(self, weights, references, source_latents, real_latents, *, step, validation_seed=None,
             source_reference_structure=None, source_query_structure=None):
        evaluation = validation_seed is not None
        augment = self.diffaugment["enabled"] and not evaluation and self.diffaugment["translation_ratio"] > 0
        feature_device = next(self.features.parameters()).device
        structure_losses, adversarial_losses, metrics = [], [], {}
        metrics["adversarial_augmentation"] = {**self.diffaugment, "applied": augment}
        contrastive_losses, consistency_losses = [], []
        contrastive_weight = float(self.options.get("structure_contrastive_weight", 0.0))
        consistency_weight = float(self.options.get("code_consistency_weight", 0.0))
        consistency_mode = self.options.get("code_consistency_mode", "cosine")
        self.code_consistency_objective = torch.zeros((), device=feature_device)
        if contrastive_weight > 0 and any(bank is None or any(domain not in bank for domain in self.domains)
                                         for bank in (source_reference_structure, source_query_structure)):
            raise ValueError("Decoded contrastive supervision requires cached source query and disjoint reference structures for both domains.")
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
            if augment:
                # Structure objectives retain the original image coordinates.
                # Draw once outside DINO checkpoints; reuse these live tokens
                # for G and detached tokens for D with identical fake offsets.
                _, fake_tokens = self.features(random_translation(
                    images, ratio=self.diffaugment["translation_ratio"],
                    generator=self.augmentation_generators[target]))
            with torch.no_grad():
                source_context = self.domains[source]
                source_images = decode_training_images(source_context.vae, source_latents[source][:count].to(
                    source_context.device, dtype=source_context.dtype)).to(feature_device)
                source_structure, _ = self.features(source_images)
                real_images = decode_training_images(context.vae, real_latents[target][:count].to(
                    context.device, dtype=context.dtype)).to(feature_device)
                if augment:
                    # Real and fake images use the same distribution, with
                    # independent per-image offsets (fake draws precede real).
                    real_images = random_translation(
                        real_images, ratio=self.diffaugment["translation_ratio"],
                        generator=self.augmentation_generators[target])
                _, real_tokens = self.features(real_images)
            structure_loss = (1 - F.cosine_similarity(structure, source_structure.detach(), dim=-1)).mean()
            structure_losses.append(structure_loss)
            pairs[target] = (fake_tokens, real_tokens)
            metrics[f"{source}_to_{target}"] = {"structure_loss": float(structure_loss.detach()),
                                               "samples": count, "reference_targets": references[target].shape[0]}
            if contrastive_weight > 0:
                contrastive, contrastive_metrics = decoded_structure_contrastive_loss(
                    structure, source_query_structure[source][:count], source_reference_structure[source],
                    temperature=float(self.options.get("structure_contrastive_temperature", .2)),
                    negative_similarity_threshold=float(self.options.get("structure_contrastive_negative_similarity_threshold", .95)),
                )
                contrastive_losses.append(contrastive)
                metrics[f"{source}_to_{target}"]["structure_contrastive"] = contrastive_metrics
            if consistency_weight > 0:
                condition_bank = None
                if consistency_mode == "contrastive":
                    # All current query means, not raw target codes or a stale
                    # queue. Undecoded queries add negatives without rollouts.
                    with torch.no_grad():
                        condition_bank = (weights[f"{source}_to_{target}"].detach()
                                          @ references[target].detach().float()).to(context.device, dtype=context.dtype)
                        # Exactly the conditions actually supplied to G, even
                        # if GEMM rounding differs for the larger matrix.
                        condition_bank[:count] = codes.detach()
                consistency, consistency_metrics = generated_code_consistency_loss(
                    context.branch.encoder, generated, codes, mode=consistency_mode,
                    condition_bank=condition_bank,
                    temperature=float(self.options.get("code_consistency_temperature", .2)),
                    negative_similarity_threshold=float(self.options.get("code_consistency_negative_similarity_threshold", .95)))
                consistency_losses.append(consistency.to(feature_device))
                metrics[f"{source}_to_{target}"]["code_consistency"] = consistency_metrics
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
            metrics["discriminator_clip_scale"] = min(
                1.0, float(self.options.get("discriminator_grad_clip", 1.0)) / max(float(norm), 1e-12))
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
        ramp = 1.0 if evaluation else self.ramp(step)
        if contrastive_losses:
            contrastive = torch.stack(contrastive_losses).mean()
            total = total + contrastive_weight * contrastive
            metrics.update(structure_contrastive_loss=float(contrastive.detach()),
                           structure_contrastive_weight=contrastive_weight)
        if consistency_losses:
            consistency = torch.stack(consistency_losses).mean()
            self.code_consistency_objective = ramp * consistency_weight * consistency
            metrics.update(code_consistency_loss=float(consistency.detach()),
                           code_consistency_mode=consistency_mode,
                           code_consistency_weight=consistency_weight,
                           code_consistency_weighted_loss=float(consistency.detach()) * consistency_weight,
                           code_consistency_effective_weighted_loss=float(self.code_consistency_objective.detach()),
                           code_consistency_gradient_routing="generator_only")
        metrics.update(structure_loss=float(structure.detach()), adversarial_loss=float(adversarial.detach()),
                       weighted_loss=float(total.detach()), ramp=ramp,
                       effective_weighted_loss=float(total.detach()) * ramp,
                       discriminator_updates=self.discriminator_updates,
                       num_steps=int(self.options.get("validation_num_steps", 50) if evaluation else self.options.get("num_steps", 50)))
        # Main loss reaches encoders, matching heads and G. Code recovery must
        # be consumed through autograd.grad(..., G_parameters) by the trainer.
        return total * ramp, metrics

    def checkpoint_state(self):
        state = {"generators": {d: self._cpu_copy(v.branch.generator_state_dict()) for d, v in self.domains.items()},
                "fixed_generators": self.fixed_generators,
                "generator_baseline_parameters": self.baselines,
                "generator_ema": self.ema.export() if self.ema else None,
                "generator_ema_state": self.ema.state_dict() if self.ema else None,
                "decoded_discriminators": self._cpu_copy(self.discriminators.state_dict()),
                "decoded_discriminator_optimizer": self.optimizer.state_dict(),
                "decoded_discriminator_updates": self.discriminator_updates,
                "decoded_noise_states": {d: gen.get_state().cpu() for d, gen in self.noise_generators.items()}}
        if self.diffaugment["enabled"]:
            state["decoded_augmentation_states"] = {
                d: gen.get_state().cpu() for d, gen in self.augmentation_generators.items()}
        return state

    def load_checkpoint(self, payload):
        if self.diffaugment["enabled"]:
            states = payload.get("decoded_augmentation_states")
            if not isinstance(states, dict) or set(states) != set(self.augmentation_generators):
                raise ValueError("DiffAugment resume requires per-domain decoded_augmentation_states; start a fresh run for a changed objective.")
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
        if self.diffaugment["enabled"]:
            for d, gen in self.augmentation_generators.items():
                gen.set_state(payload["decoded_augmentation_states"][d].cpu())
