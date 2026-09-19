"""Optional same-domain image fidelity and native-code InfoNCE refinement.

Real-data flow matching remains the main Stage 1A objective. Frozen DINO
features supervise decoded outputs against original RGB. Code recovery uses
the current encoder as a fixed differentiable readout of the final diffusion
latent and routes gradients only to G; it is not a VAE RGB round trip.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from diffusion_ot.data.ground_truth import load_ground_truth_images
from diffusion_ot.integrations.hf_snapshot import resolve_project_local_path
from diffusion_ot.losses.semantic_prior import SemanticPriorBank
from diffusion_ot.training.decoded_translation import (
    decode_training_images, generated_code_consistency_loss,
    integrate_training_flow, load_image_features,
)


def validate_refinement_config(config):
    options = config.get("refinement") or {}
    if not isinstance(options, dict):
        raise ValueError("refinement must be a mapping.")
    if not options.get("enabled", False):
        return
    if (config.get("flow") or {}).get("direction", "noise_to_data") != "noise_to_data":
        raise ValueError("Stage 1A refinement requires noise_to_data flow.")
    if not (config.get("adapter") or {}).get("freeze_base", True):
        raise ValueError("Stage 1A refinement requires a frozen SiT backbone.")
    for key, default in (("batch_size", 4), ("num_steps", 20), ("every_steps", 4),
                         ("warmup_steps", 500), ("validation_batch_size", 8),
                         ("validation_num_steps", 50)):
        value = options.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"refinement.{key} must be a positive integer.")
    for key, default in (("perceptual_weight", .05), ("structure_weight", .025),
                         ("code_contrastive_weight", .01)):
        value = float(options.get(key, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"refinement.{key} must be finite and nonnegative.")
    if not options.get("feature_prior_path"):
        raise ValueError("refinement.feature_prior_path must identify the pinned DINO cache.")
    temperature = float(options.get("code_temperature", .2))
    threshold = float(options.get("negative_similarity_threshold", .95))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("refinement.code_temperature must be finite and positive.")
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("refinement.negative_similarity_threshold must be in [-1, 1].")
    batch_size = int((config.get("dataloader") or {}).get("batch_size", 16))
    if options.get("batch_size", 4) > batch_size:
        raise ValueError("Refinement batch_size cannot exceed the native training batch.")
    if float(options.get("code_contrastive_weight", .01)) > 0 and batch_size < 2:
        raise ValueError("Native-code InfoNCE requires at least two training codes.")


def validate_refinement_resume(saved, current):
    old, new = saved.get("refinement") or {}, current.get("refinement") or {}
    if (old.get("enabled", False) or new.get("enabled", False)) and old != new:
        raise ValueError("Refinement objective changed: use train.initialize_from in a new output directory, not resume.")


@contextmanager
def deterministic_branch(branch):
    """Keep eval mode throughout checkpoint recomputation, then restore it."""
    was_training = branch.training
    branch.eval()
    try:
        yield
    finally:
        branch.train(was_training)


class Stage1ARefinement:
    def __init__(self, config, *, vae, data_config_path, root, device, seed):
        from diffusion_ot.data.afhq import load_afhq_dataset

        validate_refinement_config(config)
        self.options = dict(config["refinement"])
        self.vae = vae.eval().requires_grad_(False)
        self.data_config_path = data_config_path
        # Reuse pinned model/preprocessing provenance, not cached image targets:
        # targets below come from original RGB, with the current batch's flip.
        prior = SemanticPriorBank(resolve_project_local_path(
            self.options["feature_prior_path"], root, field_name="refinement.feature_prior_path"))
        self.prior_fingerprint = prior.fingerprint
        self.features = load_image_features(prior, root, device,
            checkpoint_features=self.options.get("checkpoint_features", True))
        self.original_dataset = load_afhq_dataset(data_config_path)
        self.generator = torch.Generator(device=torch.device(device)).manual_seed(seed + 3101)
        self.validation_seed = int(self.options.get("validation_seed", seed + 3102))

    def state_dict(self):
        return {"noise_state": self.generator.get_state(),
                "prior_fingerprint": self.prior_fingerprint}

    def load_state_dict(self, state):
        if not state or state.get("prior_fingerprint") != self.prior_fingerprint:
            raise ValueError("Missing refinement state or changed frozen-feature cache on resume.")
        self.generator.set_state(state["noise_state"].cpu())

    def scheduled(self, step):
        return (step - 1) % int(self.options.get("every_steps", 4)) == 0

    def losses(self, branch, transformer, batch, x0, *, step, null_label=None, validation=False):
        options = self.options
        count = min(len(x0), int(options.get("validation_batch_size", 8) if validation
                                else options.get("batch_size", 4)))
        steps = int(options.get("validation_num_steps", 50) if validation else options.get("num_steps", 20))
        ramp = 1.0 if validation else min(1.0, step / int(options.get("warmup_steps", 500)))
        generator = (torch.Generator(device=x0.device).manual_seed(self.validation_seed)
                     if validation else self.generator)
        code_weight = float(options.get("code_contrastive_weight", .01))
        # Full native minibatch supplies negatives; only a small prefix needs
        # expensive image rollouts. With zero code weight, no code recovery or
        # retrieval is computed. No semantic dropout for these conditions.
        bank = branch.encode(x0)
        condition = bank[:count]
        noise = torch.randn(x0[:count].shape, device=x0.device, dtype=x0.dtype, generator=generator)
        generated = integrate_training_flow(branch, transformer, noise, condition, num_steps=steps,
            guidance_scale=1.0, null_label=null_label,
            checkpoint_steps=options.get("checkpoint_steps", True))
        images = decode_training_images(self.vae, generated,
                                       checkpoint_decode=options.get("checkpoint_decode", True))
        originals = load_ground_truth_images(self.data_config_path, batch["metadata"][:count],
                                             dataset=self.original_dataset).to(images)
        flips = batch.get("horizontal_flip", [False] * len(x0))[:count]
        flip_mask = torch.as_tensor(flips, device=images.device, dtype=torch.bool)
        originals = torch.where(flip_mask[:, None, None, None], originals.flip(-1), originals)
        if originals.shape != images.shape:
            raise ValueError("Refinement generated and original RGB shapes differ; check cache preprocessing.")
        with torch.no_grad():
            target_structure, target_tokens = self.features(originals)
        structure, tokens = self.features(images)
        structure_loss = (1 - F.cosine_similarity(structure, target_structure, dim=-1)).mean()
        # Both global appearance and corresponding spatial tokens matter in
        # same-domain reconstruction (unlike a cross-domain appearance target).
        token_distance = 1 - F.cosine_similarity(tokens, target_tokens, dim=-1)
        perceptual_loss = .5 * (token_distance[:, 0].mean() + token_distance[:, 1:].mean())
        image_loss = ramp * (float(options.get("perceptual_weight", .05)) * perceptual_loss
                            + float(options.get("structure_weight", .025)) * structure_loss)
        code_weighted, code_metrics = image_loss.new_zeros(()), None
        if code_weight > 0:
            code_loss, code_metrics = generated_code_consistency_loss(
                branch.encoder, generated, condition, mode="contrastive", condition_bank=bank,
                temperature=float(options.get("code_temperature", .2)),
                negative_similarity_threshold=float(options.get("negative_similarity_threshold", .95)))
            code_metrics.update(negative_source="current_same_domain_native_codes")
            code_weighted = ramp * code_weight * code_loss
        if not bool(torch.isfinite(image_loss) & torch.isfinite(code_weighted)):
            raise FloatingPointError("Non-finite Stage 1A refinement loss.")
        with torch.no_grad():
            mse = (images - originals).square().flatten(1).mean(1)
            metrics = {"ramp": ramp, "samples": count, "num_steps": steps,
                       "target": "original_rgb", "guidance_scale": 1.0,
                       "sample_ids": batch["sample_id"][:count],
                       "perceptual_loss": float(perceptual_loss.detach()),
                       "structure_loss": float(structure_loss.detach()),
                       "weighted_image_loss": float(image_loss.detach()),
                       "weighted_code_loss": float(code_weighted.detach()),
                       "weighted_loss": float((image_loss + code_weighted).detach()),
                       "rgb_mse": float(mse.mean()),
                       "rgb_mae": float((images - originals).abs().mean()),
                       "rgb_psnr": float((-10 * mse.clamp_min(1e-12).log10()).mean()),
                       "native_latent_mse": float((generated - x0[:count]).square().mean()),
                       "code_contrastive": code_metrics,
                       "code_gradient_routing": "generator_only" if code_weight > 0 else "disabled",
                       "image_gradient_routing": "encoder_and_generator",
                       "interpretation": "DINO and code retrieval are training proxies; RGB metrics target original inputs."}
        return image_loss, code_weighted, metrics, (originals, images)

    def backward(self, branch, transformer, batch, x0, *, step, null_label=None):
        # Called ONCE per optimizer update, after accumulated real-data flow
        # gradients. Auxiliary weights are not divided by accumulation_steps
        # or multiplied by the sparse interval.
        generator_parameters = [p for n, p in branch.named_parameters()
                                if p.requires_grad and not n.startswith("encoder.")]
        with deterministic_branch(branch):
            image_loss, code_loss, metrics, _ = self.losses(
                branch, transformer, batch, x0, step=step, null_label=null_label)
            code_grads = (torch.autograd.grad(code_loss, generator_parameters,
                                             retain_graph=True, allow_unused=True)
                          if code_loss.requires_grad else [])
            image_loss.backward()
            for parameter, gradient in zip(generator_parameters, code_grads):
                if gradient is not None:
                    parameter.grad = (gradient.detach() if parameter.grad is None
                                      else parameter.grad + gradient.detach())
            metrics["code_generator_grad_norm"] = math.sqrt(sum(
                float(g.detach().float().square().sum()) for g in code_grads if g is not None))
            metrics["applied_code_encoder_grad_norm"] = 0.0
        return metrics

    @torch.no_grad()
    def evaluate(self, branch, transformer, loader, *, device, dtype, step, null_label=None,
                 grid_path: Path | None = None):
        from diffusion_ot.evaluation.stage1a_eval import _save_grid

        # DataLoader iterator creation can consume global RNG even in eval.
        # Preserve training RNG and use a private fixed rollout-noise stream.
        devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
        with torch.random.fork_rng(devices=devices), deterministic_branch(branch):
            batch = next(iter(loader))
            x0 = batch["x0_latent"].to(device=device, dtype=dtype)
            _, _, metrics, (originals, images) = self.losses(
                branch, transformer, batch, x0, step=step, null_label=null_label, validation=True)
            metrics["noise_seed"] = self.validation_seed
            metrics["bank_sample_ids"] = batch["sample_id"]
            if grid_path is not None:
                vae_images = decode_training_images(self.vae, x0[:len(images)], checkpoint_decode=False)
                _save_grid(grid_path, [originals, vae_images, images], len(images))
                metrics.update(grid_path=str(grid_path),
                               grid_rows=["original_rgb", "vae_reconstruction", "generated_native"])
            return metrics
