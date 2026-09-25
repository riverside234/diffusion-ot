"""Own-encoder RGB translation contrast, without DINO or a discriminator.

For cat -> dog, the default live readout is the dog PDAE encoder. It retrieves
the detached original cat code among current cat queries and references. This
deliberately learns cross-domain coordinate agreement; it is not recovery of
the projected dog condition and does not independently measure dog realism.
The opt-in PatchNCE experiment instead matches corresponding spatial features
from the same encoders using CUT's within-image negative construction.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from diffusion_ot.losses.contrastive import detached_key_contrastive_loss
from diffusion_ot.models.generator_adaptation import (
    baseline_parameter_snapshot,
    configure_generator_adaptation,
    load_joint_generator,
)
from diffusion_ot.training.decoded_translation import (
    DecoderTraining,
    decode_training_images,
    integrate_training_flow,
    self_supervised_translation_options,
)
from diffusion_ot.training.self_supervised_diagnostics import image_correspondence, native_rgb_reconstruction


def encode_generated_images(vae, images, *, checkpoint_encode=True):
    """Encode actual [0,1] RGB using the cache's scaled VAE posterior mean.

    VAE parameters are frozen by the runtime, but the RGB input remains live.
    A deterministic mean avoids extra posterior noise in the retrieval target.
    The scaling-only convention exactly matches data/latent_cache.py.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Generated VAE encoding requires [batch, 3, height, width] RGB.")
    parameter = next(vae.parameters())
    scale = float(getattr(vae.config, "scaling_factor", 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("VAE scaling_factor must be finite and positive.")

    def encode(value):
        posterior = vae.encode((value * 2 - 1).to(parameter)).latent_dist
        return posterior.mean * scale

    return (checkpoint(encode, images, use_reentrant=False)
            if checkpoint_encode and torch.is_grad_enabled() else encode(images))


def source_code_contrastive_loss(recovered, positive, source_bank, *, temperature=.2,
                                 negative_similarity_threshold=.95):
    """Contrast a live recovered code against detached original-source keys."""
    loss, metrics = detached_key_contrastive_loss(
        recovered, positive, source_bank, temperature=temperature,
        negative_similarity_threshold=negative_similarity_threshold,
    )
    with torch.no_grad():
        target = positive.detach().to(recovered).float()
        values = recovered.detach().float()
        norms = target.norm(dim=-1)
        valid = norms > 1e-6
        metrics.update(
            matched_cosine=float(F.cosine_similarity(values[valid], target[valid], dim=-1).mean())
            if valid.any() else None,
            recovered_to_source_norm_ratio=float((values[valid].norm(dim=-1) / norms[valid]).mean())
            if valid.any() else None,
            source_norm=float(norms.mean()),
            recovered_norm=float(values.norm(dim=-1).mean()),
            source_batch_variance=float(target.var(dim=0, unbiased=False).mean()),
            recovered_batch_variance=float(values.var(dim=0, unbiased=False).mean()),
            negative_source="current_source_queries_and_disjoint_source_references",
            positive_target="detached_original_source_encoder_code",
        )
    return loss, metrics


class SelfSupervisedDecoderTraining(DecoderTraining):
    """Opt-in self-supervised translation branch using existing generator views.

    Main gradients reach the live readout encoder, generator, and InfoOT
    conditioning path. Positive/negative keys and VAE parameters are detached
    or frozen. No frozen semantic image teacher or critic is instantiated.
    """

    def __init__(self, config, domains, prior=None, root=None, *, seed):
        from diffusion_ot.training.train_joint_infoot import EncoderEMA

        self.config, self.options, self.domains = config, config["decoded_translation"], domains
        if self.options.get("supervision") != "self_supervised":
            raise ValueError("Self-supervised decoder requires supervision: self_supervised.")
        self.contrastive_options = self_supervised_translation_options(self.options)
        self.objective = self.contrastive_options["objective"]
        self.objective_name = self.contrastive_options["name"]
        self.readout = self.contrastive_options["readout"]
        for key in ("structure_weight", "perceptual_weight", "structure_contrastive_weight",
                    "adversarial_weight", "code_consistency_weight"):
            if float(self.options.get(key, 0)) != 0:
                raise ValueError(f"Self-supervised decoder requires {key}: 0.")
        if float((self.options.get("color_histogram") or {}).get("weight", 0)) != 0:
            raise ValueError("Self-supervised decoder requires color_histogram.weight: 0.")
        for key in ("null_preservation_weight", "conditioned_preservation_weight"):
            if float(config["generator_adaptation"].get(key, 0)) != 0:
                raise ValueError(f"Self-supervised decoder requires {key}: 0.")
        self.weight = self.contrastive_options["weight"]
        self.patch_options = None
        if self.objective == "patchnce":
            self.patch_options = {key: self.contrastive_options[key]
                                  for key in ("weight", "temperature", "num_patches", "layers")}
            widths = []
            for context in domains.values():
                encoder = context.branch.encoder
                if not callable(getattr(encoder, "forward_spatial_features", None)):
                    raise ValueError("PatchNCE requires a PDAE encoder exposing forward_spatial_features.")
                channels = encoder.spatial_feature_channels
                if any(index >= len(channels) for index in self.patch_options["layers"]):
                    raise ValueError("PatchNCE layers exceed the PDAE encoder convolution stages.")
                widths.append(tuple(channels[index] for index in self.patch_options["layers"]))
            if any(width != widths[0] for width in widths):
                raise ValueError("Cross-domain PatchNCE requires matching encoder feature channel widths.")

        self.views = {d: configure_generator_adaptation(v.branch) for d, v in domains.items()}
        self.parameters = [p for view in self.views.values() for p in view.parameters()]
        # Retain the common checkpoint/evaluation contract, without executing a
        # baseline-teacher objective or constructing another generator model.
        self.baselines = {d: baseline_parameter_snapshot(v.branch, self.views[d]) for d, v in domains.items()}
        self.fixed_generators = {d: self._cpu_copy(v.branch.generator_state_dict()) for d, v in domains.items()}
        for value in domains.values():
            if value.vae is None or not callable(getattr(value.vae, "encode", None)):
                raise ValueError("Self-supervised translation requires the frozen VAE encoder and decoder.")
            value.vae.eval().requires_grad_(False)
            if (value.training_config.get("flow") or {}).get("direction", "noise_to_data") != "noise_to_data":
                raise ValueError("Self-supervised translation requires noise_to_data flow.")
            if abs(value.branch.semantic_conditioner.dropout_probability - .1) > 1e-8:
                raise ValueError("Self-supervised translation preserves semantic dropout_probability=0.1.")
        ema = config.get("ema") or {}
        self.ema = EncoderEMA(self.views, decay=float(ema.get("decay", .995)),
                              warmup_steps=int(ema.get("warmup_steps", 500))) if ema.get("enabled", True) else None
        self.noise_generators = {d: torch.Generator(device=v.device).manual_seed(seed + 901 + i)
                                 for i, (d, v) in enumerate(domains.items())}
        # Separate from diffusion noise and global RNG. Changing patch sampling
        # must not change the initial noise used by the paired control rollout.
        self.patch_generators = ({d: torch.Generator(device="cpu").manual_seed(seed + 1901 + i)
                                  for i, d in enumerate(domains)} if self.patch_options else {})
        self.code_consistency_objective = torch.zeros((), device=domains["cat"].device)
        self.image_objectives = {}
        self.original_datasets = {}
        self.diagnostic_options = config.get("self_supervised_diagnostics") or {}

    @torch.no_grad()
    def _validation_images(self, source, images, originals, query_keys, source_latents, *, seed, steps):
        """Original-RGB checks plus a native rollout using a private fixed RNG."""
        count = min(len(images), int(self.diagnostic_options.get("image_samples", 32)))
        images, originals = images[:count], originals[:count]
        context = self.domains[source]
        generator = torch.Generator(device=context.device).manual_seed(seed)
        noise = torch.randn((count, *source_latents.shape[1:]), generator=generator,
                            device=context.device, dtype=context.dtype)
        native_latents = integrate_training_flow(
            context.branch, context.transformer, noise,
            query_keys[:count].to(context.device, dtype=context.dtype), num_steps=steps,
            guidance_scale=float(self.options.get("guidance_scale", 1.)),
            null_label=(context.training_config.get("class_conditioning") or {}).get("null_label"),
            checkpoint_steps=False,
        )
        native = decode_training_images(context.vae, native_latents, checkpoint_decode=False)
        correspondence = image_correspondence(images, originals,
            pooled_size=int(self.diagnostic_options.get("pooled_size", 16)))
        reconstruction = native_rgb_reconstruction(native, originals)
        reconstruction.update(seed=seed, num_steps=steps, guidance_scale=float(self.options.get("guidance_scale", 1.)))
        return correspondence, reconstruction, native

    def null_preservation_loss(self, latents, codes):
        return torch.zeros((), device=self.domains["cat"].device)

    def conditioned_preservation_loss(self, latents, baseline_codes):
        return torch.zeros((), device=self.domains["cat"].device)

    def _patchnce_loss(self, source, readout, recovered_latents, original_latents, *, validation_seed):
        from diffusion_ot.losses.patchnce import patchnce_loss

        layers = self.patch_options["layers"]
        def query_features(value):
            return tuple(readout.branch.encoder.forward_spatial_features(value, layers))

        values = recovered_latents.to(readout.device, dtype=readout.dtype)
        features = (checkpoint(query_features, values, use_reentrant=False)
                    if bool(self.options.get("checkpoint_features", True)) and torch.is_grad_enabled()
                    else query_features(values))
        source_context = self.domains[source]
        with torch.no_grad():
            keys = source_context.branch.encoder.forward_spatial_features(
                original_latents.to(source_context.device, dtype=source_context.dtype), layers)
        generator = (torch.Generator(device="cpu").manual_seed(validation_seed)
                     if validation_seed is not None else self.patch_generators[source])
        loss, metrics = patchnce_loss(features, keys, temperature=self.patch_options["temperature"],
                                     num_patches=self.patch_options["num_patches"], generator=generator)
        metrics.update(encoder_layers=layers, source_encoder_domain=source,
                       key_input="original_cached_vae_latent",
                       query_input="generated_rgb_vae_posterior_mean")
        return loss, metrics

    def loss(self, weights, references, source_latents, real_latents, *, step, validation_seed=None,
             source_reference_structure=None, source_query_structure=None, source_query_metadata=None,
             validation_image_dir=None, source_query_codes=None):
        evaluation = validation_seed is not None
        device = self.domains["cat"].device
        losses, metrics = [], {}
        self.code_consistency_objective = torch.zeros((), device=device)
        self.image_objectives = {}
        save_images = (evaluation and validation_image_dir is not None
                       and bool(self.options.get("save_validation_images", False)))
        image_diagnostics = evaluation and bool(self.diagnostic_options.get("enabled", False))
        if (save_images or image_diagnostics) and (source_query_metadata is None or any(d not in source_query_metadata for d in self.domains)):
            raise ValueError("Self-supervised validation images/diagnostics require original source metadata.")
        if source_query_codes is None:
            with torch.no_grad():
                source_query_codes = {
                    d: value.branch.encode(source_latents[d].to(value.device, dtype=value.dtype))
                    for d, value in self.domains.items()
                }
        for offset, (source, target) in enumerate((("cat", "dog"), ("dog", "cat"))):
            direction = f"{source}_to_{target}"
            context = self.domains[target]
            probability = weights[direction]
            count = min(int(self.options.get("batch_size", 4)), len(probability))
            query_keys = source_query_codes[source].detach()
            if len(query_keys) != len(probability) or len(query_keys) > len(source_latents[source]):
                raise ValueError("Source query codes must match all conditional query rows.")
            # Keep the complete weighted target mean and its live gradient.
            codes = (probability[:count] @ references[target].float()).to(context.device, dtype=context.dtype)
            generator = (torch.Generator(device=context.device).manual_seed(validation_seed + offset)
                         if evaluation else self.noise_generators[target])
            noise = torch.randn((count, *source_latents[source].shape[1:]), generator=generator,
                                device=context.device, dtype=context.dtype)
            steps = int(self.options.get("validation_num_steps", 50) if evaluation else self.options.get("num_steps", 50))
            generated = integrate_training_flow(
                context.branch, context.transformer, noise, codes, num_steps=steps,
                guidance_scale=float(self.options.get("guidance_scale", 1.0)),
                null_label=(context.training_config.get("class_conditioning") or {}).get("null_label"),
                checkpoint_steps=bool(self.options.get("checkpoint_sampler", True)),
            )
            images = decode_training_images(context.vae, generated,
                                            checkpoint_decode=bool(self.options.get("checkpoint_decode", True)))
            readout_domain = target if self.readout == "target" else source
            readout = self.domains[readout_domain]
            recovered_latents = encode_generated_images(
                readout.vae, images.to(readout.device),
                checkpoint_encode=bool(self.options.get("checkpoint_encode", True)),
            )
            if self.objective == "patchnce":
                loss, retrieval = self._patchnce_loss(
                    source, readout, recovered_latents, source_latents[source][:count],
                    validation_seed=validation_seed + 20000 + offset if evaluation else None)
            else:
                recovered = readout.branch.encode(recovered_latents.to(readout.device, dtype=readout.dtype)).float()
                source_bank = torch.cat((query_keys, references[source].detach().to(query_keys)), dim=0)
                loss, retrieval = source_code_contrastive_loss(
                    recovered, query_keys[:count], source_bank,
                    temperature=self.contrastive_options["temperature"],
                    negative_similarity_threshold=self.contrastive_options["negative_similarity_threshold"],
                )
            losses.append(loss.to(device))
            metrics[direction] = {
                self.objective_name: retrieval, f"{self.objective_name}_loss": float(loss.detach()),
                "samples": count, "reference_targets": len(references[target]),
                "readout_domain": readout_domain,
                "readout": "generated_rgb_to_scaled_vae_posterior_mean_to_live_pdae_encoder",
            }
            if save_images or image_diagnostics:
                records = source_query_metadata[source][:count]
                if len(records) != count:
                    raise ValueError("Original source metadata must match the decoded query count.")
                with torch.no_grad():
                    originals = self.original_source_images(source, records).to(device=images.device, dtype=torch.float32)
                if originals.shape != images.shape:
                    raise ValueError("Decoded and original source RGB shapes differ; check cache preprocessing.")
            if image_diagnostics:
                correspondence, native_metrics, native = self._validation_images(
                    source, images, originals, query_keys, source_latents[source],
                    seed=validation_seed + 10000 + offset, steps=steps)
                metrics[direction]["source_correspondence"] = correspondence
                metrics.setdefault("native_reconstruction", {})[source] = native_metrics
            if save_images:
                from torchvision.utils import save_image

                path = Path(validation_image_dir) / f"{direction}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                pairs = torch.stack((originals.detach().cpu(), images.detach().cpu()), dim=1).flatten(0, 1)
                save_image(pairs, str(path), nrow=8, padding=2, pad_value=1.)
                metrics[direction].update(validation_grid=str(path),
                    validation_grid_layout="original_source,generated_target pairs; four pairs per row")
                if image_diagnostics:
                    native_path = Path(validation_image_dir) / f"native_{source}.png"
                    native_pairs = torch.stack((originals[:len(native)].cpu(), native.float().cpu()), dim=1).flatten(0, 1)
                    save_image(native_pairs, str(native_path), nrow=8, padding=2, pad_value=1.)
                    native_metrics.update(validation_grid=str(native_path),
                        validation_grid_layout="original_source,native_reconstruction pairs; four pairs per row")
        contrastive = torch.stack(losses).mean()
        ramp = 1.0 if evaluation else self.ramp(step)
        weighted = self.weight * contrastive
        objective = ramp * weighted
        self.image_objectives[self.objective_name] = objective
        metrics.update({
            f"{self.objective_name}_loss": float(contrastive.detach()),
            f"{self.objective_name}_weight": self.weight,
            f"{self.objective_name}_effective_weight": self.weight * ramp,
            f"{self.objective_name}_readout": self.readout,
            f"{self.objective_name}_gradient_routing":
                "live_readout_encoder_and_translation_conditioning_and_generator; detached_source_keys",
        })
        metrics.update(
            supervision="self_supervised", objective=self.objective,
            weighted_loss=float(weighted.detach()), effective_weighted_loss=float(objective.detach()),
            ramp=ramp, guidance_scale=float(self.options.get("guidance_scale", 1.0)), num_steps=steps,
            interpretation=("Own-encoder spatial source correspondence; training objective, not independent target realism validation."
                            if self.patch_options else
                            "Own-encoder source retrieval; training objective, not independent target realism validation."),
        )
        if image_diagnostics:
            metrics["image_diagnostics_protocol"] = "original_rgb_pooled_correspondence_and_native_reconstruction_v1"
        return objective, metrics

    def checkpoint_state(self):
        return {
            "decoded_supervision": "self_supervised",
            "decoded_objective": self.objective,
            **({"decoded_patchnce_options": self.patch_options,
                "decoded_patch_sampling_states": {d: gen.get_state().cpu() for d, gen in self.patch_generators.items()}}
               if self.patch_options else {}),
            "decoded_source_contrastive_readout": self.readout,
            "generators": {d: self._cpu_copy(v.branch.generator_state_dict()) for d, v in self.domains.items()},
            "fixed_generators": self.fixed_generators,
            "generator_baseline_parameters": self.baselines,
            "generator_ema": self.ema.export() if self.ema else None,
            "generator_ema_state": self.ema.state_dict() if self.ema else None,
            "decoded_noise_states": {d: gen.get_state().cpu() for d, gen in self.noise_generators.items()},
        }

    def load_checkpoint(self, payload):
        if (payload.get("decoded_supervision") != "self_supervised"
                or payload.get("decoded_objective", "source_infonce") != self.objective
                or payload.get("decoded_source_contrastive_readout") != self.readout):
            raise ValueError("Cannot resume a different decoded supervision/readout; start a fresh Stage 1B run.")
        if self.patch_options and (payload.get("decoded_patchnce_options") != self.patch_options
                                  or set(payload.get("decoded_patch_sampling_states", {})) != set(self.patch_generators)):
            raise ValueError("Cannot resume a different PatchNCE protocol or missing patch sampling state.")
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
        for d, gen in self.noise_generators.items():
            gen.set_state(payload["decoded_noise_states"][d].cpu())
        for d, gen in self.patch_generators.items():
            gen.set_state(payload["decoded_patch_sampling_states"][d].cpu())
