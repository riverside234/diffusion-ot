"""Matched RGB augmentation and online frozen-VAE targets for PDAE v2."""
from __future__ import annotations

import math

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from diffusion_ot.data.latent_cache import select_posterior_latent


def augmentation_options(config: dict) -> dict | None:
    options = config.get("augmentation") or {}
    if not isinstance(options, dict):
        raise ValueError("augmentation must be a mapping.")
    if not options.get("enabled", False):
        return None
    encoder = config.get("encoder") or {}
    if encoder.get("kind") != "siglip2_vit_b16" or encoder.get("input_space") != "rgb":
        raise ValueError("Matched augmentation currently requires the PDAE v2 RGB SigLIP encoder.")
    color = {"probability": .25, "brightness": .1, "contrast": .1, "saturation": .05,
             "hue": 0., **(options.get("color_jitter") or {})}
    affine = {"probability": .2, "translate": [.03, .03], "scale": [.97, 1.03],
              **(options.get("affine") or {})}
    flip = float(options.get("horizontal_flip_probability", .5))
    for value in (flip, color["probability"], affine["probability"]):
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError("Augmentation probabilities must be finite and in [0,1].")
    for key in ("brightness", "contrast", "saturation", "hue"):
        value = float(color[key])
        if not math.isfinite(value) or not 0 <= value <= (0.5 if key == "hue" else 1):
            raise ValueError(f"Invalid augmentation color_jitter.{key}.")
        color[key] = value
    translate, scale = affine["translate"], affine["scale"]
    if (len(translate) != 2 or any(not math.isfinite(float(v)) or not 0 <= v <= .5 for v in translate)
            or len(scale) != 2 or not all(math.isfinite(float(v)) for v in scale)
            or not 0 < scale[0] <= scale[1]):
        raise ValueError("Invalid augmentation affine translate/scale ranges.")
    if set(affine) - {"probability", "translate", "scale"}:
        raise ValueError("Augmentation affine only supports translation and scale (no rotation/shear).")
    batch_size = int(options.get("vae_batch_size", 16))
    if batch_size <= 0:
        raise ValueError("augmentation.vae_batch_size must be positive.")
    return {"enabled": True, "horizontal_flip_probability": flip, "color_jitter": color,
            "affine": affine, "vae_batch_size": batch_size, "target_source": "augmented_rgb_frozen_vae"}


class MatchedRGBAugmentation:
    def __init__(self, options: dict, *, use_posterior_mean: bool):
        self.options = options
        self.use_posterior_mean = use_posterior_mean
        self.jitter = T.ColorJitter(**{k: v for k, v in options["color_jitter"].items() if k != "probability"})

    def transform(self, images: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Sample per image on CPU, using the trainer's checkpointed torch RNG."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Augmentation requires RGB images [B,3,H,W].")
        images = images.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(images).all() or images.min() < -1.00001 or images.max() > 1.00001:
            raise ValueError("Augmentation expects finite RGB in [-1,1].")
        stats = {"samples": len(images), "horizontal_flip": 0, "color_jitter": 0, "affine": 0}
        output = []
        for image in (images + 1).mul(.5).clamp(0, 1):
            if torch.rand(()) < self.options["horizontal_flip_probability"]:
                image = TF.hflip(image)
                stats["horizontal_flip"] += 1
            if torch.rand(()) < self.options["color_jitter"]["probability"]:
                image = self.jitter(image)
                stats["color_jitter"] += 1
            affine = self.options["affine"]
            if torch.rand(()) < affine["probability"]:
                h, w = image.shape[-2:]
                angle, translation, scale, shear = T.RandomAffine.get_params(
                    [0., 0.], affine["translate"], affine["scale"], None, [w, h])
                # Translation is sampled relative to ORIGINAL dimensions. Reflect enough
                # pixels for every output point; black affine fill stays outside the crop.
                pad = math.ceil((max(abs(v) for v in translation)
                                 + max(h, w) * max(0., 1 - scale) / 2) / scale) + 2
                if pad >= min(h, w):
                    raise ValueError("Affine range needs more reflection padding than this image supports.")
                image = TF.pad(image, [pad] * 4, padding_mode="reflect")
                image = TF.affine(image, angle, translation, scale, shear,
                                  interpolation=T.InterpolationMode.BILINEAR)
                image = TF.crop(image, pad, pad, h, w)
                stats["affine"] += 1
            output.append(image.clamp(0, 1).mul(2).sub(1))
        return torch.stack(output), stats

    @torch.no_grad()
    def prepare_batch(self, batch: dict, vae, *, device, dtype) -> tuple[dict, dict]:
        if any(batch.get("horizontal_flip", [])):
            raise ValueError("Disable cached-latent flips when matched RGB augmentation is enabled.")
        expected = "mean" if self.use_posterior_mean else "sample"
        for record in batch.get("metadata", []):
            if record.get("posterior_statistic", expected) != expected:
                raise ValueError("Augmentation posterior policy differs from the latent-cache metadata.")
        images, stats = self.transform(batch["encoder_image"])
        vae.requires_grad_(False).eval()
        parameter = next(vae.parameters(), None)
        vae_dtype = parameter.dtype if parameter is not None else dtype
        latents = []
        for chunk in images.split(self.options["vae_batch_size"]):
            posterior = vae.encode(chunk.to(device=device, dtype=vae_dtype)).latent_dist
            latents.append(select_posterior_latent(posterior, use_posterior_mean=self.use_posterior_mean))
        x0 = torch.cat(latents).mul(float(vae.config.scaling_factor)).to(dtype=dtype)
        if x0.shape != batch["x0_latent"].shape:
            raise ValueError("Online VAE target shape differs from the cached latent shape.")
        return {**batch, "encoder_image": images, "x0_latent": x0}, stats
