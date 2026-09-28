"""Teacher-free RGB objectives for unpaired translation.

Source RGB supplies appearance/layout targets; unpaired real target RGB supplies
texture statistics. Fixed local self-similarity and Laplacian-patch SW1 are
adaptations, not the VGG/GAN training recipes in F-LSeSim or texture synthesis.
See docs/analysis/stage1b_v3_2500/loss_design.md for references and reductions.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from diffusion_ot.losses.color_histogram import color_histogram_options, histogan_color_distance


IMAGE_LOSS_PROTOCOL = "source_rgb_layout_target_laplacian_sw1_v1"
IMAGE_LOSS_NAMES = ("coarse_rgb", "color_histogram", "local_layout", "target_patch_swd")


def translation_image_options(image):
    """Resolve only enabled terms, keeping legacy zero-weight recipes unchanged."""
    defaults = {
        "coarse_rgb": {"weight": 0., "sizes": [8, 16]},
        "local_layout": {"weight": 0., "sizes": [32, 64], "contrast_eps": .01},
        "target_patch_swd": {"weight": 0., "sizes": [256, 128, 64],
                             "patch_size": 7, "patches_per_image": 32,
                             "directions": 64, "scale_floor": .01},
    }
    resolved = {}
    for name in IMAGE_LOSS_NAMES:
        supplied = image.get(name) or {}
        if name == "color_histogram":
            options = color_histogram_options(supplied)
        else:
            if not isinstance(supplied, dict) or set(supplied) - set(defaults[name]):
                raise ValueError(f"Unknown {name} options.")
            options = {**defaults[name], **supplied}
            for key in ("weight", "contrast_eps", "scale_floor"):
                if key not in options:
                    continue
                value = options[key]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0 or (key != "weight" and value == 0)):
                    raise ValueError(f"{name}.{key} must be finite and {'nonnegative' if key == 'weight' else 'positive'}.")
                options[key] = float(value)
            sizes = options["sizes"]
            if (not isinstance(sizes, (list, tuple)) or not sizes
                    or any(isinstance(s, bool) or not isinstance(s, int) or s < 4 for s in sizes)
                    or len(set(sizes)) != len(sizes)):
                raise ValueError(f"{name}.sizes must contain distinct integer resolutions >= 4.")
            options["sizes"] = list(sizes)
            if name == "target_patch_swd":
                for key in ("patch_size", "patches_per_image", "directions"):
                    value = options[key]
                    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                        raise ValueError(f"{name}.{key} must be a positive integer.")
                if options["patch_size"] > min(sizes):
                    raise ValueError("target_patch_swd.patch_size exceeds a band resolution.")
        if options["weight"] > 0:
            resolved[name] = options
    return resolved


def _rgb(images):
    if (images.ndim != 4 or images.shape[1] != 3 or images.shape[0] == 0
            or min(images.shape[-2:]) < 1 or not images.is_floating_point()
            or not torch.isfinite(images).all()):
        raise ValueError("Image objectives require finite nonempty [B,3,H,W] floating RGB.")
    # The runtime decodes into [0,1]; clamp also keeps standalone callers safe.
    return images.float().clamp(0, 1)


def _luminance(images):
    return (images * images.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)


def coarse_rgb_distance(generated, source, *, sizes=(8, 16)):
    """Per-image mean L1 over area-averaged RGB scales; source is detached."""
    with torch.autocast(device_type=generated.device.type, enabled=False):
        actual, target = _rgb(generated), _rgb(source.detach().to(generated.device))
        if actual.shape != target.shape:
            raise ValueError("Coarse RGB requires corresponding source/generated shapes.")
        return torch.stack([(F.interpolate(actual, size=(s, s), mode="area")
                             - F.interpolate(target, size=(s, s), mode="area"))
                            .abs().flatten(1).mean(1) for s in sizes]).mean(0)


def local_layout_distance(generated, source, *, sizes=(32, 64), contrast_eps=.01):
    """Per-image local cosine-relation error with detached source contrast weights.

    Valid 3x3 descriptors, eight neighbor offsets, no wraparound or padded
    relations. Low-contrast source patches contribute proportionally less.
    """
    with torch.autocast(device_type=generated.device.type, enabled=False):
        actual, target = _rgb(generated), _rgb(source.detach().to(generated.device))
        if actual.shape != target.shape:
            raise ValueError("Local layout requires corresponding source/generated shapes.")
        actual, target = _luminance(actual), _luminance(target)
        losses, fractions = [], []
        for size in sizes:
            def descriptor(value):
                small = F.interpolate(value, (size, size), mode="bilinear", align_corners=False, antialias=True)
                patches = F.unfold(small, 3).reshape(len(value), 9, size - 2, size - 2)
                centered = patches - patches.mean(1, keepdim=True)
                norm = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
                return centered / norm.clamp_min(3 * contrast_eps), norm[:, 0] / 3

            query, _ = descriptor(actual)
            key, contrast = descriptor(target)
            weights = contrast / (contrast + contrast_eps)
            numerator, denominator = actual.new_zeros(len(actual)), actual.new_zeros(len(actual))
            contributing, total_pairs = actual.new_zeros(len(actual)), 0
            height, width = query.shape[-2:]
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                           (0, 1), (1, -1), (1, 0), (1, 1)):
                ya, yb = slice(max(0, -dy), min(height, height - dy)), slice(max(0, dy), min(height, height + dy))
                xa, xb = slice(max(0, -dx), min(width, width - dx)), slice(max(0, dx), min(width, width + dx))
                similarity = (query[:, :, ya, xa] * query[:, :, yb, xb]).sum(1)
                reference = (key[:, :, ya, xa] * key[:, :, yb, xb]).sum(1)
                confidence = (weights[:, ya, xa] * weights[:, yb, xb]).detach()
                numerator += ((similarity - reference).abs() * .5 * confidence).flatten(1).sum(1)
                denominator += confidence.flatten(1).sum(1)
                contributing += (confidence > .25).flatten(1).sum(1)
                total_pairs += confidence[0].numel()
            losses.append(numerator / denominator.clamp_min(1e-6))
            fractions.append(contributing / total_pairs)
        return torch.stack(losses).mean(0), torch.stack(fractions).mean(0)


def sliced_wasserstein_l1(actual, target, directions):
    """Equal-count empirical SW1, averaging over samples and unit directions."""
    if actual.ndim != 2 or actual.shape != target.shape or actual.shape[0] == 0:
        raise ValueError("SW1 requires nonempty equal-size [samples, dimensions] sets.")
    if directions.ndim != 2 or directions.shape[0] != actual.shape[1] or directions.shape[1] == 0:
        raise ValueError("SW1 direction dimensions do not match descriptors.")
    axes = F.normalize(directions.detach().to(actual), dim=0, eps=1e-8)
    return ((actual @ axes).sort(dim=0).values
            - (target.detach().to(actual) @ axes).sort(dim=0).values).abs().mean()


def target_patch_swd(generated, real_target, *, generator, sizes=(256, 128, 64),
                     patch_size=7, patches_per_image=32, directions=64, scale_floor=.01,
                     real_baseline=None):
    """Fixed luminance Laplacian patches, normalized by real-only RMS.

    One CPU RNG, independent of diffusion noise. Shared sampled locations across
    images/sets make identical inputs zero and batch duplication invariant. Sort
    compares bags of patches, not source-to-target patch correspondences.
    """
    with torch.autocast(device_type=generated.device.type, enabled=False):
        actual = _luminance(_rgb(generated))
        real = _luminance(_rgb(real_target.detach().to(generated.device)))
        baseline = (_luminance(_rgb(real_baseline.detach().to(generated.device)))
                    if real_baseline is not None else None)
        if len(actual) != len(real) or (baseline is not None and len(baseline) != len(real)):
            raise ValueError("Patch SWD requires equal generated/real/baseline image counts.")
        losses, records = [], {}
        for size in sizes:
            def band(value):
                fine = F.interpolate(value, (size, size), mode="bilinear", align_corners=False, antialias=True)
                coarse = F.interpolate(fine, (size // 2, size // 2), mode="bilinear", align_corners=False, antialias=True)
                return fine - F.interpolate(coarse, (size, size), mode="bilinear", align_corners=False)

            positions = torch.randint(size - patch_size + 1, (2, patches_per_image), generator=generator)
            y = (positions[0, :, None, None] + torch.arange(patch_size)[None, :, None]).to(actual.device)
            x = (positions[1, :, None, None] + torch.arange(patch_size)[None, None, :]).to(actual.device)
            def patches(value):
                return band(value)[:, 0, y, x].reshape(-1, patch_size ** 2)

            query, reference = patches(actual), patches(real).detach()
            # A common scalar transform preserves relative generated amplitudes.
            center = reference.mean()
            real_scale = (reference - center).square().mean().sqrt()
            scale = real_scale.clamp_min(scale_floor)
            axes = torch.randn(patch_size ** 2, directions, generator=generator).to(actual.device)
            loss = sliced_wasserstein_l1((query - center) / scale, (reference - center) / scale, axes)
            losses.append(loss)
            record = {"loss": float(loss.detach()), "real_rms": float(real_scale),
                      "normalization_scale": float(scale), "patches_per_set": len(query)}
            if baseline is not None:
                record["real_to_real_distance"] = float(sliced_wasserstein_l1(
                    (patches(baseline) - center) / scale, (reference - center) / scale, axes))
            records[str(size)] = record
        return torch.stack(losses).mean(), {"bands": records,
            "real_to_real_distance": (sum(v["real_to_real_distance"] for v in records.values()) / len(records)
                                      if baseline is not None else None)}


def translation_image_losses(generated, source, options, *, generator=None,
                             real_target=None, real_baseline=None, per_image=False):
    """Unweighted scalar losses and diagnostics, evaluating only enabled terms."""
    losses, metrics = {}, {}
    for name, settings in options.items():
        parameters = {k: v for k, v in settings.items() if k != "weight"}
        extra = {}
        if name == "coarse_rgb":
            distances = coarse_rgb_distance(generated, source, **parameters)
        elif name == "color_histogram":
            distances = histogan_color_distance(generated, source, **parameters)
        elif name == "local_layout":
            distances, fraction = local_layout_distance(generated, source, **parameters)
            extra["contributing_pair_fraction"] = float(fraction.mean())
        elif name == "target_patch_swd":
            if real_target is None or generator is None:
                raise ValueError("Target patch SWD requires original target RGB and a private generator.")
            distances, extra = target_patch_swd(generated, real_target, generator=generator,
                real_baseline=real_baseline, **parameters)
        else:
            raise ValueError(f"Unknown image loss: {name}.")
        losses[name] = distances.mean()
        metrics[name] = {"loss": float(losses[name].detach()), **extra}
        if per_image and distances.ndim == 1:
            metrics[name]["per_image_loss"] = distances.detach().cpu().tolist()
    return losses, metrics
