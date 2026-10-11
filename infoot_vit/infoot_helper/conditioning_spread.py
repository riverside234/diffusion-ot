"""Opt-in, mean-preserving conditioning intervention; never modifies an OT plan.

This restores one scalar moment, not semantic information lost by averaging.
Image quality must be assessed with a frozen decoder and matched native controls.
"""
from __future__ import annotations

import math

import torch


MAX_SPREAD_GAIN = 2.0


def token_moments(features, valid_mask):
    """Per-image valid-token mean and population variance, calculated in float64."""
    if (features.ndim != 3 or min(features.shape) < 1 or not features.is_floating_point()
            or not torch.isfinite(features).all() or valid_mask.dtype != torch.bool
            or valid_mask.shape != features.shape[:2] or valid_mask.device != features.device):
        raise ValueError("Need finite floating [N,P,D] features and a matching boolean valid mask.")
    counts = valid_mask.sum(1)
    if (counts < 2).any():
        raise ValueError("Spread intervention requires at least two valid tokens per image.")
    x = features.detach().double()
    mean = x.masked_fill(~valid_mask[..., None], 0).sum(1) / counts[:, None]
    residual = (x - mean[:, None]).masked_fill(~valid_mask[..., None], 0)
    variance = residual.square().sum((1, 2)) / (counts * x.shape[-1])
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise ValueError("Token moments overflowed.")
    return mean, variance


@torch.inference_mode()
def scale_token_deviations(features, valid_mask, gains):
    """Scale deviations around each image mean; leave padding values untouched."""
    mean, _ = token_moments(features, valid_mask)
    gains = torch.as_tensor(gains, device=features.device, dtype=torch.float64)
    if gains.ndim == 0:
        gains = gains.expand(len(features))
    if gains.shape != (len(features),) or not torch.isfinite(gains).all() or (gains <= 0).any():
        raise ValueError("Need one finite positive spread gain per image.")
    changed = mean[:, None] + gains[:, None, None] * (features.double() - mean[:, None])
    changed = torch.where(valid_mask[..., None], changed, features.double()).to(features.dtype)
    if not torch.isfinite(changed).all():
        raise ValueError("Conditioning intervention overflowed the output dtype.")
    return changed


@torch.inference_mode()
def calibrate_native_spread(features, *, chunk_size=32):
    """Equal-image mean within-image variance of the selected target TRAIN bank."""
    if not isinstance(chunk_size, int) or chunk_size < 1 or len(features) < 1:
        raise ValueError("Native calibration needs nonempty features and a positive chunk size.")
    variances = []
    for start in range(0, len(features), chunk_size):
        x = features[start:start + chunk_size]
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        variances.append(token_moments(x, valid)[1].cpu())
    values = torch.cat(variances)
    target = float(values.mean())
    if not math.isfinite(target) or target <= 1e-12:
        raise ValueError("Native training-token spread is degenerate.")
    return dict(target_within_image_variance=target, images=len(features),
                per_image_within_variance=values.tolist(),
                estimator="equal_image_mean_population_variance_over_tokens_and_channels")


@torch.inference_mode()
def restore_token_spread(features, valid_mask, target_variance):
    """Amplify existing directions toward native spread, with a fixed 2x cap.

The gain is per image, so results do not depend on query order or batch size.
Never shrink high-variance conditions or silently synthesize collapsed tokens.
"""
    if not math.isfinite(target_variance) or target_variance <= 1e-12:
        raise ValueError("Target variance must be finite and positive.")
    mean, before = token_moments(features, valid_mask)
    if (before <= 1e-12).any():
        raise ValueError("Collapsed mapped tokens cannot be repaired by spread scaling.")
    requested = (target_variance / before).sqrt().clamp_min(1)
    gains = requested.clamp_max(MAX_SPREAD_GAIN)
    changed = scale_token_deviations(features, valid_mask, gains)
    after_mean, after = token_moments(changed, valid_mask)
    return changed, dict(method="valid_token_centered_spread_v1", max_gain=MAX_SPREAD_GAIN,
        target_within_image_variance=float(target_variance), gains=gains.tolist(),
        requested_gains=requested.tolist(), capped=(requested > MAX_SPREAD_GAIN).tolist(),
        variance_before=before.tolist(), variance_after=after.tolist(),
        mean_max_abs_change=float((after_mean - mean).abs().max()),
        interpretation="Inference-only hypothesis; matching variance does not establish image quality.")


@torch.inference_mode()
def conditioning_variants(mapped, valid_mask, native_references, target_variance):
    """Five matched conditions, including a native contraction counterfactual."""
    if native_references.shape != mapped.shape:
        raise ValueError("Native reference and mapped grids must have identical shapes.")
    corrected, correction = restore_token_spread(mapped, valid_mask, target_variance)
    mapped_variance = token_moments(mapped, valid_mask)[1]
    native_variance = token_moments(native_references, valid_mask)[1]
    if (native_variance <= 1e-12).any():
        raise ValueError("Native references have degenerate spread under the mapped mask.")
    contraction = (mapped_variance / native_variance).sqrt().clamp_max(1)
    contracted = scale_token_deviations(native_references, valid_mask, contraction)
    full_mask = torch.ones_like(valid_mask)
    variants = dict(
        native_reference_full=(native_references, full_mask),
        native_reference_mapped_mask=(native_references, valid_mask),
        native_reference_contracted=(contracted, valid_mask),
        mapped_baseline=(mapped, valid_mask),
        mapped_spread_restored=(corrected, valid_mask),
    )
    return variants, dict(correction=correction, native_contraction_gains=contraction.tolist(),
        native_contraction_reached_mapped_variance=(mapped_variance <= native_variance).tolist(),
        native_mask_interpretation="Same token indices/count only; cat and dog patches are not paired ground truth.")
