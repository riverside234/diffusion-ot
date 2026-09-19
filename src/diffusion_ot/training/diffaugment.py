"""Translation-only DiffAugment for image-space adversarial supervision.

Algorithm reference: Zhao et al., NeurIPS 2020, https://arxiv.org/abs/2006.10738
and https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py.
Use private RNG streams and apply before the frozen feature extractor so that
generator gradients pass through the same augmentation used by the discriminator.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def diffaugment_options(options: dict | None) -> dict:
    """Normalize and validate the supported adversarial augmentation policy."""
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("decoded_translation.diffaugment must be a mapping.")
    enabled = options.get("enabled", False)
    policy = options.get("policy", "translation")
    ratio = options.get("translation_ratio", .0625)
    if not isinstance(enabled, bool):
        raise ValueError("decoded_translation.diffaugment.enabled must be boolean.")
    if policy != "translation":
        raise ValueError("decoded_translation.diffaugment.policy must be translation.")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= .5:
        raise ValueError("decoded_translation.diffaugment.translation_ratio must be finite and in [0, 0.5].")
    return {"enabled": enabled, "policy": policy, "translation_ratio": float(ratio)}


def random_translation(images: torch.Tensor, *, ratio: float, generator: torch.Generator) -> torch.Tensor:
    """Independently translate NCHW images by integer offsets with zero padding.

    Each axis is uniform on +/-floor(size * ratio + 0.5), inclusive.
    Out-of-bounds pixels are black, never wrapped. Indexing preserves image
    gradients. Call outside checkpoint closures: backward must reuse the sampled
    transform, without advancing this private stream again.
    """
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("DiffAugment expects floating-point NCHW images.")
    if not math.isfinite(ratio) or not 0 <= ratio <= .5:
        raise ValueError("DiffAugment translation ratio must be finite and in [0, 0.5].")
    batch, _, height, width = images.shape
    max_y, max_x = int(height * ratio + .5), int(width * ratio + .5)
    if max_y == 0 and max_x == 0:
        return images
    offset_y = torch.randint(-max_y, max_y + 1, (batch, 1, 1), device=images.device, generator=generator)
    offset_x = torch.randint(-max_x, max_x + 1, (batch, 1, 1), device=images.device, generator=generator)
    rows = (torch.arange(height, device=images.device)[None, :, None] + offset_y + 1).clamp(0, height + 1)
    cols = (torch.arange(width, device=images.device)[None, None, :] + offset_x + 1).clamp(0, width + 1)
    samples = torch.arange(batch, device=images.device)[:, None, None]
    padded = F.pad(images, (1, 1, 1, 1)).permute(0, 2, 3, 1)
    return padded[samples, rows, cols].permute(0, 3, 1, 2).contiguous()
