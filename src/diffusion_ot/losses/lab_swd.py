"""Training-free source color comparison, inspired by ECCV 2024 MS-SWD.

Reference: https://github.com/real-hjq/MS-SWD/blob/main/MS_SWD.py
Independent adaptation: explicit normalized D65 Lab units, low-resolution
Gaussian scales, detached source, FP32 arithmetic and caller-owned CPU RNG.
This is a source appearance constraint, not a target-realism metric.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_rgb(rgb):
    if (rgb.ndim != 4 or rgb.shape[1] != 3 or not all(rgb.shape)
            or not rgb.is_floating_point() or not torch.isfinite(rgb).all()):
        raise ValueError("Lab conversion requires finite nonempty [B,3,H,W] RGB.")


def normalized_lab(rgb):
    """sRGB [0,1] -> D65 Lab (L/100, a/128, b/128), with finite black gradients."""
    _validate_rgb(rgb)
    with torch.autocast(device_type=rgb.device.type, enabled=False):
        rgb = rgb.float().clamp(0, 1)
        linear = torch.where(rgb > .04045, ((rgb.clamp_min(.04045) + .055) / 1.055).pow(2.4), rgb / 12.92)
        matrix = rgb.new_tensor([[.4124564, .3575761, .1804375],
                                 [.2126729, .7151522, .0721750],
                                 [.0193339, .1191920, .9503041]])
        xyz = torch.einsum("ij,bjhw->bihw", matrix, linear)
        xyz = xyz / rgb.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
        delta = 6 / 29
        f = torch.where(xyz > delta ** 3, xyz.clamp_min(delta ** 3).pow(1 / 3),
                        xyz / (3 * delta ** 2) + 4 / 29)
        return torch.stack(((116 * f[:, 1] - 16) / 100,
                            500 * (f[:, 0] - f[:, 1]) / 128,
                            200 * (f[:, 1] - f[:, 2]) / 128), dim=1)


def _gaussian_scales(rgb, sizes):
    value = F.interpolate(rgb.float().clamp(0, 1), (sizes[0], sizes[0]),
                          mode="bilinear", align_corners=False, antialias=True)
    yield value
    axis = value.new_tensor([1., 4., 6., 4., 1.])
    kernel = ((axis[:, None] * axis[None, :]) / 256)[None, None].expand(3, 1, 5, 5)
    for size in sizes[1:]:
        value = F.conv2d(F.pad(value, (2, 2, 2, 2), mode="reflect"), kernel, groups=3)[:, :, ::2, ::2]
        if value.shape[-2:] != (size, size):
            raise ValueError("Lab SWD scales must halve successively.")
        yield value


def source_lab_swd(generated, source, *, generator, sizes=(64, 32, 16), patch_size=5, directions=128):
    """Per-image multiscale SW1 of normalized Lab patch projections."""
    if generator is None or generator.device.type != "cpu":
        raise ValueError("Lab SWD requires a private CPU generator.")
    if generated.shape != source.shape:
        raise ValueError("Lab SWD requires corresponding source/generated RGB shapes.")
    if (not sizes or any(s < 4 for s in sizes) or any(a != 2 * b for a, b in zip(sizes, sizes[1:]))
            or patch_size < 1 or patch_size % 2 != 1 or patch_size > min(sizes) or directions < 1):
        raise ValueError("Invalid Lab SWD scales, odd patch size, or direction count.")
    with torch.autocast(device_type=generated.device.type, enabled=False):
        # Validate before interpolation as NaNs/invalid channels must fail explicitly.
        _validate_rgb(generated)
        _validate_rgb(source)
        actual = _gaussian_scales(generated, sizes)
        target = _gaussian_scales(source.detach().to(generated.device), sizes)
        values = []
        for x, y in zip(actual, target):
            x, y = normalized_lab(x), normalized_lab(y)
            axes = F.normalize(torch.randn(directions, 3 * patch_size ** 2, generator=generator), dim=1)
            filters = axes.reshape(directions, 3, patch_size, patch_size).to(x.device)
            pad = patch_size // 2
            def project(image):
                return F.conv2d(F.pad(image, (pad, pad, pad, pad), mode="reflect"), filters).flatten(2).sort(2).values
            values.append((project(x) - project(y)).abs().mean((1, 2)))
        return torch.stack(values).mean(0)


@torch.no_grad()
def coarse_lab_descriptor(rgb, size=8):
    """Cheap image-space selection descriptor: Lab of area-averaged sRGB."""
    with torch.autocast(device_type=rgb.device.type, enabled=False):
        return normalized_lab(F.adaptive_avg_pool2d(rgb.detach().float(), (size, size))).flatten(1)
