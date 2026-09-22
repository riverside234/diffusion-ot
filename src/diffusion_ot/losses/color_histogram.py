"""HistoGAN RGB-uv palette supervision against detached original source RGB.

RGB-uv, intensity weighting and inverse-quadratic kernels follow Afifi et al.,
HistoGAN (CVPR 2021), https://arxiv.org/abs/2011.11731 and the reference
https://github.com/mahmoudnafifi/HistoGAN/blob/master/histogram_classes/RGBuvHistBlock.py.
This implementation batches the three projections, checkpoints small chunks,
and averages per-image Hellinger distances (independent of batch size).
It does not implement HistoGAN's generator or add a brightness/spatial loss.

Reference implementation license: MIT, Copyright (c) 2021 Mahmoud Afifi.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


COLOR_PROTOCOL = "histogan_rgbuv_hellinger_v1"


def color_histogram_options(options=None):
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options) - {"weight", "bins", "input_size", "sigma"}:
        raise ValueError("color_histogram expects weight, bins, input_size and sigma only.")
    result = {"weight": 0.0, "bins": 64, "input_size": 128, "sigma": .02, **options}
    for name, minimum in (("bins", 2), ("input_size", 1)):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"color_histogram.{name} must be an integer >= {minimum}.")
    for name in ("weight", "sigma"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"color_histogram.{name} must be finite.")
        if value < 0 or (name == "sigma" and value == 0):
            raise ValueError(f"Invalid color_histogram.{name}.")
        result[name] = float(value)
    return result


def rgbuv_histogram(images, *, bins=64, input_size=128, sigma=.02):
    """Return normalized [B, 3, bins, bins] soft histograms for RGB in [0, 1].

All three projections are normalized jointly, as in the reference code.
Whole-image, deterministic bilinear resizing; no foreground masks are assumed.
FP32 computation is intentional even inside mixed-precision training.
"""
    color_histogram_options({"bins": bins, "input_size": input_size, "sigma": sigma})
    if images.ndim != 4 or images.shape[1] != 3 or not images.shape[0] or min(images.shape[2:]) < 1:
        raise ValueError("RGB-uv histogram requires nonempty [batch, 3, height, width] images.")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("RGB-uv histogram requires finite floating-point RGB images.")
    with torch.autocast(device_type=images.device.type, enabled=False):
        value = images.float().clamp(0, 1)
        if max(value.shape[-2:]) > input_size:
            value = F.interpolate(value, size=(input_size, input_size), mode="bilinear", align_corners=False)
        centers = torch.linspace(-3., 3., bins, device=value.device, dtype=torch.float32)

        def histogram(chunk):
            pixels = chunk.flatten(2).transpose(1, 2)  # B, pixels, RGB
            logs = (pixels + 1e-6).log()
            # (log R/G, log R/B), (log G/R, log G/B), (log B/R, log B/G).
            u = logs[..., [0, 1, 2]] - logs[..., [1, 0, 0]]
            v = logs[..., [0, 1, 2]] - logs[..., [2, 2, 1]]
            ku = 1 / (1 + ((u.transpose(1, 2).unsqueeze(-1) - centers) / sigma).square())
            kv = 1 / (1 + ((v.transpose(1, 2).unsqueeze(-1) - centers) / sigma).square())
            intensity = (pixels.square().sum(-1) + 1e-6).sqrt()[:, None, :, None]
            hist = (intensity * ku).transpose(-1, -2) @ kv
            # Positive inverse-quadratic kernels + intensity epsilon also make
            # black images well-defined. Normalize exactly rather than adding
            # epsilon to every bin (which would dilute very sparse histograms).
            return hist / hist.sum((1, 2, 3), keepdim=True).clamp_min(1e-12)

        pieces = []
        for chunk in value.split(4):
            pieces.append(checkpoint(histogram, chunk, use_reentrant=False)
                          if torch.is_grad_enabled() and chunk.requires_grad else histogram(chunk))
        return torch.cat(pieces)


def histogan_color_distance(generated, source, *, bins=64, input_size=128, sigma=.02):
    """Per-image Hellinger distances; only generated images receive gradients.

The normalized RGB-uv histogram is mostly insensitive to overall exposure and
contains no pixel correspondence: report brightness/structure separately.
"""
    if generated.shape != source.shape:
        raise ValueError("Color supervision requires corresponding generated/source RGB shapes.")
    options = dict(bins=bins, input_size=input_size, sigma=sigma)
    actual = rgbuv_histogram(generated, **options)
    with torch.no_grad():
        target = rgbuv_histogram(source.detach().to(generated), **options)
    # vector_norm has a defined zero gradient at identical histograms, unlike
    # sqrt(sum(square(...))) at zero. Clamp only before per-bin square roots.
    delta = actual.clamp_min(1e-12).sqrt() - target.clamp_min(1e-12).sqrt()
    return torch.linalg.vector_norm(delta.flatten(1), dim=1) / math.sqrt(2.)
