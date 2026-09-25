"""CUT PatchSampleF MLPs, eagerly built for the two PDAE feature samplers.

Architecture/initialization follow the pinned CUT networks.py (BSD-2-Clause;
see third_party/cut/LICENSE). Sampling and normalization live in losses.patchnce.
Unlike upstream's lazy create_mlp, all parameters exist before optimizer/EMA
creation. These heads do not change the global InfoOT matching representation.
"""
from __future__ import annotations

import torch
from torch import nn


class PatchSampleMLP(nn.Module):
    """One Linear(C,D)-ReLU-Linear(D,D) transform per selected encoder layer."""

    def __init__(self, channels, projection_dim=256, *, seed=0):
        super().__init__()
        self.channels = tuple(channels)
        self.projection_dim = projection_dim
        if (not self.channels or any(isinstance(c, bool) or not isinstance(c, int) or c < 1
                                     for c in self.channels)
                or isinstance(projection_dim, bool) or not isinstance(projection_dim, int)
                or projection_dim < 1):
            raise ValueError("Patch MLP channel widths and projection_dim must be positive integers.")
        # CPU-only initialization preserves diffusion, DataLoader and CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for index, width in enumerate(self.channels):
                self.add_module(f"mlp_{index}", nn.Sequential(
                    nn.Linear(width, projection_dim, device="cpu", dtype=torch.float32),
                    nn.ReLU(),
                    nn.Linear(projection_dim, projection_dim, device="cpu", dtype=torch.float32),
                ))
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0., .02)
                    nn.init.zeros_(module.bias)

    def forward(self, patches, layer):
        # The generated-image branch remains live even under mixed precision.
        with torch.autocast(device_type=patches.device.type, enabled=False):
            return getattr(self, f"mlp_{layer}")(patches.float())


def make_patch_projector(encoder, options, *, device, seed=0):
    if options.get("sampler", "sample") == "sample":
        return None
    channels = encoder.spatial_feature_channels
    return PatchSampleMLP([channels[i] for i in options["layers"]],
                          options["projection_dim"], seed=seed).to(device)


def load_patch_projector(encoder, options, checkpoint, domain, *, weights, device):
    """Load the matching raw/EMA sampler; never evaluate random MLP weights."""
    if weights not in {"raw", "ema"}:
        raise ValueError("Patch projector weights must be raw or ema.")
    projector = make_patch_projector(encoder, options, device=device)
    if projector is None:
        return None
    state_key = "patch_projector_ema" if weights == "ema" else "patch_projectors"
    state = (checkpoint.get(state_key) or {}).get(domain)
    if state is None:
        raise ValueError(f"Checkpoint has no {state_key}.{domain}; MLP PatchNCE requires a fresh run.")
    projector.load_state_dict(state, strict=True)
    return projector.eval().requires_grad_(False)
