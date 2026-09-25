"""Residual semantic CNN for clean VAE latents (not a diffusion denoiser)."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"encoder.{name} must be a positive integer.")
    return value


def _norm(channels: int, preferred_groups: int) -> nn.GroupNorm:
    groups = min(channels, preferred_groups)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class EncoderResidualBlock(nn.Module):
    """Pre-activation convolutions with an unnormalized shortcut."""

    def __init__(self, in_channels: int, out_channels: int, *, num_groups: int, dropout: float):
        super().__init__()
        self.residual = nn.Sequential(
            _norm(in_channels, num_groups), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            _norm(out_channels, num_groups), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = (nn.Identity() if in_channels == out_channels
                         else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.shortcut(value) + self.residual(value)


class EncoderSpatialAttention(nn.Module):
    """Residual multi-head attention over spatial positions at one resolution."""

    def __init__(self, channels: int, *, num_groups: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm = _norm(channels, num_groups)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        qkv = self.qkv(self.norm(value)).reshape(
            batch, 3, self.num_heads, channels // self.num_heads, height * width,
        )
        query, key, content = qkv.unbind(dim=1)
        attended = F.scaled_dot_product_attention(
            query.transpose(-1, -2), key.transpose(-1, -2), content.transpose(-1, -2),
            dropout_p=0.0,
        )
        attended = attended.transpose(-1, -2).reshape(batch, channels, height, width)
        return value + self.proj(attended)


class PDAEResidualLatentEncoder(nn.Module):
    """32x32 -> 16x16 -> 8x8 -> 4x4 features -> spatially projected code.

    Stage indices identify outputs *after* residual blocks and optional attention.
    The input stem has no normalization or activation before its stride-1 conv.
    Configuration is saved with weights: head/group counts alter the computation
    without necessarily altering any tensor shape, so shape checks alone fail.
    """

    def __init__(
        self, *, input_channels: int = 4, input_size: int = 32, z_dim: int = 512,
        channels: Iterable[int] = (64, 128, 256, 256),
        blocks_per_stage: Iterable[int] = (2, 2, 2, 2), spatial_size: int = 4,
        num_groups: int = 32, normalize_z: bool = True,
        attention_resolutions: Iterable[int] = (16,), attention_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_channels = _positive_int(input_channels, "input_channels")
        self.input_size = _positive_int(input_size, "input_size")
        self.z_dim = _positive_int(z_dim, "z_dim")
        self.spatial_size = _positive_int(spatial_size, "spatial_size")
        self.num_groups = _positive_int(num_groups, "num_groups")
        self.attention_heads = _positive_int(attention_heads, "attention_heads")
        self.channels = tuple(_positive_int(c, "channels") for c in channels)
        self.blocks_per_stage = tuple(_positive_int(n, "blocks_per_stage") for n in blocks_per_stage)
        if not self.channels or len(self.channels) != len(self.blocks_per_stage):
            raise ValueError("encoder.channels and blocks_per_stage must be non-empty lists of equal length.")
        factor = 2 ** (len(self.channels) - 1)
        if self.input_size % factor or self.input_size // factor != self.spatial_size:
            raise ValueError("encoder.input_size must downsample exactly to spatial_size through its stages.")
        self.resolutions = tuple(self.input_size // (2 ** i) for i in range(len(self.channels)))
        attention = tuple(_positive_int(r, "attention_resolutions") for r in attention_resolutions)
        if len(set(attention)) != len(attention) or not set(attention).issubset(self.resolutions):
            raise ValueError("encoder.attention_resolutions must be distinct resolutions present in the encoder.")
        self.attention_resolutions = tuple(sorted(attention, reverse=True))
        for width, resolution in zip(self.channels, self.resolutions):
            if resolution in attention and width % self.attention_heads:
                raise ValueError("encoder attention channels must be divisible by attention_heads.")
        self.dropout = float(dropout)
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("encoder.dropout must be finite and in [0, 1).")
        if not isinstance(normalize_z, bool):
            raise ValueError("encoder.normalize_z must be a Boolean.")
        self.normalize_z = normalize_z

        self.stem = nn.Conv2d(self.input_channels, self.channels[0], 3, stride=1, padding=1)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_channels = self.channels[0]
        for index, (width, count, resolution) in enumerate(zip(
            self.channels, self.blocks_per_stage, self.resolutions,
        )):
            blocks: list[nn.Module] = []
            for _ in range(count):
                blocks.append(EncoderResidualBlock(
                    in_channels, width, num_groups=self.num_groups, dropout=self.dropout,
                ))
                in_channels = width
            if resolution in attention:
                blocks.append(EncoderSpatialAttention(
                    width, num_groups=self.num_groups, num_heads=self.attention_heads,
                ))
            self.stages.append(nn.Sequential(*blocks))
            if index < len(self.channels) - 1:
                self.downsamples.append(nn.Conv2d(width, width, 3, stride=2, padding=1))
        self.out_norm = _norm(self.channels[-1], self.num_groups)
        self.out_act = nn.SiLU()
        self.proj = nn.Linear(self.channels[-1] * self.spatial_size ** 2, self.z_dim)
        self.z_norm = nn.LayerNorm(self.z_dim) if self.normalize_z else nn.Identity()

    @property
    def architecture_spec(self) -> dict[str, Any]:
        return {
            "kind": "residual_cnn_v1", "input_channels": self.input_channels,
            "input_size": self.input_size, "stem_stride": 1, "normalize_input": False,
            "channels": list(self.channels), "blocks_per_stage": list(self.blocks_per_stage),
            "spatial_size": self.spatial_size, "num_groups": self.num_groups,
            "attention_resolutions": list(self.attention_resolutions),
            "attention_heads": self.attention_heads, "dropout": self.dropout,
            "z_dim": self.z_dim, "normalize_z": self.normalize_z,
        }

    @property
    def spatial_feature_channels(self) -> tuple[int, ...]:
        return self.channels

    def _feature_maps(self, value: torch.Tensor):
        expected = (self.input_channels, self.input_size, self.input_size)
        if value.ndim != 4 or tuple(value.shape[1:]) != expected:
            raise ValueError(f"Residual latent encoder expects [batch,{','.join(map(str, expected))}].")
        h = self.stem(value)
        for index, stage in enumerate(self.stages):
            if index:
                h = self.downsamples[index - 1](h)
            h = stage(h)
            yield h

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for h in self._feature_maps(value):
            pass
        h = self.out_act(self.out_norm(h))
        return self.z_norm(self.proj(h.flatten(start_dim=1)))

    def forward_spatial_features(self, value: torch.Tensor, layers: Iterable[int]) -> list[torch.Tensor]:
        layers = tuple(layers)
        if (not layers or len(set(layers)) != len(layers)
                or any(isinstance(i, bool) or not isinstance(i, int)
                       or i < 0 or i >= len(self.channels) for i in layers)):
            raise ValueError("Spatial feature layers must be distinct valid residual-stage indices.")
        selected = {}
        for index, h in enumerate(self._feature_maps(value)):
            if index in layers:
                selected[index] = h
            if index == max(layers):
                break
        return [selected[index] for index in layers]
