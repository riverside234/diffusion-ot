"""Balance decoded and reconstruction gradients on the semantic encoders only.

The ratio is a configurable translation-first heuristic, not an optimum or a
constraint on Adam updates. The controller rescales the existing decoded
gradient without projecting conflicts. Other losses and other parameter groups
keep their ordinary weighted-sum gradients.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DecodedEncoderBalanceConfig:
    enabled: bool = False
    target_ratio: float = 2.0
    min_scale: float = 0.25
    max_scale: float = 4.0
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("decoded_encoder_balance.enabled must be a boolean.")
        for name in ("target_ratio", "min_scale", "max_scale", "eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"decoded_encoder_balance.{name} must be numeric.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"decoded_encoder_balance.{name} must be finite and positive.")
        if self.min_scale > self.max_scale:
            raise ValueError("decoded_encoder_balance.min_scale must not exceed max_scale.")

    @classmethod
    def from_mapping(cls, value: Mapping | None) -> DecodedEncoderBalanceConfig:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("decoded_encoder_balance must be a mapping.")
        unknown = set(value) - {"enabled", "target_ratio", "min_scale", "max_scale", "eps"}
        if unknown:
            raise ValueError(f"Unknown decoded_encoder_balance options: {sorted(unknown)}")
        return cls(**value)


def _loss_gradients(
    loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor | None, ...]:
    if loss.numel() != 1:
        raise ValueError("Encoder gradient balancing requires scalar losses.")
    if not loss.requires_grad:
        return (None,) * len(parameters)
    return tuple(
        None if gradient is None else gradient.detach()
        for gradient in torch.autograd.grad(
            loss, parameters, retain_graph=True, create_graph=False, allow_unused=True,
        )
    )


def decoded_encoder_gradient_correction(
    reconstruction: torch.Tensor,
    decoded: torch.Tensor,
    encoder_parameters: Sequence[torch.nn.Parameter],
    *,
    config: DecodedEncoderBalanceConfig,
    ramp: float = 1.0,
) -> tuple[tuple[torch.Tensor | None, ...], dict[str, float | bool | str | None]]:
    """Return detached encoder corrections to add after ``total.backward()``.

    Supply the weighted reconstruction and weighted, already-ramped decoded
    losses that participate in ``total``. The target norm ratio also receives
    the decoded ramp so balancing does not cancel the image-loss warmup. Only
    gradients with respect to the supplied encoder parameters are measured.

    Add each non-None correction to its parameter's ``.grad`` after the ordinary
    total backward and before clipping. The correction is
    ``(scale - 1) * grad(decoded)``; no matching-head or generator correction is
    produced. Do not combine this controller with the legacy gradient guard.
    Near-zero reconstruction/decoded gradients and a zero ramp skip correction,
    preserving the ordinary gradient instead of dividing by a tiny norm.
    """
    if isinstance(ramp, bool) or not math.isfinite(ramp) or not 0 <= ramp <= 1:
        raise ValueError("Decoded encoder balance ramp must be finite and in [0, 1].")
    parameters = tuple(encoder_parameters)
    if not parameters:
        raise ValueError("Decoded encoder balancing needs nonempty encoder parameters.")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("Decoded encoder balance parameters must not overlap.")
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("Decoded encoder balance parameters must require gradients.")
    empty = (None,) * len(parameters)
    metrics: dict[str, float | bool | str | None] = {
        "enabled": config.enabled,
        "applied": False,
        "scale": 1.0,
        "target_ratio": float(config.target_ratio),
        "ramp": float(ramp),
        "effective_target_ratio": float(config.target_ratio * ramp),
        "reconstruction_gradient_norm": None,
        "decoded_gradient_norm_before": None,
        "decoded_gradient_norm_after": None,
        "prebalanced_ratio": None,
        "effective_ratio": None,
        "cosine": None,
        "clamped": False,
        "skip_reason": "disabled" if not config.enabled else "",
    }
    if not config.enabled:
        return empty, metrics

    reconstruction_gradients = _loss_gradients(reconstruction, parameters)
    decoded_gradients = _loss_gradients(decoded, parameters)
    # Float32 accumulation avoids half-precision overflow. Each domain may
    # live on a separate device; transfer only its three accumulated scalars.
    # No second-order graphs or parameter-sized diagnostics are retained.
    device_statistics = {
        device: torch.zeros(3, device=device, dtype=torch.float32)
        for device in dict.fromkeys(parameter.device for parameter in parameters)
    }
    for parameter, reconstruction_gradient, decoded_gradient in zip(
        parameters, reconstruction_gradients, decoded_gradients,
    ):
        statistics = device_statistics[parameter.device]
        if reconstruction_gradient is not None:
            statistics[0] += reconstruction_gradient.float().square().sum()
        if decoded_gradient is not None:
            statistics[1] += decoded_gradient.float().square().sum()
        if reconstruction_gradient is not None and decoded_gradient is not None:
            statistics[2] += (reconstruction_gradient.float() * decoded_gradient.float()).sum()
    device_values = [statistics.cpu().tolist() for statistics in device_statistics.values()]
    reconstruction_squared, decoded_squared, dot = (
        sum(values[index] for values in device_values) for index in range(3)
    )
    if not all(math.isfinite(value) for value in (reconstruction_squared, decoded_squared, dot)):
        raise FloatingPointError("Non-finite decoded/reconstruction encoder gradients.")
    reconstruction_norm = math.sqrt(reconstruction_squared)
    decoded_norm = math.sqrt(decoded_squared)
    ratio = decoded_norm / reconstruction_norm if reconstruction_norm > config.eps else None
    cosine = (
        max(-1.0, min(1.0, dot / (reconstruction_norm * decoded_norm)))
        if reconstruction_norm > config.eps and decoded_norm > config.eps else None
    )
    metrics.update({
        "reconstruction_gradient_norm": reconstruction_norm,
        "decoded_gradient_norm_before": decoded_norm,
        "decoded_gradient_norm_after": decoded_norm,
        "prebalanced_ratio": ratio,
        "effective_ratio": ratio,
        "cosine": cosine,
    })
    if ramp == 0 or reconstruction_norm <= config.eps or decoded_norm <= config.eps:
        metrics["skip_reason"] = (
            "zero_ramp" if ramp == 0 else
            "tiny_reconstruction_gradient" if reconstruction_norm <= config.eps else
            "tiny_decoded_gradient"
        )
        return empty, metrics

    requested_scale = config.target_ratio * ramp * reconstruction_norm / decoded_norm
    scale = min(config.max_scale, max(config.min_scale, requested_scale))
    corrections = tuple(
        None if gradient is None or scale == 1.0 else gradient * (scale - 1.0)
        for gradient in decoded_gradients
    )
    metrics.update({
        "applied": scale != 1.0,
        "scale": scale,
        "decoded_gradient_norm_after": decoded_norm * scale,
        "effective_ratio": float(ratio) * scale,
        "clamped": requested_scale < config.min_scale or requested_scale > config.max_scale,
    })
    return corrections, metrics
