from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FlowTarget:
    x_t: Any
    target_v: Any
    noise: Any
    t: Any
    alpha_t: Any
    sigma_t: Any


def mean_flat(value):
    return value.flatten(start_dim=1).mean(dim=1)


def expand_time_like(t, value):
    while t.ndim < value.ndim:
        t = t.view(*t.shape, 1)
    return t


def sample_flow_timesteps(
    batch_size: int,
    device,
    dtype=None,
    eps: float = 1.0e-5,
):
    import torch

    t = torch.rand(batch_size, device=device, dtype=dtype)
    if eps > 0:
        t = t.clamp(eps, 1.0 - eps)
    return t


def linear_alpha_sigma(t, direction: str = "noise_to_data"):
    if direction == "noise_to_data":
        return t, 1.0 - t
    if direction == "data_to_noise":
        return 1.0 - t, t
    raise ValueError(f"Unsupported flow direction: {direction}")


def make_linear_flow_target(
    x0,
    noise=None,
    t=None,
    eps: float = 1.0e-5,
    direction: str = "noise_to_data",
) -> FlowTarget:
    import torch

    if noise is None:
        noise = torch.randn_like(x0)
    if t is None:
        t = sample_flow_timesteps(x0.shape[0], device=x0.device, dtype=x0.dtype, eps=eps)

    alpha_t, sigma_t = linear_alpha_sigma(t, direction=direction)
    alpha = expand_time_like(alpha_t, x0)
    sigma = expand_time_like(sigma_t, x0)
    x_t = alpha * x0 + sigma * noise
    if direction == "noise_to_data":
        target_v = x0 - noise
    elif direction == "data_to_noise":
        target_v = noise - x0
    else:
        raise ValueError(f"Unsupported flow direction: {direction}")
    return FlowTarget(
        x_t=x_t,
        target_v=target_v,
        noise=noise,
        t=t,
        alpha_t=alpha_t,
        sigma_t=sigma_t,
    )


def velocity_gap_target(target_v, base_v, detach: bool = True):
    target_delta_v = target_v - base_v
    return target_delta_v.detach() if detach else target_delta_v


def flow_snr(alpha_t, sigma_t, eps: float = 1.0e-8):
    return alpha_t.square() / (sigma_t.square() + eps)


def pdae_flow_snr_weight_from_snr(
    snr,
    gamma: float = 0.25,
    eps: float = 1.0e-8,
):
    one = snr.new_tensor(1.0)
    denom = one + snr
    clean_factor = (snr / denom).clamp_min(eps).pow(gamma)
    noise_factor = (one / denom).clamp_min(eps).pow(1.0 - gamma)
    return noise_factor * clean_factor


def pdae_flow_snr_weight(
    alpha_t=None,
    sigma_t=None,
    t=None,
    direction: str = "noise_to_data",
    gamma: float = 0.25,
    eps: float = 1.0e-8,
    normalize_mean_to: float | None = 1.0,
    clamp_min: float | None = 0.05,
    clamp_max: float | None = 5.0,
):
    if t is not None:
        alpha_t, sigma_t = linear_alpha_sigma(t, direction=direction)
    if alpha_t is None or sigma_t is None:
        raise ValueError("Provide either t or both alpha_t and sigma_t.")

    weight = pdae_flow_snr_weight_from_snr(
        flow_snr(alpha_t, sigma_t, eps=eps),
        gamma=gamma,
        eps=eps,
    )
    if normalize_mean_to is not None:
        target_mean = weight.new_tensor(float(normalize_mean_to))
        weight = weight * target_mean / weight.mean().clamp_min(eps)
    if clamp_min is not None or clamp_max is not None:
        weight = weight.clamp(min=clamp_min, max=clamp_max)
    return weight


def weighted_mean_flat(value, weight=None):
    per_sample = mean_flat(value)
    if weight is None:
        return per_sample.mean()
    return (per_sample * weight).mean()


def pdae_residual_mse(
    pred_delta_v,
    target_delta_v,
    weight=None,
):
    return weighted_mean_flat((pred_delta_v - target_delta_v).square(), weight=weight)


def pdae_velocity_gap_loss(
    pred_delta_v,
    target_v,
    base_v,
    weight=None,
    detach_target: bool = True,
):
    target_delta_v = velocity_gap_target(target_v, base_v, detach=detach_target)
    return pdae_residual_mse(pred_delta_v, target_delta_v, weight=weight)


class ResidualRMSNormalizer:
    def __init__(
        self,
        num_bins: int = 100,
        momentum: float = 0.99,
        eps: float = 1.0e-8,
    ) -> None:
        import torch

        self.num_bins = num_bins
        self.momentum = momentum
        self.eps = eps
        self.ema = torch.ones(num_bins)
        self.initialized = torch.zeros(num_bins, dtype=torch.bool)

    def _bins(self, t):
        return (t.detach().float().clamp(0, 1) * (self.num_bins - 1)).long()

    def update(self, t, target_delta_v) -> None:
        import torch

        bins = self._bins(t).cpu()
        values = mean_flat(target_delta_v.detach().float().square()).cpu()
        for bin_index in torch.unique(bins):
            mask = bins == bin_index
            value = values[mask].mean().clamp_min(self.eps)
            idx = int(bin_index)
            if not self.initialized[idx]:
                self.ema[idx] = value
                self.initialized[idx] = True
            else:
                self.ema[idx] = self.ema[idx] * self.momentum + value * (1.0 - self.momentum)

    def scale_for(self, t, device=None, dtype=None):
        bins = self._bins(t)
        scale = self.ema.to(device=device or t.device, dtype=dtype or t.dtype)[bins].clamp_min(self.eps)
        return scale

    def normalize_per_sample_loss(self, per_sample_loss, t):
        return per_sample_loss / self.scale_for(t, device=per_sample_loss.device, dtype=per_sample_loss.dtype)
