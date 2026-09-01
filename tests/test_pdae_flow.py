from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def test_noise_to_data_linear_flow_target_uses_sit_direction():
    from diffusion_ot.losses.pdae_flow import make_linear_flow_target

    x0 = torch.ones(3, 1, 2, 2)
    noise = torch.zeros_like(x0)
    t = torch.tensor([0.0, 1.0, 0.25])

    target = make_linear_flow_target(x0, noise=noise, t=t, direction="noise_to_data")

    torch.testing.assert_close(target.x_t[0], noise[0])
    torch.testing.assert_close(target.x_t[1], x0[1])
    torch.testing.assert_close(target.x_t[2], torch.full_like(x0[2], 0.25))
    torch.testing.assert_close(target.target_v, x0 - noise)


def test_data_to_noise_direction_remains_available():
    from diffusion_ot.losses.pdae_flow import make_linear_flow_target

    x0 = torch.ones(1, 1, 2, 2)
    noise = torch.zeros_like(x0)
    target = make_linear_flow_target(x0, noise=noise, t=torch.tensor([0.25]), direction="data_to_noise")

    torch.testing.assert_close(target.x_t, torch.full_like(x0, 0.75))
    torch.testing.assert_close(target.target_v, noise - x0)


def test_flow_snr_weight_is_finite_and_positive():
    from diffusion_ot.losses.pdae_flow import pdae_flow_snr_weight

    t = torch.linspace(0.05, 0.95, 17)
    weight = pdae_flow_snr_weight(t=t, direction="noise_to_data", clamp_min=None, clamp_max=None)

    assert torch.isfinite(weight).all()
    assert torch.all(weight > 0)


def test_velocity_gap_target_detaches_by_default():
    from diffusion_ot.losses.pdae_flow import velocity_gap_target

    target_v = torch.ones(2, 4, 8, 8, requires_grad=True)
    base_v = torch.zeros_like(target_v, requires_grad=True)

    target_delta = velocity_gap_target(target_v, base_v)

    assert not target_delta.requires_grad
    torch.testing.assert_close(target_delta, torch.ones_like(target_delta))
