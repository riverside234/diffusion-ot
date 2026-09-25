from __future__ import annotations

import math
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


def test_fixed_uniform_weight_normalization_has_unit_population_mean():
    from diffusion_ot.losses.pdae_flow import pdae_flow_snr_weight

    sample_count = 65536
    t = (torch.arange(sample_count, dtype=torch.float64) + 0.5) / sample_count
    weight = pdae_flow_snr_weight(
        t=t,
        direction="noise_to_data",
        gamma=0.1,
        normalization_mode="fixed_uniform",
        normalization_samples=sample_count,
        clamp_min=None,
        clamp_max=None,
    )

    torch.testing.assert_close(weight.mean(), torch.tensor(1.0, dtype=weight.dtype))


def test_fixed_uniform_weight_does_not_depend_on_minibatch_composition():
    from diffusion_ot.losses.pdae_flow import pdae_flow_snr_weight

    kwargs = {
        "direction": "noise_to_data",
        "gamma": 0.1,
        "normalization_mode": "fixed_uniform",
        "clamp_min": None,
        "clamp_max": None,
    }
    single = pdae_flow_snr_weight(t=torch.tensor([0.25]), **kwargs)
    mixed = pdae_flow_snr_weight(t=torch.tensor([0.25, 0.9]), **kwargs)

    torch.testing.assert_close(single[0], mixed[0])


def test_gamma_point_one_weight_peaks_near_noisy_quarter_path():
    from diffusion_ot.losses.pdae_flow import pdae_flow_snr_weight

    t = torch.linspace(0.001, 0.999, 10000)
    weight = pdae_flow_snr_weight(
        t=t,
        direction="noise_to_data",
        gamma=0.1,
        normalize_mean_to=None,
        normalization_mode="none",
        clamp_min=None,
        clamp_max=None,
    )

    peak_t = float(t[weight.argmax()])
    assert peak_t == pytest.approx(0.25, abs=0.002)


def test_velocity_gap_target_detaches_by_default():
    from diffusion_ot.losses.pdae_flow import velocity_gap_target

    target_v = torch.ones(2, 4, 8, 8, requires_grad=True)
    base_v = torch.zeros_like(target_v, requires_grad=True)

    target_delta = velocity_gap_target(target_v, base_v)

    assert not target_delta.requires_grad
    torch.testing.assert_close(target_delta, torch.ones_like(target_delta))


def test_uniform_flow_loss_is_direct_velocity_mse_with_the_same_gradient():
    from diffusion_ot.losses.pdae_flow import flow_loss_weight, pdae_velocity_gap_loss

    t = torch.tensor([0., .25, .5, .9, 1.], dtype=torch.float64)
    delta = torch.arange(10, dtype=t.dtype).reshape(5, 2).requires_grad_()
    base = torch.full_like(delta, 0.4, requires_grad=True)
    target = torch.full_like(delta, 0.7, requires_grad=True)
    weight = flow_loss_weight(t, {"type": "uniform"})
    loss = pdae_velocity_gap_loss(delta, target, base, weight=weight)
    direct_loss = (delta + base.detach() - target.detach()).square().mean()
    actual_grad = torch.autograd.grad(loss, (delta, base, target), allow_unused=True)
    expected_grad = torch.autograd.grad(direct_loss, delta)[0]
    torch.testing.assert_close(loss, direct_loss)
    torch.testing.assert_close(actual_grad[0], expected_grad)
    assert actual_grad[1:] == (None, None)
    torch.testing.assert_close(weight, torch.ones_like(t), rtol=0, atol=0)


def test_cosmap_has_unit_integral_symmetry_and_fixed_endpoint_weights():
    from diffusion_ot.losses.pdae_flow import flow_loss_weight

    t = torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)
    weight = flow_loss_weight(t, {"type": "cosmap"})
    expected = torch.tensor([2., 3.2, 4., 3.2, 2.], dtype=t.dtype) / math.pi
    torch.testing.assert_close(weight, expected)
    torch.testing.assert_close(weight, flow_loss_weight(t, {"type": "cosmap"}, direction="data_to_noise"))
    torch.testing.assert_close(weight[1:2], flow_loss_weight(t[1:2], {"type": "cosmap"}))
    midpoints = (torch.arange(10000, dtype=t.dtype) + .5) / 10000
    assert flow_loss_weight(midpoints, {"type": "cosmap"}).mean().item() == pytest.approx(1., abs=1e-8)


def test_cosmap_weighting_matches_its_sampling_measure():
    from diffusion_ot.losses.pdae_flow import flow_loss_weight

    u = (torch.arange(10000, dtype=torch.float64) + .5) / 10000
    sampled_t = 1 - 1 / (torch.tan(math.pi * u / 2) + 1)
    # Integrate a nonconstant loss by two independently expressed measures.
    weighted = (flow_loss_weight(u, {"type": "cosmap"}) * (1 + 2*u + 3*u.square())).mean()
    sampled = (1 + 2*sampled_t + 3*sampled_t.square()).mean()
    torch.testing.assert_close(weighted, sampled, rtol=0, atol=1e-8)


@pytest.mark.parametrize("weight_type", ["uniform", "cosmap"])
def test_new_flow_weighting_does_not_consume_rng_or_stack_snr_options(weight_type):
    from diffusion_ot.losses.pdae_flow import flow_loss_weight, resolve_flow_loss_weighting

    before = torch.get_rng_state().clone()
    t = torch.tensor([.2, .8], dtype=torch.float64)
    weight = flow_loss_weight(t, {"type": weight_type})
    assert weight.dtype == t.dtype and weight.device == t.device and weight.shape == t.shape
    assert torch.equal(before, torch.get_rng_state())
    for option in ("gamma", "normalize_mean_to", "normalization_mode", "clamp_min"):
        with pytest.raises(ValueError, match="accepts only 'type'"):
            resolve_flow_loss_weighting({"type": weight_type, option: .1})


@pytest.mark.parametrize("overrides", [{}, {"gamma": .25, "normalization_mode": "batch"},
                                       {"normalize_mean_to": None, "clamp_min": None}])
def test_flow_weight_dispatch_preserves_previous_stage1a_weights(overrides):
    from diffusion_ot.losses.pdae_flow import flow_loss_weight, pdae_flow_snr_weight

    t = torch.tensor([0., .01, .25, .5, .9, 1.], dtype=torch.float64)
    old_options = dict(gamma=.1, normalization_mode="fixed_uniform", normalization_samples=65536,
                       normalize_mean_to=1., clamp_min=.001, clamp_max=None)
    old_options.update(overrides)
    expected = pdae_flow_snr_weight(t=t, **old_options)
    # Omitted type/defaults are how historical checkpoints encode this recipe.
    torch.testing.assert_close(flow_loss_weight(t, overrides), expected, rtol=0, atol=0)
