import math

import pytest
import torch

from diffusion_ot.training.gradient_balance import (
    DecodedEncoderBalanceConfig,
    decoded_encoder_gradient_correction,
)


def apply_corrections(parameters, corrections):
    with torch.no_grad():
        for parameter, correction in zip(parameters, corrections):
            if correction is not None:
                if parameter.grad is None:
                    parameter.grad = correction.clone()
                else:
                    parameter.grad.add_(correction)


def test_controller_changes_only_decoded_encoder_contribution_and_actual_update():
    encoder = torch.nn.Parameter(torch.tensor([1., 2.]))
    head = torch.nn.Parameter(torch.tensor(3.))
    generator = torch.nn.Parameter(torch.tensor(4.))
    reconstruction = encoder[0] + 2 * generator
    decoded = 8 * encoder[1] + 3 * head + 5 * generator
    other_objectives = 7 * encoder[0] + 11 * head
    total = reconstruction + decoded + other_objectives
    correction, metrics = decoded_encoder_gradient_correction(
        reconstruction, decoded, [encoder], config=DecodedEncoderBalanceConfig(enabled=True),
    )
    assert encoder.grad is head.grad is generator.grad is None
    total.backward()
    apply_corrections([encoder], correction)
    torch.testing.assert_close(encoder.grad, torch.tensor([8., 2.]))
    assert head.grad == 14
    assert generator.grad == 7
    assert metrics["prebalanced_ratio"] == 8
    assert metrics["effective_ratio"] == 2
    assert metrics["cosine"] == 0
    optimizer = torch.optim.SGD([encoder, head, generator], lr=.1)
    optimizer.step()
    torch.testing.assert_close(encoder.detach(), torch.tensor([.2, 1.8]))
    assert not correction[0].requires_grad and correction[0].grad_fn is None
    assert all(not isinstance(value, torch.Tensor) for value in metrics.values())


@pytest.mark.parametrize("sign", [-1., 1.])
def test_controller_preserves_decoded_direction_even_when_conflicting(sign):
    encoder = torch.nn.Parameter(torch.ones(2))
    reconstruction = encoder[0]
    decoded = sign * encoder[0]
    correction, metrics = decoded_encoder_gradient_correction(
        reconstruction, decoded, [encoder], config=DecodedEncoderBalanceConfig(enabled=True),
    )
    (reconstruction + decoded).backward()
    apply_corrections([encoder], correction)
    torch.testing.assert_close(encoder.grad, torch.tensor([1. + 2. * sign, 0.]))
    assert metrics["cosine"] == sign
    assert metrics["effective_ratio"] == 2


@pytest.mark.parametrize("ramp", [0., .1, .5, 1.])
def test_target_tracks_warmup_instead_of_cancelling_it(ramp):
    encoder = torch.nn.Parameter(torch.ones(2))
    reconstruction = encoder[0]
    decoded = ramp * encoder[1]
    correction, metrics = decoded_encoder_gradient_correction(
        reconstruction, decoded, [encoder], config=DecodedEncoderBalanceConfig(enabled=True), ramp=ramp,
    )
    (reconstruction + decoded).backward()
    apply_corrections([encoder], correction)
    torch.testing.assert_close(encoder.grad, torch.tensor([1., 2. * ramp]))
    assert metrics["effective_target_ratio"] == 2. * ramp
    if ramp == 0:
        assert metrics["scale"] == 1
        assert metrics["skip_reason"] == "zero_ramp"
    else:
        assert metrics["effective_ratio"] == pytest.approx(2. * ramp)


@pytest.mark.parametrize("decoded_weight,expected_scale", [(.001, 4.), (100., .25)])
def test_scale_is_bounded_and_effective_ratio_reports_missed_target(decoded_weight, expected_scale):
    encoder = torch.nn.Parameter(torch.ones(2))
    correction, metrics = decoded_encoder_gradient_correction(
        encoder[0], decoded_weight * encoder[1], [encoder],
        config=DecodedEncoderBalanceConfig(enabled=True),
    )
    assert metrics["scale"] == expected_scale
    assert metrics["clamped"]
    assert metrics["effective_ratio"] == pytest.approx(decoded_weight * expected_scale)
    assert metrics["effective_ratio"] != metrics["effective_target_ratio"]
    assert correction[0] is not None


@pytest.mark.parametrize("zero_reconstruction,zero_decoded", [(True, False), (False, True), (True, True)])
def test_zero_gradients_skip_balancing_without_suppressing_other_gradients(zero_reconstruction, zero_decoded):
    encoder = torch.nn.Parameter(torch.ones(2))
    reconstruction = encoder[0] * (0 if zero_reconstruction else 1)
    decoded = encoder[1] * (0 if zero_decoded else 10)
    correction, metrics = decoded_encoder_gradient_correction(
        reconstruction, decoded, [encoder], config=DecodedEncoderBalanceConfig(enabled=True),
    )
    assert correction == (None,)
    assert not metrics["applied"]
    assert metrics["scale"] == 1
    assert metrics["cosine"] is None
    (reconstruction + decoded).backward()
    torch.testing.assert_close(encoder.grad, torch.tensor([0. if zero_reconstruction else 1., 0. if zero_decoded else 10.]))


def test_unused_parameters_and_constant_losses_are_supported():
    encoder = torch.nn.Parameter(torch.ones(2))
    unused = torch.nn.Parameter(torch.ones(3))
    correction, metrics = decoded_encoder_gradient_correction(
        encoder[0], encoder[1], [encoder, unused], config=DecodedEncoderBalanceConfig(enabled=True),
    )
    assert correction[1] is None
    assert metrics["effective_ratio"] == 2
    for reconstruction, decoded in ((torch.tensor(0.), encoder.sum()), (encoder.sum(), torch.tensor(0.))):
        correction, metrics = decoded_encoder_gradient_correction(
            reconstruction, decoded, [encoder], config=DecodedEncoderBalanceConfig(enabled=True),
        )
        assert correction == (None,)
        assert not metrics["applied"]


@pytest.mark.parametrize("tiny_component", ["reconstruction", "decoded"])
def test_tiny_gradient_skips_an_unreliable_ratio(tiny_component):
    encoder = torch.nn.Parameter(torch.ones(2))
    reconstruction = encoder[0] * (1e-14 if tiny_component == "reconstruction" else 1)
    decoded = encoder[1] * (1e-14 if tiny_component == "decoded" else 1)
    corrections, metrics = decoded_encoder_gradient_correction(
        reconstruction, decoded, [encoder], config=DecodedEncoderBalanceConfig(enabled=True),
    )
    assert corrections == (None,)
    assert metrics["skip_reason"] == f"tiny_{tiny_component}_gradient"
    assert metrics["scale"] == 1


def test_half_precision_gradients_accumulate_norms_in_float32():
    encoder = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
    reconstruction = 300 * encoder[0]
    decoded = 600 * encoder[1]
    corrections, metrics = decoded_encoder_gradient_correction(
        reconstruction, decoded, [encoder], config=DecodedEncoderBalanceConfig(enabled=True, target_ratio=1.),
    )
    assert metrics["reconstruction_gradient_norm"] == 300
    assert metrics["decoded_gradient_norm_before"] == 600
    assert metrics["scale"] == .5
    (reconstruction + decoded).backward()
    apply_corrections([encoder], corrections)
    torch.testing.assert_close(encoder.grad, torch.tensor([300., 300.], dtype=torch.float16))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_separate_domain_devices_share_global_ratio_without_moving_gradients():
    cat = torch.nn.Parameter(torch.ones(2, device="cuda:0"))
    dog = torch.nn.Parameter(torch.ones(2, device="cuda:1"))
    reconstruction = cat[0] + 2 * dog[1].to(cat.device)
    decoded = 4 * cat[1] + 8 * dog[0].to(cat.device)
    corrections, metrics = decoded_encoder_gradient_correction(
        reconstruction, decoded, [cat, dog], config=DecodedEncoderBalanceConfig(enabled=True),
    )
    assert metrics["prebalanced_ratio"] == pytest.approx(4.)
    assert metrics["effective_ratio"] == pytest.approx(2.)
    assert metrics["cosine"] == 0
    assert all(correction.device == parameter.device for correction, parameter in zip(corrections, [cat, dog]))
    (reconstruction + decoded).backward()
    apply_corrections([cat, dog], corrections)
    torch.testing.assert_close(cat.grad, torch.tensor([1., 2.], device=cat.device))
    torch.testing.assert_close(dog.grad, torch.tensor([4., 2.], device=dog.device))


def test_disabled_config_preserves_legacy_backward_and_no_autograd_work(monkeypatch):
    encoder = torch.nn.Parameter(torch.ones(2))
    reconstruction, decoded = encoder[0], 10 * encoder[1]
    def unexpected_autograd(*args, **kwargs):
        raise AssertionError("disabled controller requested gradients")
    monkeypatch.setattr(torch.autograd, "grad", unexpected_autograd)
    correction, metrics = decoded_encoder_gradient_correction(
        reconstruction, decoded, [encoder], config=DecodedEncoderBalanceConfig.from_mapping(None),
    )
    assert correction == (None,)
    assert metrics["skip_reason"] == "disabled"
    (reconstruction + decoded).backward()
    torch.testing.assert_close(encoder.grad, torch.tensor([1., 10.]))


@pytest.mark.parametrize("settings", [
    {"target_ratio": math.nan}, {"target_ratio": 0}, {"max_scale": math.inf},
    {"min_scale": -.1}, {"min_scale": 4., "max_scale": 2.}, {"eps": 0},
    {"eps": 1e-12, "enabled": "false"}, {"target_ratio": True}, {"unknown": 1},
])
def test_invalid_controller_options_are_rejected_even_when_disabled(settings):
    with pytest.raises(ValueError, match="decoded_encoder_balance"):
        DecodedEncoderBalanceConfig.from_mapping(settings)


@pytest.mark.parametrize("ramp", [-1, 2, math.nan, math.inf])
def test_invalid_ramp_rejected(ramp):
    encoder = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="ramp"):
        decoded_encoder_gradient_correction(
            encoder.sum(), encoder.sum(), [encoder], config=DecodedEncoderBalanceConfig(enabled=True), ramp=ramp,
        )


def test_nonfinite_gradients_fail_before_correction():
    encoder = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(FloatingPointError, match="Non-finite"):
        decoded_encoder_gradient_correction(
            encoder.sum(), encoder.sum() * math.inf, [encoder], config=DecodedEncoderBalanceConfig(enabled=True),
        )


def test_retained_graph_can_be_consumed_once_and_corrections_do_not_keep_it():
    encoder = torch.nn.Parameter(torch.tensor([1., 2.]))
    reconstruction = encoder[0].square()
    decoded = encoder[1].square()
    corrections, metrics = decoded_encoder_gradient_correction(
        reconstruction, decoded, [encoder], config=DecodedEncoderBalanceConfig(enabled=True, target_ratio=1.5),
    )
    total = reconstruction + decoded
    total.backward()
    apply_corrections([encoder], corrections)
    torch.testing.assert_close(encoder.grad, torch.tensor([2., 3.]))
    assert all(correction is None or correction.grad_fn is None for correction in corrections)
    with pytest.raises(RuntimeError, match="second time"):
        total.backward()
