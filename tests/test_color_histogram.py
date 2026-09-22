import math

import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.losses.color_histogram import color_histogram_options, histogan_color_distance, rgbuv_histogram


def reference_rgbuv(images, bins, sigma):
    # Scalar-image/plane formulation of HistoGAN's reference, independent of
    # the batched production implementation and its checkpoint/chunk layout.
    centers = torch.linspace(-3., 3., bins)
    result = []
    for image in images:
        pixels = image.flatten(1).T
        intensity = (pixels.square().sum(1) + 1e-6).sqrt()
        planes = []
        for primary, first, second in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
            u = (pixels[:, primary] + 1e-6).log() - (pixels[:, first] + 1e-6).log()
            v = (pixels[:, primary] + 1e-6).log() - (pixels[:, second] + 1e-6).log()
            hu = 1 / (1 + (u[:, None] - centers).square() / sigma ** 2)
            hv = 1 / (1 + (v[:, None] - centers).square() / sigma ** 2)
            planes.append((intensity[:, None] * hu).T @ hv)
        hist = torch.stack(planes)
        result.append(hist / hist.sum())
    return torch.stack(result)


def test_rgbuv_matches_reference_values_and_gradients_across_chunks():
    torch.manual_seed(2)
    images = (.1 + .8 * torch.rand(5, 3, 5, 7)).requires_grad_()
    actual = rgbuv_histogram(images, bins=16, sigma=.1)
    expected = reference_rgbuv(images, 16, .1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual.sum((1, 2, 3)), torch.ones(5))
    weight = torch.rand_like(actual)
    actual_grad = torch.autograd.grad((actual * weight).sum(), images)[0]
    expected_grad = torch.autograd.grad((expected * weight).sum(), images)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("value", [0., .5, 1.])
def test_identical_and_black_images_have_zero_distance_and_finite_gradients(value):
    image = torch.full((2, 3, 8, 8), value, requires_grad=True)
    target = image.detach().clone().requires_grad_()
    distances = histogan_color_distance(image, target)
    torch.testing.assert_close(distances, torch.zeros(2), atol=0, rtol=0)
    distances.mean().backward()
    assert torch.isfinite(image.grad).all() and image.grad.count_nonzero() == 0
    assert target.grad is None


def test_palette_mismatch_has_live_gradients_and_batch_independent_mean():
    source = torch.tensor([.7, .2, .1]).view(1, 3, 1, 1).expand(1, 3, 7, 7).clone().requires_grad_()
    generated = torch.tensor([.1, .25, .8]).view(1, 3, 1, 1).expand_as(source).clone().requires_grad_()
    distance = histogan_color_distance(generated, source)
    assert .1 < distance.item() <= 1.
    repeated = histogan_color_distance(generated.repeat(5, 1, 1, 1), source.repeat(5, 1, 1, 1))
    assert repeated.mean().item() == pytest.approx(distance.item(), abs=1e-6)
    distance.mean().backward()
    assert generated.grad.norm() > 0 and torch.isfinite(generated.grad).all()
    assert source.grad is None


def test_histogram_is_spatially_invariant_and_mostly_exposure_invariant():
    image = .1 + .4 * torch.rand(1, 3, 8, 8)
    shuffled = image.flatten(2)[:, :, torch.randperm(64)].reshape_as(image)
    assert histogan_color_distance(image, shuffled).item() < 1e-6
    # This is a palette loss, not a black/white brightness or spatial-marking loss.
    assert histogan_color_distance(image, image * 1.5).item() < 1e-4


def test_resize_mixed_precision_and_rng_are_deterministic():
    image = torch.rand(2, 3, 20, 24, dtype=torch.bfloat16, requires_grad=True)
    state = torch.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = rgbuv_histogram(image, bins=8, input_size=12)
    expected = rgbuv_histogram(F.interpolate(image.float(), (12, 12), mode="bilinear", align_corners=False), bins=8)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert torch.isfinite(image.grad).all()
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("options", [[], {"weight": -1}, {"weight": True}, {"sigma": 0},
    {"sigma": math.nan}, {"sigma": math.inf}, {"bins": 1}, {"bins": True},
    {"input_size": 0}, {"input_size": 1.5}, {"mask": "invented"}])
def test_invalid_color_options_fail_early(options):
    with pytest.raises(ValueError, match="color_histogram"):
        color_histogram_options(options)
