"""Numerics and counterexamples for the four fixed-image translation objectives."""
from copy import deepcopy

import pytest
import torch

from diffusion_ot.losses.translation_image import (
    coarse_rgb_distance, local_layout_distance, sliced_wasserstein_l1,
    target_patch_swd, translation_image_losses, translation_image_options,
)


def image_options():
    return {
        "coarse_rgb": {"weight": .02, "sizes": [4, 8]},
        "color_histogram": {"weight": .02, "bins": 8, "input_size": 16, "sigma": .1},
        "local_layout": {"weight": .02, "sizes": [8, 16]},
        "target_patch_swd": {"weight": .01, "sizes": [16, 8], "patch_size": 3,
                             "patches_per_image": 8, "directions": 12},
    }


def rng(seed=7):
    return torch.Generator(device="cpu").manual_seed(seed)


def test_enabled_terms_detach_all_references_and_leave_generated_rgb_live():
    generated = torch.rand(3, 3, 16, 16, generator=rng()).requires_grad_()
    source = torch.rand(3, 3, 16, 16, generator=rng(8)).requires_grad_()
    real = torch.rand(3, 3, 16, 16, generator=rng(9)).requires_grad_()
    baseline = torch.rand(3, 3, 16, 16, generator=rng(10)).requires_grad_()
    options = translation_image_options(image_options())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        losses, metrics = translation_image_losses(generated, source, options,
            real_target=real, real_baseline=baseline, generator=rng(), per_image=True)
    for name, loss in losses.items():
        assert loss.dtype == torch.float32 and loss.isfinite() and loss > 0
        gradients = torch.autograd.grad(loss, (generated, source, real, baseline),
                                        allow_unused=True, retain_graph=True)
        assert gradients[0].isfinite().all() and gradients[0].norm() > 0, name
        assert all(g is None for g in gradients[1:]), name
    assert len(metrics["coarse_rgb"]["per_image_loss"]) == 3
    assert 0 <= metrics["local_layout"]["contributing_pair_fraction"] <= 1
    assert metrics["target_patch_swd"]["real_to_real_distance"] > 0


@pytest.mark.parametrize("constant", [False, True])
def test_identical_inputs_zero_with_finite_zero_gradients_even_for_flat_images(constant):
    image = (torch.full((2, 3, 16, 16), .5) if constant else
             torch.rand(2, 3, 16, 16, generator=rng())).requires_grad_()
    losses, _ = translation_image_losses(image, image, translation_image_options(image_options()),
                                          real_target=image, real_baseline=image, generator=rng())
    total = sum(losses.values())
    assert total.item() == pytest.approx(0, abs=1e-7)
    total.backward()
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().max() == 0


def test_sw1_known_quantiles_permutation_and_target_detach():
    actual = torch.tensor([[0.], [2.], [4.]], requires_grad=True)
    target = torch.tensor([[5.], [1.], [3.]], requires_grad=True)
    loss = sliced_wasserstein_l1(actual, target, torch.ones(1, 1))
    assert loss == 1
    loss.backward()
    torch.testing.assert_close(actual.grad, torch.full_like(actual, -1 / 3))
    assert target.grad is None


def test_batch_duplication_preserves_all_reductions_and_private_rng_leaves_global_rng_alone():
    a, b = (torch.rand(2, 3, 16, 16, generator=rng(seed)) for seed in (3, 4))
    options = translation_image_options(image_options())
    state = torch.get_rng_state().clone()
    first, _ = translation_image_losses(a, b, options, real_target=b, generator=rng())
    second, _ = translation_image_losses(a.repeat(2, 1, 1, 1), b.repeat(2, 1, 1, 1), options,
                                         real_target=b.repeat(2, 1, 1, 1), generator=rng())
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    torch.testing.assert_close(torch.get_rng_state(), state)


def test_grid_can_evade_coarse_rgb_but_is_detected_by_real_patch_statistics():
    source = torch.full((2, 3, 16, 16), .5)
    checker = (torch.arange(16)[:, None] + torch.arange(16)[None, :]) % 2 * .4 - .2
    artifact = source + checker
    assert coarse_rgb_distance(artifact, source, sizes=[4, 8]).max() < 1e-7
    loss, metrics = target_patch_swd(artifact, source, generator=rng(), sizes=[16, 8], patch_size=3)
    assert loss > 1
    assert metrics["bands"]["16"]["normalization_scale"] == pytest.approx(.01)


def test_layout_responds_to_spatial_shuffle_and_downweights_flat_source():
    source = torch.rand(2, 3, 16, 16, generator=rng()) * .4 + .2
    actual = source.flatten(2).roll(13, 2).reshape_as(source)
    changed, fraction = local_layout_distance(actual, source, sizes=[8, 16])
    same, _ = local_layout_distance(source, source, sizes=[8, 16])
    assert (changed > same + .01).all() and (fraction > .5).all()
    flat, contribution = local_layout_distance(actual, torch.full_like(source, .5), sizes=[8, 16])
    assert flat.max() == 0 and contribution.max() == 0


def test_coarse_rgb_exposure_error_and_histogram_palette_error_are_complementary():
    source = torch.rand(2, 3, 16, 16, generator=rng()) * .4 + .2
    options = translation_image_options(image_options())
    exposure, _ = translation_image_losses(source * 1.2, source,
        {k: options[k] for k in ("coarse_rgb", "color_histogram")})
    palette, _ = translation_image_losses(source[:, [1, 2, 0]], source,
        {"color_histogram": options["color_histogram"]})
    assert exposure["coarse_rgb"] > .05
    assert exposure["color_histogram"] < .001
    assert palette["color_histogram"] > .001


@pytest.mark.parametrize("term,key,value", [
    ("coarse_rgb", "weight", float("nan")), ("coarse_rgb", "weight", -1),
    ("local_layout", "sizes", [1]), ("local_layout", "contrast_eps", 0),
    ("target_patch_swd", "directions", True), ("target_patch_swd", "scale_floor", 0),
    ("target_patch_swd", "patch_size", 99), ("coarse_rgb", "typo", 1),
])
def test_invalid_image_protocol_is_rejected(term, key, value):
    options = deepcopy(image_options())
    options[term][key] = value
    with pytest.raises(ValueError):
        translation_image_options(options)


def test_legacy_configs_enable_no_image_objective():
    assert translation_image_options({}) == {}
    assert translation_image_options({"color_histogram": {"weight": 0}}) == {}
