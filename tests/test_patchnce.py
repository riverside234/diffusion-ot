"""Behavior and gradient parity against the pinned official CUT implementation."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_ot.losses.patchnce import patchnce_loss
from diffusion_ot.third_party.cut.patchnce import PatchNCELoss
import diffusion_ot.third_party.cut.patchnce as upstream_module


def private_rng(seed=9):
    return torch.Generator(device="cpu").manual_seed(seed)


def upstream_sample(features, patch_ids):
    # Execute the verbatim pinned Normalize/PatchSampleF reference. No MLP is
    # constructed, and fixed IDs bypass its NumPy-global random sampler.
    scope = {"torch": torch, "nn": torch.nn, "np": np}
    path = Path(upstream_module.__file__).with_name("sampling_reference.txt")
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
    return scope["PatchSampleF"](use_mlp=False)(
        features, num_patches=len(patch_ids[0]),
        patch_ids=[ids.detach().cpu().numpy() for ids in patch_ids],
    )[0]


@pytest.mark.parametrize("batch", [1, 2, 5])
def test_value_and_query_gradient_match_official_loss_and_sampler(batch):
    rng = private_rng(61)
    query = torch.randn(batch, 9, 3, 4, generator=rng, requires_grad=True)
    key = torch.randn(batch, 9, 3, 4, generator=rng, requires_grad=True)
    loss, metrics = patchnce_loss([query], [key], num_patches=7, generator=private_rng())
    ids = [torch.tensor(metrics["layers"][0]["patch_ids"])]
    official_query = query.detach().clone().requires_grad_()
    official_key = key.detach().clone().requires_grad_()
    sampled_query = upstream_sample([official_query], ids)[0]
    sampled_key = upstream_sample([official_key], ids)[0]
    official = PatchNCELoss(SimpleNamespace(
        batch_size=batch, nce_T=.2, nce_includes_all_negatives_from_minibatch=False,
    ))(sampled_query, sampled_key).mean()
    torch.testing.assert_close(loss, official, atol=2e-7, rtol=1e-6)
    gradient, = torch.autograd.grad(loss, query)
    official_gradient, = torch.autograd.grad(official, official_query)
    torch.testing.assert_close(gradient, official_gradient, atol=3e-8, rtol=1e-5)
    assert key.grad is None and official_key.grad is None
    assert metrics["batch_size"] == batch
    assert metrics["layers"][0]["samples"] == batch * 7


def test_keys_are_detached_and_unsampled_query_positions_have_no_gradient():
    query = torch.randn(2, 5, 4, 4, generator=private_rng(), requires_grad=True)
    key = torch.randn(2, 5, 4, 4, generator=private_rng(10), requires_grad=True)
    loss, metrics = patchnce_loss([query], [key], num_patches=4, generator=private_rng())
    loss.backward()
    assert key.grad is None
    selected = torch.tensor(metrics["layers"][0]["patch_ids"])
    keep = torch.ones(16, dtype=torch.bool)
    keep[selected] = False
    assert query.grad.flatten(2)[:, :, keep].count_nonzero() == 0
    assert query.grad.flatten(2)[:, :, selected].norm() > 0


def test_negatives_are_within_images_and_actual_batch_size_is_used():
    query = torch.randn(3, 8, 3, 3, generator=private_rng())
    key = torch.randn(3, 8, 3, 3, generator=private_rng(18))
    batch_loss, _ = patchnce_loss([query], [key], num_patches=6, generator=private_rng())
    image_losses = [patchnce_loss([query[i:i+1]], [key[i:i+1]], num_patches=6,
                                  generator=private_rng())[0] for i in range(3)]
    torch.testing.assert_close(batch_loss, torch.stack(image_losses).mean())
    # Repeating a different image cannot alter an anchor's negative set.
    duplicate, _ = patchnce_loss([query[:1].repeat(4, 1, 1, 1)],
                               [key[:1].repeat(4, 1, 1, 1)], num_patches=6,
                               generator=private_rng())
    torch.testing.assert_close(duplicate, image_losses[0])


def test_same_positions_beat_shuffled_and_spatial_collapse_is_chance():
    own = torch.eye(8).reshape(1, 8, 2, 4)
    matched, good = patchnce_loss([own], [own], num_patches=8, generator=private_rng())
    shuffled, bad = patchnce_loss([own.roll(1, dims=3)], [own], num_patches=8,
                                 generator=private_rng())
    assert matched < shuffled
    assert good["retrieval_top1"] == 1
    assert bad["retrieval_top1"] == 0
    assert good["matched_cosine"] > .999
    constant = torch.ones(2, 8, 2, 4)
    collapsed_loss, collapsed = patchnce_loss([constant], [constant], num_patches=8,
                                             generator=private_rng())
    assert collapsed_loss.item() == pytest.approx(np.log(8), abs=1e-6)
    assert collapsed["retrieval_top1"] == collapsed["retrieval_chance"] == .125
    assert collapsed["query_collapsed_image_fraction"] == 1
    assert collapsed["key_collapsed_image_fraction"] == 1


def test_zero_maps_are_finite_and_reported_not_silently_skipped():
    query = torch.zeros(2, 4, 2, 2, requires_grad=True)
    key = torch.zeros_like(query, requires_grad=True)
    loss, metrics = patchnce_loss([query], [key], generator=private_rng())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(query.grad).all()
    assert key.grad is None
    assert metrics["query_zero_norm_fraction"] == 1
    assert metrics["key_zero_norm_fraction"] == 1
    assert metrics["retrieval_top1"] == .25
    assert metrics["samples"] == 8


def test_layer_reduction_is_equal_mean_and_patches_are_capped_by_resolution():
    query = [torch.randn(2, 8, 8, 8, generator=private_rng()),
             torch.randn(2, 16, 2, 2, generator=private_rng(10))]
    key = [value.roll(1, dims=-1) for value in query]
    loss, metrics = patchnce_loss(query, key, num_patches=64, generator=private_rng())
    assert [layer["patches_per_image"] for layer in metrics["layers"]] == [64, 4]
    assert float(loss) == pytest.approx(np.mean([item["loss"] for item in metrics["layers"]]))
    assert metrics["retrieval_chance"] == pytest.approx((1/64 + 1/4)/2)


def test_private_rng_is_repeatable_restorable_and_does_not_touch_global_rng():
    features = torch.randn(2, 4, 4, 4, generator=private_rng())
    global_state = torch.random.get_rng_state().clone()
    rng = private_rng()
    state = rng.get_state().clone()
    first, metrics = patchnce_loss([features], [features], num_patches=4, generator=rng)
    second, next_metrics = patchnce_loss([features], [features], num_patches=4, generator=rng)
    assert metrics["layers"][0]["patch_ids"] != next_metrics["layers"][0]["patch_ids"]
    rng.set_state(state)
    repeat, repeat_metrics = patchnce_loss([features], [features], num_patches=4, generator=rng)
    torch.testing.assert_close(first, repeat, atol=0, rtol=0)
    assert metrics == repeat_metrics
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert torch.isfinite(second)


def test_bfloat16_autocast_uses_fp32_loss_with_finite_gradient():
    query = torch.randn(2, 8, 4, 4, generator=private_rng()).bfloat16().requires_grad_()
    key = query.detach().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, metrics = patchnce_loss([query], [key], generator=private_rng())
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(query.grad).all()
    assert metrics["retrieval_top1"] == 1


@pytest.mark.parametrize("kwargs", [{"num_patches": 1}, {"num_patches": True},
                                    {"temperature": 0}, {"temperature": float("nan")},
                                    {"generator": None}])
def test_invalid_options_are_rejected(kwargs):
    feature = torch.ones(1, 4, 2, 2)
    options = {"generator": private_rng(), **kwargs}
    with pytest.raises(ValueError):
        patchnce_loss([feature], [feature], **options)


@pytest.mark.parametrize("kind", ["empty", "mismatch", "one_location", "nonfinite"])
def test_invalid_inputs_do_not_advance_private_rng(kind):
    query, key = [torch.ones(2, 4, 2, 2)], [torch.ones(2, 4, 2, 2)]
    if kind == "empty":
        query, key = [], []
    elif kind == "mismatch":
        key[0] = key[0][:1]
    elif kind == "one_location":
        query, key = [torch.ones(2, 4, 1, 1)], [torch.ones(2, 4, 1, 1)]
    else:
        query[0][0, 0, 0, 0] = float("nan")
    rng = private_rng()
    state = rng.get_state().clone()
    with pytest.raises((ValueError, FloatingPointError)):
        patchnce_loss(query, key, generator=rng)
    assert torch.equal(state, rng.get_state())
