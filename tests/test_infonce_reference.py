"""Compare values and query gradients with the pinned RElbers implementation."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

from diffusion_ot.losses.contrastive import detached_key_contrastive_loss
from diffusion_ot.losses.patchnce import patchnce_loss


@pytest.fixture(scope="module")
def relbers():
    path = Path(__file__).parent / "reference" / "relbers_info_nce" / "info_nce.py"
    spec = spec_from_file_location("relbers_reference", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.info_nce


def rng(seed=930):
    return torch.Generator(device="cpu").manual_seed(seed)


def assert_query_parity(actual, expected, query, reference_query):
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    gradient, = torch.autograd.grad(actual, query)
    reference_gradient, = torch.autograd.grad(expected, reference_query)
    torch.testing.assert_close(gradient, reference_gradient, atol=5e-8, rtol=2e-5)


@pytest.mark.parametrize("batch", [2, 7, 32])
@pytest.mark.parametrize("temperature", [.07, .2])
def test_global_implicit_negatives_match_relbers_values_and_gradients(relbers, batch, temperature):
    query = torch.randn(batch, 64, generator=rng(), requires_grad=True)
    keys = torch.randn(batch, 64, generator=rng(931), requires_grad=True)
    reference_query = query.detach().clone().requires_grad_()
    actual, metrics = detached_key_contrastive_loss(query, keys, keys, temperature=temperature)
    expected = relbers(reference_query, keys.detach(), temperature=temperature)
    assert metrics["mean_usable_negatives"] == batch - 1
    assert_query_parity(actual, expected, query, reference_query)
    assert keys.grad is None


@pytest.mark.parametrize("temperature", [.07, .2])
def test_global_explicit_negatives_match_relbers(relbers, temperature):
    query = torch.randn(4, 32, generator=rng(), requires_grad=True)
    positive = torch.randn(4, 32, generator=rng(931), requires_grad=True)
    bank = torch.randn(13, 32, generator=rng(932), requires_grad=True)
    reference_query = query.detach().clone().requires_grad_()
    actual, metrics = detached_key_contrastive_loss(query, positive, bank, temperature=temperature)
    expected = relbers(reference_query, positive.detach(), bank.detach(), temperature=temperature)
    assert metrics["mean_usable_negatives"] == len(bank)
    # Detachment is our caller's policy, not provided by the upstream function.
    assert torch.autograd.grad(actual, (positive, bank), allow_unused=True, retain_graph=True) == (None, None)
    assert_query_parity(actual, expected, query, reference_query)


def test_filtered_ragged_negatives_and_invalid_row_match_relbers(relbers):
    query = torch.randn(3, 4, generator=rng(), requires_grad=True)
    positive = torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0], [0., 0, 0, 0]], requires_grad=True)
    bank = torch.cat((torch.eye(4), torch.tensor([[1., 0, 0, 0], [.999, .01, 0, 0], [0., 0, 0, 0]])))
    reference_query = query.detach().clone().requires_grad_()
    actual, metrics = detached_key_contrastive_loss(query, positive, bank)
    # Explicit independent candidate lists: duplicate/near-positive and zero
    # keys are removed; the row with a zero positive contributes no gradient.
    expected = torch.stack([
        relbers(reference_query[0:1], positive[0:1].detach(), bank[[1, 2, 3]], temperature=.2),
        relbers(reference_query[1:2], positive[1:2].detach(), bank[[0, 2, 3, 4, 5]], temperature=.2),
    ]).mean()
    assert metrics["usable_samples"] == 2 and metrics["skipped_samples"] == 1
    assert metrics["mean_usable_negatives"] == 4
    assert_query_parity(actual, expected, query, reference_query)


def test_confident_global_prediction_does_not_lose_small_positive_loss(relbers):
    query = torch.tensor([[1., 0.]], requires_grad=True)
    positive = torch.tensor([[1., 0.]])
    negatives = torch.tensor([[-.6, .8]])
    actual, _ = detached_key_contrastive_loss(query, positive, negatives, temperature=.1)
    expected = relbers(query, positive, negatives, temperature=.1)
    # logsumexp([10, -6]) - 10 rounds to zero in FP32, although the
    # max-shifted cross-entropy still resolves a positive loss (~1.19e-7).
    assert actual.item() > 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=1e-6)


@pytest.mark.parametrize("temperature", [.07, .2])
@pytest.mark.parametrize("batch,shapes,patches", [
    (2, [(7, 3, 4)], 7),
    (32, [(64, 16, 16), (128, 8, 8), (256, 4, 4)], 64),
])
def test_patchnce_matches_relbers_per_image_including_current_batch_and_layers(relbers, temperature, batch, shapes, patches):
    generator = rng()
    queries = [torch.randn(batch, *shape, generator=generator, requires_grad=True) for shape in shapes]
    keys = [torch.randn(batch, *shape, generator=generator, requires_grad=True) for shape in shapes]
    reference_queries = [q.detach().clone().requires_grad_() for q in queries]
    actual, metrics = patchnce_loss(queries, keys, temperature=temperature, num_patches=patches, generator=rng(934))
    layer_losses = []
    for q, k, layer in zip(reference_queries, keys, metrics["layers"]):
        ids = layer["patch_ids"]
        q = q.permute(0, 2, 3, 1).flatten(1, 2)[:, ids]
        k = k.detach().permute(0, 2, 3, 1).flatten(1, 2)[:, ids]
        # One upstream implicit-negative problem per image, never flattening
        # B*P into a single contrast set with cross-image negatives.
        layer_losses.append(torch.stack([
            relbers(q_image, k_image, temperature=temperature)
            for q_image, k_image in zip(q, k)
        ]).mean())
    expected = torch.stack(layer_losses).mean()
    # CUT divides by norm+1e-7; RElbers clamps norm at 1e-12. Their values and
    # derivatives agree to FP32 tolerance for these nonzero feature vectors.
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.autograd.grad(actual, keys, allow_unused=True, retain_graph=True) == (None,) * len(keys)
    gradients = torch.autograd.grad(actual, queries)
    reference_gradients = torch.autograd.grad(expected, reference_queries)
    for gradient, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, reference, atol=5e-8, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_global_autocast_matches_fp32_relbers_after_key_filtering(relbers, dtype):
    query = torch.randn(5, 32, generator=rng()).to(dtype).requires_grad_()
    keys = torch.randn(5, 32, generator=rng(931)).to(dtype)
    reference_query = query.detach().float().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, metrics = detached_key_contrastive_loss(query, keys, keys)
    expected = relbers(reference_query, keys.float(), temperature=.2)
    assert actual.dtype == torch.float32
    assert metrics["mean_usable_negatives"] == 4
    torch.testing.assert_close(actual, expected)
    gradient, = torch.autograd.grad(actual, query)
    reference_gradient, = torch.autograd.grad(expected, reference_query)
    torch.testing.assert_close(gradient, reference_gradient.to(dtype), rtol=0, atol=0)
