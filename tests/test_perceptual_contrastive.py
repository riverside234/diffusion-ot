import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.training.decoded_translation import decoded_perceptual_contrastive_loss


def test_info_nce_matches_half_cls_half_spatial_logits_and_detaches_keys():
    torch.manual_seed(5)
    source = torch.randn(4, 5, 7, requires_grad=True)
    generated = (source.detach() + .2 * torch.randn_like(source)).requires_grad_()
    actual, metrics = decoded_perceptual_contrastive_loss(
        generated, source, temperature=.2, negative_similarity_threshold=1.)
    q, k = F.normalize(generated, dim=-1), F.normalize(source.detach(), dim=-1)
    scores = .5 * (q[:, 0] @ k[:, 0].T) + .5 * torch.einsum('ipd,jpd->ij', q[:, 1:], k[:, 1:]) / 4
    expected = F.cross_entropy(scores / .2, torch.arange(4))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert metrics["usable_samples"] == 4 and metrics["mean_usable_negatives"] == 3
    assert metrics["retrieval_top1"] == 1.
    actual.backward()
    assert generated.grad.norm() > 0 and source.grad is None


def test_patch_locations_matter_even_when_cls_and_patch_means_are_equal():
    source = torch.tensor([[[1., 1.], [1., 0.], [0., 1.]],
                           [[1., 1.], [0., 1.], [1., 0.]]])
    good, correct = decoded_perceptual_contrastive_loss(source, source)
    bad, swapped = decoded_perceptual_contrastive_loss(source.flip(0), source)
    assert good < bad
    assert correct["retrieval_top1"] == 1. and swapped["retrieval_top1"] == 0.


@pytest.mark.parametrize("count", [1, 3])
def test_singleton_or_duplicate_only_bank_has_no_fake_success_or_gradient(count):
    source = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]]).expand(count, -1, -1)
    query = torch.randn(count, 3, 2, requires_grad=True)
    loss, metrics = decoded_perceptual_contrastive_loss(query, source)
    assert loss == 0 and metrics["usable_samples"] == 0
    assert metrics["retrieval_top1"] is None and metrics["skipped_samples"] == count
    loss.backward()
    assert query.grad is not None and torch.count_nonzero(query.grad) == 0


def test_invalid_token_shapes_are_rejected():
    with pytest.raises(ValueError, match="CLS"):
        decoded_perceptual_contrastive_loss(torch.ones(2, 1, 3), torch.ones(2, 1, 3))
    with pytest.raises(ValueError, match="CLS"):
        decoded_perceptual_contrastive_loss(torch.ones(2, 4, 3), torch.ones(3, 4, 3))


def test_permuting_all_samples_keeps_loss_and_gradients_equivariant():
    torch.manual_seed(9)
    keys = torch.randn(4, 4, 8)
    query = torch.randn_like(keys).requires_grad_()
    loss, _ = decoded_perceptual_contrastive_loss(query, keys)
    grad, = torch.autograd.grad(loss, query)
    order = torch.tensor([2, 0, 3, 1])
    permuted = query.detach()[order].requires_grad_()
    reordered_loss, _ = decoded_perceptual_contrastive_loss(permuted, keys[order])
    reordered_grad, = torch.autograd.grad(reordered_loss, permuted)
    torch.testing.assert_close(loss, reordered_loss)
    torch.testing.assert_close(grad[order], reordered_grad)
