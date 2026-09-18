"""Generated-code InfoNCE: correct retrieval, false negatives, and routing."""
import math

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from diffusion_ot.losses.contrastive import detached_key_contrastive_loss
from diffusion_ot.training.decoded_translation import generated_code_consistency_loss


def recover(query, condition, bank, **options):
    return generated_code_consistency_loss(
        nn.Identity(), query, condition, mode="contrastive", condition_bank=bank, **options)


def test_code_infonce_matches_cross_entropy_over_full_condition_bank():
    bank = torch.eye(32, requires_grad=True)
    condition = bank[:4].detach().clone().requires_grad_()
    query = (condition.detach() + .03 * torch.randn_like(condition)).requires_grad_()
    loss, metrics = recover(query, condition, bank, temperature=.2)
    expected = F.cross_entropy(F.normalize(query, dim=-1) @ bank.detach().T / .2, torch.arange(4))
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, query, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, query)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    loss.backward()
    assert condition.grad is None and bank.grad is None
    assert metrics["condition_bank_size"] == 32
    assert metrics["mean_usable_negatives"] == 31
    assert metrics["usable_samples"] == 4
    assert metrics["retrieval_top1"] == 1
    assert metrics["negative_source"] == "current_projected_query_conditions"


def test_undecoded_conditions_supply_negatives_even_for_one_generated_image():
    positive = torch.tensor([[1., 0., 0.]])
    query = torch.tensor([[.8, .2, 0.]], requires_grad=True)
    only_self, self_metrics = recover(query, positive, positive)
    with_others, metrics = recover(query, positive, torch.eye(3))
    assert only_self == 0 and self_metrics["usable_samples"] == 0
    assert with_others > 0 and metrics["mean_usable_negatives"] == 2
    with_others.backward()
    assert query.grad.norm() > 0


def test_retrieval_prefers_intended_condition_and_not_an_unrelated_condition():
    bank = torch.eye(4)
    matched, right_metrics = recover(bank[:2] * 7, bank[:2], bank)
    mismatched, wrong_metrics = recover(bank[2:] * 7, bank[:2], bank)
    assert matched < mismatched
    assert right_metrics["retrieval_top1"] == 1 and wrong_metrics["retrieval_top1"] == 0
    assert right_metrics["positive_minus_hardest_negative_cosine"] == pytest.approx(1.)
    assert wrong_metrics["positive_minus_hardest_negative_cosine"] == pytest.approx(-1.)
    assert right_metrics["recovered_to_condition_norm_ratio"] == 7
    assert matched.item() == pytest.approx(math.log1p(3 * math.exp(-5)), abs=1e-6)


def test_code_infonce_masks_duplicate_and_near_duplicate_condition_keys():
    query = torch.tensor([[.8, .3, .1]], requires_grad=True)
    positive = torch.tensor([[1., 0, 0]])
    bank = torch.tensor([[1., 0, 0], [.999, .01, 0], [0., 1, 0], [0., 0, 0]])
    loss, metrics = recover(query, positive, bank)
    expected, _ = recover(query, positive, torch.tensor([[0., 1, 0]]))
    torch.testing.assert_close(loss, expected)
    assert metrics["mean_usable_negatives"] == 1
    assert metrics["condition_bank_size"] == 4


@pytest.mark.parametrize("bank", [torch.empty(0, 3), torch.zeros(3, 3), torch.ones(4, 3)])
def test_no_distinct_keys_produce_zero_gradient_without_fake_retrieval_success(bank):
    query = torch.randn(2, 3, requires_grad=True)
    loss, metrics = recover(query, torch.ones(2, 3), bank, negative_similarity_threshold=1.)
    assert loss.item() == 0 and metrics["usable_samples"] == 0
    assert metrics["skipped_samples"] == 2
    assert metrics["retrieval_top1"] is None and metrics["positive_probability"] is None
    loss.backward()
    torch.testing.assert_close(query.grad, torch.zeros_like(query))


def test_tied_retrieval_scores_report_chance_and_not_optimistic_first_column():
    bank = torch.eye(4)
    query = torch.ones(2, 4, requires_grad=True)
    loss, metrics = recover(query, bank[:2], bank)
    assert loss.item() == pytest.approx(math.log(4))
    assert metrics["retrieval_top1"] == .25
    assert metrics["retrieval_chance"] == .25
    assert metrics["uniform_loss"] == pytest.approx(math.log(4))
    assert metrics["positive_probability"] == pytest.approx(.25)
    assert metrics["positive_minus_hardest_negative_cosine"] == 0


def test_invalid_zero_targets_are_skipped_but_distinct_rows_still_train():
    query = torch.tensor([[.7, .3, 0], [1., 1., 1.]], requires_grad=True)
    target = torch.tensor([[1., 0, 0], [0., 0, 0]])
    loss, metrics = recover(query, target, torch.eye(3))
    assert loss > 0 and metrics["usable_samples"] == metrics["valid_conditions"] == 1
    loss.backward()
    assert query.grad[0].norm() > 0
    assert query.grad[1].norm() == 0


def test_recovery_keeps_readout_buffers_parameters_and_all_keys_fixed():
    encoder = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 3, bias=False))
    before = {name: value.clone() for name, value in encoder.state_dict().items()}
    query = torch.tensor([[1., .2, -.1], [.3, 1., .2]], requires_grad=True)
    keys = torch.eye(3, requires_grad=True)
    loss, metrics = generated_code_consistency_loss(
        encoder, query, keys[:2], mode="contrastive", condition_bank=keys)
    loss.backward()
    assert query.grad.norm() > 0 and keys.grad is None
    assert all(p.grad is None for p in encoder.parameters())
    for name, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert metrics["mode"] == "contrastive"


def test_info_nce_backward_improves_retrieval_on_fixed_keys():
    keys = torch.eye(4)
    query = nn.Parameter(keys.roll(1, dims=0).clone())
    optimizer = torch.optim.SGD([query], lr=.3)
    initial, initial_metrics = recover(query, keys, keys)
    for _ in range(20):
        optimizer.zero_grad()
        loss, _ = recover(query, keys, keys)
        loss.backward()
        optimizer.step()
    final, final_metrics = recover(query, keys, keys)
    assert final < initial * .1
    assert initial_metrics["retrieval_top1"] == 0
    assert final_metrics["retrieval_top1"] == 1


def test_contrastive_code_recovery_requires_bank_and_does_not_fall_back_to_cosine():
    with pytest.raises(ValueError, match="condition bank"):
        generated_code_consistency_loss(nn.Identity(), torch.eye(2), torch.eye(2), mode="contrastive")


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_code_contrastive_rejects_invalid_temperatures(temperature):
    with pytest.raises(ValueError, match="temperature"):
        recover(torch.eye(2), torch.eye(2), torch.eye(2), temperature=temperature)


@pytest.mark.parametrize("threshold", [-1.1, 1.1, float("nan")])
def test_code_contrastive_rejects_invalid_duplicate_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        recover(torch.eye(2), torch.eye(2), torch.eye(2), negative_similarity_threshold=threshold)


def test_info_nce_is_finite_for_large_half_precision_codes_and_low_temperature():
    query = (torch.eye(3) * 50000).half().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = detached_key_contrastive_loss(query, torch.eye(3), torch.eye(3), temperature=1e-3)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(query.grad).all()


def test_nonfinite_negative_keys_fail_loudly():
    with pytest.raises(FloatingPointError, match="Non-finite"):
        recover(torch.eye(2), torch.eye(2), torch.full((3, 2), float("nan")))
