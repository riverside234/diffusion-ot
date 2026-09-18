from copy import deepcopy
import math

import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.losses.contrastive import (
    matching_contrastive_loss, matching_contrastive_options, matching_neighborhood_contrastive_loss,
    teacher_multi_positive_contrastive_loss,
)


def test_mass_infonce_value_and_gradient_push_positive_up_negative_down():
    logits = torch.tensor([[.3, -.2, .1, -.8]], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([[.6, .3, .09, .01]], dtype=torch.float64, requires_grad=True)
    before = logits.detach().clone()
    loss, metrics = teacher_multi_positive_contrastive_loss(logits, teacher, positive_count=2)
    expected = -logits.softmax(1)[0, :2].sum().log()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (logits.grad[0, :2] < 0).all()
    assert (logits.grad[0, 2:] > 0).all()
    assert teacher.grad is None
    torch.testing.assert_close(logits, before)
    assert metrics["usable_rows"] == 1
    assert metrics["teacher_positive_mass"] == pytest.approx(.9)


def test_self_pairs_never_appear_as_positive_or_negative_and_have_zero_gradient():
    logits = torch.tensor([[100., 2., 0.], [0., 100., 2.], [2., 0., 100.]], requires_grad=True)
    teacher = torch.tensor([[.9, .09, .01], [.01, .9, .09], [.09, .01, .9]])
    loss, metrics = teacher_multi_positive_contrastive_loss(logits, teacher, positive_count=1, exclude_diagonal=True)
    assert float(loss.detach()) == pytest.approx(math.log1p(math.exp(-2)))
    loss.backward()
    assert logits.grad.diagonal().count_nonzero() == 0
    assert metrics["valid_candidates"] == 2
    assert metrics["mean_positive_count"] == 1


def test_structural_ties_are_all_positive_and_no_arbitrary_negatives_are_created():
    logits = torch.zeros(2, 4, requires_grad=True)
    teacher = torch.tensor([[.4, .4, .1, .1], [.25, .25, .25, .25]])
    loss, metrics = teacher_multi_positive_contrastive_loss(logits, teacher, positive_count=1)
    assert float(loss.detach()) == pytest.approx(math.log(2) / 2)
    assert metrics["usable_rows"] == 1
    assert metrics["fraction_usable_rows"] == .5
    assert metrics["mean_positive_count"] == 3
    loss.backward()
    torch.testing.assert_close(logits.grad[0, 0], logits.grad[0, 1])
    assert logits.grad[1].count_nonzero() == 0


def test_all_tied_batch_has_differentiable_zero_loss():
    logits = torch.randn(3, 5, requires_grad=True)
    loss, metrics = teacher_multi_positive_contrastive_loss(logits, torch.ones_like(logits), positive_count=2)
    assert loss.requires_grad
    assert float(loss.detach()) == 0
    assert metrics["usable_rows"] == 0
    loss.backward()
    assert logits.grad.count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_low_precision_accumulates_in_fp32_and_large_logits_remain_finite(dtype):
    logits = torch.tensor([[1000., -1000., 0.]], dtype=dtype, requires_grad=True)
    teacher = torch.tensor([[.1, .8, .1]], dtype=dtype)
    loss, _ = teacher_multi_positive_contrastive_loss(logits, teacher, positive_count=1)
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert float(loss.detach()) == pytest.approx(2000)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_log_probabilities_keep_full_support_and_supplement_existing_kl():
    logits = torch.tensor([[.1, .3, -.7, .4]], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([[.6, .3, .09, .01]], dtype=torch.float64)
    log_weights = logits.log_softmax(1)
    weights_before = log_weights.exp().detach().clone()
    kl = F.kl_div(log_weights, teacher, reduction="batchmean")
    additional, _ = teacher_multi_positive_contrastive_loss(log_weights, teacher, positive_count=2)
    weighted = kl + .01 * additional
    expected = kl - .01 * log_weights.exp()[0, :2].sum().log()
    torch.testing.assert_close(weighted, expected)
    weighted.backward()
    assert (logits.grad != 0).all()
    torch.testing.assert_close(log_weights.exp(), weights_before)
    assert weights_before.count_nonzero() == 4


def test_generic_and_neighborhood_gradcheck_and_teacher_stop_gradient():
    torch.manual_seed(913)
    logits = torch.randn(4, 7, dtype=torch.float64, requires_grad=True)
    teacher = torch.rand_like(logits)
    assert torch.autograd.gradcheck(
        lambda x: teacher_multi_positive_contrastive_loss(x, teacher, positive_count=2)[0], (logits,),
    )
    features = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    structure = torch.randn(7, 9, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: matching_neighborhood_contrastive_loss(x, structure, positive_count=2)[0], (features,),
    )
    loss, _ = matching_neighborhood_contrastive_loss(features, structure, positive_count=2)
    loss.backward()
    assert features.grad.norm() > 0
    assert structure.grad is None


def test_neighborhood_is_invariant_to_independent_descriptor_rotation_and_feature_scale():
    torch.manual_seed(916)
    features = torch.randn(8, 5)
    structure = torch.randn(8, 3)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    first, _ = matching_neighborhood_contrastive_loss(features, structure, positive_count=2)
    second, _ = matching_neighborhood_contrastive_loss(5 * features, structure @ rotation, positive_count=2)
    torch.testing.assert_close(first, second)


def test_neighborhood_keeps_fp32_under_autocast():
    torch.manual_seed(917)
    features = torch.randn(8, 5, requires_grad=True)
    structure = torch.randn(8, 3)
    expected, _ = matching_neighborhood_contrastive_loss(features, structure, positive_count=2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, _ = matching_neighborhood_contrastive_loss(features, structure, positive_count=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("value", [0, -1, 1.5, True, 4])
def test_invalid_positive_counts(value):
    with pytest.raises(ValueError, match="positive_count"):
        teacher_multi_positive_contrastive_loss(torch.zeros(2, 4), torch.ones(2, 4), positive_count=value)


@pytest.mark.parametrize("student,teacher,kwargs", [
    (torch.zeros(4), torch.ones(4), {}),
    (torch.zeros(2, 4), torch.ones(3, 4), {}),
    (torch.zeros(0, 4), torch.ones(0, 4), {}),
    (torch.zeros(3, 1), torch.ones(3, 1), {}),
    (torch.zeros(2, 4), torch.ones(2, 4), {"exclude_diagonal": True}),
    (torch.zeros(2, 2), torch.ones(2, 2), {"exclude_diagonal": True}),
    (torch.zeros(2, 4), -torch.ones(2, 4), {}),
    (torch.zeros(2, 4), torch.zeros(2, 4), {}),
    (torch.zeros(3, 3), torch.eye(3), {"exclude_diagonal": True}),
    (torch.full((2, 4), float("nan")), torch.ones(2, 4), {}),
    (torch.zeros(2, 4), torch.full((2, 4), float("inf")), {}),
    (torch.zeros(2, 4, dtype=torch.int64), torch.ones(2, 4), {}),
])
def test_invalid_inputs(student, teacher, kwargs):
    with pytest.raises(ValueError):
        teacher_multi_positive_contrastive_loss(student, teacher, positive_count=1, **kwargs)


@pytest.mark.parametrize("overrides", [
    {"neighborhood_weight": -1}, {"conditional_weight": float("nan")},
    {"neighborhood_weight": 0, "conditional_weight": 0},
    {"temperature": 0}, {"temperature": float("inf")},
    {"positive_count": 1.5}, {"positive_count": True}, {"bad_option": 1},
])
def test_invalid_enabled_options(overrides):
    with pytest.raises(ValueError):
        matching_contrastive_options({"enabled": True, **overrides})


def test_options_are_opt_in_and_do_not_modify_config():
    assert matching_contrastive_options({}) is None
    assert matching_contrastive_options({"enabled": False}) is None
    config = {"enabled": True, "neighborhood_weight": .02, "conditional_weight": .01,
              "positive_count": 3, "temperature": .15}
    before = deepcopy(config)
    resolved = matching_contrastive_options(config)
    assert resolved == {key: value for key, value in config.items() if key != "enabled"}
    assert config == before


def test_combined_loss_means_each_objective_and_retains_independent_domain_gradients():
    torch.manual_seed(919)
    features = {"cat": torch.randn(6, 4, requires_grad=True), "dog": torch.randn(7, 4, requires_grad=True)}
    structure = {domain: torch.randn(len(values), 9, requires_grad=True) for domain, values in features.items()}
    logits = {"cat_to_dog": torch.randn(3, 7, requires_grad=True), "dog_to_cat": torch.randn(3, 6, requires_grad=True)}
    teacher = {direction: torch.rand_like(values, requires_grad=True) for direction, values in logits.items()}
    log_weights = {direction: values.log_softmax(1) for direction, values in logits.items()}
    loss, metrics = matching_contrastive_loss(
        features, structure, log_weights, teacher,
        neighborhood_weight=.01, conditional_weight=.005, positive_count=2, temperature=.2,
    )
    expected_neighborhood = torch.stack([
        matching_neighborhood_contrastive_loss(features[d], structure[d], positive_count=2)[0]
        for d in features
    ]).mean()
    expected_conditional = torch.stack([
        teacher_multi_positive_contrastive_loss(log_weights[d], teacher[d], positive_count=2)[0]
        for d in logits
    ]).mean()
    torch.testing.assert_close(loss, .01 * expected_neighborhood + .005 * expected_conditional)
    assert metrics["neighborhood_loss"] == pytest.approx(float(expected_neighborhood.detach()))
    assert metrics["conditional_loss"] == pytest.approx(float(expected_conditional.detach()))
    assert metrics["weighted_loss"] == pytest.approx(float(loss.detach()))
    loss.backward()
    assert all(v.grad is not None and v.grad.norm() > 0 for v in (*features.values(), *logits.values()))
    assert all(v.grad is None for v in (*structure.values(), *teacher.values()))


@pytest.mark.parametrize("neighborhood_weight,conditional_weight", [(0, .005), (.01, 0)])
def test_zero_weight_branch_can_be_absent(neighborhood_weight, conditional_weight):
    features = {"cat": torch.randn(6, 4, requires_grad=True)} if neighborhood_weight else {}
    structure = {"cat": torch.randn(6, 3)} if neighborhood_weight else {}
    logits = {"cat_to_dog": torch.randn(3, 6, requires_grad=True)} if conditional_weight else {}
    teacher = {"cat_to_dog": torch.rand(3, 6)} if conditional_weight else {}
    loss, metrics = matching_contrastive_loss(
        features, structure, logits, teacher, neighborhood_weight=neighborhood_weight,
        conditional_weight=conditional_weight, positive_count=2, temperature=.2,
    )
    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert metrics["neighborhood_loss" if not neighborhood_weight else "conditional_loss"] == 0


def test_active_reference_and_query_sizes_with_eight_positives():
    torch.manual_seed(920)
    features = {domain: F.normalize(torch.randn(96, 512), dim=1).requires_grad_() for domain in ("cat", "dog")}
    structure = {domain: torch.randn(96, 240) for domain in features}
    logits = {direction: torch.randn(32, 96, requires_grad=True) for direction in ("cat_to_dog", "dog_to_cat")}
    teacher = {direction: torch.rand_like(values) for direction, values in logits.items()}
    loss, metrics = matching_contrastive_loss(
        features, structure, logits, teacher, neighborhood_weight=.01, conditional_weight=.005,
        positive_count=8, temperature=.2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["neighborhood"]["cat"]["valid_candidates"] == 95
    assert metrics["conditional"]["cat_to_dog"]["valid_candidates"] == 96
    assert metrics["conditional"]["cat_to_dog"]["usable_rows"] == 32
    with pytest.raises(ValueError, match="positive_count"):
        matching_neighborhood_contrastive_loss(torch.randn(9, 4), torch.randn(9, 3), positive_count=8)


@pytest.mark.parametrize("neighborhood_weight,conditional_weight", [(0, 0), (-1, .01), (.01, float("nan"))])
def test_combined_rejects_invalid_weights(neighborhood_weight, conditional_weight):
    with pytest.raises(ValueError, match="weight"):
        matching_contrastive_loss({}, {}, {}, {}, neighborhood_weight=neighborhood_weight,
                                 conditional_weight=conditional_weight, positive_count=2, temperature=.2)


def test_combined_rejects_mismatched_domain_or_direction_banks():
    with pytest.raises(ValueError, match="domain banks"):
        matching_contrastive_loss({"cat": torch.randn(6, 4)}, {}, {}, {}, neighborhood_weight=.01,
                                 conditional_weight=0, positive_count=2, temperature=.2)
    with pytest.raises(ValueError, match="direction banks"):
        matching_contrastive_loss({}, {}, {"cat_to_dog": torch.randn(3, 6)}, {}, neighborhood_weight=0,
                                 conditional_weight=.005, positive_count=2, temperature=.2)
