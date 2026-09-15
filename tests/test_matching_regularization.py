from copy import deepcopy
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
import yaml

from diffusion_ot.losses.matching_regularization import (
    matching_regularization_loss, matching_regularization_options,
)
from diffusion_ot.losses.semantic_prior import validate_prior_resume


@pytest.mark.parametrize("dimension", [4, 16])
def test_isotropic_and_correlated_unit_banks_have_equal_spread_but_different_covariance(dimension):
    # Both have covariance trace 1 and scaled coordinate std 1. Only the
    # second bank repeats the same information across every coordinate.
    basis = torch.eye(dimension, dtype=torch.float64)
    isotropic = torch.cat([basis, -basis])
    vector = torch.ones(dimension, dtype=torch.float64) / math.sqrt(dimension)
    correlated = torch.stack([vector, -vector] * dimension)
    white = matching_regularization_loss({"cat": isotropic})
    redundant = matching_regularization_loss({"cat": correlated})
    for result in (white, redundant):
        assert result.metrics["cat"]["matching_variance"] == pytest.approx(1)
        assert result.metrics["variance_loss"] == 0
    assert white.loss == pytest.approx(0)
    assert redundant.metrics["covariance_loss"] == pytest.approx(1)
    assert redundant.loss == pytest.approx(.001)


def test_narrow_cone_gets_expansion_gradient_with_modest_covariance_penalty():
    # Contract all unit vectors toward the first axis while retaining small
    # differences along the others, as in the observed concentration.
    tangent = torch.cat([torch.eye(7), -torch.eye(7)]).double()
    def bank(amount):
        return F.normalize(torch.cat([torch.ones(14, 1), amount * tangent], dim=1), dim=1)
    amount = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    before = matching_regularization_loss({"cat": bank(amount)})
    gradient, = torch.autograd.grad(before.loss, amount)
    assert before.metrics["variance_loss"] > 0
    assert gradient < 0  # Gradient descent expands, not contracts, the cone.
    after = matching_regularization_loss({"cat": bank(amount.detach() - gradient)})
    assert after.loss < before.loss
    assert after.metrics["cat"]["matching_variance"] > before.metrics["cat"]["matching_variance"]


def test_normalized_features_have_correct_numerical_gradients_and_are_not_modified():
    torch.manual_seed(915)
    raw = (torch.randn(6, 4, dtype=torch.float64) * .1 + 1).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda x: matching_regularization_loss({"cat": F.normalize(x, dim=1)}).loss, (raw,), atol=1e-5,
    )
    features = F.normalize(raw, dim=1)
    saved = features.detach().clone()
    result = matching_regularization_loss({"cat": features})
    result.loss.backward()
    torch.testing.assert_close(features, saved, rtol=0, atol=0)
    assert result.metrics["cat"]["matching_variance"] == pytest.approx(float(saved.var(0, unbiased=False).sum()))
    assert torch.isfinite(raw.grad).all() and raw.grad.norm() > 0


def test_separate_domain_variance_cannot_be_satisfied_by_domain_separation():
    cat = F.normalize(torch.ones(3, 4), dim=1)
    dog = -cat[:2]
    independent = matching_regularization_loss({"cat": cat, "dog": dog})
    pooled = matching_regularization_loss({"pooled": torch.cat([cat, dog])})
    assert independent.metrics["variance_loss"] == pytest.approx(.69)
    assert pooled.metrics["variance_loss"] == 0
    # Equal domain weights, even for unequal bank sizes.
    dog = torch.cat([torch.eye(4), -torch.eye(4)])
    averaged = matching_regularization_loss({"cat": cat, "dog": dog})
    expected = (matching_regularization_loss({"cat": cat}).loss + matching_regularization_loss({"dog": dog}).loss) / 2
    torch.testing.assert_close(averaged.loss, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_exact_collapse_is_finite_but_is_not_a_recovery_mechanism(dtype):
    collapsed = torch.full((8, 4), .5, dtype=dtype, requires_grad=True)
    result = matching_regularization_loss({"cat": collapsed})
    result.loss.backward()
    assert result.loss.dtype == torch.float32
    assert float(result.loss.detach()) == pytest.approx(.02 * .69)
    assert torch.isfinite(collapsed.grad).all()
    assert collapsed.grad.count_nonzero() == 0


def test_statistics_keep_fp32_under_autocast():
    torch.manual_seed(916)
    features = F.normalize(torch.randn(10, 8), dim=1).requires_grad_()
    expected = matching_regularization_loss({"cat": features})
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = matching_regularization_loss({"cat": features})
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    actual.loss.backward()
    assert torch.isfinite(features.grad).all()


def test_protection_reaches_encoder_and_head_through_reference_samples_only():
    from diffusion_ot.models.matching_head import ResidualMatchingHead
    torch.manual_seed(917)
    encoder = torch.nn.Linear(5, 4)
    head = ResidualMatchingHead(4, 8)
    inputs = torch.randn(9, 5, requires_grad=True)
    matching = head(encoder(inputs))
    matching_regularization_loss({"cat": matching[:6]}).loss.backward()
    assert encoder.weight.grad.norm() > 0
    assert head.residual[-1].weight.grad.norm() > 0
    assert inputs.grad[:6].norm() > 0
    assert inputs.grad[6:].count_nonzero() == 0


@pytest.mark.parametrize("bank", [{}, {"cat": torch.ones(1, 4)}, {"cat": torch.ones(4, 1)}, {"cat": torch.ones(2, 2, 2)}])
def test_degenerate_bank_shape_is_rejected(bank):
    with pytest.raises(ValueError, match="Matching regularization"):
        matching_regularization_loss(bank)


@pytest.mark.parametrize("key,value", [
    ("std_target", 0), ("std_target", 1.1), ("variance_weight", 0), ("variance_weight", float("nan")),
    ("covariance_weight", -.1), ("covariance_weight", float("inf")), ("eps", 0), ("eps", .5), ("unknown", 1),
])
def test_invalid_options_rejected(key, value):
    with pytest.raises(ValueError, match="matching_regularization"):
        matching_regularization_options({"enabled": True, key: value})


def test_default_disabled_and_variance_only_ablation():
    assert matching_regularization_options({}) is None
    assert matching_regularization_options({"enabled": False}) is None
    options = matching_regularization_options({"enabled": True, "covariance_weight": 0})
    assert options == {"std_target": .7, "variance_weight": .02, "covariance_weight": 0, "eps": .0001}


def test_covariance_value_and_gradient_match_sample_covariance_reference_after_rescaling():
    # Official VICReg / Lightly use sample covariance and sum(offdiag^2)/D.
    # Our documented population + mean reduction differs by this exact factor.
    g = torch.Generator().manual_seed(76)
    x = F.normalize(torch.randn(12, 20, generator=g, dtype=torch.float64), dim=1).requires_grad_()
    result = matching_regularization_loss({"cat": x}, covariance_weight=.001)
    sample_cov = torch.cov((x * x.shape[1] ** .5).T)
    mask = ~torch.eye(x.shape[1], dtype=torch.bool)
    reference = sample_cov[mask].square().sum() / x.shape[1]
    reference *= ((len(x) - 1) / len(x)) ** 2 / (x.shape[1] - 1) * .001
    torch.testing.assert_close(result.weighted_covariance_loss, reference)
    torch.testing.assert_close(torch.autograd.grad(result.weighted_covariance_loss, x, retain_graph=True)[0],
                               torch.autograd.grad(reference, x)[0])
    torch.testing.assert_close(result.loss, result.weighted_variance_loss + result.weighted_covariance_loss)


@pytest.mark.parametrize("change", ["enable", "disable", "variance_weight", "covariance_weight", "std_target", "eps"])
def test_resume_requires_same_protection_objective(change):
    current = {"matching_regularization": {"enabled": True, **matching_regularization_options({"enabled": True})}}
    saved = deepcopy(current)
    validate_prior_resume(saved, current)
    if change == "enable":
        saved = {}
    elif change == "disable":
        current = {}
    else:
        current["matching_regularization"][change] *= .5
    with pytest.raises(ValueError, match="Resume cannot change matching_regularization"):
        validate_prior_resume(saved, current)
    validate_prior_resume({}, {})


def test_cosine_protection_control_differs_from_rms_only_in_regularizer_and_paths():
    root = Path(__file__).resolve().parents[1]
    def read(relative):
        return yaml.safe_load((root / relative).read_text())
    active = read("configs/stage1b_infoot/structure_decoder_vicreg_cosine_sit_b2.yaml")
    control = read("configs/stage1b_infoot/structure_decoder_rmsgrad_sit_b2.yaml")
    assert active["matching"]["distance_scale_gradient"] == control["matching"]["distance_scale_gradient"] == "full"
    assert matching_regularization_options(active.pop("matching_regularization")) is not None
    assert matching_regularization_options(control.get("matching_regularization", {})) is None
    assert active.pop("output_dir") != control.pop("output_dir")
    eval_active = read(active.pop("quick_evaluation")["config"])
    eval_control = read(control.pop("quick_evaluation")["config"])
    assert active == control
    assert eval_active.pop("output_dir") != eval_control.pop("output_dir")
    assert eval_active == eval_control


def test_active_relational_config_preserves_vicreg_control_except_neighborhood_and_paths():
    root = Path(__file__).resolve().parents[1]
    active = yaml.safe_load((root / "configs/stage1b_infoot/structure_decoder_sit_b2.yaml").read_text())
    control = yaml.safe_load((root / "configs/stage1b_infoot/structure_decoder_vicreg_cosine_sit_b2.yaml").read_text())
    assert active["semantic_prior"].pop("neighborhood_geometry") == "rms_distance"
    assert active.pop("output_dir") != control.pop("output_dir")
    eval_active = yaml.safe_load((root / active.pop("quick_evaluation")["config"]).read_text())
    eval_control = yaml.safe_load((root / control.pop("quick_evaluation")["config"]).read_text())
    assert active == control
    assert eval_active.pop("output_dir") != eval_control.pop("output_dir")
    assert eval_active == eval_control
