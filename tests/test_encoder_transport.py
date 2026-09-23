"""Teacher-free transport must retain query dependence and usable gradients."""
import math

import pytest
import torch

from diffusion_ot.losses.encoder_transport import (
    encoder_conditional_readout,
    encoder_transport_cost,
)
from diffusion_ot.losses.infoot import solve_infoot


def test_encoder_cost_is_bounded_normalized_and_detached():
    cat = torch.tensor([[3., 0.], [0., -4.]], requires_grad=True)
    dog = torch.tensor([[7., 0.], [-2., 0.], [0., 5.]], requires_grad=True)
    cost = encoder_transport_cost(cat, dog)
    torch.testing.assert_close(cost, torch.tensor([[0., 2., 1.], [1., 1., 2.]]))
    assert not cost.requires_grad
    assert cat.grad is None and dog.grad is None


@pytest.mark.parametrize("cat,dog,message", [
    (torch.zeros(0, 2), torch.ones(2, 2), "nonempty"),
    (torch.zeros(2, 0), torch.ones(2, 0), "nonempty"),
    (torch.zeros(2, 2), torch.ones(2, 2), "nonzero"),
    (torch.ones(2, 3), torch.ones(2, 2), "matching dimensions"),
    (torch.full((2, 2), float("nan")), torch.ones(2, 2), "finite"),
    (torch.ones(2, 2, dtype=torch.long), torch.ones(2, 2), "floating-point"),
])
def test_encoder_cost_rejects_invalid_features(cat, dog, message):
    with pytest.raises(ValueError, match=message):
        encoder_transport_cost(cat, dog)


def test_encoder_cost_breaks_stationary_independent_plan_without_external_teacher():
    cat = torch.tensor([[1., 0.], [.6, .8], [-1., 0.]])
    dog = torch.tensor([[-.8, .6], [1., 0.], [0., -1.]])
    options = dict(bandwidth=.55, mi_weight=.1, entropy_epsilon=.2,
                   inner_iterations=300, projection_iterations=2000,
                   outer_tolerance=1e-5, strict_convergence=True, require_outer_convergence=True)
    plain = solve_infoot(cat, dog, **options)
    own = solve_infoot(cat, dog, cross_cost=encoder_transport_cost(cat, dog), **options)
    torch.testing.assert_close(plain.coupling, torch.full((3, 3), 1 / 9))
    assert own.outer_converged and own.sinkhorn_converged
    assert float((own.coupling - plain.coupling).abs().sum()) > .5
    torch.testing.assert_close(own.coupling.sum(0), torch.full((3,), 1 / 3), atol=1e-5, rtol=0)
    torch.testing.assert_close(own.coupling.sum(1), torch.full((3,), 1 / 3), atol=1e-5, rtol=0)
    references = {"cat": cat, "dog": dog}
    # Separate query tensors and counts; no reference slicing is performed by the helper.
    queries = {"cat": torch.tensor([[.95, .05], [-.95, .05]]),
               "dog": torch.tensor([[-.75, .65], [.1, -.9]])}
    readout = encoder_conditional_readout(references, queries, own.coupling, bandwidth=.4)
    independent = encoder_conditional_readout(references, queries, plain.coupling, bandwidth=.4)
    for direction in ("cat_to_dog", "dog_to_cat"):
        assert readout.metrics[direction]["query_information_nats"] > .05
        assert abs(independent.metrics[direction]["query_information_nats"]) < 1e-6
        assert readout.weights[direction].shape == (2, 3)
        torch.testing.assert_close(readout.weights[direction].sum(1), torch.ones(2))


def banks():
    generator = torch.Generator().manual_seed(73)
    references = {"cat": torch.randn(4, 5, generator=generator),
                  "dog": torch.randn(5, 6, generator=generator)}
    queries = {"cat": torch.randn(3, 5, generator=generator),
               "dog": torch.randn(2, 6, generator=generator)}
    matching_refs = {d: torch.randn(len(v), 3, generator=generator, dtype=torch.float64,
                                    requires_grad=True) for d, v in references.items()}
    matching_queries = {d: torch.randn(len(v), 3, generator=generator, dtype=torch.float64,
                                      requires_grad=True) for d, v in queries.items()}
    # Balanced, rectangular, non-independent plan with positive entries.
    coupling = torch.full((4, 5), .05, dtype=torch.float64)
    coupling[:2, :2] += torch.tensor([[.04, -.04], [-.04, .04]])
    return references, queries, matching_refs, matching_queries, coupling.requires_grad_()


def test_readout_backpropagates_through_matching_features_not_plan():
    refs, queries, match_ref, match_query, coupling = banks()
    result = encoder_conditional_readout(
        refs, queries, coupling, bandwidth=.7,
        reference_matching=match_ref, query_matching=match_query,
        differentiate_distance_scale=True,
    )
    loss = sum((weights * torch.arange(weights.shape[1], dtype=weights.dtype)).sum()
               for weights in result.weights.values())
    loss.backward()
    assert coupling.grad is None
    for collection in (match_ref, match_query):
        for features in collection.values():
            assert features.grad is not None and torch.isfinite(features.grad).all()
            assert features.grad.norm() > 1e-8
    for direction, metrics in result.metrics.items():
        assert "kl" not in metrics and not any("teacher" in key for key in metrics)
        assert all(value is None or math.isfinite(value) for value in metrics.values())
        torch.testing.assert_close(result.log_weights[direction].exp(), result.weights[direction])


def test_readout_full_rms_gradients_match_finite_difference():
    refs, queries, match_ref, match_query, coupling = banks()

    def readout(cat_ref, dog_ref, cat_query, dog_query):
        result = encoder_conditional_readout(
            refs, queries, coupling, bandwidth=.7,
            reference_matching={"cat": cat_ref, "dog": dog_ref},
            query_matching={"cat": cat_query, "dog": dog_query},
            differentiate_distance_scale=True,
        )
        return result.log_weights["cat_to_dog"], result.log_weights["dog_to_cat"]

    assert torch.autograd.gradcheck(readout, (*match_ref.values(), *match_query.values()), fast_mode=True)


def test_readout_rejects_unpaired_matching_banks_and_wrong_plan_shape():
    refs, queries, match_ref, match_query, coupling = banks()
    with pytest.raises(ValueError, match="both reference and query"):
        encoder_conditional_readout(refs, queries, coupling, bandwidth=.5, reference_matching=match_ref)
    with pytest.raises(ValueError, match="coupling dimensions"):
        encoder_conditional_readout(refs, queries, coupling.T, bandwidth=.5)
    match_query["cat"] = match_query["cat"][:1]
    with pytest.raises(ValueError, match="rows must correspond"):
        encoder_conditional_readout(refs, queries, coupling, bandwidth=.5,
                                    reference_matching=match_ref, query_matching=match_query)


def test_readout_stays_finite_at_narrow_projection_bandwidth():
    refs, queries, match_ref, match_query, coupling = banks()
    result = encoder_conditional_readout(
        refs, queries, coupling, bandwidth=.01,
        reference_matching=match_ref, query_matching=match_query,
        differentiate_distance_scale=True,
    )
    assert all(torch.isfinite(value).all() for value in result.log_weights.values())
    assert all(all(value is None or math.isfinite(value) for value in metrics.values())
               for metrics in result.metrics.values())


def test_query_dependence_is_zero_for_identical_queries_and_positive_for_distinct_queries():
    refs = {d: torch.eye(4, dtype=torch.float64) for d in ("cat", "dog")}
    coupling = torch.eye(4, dtype=torch.float64) / 4
    identical = {d: value[:1].expand(4, -1) for d, value in refs.items()}
    flat = encoder_conditional_readout(refs, identical, coupling, bandwidth=.1)
    dependent = encoder_conditional_readout(refs, refs, coupling, bandwidth=.1)
    for direction in flat.metrics:
        assert flat.metrics[direction]["query_information_nats"] == pytest.approx(0, abs=1e-10)
        assert flat.metrics[direction]["mean_query_total_variation_from_marginal"] == pytest.approx(0, abs=1e-10)
        assert dependent.metrics[direction]["query_information_fraction_of_bound"] > .9
        assert dependent.metrics[direction]["mean_query_total_variation_from_marginal"] > .5
