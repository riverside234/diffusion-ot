import pytest
import torch

from diffusion_ot.training.gradient_diagnostics import (
    GradientConflictMonitor, gradient_pair_metrics, training_gradient_conflicts,
)


@pytest.mark.parametrize("a,b,cosine,descent", [
    ([1., 0.], [2., 0.], 1., 3.),
    ([1., 0.], [0., 2.], 0., 1.),
    ([1., 0.], [-2., 0.], -1., -1.),
])
def test_pair_geometry_and_pair_only_descent(a, b, cosine, descent):
    first, second = torch.tensor(a), torch.tensor(b)
    result = gradient_pair_metrics([first], [second])
    assert result["valid"]
    assert result["cosine"] == pytest.approx(cosine)
    assert result["conflict"] == (cosine < 0)
    assert result["first_pair_descent_fraction"] == pytest.approx(descent)
    assert result["projection_removed_norm_fraction"] == pytest.approx(max(0., -cosine))
    torch.testing.assert_close(first, torch.tensor(a))


def test_unused_and_zero_gradients_are_not_reported_as_aligned():
    result = gradient_pair_metrics([torch.ones(2), None], [None, torch.ones(2)])
    assert result["cosine"] == 0.
    empty = gradient_pair_metrics([None], [torch.ones(1)])
    assert not empty["valid"] and empty["cosine"] is empty["conflict"] is None
    monitor = GradientConflictMonitor(window_size=2)
    opposing = gradient_pair_metrics([torch.ones(1)], [-torch.ones(1)])
    monitor.record("encoder", "pair", opposing)
    row = monitor.record("encoder", "pair", empty)
    assert row["window_probes"] == 2 and row["window_valid_pairs"] == 1
    assert row["window_conflict_rate"] == 1.
    aligned = gradient_pair_metrics([torch.ones(1)], [torch.ones(1)])
    row = monitor.record("encoder", "pair", aligned)
    assert row["window_conflict_rate"] == 0.
    monitor.set_phase("full_weight")
    row = monitor.record("encoder", "pair", opposing)
    assert row["window_probes"] == row["window_valid_pairs"] == 1
    assert row["window_conflict_rate"] == 1.


def test_routing_scaling_domain_conflicts_and_read_only_autograd():
    cat, dog, gen = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(3)]
    reconstruction = cat + dog + gen
    perceptual = -cat + 3 * dog + gen
    adversarial = 2 * cat - dog - gen
    decoded = perceptual + adversarial
    losses = {"reconstruction": reconstruction, "decoded": decoded,
              "perceptual": perceptual, "adversarial": adversarial,
              "matching": cat - dog}
    cat.grad = torch.tensor(17.)
    rng = torch.get_rng_state().clone()
    report, norms = training_gradient_conflicts(
        losses, {"encoder.cat": [cat], "encoder.dog": [dog], "generator.dog": [gen]},
        code_gradients={id(gen): torch.tensor(-3.)}, encoder_scale=2., monitor=GradientConflictMonitor())
    assert cat.grad == 17. and dog.grad is None and gen.grad is None
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert norms["decoded"]["encoder.all"] == pytest.approx(5 ** .5)
    groups = report["groups"]
    assert groups["encoder.cat"]["perceptual_vs_adversarial"]["cosine"] == -1.
    assert groups["encoder.cat"]["translation_vs_reconstruction"]["first_to_second_norm_ratio"] == 2.
    assert groups["encoder.dog"]["translation_vs_reconstruction"]["first_to_second_norm_ratio"] == 4.
    # Code contributes to the generator translation, and never to encoders.
    pair = groups["generator.dog"]["translation_vs_reconstruction"]
    assert pair["cosine"] == -1. and pair["first_norm"] == 3.
    # The original graph remains usable, including the gradients at conflict.
    grads = torch.autograd.grad(reconstruction + decoded, (cat, dog, gen))
    assert [float(g) for g in grads] == [2., 3., 1.]
