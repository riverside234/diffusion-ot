"""Execute the pinned official InfoOT implementation, including projection.

These complement the independent formulas in test_infoot_reference_audit.py.
The reference remains test-only; no training objective or weights are changed.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.losses.encoder_transport import encoder_conditional_readout, encoder_transport_cost
from diffusion_ot.losses.infoot import (
    conditional_density_ratio, conditional_reference_log_weights,
    conditional_reference_weights, gaussian_kernel, infoot_cross_distance_scale,
    infoot_distance_scale, infoot_mutual_information, infoot_plan_gradient,
    plain_infoot_feature_loss, solve_infoot, weighted_target_codes,
)


@pytest.fixture(scope="module")
def upstream():
    for name in ("ot", "sklearn", "scipy", "tqdm"):
        pytest.importorskip(name)
    path = Path(__file__).parent / "reference" / "chingyaoc_infoot" / "infoot.py"
    source = path.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(source).hexdigest() == "0d636765545256a0062ec697cc2e88ca17cc9ca40f6e185999eaf1b93ff2253b"
    spec = importlib.util.spec_from_file_location("_official_infoot_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _banks():
    rng = np.random.default_rng(20260925)
    return [F.normalize(torch.from_numpy(rng.normal(size=shape)), dim=1)
            for shape in ((5, 4), (7, 4), (3, 4), (2, 4))]


def _fitted(upstream, x, y):
    fitted = upstream.FusedInfoOT(x.numpy(), y.numpy(), h=.55, lam=.1, reg=.4)
    fitted.solve(numIter=6, verbose=False)
    return fitted


def test_live_upstream_kernels_density_ratio_mi_and_complete_gradient(upstream):
    x, y, _, _ = _banks()
    fitted = _fitted(upstream, x, y)
    plan = torch.from_numpy(fitted.P)
    kx = gaussian_kernel(x, bandwidth=.55, distance_scale=infoot_distance_scale(x))
    ky = gaussian_kernel(y, bandwidth=.55, distance_scale=infoot_distance_scale(y))
    torch.testing.assert_close(kx, torch.from_numpy(fitted.Ks), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ky, torch.from_numpy(fitted.Kt), rtol=1e-12, atol=1e-12)
    expected_ratio = upstream.ratio(fitted.P, fitted.Ks, fitted.Kt)
    actual_ratio = conditional_density_ratio(x, y, x, y, plan, bandwidth=.55,
        distance_scale_x=infoot_distance_scale(x), distance_scale_y=infoot_distance_scale(y))
    torch.testing.assert_close(actual_ratio, torch.from_numpy(expected_ratio), rtol=1e-12, atol=1e-12)
    assert float(infoot_mutual_information(plan, kx, ky)) == pytest.approx(
        np.sum(fitted.P * np.log(expected_ratio)), abs=1e-12)
    torch.testing.assert_close(infoot_plan_gradient(plan, kx, ky),
        torch.from_numpy(-upstream.migrad(fitted.P, fitted.Ks, fitted.Kt)), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("iterations", [1, 6])
@pytest.mark.parametrize("cost_kind", ["euclidean", "encoder_cosine"])
def test_multiple_fused_updates_match_actual_official_solver(upstream, iterations, cost_kind):
    x, y, _, _ = _banks()
    fitted = upstream.FusedInfoOT(x.numpy(), y.numpy(), h=.55, lam=.1, reg=.4)
    if cost_kind == "encoder_cosine":
        # Hold the prescribed cost equal while keeping the official algorithm intact.
        fitted.C = encoder_transport_cost(x, y).numpy()
    expected = fitted.solve(numIter=iterations, verbose=False)
    actual = solve_infoot(x, y, cross_cost=torch.from_numpy(fitted.C), bandwidth=.55,
        distance_scale_x=infoot_distance_scale(x), distance_scale_y=infoot_distance_scale(y),
        mi_weight=.1, entropy_epsilon=.4, inner_iterations=iterations,
        projection_iterations=10000, projection_tolerance=1e-12)
    torch.testing.assert_close(actual.coupling, torch.from_numpy(expected), rtol=1e-8, atol=2e-10)
    assert actual.sinkhorn_converged


def test_plain_infoot_matches_and_irregular_kde_can_leave_independent_initialization(upstream):
    x, y, _, _ = _banks()
    y = y[:, :3]  # Plain InfoOT also supports unrelated coordinate dimensions.
    fitted = upstream.InfoOT(x.numpy(), y.numpy(), h=.55, reg=.4)
    expected = fitted.solve(numIter=6, verbose=False)
    actual = solve_infoot(x, y, bandwidth=.55, mi_weight=1., entropy_epsilon=.4,
        distance_scale_x=infoot_distance_scale(x), distance_scale_y=infoot_distance_scale(y),
        inner_iterations=6, projection_iterations=10000, projection_tolerance=1e-12)
    torch.testing.assert_close(actual.coupling, torch.from_numpy(expected), rtol=1e-12, atol=1e-12)
    # The finite KDE gradient need not be constant at independence. In
    # particular, uneven kernel row sums can give a nonseparable derivative.
    assert (actual.coupling - 1 / 35).abs().max() > 1e-3


def test_plain_infoot_can_remain_independent_for_symmetric_kernels(upstream):
    x, y = torch.eye(4, dtype=torch.float64), torch.eye(5, dtype=torch.float64)
    reference = upstream.InfoOT(x.numpy(), y.numpy(), h=.55, reg=.4)
    expected = reference.solve(numIter=6, verbose=False)
    actual = solve_infoot(x, y, bandwidth=.55, mi_weight=1., entropy_epsilon=.4,
        distance_scale_x=infoot_distance_scale(x), distance_scale_y=infoot_distance_scale(y),
        inner_iterations=6, projection_iterations=10000, projection_tolerance=1e-12)
    torch.testing.assert_close(actual.coupling, torch.from_numpy(expected), rtol=0, atol=1e-12)
    torch.testing.assert_close(actual.coupling, torch.full_like(actual.coupling, 1 / 20), rtol=0, atol=1e-12)


@pytest.mark.parametrize("bandwidth", [.55, .1])
@pytest.mark.parametrize("reverse", [False, True])
def test_in_sample_and_unseen_projection_match_official_readouts(upstream, bandwidth, reverse):
    x, y, qx, qy = _banks()
    fitted = _fitted(upstream, x, y)
    if reverse:
        fitted = upstream.FusedInfoOT(y.numpy(), x.numpy(), h=.55, lam=.1, reg=.4)
        fitted.P = _fitted(upstream, x, y).P.T
        x, y, qx = y, x, qy
    plan = torch.from_numpy(fitted.P)
    expected_barycentric = fitted.project(x.numpy(), method="barycentric")
    torch.testing.assert_close(weighted_target_codes(plan, y), torch.from_numpy(expected_barycentric))
    for query in (x, qx):
        settings = dict(bandwidth=bandwidth,
            distance_scale_x=infoot_cross_distance_scale(query, x),
            distance_scale_y=infoot_distance_scale(y))
        scores = fitted.conditional_score(query.numpy(), h=bandwidth)
        expected_weights = scores / scores.sum(1, keepdims=True)
        expected_codes = upstream.projection(scores, y.numpy())
        actual = conditional_reference_weights(query, x, y, plan, **settings)
        log_actual = conditional_reference_log_weights(query, x, y, plan, **settings)
        torch.testing.assert_close(actual, torch.from_numpy(expected_weights), rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(log_actual.exp(), actual, rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(weighted_target_codes(actual, y), torch.from_numpy(expected_codes),
                                   rtol=1e-9, atol=1e-12)
        if query is x:
            torch.testing.assert_close(weighted_target_codes(actual, y),
                torch.from_numpy(fitted.project(x.numpy(), method="conditional", h=bandwidth)),
                rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("bandwidth", [.55, .1])
def test_actual_bidirectional_training_readout_uses_matching_keys_and_raw_values(upstream, bandwidth):
    x, y, qx, qy = _banks()
    fitted = _fitted(upstream, x, y)
    plan = torch.from_numpy(fitted.P).requires_grad_()
    generator = torch.Generator().manual_seed(53)
    raw_refs = {d: torch.randn(n, 9, dtype=torch.float64, generator=generator, requires_grad=True)
                for d, n in (("cat", len(x)), ("dog", len(y)))}
    raw_queries = {d: torch.randn(n, 9, dtype=torch.float64, generator=generator)
                   for d, n in (("cat", len(qx)), ("dog", len(qy)))}
    match_refs = {"cat": x.clone().requires_grad_(), "dog": y.clone().requires_grad_()}
    match_queries = {"cat": qx.clone().requires_grad_(), "dog": qy.clone().requires_grad_()}
    result = encoder_conditional_readout(raw_refs, raw_queries, plan, bandwidth=bandwidth,
        reference_matching=match_refs, query_matching=match_queries, differentiate_distance_scale=True)
    objectives = []
    for source, target in (("cat", "dog"), ("dog", "cat")):
        reference = upstream.FusedInfoOT(match_refs[source].detach().numpy(),
            match_refs[target].detach().numpy(), h=.55)
        reference.P = fitted.P if source == "cat" else fitted.P.T
        scores = reference.conditional_score(match_queries[source].detach().numpy(), h=bandwidth)
        expected = upstream.projection(scores, raw_refs[target].detach().numpy())
        actual = result.weights[f"{source}_to_{target}"] @ raw_refs[target]
        torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=1e-9, atol=1e-12)
        objectives.append(actual.square().mean())
    sum(objectives).backward()
    assert plan.grad is None
    for collection in (match_refs, match_queries, raw_refs):
        for value in collection.values():
            assert value.grad is not None and torch.isfinite(value.grad).all()
            assert value.grad.norm() > 1e-12


@pytest.mark.parametrize("domain", ["cat", "dog"])
def test_neural_mi_rms_gradient_matches_finite_difference_of_official_density(upstream, domain):
    x, y, _, _ = _banks()
    fitted = _fitted(upstream, x, y)
    plan = torch.from_numpy(fitted.P)
    x, y = x.requires_grad_(), y.requires_grad_()
    loss = plain_infoot_feature_loss(x, y, plan, bandwidth=.55, mi_weight=.1,
        distance_scale_x=infoot_distance_scale(x, detach=False),
        distance_scale_y=infoot_distance_scale(y, detach=False))
    value = x if domain == "cat" else y
    gradient, = torch.autograd.grad(loss, value)
    direction = torch.randn(value.shape, generator=torch.Generator().manual_seed(34), dtype=value.dtype)

    def reference_loss(shift):
        nx, ny = x.detach().numpy().copy(), y.detach().numpy().copy()
        if domain == "cat":
            nx += shift * direction.numpy()
        else:
            ny += shift * direction.numpy()
        reference = upstream.FusedInfoOT(nx, ny, h=.55)
        return -.1 * np.sum(fitted.P * np.log(upstream.ratio(fitted.P, reference.Ks, reference.Kt)))

    delta = 1e-5
    numerical = (reference_loss(delta) - reference_loss(-delta)) / (2 * delta)
    assert float((gradient * direction).sum()) == pytest.approx(numerical, rel=2e-6, abs=1e-9)


@pytest.mark.parametrize("class_name", ["InfoOT", "FusedInfoOT"])
def test_record_upstream_unseen_project_bug_but_working_score_path(upstream, class_name):
    x, y, qx, _ = _banks()
    reference = getattr(upstream, class_name)(x.numpy(), y.numpy(), h=.55)
    reference.P = np.full((len(x), len(y)), 1 / (len(x) * len(y)))
    with pytest.raises(NameError, match="Xs"):
        reference.project(qx.numpy(), method="conditional")
    assert np.isfinite(upstream.projection(reference.conditional_score(qx.numpy()), y.numpy())).all()


def test_query_batch_bandwidth_dependence_is_inherited_from_upstream(upstream):
    x = F.normalize(torch.tensor([[1., 0.], [1., .1], [1., -.1], [1., .3]], dtype=torch.float64), dim=1)
    y = torch.eye(4, dtype=torch.float64)
    queries = F.normalize(torch.tensor([[1., .15], [-1., 0.]], dtype=torch.float64), dim=1)
    plan = y / 4
    reference = upstream.InfoOT(x.numpy(), y.numpy(), h=.55)
    reference.P = plan.numpy()
    adaptive, fixed = [], []
    for query in (queries[:1], queries):
        expected = reference.conditional_score(query.numpy(), h=.1)
        expected /= expected.sum(1, keepdims=True)
        adaptive.append(conditional_reference_log_weights(query, x, y, plan, bandwidth=.1,
            distance_scale_x=infoot_cross_distance_scale(query, x),
            distance_scale_y=infoot_distance_scale(y)).exp()[0])
        torch.testing.assert_close(adaptive[-1], torch.from_numpy(expected[0]), rtol=1e-9, atol=1e-12)
        fixed.append(conditional_reference_log_weights(query, x, y, plan, bandwidth=.1,
            distance_scale_x=infoot_distance_scale(x), distance_scale_y=infoot_distance_scale(y)).exp()[0])
    # Query-batch-dependent RMS is mathematically upstream-compatible, but a
    # future fixed-reference-scale experiment can provide a pointwise mapping.
    assert (adaptive[0] - adaptive[1]).abs().sum() > .9
    torch.testing.assert_close(fixed[0], fixed[1], rtol=0, atol=1e-12)
