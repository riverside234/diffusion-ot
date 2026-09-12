from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def test_sinkhorn_projection_supports_unequal_batches():
    from diffusion_ot.losses.infoot import sinkhorn_project, uniform_marginals

    a, b = uniform_marginals(3, 5, dtype=torch.float64)
    matrix = torch.rand(3, 5, dtype=torch.float64)
    coupling = sinkhorn_project(matrix, a, b, max_iterations=1000, tolerance=1.0e-10)

    assert torch.all(coupling >= 0)
    torch.testing.assert_close(coupling.sum(dim=1), a, atol=1.0e-9, rtol=0)
    torch.testing.assert_close(coupling.sum(dim=0), b, atol=1.0e-9, rtol=0)


def test_infoot_distance_scale_matches_official_compute_kernel():
    from diffusion_ot.losses.infoot import (
        infoot_cross_distance_scale,
        infoot_distance_scale,
    )

    features = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]], dtype=torch.float64
    )
    distances = torch.cdist(features, features)
    expected = ((distances.square().mean() / 2.0).sqrt())

    torch.testing.assert_close(infoot_distance_scale(features), expected)
    query = features[:2] + 0.25
    cross_distances = torch.cdist(query, features)
    cross_expected = (cross_distances.square().mean() / 2.0).sqrt()
    torch.testing.assert_close(
        infoot_cross_distance_scale(query, features), cross_expected
    )


def test_complete_plan_gradient_matches_autograd():
    from diffusion_ot.losses.infoot import (
        gaussian_kernel,
        infoot_mutual_information,
        infoot_plan_gradient,
        uniform_marginals,
    )

    torch.manual_seed(3)
    coupling = torch.rand(3, 4, dtype=torch.float64, requires_grad=True)
    a, b = uniform_marginals(3, 4, dtype=torch.float64)
    kernel_x = gaussian_kernel(torch.randn(3, 2, dtype=torch.float64), bandwidth=0.8)
    kernel_y = gaussian_kernel(torch.randn(4, 2, dtype=torch.float64), bandwidth=1.1)
    value = infoot_mutual_information(coupling, kernel_x, kernel_y, a, b)
    (autograd_gradient,) = torch.autograd.grad(value, coupling)
    analytic_gradient = infoot_plan_gradient(coupling.detach(), kernel_x, kernel_y, a, b)

    torch.testing.assert_close(analytic_gradient, autograd_gradient, atol=1.0e-9, rtol=1.0e-8)


def test_complete_plan_gradient_matches_finite_difference():
    from diffusion_ot.losses.infoot import (
        gaussian_kernel,
        infoot_mutual_information,
        infoot_plan_gradient,
        uniform_marginals,
    )

    torch.manual_seed(4)
    coupling = torch.rand(3, 3, dtype=torch.float64)
    a, b = uniform_marginals(3, 3, dtype=torch.float64)
    kernel_x = gaussian_kernel(torch.randn(3, 2, dtype=torch.float64))
    kernel_y = gaussian_kernel(torch.randn(3, 2, dtype=torch.float64))
    analytic = infoot_plan_gradient(coupling, kernel_x, kernel_y, a, b)
    row, column = 1, 2
    delta = 1.0e-6
    plus = coupling.clone()
    minus = coupling.clone()
    plus[row, column] += delta
    minus[row, column] -= delta
    numerical = (
        infoot_mutual_information(plus, kernel_x, kernel_y, a, b)
        - infoot_mutual_information(minus, kernel_x, kernel_y, a, b)
    ) / (2.0 * delta)

    assert float(analytic[row, column]) == pytest.approx(float(numerical), rel=1.0e-5, abs=1.0e-6)


def test_plain_infoot_is_invariant_to_independent_rotations():
    from diffusion_ot.losses.infoot import (
        gaussian_kernel,
        infoot_mutual_information,
        seeded_nonindependent_coupling,
        uniform_marginals,
    )

    torch.manual_seed(5)
    x = torch.randn(6, 4, dtype=torch.float64)
    y = torch.randn(7, 4, dtype=torch.float64)
    qx, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
    qy, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
    a, b = uniform_marginals(6, 7, dtype=torch.float64)
    coupling = seeded_nonindependent_coupling(a, b, seed=9)

    original = infoot_mutual_information(
        coupling, gaussian_kernel(x), gaussian_kernel(y), a, b
    )
    rotated = infoot_mutual_information(
        coupling, gaussian_kernel(x @ qx), gaussian_kernel(y @ qy), a, b
    )
    torch.testing.assert_close(original, rotated, atol=1.0e-10, rtol=1.0e-9)


def test_joint_sample_and_plan_permutation_preserves_infoot_value():
    from diffusion_ot.losses.infoot import (
        gaussian_kernel,
        infoot_mutual_information,
        seeded_nonindependent_coupling,
        uniform_marginals,
    )

    torch.manual_seed(6)
    x = torch.randn(5, 3, dtype=torch.float64)
    y = torch.randn(6, 3, dtype=torch.float64)
    a, b = uniform_marginals(5, 6, dtype=torch.float64)
    coupling = seeded_nonindependent_coupling(a, b, seed=8)
    px = torch.tensor([3, 0, 4, 1, 2])
    py = torch.tensor([2, 5, 0, 4, 1, 3])
    original = infoot_mutual_information(
        coupling, gaussian_kernel(x), gaussian_kernel(y), a, b
    )
    permuted = infoot_mutual_information(
        coupling[px][:, py],
        gaussian_kernel(x[px]),
        gaussian_kernel(y[py]),
        a[px],
        b[py],
    )
    torch.testing.assert_close(original, permuted, atol=1.0e-10, rtol=1.0e-9)


def test_projection_matches_pot_kl_reference_when_available():
    ot = pytest.importorskip("ot")
    from diffusion_ot.losses.infoot import sinkhorn_project, uniform_marginals

    torch.manual_seed(10)
    matrix = torch.rand(4, 6, dtype=torch.float64).add(0.1)
    a, b = uniform_marginals(4, 6, dtype=torch.float64)
    actual = sinkhorn_project(matrix, a, b, max_iterations=2000, tolerance=1.0e-12)
    reference = ot.sinkhorn(
        a.numpy(),
        b.numpy(),
        (-matrix.log()).numpy(),
        reg=1.0,
        numItermax=20_000,
        stopThr=1.0e-12,
    )
    torch.testing.assert_close(actual, torch.from_numpy(reference), atol=1.0e-8, rtol=1.0e-7)


def test_cost_sinkhorn_matches_pot_reference_when_available():
    ot = pytest.importorskip("ot")
    from diffusion_ot.losses.infoot import (
        sinkhorn_transport_from_cost,
        uniform_marginals,
    )

    torch.manual_seed(12)
    cost = torch.randn(4, 6, dtype=torch.float64)
    a, b = uniform_marginals(4, 6, dtype=torch.float64)
    actual = sinkhorn_transport_from_cost(
        cost,
        a,
        b,
        regularization=0.7,
        max_iterations=2000,
        tolerance=1.0e-12,
    )
    reference = ot.sinkhorn(
        a.numpy(),
        b.numpy(),
        cost.numpy(),
        reg=0.7,
        numItermax=20_000,
        stopThr=1.0e-12,
    )

    torch.testing.assert_close(
        actual, torch.from_numpy(reference), atol=1.0e-9, rtol=1.0e-8
    )


def test_independent_plan_has_zero_kernelized_mutual_information():
    from diffusion_ot.losses.infoot import gaussian_kernel, infoot_mutual_information, uniform_marginals

    torch.manual_seed(7)
    a, b = uniform_marginals(5, 8, dtype=torch.float64)
    independent = a[:, None] * b[None, :]
    value = infoot_mutual_information(
        independent,
        gaussian_kernel(torch.randn(5, 3, dtype=torch.float64)),
        gaussian_kernel(torch.randn(8, 3, dtype=torch.float64)),
        a,
        b,
    )
    assert float(value.abs()) < 1.0e-10


def test_fixed_plan_feature_shuffle_changes_infoot_value():
    from diffusion_ot.losses.infoot import (
        gaussian_kernel,
        infoot_mutual_information,
        seeded_nonindependent_coupling,
        uniform_marginals,
    )

    torch.manual_seed(11)
    x = torch.randn(6, 3, dtype=torch.float64)
    y = torch.randn(6, 3, dtype=torch.float64)
    a, b = uniform_marginals(6, 6, dtype=torch.float64)
    coupling = seeded_nonindependent_coupling(a, b, seed=13)
    first = infoot_mutual_information(coupling, gaussian_kernel(x), gaussian_kernel(y), a, b)
    second = infoot_mutual_information(
        coupling, gaussian_kernel(x), gaussian_kernel(y[torch.tensor([2, 5, 1, 4, 0, 3])]), a, b
    )
    assert not torch.isclose(first, second, atol=1.0e-8, rtol=1.0e-5)


def test_solver_returns_a_finite_feasible_plan():
    from diffusion_ot.losses.infoot import normalize_matching_features, solve_plain_infoot

    torch.manual_seed(17)
    x = normalize_matching_features(torch.randn(7, 5))
    y = normalize_matching_features(torch.randn(9, 5))
    result = solve_plain_infoot(x, y, inner_iterations=4)

    assert torch.isfinite(result.coupling).all()
    assert result.row_residual < 1.0e-4
    assert result.column_residual < 1.0e-4
    assert result.coupling.grad_fn is None


def test_plain_solver_one_step_matches_official_projected_sinkhorn_update():
    from diffusion_ot.losses.infoot import (
        gaussian_kernel,
        infoot_plan_gradient,
        sinkhorn_project,
        solve_plain_infoot,
        uniform_marginals,
    )

    torch.manual_seed(18)
    x = torch.randn(4, 3, dtype=torch.float64)
    y = torch.randn(5, 2, dtype=torch.float64)
    a, b = uniform_marginals(4, 5, dtype=torch.float64)
    initial = a[:, None] * b[None, :]
    kernel_x = gaussian_kernel(x, bandwidth=0.7, distance_scale=0.9)
    kernel_y = gaussian_kernel(y, bandwidth=0.7, distance_scale=1.1)
    negative_mi_gradient = -infoot_plan_gradient(
        initial, kernel_x, kernel_y, a, b
    )
    regularization = 0.5
    expected = sinkhorn_project(
        (-negative_mi_gradient / regularization).exp(),
        a,
        b,
        max_iterations=2000,
        tolerance=1.0e-12,
    )

    result = solve_plain_infoot(
        x,
        y,
        a=a,
        b=b,
        bandwidth=0.7,
        distance_scale_x=0.9,
        distance_scale_y=1.1,
        entropy_epsilon=regularization,
        inner_iterations=1,
        projection_iterations=2000,
        projection_tolerance=1.0e-12,
    )

    torch.testing.assert_close(result.coupling, expected, atol=1.0e-10, rtol=1.0e-9)
    assert result.restart == 0


def test_plain_solver_starts_from_independent_product():
    from diffusion_ot.losses.infoot import solve_plain_infoot, uniform_marginals

    x = torch.randn(3, 2, dtype=torch.float64)
    y = torch.randn(5, 4, dtype=torch.float64)
    a, b = uniform_marginals(3, 5, dtype=torch.float64)
    result = solve_plain_infoot(x, y, a=a, b=b, inner_iterations=0)

    torch.testing.assert_close(result.coupling, a[:, None] * b[None, :])


def test_detached_plan_outer_loss_updates_both_feature_extractors():
    from diffusion_ot.losses.infoot import (
        normalize_matching_features,
        plain_infoot_feature_loss,
        solve_plain_infoot,
    )

    torch.manual_seed(21)
    encoder_x = torch.nn.Linear(4, 3)
    encoder_y = torch.nn.Linear(5, 3)
    features_x = normalize_matching_features(encoder_x(torch.randn(6, 4)))
    features_y = normalize_matching_features(encoder_y(torch.randn(7, 5)))
    solution = solve_plain_infoot(
        features_x.detach(), features_y.detach(), inner_iterations=3
    )
    loss = plain_infoot_feature_loss(
        features_x, features_y, solution.coupling.detach()
    )
    loss.backward()

    assert solution.coupling.grad_fn is None
    assert all(parameter.grad is not None for parameter in encoder_x.parameters())
    assert all(parameter.grad is not None for parameter in encoder_y.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in encoder_x.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in encoder_y.parameters())


def test_conditional_projection_weights_are_normalized():
    from diffusion_ot.losses.infoot import (
        conditional_reference_weights,
        seeded_nonindependent_coupling,
        uniform_marginals,
        weighted_target_codes,
    )

    torch.manual_seed(23)
    reference_x = torch.randn(5, 4)
    target_codes = torch.randn(7, 4)
    query = torch.randn(3, 4)
    a, b = uniform_marginals(5, 7)
    coupling = seeded_nonindependent_coupling(a, b, seed=29)
    weights = conditional_reference_weights(query, reference_x, target_codes, coupling, a=a, b=b)
    projected = weighted_target_codes(weights, target_codes)

    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3), atol=1.0e-6, rtol=0)
    assert projected.shape == (3, 4)


@pytest.mark.parametrize("reverse", [False, True])
def test_conditional_reference_weights_match_equation7_with_target_correction(reverse):
    import math

    from diffusion_ot.losses.infoot import conditional_reference_weights, weighted_target_codes

    source = torch.tensor([[0.0], [1.4]], dtype=torch.float64)
    target = torch.tensor([[0.0, 0.0], [0.2, 0.4], [2.0, 1.4]], dtype=torch.float64)
    coupling = torch.tensor([[0.20, 0.06, 0.04], [0.10, 0.14, 0.46]], dtype=torch.float64)
    if reverse:
        source, target, coupling = target, source, coupling.T
    query = source[:1] + 0.3
    a, b = coupling.sum(1), coupling.sum(0)
    bandwidth, scale_x, scale_y = 0.7, 0.8, 1.3

    def kernel(x, y, scale):
        return math.exp(-float(((x - y) ** 2).sum()) / (2 * (bandwidth * scale) ** 2))

    # Independent scalar evaluation of the joint and both marginals in Eq. (7).
    source_density = sum(float(a[i]) * kernel(query[0], x, scale_x) for i, x in enumerate(source))
    scores = []
    uncorrected = []
    for j, y in enumerate(target):
        joint = sum(
            float(coupling[i, l]) * kernel(query[0], x, scale_x) * kernel(y, v, scale_y)
            for i, x in enumerate(source) for l, v in enumerate(target)
        )
        target_density = sum(float(b[l]) * kernel(y, v, scale_y) for l, v in enumerate(target))
        scores.append(float(b[j]) * joint / (source_density * target_density))
        uncorrected.append(float(b[j]) * joint)
    expected = torch.tensor([scores], dtype=torch.float64)
    expected /= expected.sum(1, keepdim=True)
    weights = conditional_reference_weights(
        query, source, target, coupling, a=a, b=b,
        bandwidth=bandwidth, distance_scale_x=scale_x, distance_scale_y=scale_y,
    )
    torch.testing.assert_close(weights, expected)
    source_only = torch.tensor(
        [[kernel(query[0], x, scale_x) for x in source]], dtype=torch.float64
    ) @ coupling
    source_only /= source_only.sum(1, keepdim=True)
    no_target_correction = torch.tensor([uncorrected], dtype=torch.float64)
    no_target_correction /= no_target_correction.sum(1, keepdim=True)
    assert not torch.allclose(weights, source_only)
    assert not torch.allclose(weights, no_target_correction)
    # Decoder values are raw codes, separate from the lower-dimensional KDE features.
    raw = torch.arange(len(target) * 4, dtype=torch.float64).reshape(len(target), 4)
    torch.testing.assert_close(weighted_target_codes(weights, raw), expected @ raw)


def test_conditional_projection_normalizes_when_source_kernels_underflow():
    from diffusion_ot.losses.infoot import conditional_reference_weights, gaussian_kernel

    source = torch.tensor([[0.0], [1.0]])
    target = torch.tensor([[0.0], [0.3], [2.0]])
    query = torch.tensor([[100.0]])
    coupling = torch.tensor([[0.20, 0.06, 0.04], [0.10, 0.14, 0.46]])
    a, b = coupling.sum(1), coupling.sum(0)
    assert gaussian_kernel(query, source).count_nonzero() == 0
    weights = conditional_reference_weights(query, source, target, coupling, a=a, b=b)
    target_kernel = gaussian_kernel(target)
    expected = (coupling[1:2] @ target_kernel.T) * (b / (target_kernel @ b))
    expected /= expected.sum(1, keepdim=True)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.sum(1), torch.ones(1))


def test_conditional_projection_independent_plan_recovers_target_masses():
    from diffusion_ot.losses.infoot import conditional_reference_weights

    source = torch.tensor([[0.0], [1.0]])
    target = torch.tensor([[0.0], [0.1], [2.0]])
    a, b = torch.tensor([0.3, 0.7]), torch.tensor([0.2, 0.3, 0.5])
    weights = conditional_reference_weights(source, source, target, torch.outer(a, b), a=a, b=b)
    torch.testing.assert_close(weights, b.expand(2, -1))


def test_conditional_projection_rejects_zero_normalization():
    from diffusion_ot.losses.infoot import conditional_reference_weights

    source = torch.tensor([[0.0], [1.0]])
    with pytest.raises(FloatingPointError, match="normalization"):
        conditional_reference_weights(source, source, source, torch.zeros(2, 2))


def test_conditional_projection_uses_larger_target_bank_and_empirical_masses():
    from diffusion_ot.losses.infoot import conditional_projection_weights

    source_reference = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    target_reference = torch.tensor([[0.0], [0.5], [2.0]], dtype=torch.float64)
    projection = torch.tensor([[-0.2], [0.3], [1.1], [2.4]], dtype=torch.float64)
    query = torch.tensor([[0.35], [0.9]], dtype=torch.float64)
    coupling = torch.tensor(
        [[0.22, 0.08, 0.10], [0.08, 0.22, 0.30]], dtype=torch.float64
    )
    a, b = coupling.sum(1), coupling.sum(0)
    projection_masses = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    bandwidth, scale_x, scale_y = 0.8, 0.7, 1.2

    weights = conditional_projection_weights(
        query,
        projection,
        source_reference,
        target_reference,
        coupling,
        a=a,
        b=b,
        projection_masses=projection_masses,
        bandwidth=bandwidth,
        distance_scale_x=scale_x,
        distance_scale_y=scale_y,
    )
    kx = torch.exp(
        -torch.cdist(query, source_reference).square() / (2 * (bandwidth * scale_x) ** 2)
    )
    ky = torch.exp(
        -torch.cdist(projection, target_reference).square()
        / (2 * (bandwidth * scale_y) ** 2)
    )
    expected = (kx @ coupling @ ky.T) / ((kx @ a)[:, None] * (ky @ b)[None, :])
    expected *= projection_masses
    expected /= expected.sum(1, keepdim=True)

    assert weights.shape == (2, 4)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, dtype=torch.float64))
