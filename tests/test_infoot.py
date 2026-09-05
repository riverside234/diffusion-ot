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
    result = solve_plain_infoot(x, y, inner_iterations=4, restarts=2, seed=19)

    assert torch.isfinite(result.coupling).all()
    assert result.row_residual < 1.0e-4
    assert result.column_residual < 1.0e-4
    assert result.coupling.grad_fn is None


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
        features_x.detach(), features_y.detach(), inner_iterations=3, restarts=1, seed=22
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
    weights = conditional_reference_weights(query, reference_x, coupling, a=a)
    projected = weighted_target_codes(weights, target_codes)

    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3), atol=1.0e-6, rtol=0)
    assert projected.shape == (3, 4)
