"""InfoOT audit against the author's mathematical update and local safeguards.

Source reviewed: https://github.com/chingyaoc/InfoOT/blob/main/infoot.py
The reference expressions below independently evaluate its Gaussian KDE and
fused entropic-OT update; they do not vendor or execute downloaded source code.
"""
import numpy as np
import pytest
import torch

from diffusion_ot.losses.infoot import (
    gaussian_kernel, infoot_distance_scale, infoot_mutual_information,
    infoot_plan_gradient, solve_infoot,
)


def _reference_kernel(features, bandwidth):
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)
    rms = np.sqrt(np.mean(distances ** 2) / 2)
    return np.exp(-.5 * (distances / (bandwidth * rms)) ** 2), rms


def _reference_positive_mi_gradient(plan, kernel_x, kernel_y):
    """Evaluate the KDE derivative with scalar sums, independent of local matmuls.

    The author's migrad returns the NEGATIVE of this expression. Uniform
    marginals match the author's implementation; all test densities exceed eps.
    """
    rows, columns = plan.shape
    joint = np.empty_like(plan)
    for i in range(rows):
        for j in range(columns):
            joint[i, j] = sum(kernel_x[i, r] * plan[r, s] * kernel_y[j, s]
                              for r in range(rows) for s in range(columns))
    ratios = joint / np.outer(kernel_x.mean(axis=1), kernel_y.mean(axis=1))
    result = np.log(ratios)
    for r in range(rows):
        for s in range(columns):
            result[r, s] += sum(plan[i, j] * kernel_x[i, r] * kernel_y[j, s] / joint[i, j]
                                for i in range(rows) for j in range(columns))
    return result


def test_rms_gaussian_and_gradient_match_author_formula_without_active_floors():
    rng = np.random.default_rng(20260925)
    x, y = rng.normal(size=(3, 2)), rng.normal(size=(4, 2))
    plan = rng.uniform(.1, 1., size=(3, 4))
    plan /= plan.sum()
    kx, scale_x = _reference_kernel(x, .55)
    ky, scale_y = _reference_kernel(y, .55)
    tx, ty, tp = (torch.from_numpy(value) for value in (x, y, plan))
    torch.testing.assert_close(infoot_distance_scale(tx), torch.tensor(scale_x, dtype=torch.float64))
    torch.testing.assert_close(infoot_distance_scale(ty), torch.tensor(scale_y, dtype=torch.float64))
    torch.testing.assert_close(gaussian_kernel(tx, bandwidth=.55, distance_scale=scale_x), torch.from_numpy(kx))
    torch.testing.assert_close(gaussian_kernel(ty, bandwidth=.55, distance_scale=scale_y), torch.from_numpy(ky))
    actual = infoot_plan_gradient(tp, torch.from_numpy(kx), torch.from_numpy(ky))
    torch.testing.assert_close(actual, torch.from_numpy(_reference_positive_mi_gradient(plan, kx, ky)),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("off_diagonal", [5e-9, 1e-8, 2e-8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_plan_gradient_matches_clamped_mi_below_at_and_above_density_floor(off_diagonal, dtype):
    # A feasible plan with narrow Gaussian kernels makes the two off-diagonal
    # joint densities hit both sides and the inclusive boundary of the floor.
    plan = torch.tensor([[.5 - off_diagonal, off_diagonal],
                         [off_diagonal, .5 - off_diagonal]], dtype=dtype, requires_grad=True)
    marginal = torch.full((2,), .5, dtype=dtype)
    features = torch.tensor([[0.], [1.]], dtype=dtype)
    kernel = gaussian_kernel(features, bandwidth=.01)
    torch.testing.assert_close(kernel, torch.eye(2, dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(plan.sum(0), marginal)
    torch.testing.assert_close(plan.sum(1), marginal)
    value = infoot_mutual_information(plan, kernel, kernel, marginal, marginal, eps=1e-8)
    autograd_gradient, = torch.autograd.grad(value, plan)
    actual = infoot_plan_gradient(plan.detach(), kernel, kernel, marginal, marginal, eps=1e-8)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(actual, autograd_gradient, rtol=tolerance, atol=tolerance)
    expected_off_diagonal = np.log(max(off_diagonal, 1e-8) / .25) + (1. if off_diagonal >= 1e-8 else 0.)
    assert actual[0, 1].item() == pytest.approx(expected_off_diagonal)


def test_floor_active_plan_gradient_matches_feasible_directional_finite_difference():
    eps = 1e-8
    plan = torch.tensor([[.5 - 5e-9, 5e-9], [5e-9, .5 - 5e-9]], dtype=torch.float64)
    kernel = torch.eye(2, dtype=torch.float64)
    direction = torch.tensor([[-1., 1.], [1., -1.]], dtype=torch.float64)
    step = 1e-10  # Neither perturbation crosses the density floor.
    numerical = (infoot_mutual_information(plan + step * direction, kernel, kernel, eps=eps)
                 - infoot_mutual_information(plan - step * direction, kernel, kernel, eps=eps)) / (2 * step)
    analytic = (infoot_plan_gradient(plan, kernel, kernel, eps=eps) * direction).sum()
    torch.testing.assert_close(analytic, numerical, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("floor_active", [False, True])
def test_complete_derivative_uses_adjoint_for_nonsymmetric_kernel_matrices(floor_active):
    generator = torch.Generator().manual_seed(55)
    plan = torch.rand(3, 4, dtype=torch.float64, generator=generator).requires_grad_()
    kx = torch.rand(3, 3, dtype=torch.float64, generator=generator) + .1
    ky = torch.rand(4, 4, dtype=torch.float64, generator=generator) + .1
    if floor_active:
        # Suppress one row to exercise masking plus the nonsymmetric adjoint.
        kx[0] *= 1e-10
    value = infoot_mutual_information(plan, kx, ky)
    expected, = torch.autograd.grad(value, plan)
    actual = infoot_plan_gradient(plan.detach(), kx, ky)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_fused_outer_update_matches_author_gradient_sign_and_pot_entropic_solve():
    ot = pytest.importorskip("ot")
    rng = np.random.default_rng(10)
    x, y = rng.normal(size=(3, 2)), rng.normal(size=(4, 2))
    kx, scale_x = _reference_kernel(x, .55)
    ky, scale_y = _reference_kernel(y, .55)
    a, b = np.full(3, 1 / 3), np.full(4, .25)
    initial = np.outer(a, b)
    # FusedInfoOT uses Euclidean cross-cost; our generic solver must reproduce
    # that update when supplied the same cost, despite training choosing cosine.
    cost = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    gradient = _reference_positive_mi_gradient(initial, kx, ky)
    expected = ot.bregman.sinkhorn(a, b, cost - .1 * gradient, reg=.4,
                                    numItermax=20000, stopThr=1e-13)
    actual = solve_infoot(torch.from_numpy(x), torch.from_numpy(y),
        cross_cost=torch.from_numpy(cost), bandwidth=.55,
        distance_scale_x=scale_x, distance_scale_y=scale_y,
        mi_weight=.1, entropy_epsilon=.4, inner_iterations=1,
        projection_iterations=20000, projection_tolerance=1e-13)
    torch.testing.assert_close(actual.coupling, torch.from_numpy(expected), rtol=1e-10, atol=1e-12)
    assert actual.sinkhorn_converged
