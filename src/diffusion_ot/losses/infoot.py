from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class InfoOTSolveResult:
    coupling: torch.Tensor
    objective: float
    mutual_information: float
    entropy: float
    row_residual: float
    column_residual: float
    iterations: int
    restart: int
    sinkhorn_converged: bool = True
    unconverged_inner_steps: int = 0
    outer_converged: bool = False
    plan_delta_l1: float = 0.0


def uniform_marginals(
    n: int,
    m: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n <= 0 or m <= 0:
        raise ValueError("Both marginal sizes must be positive.")
    return (
        torch.full((n,), 1.0 / n, device=device, dtype=dtype),
        torch.full((m,), 1.0 / m, device=device, dtype=dtype),
    )


def _validate_marginals(
    a: torch.Tensor,
    b: torch.Tensor,
    n: int,
    m: int,
) -> None:
    if a.shape != (n,) or b.shape != (m,):
        raise ValueError(f"Expected marginal shapes {(n,)} and {(m,)}, got {a.shape} and {b.shape}.")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Marginals must be finite.")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("Marginals must be nonnegative.")
    if not torch.allclose(a.sum(), b.sum(), atol=1.0e-6, rtol=1.0e-5):
        raise ValueError("Source and target marginals must have equal total mass.")


def normalize_matching_features(features: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("Matching features must have shape [batch, dimension].")
    return F.normalize(features, dim=-1, eps=eps)


def pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    if x.ndim != 2 or (y is not None and y.ndim != 2):
        raise ValueError("Pairwise distances require rank-two feature matrices.")
    y = x if y is None else y
    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimensions do not match.")
    x2 = x.square().sum(dim=1, keepdim=True)
    y2 = y.square().sum(dim=1, keepdim=True).transpose(0, 1)
    return (x2 + y2 - 2.0 * x @ y.transpose(0, 1)).clamp_min(0.0)


def median_distance_scale(features: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    distances = pairwise_squared_distances(features).detach().sqrt()
    mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
    values = distances[mask]
    if values.numel() == 0:
        return distances.new_tensor(1.0)
    return values.median().clamp_min(eps)


def infoot_cross_distance_scale(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return the official InfoOT scale for a pairwise-distance matrix."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("InfoOT distance scaling requires rank-two feature matrices.")
    mean_squared_distance = pairwise_squared_distances(x, y).detach().mean()
    return (mean_squared_distance / 2.0).sqrt().clamp_min(eps)


def infoot_distance_scale(features: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Return the Gaussian-kernel scale used by the official InfoOT code.

    The reference implementation computes ``sqrt(mean(D ** 2) / 2)`` from
    the full within-domain pairwise-distance matrix, including its diagonal.
    """
    return infoot_cross_distance_scale(features, features, eps=eps)


def _log_gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    bandwidth: float | torch.Tensor = 1.0,
    distance_scale: float | torch.Tensor = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    scale = torch.as_tensor(distance_scale, device=x.device, dtype=x.dtype)
    width = torch.as_tensor(bandwidth, device=x.device, dtype=x.dtype) * scale
    width = width.detach().clamp_min(eps)
    return -pairwise_squared_distances(x, y) / (2.0 * width.square())


def gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    bandwidth: float | torch.Tensor = 1.0,
    distance_scale: float | torch.Tensor = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    return _log_gaussian_kernel(
        x, y, bandwidth=bandwidth, distance_scale=distance_scale, eps=eps
    ).exp()


@torch.no_grad()
def kernel_offdiagonal_stats(
    features: torch.Tensor,
    *,
    bandwidth: float = 1.0,
    distance_scale: float | torch.Tensor = 1.0,
) -> dict[str, float]:
    kernel = gaussian_kernel(
        features, bandwidth=bandwidth, distance_scale=distance_scale
    )
    mask = ~torch.eye(kernel.shape[0], dtype=torch.bool, device=kernel.device)
    values = kernel[mask]
    if values.numel() == 0:
        values = kernel.flatten()
    return {
        "offdiagonal_min": float(values.min().cpu()),
        "offdiagonal_mean": float(values.mean().cpu()),
        "offdiagonal_max": float(values.max().cpu()),
    }


def kernel_densities(
    coupling: torch.Tensor,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, m = coupling.shape
    if kernel_x.shape != (n, n) or kernel_y.shape != (m, m):
        raise ValueError("Kernel matrices do not match the coupling dimensions.")
    _validate_marginals(a, b, n, m)
    joint = kernel_x @ coupling @ kernel_y.transpose(0, 1)
    marginal_x = kernel_x @ a
    marginal_y = kernel_y @ b
    return joint, marginal_x, marginal_y


def infoot_mutual_information(
    coupling: torch.Tensor,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    n, m = coupling.shape
    if a is None or b is None:
        default_a, default_b = uniform_marginals(n, m, device=coupling.device, dtype=coupling.dtype)
        a = default_a if a is None else a
        b = default_b if b is None else b
    joint, marginal_x, marginal_y = kernel_densities(coupling, kernel_x, kernel_y, a, b)
    log_ratio = (
        joint.clamp_min(eps).log()
        - marginal_x.clamp_min(eps).log()[:, None]
        - marginal_y.clamp_min(eps).log()[None, :]
    )
    return (coupling * log_ratio).sum()


def infoot_plan_gradient(
    coupling: torch.Tensor,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Complete derivative of the KDE mutual information with respect to Gamma."""
    n, m = coupling.shape
    if a is None or b is None:
        default_a, default_b = uniform_marginals(n, m, device=coupling.device, dtype=coupling.dtype)
        a = default_a if a is None else a
        b = default_b if b is None else b
    joint, marginal_x, marginal_y = kernel_densities(coupling, kernel_x, kernel_y, a, b)
    safe_joint = joint.clamp_min(eps)
    log_ratio = (
        safe_joint.log()
        - marginal_x.clamp_min(eps).log()[:, None]
        - marginal_y.clamp_min(eps).log()[None, :]
    )
    joint_dependency = (
        kernel_x @ (coupling / safe_joint) @ kernel_y.transpose(0, 1)
    )
    return log_ratio + joint_dependency


def coupling_entropy(coupling: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return -(coupling * coupling.clamp_min(eps).log()).sum()


def infoot_objective(
    coupling: torch.Tensor,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    *,
    mi_weight: float = 1.0,
    entropy_epsilon: float = 0.05,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    mutual_information = infoot_mutual_information(
        coupling, kernel_x, kernel_y, a, b, eps=eps
    )
    return -float(mi_weight) * mutual_information - float(entropy_epsilon) * coupling_entropy(
        coupling, eps=eps
    )


@torch.no_grad()
def sinkhorn_project(
    matrix: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    max_iterations: int = 200,
    tolerance: float = 1.0e-6,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Projection input must be a matrix.")
    n, m = matrix.shape
    _validate_marginals(a, b, n, m)
    coupling = matrix.clamp_min(eps).clone()
    for _ in range(int(max_iterations)):
        coupling.mul_((a / coupling.sum(dim=1).clamp_min(eps))[:, None])
        coupling.mul_((b / coupling.sum(dim=0).clamp_min(eps))[None, :])
        row_error = (coupling.sum(dim=1) - a).abs().max()
        column_error = (coupling.sum(dim=0) - b).abs().max()
        if max(float(row_error), float(column_error)) <= tolerance:
            break
    return coupling


@torch.no_grad()
def sinkhorn_transport_from_cost(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    regularization: float,
    max_iterations: int = 200,
    tolerance: float = 1.0e-6,
) -> torch.Tensor:
    """Solve entropic OT from a cost matrix with log-domain Sinkhorn scaling."""
    if cost.ndim != 2:
        raise ValueError("Sinkhorn cost must be a matrix.")
    n, m = cost.shape
    _validate_marginals(a, b, n, m)
    if not torch.isfinite(cost).all():
        raise ValueError("Sinkhorn cost must be finite.")
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError("Entropic Sinkhorn requires strictly positive marginals.")
    regularization = float(regularization)
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Entropic regularization must be positive.")
    if max_iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Sinkhorn iterations and tolerance must be positive.")

    log_kernel = -cost / regularization
    log_a = a.log()
    log_b = b.log()
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    coupling = torch.empty_like(cost)
    for iteration in range(max(int(max_iterations), 1)):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
        if iteration % 10 == 0 or iteration + 1 == int(max_iterations):
            coupling = (log_kernel + log_u[:, None] + log_v[None, :]).exp()
            row_error = (coupling.sum(dim=1) - a).abs().max()
            column_error = (coupling.sum(dim=0) - b).abs().max()
            if max(float(row_error), float(column_error)) <= tolerance:
                break
    return coupling


@torch.no_grad()
def seeded_nonindependent_coupling(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    seed: int,
    projection_iterations: int = 200,
    projection_tolerance: float = 1.0e-6,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    _validate_marginals(a, b, a.numel(), b.numel())
    generator = torch.Generator(device=a.device)
    generator.manual_seed(int(seed))
    matrix = torch.rand(
        (a.numel(), b.numel()),
        generator=generator,
        device=a.device,
        dtype=a.dtype,
    ).add_(0.05)
    return sinkhorn_project(
        matrix,
        a,
        b,
        max_iterations=projection_iterations,
        tolerance=projection_tolerance,
        eps=eps,
    )


@torch.no_grad()
def solve_infoot(
    features_x: torch.Tensor,
    features_y: torch.Tensor,
    *,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    bandwidth: float = 1.0,
    distance_scale_x: float | torch.Tensor = 1.0,
    distance_scale_y: float | torch.Tensor = 1.0,
    mi_weight: float = 1.0,
    entropy_epsilon: float = 0.05,
    inner_iterations: int = 50,
    projection_iterations: int = 200,
    projection_tolerance: float = 1.0e-5,
    eps: float = 1.0e-8,
    cross_cost: torch.Tensor | None = None,
    cross_cost_weight: float = 1.0,
    outer_tolerance: float = 0.0,
    min_inner_iterations: int = 5,
    outer_patience: int = 3,
    strict_convergence: bool = False,
) -> InfoOTSolveResult:
    if features_x.ndim != 2 or features_y.ndim != 2:
        raise ValueError("InfoOT expects [batch, dimension] feature matrices.")
    if features_x.device != features_y.device:
        raise ValueError("InfoOT feature matrices must be on the same device.")
    if not math.isfinite(mi_weight) or mi_weight < 0:
        raise ValueError("InfoOT mi_weight must be finite and nonnegative.")
    if not math.isfinite(outer_tolerance) or outer_tolerance < 0:
        raise ValueError("InfoOT outer_tolerance must be finite and nonnegative.")
    if min_inner_iterations < 1 or outer_patience < 1:
        raise ValueError("InfoOT minimum iterations and patience must be positive.")
    n, m = features_x.shape[0], features_y.shape[0]
    if a is None or b is None:
        default_a, default_b = uniform_marginals(n, m, device=features_x.device, dtype=features_x.dtype)
        a = default_a if a is None else a
        b = default_b if b is None else b
    _validate_marginals(a, b, n, m)
    kernel_x = gaussian_kernel(
        features_x, bandwidth=bandwidth, distance_scale=distance_scale_x, eps=eps
    )
    kernel_y = gaussian_kernel(
        features_y, bandwidth=bandwidth, distance_scale=distance_scale_y, eps=eps
    )

    fixed_cost = torch.zeros((n, m), device=features_x.device, dtype=features_x.dtype)
    if cross_cost is not None:
        if cross_cost.shape != (n, m) or not torch.isfinite(cross_cost).all():
            raise ValueError("Fused InfoOT cross_cost must be a finite [n, m] matrix.")
        if not torch.isfinite(torch.tensor(cross_cost_weight)) or cross_cost_weight < 0:
            raise ValueError("cross_cost_weight must be finite and nonnegative.")
        fixed_cost = cross_cost.detach().to(fixed_cost) * float(cross_cost_weight)

    # Official InfoOT initializes with the independent coupling. Each outer
    # iteration applies Eq. (5): use the negative MI gradient as the cost of a
    # fresh entropic OT solve. The entropy belongs to that Sinkhorn subproblem;
    # it is not added as an explicit gradient or accumulated in a mirror step.
    coupling = a[:, None] * b[None, :]
    completed_iterations = 0
    unconverged_inner_steps = 0
    stable_iterations = 0
    outer_converged = False
    plan_delta_l1 = 0.0
    for iteration in range(max(int(inner_iterations), 0)):
        mi_gradient = infoot_plan_gradient(
            coupling, kernel_x, kernel_y, a, b, eps=eps
        )
        previous_coupling = coupling
        coupling = sinkhorn_transport_from_cost(
            fixed_cost - float(mi_weight) * mi_gradient,
            a,
            b,
            regularization=entropy_epsilon,
            max_iterations=projection_iterations,
            tolerance=projection_tolerance,
        )
        completed_iterations = iteration + 1
        residual = max(
            float((coupling.sum(1) - a).abs().max()),
            float((coupling.sum(0) - b).abs().max()),
        )
        inner_converged = residual <= projection_tolerance
        unconverged_inner_steps += int(not inner_converged)
        plan_delta_l1 = float((coupling - previous_coupling).abs().sum())
        stable_iterations = (
            stable_iterations + 1
            if outer_tolerance > 0 and inner_converged and plan_delta_l1 <= outer_tolerance
            else 0
        )
        if completed_iterations >= min_inner_iterations and stable_iterations >= outer_patience:
            outer_converged = True
            break

    row_residual = float((coupling.sum(dim=1) - a).abs().max().cpu())
    column_residual = float((coupling.sum(dim=0) - b).abs().max().cpu())
    sinkhorn_converged = max(row_residual, column_residual) <= projection_tolerance
    if strict_convergence and not sinkhorn_converged:
        raise RuntimeError(
            f"InfoOT Sinkhorn marginals did not converge: row={row_residual:.3g}, "
            f"column={column_residual:.3g}, tolerance={projection_tolerance:.3g}. "
            "Increase projection_iterations or reassess the cost/MI/entropy scales."
        )
    mi = infoot_mutual_information(coupling, kernel_x, kernel_y, a, b, eps=eps)
    entropy = coupling_entropy(coupling, eps=eps)
    objective = (coupling * fixed_cost).sum() - float(mi_weight) * mi - float(entropy_epsilon) * entropy
    return InfoOTSolveResult(
        coupling=coupling,
        objective=float(objective.cpu()),
        mutual_information=float(mi.cpu()),
        entropy=float(entropy.cpu()),
        row_residual=row_residual,
        column_residual=column_residual,
        iterations=completed_iterations,
        restart=0,
        sinkhorn_converged=sinkhorn_converged,
        unconverged_inner_steps=unconverged_inner_steps,
        outer_converged=outer_converged,
        plan_delta_l1=plan_delta_l1,
    )


def solve_plain_infoot(features_x: torch.Tensor, features_y: torch.Tensor, **kwargs) -> InfoOTSolveResult:
    """Backward-compatible plain solver; fused callers use solve_infoot."""
    if "cross_cost" in kwargs:
        raise ValueError("Use solve_infoot for fused transport.")
    return solve_infoot(features_x, features_y, **kwargs)


@torch.no_grad()
def transport_diagnostics(coupling: torch.Tensor) -> dict[str, float]:
    """Concentration conditioned on each source, independent of batch size."""
    rows = coupling / coupling.sum(1, keepdim=True).clamp_min(1e-30)
    cols = coupling / coupling.sum(0, keepdim=True).clamp_min(1e-30)
    return {
        "mean_row_entropy": float(-(rows * rows.clamp_min(1e-30).log()).sum(1).mean()),
        "normalized_row_entropy": float(
            -(rows * rows.clamp_min(1e-30).log()).sum(1).mean() / max(math.log(coupling.shape[1]), 1e-30)
        ),
        "mean_row_effective_targets": float((-(rows * rows.clamp_min(1e-30).log()).sum(1)).exp().mean()),
        "mean_column_effective_sources": float((-(cols * cols.clamp_min(1e-30).log()).sum(0)).exp().mean()),
        "mean_row_max_probability": float(rows.max(1).values.mean()),
    }


def plain_infoot_feature_loss(
    features_x: torch.Tensor,
    features_y: torch.Tensor,
    coupling: torch.Tensor,
    *,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    bandwidth: float = 1.0,
    distance_scale_x: float | torch.Tensor = 1.0,
    distance_scale_y: float | torch.Tensor = 1.0,
    mi_weight: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    kernel_x = gaussian_kernel(
        features_x, bandwidth=bandwidth, distance_scale=distance_scale_x, eps=eps
    )
    kernel_y = gaussian_kernel(
        features_y, bandwidth=bandwidth, distance_scale=distance_scale_y, eps=eps
    )
    return -float(mi_weight) * infoot_mutual_information(
        coupling, kernel_x, kernel_y, a, b, eps=eps
    )


def conditional_density_ratio(
    query_x: torch.Tensor,
    gallery_y: torch.Tensor,
    reference_x: torch.Tensor,
    reference_y: torch.Tensor,
    coupling: torch.Tensor,
    *,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    bandwidth: float = 1.0,
    distance_scale_x: float | torch.Tensor = 1.0,
    distance_scale_y: float | torch.Tensor = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    n, m = coupling.shape
    if reference_x.shape[0] != n or reference_y.shape[0] != m:
        raise ValueError("Reference banks do not match the coupling dimensions.")
    if a is None or b is None:
        default_a, default_b = uniform_marginals(n, m, device=coupling.device, dtype=coupling.dtype)
        a = default_a if a is None else a
        b = default_b if b is None else b
    log_kernel_qx = _log_gaussian_kernel(
        query_x, reference_x, bandwidth=bandwidth, distance_scale=distance_scale_x, eps=eps
    )
    log_kernel_gy = _log_gaussian_kernel(
        gallery_y, reference_y, bandwidth=bandwidth, distance_scale=distance_scale_y, eps=eps
    )
    # Multiplying each KDE-kernel row by a positive constant cancels in the
    # joint-to-marginal ratio. Row softmax therefore preserves the ratios while
    # avoiding all-zero rows for low-density queries or gallery samples.
    kernel_qx = torch.softmax(log_kernel_qx, dim=1)
    kernel_gy = torch.softmax(log_kernel_gy, dim=1)
    joint = kernel_qx @ coupling @ kernel_gy.transpose(0, 1)
    marginal_q = kernel_qx @ a
    marginal_g = kernel_gy @ b
    ratios = joint / (marginal_q[:, None] * marginal_g[None, :]).clamp_min(
        torch.finfo(joint.dtype).tiny
    )
    if not torch.isfinite(ratios).all():
        raise FloatingPointError("Conditional density ratios are non-finite.")
    return ratios


def normalize_rows(weights: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return weights.clamp_min(0.0) / weights.clamp_min(0.0).sum(dim=1, keepdim=True).clamp_min(eps)


def conditional_projection_weights(
    query_x: torch.Tensor,
    projection_y: torch.Tensor,
    reference_x: torch.Tensor,
    reference_y: torch.Tensor,
    coupling: torch.Tensor,
    *,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    projection_masses: torch.Tensor | None = None,
    bandwidth: float = 1.0,
    distance_scale_x: float | torch.Tensor = 1.0,
    distance_scale_y: float | torch.Tensor = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Eq. (7) weights over a target projection bank, without top-k.

    The fit references define the KDE and may be smaller than ``projection_y``.
    The source marginal cancels in row normalization. Target smoothing and
    division by the target marginal do not cancel.
    """
    n, m = coupling.shape
    if reference_x.shape[0] != n or reference_y.shape[0] != m:
        raise ValueError("Reference banks do not match the coupling dimensions.")
    if query_x.ndim != 2 or projection_y.ndim != 2:
        raise ValueError("Query and projection features must be rank two.")
    if query_x.shape[1] != reference_x.shape[1]:
        raise ValueError("Query and source-reference feature dimensions differ.")
    if projection_y.shape[1] != reference_y.shape[1]:
        raise ValueError("Projection and target-reference feature dimensions differ.")
    if a is None or b is None:
        default_a, default_b = uniform_marginals(n, m, device=coupling.device, dtype=coupling.dtype)
        a = default_a if a is None else a
        b = default_b if b is None else b
    _validate_marginals(a, b, n, m)
    if not torch.isfinite(coupling).all() or (coupling < 0).any():
        raise ValueError("Coupling must be finite and nonnegative.")
    projection_count = projection_y.shape[0]
    if projection_count <= 0:
        raise ValueError("Projection bank must contain at least one sample.")
    if projection_masses is None:
        projection_masses = torch.full(
            (projection_count,),
            1.0 / projection_count,
            device=coupling.device,
            dtype=coupling.dtype,
        )
    if projection_masses.shape != (projection_count,):
        raise ValueError("Projection masses do not match the projection bank.")
    if (
        not torch.isfinite(projection_masses).all()
        or (projection_masses < 0).any()
        or projection_masses.sum() <= 0
    ):
        raise ValueError("Projection masses must be finite, nonnegative, and have positive mass.")
    ratios = conditional_density_ratio(
        query_x,
        projection_y,
        reference_x,
        reference_y,
        coupling,
        a=a,
        b=b,
        bandwidth=bandwidth,
        distance_scale_x=distance_scale_x,
        distance_scale_y=distance_scale_y,
        eps=eps,
    )
    weights = ratios * projection_masses[None, :]
    normalizer = weights.sum(dim=1, keepdim=True)
    if (
        not torch.isfinite(weights).all()
        or not torch.isfinite(normalizer).all()
        or (normalizer <= 0).any()
    ):
        raise FloatingPointError("Conditional projection has no finite positive normalization.")
    return weights / normalizer


def conditional_reference_weights(
    query_x: torch.Tensor,
    reference_x: torch.Tensor,
    reference_y: torch.Tensor,
    coupling: torch.Tensor,
    *,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    bandwidth: float = 1.0,
    distance_scale_x: float | torch.Tensor = 1.0,
    distance_scale_y: float | torch.Tensor = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Eq. (7) weights when the target references are the projection bank."""
    n, m = coupling.shape
    if b is None:
        _, b = uniform_marginals(n, m, device=coupling.device, dtype=coupling.dtype)
    return conditional_projection_weights(
        query_x,
        reference_y,
        reference_x,
        reference_y,
        coupling,
        a=a,
        b=b,
        projection_masses=b,
        bandwidth=bandwidth,
        distance_scale_x=distance_scale_x,
        distance_scale_y=distance_scale_y,
        eps=eps,
    )


def conditional_reference_log_weights(
    query_x: torch.Tensor,
    reference_x: torch.Tensor,
    reference_y: torch.Tensor,
    coupling: torch.Tensor,
    *,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    bandwidth: float = 1.0,
    distance_scale_x: float | torch.Tensor = 1.0,
    distance_scale_y: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Log Eq. (7) probabilities for small differentiable training banks.

    Equivalent to conditional_reference_weights, including target smoothing,
    target-density correction and target masses. Log-sum-exp contractions avoid
    zero-probability clipping and its dead gradients at small bandwidths.
    Temporary storage scales as query_count * source_count * target_count;
    large evaluation galleries should use the matrix-multiply weight helper.
    """
    n, m = coupling.shape
    if reference_x.shape[0] != n or reference_y.shape[0] != m:
        raise ValueError("Reference banks do not match the coupling dimensions.")
    if a is None or b is None:
        default_a, default_b = uniform_marginals(n, m, device=coupling.device, dtype=coupling.dtype)
        a = default_a if a is None else a
        b = default_b if b is None else b
    _validate_marginals(a, b, n, m)
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError("Log conditional projection requires positive reference masses.")
    if not torch.isfinite(coupling).all() or (coupling < 0).any() or coupling.sum() <= 0:
        raise ValueError("Coupling must be finite, nonnegative and have positive mass.")
    if (coupling.sum(0) <= 0).any() or (coupling.sum(1) <= 0).any():
        raise ValueError("Positive reference masses require nonempty coupling rows and columns.")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Projection bandwidth must be finite and positive.")
    log_qx = F.log_softmax(_log_gaussian_kernel(
        query_x, reference_x, bandwidth=bandwidth, distance_scale=distance_scale_x,
    ), dim=1)
    log_yy = F.log_softmax(_log_gaussian_kernel(
        reference_y, bandwidth=bandwidth, distance_scale=distance_scale_y,
    ), dim=1)
    log_plan = torch.where(coupling > 0, coupling, torch.ones_like(coupling)).log().masked_fill(
        coupling == 0, -torch.inf
    )
    log_source_transport = torch.logsumexp(log_qx[:, :, None] + log_plan[None, :, :], dim=1)
    log_joint = torch.logsumexp(log_source_transport[:, None, :] + log_yy[None, :, :], dim=2)
    log_target_density = torch.logsumexp(log_yy + b.log()[None, :], dim=1)
    # The source marginal cancels when probabilities are normalized per query.
    return F.log_softmax(log_joint - log_target_density[None, :] + b.log()[None, :], dim=1)


def nearest_plan_row_weights(
    query_x: torch.Tensor,
    reference_x: torch.Tensor,
    coupling: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    nearest = pairwise_squared_distances(query_x, reference_x).argmin(dim=1)
    return normalize_rows(coupling[nearest], eps=eps), nearest


def weighted_target_codes(weights: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
    if weights.ndim != 2 or target_codes.ndim != 2:
        raise ValueError("Weights and target codes must both be rank two.")
    if weights.shape[1] != target_codes.shape[0]:
        raise ValueError("Weight columns must match the target bank size.")
    return normalize_rows(weights) @ target_codes


def effective_target_count(weights: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    probabilities = normalize_rows(weights, eps=eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=1)
    return entropy.exp()


def solver_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    algorithm = str(config.get("algorithm", "official_projected_sinkhorn"))
    if algorithm != "official_projected_sinkhorn":
        raise ValueError("infoot.algorithm must be 'official_projected_sinkhorn'.")
    initialization = str(config.get("initialization", "independent_product"))
    if initialization != "independent_product":
        raise ValueError("infoot.initialization must be 'independent_product'.")
    return {
        "mi_weight": float(config.get("mi_weight", 1.0)),
        "entropy_epsilon": float(config.get("entropy_epsilon", 0.05)),
        "inner_iterations": int(config.get("inner_iterations", 50)),
        "projection_iterations": int(config.get("projection_iterations", 200)),
        "projection_tolerance": float(config.get("projection_tolerance", 1.0e-5)),
        "eps": float(config.get("numerical_epsilon", 1.0e-8)),
        "outer_tolerance": float(config.get("outer_tolerance", 0.0)),
        "min_inner_iterations": int(config.get("min_inner_iterations", 5)),
        "outer_patience": int(config.get("outer_patience", 3)),
        "strict_convergence": bool(config.get("strict_convergence", False)),
    }
