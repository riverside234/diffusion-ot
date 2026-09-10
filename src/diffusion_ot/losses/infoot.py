from __future__ import annotations

from dataclasses import dataclass
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
    joint_dependency = kernel_x.transpose(0, 1) @ (coupling / safe_joint) @ kernel_y
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
def solve_plain_infoot(
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
    step_size: float = 0.25,
    gradient_clip: float | None = 50.0,
    restarts: int = 3,
    seed: int = 0,
    eps: float = 1.0e-8,
) -> InfoOTSolveResult:
    if features_x.ndim != 2 or features_y.ndim != 2:
        raise ValueError("InfoOT expects [batch, dimension] feature matrices.")
    if features_x.device != features_y.device:
        raise ValueError("InfoOT feature matrices must be on the same device.")
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

    best: InfoOTSolveResult | None = None
    for restart in range(max(int(restarts), 1)):
        coupling = seeded_nonindependent_coupling(
            a,
            b,
            seed=int(seed) + restart,
            projection_iterations=projection_iterations,
            projection_tolerance=projection_tolerance,
            eps=eps,
        )
        completed_iterations = 0
        for iteration in range(max(int(inner_iterations), 0)):
            mi_gradient = infoot_plan_gradient(
                coupling, kernel_x, kernel_y, a, b, eps=eps
            )
            gradient = -float(mi_weight) * mi_gradient + float(entropy_epsilon) * (
                coupling.clamp_min(eps).log() + 1.0
            )
            if gradient_clip is not None:
                gradient = gradient.clamp(-float(gradient_clip), float(gradient_clip))
            log_update = coupling.clamp_min(eps).log() - float(step_size) * gradient
            log_update = log_update - log_update.max()
            coupling = sinkhorn_project(
                log_update.exp().clamp_min(eps),
                a,
                b,
                max_iterations=projection_iterations,
                tolerance=projection_tolerance,
                eps=eps,
            )
            completed_iterations = iteration + 1

        mi = infoot_mutual_information(coupling, kernel_x, kernel_y, a, b, eps=eps)
        entropy = coupling_entropy(coupling, eps=eps)
        objective = -float(mi_weight) * mi - float(entropy_epsilon) * entropy
        candidate = InfoOTSolveResult(
            coupling=coupling,
            objective=float(objective.cpu()),
            mutual_information=float(mi.cpu()),
            entropy=float(entropy.cpu()),
            row_residual=float((coupling.sum(dim=1) - a).abs().max().cpu()),
            column_residual=float((coupling.sum(dim=0) - b).abs().max().cpu()),
            iterations=completed_iterations,
            restart=restart,
        )
        if best is None or candidate.objective < best.objective:
            best = candidate
    if best is None:
        raise RuntimeError("InfoOT solver did not produce a coupling.")
    return best


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
    return {
        "mi_weight": float(config.get("mi_weight", 1.0)),
        "entropy_epsilon": float(config.get("entropy_epsilon", 0.05)),
        "inner_iterations": int(config.get("inner_iterations", 50)),
        "projection_iterations": int(config.get("projection_iterations", 200)),
        "projection_tolerance": float(config.get("projection_tolerance", 1.0e-5)),
        "step_size": float(config.get("step_size", 0.25)),
        "gradient_clip": config.get("gradient_clip", 50.0),
        "restarts": int(config.get("restarts", 3)),
        "eps": float(config.get("numerical_epsilon", 1.0e-8)),
    }
