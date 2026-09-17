"""Exact full-bank InfoOT and bounded-memory conditional projection.

Tiling changes execution only: every row/column participates in a single OT
problem. Image models are not needed by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import torch

from diffusion_ot.losses.infoot import (
    _log_gaussian_kernel, gaussian_kernel, infoot_plan_gradient, uniform_marginals,
)
from diffusion_ot.evaluation.offline_artifacts import atomic_torch, append_json, read_torch


@dataclass(frozen=True)
class FullFitSettings:
    bandwidth: float = .55
    mi_weight: float = .10
    entropy_epsilon: float = .05
    cross_cost_weight: float = 1.0
    outer_iterations: int = 1200
    sinkhorn_iterations: int = 2000
    absolute_tolerance: float = 1e-7
    relative_tolerance: float = 1e-3
    outer_tolerance: float = 1e-5
    min_outer_iterations: int = 5
    patience: int = 3
    block_size: int = 512
    checkpoint_every: int = 10
    numerical_epsilon: float = 1e-8

    def validate(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid full InfoOT setting: {name}={value}")
            if name not in {"mi_weight", "cross_cost_weight"} and value == 0:
                raise ValueError(f"{name} must be positive")
        for name in ("outer_iterations", "sinkhorn_iterations", "min_outer_iterations",
                     "patience", "block_size", "checkpoint_every"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")


def reference_rms(features: torch.Tensor) -> float:
    """sqrt(mean(all pairwise squared distances)/2), including self-pairs.

    The identity E||X-X'||²/2 = E||X-E X||² avoids an N² calibration array.
    Accumulate in float64 to avoid cancellation in contracted feature banks.
    """
    if features.ndim != 2 or len(features) < 2 or not torch.isfinite(features).all():
        raise ValueError("Calibration requires at least two finite feature rows")
    centered = features.double() - features.double().mean(0)
    scale = float(centered.square().sum(1).mean().sqrt())
    if scale < 1e-8:
        raise ValueError("Matching references have collapsed; cannot calibrate RMS")
    return scale


def feasibility(plan, a, b, settings):
    rows = (plan.sum(1, dtype=torch.float64) - a.double()).abs()
    cols = (plan.sum(0, dtype=torch.float64) - b.double()).abs()
    result = {"row_absolute": float(rows.max()), "column_absolute": float(cols.max()),
              "row_relative": float((rows / a.double()).max()),
              "column_relative": float((cols / b.double()).max()),
              "mass": float(plan.sum(dtype=torch.float64))}
    result["feasible"] = bool(torch.isfinite(plan).all() and (plan >= 0).all()) and (
        max(result["row_absolute"], result["column_absolute"]) <= settings.absolute_tolerance
        and max(result["row_relative"], result["column_relative"]) <= settings.relative_tolerance
        and abs(result["mass"] - 1) <= 1e-6)
    return result


@torch.no_grad()
def tiled_sinkhorn(cost, a, b, settings):
    """Log Sinkhorn; each tile reduction spans the full opposite marginal."""
    if not torch.isfinite(cost).all():
        raise FloatingPointError("Non-finite InfoOT cost")
    kernel = -cost / settings.entropy_epsilon
    u, v = torch.zeros_like(a), torch.zeros_like(b)
    block = settings.block_size
    plan = torch.empty_like(cost)
    for iteration in range(settings.sinkhorn_iterations):
        for start in range(0, len(a), block):
            u[start:start+block] = a[start:start+block].log() - torch.logsumexp(
                kernel[start:start+block] + v[None, :], dim=1)
        for start in range(0, len(b), block):
            v[start:start+block] = b[start:start+block].log() - torch.logsumexp(
                kernel[:, start:start+block] + u[:, None], dim=0)
        if (iteration + 1) % 10 == 0 or iteration + 1 == settings.sinkhorn_iterations:
            for start in range(0, len(a), block):
                plan[start:start+block] = (
                    kernel[start:start+block] + u[start:start+block, None] + v[None, :]).exp()
            checks = feasibility(plan, a, b, settings)
            if checks["feasible"]:
                break
    return plan, checks, iteration + 1


@torch.no_grad()
def fit_global(x, y, cross_cost, *, scales, settings, output: Path, identity: str,
               resume=False, max_new_iterations=None):
    """Return (plan, report). An interrupted/preflight fit returns no plan.

    Progress is reusable only for the same artifact identity and solver settings.
    """
    settings.validate()
    if max_new_iterations is not None and max_new_iterations < 1:
        raise ValueError("max_new_iterations must be positive")
    if x.dtype not in {torch.float32, torch.float64} or y.dtype != x.dtype or y.device != x.device:
        raise ValueError("Use matching FP32/FP64 feature banks on one solver device")
    if (cross_cost.shape != (len(x), len(y)) or not torch.isfinite(cross_cost).all()
            or not torch.isfinite(x).all() or not torch.isfinite(y).all()):
        raise ValueError("Invalid full-bank cost/features")
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "solver_progress.pt"
    a, b = uniform_marginals(len(x), len(y), device=x.device, dtype=x.dtype)
    plan = a[:, None] * b[None, :]
    completed, stable = 0, 0
    if progress.exists():
        if not resume:
            raise ValueError("Fit progress already exists; pass --resume")
        saved = read_torch(progress)
        if saved["identity"] != identity or saved["settings"] != asdict(settings):
            raise ValueError("Resume input/solver fingerprint mismatch")
        plan = saved["coupling"].to(x)
        completed, stable = saved["iteration"], saved["stable_iterations"]
        if plan.shape != cross_cost.shape or not torch.isfinite(plan).all() or (plan < 0).any():
            raise ValueError("Invalid saved coupling")
        if saved["report"].get("converged"):
            if not feasibility(plan, a, b, settings)["feasible"]:
                raise ValueError("Saved converged coupling is infeasible")
            return plan, saved["report"]

    # Kernels are complete. Only their construction uses row chunks.
    kernels = []
    for features, scale in zip((x, y), scales):
        matrix = torch.empty((len(features), len(features)), device=x.device, dtype=x.dtype)
        for start in range(0, len(features), settings.block_size):
            matrix[start:start+settings.block_size] = gaussian_kernel(
                features[start:start+settings.block_size], features,
                bandwidth=settings.bandwidth, distance_scale=scale,
                eps=settings.numerical_epsilon)
        kernels.append(matrix)
    fixed_cost = cross_cost * settings.cross_cost_weight
    stop = settings.outer_iterations if max_new_iterations is None else min(
        settings.outer_iterations, completed + max_new_iterations)
    report = {"iteration": completed, "converged": False}
    for iteration in range(completed, stop):
        started = time.perf_counter()
        gradient = infoot_plan_gradient(plan, *kernels, a, b, eps=settings.numerical_epsilon)
        updated, checks, inner_steps = tiled_sinkhorn(fixed_cost - settings.mi_weight * gradient, a, b, settings)
        delta = float((updated - plan).abs().sum(dtype=torch.float64))
        plan = updated
        del gradient, updated
        stable = stable + 1 if checks["feasible"] and delta <= settings.outer_tolerance else 0
        converged = iteration + 1 >= settings.min_outer_iterations and stable >= settings.patience
        report = {**checks, "iteration": iteration + 1, "plan_delta_l1": delta,
                  "sinkhorn_iterations": inner_steps, "converged": converged,
                  "seconds": time.perf_counter() - started,
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated(x.device) if x.is_cuda else None}
        append_json(output / "solver_log.jsonl", report)
        print(f"InfoOT {iteration+1}: delta={delta:.3g}, relative="
              f"{max(checks['row_relative'], checks['column_relative']):.3g}, "
              f"{report['seconds']:.1f}s", flush=True)
        if converged or (iteration + 1) % settings.checkpoint_every == 0 or iteration + 1 == stop:
            atomic_torch(progress, {"identity": identity, "settings": asdict(settings),
                         "iteration": iteration + 1, "stable_iterations": stable,
                         "coupling": plan.cpu(), "report": report})
        if converged:
            return plan, report
    return None, report


@torch.no_grad()
def project_full(query, source, target, plan, target_codes, *, source_scale, target_scale,
                 bandwidth=.10, query_batch_size=32, target_block_size=512):
    """Exact Eq. (7) mean, with global target normalization and fixed scales."""
    if query_batch_size < 1 or target_block_size < 1 or bandwidth <= 0:
        raise ValueError("Projection sizes and bandwidth must be positive")
    if plan.shape != (len(source), len(target)) or len(target_codes) != len(target):
        raise ValueError("Coupling and projection banks do not match")
    if not torch.isfinite(plan).all() or (plan < 0).any():
        raise ValueError("Invalid coupling")
    _, b = uniform_marginals(*plan.shape, device=plan.device, dtype=plan.dtype)
    target_codes = target_codes.to(plan)
    result = []
    for qs in range(0, len(query), query_batch_size):
        q = query[qs:qs+query_batch_size].to(plan)
        left = torch.softmax(_log_gaussian_kernel(q, source, bandwidth=bandwidth,
                                                  distance_scale=source_scale), dim=1) @ plan
        # The source marginal cancels in the final row normalization.
        numerator = torch.zeros((len(q), target_codes.shape[1]), device=plan.device, dtype=plan.dtype)
        denominator = torch.zeros((len(q), 1), device=plan.device, dtype=plan.dtype)
        for ts in range(0, len(target), target_block_size):
            right = torch.softmax(_log_gaussian_kernel(
                target[ts:ts+target_block_size], target, bandwidth=bandwidth,
                distance_scale=target_scale), dim=1)
            weights = (left @ right.T) / (right @ b)[None, :].clamp_min(torch.finfo(plan.dtype).tiny)
            weights = weights * b[None, ts:ts+target_block_size]
            numerator += weights @ target_codes[ts:ts+target_block_size]
            denominator += weights.sum(1, keepdim=True)
        if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all() or (denominator <= 0).any():
            raise FloatingPointError("Conditional projection has no finite positive normalization")
        result.append((numerator / denominator).cpu())
    return torch.cat(result)
