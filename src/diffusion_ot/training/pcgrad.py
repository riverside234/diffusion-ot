"""Conflict-only PCGrad on weighted objectives, separately for E, H and G.

The pairwise rule follows Yu et al. (NeurIPS 2020). Original task gradients
are the projection references; each task visits the others in random order.
We sum projected gradients, preserving weighted-sum scale in conflict-free
cases. This is not the older reconstruction-priority gradient guard.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PCGradConfig:
    enabled: bool = False
    eps: float = 1e-12

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("pcgrad.enabled must be a boolean.")
        if isinstance(self.eps, bool) or not isinstance(self.eps, (int, float)) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("pcgrad.eps must be finite and positive.")

    @classmethod
    def from_mapping(cls, value):
        if value is None:
            return cls()
        if not isinstance(value, Mapping) or set(value) - {"enabled", "eps"}:
            raise ValueError("pcgrad supports only enabled and eps options.")
        return cls(**value)


@torch.no_grad()
def project_task_gradients(gradients, *, generator, eps=1e-12):
    """Project tuple-valued gradients without flattening models across devices.

    A small Gram matrix contains all inner products. Projection coefficients
    in this original-gradient basis implement exactly the sequential PCGrad
    rule; parameter tensors need only one final linear combination. This
    avoids keeping a second full copy of every projected task gradient.
    None means unused, and remains None when every task leaves it unused.
    """
    names = list(gradients)
    values = [tuple(gradients[name]) for name in names]
    if not values or len({len(v) for v in values}) != 1:
        raise ValueError("PCGrad needs nonempty tasks with matching gradient-list lengths.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("PCGrad eps must be finite and positive.")
    n = len(names)
    by_device = {}
    for entries in zip(*values):
        template = next((g for g in entries if g is not None), None)
        if template is None:
            continue
        if any(g is not None and (g.shape != template.shape or g.device != template.device) for g in entries):
            raise ValueError("Corresponding PCGrad gradients must have the same shape and device.")
        # Small task-by-task product for each parameter, accumulated on-device.
        # No transfer of a full gradient between the Cat/Dog devices is needed.
        with torch.autocast(device_type=template.device.type, enabled=False):
            matrix = torch.stack([g.detach().float().reshape(-1) if g is not None
                                  else torch.zeros(template.numel(), device=template.device)
                                  for g in entries])
            gram_part = matrix @ matrix.T
        if template.device not in by_device:
            by_device[template.device] = gram_part
        else:
            by_device[template.device].add_(gram_part)
    gram = sum((g.cpu().double() for g in by_device.values()), torch.zeros(n, n, dtype=torch.float64))
    if not torch.isfinite(gram).all():
        raise FloatingPointError("Non-finite PCGrad gradient inner products.")
    active = [i for i in range(n) if gram[i, i] > eps ** 2]
    coefficients = torch.eye(n, dtype=torch.float64)
    projections = 0
    for i in active:
        others = [j for j in active if j != i]
        for index in torch.randperm(len(others), generator=generator).tolist():
            j = others[index]
            dot = float(coefficients[i] @ gram[:, j])
            if dot < 0:
                coefficients[i, j] -= dot / float(gram[j, j])
                projections += 1
    summed = coefficients.sum(dim=0)
    if not torch.isfinite(summed).all():
        raise FloatingPointError("Non-finite PCGrad projection coefficients.")
    weights = summed.tolist()
    merged = []
    for entries in zip(*values):
        terms = [(weight, g) for weight, g in zip(weights, entries) if g is not None]
        if not terms:
            merged.append(None)
            continue
        combined = torch.zeros_like(terms[0][1], dtype=torch.float32)
        for weight, g in terms:
            combined.add_(g.detach().float(), alpha=weight)
        merged.append(combined.to(terms[0][1].dtype))
    original_sum = torch.ones(n, dtype=torch.float64)
    projected_gram = coefficients @ gram @ coefficients.T
    pair_count = len(active) * (len(active) - 1) // 2
    conflicts = sum(gram[i, j] < 0 for k, i in enumerate(active) for j in active[k + 1:])
    metrics = {
        "active_tasks": [names[i] for i in active], "projection_count": projections,
        "conflicting_pairs_before": int(conflicts), "pair_count": pair_count,
        "weighted_sum_norm": math.sqrt(max(0., float(original_sum @ gram @ original_sum))),
        "projected_sum_norm": math.sqrt(max(0., float(summed @ gram @ summed))),
        "correction_norm": math.sqrt(max(0., float((summed - original_sum) @ gram @ (summed - original_sum)))),
        "tasks": {
            names[i]: {
                "norm_before": math.sqrt(max(0., float(gram[i, i]))),
                "norm_after": math.sqrt(max(0., float(projected_gram[i, i]))),
                # First-order descent under ONLY this group's SGD direction;
                # neither a guarantee for the full model nor an Adam forecast.
                "sum_descent_fraction_before": float(gram[i] @ original_sum / gram[i, i]),
                "sum_descent_fraction_after": float(gram[i] @ summed / gram[i, i]),
            } for i in active
        },
    }
    return tuple(merged), metrics


def pcgrad_backward(losses, groups, *, seed, step, eps=1e-12, routed_gradients=None):
    """Assign gradients once, before clipping/Adam; do not call total.backward.

    ``routed_gradients`` are optional precomputed task gradients keyed by
    parameter id (e.g. generator-only code recovery in legacy recipes). This
    prevents a second autograd traversal from leaking that task into E/H.
    Random order uses a local CPU generator derived from train seed and step:
    resumes reproduce ordering without consuming diffusion/DataLoader RNG.
    """
    parameters = [p for group in groups.values() for p in group]
    if not parameters or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("PCGrad requires nonempty, disjoint parameter groups.")
    collected = {}
    live_names = [name for name, loss in losses.items() if loss.requires_grad]
    for name, loss in losses.items():
        if loss.numel() != 1 or not torch.isfinite(loss.detach()).all():
            raise ValueError("PCGrad losses must be finite scalars.")
        collected[name] = (tuple(None if g is None else g.detach() for g in torch.autograd.grad(
            loss, parameters, retain_graph=name != live_names[-1], allow_unused=True)) if loss.requires_grad
            else (None,) * len(parameters))
    for name, grads in (routed_gradients or {}).items():
        if name in collected:
            raise ValueError("PCGrad routed task names must be distinct.")
        collected[name] = tuple(grads.get(id(p)) for p in parameters)
    generator = torch.Generator(device="cpu").manual_seed((int(seed) + 104729 * int(step) + 1701) % (2 ** 63 - 1))
    report = {"enabled": True, "reduction": "sum", "scope": "separate_encoder_matching_head_generator",
              "order": "private_seed_and_step", "step": step, "groups": {}}
    if groups.get("patch_projector"):
        report["scope"] += "_patch_projector"
    start = 0
    for name, group in groups.items():
        end = start + len(group)
        if group:
            merged, metrics = project_task_gradients(
                {key: value[start:end] for key, value in collected.items()}, generator=generator, eps=eps)
            for parameter, gradient in zip(group, merged):
                parameter.grad = gradient
            report["groups"][name] = metrics
        start = end
    return report
