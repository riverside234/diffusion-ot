"""Safeguarded Newton steps for the capped-mass entropic OT dual.

Uses batched PyTorch linear algebra, on the input device. This accelerates the
existing partial objective; no dummy masses, balanced fallback or plan repair.
The linear system is diagonally scaled and damped; the *objective* is unchanged.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def dual_newton_step(base, a, b, mass, u, v, w):
    """One projected, dual-descent Newton step per independent partial problem.

    Minimize sum(exp(base+u+v+w)) - a.u - b.v - mass*w, with u,v <= 0.
    The free coordinates are negative potentials or violated capacities, plus
    the mass multiplier. Backtracking enforces descent of this SAME dual.
    A rejected/nonfinite/singular step leaves the block-ascent iterate intact.
    Callers still certify primal feasibility, complementarity and duality gap.
    """
    batch, n, m = base.shape
    z = torch.cat((u, v, w[:, None]), -1)
    plan = (base + u[:, :, None] + v[:, None, :] + w[:, None, None]).exp()
    rows, cols, total = plan.sum(-1), plan.sum(-2), plan.sum((-2, -1))
    gradient = torch.cat((rows - a, cols - b, (total - mass)[:, None]), -1)
    free = torch.cat(((u < 0) | (rows > a), (v < 0) | (cols > b),
                      torch.ones((batch, 1), dtype=torch.bool, device=base.device)), -1)
    hessian = torch.cat((rows, cols, total[:, None]), -1).diag_embed()
    hessian[:, :n, n:n+m] = plan
    hessian[:, n:n+m, :n] = plan.transpose(-2, -1)
    hessian[:, :n, -1] = hessian[:, -1, :n] = rows
    hessian[:, n:n+m, -1] = hessian[:, -1, n:n+m] = cols
    hessian *= free[:, :, None] * free[:, None, :]
    hessian.diagonal(dim1=-2, dim2=-1).add_(~free)
    scale = hessian.diagonal(dim1=-2, dim2=-1).clamp_min(1e-300).rsqrt()
    hessian *= scale[:, :, None] * scale[:, None, :]
    hessian.diagonal(dim1=-2, dim2=-1).add_(1e-12)
    solution, info = torch.linalg.solve_ex(hessian, (-gradient * free * scale).unsqueeze(-1),
                                         check_errors=False)
    direction = solution.squeeze(-1) * scale
    valid = (info == 0) & torch.isfinite(direction).all(-1)
    old = total - (a*u).sum(-1) - (b*v).sum(-1) - mass*w
    # Roundoff allowance applies only to line search, never to certification.
    roundoff = 8 * torch.finfo(base.dtype).eps * (1 + old.abs())
    step = base.new_ones(batch)
    accepted = torch.zeros_like(valid)
    result = z.clone()
    for _ in range(30):
        trial = z + step[:, None] * direction
        trial[:, :-1].clamp_max_(0)
        tu, tv, tw = trial[:, :n], trial[:, n:n+m], trial[:, -1]
        log_plan = base + tu[:, :, None] + tv[:, None, :] + tw[:, None, None]
        value = log_plan.exp().sum((-2, -1)) - (a*tu).sum(-1) - (b*tv).sum(-1) - mass*tw
        slope = (gradient * (trial-z)).sum(-1)
        ok = valid & torch.isfinite(value) & (slope <= 0) & (value <= old + 1e-4*slope + roundoff)
        result = torch.where((~accepted & ok)[:, None], trial, result)
        accepted |= ok
        # No GPU-to-CPU check per backtrack; fixed bounded work with pair masks.
        step = torch.where(accepted, step, step*.5)
    return result[:, :n], result[:, n:n+m], result[:, -1], accepted
