import math

import torch
import ot


def marginal_error(P):
    return max(
        (P.sum(1) - 1 / P.shape[0]).abs().sum().item(),
        (P.sum(0) - 1 / P.shape[1]).abs().sum().item(),
    )


def is_feasible(P, tol=1e-4):
    return bool(
        torch.isfinite(P).all() and (P >= 0).all()
        and marginal_error(P) <= tol
    )


def sinkhorn_plan(cost, reg, marginal_tol=1e-4, sinkhorn_iter=5000,
                  warmstart=None):
    if not torch.isfinite(cost).all():
        raise FloatingPointError("Non-finite Sinkhorn cost.")
    cost = cost - cost.amin(1, keepdim=True)
    cost = cost - cost.amin(0, keepdim=True)
    p = cost.new_full((cost.shape[0],), 1 / cost.shape[0])
    q = cost.new_full((cost.shape[1],), 1 / cost.shape[1])
    for budget in (sinkhorn_iter, 2 * sinkhorn_iter):
        P, log = ot.bregman.sinkhorn(
            p, q, cost, reg=reg, method="sinkhorn_log",
            numItermax=budget, stopThr=marginal_tol / len(q) ** 0.5,
            warmstart=warmstart, log=True, warn=False,
        )
        warmstart = (log["log_u"], log["log_v"])
        if is_feasible(P, marginal_tol):
            return P, warmstart
        if not all(torch.isfinite(value).all() for value in warmstart):
            warmstart = None
    raise FloatingPointError(
        f"Sinkhorn marginal tolerance not met: {marginal_error(P):.2e}."
    )


def backtrack(P, Q, loss_fn, current_loss, marginal_tol, attempts=20):
    threshold = torch.finfo(P.dtype).eps * max(1.0, abs(current_loss))
    alpha = 1.0
    for _ in range(attempts):
        candidate = (1 - alpha) * P + alpha * Q
        value = loss_fn(candidate).item()
        if (math.isfinite(value) and value < current_loss - threshold
                and is_feasible(candidate, marginal_tol)):
            return candidate, value, alpha
        alpha *= 0.5
    return P, current_loss, 0.0
