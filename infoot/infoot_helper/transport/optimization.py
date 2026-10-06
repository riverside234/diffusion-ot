from functools import partial
import math
from time import perf_counter

import torch

from .objective import fitting_loss, loss_terms, migrad
from .transport_utils import backtrack, is_feasible, marginal_error, sinkhorn_plan


@torch.no_grad()
def solve(solver, numIter=50, verbose=True, P0=None, *, tol=1e-5,
          marginal_tol=1e-4, patience=3, sinkhorn_iter=5000,
          line_search=True, reg=None, eps=1e-8):
    started = perf_counter()
    reg = solver.reg if reg is None else reg
    lam = getattr(solver, "lam", 1.0)
    C = getattr(solver, "C", None)
    if any(not math.isfinite(x) or x <= 0 for x in (reg, solver.h, tol, marginal_tol, eps)):
        raise ValueError("Regularization, bandwidth, and tolerances must be positive.")
    if min(numIter, patience, sinkhorn_iter) < 1 or not math.isfinite(lam) or lam < 0:
        raise ValueError("Iteration budgets must be positive and MI weight nonnegative.")
    if not all(torch.isfinite(K).all() for K in (solver.Ks, solver.Kt)):
        raise ValueError("Non-finite kernels; check reference features and bandwidth.")
    shape = (len(solver.Xs), len(solver.Xt))
    if 0 in shape:
        raise ValueError("Reference sets must be nonempty.")
    P = (solver.Xs.new_full(shape, 1 / math.prod(shape)) if P0 is None
         else P0.detach().to(solver.Xs).clone())
    if P.shape != shape or not is_feasible(P, marginal_tol):
        raise ValueError("P0 must be a finite, nonnegative plan with uniform marginals.")
    loss_fn = partial(fitting_loss, Kx=solver.Ks, Ky=solver.Kt,
                      reg=reg, C=C, mi_weight=lam, eps=eps)
    value = loss_fn(P).item()
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite initial fitting loss.")
    best, best_loss, best_iteration = P, value, 0
    history = [{"iteration": 0, "loss": value, "marginal": marginal_error(P)}]
    warmstart, stable, residual = None, 0, None
    status, error = "iteration_limit", None

    for iteration in range(1, numIter + 1):
        cost = lam * migrad(P, solver.Ks, solver.Kt, eps)
        if C is not None:
            cost = cost + C
        alpha = 0.0
        try:
            for accuracy in ((1.0, 0.1) if line_search else (1.0,)):
                Q, warmstart = sinkhorn_plan(
                    cost, reg, marginal_tol * accuracy,
                    sinkhorn_iter * (1 if accuracy == 1 else 2), warmstart,
                )
                residual = (Q - P).abs().sum().item()
                if residual < tol:
                    stable += 1
                    break
                stable = 0
                if line_search:
                    candidate, candidate_loss, alpha = backtrack(
                        P, Q, loss_fn, value, marginal_tol,
                    )
                else:
                    candidate, candidate_loss, alpha = Q, loss_fn(Q).item(), 1.0
                if alpha:
                    if not math.isfinite(candidate_loss):
                        raise FloatingPointError("Non-finite candidate loss.")
                    P, value = candidate, candidate_loss
                    break
        except FloatingPointError as exc:
            status, error = "inner_failure", str(exc)
            break
        if value < best_loss:
            best, best_loss, best_iteration = P, value, iteration
        history.append({"iteration": iteration, "loss": value, "alpha": alpha,
                        "residual": residual, "marginal": marginal_error(P)})
        if verbose:
            print(f"{iteration}: loss={value:.6f} residual={residual:.2e} alpha={alpha:.3g}")
        if stable >= patience:
            status = "converged"
            break
        if not alpha and not stable:
            status = "stalled"
            break

    cost, mi, entropy = loss_terms(best, solver.Ks, solver.Kt, C, eps)
    solver.P = best
    solver.converged_ = status == "converged" and (best - P).abs().sum().item() < tol
    solver.diagnostics_ = {
        "solver": type(solver).__name__,
        "loss": best_loss, "cost": cost.item(), "mi": mi.item(),
        "entropy": entropy.item(), "marginal": marginal_error(best),
        "last_residual": residual, "status": status, "error": error,
        "converged": solver.converged_, "iterations": iteration,
        "best_iteration": best_iteration, "seconds": perf_counter() - started,
        "reg": reg, "line_search": line_search, "history": history,
        "parameters": {"h": solver.h, "lam": lam, "eps": eps,
                       "tol": tol, "marginal_tol": marginal_tol,
                       "patience": patience, "sinkhorn_iter": sinkhorn_iter,
                       "dtype": str(P.dtype)},
    }
    if verbose:
        print(f"{status}: best_loss={best_loss:.6f} marginal={marginal_error(best):.2e}")
    return best
