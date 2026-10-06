import torch

from .optimization import solve
from .transport_utils import sinkhorn_plan


def initial_plan(solver, index, seed, marginal_tol=1e-4, sinkhorn_iter=5000):
    shape = (len(solver.Xs), len(solver.Xt))
    if index == 0:
        return "uniform", solver.Xs.new_full(shape, 1 / (shape[0] * shape[1]))
    generator = torch.Generator(device=solver.Xs.device).manual_seed(seed + index)
    noise = torch.randn(shape, device=solver.Xs.device, dtype=solver.Xs.dtype,
                        generator=generator)
    C = getattr(solver, "C", None)
    if C is not None and index == 2:
        name, cost = "geometric", C
    elif C is not None and index in (3, 4):
        scale = 1.0 if index == 3 else 4.0
        name, cost = f"perturbed_{scale:g}", C + scale * solver.reg * noise
    else:
        scale = 4.0 if index == 1 else 1.0
        name, cost = f"random_{scale:g}", -solver.reg * scale * noise
    P, _ = sinkhorn_plan(cost, solver.reg, marginal_tol, sinkhorn_iter)
    return name, P


@torch.no_grad()
def solve_multistart(solver, numIter=50, restarts=6, seed=0,
                     continuation=False, verbose=True, **options):
    if restarts < 1:
        raise ValueError("restarts must be positive.")
    if {"reg", "line_search", "P0"}.intersection(options):
        raise ValueError("Set reg on the solver; multistart controls P0 and line search.")
    runs, best, best_record = [], None, None
    for index in range(-1, restarts):
        name = "baseline" if index == -1 else f"start_{index}"
        try:
            P0 = None
            if index >= 0:
                name, P0 = initial_plan(
                    solver, index, seed, options.get("marginal_tol", 1e-4),
                    options.get("sinkhorn_iter", 5000),
                )
            if verbose:
                print(f"\n{name} (seed={seed + max(index, 0)})")
            P = solve(solver, numIter, verbose, P0,
                      line_search=index != -1, **options)
            record = dict(solver.diagnostics_, initialization=name,
                          seed=seed + max(index, 0))
            if best_record is None or record["loss"] < best_record["loss"]:
                best, best_record = P, record
        except FloatingPointError as exc:
            record = {"initialization": name, "status": "failed", "error": str(exc)}
            if verbose:
                print(f"{name}: {exc}")
        runs.append(record)

    if continuation:
        P0, stages = None, []
        try:
            for factor in (5.0, 2.0, 1.0):
                P0 = solve(solver, numIter, verbose, P0,
                           reg=factor * solver.reg, **options)
                stages.append(solver.diagnostics_)
            record = dict(stages[-1], initialization="continuation", seed=seed,
                          stages=stages)
            if best_record is None or record["loss"] < best_record["loss"]:
                best, best_record = P0, record
        except FloatingPointError as exc:
            record = {"initialization": "continuation", "status": "failed",
                      "error": str(exc), "stages": stages}
        runs.append(record)
    if best is None:
        raise FloatingPointError("No feasible run produced a finite fitting loss.")
    solver.P = best
    solver.converged_ = best_record["converged"]
    solver.diagnostics_ = {
        "solver": type(solver).__name__,
        "selected": best_record["initialization"], "loss": best_record["loss"],
        "status": best_record["status"], "converged": solver.converged_,
        "seed": seed, "restarts": restarts, "continuation": continuation,
        "numIter": numIter, "options": options, "runs": runs,
    }
    if verbose:
        print(f"Selected {solver.diagnostics_['selected']}: loss={best_record['loss']:.6f}")
    return best
