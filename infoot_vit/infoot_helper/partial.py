"""Offline fixed-mass partial InfoOT; differentiate only the transported KDE MI.

This is the plan's project-specific extension, not upstream InfoOT's fixed-
marginal gradient. POT solves the exact entropic row/column-cap subproblem.
"""
from __future__ import annotations

import inspect
import math
import warnings

import ot
import torch

OBJECTIVE = "transported_kde_mi_mass_weighted_v1"
DEFAULTS = dict(h=.4, lam=.1, reg=.05, cost_scale="mean", max_outer_steps=100,
    max_inner_steps=10000, inner_tolerance=1e-10, feasibility_tolerance=1e-8,
    mass_tolerance=1e-8, outer_tolerance=1e-7, objective_tolerance=1e-12,
    max_backtracks=30, log_floor=1e-300, inner_acceleration="none")


def solver_config(options=None):
    options = options or {}
    if unknown := options.keys() - DEFAULTS.keys():
        raise ValueError(f"Unknown solver settings: {sorted(unknown)}")
    c = DEFAULTS | options
    if c["inner_acceleration"] not in {"none", "newton"}:
        raise ValueError("inner_acceleration must be 'none' or 'newton' (batched partial solver only).")
    for key in ("h", "reg", "inner_tolerance", "feasibility_tolerance", "mass_tolerance", "outer_tolerance"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if not math.isfinite(c["lam"]) or c["lam"] < 0 or not 0 < c["log_floor"] < 1:
        raise ValueError("lam must be nonnegative and log_floor must lie in (0,1).")
    if not math.isfinite(c["objective_tolerance"]) or c["objective_tolerance"] < 0:
        raise ValueError("objective_tolerance must be finite and nonnegative.")
    for key in ("max_outer_steps", "max_inner_steps", "max_backtracks"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if c["cost_scale"] != "mean" and (not isinstance(c["cost_scale"], (int, float))
                                      or not math.isfinite(c["cost_scale"]) or c["cost_scale"] <= 0):
        raise ValueError("cost_scale must be 'mean' or a positive fixed number.")
    return c


def distance(x, y):
    return torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")


def kernel_state(x, h):
    distances = distance(x, x)
    scale = float((distances.square().mean() / 2).sqrt())
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(h) or h <= 0:
        raise ValueError("Degenerate fit support or zero bandwidth; provide distinct training features.")
    return torch.exp(-.5 * (distances / (h * scale)).square()), scale


def information(plan, kx, ky, log_floor=1e-300):
    """Mass-weighted KDE MI; optional leading pair axis stays independent."""
    mass = plan.sum((-2, -1), keepdim=True)
    normalized = plan / mass
    joint = kx @ normalized @ ky.transpose(-2, -1)
    fx = (kx @ normalized.sum(-1).unsqueeze(-1)).squeeze(-1)
    fy = (ky @ normalized.sum(-2).unsqueeze(-1)).squeeze(-1)
    log_ratio = (joint.clamp_min(log_floor).log() - fx.clamp_min(log_floor).log().unsqueeze(-1)
                 - fy.clamp_min(log_floor).log().unsqueeze(-2))
    return mass[..., 0, 0] * (normalized * log_ratio).sum((-2, -1))


def information_gradient(plan, kx, ky, log_floor=1e-300):
    with torch.enable_grad():
        variable = plan.detach().clone().requires_grad_(True)
        return torch.autograd.grad(information(variable, kx, ky, log_floor).sum(), variable)[0].detach()


def objective(plan, cost, kx, ky, config):
    terms = dict(cost=float((plan * cost).sum()),
                 information=float(information(plan, kx, ky, config["log_floor"])),
                 entropy=float((torch.special.xlogy(plan, plan) - plan).sum()))
    terms["objective"] = terms["cost"] - config["lam"] * terms["information"] + config["reg"] * terms["entropy"]
    return terms


def feasibility(plan, a, b, mass, config):
    if plan.shape != (len(a), len(b)) or not torch.isfinite(plan).all() or (plan < 0).any():
        raise ValueError("Partial plan must have the expected shape and finite nonnegative entries.")
    result = dict(mass=float(plan.sum()), mass_error=abs(float(plan.sum()) - mass),
                  row_cap_error=float((plan.sum(1) - a).clamp_min(0).max()),
                  column_cap_error=float((plan.sum(0) - b).clamp_min(0).max()))
    if (result["mass_error"] > config["mass_tolerance"]
            or max(result["row_cap_error"], result["column_cap_error"]) > config["feasibility_tolerance"]):
        raise ValueError(f"Infeasible partial plan (no renormalization applied): {result}")
    return result


def capabilities():
    method = "method" in inspect.signature(ot.partial.entropic_partial_wasserstein).parameters
    return dict(pot_version=ot.__version__, torch_version=str(torch.__version__), partial_signature=str(inspect.signature(ot.partial.entropic_partial_wasserstein)),
                partial_method="sinkhorn_log" if method else "unavailable", working_dtype="float64",
                autodiff="torch.autograd", entropy="sum(Gamma*(log(Gamma)-1))")


def entropy_subproblem(a, b, cost, mass, config):
    if not 0 < mass <= 1:
        raise ValueError("keep_mass must be in (0,1]; it is mass, not a patch count.")
    if not torch.isfinite(cost).all():
        raise ValueError("Nonfinite effective partial cost.")
    # One scalar shift is constant on the feasible set, even for negative costs.
    shifted = cost - cost.min()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if mass == 1:
            plan, log = ot.bregman.sinkhorn(a, b, shifted, config["reg"], method="sinkhorn_log",
                numItermax=config["max_inner_steps"], stopThr=config["inner_tolerance"], log=True)
            method = "balanced_sinkhorn_log_s_equals_1"
        else:
            if capabilities()["partial_method"] == "unavailable":
                raise RuntimeError("This POT lacks log-domain partial OT. Install POT>=0.9.7 explicitly; no automatic upgrade/fallback.")
            # POT's torch log backend requires a tensor mass (nx.log(m)).
            plan, log = ot.partial.entropic_partial_wasserstein(a, b, shifted, config["reg"], m=cost.new_tensor(mass),
                method="sinkhorn_log", numItermax=config["max_inner_steps"], stopThr=config["inner_tolerance"], log=True)
            method = "partial_sinkhorn_log"
    residuals = feasibility(plan, a, b, mass, config)
    errors = log.get("err", [])
    error = float(errors[-1]) if len(errors) else None
    if error is None or not math.isfinite(error) or error > config["inner_tolerance"]:
        raise RuntimeError(f"{method} did not converge: error={error}; increase max_inner_steps or reassess scales.")
    return plan.detach(), dict(method=method, error=error, warnings=[str(w.message) for w in caught],
                               cost_shift=float(cost.min()), **residuals)


def solve_partial(cost, kx, ky, *, a=None, b=None, keep_mass=.8, config=None, on_step=None):
    config = solver_config(config)
    cost, kx, ky = [x.detach().to(dtype=torch.float64) for x in (cost, kx, ky)]
    if not 0 < keep_mass <= 1:
        raise ValueError("keep_mass must be in (0,1].")
    a = cost.new_full((len(cost),), 1 / len(cost)) if a is None else a.to(cost)
    b = cost.new_full((cost.shape[1],), 1 / cost.shape[1]) if b is None else b.to(cost)
    if any(not torch.isfinite(w).all() or (w <= 0).any() or abs(float(w.sum()) - 1) > 1e-12 for w in (a, b)):
        raise ValueError("Explicit a,b must be positive probability masses.")
    scale = float(cost.mean()) if config["cost_scale"] == "mean" else float(config["cost_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid fitted cost scale.")
    cost = cost / scale
    plan = keep_mass * torch.outer(a, b)
    history = []
    status = "max_outer_steps"
    if config["lam"] == 0:
        plan, inner = entropy_subproblem(a, b, cost, keep_mass, config)
        history.append(dict(iteration=1, step_size=1., status="converged", backtracks=0,
                            inner=inner, **objective(plan, cost, kx, ky, config)))
        status = "converged"
        if on_step:
            on_step(plan, history[-1])
    else:
        for iteration in range(1, config["max_outer_steps"] + 1):
            old = objective(plan, cost, kx, ky, config)
            gradient = information_gradient(plan, kx, ky, config["log_floor"])
            candidate, inner = entropy_subproblem(a, b, cost - config["lam"] * gradient, keep_mass, config)
            delta = float((candidate - plan).abs().sum())
            if delta <= config["outer_tolerance"]:
                status = "converged"
                history.append(dict(iteration=iteration, step_size=0., status=status, backtracks=0,
                    plan_delta_l1=delta, accepted_plan_delta_l1=0., objective_delta=0., inner=inner, **old))
                if on_step:
                    on_step(plan, history[-1])
                break
            step = 1.
            for backtracks in range(config["max_backtracks"]):
                trial = plan + step * (candidate - plan)
                terms = objective(trial, cost, kx, ky, config)
                if terms["objective"] <= old["objective"] + config["objective_tolerance"]:
                    break
                step *= .5
            else:
                status = "line_search_stalled"
                history.append(dict(iteration=iteration, step_size=0., status=status,
                    backtracks=config["max_backtracks"], plan_delta_l1=delta,
                    accepted_plan_delta_l1=0., objective_delta=0., inner=inner, **old))
                if on_step:
                    on_step(plan, history[-1])
                break
            # A vanishing accepted step is a stall, not proof of stationarity.
            if step * delta <= config["outer_tolerance"]:
                status = "line_search_stalled"
                history.append(dict(iteration=iteration, step_size=0., status=status, backtracks=backtracks,
                    plan_delta_l1=delta, accepted_plan_delta_l1=0., objective_delta=0., inner=inner, **old))
                if on_step:
                    on_step(plan, history[-1])
                break
            plan = trial.detach()
            record = dict(iteration=iteration, step_size=step, plan_delta_l1=delta,
                accepted_plan_delta_l1=step * delta, objective_delta=terms["objective"] - old["objective"],
                status="max_outer_steps" if iteration == config["max_outer_steps"] else "running",
                backtracks=backtracks, inner=inner, **terms)
            history.append(record)
            if on_step:
                on_step(plan, record)
    report = dict(status=status, objective_version=OBJECTIVE, cost_scale=scale,
                  config=config, keep_mass=keep_mass, runtime=capabilities(), history=history,
                  **feasibility(plan, a, b, keep_mass, config))
    return plan, report
