"""Exact dense partial InfoOT for independent GPU/CPU pair batches.

POT 0.9.7's partial API accepts one 2-D cost. We solve the same entropic
subproblem by exact dual block ascent: jointly optimize row potentials and
the mass multiplier, then column potentials and the mass multiplier. Grouping
each capacity constraint with mass avoids slow three-set Dykstra iterations
near saturated supports. No change to the objective or feasible set. The
mass-one control uses POT's balanced log updates, just like partial.py.
At an inner budget boundary a primal-dual/KKT certificate can resolve a
log-residual-only failure; the outer InfoOT stationarity check is unchanged.
See https://pythonot.github.io/_modules/ot/partial/partial_solvers.html.
"""
from __future__ import annotations

import math
import torch

from .partial import OBJECTIVE, capabilities, information, information_gradient, solver_config

VERSION = "dense_partial_infoot_batch_v3_primal_dual"
RUNNING, CONVERGED, BUDGET, STALLED, INVALID, INNER_FAILED, NONFINITE = range(7)
STATUS = ("running", "converged", "max_outer_steps", "line_search_stalled",
          "invalid_input", "inner_failed", "nonfinite_objective_or_gradient")


def objective_values(plan, cost, kx, ky, config):
    """[B,4] cost, transported MI, entropy, total; no host synchronization."""
    c = (plan * cost).sum((-2, -1))
    mi = information(plan, kx, ky, config["log_floor"])
    ent = (torch.special.xlogy(plan, plan) - plan).sum((-2, -1))
    return torch.stack((c, mi, ent, c - config["lam"] * mi + config["reg"] * ent), -1)


def residual_values(plan, a, b, mass):
    total = plan.sum((-2, -1))
    return torch.stack((total, (total - mass).abs(),
        (plan.sum(-1) - a).clamp_min(0).amax(-1),
        (plan.sum(-2) - b).clamp_min(0).amax(-1)), -1)


def feasible(residuals, config):
    return (torch.isfinite(residuals).all(-1)
            & (residuals[:, 1] <= config["mass_tolerance"])
            & (residuals[:, 2:].amax(-1) <= config["feasibility_tolerance"]))


def _capped_mass_potentials(log_weights, capacities, mass):
    """Exact log water filling: sum min(cap_i, exp(log_weight_i + w)) = mass.

    Sorting log(cap_i)-log_weight_i identifies the saturated prefix. Reverse
    logcumsumexp keeps the unsaturated tail accurate even for tiny weights.
    Returned inequality potentials are <= 0; the mass potential is free.
    Requires 0 < mass < sum(capacities), with all capacities positive.
    """
    log_caps = capacities.log()
    breaks, order = (log_caps - log_weights).sort(-1)
    caps = capacities.gather(-1, order)
    remaining = mass - (caps.cumsum(-1) - caps)
    tail = log_weights.gather(-1, order).flip(-1).logcumsumexp(-1).flip(-1)
    candidates = remaining.clamp_min(torch.finfo(log_weights.dtype).tiny).log() - tail
    admissible = (remaining > 0) & (candidates <= breaks)
    index = admissible.to(torch.int64).argmax(-1, keepdim=True)
    w = candidates.gather(-1, index).squeeze(-1)
    # No silent fallback if extreme inputs exhaust floating-point resolution.
    w = w.masked_fill(~admissible.any(-1), float("nan"))
    return (log_caps - log_weights - w[:, None]).clamp_max(0), w


def _duality_gap(plan, log_plan, base, a, b, mass, u, v, w, reg):
    """Primal minus dual for shifted cost -reg*base; no entropy approximation."""
    primal = reg * (plan * (log_plan - base - 1)).sum((-2, -1))
    dual = reg * ((u * a).sum(-1) + (v * b).sum(-1) + w * mass - plan.sum((-2, -1)))
    gap = primal - dual
    return gap, gap.abs() / torch.maximum(torch.ones_like(gap), torch.maximum(primal.abs(), dual.abs()))


def _partial_certificate(plan, log_plan, previous, base, a, b, mass, u, v, w):
    """Primal stability and KKT residual for the SAME entropic subproblem.

    Inequality potentials u,v must be nonpositive and complementary to unused
    capacity. Stationarity is exact up to roundoff for our exponential plan.
    Unlike unweighted log-plan L2, these checks cannot be dominated solely by
    changes in negligible-probability entries. No plan repair is performed.
    """
    rows, cols = plan.sum(-1), plan.sum(-2)
    delta = (plan - previous.exp()).abs().sum((-2, -1)) / mass
    residuals = torch.stack((
        ((rows - a) / a).clamp_min(0).amax(-1),
        ((cols - b) / b).clamp_min(0).amax(-1),
        (plan.sum((-2, -1)) - mass).abs() / mass,
        ((u * (rows - a)).abs().sum(-1) + (v * (cols - b)).abs().sum(-1)) / mass,
        u.clamp_min(0).amax(-1), v.clamp_min(0).amax(-1),
        (log_plan - base - u[:, :, None] - v[:, None, :] - w[:, None, None]).abs().amax((-2, -1)),
    ), -1)
    return delta, residuals.amax(-1)


@torch.no_grad()
def entropy_subproblem_batch(a, b, cost, mass, config):
    """Exact entropic subproblems, with pair-local stopping/failure tensors.

    Only one batch-wide host check per ten iterations. Converged/failed state
    is frozen by GPU masks. Counts include every update, including the first
    convergence check at iteration 1 (POT cpt=0).
    """
    batch = len(cost)
    valid = torch.isfinite(cost).all((-2, -1))
    shift = cost.amin((-2, -1))
    base = -(torch.where(valid[:, None, None], cost - shift[:, None, None], 0.)) / config["reg"]
    active = valid.clone()
    numerical = ~valid
    error = cost.new_full((batch,), float("inf"))
    plan_error, kkt_error = error.clone(), error.clone()
    stop_reason = torch.zeros(batch, dtype=torch.int64, device=cost.device)
    counts = torch.zeros(batch, dtype=torch.int64, device=cost.device)
    loga, logb = a.log(), b.log()
    u, v = torch.zeros_like(a), torch.zeros_like(b)
    w = cost.new_zeros(batch)
    if mass == 1.:
        lk = base.clone()
    else:
        w = math.log(mass) - base.logsumexp((-2, -1))
        lk = base + w[:, None, None]
    for iteration in range(config["max_inner_steps"]):
        previous = lk
        if mass == 1.:
            next_v = logb - (base + u[:, :, None]).logsumexp(-2)
            next_u = loga - (base + next_v[:, None, :]).logsumexp(-1)
            update = base + next_u[:, :, None] + next_v[:, None, :]
        else:
            next_u, _ = _capped_mass_potentials((base + v[:, None, :]).logsumexp(-1), a, mass)
            next_v, next_w = _capped_mass_potentials((base + next_u[:, :, None]).logsumexp(-2), b, mass)
            update = base + next_u[:, :, None] + next_v[:, None, :] + next_w[:, None, None]
        finite = torch.isfinite(update).all((-2, -1))
        counts += active
        numerical |= active & ~finite
        active &= finite
        mask = active[:, None, None]
        lk = torch.where(mask, update, previous)
        u = torch.where(active[:, None], next_u, u)
        v = torch.where(active[:, None], next_v, v)
        if mass != 1.:
            w = torch.where(active, next_w, w)
        if iteration % 10 == 0 or iteration + 1 == config["max_inner_steps"]:
            err = (torch.linalg.vector_norm(lk.exp().sum(-2) - b, dim=-1) if mass == 1.
                   else torch.linalg.vector_norm((previous - lk).flatten(1), dim=-1))
            error = torch.where(active, err, error)
            # Retain the legacy residual as a diagnostic and one stopping path.
            stopped = error < config["inner_tolerance"] if mass == 1. else error <= config["inner_tolerance"]
            reason = torch.ones_like(stop_reason)
            if mass != 1.:
                plan = lk.exp()
                _, gap_error = _duality_gap(plan, lk, base, a, b, mass, u, v, w, config["reg"])
                delta, kkt = _partial_certificate(plan, lk, previous, base, a, b, mass, u, v, w)
                plan_error = torch.where(active, delta, plan_error)
                kkt_error = torch.where(active, kkt, kkt_error)
                # At the budget boundary, certify the convex subproblem rather
                # than reject an optimal plan solely on an unweighted log norm.
                # Ordinary iterations retain the legacy stopping rule. A budget
                # stop alone is NEVER sufficient, nor is feasibility alone.
                certified = ((iteration + 1 == config["max_inner_steps"])
                             & (plan_error <= config["inner_tolerance"])
                             & (kkt_error <= config["inner_tolerance"]))
                reason = torch.where(stopped, 1, 2)
                stopped |= certified
                stopped &= feasible(residual_values(plan, a, b, mass), config) & (gap_error <= config["inner_tolerance"])
            stop_reason = torch.where(active & stopped, reason, stop_reason)
            active &= ~stopped
            if not bool(active.any()):
                break
    plan = lk.exp()
    residuals = residual_values(plan, a, b, mass)
    gap, gap_error = _duality_gap(plan, lk, base, a, b, mass, u, v, w, config["reg"])
    ok = valid & ~numerical & ~active & feasible(residuals, config)
    # 0 success, 1 input/numerical failure, 2 iteration budget, 3 feasibility.
    failure = torch.where(numerical, 1, torch.where(active, 2, torch.where(ok, 0, 3)))
    return plan, dict(ok=ok, error=error, iterations=counts, failure=failure,
        cost_shift=shift, residuals=residuals, duality_gap=gap, relative_duality_gap=gap_error,
        relative_plan_delta_l1=plan_error, kkt_error=kkt_error, stop_reason=stop_reason,
        method="balanced_sinkhorn_log_s_equals_1" if mass == 1. else "partial_capped_mass_log")


def _number(value):
    return value if math.isfinite(value) else None


@torch.no_grad()
def solve_partial_batch(cost, kx, ky, *, a=None, b=None, keep_mass=.8, config=None, on_step=None):
    """Return [B,P,Q] last accepted plans and B independent reports.

    A numerical/iteration failure retires just that pair. ``on_step(plans,
    reports)`` receives None for inactive pairs and full reports for updated
    pairs, at an outer-iteration boundary. It may persist successful pairs
    immediately, even while their neighbors are still running.
    """
    c = solver_config(config)
    if cost.ndim != 3 or min(cost.shape) < 1:
        raise ValueError("Batched costs must have shape [B,P,Q] with nonempty axes.")
    batch, n, m = cost.shape
    if kx.shape != (batch, n, n) or ky.shape != (batch, m, m):
        raise ValueError("Batched kernels must have shapes [B,P,P] and [B,Q,Q].")
    if not 0 < keep_mass <= 1 or kx.device != cost.device or ky.device != cost.device:
        raise ValueError("Require keep_mass in (0,1] and costs/kernels on the same device.")
    cost, kx, ky = (v.detach().double() for v in (cost, kx, ky))
    a = cost.new_full((batch, n), 1/n) if a is None else a.to(cost).expand(batch, n)
    b = cost.new_full((batch, m), 1/m) if b is None else b.to(cost).expand(batch, m)
    if any(not torch.isfinite(w).all() or (w <= 0).any() or ((w.sum(-1) - 1).abs() > 1e-12).any() for w in (a, b)):
        raise ValueError("Each a,b must contain positive probability masses.")
    scales = cost.mean((-2, -1)) if c["cost_scale"] == "mean" else cost.new_full((batch,), c["cost_scale"])
    valid = torch.isfinite(cost).all((-2, -1)) & torch.isfinite(scales) & (scales > 0)
    for kernel in (kx, ky):
        valid &= torch.isfinite(kernel).all((-2, -1)) & (kernel >= 0).all((-2, -1))
    # Quarantine malformed members, so they cannot poison other pairs' autograd.
    cost = torch.where(valid[:, None, None], cost, 0.) / torch.where(valid, scales, 1.)[:, None, None]
    kx, ky = (torch.where(valid[:, None, None], k, 0.) for k in (kx, ky))
    plans = keep_mass * (a[:, :, None] * b[:, None, :])
    statuses = torch.where(valid, RUNNING, INVALID)
    counts = torch.zeros(batch, dtype=torch.int64, device=cost.device)
    reports = [None] * batch
    histories = [[] for _ in range(batch)]
    runtime = dict(capabilities(), batch_solver=VERSION, device=str(cost.device))
    initial = torch.stack((statuses.double(), scales), -1).cpu().tolist()
    for idx, (status, scale) in enumerate(initial):
        if int(status) == INVALID:
            reports[idx] = dict(status=STATUS[INVALID], history=[], cost_scale=_number(scale),
                config=c, keep_mass=keep_mass, objective_version=OBJECTIVE, runtime=runtime,
                iterations=0, error="Nonfinite cost/kernel or invalid fitted cost scale.")
    if on_step and any(r is not None for r in reports):
        on_step(plans, list(reports))
    for iteration in range(1, c["max_outer_steps"] + 1):
        indices = (statuses == RUNNING).nonzero().flatten()
        if not len(indices):
            break
        old_plan, cc, xx, yy, aa, bb = (v[indices] for v in (plans, cost, kx, ky, a, b))
        old = objective_values(old_plan, cc, xx, yy, c)
        gradient = information_gradient(old_plan, xx, yy, c["log_floor"]) if c["lam"] else torch.zeros_like(old_plan)
        healthy = torch.isfinite(old).all(-1) & torch.isfinite(gradient).all((-2, -1))
        effective = cc - c["lam"] * gradient
        candidate, inner = entropy_subproblem_batch(aa, bb, effective, keep_mass, c)
        ok = inner["ok"] & healthy
        delta = (candidate - old_plan).abs().sum((-2, -1))
        local_status = torch.where(healthy, torch.where(ok, RUNNING, INNER_FAILED), NONFINITE)
        stationary = ok & (delta <= c["outer_tolerance"])
        accepted_plan, terms = old_plan.clone(), old.clone()
        steps = delta.new_zeros(len(indices))
        backtracks = torch.zeros_like(local_status)
        if c["lam"] == 0:
            local_status = torch.where(ok, CONVERGED, local_status)
            accepted_plan = torch.where(ok[:, None, None], candidate, old_plan)
            terms = objective_values(accepted_plan, cc, xx, yy, c)
            steps = ok.to(delta.dtype)
        else:
            local_status = torch.where(stationary, CONVERGED, local_status)
            pending = ok & ~stationary
            step = torch.ones_like(delta)
            for attempt in range(c["max_backtracks"]):
                if not bool(pending.any()):
                    break
                trial = old_plan + step[:, None, None] * (candidate - old_plan)
                trial_terms = objective_values(trial, cc, xx, yy, c)
                accepted = pending & torch.isfinite(trial_terms).all(-1) & (trial_terms[:, 3] <= old[:, 3] + c["objective_tolerance"])
                stalled = accepted & (step * delta <= c["outer_tolerance"])
                local_status = torch.where(stalled, STALLED, local_status)
                improved = accepted & ~stalled
                accepted_plan = torch.where(improved[:, None, None], trial, accepted_plan)
                terms = torch.where(improved[:, None], trial_terms, terms)
                steps = torch.where(improved, step, steps)
                pending &= ~accepted
                backtracks += pending
                step = torch.where(pending, step * .5, step)
            local_status = torch.where(pending, STALLED, local_status)
        if iteration == c["max_outer_steps"]:
            local_status = torch.where(local_status == RUNNING, BUDGET, local_status)
        plans[indices] = accepted_plan
        statuses[indices] = local_status
        counts[indices] += 1
        residuals = residual_values(accepted_plan, aa, bb, keep_mass)
        # Pack diagnostics for one device-to-host copy per outer iteration;
        # all objective/gradient/feasibility computation above stays on device.
        packed = torch.cat((indices[:, None], local_status[:, None], counts[indices, None],
            scales[indices, None], steps[:, None], backtracks[:, None], delta[:, None],
            (steps * delta)[:, None], (terms[:, 3]-old[:, 3])[:, None], terms, residuals,
            inner["error"][:, None], inner["iterations"][:, None], inner["failure"][:, None],
            inner["cost_shift"][:, None], inner["residuals"],
            inner["duality_gap"][:, None], inner["relative_duality_gap"][:, None],
            inner["relative_plan_delta_l1"][:, None], inner["kkt_error"][:, None],
            inner["stop_reason"][:, None]), -1).cpu().tolist()
        emitted = [None] * batch
        for row in packed:
            idx, status, count = map(int, row[:3])
            inner_report = dict(method=inner["method"], error=_number(row[17]), iterations=int(row[18]),
                failure_code=int(row[19]), cost_shift=_number(row[20]), warnings=[],
                duality_gap=_number(row[25]), relative_duality_gap=_number(row[26]),
                relative_plan_delta_l1=_number(row[27]), kkt_error=_number(row[28]),
                convergence_reason=("not_converged", "balanced_marginal_residual" if keep_mass == 1.
                    else "legacy_residual_and_gap", "primal_dual_certificate")[int(row[29])],
                error_metric="column_marginal_l2" if keep_mass == 1. else "log_plan_delta_l2",
                **dict(zip(("mass", "mass_error", "row_cap_error", "column_cap_error"), map(_number, row[21:25]))))
            record = dict(iteration=count, status=STATUS[status], step_size=row[4], backtracks=int(row[5]),
                plan_delta_l1=_number(row[6]), accepted_plan_delta_l1=_number(row[7]), objective_delta=_number(row[8]),
                inner=inner_report, **dict(zip(("cost", "information", "entropy", "objective"), map(_number, row[9:13]))))
            histories[idx].append(record)
            reports[idx] = dict(status=STATUS[status], objective_version=OBJECTIVE, cost_scale=row[3], config=c,
                keep_mass=keep_mass, runtime=runtime, iterations=count, history=histories[idx],
                **dict(zip(("mass", "mass_error", "row_cap_error", "column_cap_error"), row[13:17])))
            emitted[idx] = reports[idx]
        if on_step:
            on_step(plans, emitted)
    return plans, reports
