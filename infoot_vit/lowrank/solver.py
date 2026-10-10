"""InfoOT mirror descent using POT's constrained low-rank Dykstra projection.

POT's lowrank_sinkhorn itself is not used: its squared cost and factor entropy
would change this experiment's Euclidean-cost, KDE-MI, plan-entropy objective.
"""
import warnings
import torch
from ot.lowrank import _LR_Dysktra

from .objective import evaluate
from .constraints import partial_project


def residuals(q, r, g, *, partial=False, mass=1.):
    if (q.ndim != 2 or r.ndim != 2 or g.ndim != 1 or q.shape[1] != len(g) or r.shape[1] != len(g)
            or not all(torch.isfinite(t).all() and (t >= 0).all() for t in (q,r,g)) or (g <= 0).any()):
        raise ValueError("Invalid/negative low-rank transport factors or nonpositive g.")
    sq, sr = q.sum(0), r.sum(0)
    error = (lambda t: t.clamp_min(0)) if partial else (lambda t: t.abs())
    return dict(q_row_relative_max=float(error(q.sum(1)*len(q)-1).max()),
        r_row_relative_max=float(error(r.sum(1)*len(r)-1).max()),
        q_column_relative_max=float(((sq-g)/g).abs().max()),
        r_column_relative_max=float(((sr-g)/g).abs().max()),
        plan_row_relative_max=float(error((q@(sr/g))*len(q)-1).max()),
        plan_column_relative_max=float(error((r@(sq/g))*len(r)-1).max()),
        mass_error=abs(float((sq*sr/g).sum())-mass), g_sum_error=abs(float(g.sum())-mass),
        g_min=float(g.min()))


def check(q, r, g, tolerance, min_g, *, partial=False, mass=1.):
    report = residuals(q,r,g,partial=partial,mass=mass)
    if max(v for k,v in report.items() if k != "g_min") > tolerance or report["g_min"] < min_g*(1-tolerance):
        raise ValueError(f"Low-rank constraints did not converge: {report}; tolerance={tolerance}.")
    return report


def constraint_args(config):
    return dict(partial=config.get("constraint") == "partial", mass=config.get("transported_mass", 1.))


@torch.no_grad()
def project(q, r, g, config):
    if config.get("constraint") == "partial":
        out, iterations = partial_project(q,r,g,config)
        return out, dict(check(*out,config["constraint_tolerance"],config["min_g"],**constraint_args(config)),
                         projection_iterations=iterations,warnings=[])
    a, b = q.new_full((len(q),),1/len(q)), r.new_full((len(r),),1/len(r))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _LR_Dysktra(q,r,g,a,b,config["min_g"],config["projection_tolerance"],
                          config["projection_iterations"],True)
    report = check(*out,config["constraint_tolerance"],config["min_g"])
    report["warnings"] = [str(w.message) for w in caught]
    return out, report


@torch.no_grad()
def initialize(n,m,rank,config,device):
    rng = torch.Generator(device=device).manual_seed(config["seed"])
    q = torch.rand(n,rank,generator=rng,dtype=torch.float64,device=device)+.1
    r = torch.rand(m,rank,generator=rng,dtype=torch.float64,device=device)+.1
    q /= q.sum(1,keepdim=True)*n; r /= r.sum(1,keepdim=True)*m
    g = q.new_full((rank,),1/rank)
    return project(q,r,g,config)[0]


def relative_change(old,new):
    return sum(float((a-b).abs().sum()) for a,b in zip(old,new))/3


@torch.no_grad()
def solve(factors,fx,fy,pairs,audit,config,*,start_step=0,on_step=None):
    args = dict(lam=config["lam"],reg=config["reg"],log_floor=config["log_floor"],chunk_size=config["chunk_size"],
                partial=config.get("constraint") == "partial")
    status, history = "max_steps", []
    for step in range(start_step+1,config["max_steps"]+1):
        before, gradients = evaluate(*factors,fx,fy,pairs,gradient=True,**args)
        magnitude = max(float(v.abs().max()) for v in gradients)
        base_step = config["step_size"]/max(1.,magnitude)
        accepted = False
        for backtrack in range(config["max_backtracks"]+1):
            step_size = base_step * .5**backtrack
            proposals = [x*torch.exp(-step_size*grad) for x,grad in zip(factors,gradients)]
            candidate, constraints = project(*proposals,config)
            after = evaluate(*candidate,fx,fy,pairs,**args)
            if after["objective"] <= before["objective"] + config["objective_tolerance"]:
                accepted = True
                break
        if not accepted:
            status = "line_search_failed"
            record = dict(step=step,status=status,**before,gradient_max=magnitude)
            if on_step: on_step(factors,record)
            history.append(record)
            break
        delta = relative_change(factors,candidate)
        # Do not label a tiny backtracked update convergence. Measure the
        # displacement per step, and only certify at the unreduced step size.
        stationarity = delta/max(step_size,1e-300)
        factors = tuple(candidate)
        status = ("converged_sampled_objective" if backtrack == 0 and stationarity <= config["stationarity_tolerance"]
                  else "max_steps")
        record = dict(step=step,status=status,**after,constraints=constraints,factor_delta_l1_mean=delta,
            mirror_step_residual=stationarity,gradient_max=magnitude,step_size=step_size,backtracks=backtrack,
            objective_change=after["objective"]-before["objective"])
        if step % config["audit_every"] == 0 or status.startswith("converged") or step == config["max_steps"]:
            record["audit"] = evaluate(*factors,fx,fy,audit,**args)
            record["audit_objective_gap"] = record["audit"]["objective"]-after["objective"]
        history.append(record)
        if on_step: on_step(factors,record)
        if status.startswith("converged"): break
    return factors, dict(status=status,history=history,last_step=history[-1]["step"] if history else start_step)
