"""Float64 fixed-support conditional projection, retaining target KDE correction."""
from __future__ import annotations

import math
import torch

from . import infoot as legacy
from .partial import distance, kernel_state, solver_config, entropy_subproblem
from .storage import validate_plan


def normalize_rows(scores):
    if not torch.isfinite(scores).all() or (scores < 0).any() or (scores.sum(1) <= 0).any():
        raise ValueError("Nonfinite/negative/zero-mass conditional score row; no uniform fallback.")
    return scores / scores.sum(1, keepdim=True)


def log_query_kernel(query, support, scale, h):
    if not math.isfinite(h) or h <= 0:
        raise ValueError("Projection bandwidth must be finite and positive.")
    logs = -.5 * (distance(query, support) / (scale * h)).square()
    if not torch.isfinite(logs).all():
        raise ValueError("Nonfinite query distances/log-kernels.")
    return logs


def _balanced_mi_gradient(plan, kx, ky, log_floor):
    """Negative derivative of the saved, floored fixed-marginal MI objective.

    Narrow kernels and low entropy can produce exact zeros in float64. The
    upstream P / f_xy formula then evaluates 0/0, although the floored objective
    has a finite derivative. Below the floor, only the direct P*log(ratio)
    derivative remains; the derivative through the clamped ratio is zero.
    """
    joint = kx @ plan @ ky.T
    ratio = joint / (kx.mean(1)[:, None] * ky.mean(1)[None])
    active = ratio >= log_floor
    quotient = torch.where(active, plan / joint.masked_fill(~active, 1.), 0.)
    gradient = -(ratio.clamp_min(log_floor).log() + kx.T @ quotient @ ky)
    return gradient, int((~active).sum())


class BalancedModel:
    """Saved local FusedInfoOT plan; projection never invokes a solver."""
    def __init__(self, source, target, state, bandwidth_multiplier=1.):
        self.source, self.target = source.detach().double(), target.detach().double()
        self.state = state
        self.plan = state["plan"].to(self.source)
        n, m = len(source), len(target)
        if state.get("storage"):
            self.storage_validation = validate_plan(state, state["plan"].new_full((n,), 1/n,dtype=torch.float64), state["plan"].new_full((m,), 1/m,dtype=torch.float64),
                                                    1., state["config"], balanced=True)
        else:
            tol = state["config"]["feasibility_tolerance"]
            if (self.plan.shape != (n, m) or not torch.isfinite(self.plan).all() or (self.plan < 0).any()
                    or (self.plan.sum(1) - 1 / n).abs().max() > tol
                    or (self.plan.sum(0) - 1 / m).abs().max() > tol):
                raise ValueError("Invalid balanced plan/marginals. Refit; no post-hoc row/column repair.")
        self.h = state["config"]["h"] * bandwidth_multiplier
        if not math.isfinite(self.h) or self.h <= 0:
            raise ValueError("Invalid projection bandwidth.")
        self.ky = torch.exp(-.5 * (distance(self.target, self.target) / (state["target_scale"] * self.h)).square())
        self.fy = self.ky.mean(1)
        # A is proportional to the conditional ratio (query-only factor cancels).
        self.smoothing = (self.plan @ self.ky.T) / self.fy[None, :]

    @classmethod
    def fit(cls, source, target, options=None, on_step=None):
        c = solver_config(options)
        source, target = source.detach().double(), target.detach().double()
        _, sx = kernel_state(source, c["h"])
        _, sy = kernel_state(target, c["h"])
        # Reuse local FusedInfoOT geometry/kernels and its fixed-marginal MI.
        solver = legacy.FusedInfoOT(source, target, h=c["h"], lam=c["lam"], reg=c["reg"])
        scale = float(solver.C.mean()) if c["cost_scale"] == "mean" else float(c["cost_scale"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid balanced cost scale.")
        cost = solver.C / scale
        a, b = source.new_full((len(source),), 1 / len(source)), target.new_full((len(target),), 1 / len(target))
        plan, history, status = torch.outer(a, b), [], "max_outer_steps"
        for iteration in range(1, c["max_outer_steps"] + 1):
            floored = 0
            effective = cost
            if c["lam"] != 0:
                gradient, floored = _balanced_mi_gradient(plan, solver.Ks, solver.Kt, c["log_floor"])
                effective = cost + c["lam"] * gradient
            candidate, inner = entropy_subproblem(a, b, effective, 1., c)
            delta = float((candidate - plan).abs().sum())
            plan = candidate
            loss, terms = legacy.fitting_loss(plan, solver.Ks, solver.Kt, c["reg"], C=cost,
                                               mi_weight=c["lam"], eps=c["log_floor"], return_terms=True)
            record = dict(iteration=iteration, objective=float(loss), cost=float(terms[0]), mi_term=float(terms[1]),
                          entropy_term=float(terms[2]), plan_delta_l1=delta, inner=inner,
                          gradient_log_floor_entries=floored)
            history.append(record)
            if on_step:
                on_step(plan, record)
            if c["lam"] == 0 or delta <= c["outer_tolerance"]:
                status = "converged"
                break
        state = dict(plan=plan.cpu(), config=c, source_scale=sx, target_scale=sy, cost_scale=scale,
                     status=status, history=history, solver="local_FusedInfoOT_checked_fixed_point_v1",
                     distance="euclidean", source_count=len(source), target_count=len(target),
                     row_residual=float((plan.sum(1) - a).abs().max()), column_residual=float((plan.sum(0) - b).abs().max()))
        return cls(source, target, state)

    def query_kernel(self, query):
        return log_query_kernel(query.to(self.source), self.source, self.state["source_scale"], self.h)

    def scores(self, query, columns=None):
        smoothing = self.smoothing if columns is None else self.smoothing[:, columns]
        return self.query_kernel(query).softmax(1) @ smoothing

    def conditional_weights(self, query):
        return normalize_rows(self.scores(query))

    def pair_weights(self, query):
        theta = self.query_kernel(query).softmax(1)[:, :, None] * self.smoothing[None]
        denominator = theta.sum((1, 2), keepdim=True)
        if (denominator <= 0).any() or not torch.isfinite(theta).all():
            raise ValueError("Invalid image-pair routing scores.")
        return theta / denominator

    def project(self, query):
        return self.conditional_weights(query) @ self.target


def calibrate_support(source, scale, h, confidence):
    policy = confidence["support_calibration"]
    if policy == "disabled":
        return dict(policy=policy, threshold=None)
    if policy == "fixed":
        value = float(confidence["support_log_threshold"])
        if not math.isfinite(value):
            raise ValueError("Fixed support_log_threshold must be finite.")
        return dict(policy=policy, threshold=value)
    if policy != "fit_leave_one_out_log_density" or len(source) < 2:
        raise ValueError("Support calibration requires >=2 training patches, or an explicit fixed threshold.")
    logs = log_query_kernel(source, source, scale, h)
    logs.fill_diagonal_(-torch.inf)
    loo = logs.logsumexp(1) - math.log(len(source) - 1)
    if not torch.isfinite(loo).all():
        raise ValueError("Degenerate leave-one-out support density; use an explicit fixed threshold.")
    q = confidence["support_quantile"]
    if not 0 <= q <= 1:
        raise ValueError("support_quantile must be in [0,1].")
    return dict(policy=policy, threshold=float(torch.quantile(loo, q)), quantile=q,
                normalization="mean over P-1 nonself training kernels", heuristic=True)


def partial_projection(query, source, target, plan, a, b, source_scale, target_scale, h, support,
                       *, target_kernel=None, query_logs=None):
    """Original-proposal importance weights and separate retained-mass confidence."""
    logs = log_query_kernel(query, source, source_scale, h) if query_logs is None else query_logs
    # Normalize wrt the ORIGINAL source measure; keeps nonuniform a explicit.
    row_max = logs.max(1).values
    centered = logs - torch.where(torch.isfinite(row_max), row_max, 0.)[:, None]
    log_base_centered = torch.logsumexp(centered + a.log()[None], dim=1)
    log_density = log_base_centered + row_max
    has_support = torch.isfinite(log_base_centered)
    posterior = torch.zeros_like(logs)
    posterior[has_support] = (centered[has_support] + a.log()[None] - log_base_centered[has_support, None]).exp()
    retained = plan.sum(1)
    log_kept = torch.logsumexp(centered + retained.log()[None], dim=1)
    raw_confidence = torch.zeros_like(log_density)
    raw_confidence[has_support] = (log_kept[has_support] - log_base_centered[has_support]).exp()
    support_valid = (torch.ones_like(raw_confidence, dtype=torch.bool) if support["threshold"] is None
                     else log_density >= support["threshold"])
    support_valid &= has_support
    confidence_underflow = has_support & torch.isfinite(log_kept) & (raw_confidence == 0)
    if (confidence_underflow & support_valid).any():
        raise ValueError("Retained confidence underflow for a support-valid query; this is not genuine zero retained mass.")
    ky = (torch.exp(-.5 * (distance(target, target) / (h * target_scale)).square())
          if target_kernel is None else target_kernel)
    density = ky @ b  # Original target proposal, NOT transported c/s.
    smoothing = (plan @ ky.T) * (b / density)[None]
    scores = (posterior / a[None]) @ smoothing
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Numerical failure in partial conditional scores.")
    totals = scores.sum(1)
    weights = torch.zeros_like(scores)
    positive = totals >= 1e-200
    weights[positive] = scores[positive] / totals[positive, None]
    # Retry tiny score rows in log space, including queries whose nearest
    # source token is fully rejected. Compact-kernel true zeros stay masked.
    retry = (~positive) & torch.isfinite(log_kept)
    if retry.any():
        log_scores = []
        for start in range(0, len(target), 32):
            log_scores.append(torch.logsumexp(centered[retry, :, None]
                + smoothing[:, start:start + 32].log()[None], dim=1))
        log_scores = torch.cat(log_scores, dim=1)
        if not torch.isfinite(log_scores.logsumexp(1)).all():
            raise ValueError("Conditional score underflow remains after float64 log-space retry.")
        weights[retry] = log_scores.softmax(1)
    g = raw_confidence * support_valid
    return weights @ target, g, dict(raw_confidence=raw_confidence, support_valid=support_valid,
        log_density=log_density, weights=weights, confidence_underflow=confidence_underflow,
        score_log_retry=retry)
