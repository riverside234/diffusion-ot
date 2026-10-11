"""Experimental out-of-sample transport of displacements, not absolute tokens.

For one saved pair let P[i,j] be its unchanged partial plan, K[k,j] the
unchanged target projection kernel, and b the original target proposal.
Integrating the existing target-density correction gives

    t[j] = sum_k K[k,j] b[k] / (K @ b)[k]
    w(q,i,j) proportional to exp(log_query_kernel[q,i]) P[i,j] t[j].

The candidate is q + E_w[y[j] - x[i]]. It keeps the query's local residual
instead of replacing it with a conditional average of complete target tokens.
This is an experimental value estimator, NOT the original InfoOT conditional
expectation of y[k], and NOT a guarantee of native-dog conditioning.
Confidence, support rejection, image routing and saved plans are untouched.
"""
import torch


VALUE_MODES = ("barycentric", "conditional_displacement")


def validate_value_mode(mode):
    if not isinstance(mode, str) or mode not in VALUE_MODES:
        raise ValueError(f"patch_value_mode must be one of {VALUE_MODES}.")
    return mode


def conditional_displacement(query, source, target, plans, logs, target_kernel, b):
    """Batched float64 [B,Q,D] values; never materialize [B,Q,P,P,D].

    Normalize latent plan rows and query weights in log space. Zero retained
    rows have exactly zero weight; an entirely unsupported query returns zero,
    not an unmasked source copy. Caller retains its original confidence gates.
    """
    if (plans.dtype != torch.float64 or any(x.dtype != torch.float64 for x in
            (query, source, target, logs, target_kernel, b))):
        raise ValueError("Conditional displacement requires float64 projection.")
    if (plans.ndim != 3 or source.ndim != 3 or target.ndim != 3 or query.ndim != 3
            or source.shape[:2] != plans.shape[:2]
            or target.shape[:2] != (plans.shape[0], plans.shape[2])
            or query.shape[0] != plans.shape[0] or query.shape[-1] != source.shape[-1]
            or target.shape[-1] != source.shape[-1]
            or logs.shape != (plans.shape[0], query.shape[1], plans.shape[1])
            or target_kernel.shape != (plans.shape[0], plans.shape[2], plans.shape[2])
            or b.shape != target.shape[:2]):
        raise ValueError("Incompatible conditional-displacement tensor shapes.")
    density = (target_kernel @ b[..., None]).squeeze(-1)
    target_importance = (target_kernel.transpose(-1, -2) @ (b / density)[..., None]).squeeze(-1)
    log_edges = plans.log() + target_importance.log()[:, None]
    log_row_mass = log_edges.logsumexp(-1)
    active = torch.isfinite(log_row_mass)
    row_weights = torch.where(active[..., None],
        (log_edges - log_row_mass.masked_fill(~active, 0.)[..., None]).exp(), 0.)
    # A common displacement is reproduced even when K mixes many target tokens.
    row_delta = (row_weights @ target - source).masked_fill(~active[..., None], 0.)
    log_scores = logs + log_row_mass[:, None]
    log_total = log_scores.logsumexp(-1)
    positive = torch.isfinite(log_total)
    weights = torch.where(positive[..., None],
        (log_scores - log_total.masked_fill(~positive, 0.)[..., None]).exp(), 0.)
    delta = weights @ row_delta
    values = (query + delta).masked_fill(~positive[..., None], 0.)
    source_residual = query - weights @ source
    stats = dict(
        displacement_source_effective_patches=(-torch.special.xlogy(weights, weights).sum(-1)).exp(),
        displacement_norm=delta.norm(dim=-1),
        source_residual_norm=source_residual.norm(dim=-1).masked_fill(~positive, 0.),
    )
    # Reject corrupted inputs and numerical failures, never substitute a native
    # reference or silently fall back to the source to manufacture sharpness.
    failures = [(plans < 0).any(), ~torch.isfinite(plans).all(),
        ~torch.isfinite(density).all(), (density <= 0).any(),
        ~torch.isfinite(target_importance).all(), (target_importance <= 0).any(),
        ~torch.isfinite(row_delta).all(), ~torch.isfinite(values).all(),
        torch.isnan(logs).any(), torch.isposinf(logs).any(),
        (b <= 0).any(), (target_kernel < 0).any()]
    failures += [~torch.isfinite(value).all() for value in (*stats.values(), query, source, target)]
    if bool(torch.stack(failures).any()):
        raise ValueError("Nonfinite/invalid conditional-displacement projection.")
    return values, stats
