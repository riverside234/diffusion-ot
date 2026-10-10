"""Positive Gaussian random features; no patch-by-patch kernel allocation.

For z=(x-training_mean)/sigma, exp(w.z-||z||^2), w~N(0,I), is an
unbiased positive feature for the Gaussian kernel. Row L2 normalization is
an explicit finite-rank bias: it fixes diagonal K=1 and bounds K in [0,1].
The norm term cancels under this normalization. See README for error checks.
Input/output SigLIP features are never normalized or projected for conditioning.
"""
import math
import torch


def features(x, state, chunk_size=4096):
    parts = []
    mean, omega = state["mean"].to(x), state["omega"].to(x)
    for rows in x.split(chunk_size):
        logits = ((rows - mean) / state["sigma"]) @ omega
        positive = (logits - logits.max(1, keepdim=True).values).exp()
        parts.append(positive / positive.norm(dim=1, keepdim=True))
    out = torch.cat(parts)
    if not torch.isfinite(out).all() or (out < 0).any():
        raise ValueError("Invalid positive kernel features.")
    return out


def fit_features(x, rank, h, seed, chunk_size=4096, *, omega=None):
    # Exact all-training-pairs RMS bandwidth from population moments:
    # mean_ij ||xi-xj||^2 / 2 = sum_d population_variance_d(x).
    mean = x.mean(0)
    variance = sum((rows - mean).square().sum() for rows in x.split(chunk_size)) / len(x)
    scale = float(variance.sqrt())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate training features for kernel bandwidth.")
    if omega is None:
        omega = torch.randn(x.shape[1], rank, dtype=torch.float64,device=x.device,
                            generator=torch.Generator(device=x.device).manual_seed(seed))
    state = dict(method="normalized_positive_gaussian_v1", mean=mean, omega=omega,
                 scale=scale, h=h, sigma=scale*h, seed=seed, rank=rank)
    return features(x, state, chunk_size), state


def error_report(x, factor, state, *, seed, count=4096, density_queries=32, chunk_size=4096):
    """Independent exact Gaussian pair/density probes, O(count*D + probes*N*D)."""
    rng = torch.Generator(device=x.device).manual_seed(seed)
    i = torch.randint(len(x), (count,), generator=rng,device=x.device)
    j = torch.randint(len(x), (count,), generator=rng,device=x.device)
    exact, approx = [], []
    for ii, jj in zip(i.split(chunk_size), j.split(chunk_size)):
        exact.append((-.5*((x[ii]-x[jj])/state["sigma"]).square().sum(1)).exp())
        approx.append((factor[ii]*factor[jj]).sum(1))
    exact, approx = torch.cat(exact), torch.cat(approx)
    error = approx-exact
    qi = torch.randperm(len(x), generator=rng,device=x.device)[:min(density_queries, len(x))]
    density = x.new_zeros(len(qi))
    for rows in x.split(chunk_size):
        distances = torch.cdist(x[qi], rows, compute_mode="donot_use_mm_for_euclid_dist")
        density += (-.5*(distances/state["sigma"]).square()).exp().sum(1)/len(x)
    approx_density = factor[qi] @ factor.mean(0)
    return dict(seed=seed, pair_count=count, density_query_indices=qi.cpu().tolist(),
        relative_rmse=float(error.square().mean().sqrt()/exact.square().mean().sqrt()),
        mean_absolute_error=float(error.abs().mean()), max_absolute_error=float(error.abs().max()),
        exact_mean=float(exact.mean()), approximate_mean=float(approx.mean()),
        density_relative_error_max=float(((approx_density-density)/density).abs().max()),
        density_relative_error_mean=float(((approx_density-density)/density).abs().mean()),
        diagonal_error_max=float((factor.square().sum(1)-1).abs().max()),
        interpretation="Sampled Gaussian approximation error; normalized positive features are biased at finite rank.")
