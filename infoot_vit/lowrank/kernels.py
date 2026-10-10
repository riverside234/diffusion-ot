"""Positive Gaussian kernel features; raw SigLIP conditions stay unchanged.

The legacy row-normalized estimator remains available for old artifacts.
PRF and OPRF retain all Gaussian scale terms, without row normalization.
OPRF uses the training-moment heuristic in Chefs' Random Tables, Eqs. 4-8
(NeurIPS 2022), optionally with Gaussian-marginal orthogonal projections.
Neither finite-rank accuracy nor image quality is guaranteed by unbiasedness.
"""
import math
import torch

LEGACY = "normalized_positive_gaussian_v1"
PRF = "positive_gaussian_v1"
OPRF = "oprf_gaussian_v1"
METHODS = (LEGACY, PRF, OPRF)


def oprf_coefficient(dimension, h):
    # E||z+z'||^2 = 2/h^2 after centering and the training RMS bandwidth.
    # Rationalized Eq. 7 avoids subtracting two nearly equal numbers.
    s = 2 / h**2
    rho = 2*dimension / (math.hypot(2*s+dimension, math.sqrt(8*dimension*s)) + 2*s+dimension)
    return (1 - 1/rho) / 8


def gaussian_projections(dimension, rank, *, seed, device, orthogonal=False):
    rng = torch.Generator(device=device).manual_seed(seed)
    if not orthogonal:
        return torch.randn(dimension, rank, dtype=torch.float64, device=device, generator=rng)
    blocks = []
    for start in range(0, rank, dimension):
        width = min(dimension, rank-start)
        noise = torch.randn(dimension, width, dtype=torch.float64, device=device, generator=rng)
        q, r = torch.linalg.qr(noise, mode="reduced")
        signs = r.diagonal().sign().masked_fill(r.diagonal() == 0, 1.)
        # Haar directions with independent chi_d radii have N(0,I) marginals.
        radii = torch.randn(dimension, width, dtype=torch.float64, device=device, generator=rng).norm(dim=0)
        blocks.append(q * signs * radii)
    return torch.cat(blocks, dim=1)


def features(x, state, chunk_size=4096):
    parts = []
    mean, omega = state["mean"].to(x), state["omega"].to(x)
    method = state["method"]
    if method not in METHODS:
        raise ValueError(f"Unknown kernel feature method: {method}")
    for rows in x.split(chunk_size):
        z = (rows - mean) / state["sigma"]
        logits = z @ omega
        if method == LEGACY:
            positive = (logits - logits.max(1, keepdim=True).values).exp()
            parts.append(positive / positive.norm(dim=1, keepdim=True))
        else:
            a = state["a"]
            log_features = (math.sqrt(1-4*a)*logits - z.square().sum(1, keepdim=True)
                + a*omega.square().sum(0) + x.shape[1]/4*math.log1p(-4*a)
                - .5*math.log(state["rank"]))
            # No row-wise shift/clipping: either would change the target kernel.
            parts.append(log_features.exp())
    out = torch.cat(parts)
    if not torch.isfinite(out).all() or (out < 0).any() or (out.sum(1) <= 0).any():
        raise ValueError("Invalid positive kernel features (overflow/zero row); inspect bandwidth and method.")
    return out


def fit_features(x, rank, h, seed, chunk_size=4096, *, omega=None, method=LEGACY, orthogonal=False):
    if method not in METHODS or not math.isfinite(h) or h <= 0 or rank < 1:
        raise ValueError("Invalid kernel method, bandwidth or rank.")
    if method == LEGACY and orthogonal:
        raise ValueError("Legacy kernel artifacts require their original IID projections.")
    # Exact all-training-pairs RMS bandwidth from population moments:
    # mean_ij ||xi-xj||^2 / 2 = sum_d population_variance_d(x).
    mean = x.mean(0)
    variance = sum((rows - mean).square().sum() for rows in x.split(chunk_size)) / len(x)
    scale = float(variance.sqrt())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate training features for kernel bandwidth.")
    if omega is None:
        omega = gaussian_projections(x.shape[1], rank, seed=seed, device=x.device, orthogonal=orthogonal)
    if omega.shape != (x.shape[1], rank):
        raise ValueError("Kernel projection shape does not match feature dimension/rank.")
    state = dict(method=method, mean=mean, omega=omega,
                 scale=scale, h=h, sigma=scale*h, seed=seed, rank=rank)
    if method != LEGACY:
        state.update(a=oprf_coefficient(x.shape[1], h) if method == OPRF else 0., orthogonal=orthogonal)
    return features(x, state, chunk_size), state


def error_report(x, factor, state, *, seed, count=4096, density_queries=32, chunk_size=4096, storage_roundtrip=False):
    """Independent exact Gaussian pair/density probes, O(count*D + probes*N*D)."""
    rng = torch.Generator(device=x.device).manual_seed(seed)
    i = torch.randint(len(x), (count,), generator=rng,device=x.device)
    j = torch.randint(len(x), (count,), generator=rng,device=x.device)
    convert = (lambda f: f.float().double()) if storage_roundtrip else (lambda f: f)
    exact, approx = [], []
    for ii, jj in zip(i.split(chunk_size), j.split(chunk_size)):
        exact.append((-.5*((x[ii]-x[jj])/state["sigma"]).square().sum(1)).exp())
        approx.append((convert(factor[ii])*convert(factor[jj])).sum(1))
    exact, approx = torch.cat(exact), torch.cat(approx)
    error = approx-exact
    qi = torch.randperm(len(x), generator=rng,device=x.device)[:min(density_queries, len(x))]
    density = x.new_zeros(len(qi))
    for rows in x.split(chunk_size):
        distances = torch.cdist(x[qi], rows, compute_mode="donot_use_mm_for_euclid_dist")
        density += (-.5*(distances/state["sigma"]).square()).exp().sum(1)/len(x)
    factor_sum = x.new_zeros(factor.shape[1])
    diagonal_error = x.new_zeros(())
    for rows in factor.split(chunk_size):
        rows = convert(rows)
        factor_sum += rows.sum(0)
        diagonal_error = torch.maximum(diagonal_error, (rows.square().sum(1)-1).abs().max())
    approx_density = convert(factor[qi]) @ (factor_sum/len(factor))
    if not all(torch.isfinite(v).all() for v in (exact, approx, density, approx_density, diagonal_error)):
        raise ValueError("Nonfinite kernel error audit; float32 factors may overflow.")
    return dict(seed=seed, pair_count=count, density_query_indices=qi.cpu().tolist(),
        relative_rmse=float(error.square().mean().sqrt()/exact.square().mean().sqrt().clamp_min(1e-300)),
        mean_absolute_error=float(error.abs().mean()), max_absolute_error=float(error.abs().max()),
        exact_mean=float(exact.mean()), approximate_mean=float(approx.mean()),
        density_relative_error_max=float(((approx_density-density)/density).abs().max()),
        density_relative_error_mean=float(((approx_density-density)/density).abs().mean()),
        diagonal_error_max=float(diagonal_error), method=state["method"], storage_roundtrip=storage_roundtrip,
        interpretation="Sampled error against the exact training-bandwidth Gaussian kernel; finite rank can be inaccurate.")
