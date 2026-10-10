"""Bounded exact-Gaussian diagnostics, never the production transport solver.

Use sampled TRAIN image/patch IDs, fitted full-training bandwidths, and one
fixed nonuniform balanced probe plan to isolate kernel error from optimization.
The probe queries are different training patches, not held-out validation data.
"""
import math
import torch

from . import kernels


def acceptance(checks, config):
    limits = dict(relative_rmse=config["max_relative_rmse"],
                  density_relative_error_mean=config["max_density_relative_error_mean"],
                  density_relative_error_max=config["max_density_relative_error_max"])
    failures = []
    for domain, check in checks.items():
        versions = {"float64": check}
        if "float32_roundtrip" in check:
            versions["float32_roundtrip"] = check["float32_roundtrip"]
        for precision, report in versions.items():
            for metric, limit in limits.items():
                value = report[metric]
                if not math.isfinite(value) or value > limit:
                    failures.append(dict(domain=domain, precision=precision, metric=metric,
                                         value=value, limit=limit))
    return dict(accepted=not failures, policy=config["error_policy"], limits=limits, failures=failures,
                interpretation="Engineering acceptance thresholds, not guarantees of image quality.")


def gaussian(x, y, sigma):
    return (-.5*(torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")/sigma).square()).exp()


def _relative_error(exact, approximate):
    return float((approximate-exact).norm()/exact.norm().clamp_min(1e-300))


def _cosine(a, b):
    denominator = a.norm()*b.norm()
    # Undefined directions are reported as null, not as agreement.
    return None if float(denominator) < 1e-24 else float((a*b).sum()/denominator)


def compare(plan, cost, kx, ky, query_kernel, approx_kx, approx_ky, approx_query,
            target, target_group, *, lam, reg, log_floor=1e-300):
    """Dense reference on a bounded subset with the SAME plan/cost in both arms."""
    if max(plan.shape) > 512 or len(query_kernel) > 128:
        raise ValueError("Exact Gaussian audit is limited to 512 support patches and 128 queries.")
    results = []
    for source_kernel, target_kernel, query in ((kx, ky, query_kernel), (approx_kx, approx_ky, approx_query)):
        with torch.enable_grad():
            p = plan.detach().clone().requires_grad_(True)
            density = source_kernel.mean(1)[:, None]*target_kernel.mean(1)[None, :]
            mi = (p*((source_kernel@p@target_kernel.T)/density).clamp_min(log_floor).log()).sum()
            entropy = (p*(p.clamp_min(log_floor).log()-1)).sum()
            transport_cost = (p*cost).sum()
            objective = transport_cost-lam*mi+reg*entropy
            mi_gradient = torch.autograd.grad(mi, p, retain_graph=True)[0]
            gradient = torch.autograd.grad(objective, p)[0]
        scores = (query@plan@target_kernel.T)/target_kernel.mean(0)
        weights = torch.zeros_like(scores)
        for group in target_group.unique():
            mask = target_group == group
            total = scores[:, mask].sum(1, keepdim=True)
            if (total <= 0).any():
                raise ValueError("Zero conditional density in the exact Gaussian audit.")
            weights[:, mask] = scores[:, mask]/total
        # Each target image gets equal router weight in this diagnostic.
        mapped = weights@target/len(target_group.unique())
        if not all(torch.isfinite(t).all() for t in (mi, objective, mi_gradient, gradient, weights, mapped)):
            raise ValueError("Nonfinite exact Gaussian reference comparison.")
        tangent = gradient-gradient.mean(0)-gradient.mean(1, keepdim=True)+gradient.mean()
        results.append(dict(mi=mi.detach(), cost=transport_cost.detach(), entropy=entropy.detach(),
                            objective=objective.detach(), mi_gradient=mi_gradient,
                            gradient=gradient, tangent=tangent, weights=weights, mapped=mapped))
    exact, approx = results
    group_count = len(target_group.unique())
    return dict(
        exact={k: float(exact[k]) for k in ("mi", "cost", "entropy", "objective")},
        approximate={k: float(approx[k]) for k in ("mi", "cost", "entropy", "objective")},
        mi_absolute_error=float((approx["mi"]-exact["mi"]).abs()),
        mi_gradient_relative_error=_relative_error(exact["mi_gradient"], approx["mi_gradient"]),
        mi_gradient_cosine=_cosine(exact["mi_gradient"], approx["mi_gradient"]),
        objective_gradient_relative_error=_relative_error(exact["gradient"], approx["gradient"]),
        objective_gradient_cosine=_cosine(exact["gradient"], approx["gradient"]),
        feasible_gradient_cosine=_cosine(exact["tangent"], approx["tangent"]),
        mean_within_image_weight_tv=float((approx["weights"]-exact["weights"]).abs().sum(1).mean()/(2*group_count)),
        mapped_feature_relative_error=_relative_error(exact["mapped"], approx["mapped"]),
        exact_mapped_feature_variance=float(exact["mapped"].var(0, unbiased=False).sum()),
        approximate_mapped_feature_variance=float(approx["mapped"].var(0, unbiased=False).sum()))


def _subset(images, image_count, patches_per_image, rng):
    image_ids = torch.randperm(len(images), generator=rng, device=images.device)[:image_count]
    indices, groups = [], []
    for i, image_id in enumerate(image_ids):
        patches = torch.randperm(images.shape[1], generator=rng, device=images.device)[:patches_per_image]
        indices.append(image_id*images.shape[1]+patches)
        groups.append(torch.full_like(patches, i))
    return torch.cat(indices), torch.cat(groups)


@torch.no_grad()
def reference_report(x, y, fx, fy, sx, sy, config, optimizer):
    seed = config["seed"]+1000
    rng = torch.Generator(device=x.device).manual_seed(seed)
    ix, _ = _subset(x, min(config["reference_images"], len(x)),
                    min(config["reference_patches_per_image"], x.shape[1]), rng)
    iy, groups = _subset(y, min(config["reference_images"], len(y)),
                        min(config["reference_patches_per_image"], y.shape[1]), rng)
    flatx, flaty = x.flatten(0, 1), y.flatten(0, 1)
    candidates = torch.ones(len(flatx), dtype=torch.bool, device=x.device)
    candidates[ix] = False
    candidates = candidates.nonzero().flatten()
    if not len(candidates):
        return dict(status="unavailable", reason="No distinct training query patches outside the reference subset.", seed=seed)
    iq = candidates[torch.randperm(len(candidates), generator=rng, device=x.device)[:config["reference_queries"]]]
    xs, ys, query = flatx[ix], flaty[iy], flatx[iq]
    # Float32 saved support factors + float64 query computation mirror inference.
    ax, ay, aq = fx[ix].float().double(), fy[iy].float().double(), kernels.features(query, sx)
    n, m = len(xs), len(ys)
    a = torch.arange(n, dtype=x.dtype, device=x.device)/n
    b = torch.arange(m, dtype=x.dtype, device=x.device)/m
    # Randomly permuted interval-overlap coupling has exact uniform marginals.
    # Mixing uniform mass keeps entropy/gradients away from structural zeros.
    plan = (torch.minimum(a[:, None]+1/n, b[None, :]+1/m)-torch.maximum(a[:, None], b[None, :])).clamp_min(0)
    plan = plan[torch.randperm(n, generator=rng, device=x.device)][:, torch.randperm(m, generator=rng, device=x.device)]
    plan = .8*plan + .2/(n*m)
    cost = torch.cdist(xs, ys, compute_mode="donot_use_mm_for_euclid_dist")
    cost_scale = float(cost.mean())
    if cost_scale <= 0:
        raise ValueError("Degenerate cost in Gaussian reference audit.")
    metrics = compare(plan, cost/cost_scale,
        gaussian(xs, xs, sx["sigma"]), gaussian(ys, ys, sy["sigma"]), gaussian(query, xs, sx["sigma"]),
        ax@ax.T, ay@ay.T, aq@ax.T, ys, groups, lam=optimizer["lam"], reg=optimizer["reg"], log_floor=optimizer["log_floor"])
    return dict(status="completed", seed=seed, source_flat_indices=ix.cpu().tolist(), target_flat_indices=iy.cpu().tolist(),
        query_flat_indices=iq.cpu().tolist(), target_groups=groups.cpu().tolist(), cost_scale=cost_scale,
        source_sigma=sx["sigma"], target_sigma=sy["sigma"], metrics=metrics,
        plan_row_residual=float((plan.sum(1)-1/n).abs().max()), plan_column_residual=float((plan.sum(0)-1/m).abs().max()),
        interpretation="Training subset, fixed feasible probe plan, exact sums, fitted full-training bandwidths; "
                       "within-image normalization and target-density correction. Diagnostic only, not a fitted-plan or image-quality claim.")
