"""Read-only diagnostics for the PDAE-only experiment; no learned evaluator.

RGB/edge correspondence is deliberately separate from encoder retrieval. It
can expose disagreement with the source, but cannot establish target species
identity or realism. These measurements never contribute training gradients.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def image_correspondence(generated, originals, *, pooled_size=16):
    """Paired appearance/layout proxies with ALL mismatched sources as control.

    Retrieval uses negative pooled RGB MSE and shares ties equally. Constant
    outputs therefore score chance, even if every argmax points at index zero.
    Edge correlation is undefined for a flat image, and is reported as null.
    """
    if (generated.shape != originals.shape or generated.ndim != 4
            or generated.shape[1] != 3 or not len(generated)):
        raise ValueError("Correspondence needs equally shaped, nonempty RGB batches.")
    if not isinstance(pooled_size, int) or isinstance(pooled_size, bool) or pooled_size < 2:
        raise ValueError("pooled_size must be an integer >= 2.")
    generated = generated.detach().float()
    originals = originals.detach().to(device=generated.device, dtype=torch.float32)
    if not torch.isfinite(generated).all() or not torch.isfinite(originals).all():
        raise ValueError("Image diagnostics require finite RGB.")
    size = min(pooled_size, *generated.shape[-2:])
    if size < 2:
        raise ValueError("Image diagnostics need at least 2x2 pixels.")
    a = F.adaptive_avg_pool2d(generated, size)
    b = F.adaptive_avg_pool2d(originals, size)
    distances = (a[:, None] - b[None]).square().flatten(2).mean(2)
    paired = distances.diagonal()
    n = len(a)
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=a.device)
    mismatched = distances[off_diagonal].mean() if n > 1 else None
    ties = torch.isclose(distances, distances.min(1, keepdim=True).values, atol=1e-8, rtol=0)

    def edges(rgb):
        gray = (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
        dx, dy = gray[:, :-1, 1:] - gray[:, :-1, :-1], gray[:, 1:, :-1] - gray[:, :-1, :-1]
        return torch.sqrt(dx.square() + dy.square()).flatten(1)

    edge_a, edge_b = edges(a), edges(b)
    centered_a = edge_a - edge_a.mean(1, keepdim=True)
    centered_b = edge_b - edge_b.mean(1, keepdim=True)
    valid = (centered_a.norm(dim=1) > 1e-8) & (centered_b.norm(dim=1) > 1e-8)
    output_var, source_var = a.var(0, unbiased=False).mean(), b.var(0, unbiased=False).mean()
    return {
        "samples": n, "pooled_size": size,
        "paired_pooled_rgb_mse": float(paired.mean()),
        "per_image_pooled_rgb_mse": paired.cpu().tolist(),
        "mismatched_pooled_rgb_mse": float(mismatched) if mismatched is not None else None,
        "paired_advantage_pooled_rgb_mse": float(mismatched - paired.mean()) if n > 1 else None,
        "pooled_rgb_source_retrieval_top1": float((ties.diagonal() / ties.sum(1)).mean()) if n > 1 else None,
        "pooled_rgb_source_retrieval_chance": 1. / n if n > 1 else None,
        "mean_rgb_l1": float((generated.mean((2, 3)) - originals.mean((2, 3))).abs().mean()),
        "edge_correlation": float(F.cosine_similarity(centered_a[valid], centered_b[valid], dim=1).mean())
            if valid.any() else None,
        "edge_correlation_valid_samples": int(valid.sum()),
        "output_pooled_rgb_variance": float(output_var),
        "source_pooled_rgb_variance": float(source_var),
        "output_to_source_pooled_rgb_variance_ratio": float(output_var / source_var)
            if source_var > 1e-12 else None,
        "interpretation": "Unlearned source appearance/layout proxies, not target realism; source copying can score well and legitimate species changes can alter them.",
    }


@torch.no_grad()
def native_rgb_reconstruction(generated, originals):
    """Native reconstruction vs original RGB (not the Stage 1A prediction)."""
    if generated.shape != originals.shape or generated.ndim != 4 or not len(generated):
        raise ValueError("Native reconstruction needs equally shaped nonempty image batches.")
    mse = (generated.detach().float() - originals.detach().to(device=generated.device, dtype=torch.float32)).square().flatten(1).mean(1)
    # Keep JSON finite for exact synthetic reconstructions; explicitly report
    # the cap so 120 dB is not mistaken for an uncapped PSNR observation.
    psnr = -10 * mse.clamp_min(1e-12).log10()
    return {
        "samples": len(mse), "target": "original_rgb",
        "pixel_mse": float(mse.mean()), "pixel_psnr_db": float(psnr.mean()),
        "per_image_pixel_mse": mse.cpu().tolist(), "per_image_pixel_psnr_db": psnr.cpu().tolist(),
        "psnr_cap_db": 120.,
    }


def solver_health(solution, options):
    """Residual-to-tolerance and iteration-budget diagnostics, never acceptance rules."""
    outer_tolerance = float(options["outer_tolerance"])
    projection_tolerance = float(options["projection_tolerance"])
    base, recovery = int(options["inner_iterations"]), int(options["recovery_iterations"])
    return {
        "outer_tolerance": outer_tolerance,
        "outer_delta_to_tolerance": solution.plan_delta_l1 / outer_tolerance if outer_tolerance > 0 else None,
        "marginal_residual_to_tolerance": max(solution.row_residual, solution.column_residual) / projection_tolerance
            if projection_tolerance > 0 else None,
        "recovery_used": solution.recovery_iterations > 0,
        "outer_total_budget_fraction": solution.iterations / max(base + recovery, 1),
        "recovery_budget_fraction": solution.recovery_iterations / recovery if recovery > 0 else None,
        "unconverged_inner_steps": solution.unconverged_inner_steps,
    }
