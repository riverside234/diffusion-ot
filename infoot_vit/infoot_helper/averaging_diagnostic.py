"""Observe the unchanged edge-mean projection and separate its two averages.

A[j,q,k] = sum_i theta[i,j] g[i,j,q] w[i,j,q,k], where w is the
original-plan target-patch marginal (including the target-density correction).
Normalize A within each dog to obtain beta; normalize dog masses to obtain pi.
Then mu[q] = sum_j pi[j,q] sum_k beta[j,q,k] y[j,k].

The law of total variance partitions conditional target-token dispersion into
within-dog patch dispersion and between-dog mean dispersion. These are NOT
spatial variance of the decoded conditioning map or percentages of image blur.
"""
from __future__ import annotations

import torch


METHOD_TITLES = dict(mixture_edge_mean="All dogs: edge mean",
                    dominant_target_mean="Dominant dog: patch mean",
                    dominant_target_argmax="Dominant dog: patch argmax")


def _nonnegative(value, scale, name):
    if not torch.isfinite(value).all() or (value < -1e-10 * scale.abs().clamp_min(1.)).any():
        raise ValueError(f"Invalid {name} in averaging decomposition.")
    return value.clamp_min(0.)


def _summary(value, mask):
    kept = value[mask]
    return dict(mean=float(kept.mean()), min=float(kept.min()), max=float(kept.max())) if len(kept) else None


def _dispersion_summary(within, between, total, mask):
    result = dict(valid_tokens=int(mask.sum()), within_target=_summary(within, mask),
                  between_targets=_summary(between, mask), total=_summary(total, mask))
    energy = total[mask].sum()
    result["within_target_fraction"] = float(within[mask].sum() / energy) if energy > 0 else None
    result["between_target_fraction"] = float(between[mask].sum() / energy) if energy > 0 else None
    return result


def target_patch_coherence(indices, mask, grid):
    """Target grid steps for adjacent valid query positions; identity has step 1."""
    h, w = grid
    if len(indices) != h * w or mask.shape != indices.shape:
        raise ValueError("Patch selections must preserve the saved query grid.")
    selected, valid = indices.reshape(h, w), mask.reshape(h, w)
    steps = []
    for a, b, active in ((selected[:, :-1], selected[:, 1:], valid[:, :-1] & valid[:, 1:]),
                         (selected[:-1], selected[1:], valid[:-1] & valid[1:])):
        a, b = a[active], b[active]
        steps.append(((a // w - b // w).double().square() + (a % w - b % w).double().square()).sqrt())
    steps = torch.cat(steps)
    chosen = indices[mask]
    counts = chosen.unique(return_counts=True)[1]
    return dict(valid_tokens=len(chosen), unique_target_patches=len(counts),
        duplicate_fraction=1 - len(counts) / len(chosen) if len(chosen) else None,
        most_repeated_patch_fraction=float(counts.max()) / len(chosen) if len(chosen) else None,
        adjacent_valid_pairs=len(steps), adjacent_target_grid_step_mean=float(steps.mean()) if len(steps) else None,
        adjacent_same_target_patch_fraction=float((steps == 0).double().mean()) if len(steps) else None)


class AveragingObserver:
    """One query's edge mass on CPU; bounded target blocks on the mapper device.

    At the reviewed 2,000 x 196 x 196 size the CPU scratch is 586.2 MiB.
    No all-target [J,Q,D] feature maps or [N,J,Q,K] history is retained. The
    observer never alters plans, routes, confidence, masks or projected values.
    """
    def __init__(self, mapper, *, target_block_size=16, max_host_bytes=1024**3):
        if (type(target_block_size) is not int or target_block_size < 1
                or type(max_host_bytes) is not int or max_host_bytes < 1):
            raise ValueError("Positive diagnostic block size and host memory budget are required.")
        self.target = mapper.y
        self.target_ids = list(mapper.target.ids)
        self.grid = list(mapper.target.representation["grid"])
        self.policy = dict(mapper.config["projection"]["confidence"])
        self.block_size, self.max_host_bytes = target_block_size, max_host_bytes
        self.edge_mass = None
        self.records, self.means, self.argmax_maps, self.masks = [], [], [], []
        self.selected_probabilities, self.selected_confidences = [], []
        self.scratch_bytes = 0

    def begin(self, query_id, patches, alpha):
        if self.edge_mass is not None or query_id in {r["query_id"] for r in self.records}:
            raise ValueError("Averaging diagnostic query lifecycle or identity mismatch.")
        if (patches.ndim != 2 or patches.shape != self.target.shape[1:]
                or alpha.shape != (len(self.target),) or not torch.isfinite(alpha).all()
                or (alpha < 0).any() or not torch.isclose(alpha.sum(), alpha.new_tensor(1.), atol=1e-10, rtol=1e-10)):
            raise ValueError("Averaging diagnostic requires the original token grid and normalized dog routes.")
        self.query_id, self.alpha = query_id, alpha.detach().clone()
        alpha_values = alpha.detach().cpu().tolist()
        self.dominant = min(range(len(alpha)), key=lambda j: (-alpha_values[j], self.target_ids[j]))
        self.scratch_bytes = len(self.target) * len(patches) * self.target.shape[1] * 8
        if self.scratch_bytes > self.max_host_bytes:
            raise ValueError(f"Averaging diagnostic needs {self.scratch_bytes} CPU scratch bytes; budget is {self.max_host_bytes}.")
        self.edge_mass = torch.zeros((len(self.target), len(patches), self.target.shape[1]), dtype=torch.float64)

    def add(self, targets, measure, weights):
        if (self.edge_mass is None or weights.dtype != torch.float64 or measure.dtype != torch.float64
                or weights.shape != (len(targets), *self.edge_mass.shape[1:])
                or measure.shape != weights.shape[:2]
                or any(type(j) is not int or not 0 <= j < len(self.target) for j in targets)
                or not torch.isfinite(weights).all() or not torch.isfinite(measure).all()
                or (measure < 0).any() or (weights < 0).any()):
            raise ValueError("Invalid plan-edge mass supplied to the averaging diagnostic.")
        active = measure > 0
        if not torch.allclose(weights.sum(-1)[active], torch.ones_like(measure[active]), atol=2e-9, rtol=2e-9):
            raise ValueError("Supported edge weights must sum to one before dog aggregation.")
        weighted = (measure[..., None] * weights).detach().cpu()
        # Preserve pair order, including repeated targets; no GPU atomic adds.
        for index, target in enumerate(targets):
            self.edge_mass[target].add_(weighted[index])

    def finish(self, mapped, confidence, valid):
        if self.edge_mass is None:
            raise ValueError("No active averaging diagnostic query.")
        mass = self.edge_mass.sum(-1)
        reconstructed_mass = mass.sum(0).to(mapped.device)
        if not torch.allclose(reconstructed_mass, confidence, atol=2e-10, rtol=2e-10):
            raise ValueError("Target-group mass does not reconstruct the unchanged mapping confidence.")
        first = torch.zeros_like(mapped)
        second, within_sum, mean_second = [torch.zeros_like(confidence) for _ in range(3)]
        selected_mean = selected_beta = selected_mass = None
        for start in range(0, len(self.target), self.block_size):
            stop = min(start + self.block_size, len(self.target))
            c = mass[start:stop].to(mapped.device)
            beta = self.edge_mass[start:stop].to(mapped.device) / c.masked_fill(c == 0, 1.)[..., None]
            y = self.target[start:stop]
            means = beta @ y
            energy = (beta @ y.square().mean(-1)[..., None]).squeeze(-1)
            mean_energy = means.square().mean(-1)
            variance = _nonnegative(energy - mean_energy, energy, "within-target variance")
            first += (c[..., None] * means).sum(0)
            second += (c * energy).sum(0)
            within_sum += (c * variance).sum(0)
            mean_second += (c * mean_energy).sum(0)
            if start <= self.dominant < stop:
                index = self.dominant - start
                selected_mean, selected_beta, selected_mass = means[index].clone(), beta[index].clone(), c[index].clone()
        positive = confidence > self.policy["mass_floor"]
        denominator = confidence.masked_fill(~positive, 1.)
        reconstructed = (first / denominator[:, None]).masked_fill(~positive[:, None], 0.)
        if not torch.allclose(reconstructed, mapped, atol=2e-10, rtol=2e-10):
            raise ValueError("Per-dog means do not reconstruct the original conditional edge mean.")
        energy = second / denominator
        within = within_sum / denominator
        between = _nonnegative(mean_second / denominator - reconstructed.square().mean(-1), energy, "between-target variance")
        total = _nonnegative(energy - reconstructed.square().mean(-1), energy, "total variance")
        within, between, total = [x.masked_fill(~positive, 0.) for x in (within, between, total)]
        identity_error = (total - within - between).abs().max()
        if identity_error > 2e-9 * energy.abs().max().clamp_min(1.):
            raise ValueError("Conditional variance decomposition failed its total-variance identity.")
        conditional_confidence = selected_mass / self.alpha[self.dominant]
        if ((conditional_confidence < 0).any() or (conditional_confidence > 1 + 1e-7).any()
                or not torch.isfinite(conditional_confidence).all()):
            raise ValueError("Dominant-dog conditional confidence violates [0,1].")
        conditional_confidence = conditional_confidence.clamp(0, 1)
        common = valid & (selected_mass > self.policy["mass_floor"]) & (conditional_confidence >= self.policy["threshold"])
        indices = selected_beta.argmax(-1)  # Ties: first saved row-major patch index.
        hard = self.target[self.dominant, indices]
        entropy = -torch.special.xlogy(selected_beta, selected_beta).sum(-1)
        share = (selected_mass / denominator).masked_fill(~positive, 0.)
        row = dict(query_id=self.query_id, dominant_target_id=self.target_ids[self.dominant],
            dominant_target_index=self.dominant, image_routing_weight=float(self.alpha[self.dominant]),
            original_valid_tokens=int(valid.sum()), common_valid_tokens=int(common.sum()),
            removed_by_common_mask=int((valid & ~common).sum()), all_invalid_common=not bool(common.any()),
            dominant_conditional_confidence=_summary(conditional_confidence, common),
            dominant_retained_mixture_share=_summary(share, valid),
            dominant_effective_target_patches=_summary(entropy.exp(), common),
            dominant_top1_patch_probability=_summary(selected_beta.max(-1).values, common),
            conditional_dispersion_original_mask=_dispersion_summary(within, between, total, valid),
            conditional_dispersion_common_mask=_dispersion_summary(within, between, total, common),
            reconstruction_max_abs=float((reconstructed - mapped).abs().max()),
            confidence_reconstruction_max_abs=float((reconstructed_mass - confidence).abs().max()),
            variance_identity_max_abs=float(identity_error),
            hard_selection_coherence=target_patch_coherence(indices, common, self.grid),
            common_valid_mask=common.cpu().tolist(),
            selected_patch_indices=indices.masked_fill(~common, -1).cpu().tolist(),
            per_patch=dict(within_target=within.cpu().tolist(), between_targets=between.cpu().tolist(),
                total=total.cpu().tolist(), dominant_conditional_confidence=conditional_confidence.cpu().tolist(),
                dominant_retained_mixture_share=share.cpu().tolist()))
        self.records.append(row)
        self.means.append(selected_mean.masked_fill(~common[:, None], 0.).cpu())
        self.argmax_maps.append(hard.masked_fill(~common[:, None], 0.).cpu())
        self.masks.append(common.cpu())
        self.selected_probabilities.append(selected_beta.cpu())
        self.selected_confidences.append(conditional_confidence.cpu())
        self.edge_mass = None

    def report(self):
        return dict(schema="infoot_averaging_diagnostic_v1", queries=self.records,
            cpu_scratch_bytes_per_query=self.scratch_bytes, cpu_scratch_budget_bytes=self.max_host_bytes,
            target_processing_block_size=self.block_size,
            distribution="A[j,q,k]=sum_i theta[i,j]*g[i,j,q]*w[i,j,q,k]; pi=A.sum(k)/confidence; beta=A/A.sum(k)",
            variance_definition="Per-feature conditional squared Euclidean dispersion of target tokens. "
                "Within=sum_j pi_j Var_beta(y_j); between=Var_pi(E_beta[y_j]); total=within+between. "
                "Separate from within-image conditioning variance; fractions are not fractions of image blur.",
            dominant_target_rule="One dog per query: highest original post-pruning image route, stable ID breaks ties. "
                "Same dog for native reference, target mean and patch argmax.",
            mask_rule="Original valid mask intersected with dominant-dog mass>mass_floor and "
                "dominant-dog conditional confidence>=original threshold. No fallback or confidence bypass.",
            argmax_interpretation="Diagnostic only. Select the largest beta target patch at each valid query position. "
                "No nearest-neighbor retrieval or source residual. May duplicate patches and break spatial coherence; "
                "does not preserve soft-plan marginals or certify translation quality.")
