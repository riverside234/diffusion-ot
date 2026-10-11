"""Exact batched projection of saved dense partial plans; never fit at inference."""
from collections import OrderedDict

import torch

from .partial import distance
from .plan_diagnostics import patch_plan_concentration


DEFAULT_CACHE_BYTES = 3 * 1024**3
DEFAULT_PAIR_BATCH_SIZE = 32


def runtime_settings(cache_bytes=None, pair_batch_size=None):
    cache_bytes = DEFAULT_CACHE_BYTES if cache_bytes is None else cache_bytes
    pair_batch_size = DEFAULT_PAIR_BATCH_SIZE if pair_batch_size is None else pair_batch_size
    if type(cache_bytes) is not int or cache_bytes < 0:
        raise ValueError("pair_cache_bytes must be a nonnegative integer (0 disables caching).")
    if type(pair_batch_size) is not int or pair_batch_size < 1:
        raise ValueError("pair_projection_batch_size must be a positive integer.")
    return dict(pair_cache_bytes=cache_bytes, pair_projection_batch_size=pair_batch_size)


def project_pairs(logs, target, plans, a, b, ky, thresholds, tolerances):
    """Vectorize conditional.partial_projection over pairs in float64.

    Original-proposal density correction, zero retention, support gates and
    log-space score retries are unchanged. Host checks occur once per block,
    rather than once per pair. All returned statistics remain on the device.
    """
    row_max = logs.max(-1).values
    centered = logs - torch.where(torch.isfinite(row_max), row_max, 0.)[..., None]
    log_base = torch.logsumexp(centered + a.log()[:, None], dim=-1)
    log_density = log_base + row_max
    has_support = torch.isfinite(log_base)
    posterior = torch.where(has_support[..., None],
        (centered + a.log()[:, None] - log_base[..., None]).exp(), 0.)
    log_kept = torch.logsumexp(centered + plans.sum(-1).log()[:, None], dim=-1)
    raw = torch.where(has_support, (log_kept - log_base).exp(), 0.)
    valid = has_support & (log_density >= thresholds[:, None])
    underflow = has_support & torch.isfinite(log_kept) & (raw == 0)
    density = (ky @ b[..., None]).squeeze(-1)
    smoothing = (plans @ ky.transpose(-2, -1)) * (b / density)[:, None]
    scores = (posterior / a[:, None]) @ smoothing
    totals = scores.sum(-1)
    positive = totals >= 1e-200
    weights = torch.where(positive[..., None], scores / totals.masked_fill(~positive, 1.)[..., None], 0.)
    retry = (~positive) & torch.isfinite(log_kept)
    g = raw * valid
    flags = torch.stack(((underflow & valid).any(),
        (~torch.isfinite(scores) | (scores < 0)).any(),
        ((g < -tolerances[:, None]) | (g > 1 + tolerances[:, None])).any(), retry.any())).cpu().tolist()
    if flags[0]:
        raise ValueError("Retained confidence underflow for a support-valid query; this is not genuine zero retained mass.")
    if flags[1]:
        raise ValueError("Numerical failure in partial conditional scores.")
    if flags[2]:
        raise ValueError("Retained confidence violates [0,1].")
    if flags[3]:
        pair_ids, query_ids = retry.nonzero(as_tuple=True)
        retry_centered = centered[pair_ids, query_ids]
        log_scores = torch.cat([torch.logsumexp(retry_centered[..., None]
            + smoothing[pair_ids, :, start:start + 32].log(), dim=1)
            for start in range(0, target.shape[1], 32)], dim=1)
        if not torch.isfinite(log_scores.logsumexp(1)).all():
            raise ValueError("Conditional score underflow remains after float64 log-space retry.")
        weights[pair_ids, query_ids] = log_scores.softmax(1)
    return weights @ target, raw, valid, dict(
        weight_row_sums=weights.sum(-1),
        patch_entropy=-(weights * weights.clamp_min(1e-300).log()).sum(-1),
        projected_patch_top1_probability=weights.max(-1).values,
        query_source_effective_patches=(-torch.special.xlogy(posterior, posterior).sum(-1)).exp(),
        confidence_underflow=underflow, score_log_retry=retry)


class DensePartialProjector:
    """Bounded device cache of validated stored values plus block-wise projection.

    Cache float32 plans, not float64 smoothing matrices or report histories.
    Legacy float64 plans retain their saved precision. File signatures are
    checked on hits; changed/missing files cannot be served from the cache.
    """
    def __init__(self, mapper, *, cache_bytes=None, pair_batch_size=None):
        self.mapper = mapper
        self.settings = runtime_settings(cache_bytes, pair_batch_size)
        self.cache = OrderedDict()
        self.resident_bytes = self.peak_bytes = self.hits = self.loads = 0
        self.edges = sorted(mapper.pairs, key=lambda key:
            (mapper.source_index[key[0]], mapper.target_index[key[1]]))
        self.indices = torch.tensor([(mapper.source_index[s], mapper.target_index[t])
            for s, t in self.edges], device=mapper.device, dtype=torch.long).reshape(-1, 2)
        self.scales = torch.as_tensor(mapper.shared["sx"], device=mapper.device, dtype=torch.float64)
        self.thresholds = torch.tensor([float("-inf") if s["threshold"] is None else s["threshold"]
            for s in mapper.shared["support"]], device=mapper.device, dtype=torch.float64)

    @staticmethod
    def signature(path):
        try:
            stat = path.stat()
        except OSError as error:
            raise ValueError(f"Missing/changed artifact file: {path}") from error
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino

    def pair(self, key):
        mapper = self.mapper
        path = mapper.directory / mapper.pairs[key]["file"]
        signature = self.signature(path)
        if key in self.cache:
            value, saved_signature, size = self.cache.pop(key)
            if signature == saved_signature:
                self.cache[key] = value, saved_signature, size
                self.hits += 1
                return value
            self.resident_bytes -= size
        state = mapper._read_pair(*key)  # Hash, identity and quantized marginals checked on CPU.
        if self.signature(path) != signature:
            raise ValueError(f"Artifact changed while loading: {path}")
        size = sum(state[name].numel() * state[name].element_size() for name in ("plan", "a", "b"))
        limit = self.settings["pair_cache_bytes"]
        while self.cache and self.resident_bytes + size > limit:
            _, (_, _, evicted_size) = self.cache.popitem(last=False)
            self.resident_bytes -= evicted_size
        # Own compact storage even on CPU: a loaded tensor may be a view into
        # a larger serialized batch, which must not defeat the byte limit.
        value = {name: state[name].to(mapper.device, non_blocking=True, copy=True)
            for name in ("plan", "a", "b")}
        value["tolerance"] = state["storage_validation"]["confidence_roundoff_tolerance"]
        self.loads += 1
        if size <= limit:
            self.cache[key] = value, signature, size
            self.resident_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        return value

    def report(self):
        return dict(self.settings, cache_device=str(self.mapper.device), cached_pairs=len(self.cache),
            cache_resident_bytes=self.resident_bytes, cache_peak_bytes=self.peak_bytes,
            pair_file_loads=self.loads, pair_cache_hits=self.hits)

    def project(self, patches, theta, alpha):
        mapper = self.mapper
        p = len(patches)
        routes = theta[self.indices[:, 0], self.indices[:, 1]]
        # One query-level index transfer; no GPU scalar branching per pair.
        active = (routes != 0).nonzero().flatten().cpu().tolist()
        feature_sum, confidence = torch.zeros_like(patches), patches.new_zeros(p)
        rejected, support_invalid, invalid_routes = (patches.new_zeros(p) for _ in range(3))
        budgets = patches.new_zeros(p, len(mapper.target.ids))
        accounting_error, roundoff, effective = (patches.new_zeros(()) for _ in range(3))
        counts = torch.zeros(2, dtype=torch.int64, device=mapper.device)
        geometry = {}
        size = self.settings["pair_projection_batch_size"]
        for start in range(0, len(active), size):
            ids = active[start:start + size]
            keys = [self.edges[i] for i in ids]
            pairs = [self.pair(key) for key in keys]
            ij = self.indices[ids]
            route = routes[ids, None]
            sources = [mapper.source_index[s] for s, _ in keys]
            unique_sources = list(dict.fromkeys(sources))
            source_lookup = {s: i for i, s in enumerate(unique_sources)}
            inverse = [source_lookup[s] for s in sources]
            logs = -.5 * (distance(patches.expand(len(unique_sources), -1, -1), mapper.x[unique_sources])
                / (self.scales[unique_sources, None, None] * mapper.patch_projection_h)).square()
            if not torch.isfinite(logs).all():
                raise ValueError("Nonfinite query distances/log-kernels.")
            plans, a, b = [torch.stack([pair[name] for pair in pairs]).double() for name in ("plan", "a", "b")]
            tolerances = patches.new_tensor([pair["tolerance"] for pair in pairs])
            candidate, raw, valid, diag = project_pairs(logs[inverse], mapper.y[ij[:, 1]], plans, a, b,
                mapper.projection_ky[ij[:, 1]], self.thresholds[ij[:, 0]], tolerances)
            clipped = raw.clamp(0, 1)
            roundoff = torch.maximum(roundoff, (raw - clipped).abs().max())
            g = clipped * valid  # Validated storage/solver roundoff only, no plan renormalization.
            measure = route * g
            confidence += measure.sum(0)
            feature_sum += (measure[..., None] * candidate).sum(0)
            rejected += (route * (1 - clipped)).sum(0)
            support_invalid += (route * clipped * (~valid)).sum(0)
            invalid_routes += (route * (~valid)).sum(0)
            counts += torch.stack((diag["confidence_underflow"].sum(), diag["score_log_retry"].sum()))
            budget = measure * diag["weight_row_sums"] + route * (1 - g)
            # Reduce duplicate target indices deterministically, without CUDA atomic adds.
            targets = [mapper.target_index[t] for _, t in keys]
            unique_targets = list(dict.fromkeys(targets))
            membership = patches.new_tensor([[float(t == j) for j in unique_targets] for t in targets])
            budgets[:, unique_targets] += budget.T @ membership
            accounting_error = torch.maximum(accounting_error, (budget - route).abs().max())
            effective += (measure * diag["patch_entropy"].exp()).sum()
            for name, value in dict(patch_plan_concentration(plans),
                    within_image_top1_patch_probability=diag["projected_patch_top1_probability"],
                    query_source_effective_patches=diag["query_source_effective_patches"]).items():
                if value.ndim == 1:
                    value = value[:, None]
                geometry[name] = geometry.get(name, 0.) + (measure * value).sum()
        positive = confidence > mapper.config["projection"]["confidence"]["mass_floor"]
        mapped = torch.where(positive[:, None], feature_sum / confidence.masked_fill(~positive, 1.)[:, None], 0.)
        denominator = confidence.sum().clamp_min(torch.finfo(confidence.dtype).tiny)
        # Transfer aggregate diagnostics at the query boundary, not for every pair.
        names = ["matched_rejected_accounting_error", "matched_rejected_group_mass_residual",
            "within_image_effective_patches", "confidence_min", "confidence_mean", "retention_roundoff_clipped_max"]
        values = [accounting_error, (budgets - alpha[None]).abs().max(), effective / denominator,
            confidence.min(), confidence.mean(), roundoff]
        detail = dict(zip(names + list(geometry), torch.stack(values
            + [value / denominator for value in geometry.values()]).cpu().tolist()))
        detail.update(zip(("ot_rejected_mass", "support_invalid_retained_mass", "support_invalid_route_mass"),
            torch.stack((rejected, support_invalid, invalid_routes)).cpu().tolist()))
        detail.update(zip(("support_invalid_confidence_underflow_rows", "score_log_retry_rows"), counts.cpu().tolist()))
        return mapped, confidence, detail
