"""Saved-state projection only: balanced controls and confidence-aware pair maps."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .conditional import BalancedModel, normalize_rows, partial_projection, log_query_kernel
from .feature_bank import FeatureBank, compatible_banks, checked_file, digest, file_hash, save_tensor, write_json, validate_maps
from .partial import feasibility, distance, OBJECTIVE


@dataclass
class MappingResult:
    mapped_features: torch.Tensor
    match_confidence: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: dict

    def conditioning(self, query_ids, *, all_invalid_policy="error"):
        """Existing PDAE mask convention is the inverse of mapper validity."""
        if (self.mapped_features.ndim != 3 or not self.mapped_features.is_floating_point()
                or len(query_ids) != len(self.mapped_features)
                or min(self.mapped_features.shape) < 1
                or len(set(query_ids)) != len(query_ids)
                or any(not isinstance(i, str) or not i for i in query_ids)
                or self.valid_mask.shape != self.mapped_features.shape[:2] or self.valid_mask.dtype != torch.bool
                or self.match_confidence.shape != self.valid_mask.shape
                or not self.match_confidence.is_floating_point()
                or not torch.isfinite(self.mapped_features).all() or not torch.isfinite(self.match_confidence).all()
                or (self.match_confidence < 0).any() or (self.match_confidence > 1 + 1e-7).any()):
            raise ValueError("Invalid mapped-feature mask.")
        invalid = ~self.valid_mask.any(1)
        if invalid.any() and all_invalid_policy == "error":
            ids = [i for i, bad in zip(query_ids, invalid.tolist()) if bad]
            raise ValueError(f"All condition tokens rejected for query IDs {ids}. Inspect support/confidence; explicit bypass is available.")
        if all_invalid_policy not in {"error", "bypass"}:
            raise ValueError("Unknown all-invalid policy.")
        return self.mapped_features, ~self.valid_mask


def select_images(alpha, target_ids, query_id, settings):
    # Stable-ID ordering resolves ties independently of support array order.
    order = sorted(range(len(alpha)), key=lambda j: (-float(alpha[j]), target_ids[j]))
    k = settings["top_k_images"]
    kept = order if k is None else order[:k]
    retained = float(alpha[kept].sum())
    weights = torch.zeros_like(alpha)
    weights[kept] = alpha[kept] / retained
    if settings["selection"] == "argmax":
        weights.zero_(); weights[order[0]] = 1
    elif settings["selection"] == "sample":
        seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{query_id}:0".encode()).digest()[:8], "big") % (2**63 - 1)
        # Draw in stable-ID order, not physical support order.
        stable = sorted(range(len(alpha)), key=lambda j: target_ids[j])
        chosen = stable[int(torch.multinomial(weights[stable].cpu(), 1, generator=torch.Generator().manual_seed(seed)))]
        weights.zero_(); weights[chosen] = 1
    return weights, retained


class FeatureMapper:
    def __init__(self, directory, manifest, source, target):
        self.directory, self.manifest = Path(directory), manifest
        self.source, self.target = source, target
        self.source_index = {sid: i for i, sid in enumerate(source.ids)}
        self.target_index = {tid: j for j, tid in enumerate(target.ids)}
        self.config, self.mode = manifest["config"], manifest["config"]["mode"]
        self.x, self.y = source.features.double(), target.features.double()
        self.image = self.patch = None
        multiplier = self.config["projection"]["bandwidth_multiplier"]
        self.patch_groups = {}
        required = ({"patch"} if self.mode == "patch_global" else {"image", "patch"}
                    if self.mode == "grouped_patch" else {"image"})
        if set(manifest["models"]) != required:
            raise ValueError("Missing or unexpected fitted models for the selected mapping mode.")
        for name, entry in manifest["models"].items():
            state = torch.load(checked_file(directory, entry), map_location="cpu", weights_only=True)
            if state["status"] != "converged":
                raise ValueError("Cannot project with a nonconverged saved model.")
            si, ti = state["source_ids"], state["target_ids"]
            if (len(si) != len(set(si)) or len(ti) != len(set(ti))
                    or set(si) != set(source.ids) or set(ti) != set(target.ids)):
                raise ValueError("Saved support ID groups are missing, duplicate, or stale.")
            xs = self.x[[self.source_index[i] for i in si]]
            ys = self.y[[self.target_index[i] for i in ti]]
            if name == "image":
                if si != source.ids or ti != target.ids:
                    raise ValueError("Image-router order must match the mapping manifest.")
                self.image = BalancedModel(xs.reshape(len(xs), -1), ys.reshape(len(ys), -1), state, multiplier)
            elif name == "patch":
                self.patch = BalancedModel(xs.reshape(-1, xs.shape[-1]), ys.reshape(-1, ys.shape[-1]), state, multiplier)
                p = ys.shape[1]
                self.patch_groups = {tid: list(range(j * p, (j + 1) * p)) for j, tid in enumerate(ti)}
        self.pairs = {}
        if self.mode == "grouped_partial":
            index = checked_file(directory, manifest["pair_inventory"])
            entries = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
            for entry in entries:
                key = (entry["source_id"], entry["target_id"])
                if key in self.pairs:
                    raise ValueError("Duplicate pair in full pair inventory.")
                if (entry["source_index"] != self.source_index.get(key[0])
                        or entry["target_index"] != self.target_index.get(key[1])):
                    raise ValueError("Pair inventory indices do not match its stable IDs.")
                checked_file(directory, entry)  # Verify every referenced shard before using the mapper.
                self.pairs[key] = entry
            expected = {(s, t) for s in source.ids for t in target.ids}
            if set(self.pairs) != expected or manifest["pair_count"] != len(expected):
                raise ValueError("Incomplete full pair bank; resume offline fitting. Projection never fits missing pairs.")
            self.shared = torch.load(checked_file(directory, manifest["pair_kernels"]), weights_only=True)
            if multiplier == 1:
                self.projection_ky = self.shared["ky"]
            else:
                self.projection_ky = torch.stack([
                    torch.exp(-.5 * (distance(y, y) / (self.shared["sy"][j] * self.shared["h_projection"])).square())
                    for j, y in enumerate(self.y)])

    @classmethod
    def load(cls, directory):
        directory = Path(directory).resolve()
        m = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (m.get("schema") != "siglip_infoot_mapping_v2" or m.get("status") != "complete"
                or digest({k: v for k, v in m.items() if k != "artifact_id"}) != m.get("artifact_id")):
            raise ValueError("Mapping artifact is incomplete, legacy, or its manifest fingerprint changed.")
        banks = [FeatureBank.load(directory / m[key]["path"]) for key in ("source_bank", "target_bank")]
        for key, bank in zip(("source_bank", "target_bank"), banks):
            if m[key]["artifact_id"] != bank.artifact_id or m[key.replace("_bank", "_ids")] != bank.ids:
                raise ValueError("Referenced bank identity or ordered IDs changed.")
        compatible_banks(*banks)
        return cls(directory, m, *banks)

    def _pair(self, sid, tid):
        entry = self.pairs[sid, tid]
        state = torch.load(self.directory / entry["file"], weights_only=True)
        if (state["source_id"] != sid or state["target_id"] != tid
                or state["fit_fingerprint"] != self.manifest["fit_fingerprint"]
                or state["report"]["status"] != "converged" or state["report"]["objective_version"] != OBJECTIVE
                or state["source_index"] != self.source_index[sid]
                or state["target_index"] != self.target_index[tid]):
            raise ValueError(f"Pair identity/status mismatch: {sid}/{tid}")
        feasibility(state["plan"], state["a"], state["b"], self.config["partial"]["keep_mass"], self.config["partial"]["solver"])
        if not torch.equal(state["r"], state["plan"].sum(1)) or not torch.equal(state["c"], state["plan"].sum(0)):
            raise ValueError("Stored retained marginals disagree with the raw partial plan.")
        return state

    @torch.no_grad()
    def map_features(self, features, query_ids, *, valid_mask=None, return_metadata=False, chunk_size=None, on_query=None):
        if self.mode == "grouped_partial" and not return_metadata:
            raise ValueError("grouped_partial requires return_metadata=True; features alone erase rejection.")
        validate_maps(features, query_ids, self.source.representation, valid_mask)
        start = time.perf_counter()
        q = features.detach().cpu().double()
        outputs, confidences, masks, records = [], [], [], []
        settings = self.config["projection"]
        size = settings["query_chunk_size"] if chunk_size is None else chunk_size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("chunk_size must be positive.")
        p = q.shape[1]
        # One complete query image at a time bounds Theta memory to [N_source,N_target].
        for offset in range(0, len(q), size):
            for patches, query_id in zip(q[offset:offset + size], query_ids[offset:offset + size]):
                detail = dict(query_id=query_id)
                confidence = torch.ones(p, dtype=torch.float64)
                if self.mode == "patch_global":
                    weights = self.patch.conditional_weights(patches)
                    mapped = weights @ self.patch.target
                    detail["patch_effective_targets_mean"] = float(torch.exp(-(weights * weights.clamp_min(1e-300).log()).sum(1)).mean())
                    detail["query_min_fit_distance_mean"] = float(torch.cdist(patches, self.patch.source).min(1).values.mean())
                else:
                    vector = patches.reshape(1, -1)
                    original_alpha = self.image.conditional_weights(vector)[0]
                    alpha, retained = select_images(original_alpha, self.target.ids, query_id, settings)
                    entropy = float(-(alpha * alpha.clamp_min(1e-300).log()).sum())
                    detail.update(image_entropy=entropy, image_normalized_entropy=entropy / math.log(len(alpha)) if len(alpha) > 1 else 0.,
                        image_effective_targets=math.exp(entropy), image_max_weight=float(alpha.max()),
                        top_k_retained_mass=retained, top_k_discarded_mass=1 - retained,
                        image_weights=alpha.tolist(), target_ids=self.target.ids,
                        query_min_fit_distance=float(torch.cdist(vector, self.image.source).min()))
                    if self.mode == "whole_map":
                        mapped = (alpha @ self.y.reshape(len(self.y), -1)).reshape_as(patches)
                    elif self.mode == "grouped_patch":
                        mapped = torch.zeros_like(patches)
                        effective, error = 0., 0.
                        for j, tid in enumerate(self.target.ids):
                            if alpha[j] == 0:
                                continue
                            cols = self.patch_groups[tid]
                            beta = normalize_rows(self.patch.scores(patches, cols))
                            mapped += alpha[j] * (beta @ self.patch.target[cols])
                            error = max(error, float((alpha[j] * beta.sum(1) - alpha[j]).abs().max()))
                            effective += float(alpha[j] * (-(beta * beta.clamp_min(1e-300).log()).sum(1)).exp().mean())
                        detail.update(group_mass_residual=error, within_image_effective_patches=effective)
                    else:
                        theta = self.image.pair_weights(vector)[0]
                        # Selection/truncation acts once on image routes, preserving p(i|j,Q).
                        theta = theta * (alpha / original_alpha.clamp_min(1e-300))[None]
                        if not torch.allclose(theta.sum(0), alpha, atol=1e-10, rtol=1e-10):
                            raise ValueError("Pair routing does not reproduce image weights.")
                        feature_sum, confidence = torch.zeros_like(patches), torch.zeros(p, dtype=torch.float64)
                        rejected_ot, support_invalid = torch.zeros_like(confidence), torch.zeros_like(confidence)
                        invalid_routes = torch.zeros_like(confidence)
                        group_budgets = torch.zeros(p, len(self.target.ids), dtype=torch.float64)
                        accounting_error, effective = 0., 0.
                        underflow_rows = log_retry_rows = 0
                        for i, sid in enumerate(self.source.ids):
                            logs = log_query_kernel(patches, self.x[i], self.shared["sx"][i], self.shared["h_projection"])
                            for j, tid in enumerate(self.target.ids):
                                route = theta[i, j]
                                if route == 0:
                                    continue
                                pair = self._pair(sid, tid)
                                candidate, g, diag = partial_projection(patches, self.x[i], self.y[j], pair["plan"],
                                    pair["a"], pair["b"], self.shared["sx"][i], self.shared["sy"][j],
                                    self.shared["h_projection"], self.shared["support"][i],
                                    target_kernel=self.projection_ky[j], query_logs=logs)
                                if (g < -1e-8).any() or (g > 1 + 1e-8).any():
                                    raise ValueError("Retained confidence violates [0,1].")
                                g = g.clamp(0, 1)  # Only solver-tolerance boundary roundoff.
                                confidence += route * g
                                feature_sum += route * g[:, None] * candidate
                                rejected_ot += route * (1 - diag["raw_confidence"])
                                support_invalid += route * diag["raw_confidence"] * (~diag["support_valid"])
                                invalid_routes += route * (~diag["support_valid"])
                                underflow_rows += int(diag["confidence_underflow"].sum())
                                log_retry_rows += int(diag["score_log_retry"].sum())
                                budget = route * g * diag["weights"].sum(1) + route * (1 - g)
                                group_budgets[:, j] += budget
                                accounting_error = max(accounting_error, float((budget - route).abs().max()))
                                patch_entropy = -(diag["weights"] * diag["weights"].clamp_min(1e-300).log()).sum(1)
                                effective += float((route * g * patch_entropy.exp()).sum())
                        cf = settings["confidence"]
                        positive = confidence > cf["mass_floor"]
                        mapped = torch.zeros_like(feature_sum)
                        mapped[positive] = feature_sum[positive] / confidence[positive, None]
                        detail.update(matched_rejected_accounting_error=accounting_error,
                            matched_rejected_group_mass_residual=float((group_budgets - alpha[None]).abs().max()),
                            ot_rejected_mass=rejected_ot.tolist(), support_invalid_retained_mass=support_invalid.tolist(),
                            support_invalid_route_mass=invalid_routes.tolist(),
                            support_invalid_confidence_underflow_rows=underflow_rows, score_log_retry_rows=log_retry_rows,
                            within_image_effective_patches=effective / float(confidence.sum()) if confidence.sum() > 0 else 0.,
                            confidence_min=float(confidence.min()), confidence_mean=float(confidence.mean()))
                cf = settings["confidence"]
                mask = ((confidence > cf["mass_floor"]) & (confidence >= cf["threshold"])) if self.mode == "grouped_partial" else torch.ones(p, dtype=torch.bool)
                if not torch.isfinite(mapped).all():
                    raise ValueError(f"Nonfinite mapped features for {query_id}")
                detail.update(valid_fraction=float(mask.double().mean()), all_invalid=bool(not mask.any()),
                              mapped_norm_mean=float(mapped.norm(dim=1).mean()))
                outputs.append(mapped); confidences.append(confidence); masks.append(mask); records.append(detail)
                if on_query is not None:
                    on_query(detail)
        result = MappingResult(torch.stack(outputs).to(features), torch.stack(confidences).to(features),
                               torch.stack(masks).to(features.device),
                               dict(queries=records, seconds=time.perf_counter() - start, mode=self.mode,
                                    mapped_variance=float(torch.stack(outputs).flatten(0, 1).var(0, unbiased=False).mean()),
                                    note="Feature diagnostics are not independent image-quality validation."))
        # Enforce the chosen all-invalid policy at the public boundary.
        result.conditioning(query_ids, all_invalid_policy=cf["all_invalid_policy"])
        return result if return_metadata else result.mapped_features

    def project_bank(self, bank, output, *, count=None, chunk_size=None, run_log=None):
        from .run_logging import RunLog

        output = Path(output)
        if run_log is None:
            output.mkdir(parents=True, exist_ok=False)
        elif any((output / name).exists() for name in ("manifest.json", "mapped.pt")):
            raise FileExistsError(f"Mapping output already exists: {output}")
        context = RunLog(output, "mapping", metadata=dict(mapper_id=self.manifest["artifact_id"],
            query_bank_id=bank.artifact_id, mode=self.mode)) if run_log is None else nullcontext(run_log)
        with context as log:
            compatible_banks(self.source, bank, fitting=False)
            if bank.manifest["domain"] != self.source.manifest["domain"]:
                raise ValueError("Query domain must match the fitted source domain.")
            if bank.manifest["split"] not in {"val", "test"} or set(bank.ids) & (set(self.source.ids) | set(self.target.ids)):
                raise ValueError("The fit-and-test command requires disjoint held-out query IDs and a val/test bank.")
            if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
                raise ValueError("count must be a positive integer.")
            n = len(bank.ids) if count is None else min(count, len(bank.ids))
            features, ids = bank.features[:n], bank.ids[:n]
            def on_query(detail):
                log.append_jsonl("queries.jsonl", detail)
                log.event("query_completed", **detail)
            log.event("mapping_started", query_count=n, query_ids=ids, mode=self.mode,
                      projection=self.config["projection"])
            result = self.map_features(features, ids, valid_mask=torch.ones(features.shape[:2], dtype=torch.bool),
                                       return_metadata=True, chunk_size=chunk_size, on_query=on_query)
            report = mapping_summary(result, features, self.x, self.y)
            log.write_json("mapping_report.json", report)
            manifest = self._save_mapping(bank, output, ids, result)
            log.event("mapping_saved", artifact_id=manifest["artifact_id"], query_count=n,
                      seconds=result.diagnostics["seconds"])
            return result, manifest

    def _save_mapping(self, bank, output, ids, result):
        path = output / "mapped.pt"
        save_tensor(path, dict(mapped_features=result.mapped_features.cpu(), match_confidence=result.match_confidence.cpu(),
                               valid_mask=result.valid_mask.cpu(), ids=ids, records=bank.records[:len(ids)]))
        manifest = dict(schema="infoot_mapped_features_v2", mapper_id=self.manifest["artifact_id"],
            projection_implementation_sha256=file_hash(Path(__file__)),
            projection_dependency_sha256={name: file_hash(Path(__file__).with_name(name))
                for name in ("conditional.py", "partial.py", "feature_bank.py")},
            mode=self.mode, query_bank_id=bank.artifact_id, ids=ids, query_domain=bank.manifest["domain"],
            representation=bank.representation, projection=self.config["projection"],
            output=dict(file="mapped.pt", sha256=file_hash(path)), diagnostics=result.diagnostics)
        manifest["artifact_id"] = digest(manifest)
        write_json(output / "manifest.json", manifest)
        return manifest


def mapping_summary(result, query, source, target):
    """Separate within-map spread from variation between image means for tuning."""
    def statistics(features, mask=None):
        x = features.detach().cpu().double()
        if mask is None:
            population, means = x.flatten(0, 1), x.mean(1)
            position_variance = float(x.var(0, unbiased=False).mean())
        else:
            valid = mask.detach().cpu()
            population = x[valid]
            counts = valid.sum(1)
            active = counts > 0
            means = (x * valid[..., None]).sum(1)[active] / counts[active, None]
            position_counts = valid.sum(0)
            position_means = (x * valid[..., None]).sum(0) / position_counts.clamp_min(1)[:, None]
            position_energy = ((x - position_means[None]).square() * valid[..., None]).sum(0)
            observed_positions = position_counts > 0
            position_variance = (float((position_energy[observed_positions]
                / position_counts[observed_positions, None]).mean()) if observed_positions.any() else None)
        return dict(images=len(x), valid_images=len(means), valid_tokens=len(population),
            norm_mean=float(population.norm(dim=1).mean()) if len(population) else None,
            token_variance=float(population.var(0, unbiased=False).mean()) if len(population) else None,
            corresponding_patch_variance_across_images=position_variance,
            image_mean_variance=float(means.var(0, unbiased=False).mean()) if len(means) else None)
    confidence = result.match_confidence.detach().cpu().double()
    valid = result.valid_mask.detach().cpu()
    return dict(mode=result.diagnostics["mode"], seconds=result.diagnostics["seconds"],
        query=statistics(query), source_support=statistics(source), target_support=statistics(target),
        mapped=statistics(result.mapped_features, valid),
        valid_fraction=float(valid.double().mean()), all_invalid_images=int((~valid.any(1)).sum()),
        confidence_quantiles=dict(zip(("min", "p05", "p50", "p95", "max"),
            torch.quantile(confidence.flatten(), confidence.new_tensor([0, .05, .5, .95, 1])).tolist())),
        note="Descriptive feature diagnostics only; image means do not replace patch features for fitting or conditioning. Compare fixed generated images independently.")


def load_mapped(directory, *, mapper_id=None):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (m.get("schema") != "infoot_mapped_features_v2"
            or digest({k: v for k, v in m.items() if k != "artifact_id"}) != m.get("artifact_id")
            or (mapper_id is not None and m["mapper_id"] != mapper_id)):
        raise ValueError("Mapped-cache identity mismatch.")
    state = torch.load(checked_file(directory, m["output"]), weights_only=True)
    if set(state) != {"mapped_features", "match_confidence", "valid_mask", "ids", "records"} or state["ids"] != m["ids"]:
        raise ValueError("Mapped caches must load features/confidence/masks atomically by stable ID.")
    result = MappingResult(state["mapped_features"], state["match_confidence"], state["valid_mask"], m["diagnostics"])
    if (result.match_confidence.shape != result.valid_mask.shape or not torch.isfinite(result.mapped_features).all()
            or not torch.isfinite(result.match_confidence).all() or (result.match_confidence < 0).any()
            or (result.match_confidence > 1 + 1e-7).any() or len(state["ids"]) != len(result.mapped_features)
            or [r["sample_id"] for r in state["records"]] != state["ids"]):
        raise ValueError("Invalid mapped cache shapes/values.")
    result.conditioning(state["ids"], all_invalid_policy=m["projection"]["confidence"]["all_invalid_policy"])
    return result, state, m
