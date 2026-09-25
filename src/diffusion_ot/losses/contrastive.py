"""Contrastive matching supervision and generated-image/code retrieval.

Teacher-selected matching losses maximize probability mass on structurally
similar positives. The detached-key helper also serves image/code retrieval,
including the self-supervised experiment's cross-domain encoder comparison.
These objectives do not replace generative flow or truncate the InfoOT plan.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def detached_key_contrastive_loss(query, positive, bank, *, temperature=.2,
                                 negative_similarity_threshold=.95):
    """Single-positive InfoNCE with detached keys and similar-key masking.

    The bank can include the positive: its duplicate is masked. The mask is
    computed from keys only, so the query cannot evade negatives by moving.
    Rows without a valid positive and at least one distinct negative contribute
    no supervision. A zero loss with zero usable rows is not retrieval success.
    """
    if (query.ndim != 2 or positive.shape != query.shape or bank.ndim != 2
            or bank.shape[1] != query.shape[1] or not len(query) or not query.shape[1]):
        raise ValueError("Contrastive queries and keys need compatible nonempty [samples, features] shapes.")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("Contrastive temperature must be finite and positive.")
    if not math.isfinite(float(negative_similarity_threshold)) or not -1 <= negative_similarity_threshold <= 1:
        raise ValueError("Contrastive negative similarity threshold must be in [-1, 1].")
    with torch.autocast(device_type=query.device.type, enabled=False):
        query, positive, bank = query.float(), positive.detach().to(query.device).float(), bank.detach().to(query.device).float()
        if not all(torch.isfinite(value).all() for value in (query, positive, bank)):
            raise FloatingPointError("Non-finite contrastive queries or keys.")
        valid_positive, valid_bank = positive.norm(dim=-1) > 1e-6, bank.norm(dim=-1) > 1e-6
        query = F.normalize(query, dim=-1, eps=1e-6)
        positive, bank = F.normalize(positive, dim=-1, eps=1e-6), F.normalize(bank, dim=-1, eps=1e-6)
        # The tolerance also masks exact copies when fp32 cosine rounds below 1.
        allowed = ((positive @ bank.T) < negative_similarity_threshold - 1e-6) & valid_bank[None, :]
        counts = allowed.sum(-1)
        usable = valid_positive & (counts > 0)
        positive_logits = (query * positive).sum(-1) / temperature
        negative_logits = (query @ bank.T / temperature).masked_fill(~allowed, -torch.inf)
        logits = torch.cat((positive_logits[:, None], negative_logits), dim=1)
        # The positive is column zero, as in RElbers/info-nce-pytorch's
        # explicit-negative formulation. Cross-entropy shifts logits before
        # reduction: subtracting a large positive logit from logsumexp can
        # round a small but nonzero loss to zero in FP32.
        per_row = F.cross_entropy(
            logits, torch.zeros(len(query), dtype=torch.long, device=query.device), reduction="none",
        )
        loss = per_row[usable].mean() if usable.any() else query.sum() * 0
        metrics = {
            "loss": float(loss.detach()), "samples": len(query),
            "usable_samples": int(usable.sum()), "skipped_samples": int((~usable).sum()),
            "negative_bank_size": len(bank),
            "mean_usable_negatives": float(counts[usable].float().mean()) if usable.any() else 0.0,
            "temperature": float(temperature),
            "negative_similarity_threshold": float(negative_similarity_threshold),
            "positive_probability": None, "retrieval_top1": None,
            "retrieval_chance": None, "uniform_loss": None,
            "positive_minus_hardest_negative_cosine": None,
        }
        with torch.no_grad():
            if usable.any():
                usable_logits = logits[usable]
                # Fractional credit for exact score ties avoids optimistic
                # argmax accuracy when every condition has the same score.
                highest = usable_logits.max(-1, keepdim=True).values
                tied = (usable_logits - highest).abs() <= 1e-6
                metrics.update(
                    positive_probability=float((-per_row[usable]).exp().mean()),
                    retrieval_top1=float((tied[:, 0].float() / tied.sum(-1)).mean()),
                    retrieval_chance=float((counts[usable].float() + 1).reciprocal().mean()),
                    uniform_loss=float((counts[usable].float() + 1).log().mean()),
                    positive_minus_hardest_negative_cosine=float(
                        ((positive_logits[usable] - negative_logits[usable].max(-1).values) * temperature).mean()),
                )
    return loss, metrics


def _positive_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("positive_count must be a positive integer.")
    return value


def _temperature(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Contrastive temperature must be finite and positive.")
    return value


def matching_contrastive_options(config: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve an opt-in additional objective; absent/disabled stays unchanged.

    ``temperature`` scales within-domain cosine logits only. Cross-domain
    conditional log probabilities retain their existing KDE calibration.
    Weights are applied by the trainer, including its alignment warmup.
    """
    if not config.get("enabled", False):
        return None
    allowed = {"enabled", "neighborhood_weight", "conditional_weight", "positive_count", "temperature"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown matching_contrastive settings: {sorted(unknown)}")
    weights = {key: float(config.get(key, .01)) for key in ("neighborhood_weight", "conditional_weight")}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Matching contrastive weights must be finite and nonnegative.")
    if not any(weights.values()):
        raise ValueError("Enabled matching contrastive supervision needs a positive weight.")
    return {
        **weights,
        "positive_count": _positive_count(config.get("positive_count", 4)),
        "temperature": _temperature(config.get("temperature", .2)),
    }


def teacher_multi_positive_contrastive_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    *,
    positive_count: int = 4,
    exclude_diagonal: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Negative log student mass on detached teacher top-k positives.

    Each row contrasts the positive set with all remaining valid candidates:
    ``logsumexp(all logits) - logsumexp(positive logits)``. Include every
    teacher tie at the kth threshold, so tied structural neighbors are never
    arbitrarily assigned opposite labels. All-positive rows contribute zero
    and are reported as unusable, without inventing negatives. Reduction is a
    mean over all rows, including these zero contributions.

    For conditional InfoOT supervision, pass *log probabilities*, not direct
    Cat/Dog feature similarities. Teacher rows may be unnormalized but must
    have positive total mass after self-pairs are excluded. The original
    student/teacher tensors and full transport support are never modified.
    This multi-positive mass objective is not average-positive SupCon: it can
    concentrate on one positive. Retain KL and variance/covariance protection.
    """
    positive_count = _positive_count(positive_count)
    if (student_logits.ndim != 2 or teacher_probabilities.ndim != 2
            or student_logits.shape != teacher_probabilities.shape):
        raise ValueError("Contrastive student and teacher must have the same [queries, candidates] shape.")
    rows, columns = student_logits.shape
    if rows < 1 or columns < 2:
        raise ValueError("Contrastive supervision needs queries and at least two candidates.")
    if not student_logits.is_floating_point() or not teacher_probabilities.is_floating_point():
        raise ValueError("Contrastive student logits and teacher probabilities must be floating point.")
    if exclude_diagonal and rows != columns:
        raise ValueError("Self-pair exclusion requires a square within-domain matrix.")
    valid_count = columns - int(exclude_diagonal)
    if positive_count >= valid_count:
        raise ValueError("positive_count must be smaller than the number of valid candidates.")
    # Preserve double precision for numerical tests; avoid half-precision
    # reductions and autocast matrix products in actual training.
    logits = student_logits if student_logits.dtype == torch.float64 else student_logits.float()
    teacher = teacher_probabilities.detach().to(logits)
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("Contrastive logits and teacher probabilities must be finite.")
    if bool((teacher < 0).any()):
        raise ValueError("Teacher probabilities must be nonnegative.")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        valid = torch.ones_like(logits, dtype=torch.bool)
        if exclude_diagonal:
            valid.fill_diagonal_(False)
        teacher_mass = teacher.masked_fill(~valid, 0).sum(dim=1)
        if bool((teacher_mass <= 0).any()) or not bool(torch.isfinite(teacher_mass).all()):
            raise ValueError("Every teacher row must have finite positive mass on valid candidates.")
        scores = teacher.masked_fill(~valid, -torch.inf)
        threshold = scores.topk(positive_count, dim=1).values[:, -1:]
        positives = valid & (scores >= threshold)
        positive_counts = positives.sum(dim=1)
        usable = positive_counts < valid_count
        denominator = torch.logsumexp(logits.masked_fill(~valid, -torch.inf), dim=1)
        numerator = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
        per_row = torch.where(usable, (denominator - numerator).clamp_min(0), torch.zeros_like(denominator))
        loss = per_row.mean()
    with torch.no_grad():
        metrics = {
            "loss": float(loss), "rows": rows, "valid_candidates": valid_count,
            "requested_positive_count": positive_count,
            "mean_positive_count": float(positive_counts.float().mean()),
            "mean_negative_count": float((valid_count - positive_counts).float().mean()),
            "usable_rows": int(usable.sum()), "fraction_usable_rows": float(usable.float().mean()),
            "mean_positive_probability": float((-per_row).exp().mean()),
            "teacher_positive_mass": float((teacher.masked_fill(~positives, 0).sum(1) / teacher_mass).mean()),
            "exclude_diagonal": exclude_diagonal, "reduction": "mean_all_rows",
        }
    return loss, metrics


def matching_neighborhood_contrastive_loss(
    matching_features: torch.Tensor,
    structure_descriptors: torch.Tensor,
    *,
    positive_count: int = 4,
    temperature: float = .2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Contrast same-domain matching outputs using frozen structural neighbors.

    Call independently for each domain. Input features are the live post-L2
    outputs used by InfoOT; normalizing defensively preserves this convention.
    Detached descriptors select positives. Gradients reach matching heads and
    encoders through the student only. This is an additional discrimination
    objective, not a guarantee against dimensional or exact collapse.
    """
    temperature = _temperature(temperature)
    if (matching_features.ndim != 2 or structure_descriptors.ndim != 2
            or matching_features.shape[0] != structure_descriptors.shape[0]
            or matching_features.shape[1] < 1 or structure_descriptors.shape[1] < 1):
        raise ValueError("Matching features and structure descriptors need matched [samples, dimensions] rows.")
    if not matching_features.is_floating_point() or not structure_descriptors.is_floating_point():
        raise ValueError("Matching features and structure descriptors must be floating point.")
    values = matching_features if matching_features.dtype == torch.float64 else matching_features.float()
    with torch.autocast(device_type=values.device.type, enabled=False):
        student = F.normalize(values, dim=1)
        teacher = F.normalize(structure_descriptors.detach().to(values), dim=1)
        student_logits = (student @ student.T) / temperature
        teacher_probabilities = F.softmax((teacher @ teacher.T) / temperature, dim=1)
        loss, metrics = teacher_multi_positive_contrastive_loss(
            student_logits, teacher_probabilities, positive_count=positive_count, exclude_diagonal=True,
        )
    return loss, {**metrics, "temperature": temperature, "geometry": "same_domain_matching_cosine"}


def matching_contrastive_loss(
    reference_matching: dict[str, torch.Tensor],
    reference_structure: dict[str, torch.Tensor],
    conditional_log_weights: dict[str, torch.Tensor],
    teacher_weights: dict[str, torch.Tensor],
    *,
    neighborhood_weight: float,
    conditional_weight: float,
    positive_count: int,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Average each enabled objective over domains/directions, then weight.

    Neighborhoods use reference banks; conditional rows remain the existing
    disjoint queries. Each conditional matrix spans the entire target bank.
    Zero-weight branches are not evaluated, allowing independent ablations.
    The trainer adds this result to its existing KL objectives and applies
    the alignment warmup. Metrics here describe the pre-warmup result.
    """
    options = matching_contrastive_options({
        "enabled": True, "neighborhood_weight": neighborhood_weight,
        "conditional_weight": conditional_weight, "positive_count": positive_count,
        "temperature": temperature,
    })
    assert options is not None
    neighborhood_weight = options["neighborhood_weight"]
    conditional_weight = options["conditional_weight"]
    weighted_losses = []
    metrics: dict[str, Any] = {
        **options, "neighborhood_loss": 0.0, "conditional_loss": 0.0,
        "weighted_neighborhood_loss": 0.0, "weighted_conditional_loss": 0.0,
        "neighborhood": {}, "conditional": {}, "neighborhood_sample_scope": "ot_references",
        "conditional_temperature": "existing_kde_log_probabilities",
    }
    if neighborhood_weight > 0:
        if not reference_matching or set(reference_matching) != set(reference_structure):
            raise ValueError("Matching contrastive neighborhoods need matching feature/structure domain banks.")
        domain_losses = []
        for domain, features in reference_matching.items():
            loss, domain_metrics = matching_neighborhood_contrastive_loss(
                features, reference_structure[domain], positive_count=positive_count, temperature=temperature,
            )
            domain_losses.append(loss)
            metrics["neighborhood"][domain] = domain_metrics
        neighborhood = torch.stack(domain_losses).mean()
        weighted_losses.append(neighborhood_weight * neighborhood)
        metrics["neighborhood_loss"] = float(neighborhood.detach())
        metrics["weighted_neighborhood_loss"] = float(weighted_losses[-1].detach())
    if conditional_weight > 0:
        if not conditional_log_weights or set(conditional_log_weights) != set(teacher_weights):
            raise ValueError("Conditional contrastive supervision needs matching student/teacher direction banks.")
        direction_losses = []
        for direction, log_weights in conditional_log_weights.items():
            loss, direction_metrics = teacher_multi_positive_contrastive_loss(
                log_weights, teacher_weights[direction], positive_count=positive_count,
            )
            direction_losses.append(loss)
            metrics["conditional"][direction] = direction_metrics
        conditional = torch.stack(direction_losses).mean()
        weighted_losses.append(conditional_weight * conditional)
        metrics["conditional_loss"] = float(conditional.detach())
        metrics["weighted_conditional_loss"] = float(weighted_losses[-1].detach())
    loss = torch.stack(weighted_losses).sum()
    metrics["weighted_loss"] = float(loss.detach())
    return loss, metrics
