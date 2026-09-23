"""Frozen visual structure priors for encoder co-training (project extension).

The prior never compares coordinates from separately trained PDAE encoders.
Its patch self-similarities follow the structure/appearance distinction of
Splice; using pooled DINOv2 tokens here is an experimental adaptation.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from diffusion_ot.integrations.hf_snapshot import resolve_project_local_path


def patch_structure_descriptor(tokens: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
    """Pool a square token grid, then compare spatial parts within each image."""
    if tokens.ndim != 3:
        raise ValueError("Expected [batch, patches, channels] tokens.")
    side = math.isqrt(tokens.shape[1])
    if side * side != tokens.shape[1] or not 2 <= grid_size <= side:
        raise ValueError("Expected square [batch, patches, channels] tokens and a valid grid size.")
    grid = tokens.float().transpose(1, 2).reshape(tokens.shape[0], -1, side, side)
    pooled = F.adaptive_avg_pool2d(grid, grid_size).flatten(2).transpose(1, 2)
    pooled = F.normalize(pooled, dim=-1)
    similarity = pooled @ pooled.transpose(1, 2)
    mask = ~torch.eye(grid_size ** 2, dtype=torch.bool, device=tokens.device)
    # Drop constant diagonal entries and remove overall similarity per image.
    descriptor = similarity[:, mask]
    descriptor = descriptor - descriptor.mean(1, keepdim=True)
    return F.normalize(descriptor, dim=-1)


def neighborhood_options(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve legacy defaults as well as the scale-invariant neighborhood loss."""
    temperature = float(config.get("neighborhood_temperature", .2))
    geometry = str(config.get("neighborhood_geometry", "cosine"))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Neighborhood temperature must be finite and positive.")
    if geometry not in {"cosine", "rms_distance"}:
        raise ValueError("semantic_prior.neighborhood_geometry must be cosine or rms_distance.")
    return {"temperature": temperature, "geometry": geometry}


def neighborhood_distillation_loss(
    codes: torch.Tensor, teacher: torch.Tensor, *, temperature: float = 0.2,
    geometry: str = "cosine",
) -> torch.Tensor:
    """Match within-domain neighbors, excluding self-pairs.

    rms_distance divides squared pair distances by their live off-diagonal
    mean. Uniform contraction cannot flatten the student distribution, unlike
    fixed-temperature cosine logits. Teacher distances get their own scale.
    Inspired by RKD's relative distances (https://arxiv.org/abs/1904.05068),
    but using squared distances and KL rather than RKD's L2/Huber loss.
    This removes a scale shortcut; variance protection is still necessary.
    """
    if codes.ndim != 2 or teacher.ndim != 2 or codes.shape[0] != teacher.shape[0] or codes.shape[0] < 3:
        raise ValueError("Neighborhood distillation needs at least three matched samples.")
    neighborhood_options({"neighborhood_temperature": temperature, "neighborhood_geometry": geometry})
    # Legacy mode keeps its original precision/autocast behavior for controls.
    if geometry == "cosine":
        student = F.normalize(codes.float(), dim=-1)
        teacher = F.normalize(teacher.detach().to(student), dim=-1)
        mask = ~torch.eye(len(codes), device=codes.device, dtype=torch.bool)
        student_logits = (student @ student.T)[mask].view(len(codes), -1) / temperature
        teacher_logits = (teacher @ teacher.T)[mask].view(len(codes), -1) / temperature
        return F.kl_div(F.log_softmax(student_logits, dim=-1),
                        F.softmax(teacher_logits, dim=-1), reduction="batchmean")
    # Center first to avoid subtracting nearly equal unit-vector dot products
    # in a narrow cone. Keep float64 for gradient checks, otherwise use fp32.
    with torch.autocast(device_type=codes.device.type, enabled=False):
        student = F.normalize(codes if codes.dtype == torch.float64 else codes.float(), dim=-1)
        teacher = F.normalize(teacher.detach().to(student), dim=-1)
        mask = ~torch.eye(len(codes), device=codes.device, dtype=torch.bool)
        def logits(features: torch.Tensor) -> torch.Tensor:
            centered = features - features.mean(0)
            norms = centered.square().sum(1)
            distances = (norms[:, None] + norms[None, :] - 2 * centered @ centered.T).clamp_min(0)
            distances = distances[mask].view(len(codes), -1)
            # Do not detach: doing so reintroduces a spurious scale gradient.
            scale = distances.mean().clamp_min(1e-8)
            return -distances / (scale * temperature)
        return F.kl_div(F.log_softmax(logits(student), dim=-1),
                        F.softmax(logits(teacher), dim=-1), reduction="batchmean")


def descriptor_digest(payload: dict[str, Any]) -> str:
    metadata = {key: payload[key] for key in ("sample_ids", "domains", "splits", "metadata")}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(payload["features"].detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SemanticPriorBank:
    def __init__(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported semantic-prior bank format.")
        self.sample_ids = list(payload["sample_ids"])
        self.domains = list(payload["domains"])
        self.splits = list(payload["splits"])
        self.features = payload["features"].detach().float()
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Duplicate semantic-prior sample IDs.")
        if self.features.ndim != 2 or not all(
            len(values) == len(self.features) for values in (self.sample_ids, self.domains, self.splits)
        ):
            raise ValueError("Semantic-prior features and metadata lengths differ.")
        if not torch.isfinite(self.features).all() or (self.features.norm(dim=1) < 1e-6).any():
            raise ValueError("Semantic-prior descriptors must be finite and nonzero.")
        self.fingerprint = descriptor_digest(payload)
        if payload.get("fingerprint") != self.fingerprint:
            raise ValueError("Semantic-prior fingerprint does not match its contents.")
        self.index = {key: i for i, key in enumerate(self.sample_ids)}
        self.metadata = payload["metadata"]
        self.cost_scale = float(self.metadata["cost_scale"])
        if not math.isfinite(self.cost_scale) or self.cost_scale <= 0:
            raise ValueError("Semantic-prior cost scale must be finite and positive.")

    def lookup(self, ids: list[str], domain: str, split: str) -> torch.Tensor:
        missing = [key for key in ids if key not in self.index]
        if missing:
            raise ValueError(f"Semantic-prior bank missing {len(missing)} IDs, including {missing[:3]}.")
        indices = [self.index[key] for key in ids]
        if any(self.domains[i] != domain or self.splits[i] != split for i in indices):
            raise ValueError("Semantic-prior domain/split mismatch; held-out features cannot enter training.")
        return self.features[indices]

    def cost(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Fixed training-calibrated scale, shared by minibatches and evaluation.
        return torch.cdist(source.detach().float(), target.detach().float()) / self.cost_scale


def load_semantic_prior(config: dict[str, Any], root: Path) -> SemanticPriorBank | None:
    variant = str((config.get("infoot") or {}).get("variant", "plain"))
    if variant not in {"plain", "fused"}:
        raise ValueError("infoot.variant must be plain or fused.")
    if variant == "plain":
        return None
    cost_source = str((config.get("infoot") or {}).get("cross_cost_source", "dino"))
    if cost_source not in {"dino", "encoder"}:
        raise ValueError("infoot.cross_cost_source must be dino or encoder.")
    if cost_source == "encoder":
        if config.get("semantic_prior"):
            raise ValueError("Encoder-only transport must not configure a semantic_prior.")
        return None
    prior = config.get("semantic_prior") or {}
    if not prior.get("path"):
        raise ValueError("Fused InfoOT requires semantic_prior.path; build the descriptor cache first.")
    path = resolve_project_local_path(prior["path"], root, field_name="semantic_prior.path")
    return SemanticPriorBank(path)


def _compatible_infoot_resume(saved: dict[str, Any], current: dict[str, Any]) -> bool:
    """A larger cap is safe when every accepted solve already had to converge.

    Allow more outer iterations and (for strict solves) more Sinkhorn iterations
    or bounded recovery after the ordinary cap fails.
    Objective, tolerances, initialization and patience must still match. Never
    mutate either checkpoint config. Truncated runs cannot use this exception.
    """
    if saved == current:
        return True
    saved, current = dict(saved), dict(current)
    previous_budget = int(saved.pop("inner_iterations", 50))
    current_budget = int(current.pop("inner_iterations", 50))
    previous_projection_budget = int(saved.pop("projection_iterations", 200))
    current_projection_budget = int(current.pop("projection_iterations", 200))
    previous_recovery_budget = int(saved.pop("recovery_iterations", 0))
    current_recovery_budget = int(current.pop("recovery_iterations", 0))
    return (saved == current and current_budget >= previous_budget
            and current_projection_budget >= previous_projection_budget
            and current_recovery_budget >= previous_recovery_budget >= 0
            and ((current_projection_budget == previous_projection_budget
                  and current_recovery_budget == previous_recovery_budget)
                 or bool(saved.get("strict_convergence", False)))
            and bool(saved.get("require_outer_convergence", False))
            and float(saved.get("outer_tolerance", 0)) > 0)


def validate_prior_resume(saved_config: dict[str, Any], current_config: dict[str, Any]) -> None:
    from diffusion_ot.training.pcgrad import PCGradConfig
    saved_pcgrad = PCGradConfig.from_mapping(saved_config.get("pcgrad"))
    current_pcgrad = PCGradConfig.from_mapping(current_config.get("pcgrad"))
    if saved_pcgrad != current_pcgrad:
        raise ValueError("Resume cannot change pcgrad; start a new run.")
    if current_pcgrad.enabled and (saved_config.get("train") or {}).get("seed", 20260905) != (current_config.get("train") or {}).get("seed", 20260905):
        raise ValueError("Resume cannot change train.seed with PCGrad's per-step random ordering.")
    saved_variant = (saved_config.get("infoot") or {}).get("variant", "plain")
    current_variant = (current_config.get("infoot") or {}).get("variant", "plain")
    saved = (saved_config.get("semantic_prior") or {}).get("fingerprint")
    current = (current_config.get("semantic_prior") or {}).get("fingerprint")
    if saved_variant != current_variant or saved != current:
        raise ValueError("Resume cannot change InfoOT variant or semantic-prior bank; start a new run.")
    if saved_variant == "fused" and neighborhood_options(saved_config.get("semantic_prior") or {}) != neighborhood_options(current_config.get("semantic_prior") or {}):
        raise ValueError("Resume cannot change semantic_prior neighborhood objective; start a new run.")
    if any(any((config.get(key) or {}).get("enabled", False) for key in
               ("conditional_structure", "conditional_projection", "gradient_guard", "projection_support", "matching_head", "matching_regularization", "matching_contrastive", "decoded_encoder_balance")) for config in (saved_config, current_config)):
        for key in ("conditional_structure", "conditional_projection", "loss_weights", "matching", "infoot", "gradient_guard", "projection_support", "matching_head", "matching_regularization", "matching_contrastive", "decoded_encoder_balance", "generator_adaptation", "decoded_translation", "trainable"):
            if key == "infoot" and _compatible_infoot_resume(saved_config.get(key) or {}, current_config.get(key) or {}):
                continue
            if (saved_config.get(key) or {}) != (current_config.get(key) or {}):
                raise ValueError(f"Resume cannot change {key} for conditional structure training; start a new run.")
