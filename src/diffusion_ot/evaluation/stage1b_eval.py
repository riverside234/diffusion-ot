from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import warnings

import torch

from diffusion_ot.integrations.hf_snapshot import (
    effective_project_root,
    find_project_root,
    load_yaml_config,
    resolve_project_local_path,
)
from diffusion_ot.losses.infoot import (
    conditional_projection_weights,
    conditional_variance_decomposition,
    effective_target_count,
    infoot_cross_distance_scale,
    infoot_distance_scale,
    kernel_offdiagonal_stats,
    median_distance_scale,
    nearest_plan_row_weights,
    normalize_matching_features,
    pairwise_squared_distances,
    solve_infoot,
    solver_kwargs,
    transport_diagnostics,
    weighted_target_codes,
)
from diffusion_ot.models.matching_head import (
    load_matching_head, matching_features, matching_head_id, matching_head_spec,
)
from diffusion_ot.losses.projection_rms import (
    checkpoint_projection_rms, projection_rms_options, reference_variances, validate_projection_scales,
)


@dataclass
class LatentBank:
    domain: str
    split: str
    raw_codes: torch.Tensor
    matching_features: torch.Tensor
    sample_ids: list[str]
    metadata: list[dict[str, Any]]
    checkpoint_id: str
    matching_id: str = "l2_raw_v1"

    def validate(self) -> None:
        count = self.raw_codes.shape[0]
        if self.raw_codes.ndim != 2 or self.matching_features.ndim != 2:
            raise ValueError("Latent bank tensors must have shape [samples, dimension].")
        if self.matching_features.shape[0] != count:
            raise ValueError("Raw-code and matching-feature bank sizes differ.")
        if len(self.sample_ids) != count or len(self.metadata) != count:
            raise ValueError("Latent bank IDs or metadata do not match its tensor length.")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError(f"Duplicate sample IDs in {self.domain}_{self.split} bank.")
        if not torch.isfinite(self.raw_codes).all() or not torch.isfinite(self.matching_features).all():
            raise FloatingPointError("Latent bank contains non-finite values.")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "format_version": 2,
            "domain": self.domain,
            "split": self.split,
            "raw_codes": self.raw_codes.detach().cpu(),
            "matching_features": self.matching_features.detach().cpu(),
            "sample_ids": self.sample_ids,
            "metadata": self.metadata,
            "checkpoint_id": self.checkpoint_id,
            "matching_id": self.matching_id,
        }


@dataclass
class DomainEvaluationContext:
    domain: str
    branch: Any
    transformer: Any
    vae: Any
    training_config: dict[str, Any]
    data_config_path: Path
    device: str
    dtype: torch.dtype
    checkpoint_path: Path
    checkpoint_step: int
    weights: str
    stage1a_checkpoint_path: Path
    stage1a_checkpoint_step: int
    stage1a_weights: str
    stage1a_architecture: dict[str, Any]
    matching_head: Any | None = None
    projection_rms: Any | None = None
    patch_projector: Any | None = None


@dataclass
class Stage1BEvaluationReport:
    alignment_config_path: str
    evaluation_config_path: str
    alignment_checkpoint: str | None
    mode: str
    output_dir: str
    seed: int
    stage1a_architectures: dict[str, dict[str, Any]]
    generation_protocol: dict[str, Any]
    reference_sizes: dict[str, int]
    projection_sizes: dict[str, int]
    query_sizes: dict[str, int]
    distance_scales: dict[str, float]
    solver: dict[str, Any]
    latent_diagnostics: dict[str, dict[str, float]]
    reconstruction: dict[str, Any]
    baseline_comparison: dict[str, Any]
    checkpoint_selection: dict[str, Any]
    projections: dict[str, Any]
    translation_grids: dict[str, str]
    visualization_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_self_supervised_checkpoint(alignment_config, checkpoint):
    """Do not label weights trained with external losses as the new control."""
    saved = checkpoint.get("config") or {}
    current_image, saved_image = (config.get("decoded_translation") or {} for config in (alignment_config, saved))
    current_mode, saved_mode = (image.get("supervision", "external") for image in (current_image, saved_image))
    if "self_supervised" not in (current_mode, saved_mode):
        return
    if current_mode != saved_mode:
        raise ValueError("Alignment config and checkpoint decoded supervision disagree.")
    current_objective, saved_objective = (
        image.get("objective", "source_infonce") for image in (current_image, saved_image))
    if current_objective != saved_objective:
        raise ValueError("Alignment config and checkpoint decoded objective disagree.")
    if current_objective == "patchnce":
        from diffusion_ot.training.decoded_translation import self_supervised_translation_options
        try:
            current_patch, saved_patch = (self_supervised_translation_options(image)
                                          for image in (current_image, saved_image))
        except ValueError as error:
            raise ValueError(f"Invalid PatchNCE protocol: {error}") from error
        # The separate guard below describes readout mismatches explicitly.
        current_patch.pop("readout")
        saved_patch.pop("readout")
        if current_patch != saved_patch:
            raise ValueError("Alignment config and checkpoint PatchNCE protocol disagree.")
    if current_image.get("source_contrastive_readout", "target") != saved_image.get("source_contrastive_readout", "target"):
        raise ValueError("Alignment config and checkpoint source contrastive readout disagree.")
    for field, default in (("variant", "plain"), ("cross_cost_source", "dino")):
        if (alignment_config.get("infoot") or {}).get(field, default) != (saved.get("infoot") or {}).get(field, default):
            raise ValueError(f"Alignment config and checkpoint infoot.{field} disagree.")


def _nested(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _evaluation_bandwidths(
    matching_config: dict[str, Any], projection_override: float | None = None,
) -> tuple[float, float]:
    fit = float(matching_config.get("bandwidth_multiplier", 1.0))
    projection = (
        projection_override if projection_override is not None
        else matching_config.get("projection_bandwidth_multiplier")
    )
    projection = fit if projection is None else float(projection)
    for name, value in (("bandwidth_multiplier", fit),
                        ("projection_bandwidth_multiplier", projection)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"matching.{name} must be finite and positive.")
    return fit, projection


def _warn_fit_bandwidth_mismatch(alignment_config: dict[str, Any], fit: float) -> None:
    """Keep intentional sensitivity sweeps possible but expose stale defaults."""
    configured = float(_nested(alignment_config, "matching").get("bandwidth_multiplier", 1.0))
    if not math.isclose(configured, fit, rel_tol=1e-8, abs_tol=0.0):
        warnings.warn(
            f"Evaluation fit bandwidth {fit:g} differs from alignment config {configured:g}. "
            "Use matching bandwidths for a training-matched comparison; intentional "
            "bandwidth sweeps evaluate a different fitted transport plan.",
            UserWarning, stacklevel=2,
        )


def deterministic_indices(length: int, count: int | None, seed: int) -> list[int]:
    if length <= 0:
        raise ValueError("Cannot sample from an empty dataset.")
    if count is None or int(count) >= length:
        return list(range(length))
    if int(count) <= 0:
        raise ValueError("Sample count must be positive or None for the full dataset.")
    count = min(int(count), length)
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(length, generator=generator)[:count].tolist()


def _checkpoint_identifier(path: Path, step: int, weights: str) -> str:
    payload = f"{path.resolve()}::{int(step)}::{weights}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _protocol_identifier(
    evaluation_config: dict[str, Any],
    *,
    max_reference: int | None,
    max_projection: int | None,
    max_query: int | None,
) -> str:
    payload = {
        "evaluation_config": evaluation_config,
        # Older reports used VAE-decoded x0 as the pixel target. Never reuse
        # their protocol ID for metrics against the original dataset image.
        "reconstruction_metric_protocol": "original_dataset_rgb_v1",
        "cli_overrides": {
            "max_reference": max_reference,
            "max_projection": max_projection,
            "max_query": max_query,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:8]


def save_latent_bank(path: str | Path, bank: LatentBank) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(bank.to_payload(), temporary)
    temporary.replace(target)


def load_latent_bank(path: str | Path) -> LatentBank:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location="cpu")
    bank = LatentBank(
        domain=str(payload["domain"]),
        split=str(payload["split"]),
        raw_codes=payload["raw_codes"],
        matching_features=payload["matching_features"],
        sample_ids=[str(value) for value in payload["sample_ids"]],
        metadata=list(payload["metadata"]),
        checkpoint_id=str(payload["checkpoint_id"]),
        matching_id=str(payload.get("matching_id", "l2_raw_v1")),
    )
    bank.validate()
    return bank


def validate_bank_compatibility(
    reference: LatentBank,
    query: LatentBank,
    *,
    require_disjoint: bool = True,
) -> None:
    reference.validate()
    query.validate()
    if reference.domain != query.domain:
        raise ValueError("Reference and query banks must belong to the same domain.")
    if reference.checkpoint_id != query.checkpoint_id:
        raise ValueError("Reference and query banks were produced by different encoders.")
    if reference.matching_id != query.matching_id:
        raise ValueError("Reference and query banks were produced by different matching heads.")
    if reference.raw_codes.shape[1] != query.raw_codes.shape[1]:
        raise ValueError("Reference and query raw-code dimensions differ.")
    if reference.matching_features.shape[1] != query.matching_features.shape[1]:
        raise ValueError("Reference and query matching dimensions differ.")
    if require_disjoint and set(reference.sample_ids).intersection(query.sample_ids):
        raise ValueError("Reference and query banks overlap; train-only evaluation is invalid.")


def build_latent_bank(
    encoder: Any,
    dataset: Any,
    *,
    domain: str,
    split: str,
    count: int | None,
    seed: int,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    checkpoint_id: str,
    matching_head: Any | None = None,
) -> LatentBank:
    indices = deterministic_indices(len(dataset), count, seed)
    codes: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    sample_ids: list[str] = []
    metadata: list[dict[str, Any]] = []
    encoder.eval()
    if matching_head is not None:
        matching_head.eval()
    with torch.inference_mode():
        for offset in range(0, len(indices), int(batch_size)):
            items = [dataset[index] for index in indices[offset : offset + int(batch_size)]]
            x0 = torch.stack([item["x0_latent"] for item in items]).to(
                device=device, dtype=dtype
            )
            raw = encoder(x0).detach().float()
            codes.append(raw.cpu())
            features.append(matching_features(raw, matching_head).cpu())
            sample_ids.extend(str(item["sample_id"]) for item in items)
            metadata.extend(dict(item["metadata"]) for item in items)
    raw_codes = torch.cat(codes, dim=0)
    bank = LatentBank(
        domain=domain,
        split=split,
        raw_codes=raw_codes,
        matching_features=torch.cat(features),
        sample_ids=sample_ids,
        metadata=metadata,
        checkpoint_id=checkpoint_id,
        matching_id=matching_head_id(matching_head),
    )
    bank.validate()
    return bank


def subset_latent_bank(bank: LatentBank, *, count: int, seed: int) -> LatentBank:
    """Select a deterministic fit-reference subset without re-encoding samples."""
    bank.validate()
    indices = deterministic_indices(len(bank.sample_ids), count, seed)
    selected = LatentBank(
        domain=bank.domain,
        split=bank.split,
        raw_codes=bank.raw_codes[indices].clone(),
        matching_features=bank.matching_features[indices].clone(),
        sample_ids=[bank.sample_ids[index] for index in indices],
        metadata=[dict(bank.metadata[index]) for index in indices],
        checkpoint_id=bank.checkpoint_id,
        matching_id=bank.matching_id,
    )
    selected.validate()
    return selected


def load_proxy_labels(
    path: str | Path | None,
    *,
    sample_id_key: str = "sample_id",
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    labels: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if sample_id_key not in row:
                raise ValueError(f"Missing {sample_id_key} at proxy-label line {line_number}.")
            sample_id = str(row[sample_id_key])
            if sample_id in labels:
                raise ValueError(f"Duplicate proxy label for sample_id={sample_id}.")
            labels[sample_id] = row
    return labels


def _labels_for_bank(
    bank: LatentBank,
    external: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for sample_id, metadata in zip(bank.sample_ids, bank.metadata):
        merged = dict(metadata)
        merged.update(external.get(sample_id, {}))
        result[sample_id] = merged
    return result


def _effective_rank(codes: torch.Tensor, eps: float = 1.0e-8) -> float:
    centered = codes.float() - codes.float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values.square()
    probabilities = probabilities / probabilities.sum().clamp_min(eps)
    return float((-(probabilities * probabilities.clamp_min(eps).log()).sum()).exp())


def latent_diagnostics(bank: LatentBank) -> dict[str, float]:
    codes = bank.raw_codes.float()
    return {
        "mean_norm": float(codes.norm(dim=1).mean()),
        "mean_feature_std": float(codes.std(dim=0, unbiased=False).mean()),
        "effective_rank": _effective_rank(codes),
        "matching_effective_rank": _effective_rank(bank.matching_features),
        "matching_feature_variance": float(bank.matching_features.float().var(0, unbiased=False).sum()),
    }


def _direction_projection_evaluation(
    source_reference: LatentBank,
    target_reference: LatentBank,
    source_query: LatentBank,
    coupling: torch.Tensor,
    *,
    target_projection: LatentBank | None = None,
    source_scale: float,
    target_scale: float,
    bandwidth: float,
    eps: float,
    distance_scale_mode: str = "fixed_stage1a_median",
    projection_bandwidth: float | None = None,
    projection_scales: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    bandwidth, projection_bandwidth = _evaluation_bandwidths(
        {"bandwidth_multiplier": bandwidth}, projection_bandwidth
    )
    target_projection = target_reference if target_projection is None else target_projection
    validate_bank_compatibility(source_reference, source_query, require_disjoint=False)
    validate_bank_compatibility(target_reference, target_projection, require_disjoint=False)
    device = coupling.device
    source_ref_features = source_reference.matching_features.to(device)
    target_ref_features = target_reference.matching_features.to(device)
    source_query_features = source_query.matching_features.to(device)
    target_projection_features = target_projection.matching_features.to(device)
    if projection_scales is not None:
        if distance_scale_mode != "infoot_rms":
            raise ValueError("Explicit projection RMS requires infoot_rms matching.")
        projection_scales = validate_projection_scales(projection_scales)
        query_source_scale = projection_scales[source_reference.domain]
        projection_target_scale = projection_scales[target_reference.domain]
    elif distance_scale_mode == "infoot_rms":
        query_source_scale = float(
            infoot_cross_distance_scale(source_query_features, source_ref_features)
        )
        projection_target_scale = float(
            infoot_cross_distance_scale(target_projection_features, target_ref_features)
        )
    elif distance_scale_mode == "fixed_stage1a_median":
        query_source_scale = source_scale
        projection_target_scale = target_scale
    else:
        raise ValueError(
            "matching.distance_scale must be 'infoot_rms' or "
            "'fixed_stage1a_median'."
        )

    plan_weights, nearest_source = nearest_plan_row_weights(
        source_query_features, source_ref_features, coupling, eps=eps
    )
    conditional_weights = conditional_projection_weights(
        source_query_features,
        target_projection_features,
        source_ref_features,
        target_ref_features,
        coupling,
        bandwidth=projection_bandwidth,
        distance_scale_x=query_source_scale,
        distance_scale_y=projection_target_scale,
        eps=eps,
    )

    target_reference_raw = target_reference.raw_codes.to(device)
    target_projection_raw = target_projection.raw_codes.to(device)
    barycentric_codes = weighted_target_codes(plan_weights, target_reference_raw)
    conditional_codes = weighted_target_codes(conditional_weights, target_projection_raw)
    nearest_target_distance = pairwise_squared_distances(
        normalize_matching_features(conditional_codes), normalize_matching_features(target_projection_raw)
    ).min(dim=1).values.sqrt()
    norm_ratio = conditional_codes.norm(dim=1) / target_projection_raw.norm(dim=1).mean().clamp_min(eps)

    report = {
        "fit_bandwidth_multiplier": bandwidth,
        "projection_bandwidth_multiplier": projection_bandwidth,
        "projection_method": "infoot_eq7_conditional_expectation",
        "projection_support": "target_training_projection_bank",
        "projection_target_count": len(target_projection.sample_ids),
        "mean_conditional_effective_target_count": float(
            effective_target_count(conditional_weights).mean().cpu()
        ),
        "mean_plan_row_effective_target_count": float(
            effective_target_count(plan_weights).mean().cpu()
        ),
        "mean_nearest_target_distance": float(nearest_target_distance.mean().cpu()),
        "nearest_target_distance_space": "l2_raw_decoder_codes",
        "barycentric_projection_space": "l2_raw_decoder_codes",
        "matching_representation_ids": {
            "source": source_reference.matching_id, "target": target_reference.matching_id,
        },
        "mean_projected_to_target_norm_ratio": float(norm_ratio.mean().cpu()),
        "projected_to_target_variance_ratio": float(
            conditional_codes.var(0, unbiased=False).mean()
            / target_projection_raw.var(0, unbiased=False).mean().clamp_min(eps)
        ),
        "projected_mean_shift_squared": float(
            (conditional_codes.mean(0) - target_projection_raw.mean(0)).square().mean()
            / target_projection_raw.var(0, unbiased=False).mean().clamp_min(eps)
        ),
        "nearest_source_indices": nearest_source.cpu().tolist(),
        "conditional_distance_scales": {
            "query_source": query_source_scale,
            "projection_target": projection_target_scale,
        },
        "conditional_scale_mode": "reference_ema" if projection_scales is not None else distance_scale_mode,
    }
    tensors = {
        "conditional_weights": conditional_weights.detach().cpu(),
        "plan_weights": plan_weights.detach().cpu(),
        "conditional_codes": conditional_codes.detach().cpu(),
        "barycentric_codes": barycentric_codes.detach().cpu(),
    }
    return report, tensors


def _torch_dtype_from_model(model: Any) -> torch.dtype:
    for parameter in model.parameters():
        return parameter.dtype
    return torch.float32


def require_latent_stage1a_encoder(
    stage_config: dict[str, Any], *, source: str, checkpoint: dict[str, Any] | None = None,
) -> None:
    """Keep the current latent-only Stage 1B pipeline paired with legacy encoders."""
    encoder = stage_config.get("encoder") or {}
    saved_input = ((checkpoint or {}).get("model") or {}).get("encoder_input") or {}
    if (str(encoder.get("input_space", "latent")).lower() != "latent"
            or int(encoder.get("input_channels", 4)) != 4
            or str(saved_input.get("space", "latent")).lower() != "latent"):
        raise ValueError(
            f"Stage 1B currently requires a VAE-latent semantic encoder; {source} "
            "configures an RGB or otherwise incompatible encoder. Use the matching "
            "cat/dog_sit_b2_lora.yaml recipe with its original latent checkpoint. "
            "RGB Stage 1A checkpoints require a separate Stage 1B input-pipeline migration."
        )


def _load_domain_context(
    alignment_config: dict[str, Any],
    root: Path,
    domain: str,
    *,
    checkpoint_path: Path | None,
    joint_weights: str,
    device_override: str | None,
) -> DomainEvaluationContext:
    from diffusion_ot.evaluation.stage1a_eval import (
        _apply_ema_weights,
        _load_checkpoint,
        validate_stage1a_architecture,
    )
    from diffusion_ot.integrations.sit_diffusers import load_sit_components, validate_transformer_config
    from diffusion_ot.models.pdae_sit import build_pdae_sit_branch

    domain_config = _nested(_nested(alignment_config, "stage1a"), domain)
    train_config_path = resolve_project_local_path(
        domain_config["config"], root, field_name=f"stage1a.{domain}.config"
    )
    train_config = load_yaml_config(train_config_path)
    require_latent_stage1a_encoder(train_config, source=f"Stage 1A {domain} recipe")
    stage1a_checkpoint_path = resolve_project_local_path(
        domain_config["checkpoint"], root, field_name=f"stage1a.{domain}.checkpoint"
    )
    stage1a_checkpoint = _load_checkpoint(stage1a_checkpoint_path)
    require_latent_stage1a_encoder(
        stage1a_checkpoint.get("config") or {}, source=f"Stage 1A {domain} checkpoint",
        checkpoint=stage1a_checkpoint,
    )
    model_config_path = resolve_project_local_path(
        train_config["model_config"], root, field_name="model_config"
    )
    data_config_path = resolve_project_local_path(
        train_config["data_config"], root, field_name="data_config"
    )
    model_config = load_yaml_config(model_config_path)
    pretrained_config_path = resolve_project_local_path(
        model_config["pretrained"], root, field_name="model.pretrained"
    )
    device = str(device_override or domain_config.get("device") or train_config.get("device", "cuda:0"))
    components = load_sit_components(
        pretrained_config_path,
        project_root=root,
        device=device,
        torch_dtype=str(train_config.get("torch_dtype", "float32")),
    )
    mismatches = validate_transformer_config(
        components.transformer, model_config.get("expected_transformer_config") or {}
    )
    if mismatches:
        raise ValueError("Unexpected SiT transformer config:\n" + "\n".join(mismatches))
    branch = build_pdae_sit_branch(
        components.transformer, model_config=model_config, stage_config=train_config
    )
    dtype = _torch_dtype_from_model(components.transformer)
    branch.to(device=device, dtype=dtype)

    from diffusion_ot.training.decoded_translation import validate_flow_only_stage1a
    validate_flow_only_stage1a(alignment_config, train_config, stage1a_checkpoint)
    branch.load_pdae_state_dict(stage1a_checkpoint["model"])
    initial_weights = str(_nested(alignment_config, "stage1a").get("weights", "ema"))
    if initial_weights == "ema":
        _apply_ema_weights(branch, stage1a_checkpoint)
    stage1a_architecture = validate_stage1a_architecture(
        branch, _nested(alignment_config, "stage1a")
    )

    checkpoint_step = int(stage1a_checkpoint.get("step", 0))
    selected_path = stage1a_checkpoint_path
    if checkpoint_path is not None:
        joint = _load_joint_checkpoint(checkpoint_path)
        _validate_self_supervised_checkpoint(alignment_config, joint)
        provenance = (joint.get("stage1a_provenance") or {}).get(domain)
        if not isinstance(provenance, dict):
            raise ValueError(f"Joint checkpoint has no stage1a_provenance.{domain} record.")
        expected_path = Path(str(provenance.get("checkpoint_path", "")))
        if not expected_path.is_absolute():
            expected_path = resolve_project_local_path(
                expected_path,
                root,
                field_name=f"stage1a_provenance.{domain}.checkpoint_path",
            )
        provenance_mismatches = []
        if expected_path.resolve() != stage1a_checkpoint_path.resolve():
            provenance_mismatches.append(
                f"path {stage1a_checkpoint_path} != {expected_path.resolve()}"
            )
        if int(provenance.get("checkpoint_step", -1)) != checkpoint_step:
            provenance_mismatches.append(
                f"step {checkpoint_step} != {provenance.get('checkpoint_step')}"
            )
        if str(provenance.get("weights")) != initial_weights:
            provenance_mismatches.append(
                f"weights {initial_weights} != {provenance.get('weights')}"
            )
        saved_architecture = provenance.get("architecture")
        if saved_architecture is not None and saved_architecture != stage1a_architecture:
            provenance_mismatches.append(
                f"architecture {stage1a_architecture} != {saved_architecture}"
            )
        if provenance_mismatches:
            raise ValueError(
                f"Configured Stage 1A {domain} checkpoint does not match the Stage 1B "
                "checkpoint provenance: " + "; ".join(provenance_mismatches)
            )
        from diffusion_ot.models.generator_adaptation import generator_adaptation_enabled, load_joint_generator
        if generator_adaptation_enabled(joint.get("config") or {}) != generator_adaptation_enabled(alignment_config):
            raise ValueError("Alignment config and checkpoint generator_adaptation disagree.")
        load_joint_generator(branch, joint, domain, weights=joint_weights)
        state_key = "encoder_ema" if joint_weights == "ema" else "encoders"
        state = (joint.get(state_key) or {}).get(domain)
        if state is None:
            raise ValueError(f"Joint checkpoint has no {state_key}.{domain} state.")
        branch.encoder.load_state_dict(state)
        checkpoint_step = int(joint.get("step", 0))
        selected_path = checkpoint_path

    matching_head = None
    projection_rms = None
    patch_projector = None
    if checkpoint_path is not None:
        if matching_head_spec(joint.get("config") or {}) != matching_head_spec(alignment_config):
            raise ValueError("Alignment config and checkpoint matching_head architectures disagree.")
        matching_head = load_matching_head(joint, domain, weights=joint_weights, device=device)
        projection_rms = checkpoint_projection_rms(joint, alignment_config, weights=joint_weights)
        image_options = alignment_config.get("decoded_translation") or {}
        if image_options.get("supervision") == "self_supervised" and image_options.get("objective") == "patchnce":
            from diffusion_ot.models.patch_sampler import load_patch_projector
            from diffusion_ot.training.decoded_translation import self_supervised_translation_options
            patch_projector = load_patch_projector(
                branch.encoder, self_supervised_translation_options(image_options), joint, domain,
                weights=joint_weights, device=device)
    branch.eval()
    components.vae.eval()
    return DomainEvaluationContext(
        domain=domain,
        branch=branch,
        transformer=components.transformer,
        vae=components.vae,
        training_config=train_config,
        data_config_path=data_config_path,
        device=device,
        dtype=dtype,
        checkpoint_path=selected_path,
        checkpoint_step=checkpoint_step,
        weights=joint_weights if checkpoint_path is not None else initial_weights,
        stage1a_checkpoint_path=stage1a_checkpoint_path,
        stage1a_checkpoint_step=int(stage1a_checkpoint.get("step", 0)),
        stage1a_weights=initial_weights,
        stage1a_architecture=stage1a_architecture,
        matching_head=matching_head,
        projection_rms=projection_rms,
        patch_projector=patch_projector,
    )


def _load_joint_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("stage") not in {"stage1b_plain_infoot", "stage1b_fused_infoot"}:
        raise ValueError(f"Not a Stage 1B InfoOT checkpoint: {path}")
    return checkpoint


def _dataset(context: DomainEvaluationContext, split: str, root: Path):
    from diffusion_ot.data.latent_dataset import CachedLatentDataset

    return CachedLatentDataset(
        context.data_config_path,
        context.domain,
        split=split,
        project_root=root,
        validate_exists=True,
        random_horizontal_flip=0.0,
    )


def _load_latents_from_bank(bank: LatentBank, count: int) -> torch.Tensor:
    from diffusion_ot.data.latent_dataset import load_latent_tensor

    values = [load_latent_tensor(row["latent_path"]) for row in bank.metadata[:count]]
    return torch.stack(values, dim=0)


def _mse_psnr(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = float((prediction.float() - target.float()).square().mean().cpu())
    return {"mse": mse, "psnr": float("inf") if mse == 0.0 else -10.0 * math.log10(mse)}


def _lpips(prediction: torch.Tensor, target: torch.Tensor, device: str) -> float:
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as exc:
        raise RuntimeError("LPIPS evaluation requires torchmetrics[image] from requirements.txt.") from exc
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    return float(metric(prediction.to(device), target.to(device)).detach().cpu())


@torch.inference_mode()
def _evaluate_reconstruction(
    context: DomainEvaluationContext,
    bank: LatentBank,
    *,
    count: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    include_lpips: bool,
    output_path: Path,
) -> dict[str, Any]:
    from diffusion_ot.data.ground_truth import load_ground_truth_images
    from diffusion_ot.evaluation.stage1a_eval import decode_vae_latents, integrate_pdae_flow
    from torchvision.utils import save_image

    count = min(int(count), len(bank.sample_ids))
    original_images = load_ground_truth_images(context.data_config_path, bank.metadata[:count]).to(context.device)
    x0 = _load_latents_from_bank(bank, count).to(context.device, dtype=context.dtype)
    z = bank.raw_codes[:count].to(context.device, dtype=context.dtype)
    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    noise = torch.randn(x0.shape, generator=generator, device=context.device, dtype=context.dtype)
    class_config = _nested(context.training_config, "class_conditioning")
    reconstructed = integrate_pdae_flow(
        context.branch,
        context.transformer,
        noise,
        z,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        null_label=class_config.get("null_label"),
    )
    vae_images = decode_vae_latents(context.vae, x0)
    reconstructed_images = decode_vae_latents(context.vae, reconstructed)
    if reconstructed_images.shape != original_images.shape or vae_images.shape != original_images.shape:
        raise ValueError("Ground-truth image and decoded reconstruction shapes differ; check cache preprocessing.")
    latent_metrics = _mse_psnr(reconstructed, x0)
    pixel_metrics = _mse_psnr(reconstructed_images, original_images)
    vae_metrics = _mse_psnr(vae_images, original_images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        torch.cat([original_images.cpu(), vae_images.cpu(), reconstructed_images.cpu()]),
        str(output_path),
        nrow=count,
        padding=2,
        pad_value=1.0,
    )
    report: dict[str, Any] = {
        "sample_ids": bank.sample_ids[:count],
        "image_reference": "original_dataset_rgb",
        "image_preprocessing": "same_center_crop_and_bicubic_resize_as_latent_cache",
        "grid_rows": ["ground_truth", "vae_reconstruction", "model_reconstruction"],
        "latent_mse": latent_metrics["mse"],
        "pixel_mse": pixel_metrics["mse"],
        "pixel_psnr": pixel_metrics["psnr"],
        "grid_path": str(output_path),
        "vae_reconstruction_to_ground_truth": {
            "pixel_mse": vae_metrics["mse"], "pixel_psnr": vae_metrics["psnr"],
        },
    }
    if include_lpips:
        report["lpips"] = _lpips(reconstructed_images, original_images, context.device)
        report["vae_reconstruction_to_ground_truth"]["lpips"] = _lpips(vae_images, original_images, context.device)
    return report


_TRANSLATION_ROW_NAMES = {
    "conditional_mean": "infoot_conditional_mean",
    "conditional_map": "selected_target",
    "conditional_sample": "sampled_target",
    "structure_teacher_mean": "structure_teacher_mean",
}


@torch.inference_mode()
def _save_translation_grid(
    source: DomainEvaluationContext,
    target: DomainEvaluationContext,
    source_query: LatentBank,
    target_projection: LatentBank,
    weights: torch.Tensor,
    *,
    count: int,
    num_steps: int,
    guidance_scale: float,
    temperature: float,
    seed: int,
    output_path: Path,
    teacher_codes: torch.Tensor | None = None,
    image_features: Any | None = None,
    readouts: list[str] | None = None,
    include_source: bool = True,
    color_histogram: dict[str, Any] | None = None,
    patchnce_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from diffusion_ot.evaluation.stage1a_eval import decode_vae_latents, integrate_pdae_flow
    from torchvision.utils import save_image

    count = min(int(count), weights.shape[0])
    weights = weights[:count].float()
    if readouts is None:
        readouts = ["conditional_mean", "conditional_map", "conditional_sample"]
        if teacher_codes is not None:
            readouts.append("structure_teacher_mean")
    if not readouts or len(set(readouts)) != len(readouts) or any(
        name not in _TRANSLATION_ROW_NAMES for name in readouts
    ):
        raise ValueError("Translation readouts must be a nonempty, unique list of supported readouts.")
    if "structure_teacher_mean" in readouts and teacher_codes is None:
        raise ValueError("The structure_teacher_mean readout requires teacher codes.")
    code_rows = []
    for name in readouts:
        if name == "conditional_mean":
            codes = weighted_target_codes(weights, target_projection.raw_codes)
        elif name == "conditional_map":
            codes = target_projection.raw_codes[weights.argmax(dim=1)]
        elif name == "conditional_sample":
            sampling_weights = torch.softmax(weights.clamp_min(1.0e-12).log() / float(temperature), dim=1)
            sample_generator = torch.Generator().manual_seed(int(seed) + 1)
            indices = torch.multinomial(sampling_weights.cpu(), 1, generator=sample_generator).squeeze(1)
            codes = target_projection.raw_codes[indices]
        else:
            codes = teacher_codes[:count]
        code_rows.append(codes)

    source_x0 = _load_latents_from_bank(source_query, count).to(source.device, dtype=source.dtype)
    source_images = None
    if include_source or image_features is not None:
        source_images = decode_vae_latents(source.vae, source_x0).cpu()
    noise_generator = torch.Generator(device=target.device).manual_seed(int(seed))
    noise = torch.randn(
        (count, *source_x0.shape[1:]),
        generator=noise_generator,
        device=target.device,
        dtype=target.dtype,
    )
    class_config = _nested(target.training_config, "class_conditioning")
    rows = [source_images] if include_source else []
    decoded_metrics: dict[str, Any] = {}
    source_structure = None
    original_images = None
    if patchnce_options is not None:
        from diffusion_ot.losses.patchnce import patchnce_loss
        from diffusion_ot.training.self_supervised_translation import encode_generated_images
        readout = target if patchnce_options["readout"] == "target" else source
        patch_keys = source.branch.encoder.forward_spatial_features(source_x0, patchnce_options["layers"])
        if patchnce_options["sampler"] == "mlp_sample" and any(
                getattr(context, "patch_projector", None) is None for context in (source, readout)):
            raise ValueError("PatchNCE evaluation requires trained MLP checkpoint weights.")
        decoded_metrics["patchnce_weights"] = source.weights
        decoded_metrics["patchnce_query_ids"] = source_query.sample_ids[:count]
        decoded_metrics["patchnce_seed"] = seed + 20000
    if color_histogram is not None:
        from diffusion_ot.data.ground_truth import load_ground_truth_images
        from diffusion_ot.losses.color_histogram import COLOR_PROTOCOL, color_histogram_options, histogan_color_distance
        color_options = color_histogram_options(color_histogram)
        color_parameters = {k: v for k, v in color_options.items() if k != "weight"}
        original_images = load_ground_truth_images(source.data_config_path, source_query.metadata[:count]).cpu()
        decoded_metrics["color_histogram_protocol"] = COLOR_PROTOCOL
        decoded_metrics["color_histogram_parameters"] = color_parameters
        decoded_metrics["color_histogram_target"] = "original_source_rgb_whole_image"
        decoded_metrics["color_histogram_interpretation"] = "Whole-image RGB-uv palette distance; background included, not spatial markings, exposure or independent realism."
        decoded_metrics["source_query_ids"] = source_query.sample_ids[:count]
        decoded_metrics["seed"] = seed
    if image_features is not None:
        feature_device = next(image_features.parameters()).device
        source_structure, _ = image_features(source_images.to(feature_device))
    for name, codes in zip(readouts, code_rows):
        row_name = _TRANSLATION_ROW_NAMES[name]
        latent = integrate_pdae_flow(
            target.branch,
            target.transformer,
            noise,
            codes.to(target.device, dtype=target.dtype),
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            null_label=class_config.get("null_label"),
        )
        decoded_images = decode_vae_latents(target.vae, latent).cpu()
        rows.append(decoded_images)
        if patchnce_options is not None:
            recovered = encode_generated_images(readout.vae, decoded_images.to(readout.device), checkpoint_encode=False)
            query_maps = readout.branch.encoder.forward_spatial_features(
                recovered.to(readout.device, dtype=readout.dtype), patchnce_options["layers"])
            _, patch_metrics = patchnce_loss(
                query_maps, patch_keys, temperature=patchnce_options["temperature"],
                num_patches=patchnce_options["num_patches"],
                generator=torch.Generator(device="cpu").manual_seed(seed + 20000),
                query_projector=getattr(readout, "patch_projector", None),
                key_projector=getattr(source, "patch_projector", None))
            decoded_metrics.setdefault(row_name, {"samples": count})["patchnce"] = patch_metrics
        if image_features is not None:
            structure, _ = image_features(decoded_images.to(feature_device))
            per_image = 1 - torch.nn.functional.cosine_similarity(structure, source_structure, dim=-1)
            decoded_metrics.setdefault(row_name, {"samples": count}).update(
                structure_loss=float(per_image.mean()), per_image_structure_loss=per_image.cpu().tolist())
        if original_images is not None:
            distances = histogan_color_distance(decoded_images, original_images, **color_parameters)
            decoded_metrics.setdefault(row_name, {"samples": count}).update(
                color_histogram_loss=float(distances.mean()),
                per_image_color_histogram_loss=distances.tolist())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat(rows), str(output_path), nrow=count, padding=2, pad_value=1.0)
    if original_images is not None:
        # Preserve the existing grid; also save originals beside the same
        # conditional-mean images, without an additional rollout or RNG draw.
        target_row = rows[int(include_source) + readouts.index("conditional_mean")] if "conditional_mean" in readouts else rows[int(include_source)]
        paired_path = output_path.with_name(output_path.stem + "_source_pairs.png")
        save_image(torch.stack((original_images, target_row), 1).flatten(0, 1),
                   str(paired_path), nrow=8, padding=2, pad_value=1.)
        decoded_metrics["source_pair_grid"] = str(paired_path)
        decoded_metrics["source_pair_readout"] = "conditional_mean" if "conditional_mean" in readouts else readouts[0]
    if image_features is not None:
        decoded_metrics["interpretation"] = "Decoded DINO structure against source; uses the training feature prior, not independent proxy labels or a calibrated realism score."
    return decoded_metrics


def _proxy_label_value(
    labels: dict[str, dict[str, Any]], sample_id: str, attribute: str
) -> str:
    value = labels.get(sample_id, {}).get(attribute)
    if value is None or not str(value).strip():
        return "unlabeled"
    return str(value)


def _proxy_label_coverage_caption(
    target_ids: list[str],
    source_ids: list[str],
    *,
    labels: dict[str, dict[str, Any]],
    attributes: list[str],
) -> str:
    if not attributes:
        return "Label coverage: no proxy attributes configured."
    parts = []
    for attribute in attributes:
        target_count = sum(
            _proxy_label_value(labels, sample_id, attribute) != "unlabeled"
            for sample_id in target_ids
        )
        source_count = sum(
            _proxy_label_value(labels, sample_id, attribute) != "unlabeled"
            for sample_id in source_ids
        )
        parts.append(
            f"{attribute}: target {target_count}/{len(target_ids)}, "
            f"source {source_count}/{len(source_ids)}"
        )
    return "Label coverage: " + "; ".join(parts)


def _save_umap_visualizations(
    target_bank: LatentBank,
    source_query_ids: list[str],
    conditional_codes: torch.Tensor,
    barycentric_codes: torch.Tensor,
    *,
    labels: dict[str, dict[str, Any]],
    attributes: list[str],
    direction: str,
    random_state: int,
    n_jobs: int,
    target_alpha: float,
    projection_alpha: float,
    output_paths: dict[str, Path],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import umap
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError("UMAP visualization requires umap-learn and matplotlib.") from exc

    if conditional_codes.shape[0] != len(source_query_ids):
        raise ValueError("Conditional-code rows do not match source query IDs.")
    if barycentric_codes.shape[0] != len(source_query_ids):
        raise ValueError("Barycentric-code rows do not match source query IDs.")
    if not 0.0 < target_alpha <= 1.0 or not 0.0 < projection_alpha <= 1.0:
        raise ValueError("UMAP point alpha values must be in (0, 1].")
    expected_readouts = {"conditional", "barycentric"}
    if set(output_paths) != expected_readouts:
        raise ValueError(f"UMAP output paths must contain {sorted(expected_readouts)}.")

    reducer = umap.UMAP(
        random_state=int(random_state),
        transform_seed=int(random_state),
        n_jobs=int(n_jobs),
    )
    target_embedding = reducer.fit_transform(target_bank.raw_codes.numpy())
    projected_embeddings = {
        "conditional": reducer.transform(conditional_codes.numpy()),
        "barycentric": reducer.transform(barycentric_codes.numpy()),
    }
    plotted_attributes: list[str | None] = attributes or [None]
    direction_label = direction.replace("_to_", " to ").replace("_", " ").title()
    coverage_caption = _proxy_label_coverage_caption(
        target_bank.sample_ids,
        source_query_ids,
        labels=labels,
        attributes=attributes,
    )
    caption = (
        "Color = proxy attribute when labels exist; an unlabeled panel uses blue for "
        "the target bank and orange/green for the projected source. Marker = point "
        "type. Target bank contains real target-domain codes; projected source "
        "contains mapped source-query codes.\n"
        f"{coverage_caption}"
    )
    palette = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#332288",  # indigo
    ]
    readout_style = {
        "conditional": {
            "marker": "X",
            "label": "Projected source (conditional mean)",
            "fallback_color": "#E45756",
        },
        "barycentric": {
            "marker": "P",
            "label": "Projected source (barycentric)",
            "fallback_color": "#2A9D8F",
        },
    }

    for readout, projected_embedding in projected_embeddings.items():
        output_path = output_paths[readout]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(
            1,
            len(plotted_attributes),
            figsize=(7.6 * len(plotted_attributes), 8.4),
            squeeze=False,
        )
        for axis, attribute in zip(axes[0], plotted_attributes):
            if attribute is None:
                target_values = ["all"] * len(target_bank.sample_ids)
                projected_values = ["all"] * len(source_query_ids)
            else:
                target_values = [
                    _proxy_label_value(labels, sample_id, attribute)
                    for sample_id in target_bank.sample_ids
                ]
                projected_values = [
                    _proxy_label_value(labels, sample_id, attribute)
                    for sample_id in source_query_ids
                ]
            categories = sorted(
                set(target_values).union(projected_values),
                key=lambda value: (value == "unlabeled", value),
            )
            labeled_categories = [value for value in categories if value != "unlabeled"]
            colors = {
                category: (
                    "#5F6368"
                    if category == "unlabeled"
                    else palette[labeled_categories.index(category) % len(palette)]
                )
                for category in categories
            }
            labels_available = bool(labeled_categories)
            target_role_color = "#2563EB" if not labels_available else "#374151"
            projection_role_color = (
                readout_style[readout]["fallback_color"]
                if not labels_available
                else "#374151"
            )
            target_array = np.asarray(target_values)
            projected_array = np.asarray(projected_values)
            for category in categories:
                target_mask = target_array == category
                projected_mask = projected_array == category
                axis.scatter(
                    target_embedding[target_mask, 0],
                    target_embedding[target_mask, 1],
                    color=colors[category] if labels_available else target_role_color,
                    s=13,
                    alpha=float(target_alpha),
                    marker="o",
                    edgecolors="none",
                    zorder=1,
                )
                axis.scatter(
                    projected_embedding[projected_mask, 0],
                    projected_embedding[projected_mask, 1],
                    color=(
                        colors[category] if labels_available else projection_role_color
                    ),
                    s=38,
                    alpha=float(projection_alpha),
                    marker=readout_style[readout]["marker"],
                    edgecolors="#111827",
                    linewidths=0.45,
                    zorder=3,
                )
            category_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=colors[category],
                    markeredgecolor="none",
                    label=category.replace("_", " ").title(),
                    markersize=7,
                )
                for category in categories
            ]
            marker_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=target_role_color,
                    markeredgecolor="none",
                    label="Target bank (real target codes)",
                    markersize=6,
                    alpha=0.75,
                ),
                Line2D(
                    [0],
                    [0],
                    marker=readout_style[readout]["marker"],
                    linestyle="none",
                    markerfacecolor=projection_role_color,
                    markeredgecolor="#111827",
                    label=readout_style[readout]["label"],
                    markersize=8,
                    alpha=0.95,
                ),
            ]
            if labels_available:
                category_legend = axis.legend(
                    handles=category_handles,
                    title=f"{(attribute or 'sample').replace('_', ' ').title()} label",
                    fontsize=7,
                    title_fontsize=8,
                    loc="upper left",
                    framealpha=0.94,
                    ncols=2 if len(categories) > 5 else 1,
                )
                axis.add_artist(category_legend)
            axis.legend(
                handles=marker_handles,
                title="Point type",
                fontsize=7,
                title_fontsize=8,
                loc="upper right",
                framealpha=0.94,
            )
            if attribute is not None and categories == ["unlabeled"]:
                axis.text(
                    0.015,
                    0.02,
                    f"No {attribute.replace('_', ' ')} labels found. "
                    "Set proxy_labels.path or provide this field in dataset metadata.",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="#7F1D1D",
                    bbox={
                        "boxstyle": "round,pad=0.35",
                        "facecolor": "#FEF2F2",
                        "edgecolor": "#FCA5A5",
                        "alpha": 0.96,
                    },
                    zorder=5,
                )
            axis.set_title((attribute or "unlabeled").replace("_", " ").title())
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            f"{direction_label}: {readout.title()} projection",
            fontsize=14,
        )
        figure.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=7.5)
        figure.tight_layout(rect=(0.0, 0.23, 1.0, 0.95))
        figure.savefig(output_path, dpi=220, facecolor="white")
        plt.close(figure)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _numeric_comparison(current: Any, baseline: Any, prefix: str = "") -> dict[str, dict[str, float]]:
    comparisons: dict[str, dict[str, float]] = {}
    if isinstance(current, dict) and isinstance(baseline, dict):
        for key in sorted(set(current).intersection(baseline)):
            path = f"{prefix}.{key}" if prefix else str(key)
            comparisons.update(_numeric_comparison(current[key], baseline[key], path))
    elif (
        isinstance(current, (int, float))
        and not isinstance(current, bool)
        and isinstance(baseline, (int, float))
        and not isinstance(baseline, bool)
        and math.isfinite(float(current))
        and math.isfinite(float(baseline))
    ):
        comparisons[prefix] = {
            "baseline": float(baseline),
            "current": float(current),
            "delta": float(current) - float(baseline),
        }
    return comparisons


def _checkpoint_selection_summary(
    projections: dict[str, Any],
    reconstruction: dict[str, Any],
    baseline_comparison: dict[str, Any],
) -> dict[str, Any]:
    directions = {}
    for name, result in projections.items():
        directions[name] = {
            "projection_method": result["projection_method"],
            "projection_support": result["projection_support"],
            "projection_target_count": result["projection_target_count"],
            "mean_effective_target_count": result[
                "mean_conditional_effective_target_count"
            ],
            "mean_nearest_target_distance": result["mean_nearest_target_distance"],
            "mean_target_norm_ratio": result[
                "mean_projected_to_target_norm_ratio"
            ],
        }
    return {
        "primary_readout": "infoot_eq7_conditional_expectation",
        "directions": directions,
        "source_reconstruction": reconstruction,
        "baseline_comparison_status": baseline_comparison["status"],
    }


def _stage1a_baseline_required(
    evaluation_config: dict[str, Any], override: bool | None
) -> bool:
    if override is not None:
        return bool(override)
    comparison_config = _nested(evaluation_config, "comparison")
    return bool(comparison_config.get("require_stage1a_baseline", True))


def run_stage1b_evaluation(
    alignment_config_path: str | Path,
    evaluation_config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    weights: str = "ema",
    device_cat: str | None = None,
    device_dog: str | None = None,
    max_reference: int | None = None,
    max_projection: int | None = None,
    max_query: int | None = None,
    projection_bandwidth: float | None = None,
    require_stage1a_baseline: bool | None = None,
) -> Stage1BEvaluationReport:
    alignment_path = Path(alignment_config_path).resolve()
    evaluation_path = Path(evaluation_config_path).resolve()
    alignment_config = load_yaml_config(alignment_path)
    evaluation_config = load_yaml_config(evaluation_path)
    rms_options = projection_rms_options(alignment_config)
    if projection_rms_options(evaluation_config) != rms_options:
        raise ValueError("Evaluation must match the training projection_rms protocol.")
    matching_config = dict(_nested(evaluation_config, "matching"))
    bandwidth, projection_bandwidth = _evaluation_bandwidths(
        matching_config, projection_bandwidth
    )
    _warn_fit_bandwidth_mismatch(alignment_config, bandwidth)
    # Store the effective override before hashing the evaluation protocol so
    # each projection bandwidth gets its own outputs and matching baseline.
    matching_config["projection_bandwidth_multiplier"] = projection_bandwidth
    evaluation_config["matching"] = matching_config
    if matching_head_spec(alignment_config)["enabled"]:
        evaluation_config["matching_head"] = matching_head_spec(alignment_config)
    from diffusion_ot.models.generator_adaptation import generator_adaptation_enabled
    if generator_adaptation_enabled(alignment_config):
        evaluation_config["generator_adaptation"] = {"enabled": True, "checkpoint_format": 4}
    root = effective_project_root(
        alignment_config, fallback=find_project_root(alignment_path.parent)
    )
    eval_root = effective_project_root(evaluation_config, fallback=root)
    if root != eval_root:
        raise ValueError("Alignment and evaluation configs resolve to different project roots.")
    if str(alignment_config.get("stage")) not in {"stage1b_plain_infoot", "stage1b_fused_infoot"}:
        raise ValueError("The quick evaluator supports Stage 1B plain and fused InfoOT.")
    from diffusion_ot.losses.semantic_prior import load_semantic_prior
    # Evaluation chooses a solver variant explicitly; fitted prior provenance
    # participates in the protocol hash, including the actual descriptor bytes.
    eval_variant = str(_nested(evaluation_config, "infoot").get("variant", "plain"))
    prior_config = dict(alignment_config)
    prior_config["infoot"] = dict(_nested(evaluation_config, "infoot"))
    prior_config["infoot"]["variant"] = eval_variant
    eval_cost_source = str(prior_config["infoot"].get("cross_cost_source", "dino"))
    training_cost_source = str(_nested(alignment_config, "infoot").get("cross_cost_source", "dino"))
    if eval_cost_source != training_cost_source:
        raise ValueError("Evaluation must match the training infoot.cross_cost_source.")
    prior = load_semantic_prior(prior_config, root)
    if prior is not None:
        evaluation_config["semantic_prior"] = {
            "fingerprint": prior.fingerprint, "metadata": prior.metadata,
        }
    if weights not in {"ema", "raw"}:
        raise ValueError("weights must be ema or raw.")

    resolved_checkpoint = None
    if checkpoint_path is not None:
        candidate = Path(checkpoint_path)
        resolved_checkpoint = (
            candidate.resolve()
            if candidate.is_absolute()
            else resolve_project_local_path(candidate, root, field_name="checkpoint_path")
        )
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(f"Stage 1B checkpoint not found: {resolved_checkpoint}")

    contexts = {
        "cat": _load_domain_context(
            alignment_config,
            root,
            "cat",
            checkpoint_path=resolved_checkpoint,
            joint_weights=weights,
            device_override=device_cat,
        ),
        "dog": _load_domain_context(
            alignment_config,
            root,
            "dog",
            checkpoint_path=resolved_checkpoint,
            joint_weights=weights,
            device_override=device_dog,
        ),
    }
    protocol_identifier = _protocol_identifier(
        evaluation_config,
        max_reference=max_reference,
        max_projection=max_projection,
        max_query=max_query,
    )
    baseline_identifier = "stage1a_" + "_".join(
        _checkpoint_identifier(
            contexts[domain].stage1a_checkpoint_path,
            contexts[domain].stage1a_checkpoint_step,
            contexts[domain].stage1a_weights,
        )[:8]
        for domain in ("cat", "dog")
    ) + f"_protocol_{protocol_identifier}"
    mode = "stage1a_offline_infoot" if resolved_checkpoint is None else str(alignment_config["stage"])
    identifier = (
        baseline_identifier
        if resolved_checkpoint is None
        else (
            f"stage1b_{_checkpoint_identifier(resolved_checkpoint, contexts['cat'].checkpoint_step, weights)[:8]}"
            f"_step_{contexts['cat'].checkpoint_step:06d}_{weights}_protocol_{protocol_identifier}"
        )
    )
    output_base = resolve_project_local_path(
        evaluation_config.get("output_dir", "outputs/stage1b_eval"), root, field_name="output_dir"
    )
    output_root = output_base / identifier
    baseline_report_path = output_base / baseline_identifier / "evaluation_report.json"
    baseline_required = _stage1a_baseline_required(
        evaluation_config, require_stage1a_baseline
    )
    if (
        resolved_checkpoint is not None
        and baseline_required
        and not baseline_report_path.is_file()
    ):
        raise FileNotFoundError(
            "Matching Stage 1A + offline-InfoOT baseline is missing. Run the evaluator "
            f"without --checkpoint first: {baseline_report_path}"
        )
    data_config = _nested(evaluation_config, "data")
    reference_split = str(data_config.get("reference_split", "train"))
    projection_split = str(data_config.get("projection_split", reference_split))
    query_split = str(data_config.get("query_split", "val"))
    if reference_split == query_split:
        raise ValueError("Train-only evaluation requires different reference and query splits.")
    if rms_options is not None and (reference_split != "train" or projection_split != "train"):
        raise ValueError("Reference-EMA projection uses training reference and projection banks only.")
    seed = int(evaluation_config.get("seed", 20260906))
    batch_size = int(data_config.get("batch_size", 32))
    reference_count = int(max_reference or data_config.get("reference_samples_per_domain", 512))
    configured_projection_count = data_config.get("projection_samples_per_domain")
    projection_count = max_projection if max_projection is not None else configured_projection_count
    projection_count = None if projection_count in {None, "all"} else int(projection_count)
    query_count = int(max_query or data_config.get("query_samples_per_domain", 256))

    banks: dict[str, dict[str, LatentBank]] = {"cat": {}, "dog": {}}
    for domain_index, domain in enumerate(("cat", "dog")):
        context = contexts[domain]
        checkpoint_id = _checkpoint_identifier(
            context.checkpoint_path, context.checkpoint_step, context.weights
        )
        projection_dataset = _dataset(context, projection_split, root)
        banks[domain]["projection"] = build_latent_bank(
            context.branch.encoder,
            projection_dataset,
            domain=domain,
            split=projection_split,
            count=projection_count,
            seed=seed + domain_index * 1000 + 50,
            batch_size=batch_size,
            device=context.device,
            dtype=context.dtype,
            checkpoint_id=checkpoint_id,
            matching_head=getattr(context, "matching_head", None),
        )
        if reference_split == projection_split:
            banks[domain]["reference"] = subset_latent_bank(
                banks[domain]["projection"],
                count=reference_count,
                seed=seed + domain_index * 1000,
            )
        else:
            banks[domain]["reference"] = build_latent_bank(
                context.branch.encoder,
                _dataset(context, reference_split, root),
                domain=domain,
                split=reference_split,
                count=reference_count,
                seed=seed + domain_index * 1000,
                batch_size=batch_size,
                device=context.device,
                dtype=context.dtype,
                checkpoint_id=checkpoint_id,
                matching_head=getattr(context, "matching_head", None),
            )
        banks[domain]["query"] = build_latent_bank(
            context.branch.encoder,
            _dataset(context, query_split, root),
            domain=domain,
            split=query_split,
            count=query_count,
            seed=seed + domain_index * 1000 + 100,
            batch_size=batch_size,
            device=context.device,
            dtype=context.dtype,
            checkpoint_id=checkpoint_id,
            matching_head=getattr(context, "matching_head", None),
        )
        validate_bank_compatibility(banks[domain]["reference"], banks[domain]["query"])
        validate_bank_compatibility(
            banks[domain]["reference"], banks[domain]["projection"], require_disjoint=False
        )
        validate_bank_compatibility(banks[domain]["projection"], banks[domain]["query"])

    distance_scale_mode = str(matching_config.get("distance_scale", "infoot_rms"))
    if distance_scale_mode == "infoot_rms":
        scales = {
            domain: float(
                infoot_distance_scale(banks[domain]["reference"].matching_features)
            )
            for domain in ("cat", "dog")
        }
    elif distance_scale_mode == "fixed_stage1a_median":
        scales = {
            domain: float(
                median_distance_scale(banks[domain]["reference"].matching_features)
            )
            for domain in ("cat", "dog")
        }
    else:
        raise ValueError(
            "matching.distance_scale must be 'infoot_rms' or "
            "'fixed_stage1a_median'."
        )
    alignment_device = str(evaluation_config.get("alignment_device", device_cat or "cpu"))
    projection_scales, rms_report = None, None
    if rms_options is not None:
        if distance_scale_mode != "infoot_rms":
            raise ValueError("Reference-EMA projection requires matching.distance_scale=infoot_rms.")
        if resolved_checkpoint is not None:
            trackers = [contexts[d].projection_rms for d in ("cat", "dog")]
            if any(t is None for t in trackers) or trackers[0].state_dict() != trackers[1].state_dict():
                raise ValueError("Evaluation needs matching checkpoint projection RMS states for both domains.")
            projection_scales = trackers[0].scales()
            rms_report = {**trackers[0].diagnostics(bandwidth=projection_bandwidth),
                          "weights": weights, "source": "checkpoint", "frozen": True}
        else:
            # Offline Stage 1A baseline has no training history to restore.
            # Calibrate once on its own training references, never its queries.
            variances = reference_variances({d: banks[d]["reference"].matching_features for d in banks})
            projection_scales = {d: math.sqrt(max(v, rms_options["eps"] ** 2)) for d, v in variances.items()}
            rms_report = {**rms_options, "scales": projection_scales, "num_updates": 0,
                          "source": "stage1a_training_references", "frozen": True}
    cat_features = banks["cat"]["reference"].matching_features.to(alignment_device)
    dog_features = banks["dog"]["reference"].matching_features.to(alignment_device)
    infoot_config = _nested(evaluation_config, "infoot")
    cross_cost = None
    if prior is not None:
        descriptors = {
            domain: prior.lookup(banks[domain]["reference"].sample_ids, domain, reference_split).to(alignment_device)
            for domain in ("cat", "dog")
        }
        if reference_split != "train":
            raise ValueError("Fused prior fitting is restricted to training references.")
        cross_cost = prior.cost(descriptors["cat"], descriptors["dog"])
    elif eval_variant == "fused" and eval_cost_source == "encoder":
        from diffusion_ot.losses.encoder_transport import encoder_transport_cost
        cross_cost = encoder_transport_cost(cat_features, dog_features)
    solution = solve_infoot(
        cat_features,
        dog_features,
        bandwidth=bandwidth,
        distance_scale_x=scales["cat"],
        distance_scale_y=scales["dog"],
        cross_cost=cross_cost,
        cross_cost_weight=float(infoot_config.get("cross_cost_weight", 1.0)),
        **solver_kwargs(infoot_config),
    )

    banks_dir = output_root / "banks"
    for domain in ("cat", "dog"):
        for kind, bank in banks[domain].items():
            save_latent_bank(banks_dir / f"{domain}_{kind}.pt", bank)
    torch.save(
        {
            "coupling": solution.coupling.cpu(),
            "variant": eval_variant,
            "cross_cost_source": eval_cost_source if eval_variant == "fused" else None,
            "semantic_prior_fingerprint": prior.fingerprint if prior is not None else None,
            "cross_cost_weight": float(infoot_config.get("cross_cost_weight", 1.0)),
            "cat_reference_ids": banks["cat"]["reference"].sample_ids,
            "dog_reference_ids": banks["dog"]["reference"].sample_ids,
            "cat_projection_ids": banks["cat"]["projection"].sample_ids,
            "dog_projection_ids": banks["dog"]["projection"].sample_ids,
            "matching_head_config": matching_head_spec(alignment_config),
            "matching_representation_ids": {d: banks[d]["reference"].matching_id for d in ("cat", "dog")},
            "distance_scales": scales,
            "distance_scale_mode": distance_scale_mode,
            "bandwidth": bandwidth,
            "projection_bandwidth": projection_bandwidth,
            **({"projection_rms": rms_report} if rms_report is not None else {}),
            "solver_algorithm": str(
                infoot_config.get("algorithm", "official_projected_sinkhorn")
            ),
            "entropy_epsilon": float(infoot_config.get("entropy_epsilon", 0.05)),
        },
        output_root / "coupling.pt",
    )

    proxy_config = _nested(evaluation_config, "proxy_labels")
    proxy_path_value = proxy_config.get("path")
    proxy_path = (
        resolve_project_local_path(proxy_path_value, root, field_name="proxy_labels.path")
        if proxy_path_value
        else None
    )
    external_labels = load_proxy_labels(
        proxy_path, sample_id_key=str(proxy_config.get("sample_id_key", "sample_id"))
    )
    labels: dict[str, dict[str, Any]] = {}
    for domain_banks in banks.values():
        for bank in domain_banks.values():
            labels.update(_labels_for_bank(bank, external_labels))
    attributes = [str(value) for value in proxy_config.get("attributes", [])]
    projection_config = _nested(evaluation_config, "projection")

    projections: dict[str, Any] = {}
    direction_tensors: dict[str, dict[str, torch.Tensor]] = {}
    directions = {
        "cat_to_dog": ("cat", "dog", solution.coupling),
        "dog_to_cat": ("dog", "cat", solution.coupling.transpose(0, 1)),
    }
    for name in projection_config.get("directions", list(directions)):
        source_domain, target_domain, coupling = directions[str(name)]
        direction_report, tensors = _direction_projection_evaluation(
            banks[source_domain]["reference"],
            banks[target_domain]["reference"],
            banks[source_domain]["query"],
            coupling,
            target_projection=banks[target_domain]["projection"],
            source_scale=scales[source_domain],
            target_scale=scales[target_domain],
            bandwidth=bandwidth,
            eps=float(infoot_config.get("numerical_epsilon", 1.0e-8)),
            distance_scale_mode=distance_scale_mode,
            projection_bandwidth=projection_bandwidth,
            projection_scales=projection_scales,
        )
        projections[str(name)] = direction_report
        direction_tensors[str(name)] = tensors
        if prior is not None and all(
            key in prior.index for bank in (banks[source_domain]["query"], banks[target_domain]["projection"])
            for key in bank.sample_ids
        ):
            # These diagnostics measure the frozen training prior, not
            # independent validation or generated-image structure.
            query_structure = prior.lookup(
                banks[source_domain]["query"].sample_ids, source_domain, query_split
            )
            target_structure = prior.lookup(
                banks[target_domain]["projection"].sample_ids, target_domain, projection_split
            )
            structure_costs = prior.cost(query_structure, target_structure)
            conditional_weights = tensors["conditional_weights"].detach().cpu()
            direction_report["structure_prior_diagnostics"] = {
                "status": "available",
                "conditional_expected_cost": float((conditional_weights * structure_costs).sum(1).mean()),
                "uniform_expected_cost": float(structure_costs.mean()),
                "nearest_descriptor_cost": float(structure_costs.min(1).values.mean()),
                "semantic_prior_fingerprint": prior.fingerprint,
                "interpretation": "Frozen training-prior projection cost; not independent validation or decoded-image similarity.",
            }
            teacher_temperature = float(_nested(alignment_config, "conditional_structure").get("teacher_temperature", .05))
            if not math.isfinite(teacher_temperature) or teacher_temperature <= 0:
                raise ValueError("Teacher temperature must be finite and positive.")
            teacher_weights = torch.softmax(-structure_costs / teacher_temperature, dim=1)
            target_codes = banks[target_domain]["projection"].raw_codes
            tensors["structure_teacher_codes"] = weighted_target_codes(teacher_weights, target_codes)
            direction_report["structure_prior_diagnostics"].update({
                "teacher_temperature": teacher_temperature,
                "teacher_expected_cost": float((teacher_weights * structure_costs).sum(1).mean()),
                "teacher_variance_decomposition": conditional_variance_decomposition(teacher_weights, target_codes),
                "conditional_variance_decomposition": conditional_variance_decomposition(conditional_weights, target_codes),
            })
        elif prior is not None:
            direction_report["structure_prior_diagnostics"] = {
                "status": "unavailable",
                "reason": "Prior cache lacks some query or projection descriptors; transport evaluation remains available.",
                "semantic_prior_fingerprint": prior.fingerprint,
            }

    reconstruction: dict[str, Any] = {}
    reconstruction_config = _nested(evaluation_config, "reconstruction")
    if bool(reconstruction_config.get("enabled", True)):
        requested_metrics = set(reconstruction_config.get("metrics", []))
        for domain in ("cat", "dog"):
            reconstruction[domain] = _evaluate_reconstruction(
                contexts[domain],
                banks[domain]["query"],
                count=int(reconstruction_config.get("samples_per_domain", 8)),
                num_steps=int(reconstruction_config.get("num_steps", 50)),
                guidance_scale=float(reconstruction_config.get("guidance_scale", 1.0)),
                seed=seed + (10 if domain == "cat" else 20),
                include_lpips="lpips" in requested_metrics,
                output_path=output_root / "reconstruction" / f"{domain}_grid.png",
            )

    translation_paths: dict[str, str] = {}
    translation_config = _nested(evaluation_config, "translation")
    if bool(translation_config.get("enabled", True)):
        readouts = translation_config.get("readouts")
        if readouts is None:
            readouts = ["conditional_mean", "conditional_map", "conditional_sample"]
            if translation_config.get("include_structure_teacher_mean", False):
                readouts.append("structure_teacher_mean")
        include_source = bool(translation_config.get("include_source", True))
        image_features = None
        patch_options = None
        if translation_config.get("patchnce_metrics", False):
            from diffusion_ot.training.decoded_translation import self_supervised_translation_options
            image_options = alignment_config.get("decoded_translation") or {}
            if image_options.get("supervision") != "self_supervised" or image_options.get("objective") != "patchnce":
                raise ValueError("patchnce_metrics requires the self-supervised PatchNCE experiment.")
            patch_options = self_supervised_translation_options(image_options)
        if translation_config.get("decoded_structure_metrics", False):
            if prior is None:
                raise ValueError("Decoded structure evaluation requires a frozen semantic prior.")
            from diffusion_ot.training.decoded_translation import load_image_features
            image_features = load_image_features(prior, root, contexts["cat"].device, checkpoint_features=False)
        for name, (source_domain, target_domain, _) in directions.items():
            if name not in direction_tensors:
                continue
            path = output_root / "translation" / f"{name}_grid.png"
            teacher_codes = direction_tensors[name].get("structure_teacher_codes") if "structure_teacher_mean" in readouts else None
            decoded_metrics = _save_translation_grid(
                contexts[source_domain],
                contexts[target_domain],
                banks[source_domain]["query"],
                banks[target_domain]["projection"],
                direction_tensors[name]["conditional_weights"],
                count=int(translation_config.get("samples_per_direction", 8)),
                num_steps=int(translation_config.get("num_steps", 50)),
                guidance_scale=float(translation_config.get("guidance_scale", 1.0)),
                temperature=float(translation_config.get("temperature", 1.0)),
                seed=seed + (30 if name == "cat_to_dog" else 40),
                output_path=path,
                teacher_codes=teacher_codes,
                readouts=readouts,
                include_source=include_source,
                **({"image_features": image_features} if image_features is not None else {}),
                **({"color_histogram": translation_config["color_histogram"]}
                   if translation_config.get("color_histogram") is not None else {}),
                **({"patchnce_options": patch_options} if patch_options is not None else {}),
            )
            if decoded_metrics:
                projections[name]["decoded_image_diagnostics"] = decoded_metrics
            translation_paths[name] = str(path)
            projections[name]["translation_grid_rows"] = (
                (["source"] if include_source else [])
                + [_TRANSLATION_ROW_NAMES[readout] for readout in readouts]
            )
        del image_features

    visualization_paths: dict[str, str] = {}
    visualization_config = _nested(evaluation_config, "visualization")
    if bool(visualization_config.get("enabled", True)):
        for name, (source_domain, target_domain, _) in directions.items():
            if name not in direction_tensors:
                continue
            paths = {
                readout: output_root
                / "visualization"
                / f"{name}_{readout}_umap.png"
                for readout in ("conditional", "barycentric")
            }
            _save_umap_visualizations(
                banks[target_domain]["projection"],
                banks[source_domain]["query"].sample_ids,
                direction_tensors[name]["conditional_codes"],
                direction_tensors[name]["barycentric_codes"],
                labels=labels,
                attributes=attributes,
                direction=name,
                random_state=int(visualization_config.get("random_state", seed)),
                n_jobs=int(visualization_config.get("n_jobs", 1)),
                target_alpha=float(visualization_config.get("target_alpha", 0.30)),
                projection_alpha=float(
                    visualization_config.get("projection_alpha", 0.90)
                ),
                output_paths=paths,
            )
            for readout, path in paths.items():
                visualization_paths[f"{name}_{readout}"] = str(path)

    baseline_comparison: dict[str, Any]
    if resolved_checkpoint is None:
        baseline_comparison = {"status": "baseline", "baseline_report": str(output_root / "evaluation_report.json")}
    elif baseline_report_path.is_file():
        baseline_payload = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        current_payload = {
            "latent_diagnostics": {
                f"{domain}_{kind}": latent_diagnostics(bank)
                for domain, domain_banks in banks.items()
                for kind, bank in domain_banks.items()
            },
            "reconstruction": reconstruction,
            "projections": projections,
        }
        baseline_comparison = {
            "status": "compared",
            "baseline_report": str(baseline_report_path),
            "numeric_deltas": _numeric_comparison(current_payload, baseline_payload),
        }
    else:
        baseline_comparison = {"status": "missing", "baseline_report": str(baseline_report_path)}
    checkpoint_selection = _checkpoint_selection_summary(
        projections, reconstruction, baseline_comparison
    )

    report = Stage1BEvaluationReport(
        alignment_config_path=str(alignment_path),
        evaluation_config_path=str(evaluation_path),
        alignment_checkpoint=str(resolved_checkpoint) if resolved_checkpoint else None,
        mode=mode,
        output_dir=str(output_root),
        seed=seed,
        stage1a_architectures={
            domain: context.stage1a_architecture for domain, context in contexts.items()
        },
        generation_protocol={
            "reconstruction_reference": "original_dataset_rgb",
            "fit_bandwidth_multiplier": bandwidth,
            "projection_bandwidth_multiplier": projection_bandwidth,
            **({"projection_rms": rms_report} if rms_report is not None else {}),
            "semantic_cfg_enabled": all(
                context.stage1a_architecture["semantic_cfg_enabled"]
                for context in contexts.values()
            ),
            "attention_lora_enabled": all(
                context.stage1a_architecture["attention_lora"]["enabled"]
                for context in contexts.values()
            ),
            "reconstruction_guidance_scale": float(
                reconstruction_config.get("guidance_scale", 1.0)
            ),
            "translation_guidance_scale": float(
                translation_config.get("guidance_scale", 1.0)
            ),
            "stage1b_frozen_generator_components": [
                "base_transformer",
                "adaln_adapters",
                "attention_lora",
                "learned_null_token",
            ] if not generator_adaptation_enabled(alignment_config) else ["base_transformer", "learned_null_token", "vae"],
            "stage1b_trainable_generator_components": ["adaln_adapters", "token_mlps", "z_proj", "final_adapter", "attention_lora"]
                if generator_adaptation_enabled(alignment_config) else [],
            "generator_weights": weights if checkpoint_path is not None else "stage1a",
        },
        reference_sizes={domain: len(banks[domain]["reference"].sample_ids) for domain in banks},
        projection_sizes={domain: len(banks[domain]["projection"].sample_ids) for domain in banks},
        query_sizes={domain: len(banks[domain]["query"].sample_ids) for domain in banks},
        distance_scales=scales,
        solver={
            "variant": eval_variant,
            "cross_cost_source": eval_cost_source if eval_variant == "fused" else None,
            "semantic_prior_fingerprint": prior.fingerprint if prior is not None else None,
            "transport": transport_diagnostics(solution.coupling),
            **({"transport_encoder_cost": float((solution.coupling * cross_cost).sum()),
                "independent_encoder_cost": float(cross_cost.mean())} if eval_variant == "fused" and eval_cost_source == "encoder" else
               {"transport_structure_cost": float((solution.coupling * cross_cost).sum()) if cross_cost is not None else None,
                "independent_structure_cost": float(cross_cost.mean()) if cross_cost is not None else None}),
            "fit_bandwidth_multiplier": bandwidth,
            "algorithm": str(
                infoot_config.get("algorithm", "official_projected_sinkhorn")
            ),
            "initialization": str(
                infoot_config.get("initialization", "independent_product")
            ),
            "distance_scale_mode": distance_scale_mode,
            "objective": solution.objective,
            "mutual_information": solution.mutual_information,
            "entropy": solution.entropy,
            "row_residual": solution.row_residual,
            "column_residual": solution.column_residual,
            "iterations": solution.iterations,
            "iteration_budget": int(infoot_config.get("inner_iterations", 50)),
            "recovery_iteration_budget": int(infoot_config.get("recovery_iterations", 0)),
            "recovery_iterations": solution.recovery_iterations,
            "effective_projection_tolerance": solution.effective_projection_tolerance,
            "projection_iteration_budget": int(infoot_config.get("projection_iterations", 200)),
            "projection_tolerance": float(infoot_config.get("projection_tolerance", 1.0e-5)),
            "restart": solution.restart,
            "sinkhorn_converged": solution.sinkhorn_converged,
            "unconverged_inner_steps": solution.unconverged_inner_steps,
            "outer_converged": solution.outer_converged,
            "plan_delta_l1": solution.plan_delta_l1,
            "cat_kernel": kernel_offdiagonal_stats(
                cat_features,
                bandwidth=bandwidth,
                distance_scale=scales["cat"],
            ),
            "dog_kernel": kernel_offdiagonal_stats(
                dog_features,
                bandwidth=bandwidth,
                distance_scale=scales["dog"],
            ),
        },
        latent_diagnostics={
            f"{domain}_{kind}": latent_diagnostics(bank)
            for domain, domain_banks in banks.items()
            for kind, bank in domain_banks.items()
        },
        reconstruction=reconstruction,
        baseline_comparison=baseline_comparison,
        checkpoint_selection=checkpoint_selection,
        projections=projections,
        translation_grids=translation_paths,
        visualization_paths=visualization_paths,
    )
    _write_json(output_root / "evaluation_report.json", report.to_dict())
    return report
