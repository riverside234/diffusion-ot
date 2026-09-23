from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from diffusion_ot.integrations.hf_snapshot import (
    effective_project_root,
    find_project_root,
    load_yaml_config,
    resolve_project_local_path,
)


@dataclass
class JointInfoOTTrainReport:
    config_path: str
    output_dir: str
    initial_step: int
    final_step: int
    train_samples: dict[str, int]
    transport_batch_size: int
    reconstruction_batch_size: int
    checkpoint_path: str | None
    resumed_from: str | None
    train_log_path: str | None
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedTrainingDomain:
    domain: str
    branch: Any
    transformer: Any
    training_config: dict[str, Any]
    checkpoint_path: Path
    checkpoint_step: int
    stage1a_architecture: dict[str, Any]
    data_config_path: Path
    device: str
    dtype: torch.dtype
    vae: Any | None = None


class EncoderEMA:
    def __init__(
        self,
        encoders: dict[str, torch.nn.Module],
        *,
        decay: float = 0.9999,
        warmup_steps: int = 500,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.num_updates = 0
        self.shadow = {
            domain: {
                name: value.detach().clone()
                for name, value in encoder.state_dict().items()
            }
            for domain, encoder in encoders.items()
        }

    @property
    def effective_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        fraction = min(self.num_updates / self.warmup_steps, 1.0)
        return self.decay * fraction

    @torch.no_grad()
    def update(self, encoders: dict[str, torch.nn.Module]) -> None:
        self.num_updates += 1
        decay = self.effective_decay
        for domain, encoder in encoders.items():
            for name, value in encoder.state_dict().items():
                shadow = self.shadow[domain][name]
                source = value.detach().to(device=shadow.device, dtype=shadow.dtype)
                if torch.is_floating_point(shadow):
                    shadow.mul_(decay).add_(source, alpha=1.0 - decay)
                else:
                    shadow.copy_(source)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "num_updates": self.num_updates,
            "shadow": {
                domain: {name: value.detach().cpu() for name, value in state.items()}
                for domain, state in self.shadow.items()
            },
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        encoders: dict[str, torch.nn.Module],
    ) -> None:
        self.decay = float(state["decay"])
        self.warmup_steps = int(state["warmup_steps"])
        self.num_updates = int(state["num_updates"])
        incoming = state["shadow"]
        for domain, encoder in encoders.items():
            current = encoder.state_dict()
            if set(incoming[domain]) != set(current):
                raise ValueError(f"EMA state does not match the {domain} encoder.")
            self.shadow[domain] = {
                name: incoming[domain][name].to(device=value.device, dtype=value.dtype)
                for name, value in current.items()
            }

    def export(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            domain: {name: value.detach().cpu() for name, value in state.items()}
            for domain, state in self.shadow.items()
        }


def _nested(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _cycle(loader):
    while True:
        yield from loader


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _cpu_nested_state_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _cpu_nested_state_dict(item) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def freeze_generator_train_encoder(branch: Any) -> None:
    for parameter in branch.encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in branch.semantic_transformer.parameters():
        parameter.requires_grad_(False)
    semantic_conditioner = getattr(branch, "semantic_conditioner", None)
    if semantic_conditioner is not None:
        for parameter in semantic_conditioner.parameters():
            parameter.requires_grad_(False)
        semantic_conditioner.eval()
    branch.encoder.train()
    branch.semantic_transformer.eval()


def _fixed_generator_state_dict(branch: Any) -> dict[str, Any]:
    if hasattr(branch, "generator_state_dict"):
        return branch.generator_state_dict()
    return branch.semantic_transformer.trainable_state_dict()


def encoder_anchor_loss(
    current: torch.Tensor,
    reference: torch.Tensor,
    variance: float | torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    scale = torch.as_tensor(variance, device=current.device, dtype=current.dtype).clamp_min(eps)
    return (current - reference.detach()).square().mean() / scale


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(squared)


def _autograd_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    if not parameters or not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = 0.0
    for gradient in gradients:
        if gradient is not None:
            squared += float(gradient.detach().float().square().sum().cpu())
    return math.sqrt(squared)


def _autograd_group_norms(loss: torch.Tensor, groups: dict[str, list[torch.nn.Parameter]]) -> dict[str, float]:
    """Measure all groups with one graph traversal; do not assign grads."""
    parameters = [p for group in groups.values() for p in group]
    if not parameters or not loss.requires_grad:
        return {name: 0.0 for name in groups}
    gradients = iter(torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True))
    norms = {}
    for name, group in groups.items():
        squared = 0.0
        for _ in group:
            gradient = next(gradients)
            if gradient is not None:
                squared += float(gradient.detach().float().square().sum())
        norms[name] = math.sqrt(squared)
    return norms


@torch.no_grad()
def _clip_gradient_norm(parameters: list[torch.nn.Parameter], maximum: float | None) -> float:
    norm = _gradient_norm(parameters)
    if maximum is not None and norm > float(maximum):
        multiplier = float(maximum) / max(norm, 1.0e-12)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(multiplier)
    return norm


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torch_dtype_from_model(model: Any) -> torch.dtype:
    for parameter in model.parameters():
        return parameter.dtype
    return torch.float32


def _load_training_domain(
    config: dict[str, Any],
    root: Path,
    domain: str,
    *,
    device_override: str | None,
) -> LoadedTrainingDomain:
    from diffusion_ot.evaluation.stage1a_eval import (
        _apply_ema_weights,
        _load_checkpoint,
        validate_stage1a_architecture,
    )
    from diffusion_ot.integrations.sit_diffusers import load_sit_components, validate_transformer_config
    from diffusion_ot.models.pdae_sit import build_pdae_sit_branch

    stage1a_config = _nested(config, "stage1a")
    domain_config = _nested(stage1a_config, domain)
    train_config_path = resolve_project_local_path(
        domain_config["config"], root, field_name=f"stage1a.{domain}.config"
    )
    train_config = load_yaml_config(train_config_path)
    if str(train_config.get("domain", "")).lower() != domain:
        raise ValueError(f"Stage 1A config for {domain} has a mismatched domain.")
    model_config_path = resolve_project_local_path(
        train_config["model_config"], root, field_name=f"stage1a.{domain}.model_config"
    )
    data_config_path = resolve_project_local_path(
        train_config["data_config"], root, field_name=f"stage1a.{domain}.data_config"
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
    transformer = components.transformer
    transformer.eval()
    mismatches = validate_transformer_config(
        transformer, model_config.get("expected_transformer_config") or {}
    )
    if mismatches:
        raise ValueError("Unexpected SiT transformer config:\n" + "\n".join(mismatches))
    branch = build_pdae_sit_branch(transformer, model_config=model_config, stage_config=train_config)
    dtype = _torch_dtype_from_model(transformer)
    branch.to(device=device, dtype=dtype)
    checkpoint_path = resolve_project_local_path(
        domain_config["checkpoint"], root, field_name=f"stage1a.{domain}.checkpoint"
    )
    checkpoint = _load_checkpoint(checkpoint_path)
    branch.load_pdae_state_dict(checkpoint["model"])
    if str(stage1a_config.get("weights", "ema")) == "ema":
        _apply_ema_weights(branch, checkpoint)
    stage1a_architecture = validate_stage1a_architecture(branch, stage1a_config)
    freeze_generator_train_encoder(branch)
    return LoadedTrainingDomain(
        domain=domain,
        branch=branch,
        transformer=transformer,
        training_config=train_config,
        checkpoint_path=checkpoint_path,
        checkpoint_step=int(checkpoint.get("step", 0)),
        stage1a_architecture=stage1a_architecture,
        data_config_path=data_config_path,
        device=device,
        dtype=dtype,
        vae=components.vae if (config.get("decoded_translation") or {}).get("enabled", False) else None,
    )


@torch.inference_mode()
def _calibrate_encoder(
    encoder: torch.nn.Module,
    dataset: Any,
    *,
    count: int,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    distance_scale_mode: str,
) -> tuple[float, float]:
    from diffusion_ot.losses.infoot import (
        infoot_distance_scale,
        median_distance_scale,
        normalize_matching_features,
    )

    values: list[torch.Tensor] = []
    count = min(int(count), len(dataset))
    for offset in range(0, count, int(batch_size)):
        items = [dataset[index] for index in range(offset, min(offset + int(batch_size), count))]
        x0 = torch.stack([item["x0_latent"] for item in items]).to(device=device, dtype=dtype)
        values.append(encoder(x0).float().cpu())
    codes = torch.cat(values, dim=0)
    variance = float(codes.var(dim=0, unbiased=False).mean().clamp_min(1.0e-8))
    matching_features = normalize_matching_features(codes)
    if distance_scale_mode == "infoot_rms":
        distance_scale = float(infoot_distance_scale(matching_features))
    elif distance_scale_mode == "fixed_stage1a_median":
        distance_scale = float(median_distance_scale(matching_features))
    else:
        raise ValueError(
            "matching.distance_scale must be 'infoot_rms' or "
            "'fixed_stage1a_median'."
        )
    return variance, distance_scale


def _reconstruction_loss(
    domain: LoadedTrainingDomain,
    x0: torch.Tensor,
    z: torch.Tensor,
    *,
    semantic_dropout: bool = False,
    force_null: bool = False,
    generator_parameters: dict[str, torch.Tensor] | None = None,
    diagnostics: dict[str, float] | None = None,
) -> torch.Tensor:
    from diffusion_ot.losses.pdae_flow import (
        make_linear_flow_target,
        pdae_flow_snr_weight,
        pdae_velocity_gap_loss,
    )
    from diffusion_ot.models.pdae_sit import make_null_class_labels

    flow_config = _nested(domain.training_config, "flow")
    class_config = _nested(domain.training_config, "class_conditioning")
    weighting = _nested(domain.training_config, "loss_weighting")
    direction = str(flow_config.get("direction", "noise_to_data"))
    target = make_linear_flow_target(
        x0,
        eps=float(flow_config.get("time_eps", 1.0e-5)),
        direction=direction,
    )
    labels = make_null_class_labels(
        domain.transformer,
        batch_size=x0.shape[0],
        device=x0.device,
        null_label=class_config.get("null_label"),
    )
    if force_null:
        z = domain.branch.semantic_null_like(z)
    elif semantic_dropout:
        z, drop_mask = domain.branch.condition_semantic_z(z, apply_dropout=True)
        if diagnostics is not None:
            diagnostics["semantic_dropout_fraction"] = float(drop_mask.float().mean())
    if generator_parameters is None:
        output = domain.branch.predict_with_z(target.x_t, target.t, z, class_labels=labels)
    else:
        from diffusion_ot.models.generator_adaptation import predict_with_parameters
        output = predict_with_parameters(domain.branch, generator_parameters,
                                          target.x_t, target.t, z, labels)
    weight = pdae_flow_snr_weight(
        t=target.t.float(),
        direction=direction,
        gamma=float(weighting.get("gamma", 0.1)),
        normalize_mean_to=weighting.get("normalize_mean_to", 1.0),
        normalization_mode=str(weighting.get("normalization_mode", "fixed_uniform")),
        normalization_samples=int(weighting.get("normalization_samples", 65536)),
        clamp_min=weighting.get("clamp_min", 0.001),
        clamp_max=weighting.get("clamp_max"),
    )
    return pdae_velocity_gap_loss(
        output.delta_sample.float(),
        target.target_v.float(),
        output.base_sample.float(),
        weight=weight.float(),
    )


@torch.no_grad()
def fixed_reconstruction_probe(
    domains: dict[str, LoadedTrainingDomain],
    anchors: dict[str, torch.nn.Module],
    inputs: dict[str, torch.Tensor],
    *, seed: int,
    generator_baselines: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, float]:
    """Fixed validation images/noise/times; preserve the training RNG stream."""
    result = {}
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        for offset, (domain, value) in enumerate(domains.items()):
            encoder = value.branch.encoder
            was_training = encoder.training
            try:
                encoder.eval()
                x = inputs[domain].to(device=value.device, dtype=value.dtype)
                z = encoder(x)
                reference = anchors[domain](x)
                torch.manual_seed(seed + offset)
                result[f"{domain}_raw_reconstruction"] = float(_reconstruction_loss(value, x, z))
                torch.manual_seed(seed + offset)
                if generator_baselines is None:
                    result[f"{domain}_stage1a_reconstruction"] = float(_reconstruction_loss(value, x, reference))
                else:
                    result[f"{domain}_stage1a_reconstruction"] = float(_reconstruction_loss(
                        value, x, reference, generator_parameters=generator_baselines[domain]))
                    torch.manual_seed(seed + offset)
                    result[f"{domain}_null_reconstruction"] = float(_reconstruction_loss(value, x, z, force_null=True))
                    torch.manual_seed(seed + offset)
                    result[f"{domain}_stage1a_null_reconstruction"] = float(_reconstruction_loss(
                        value, x, reference, force_null=True, generator_parameters=generator_baselines[domain]))
                    # Paired cross probes locate E/G drift using the same
                    # latents, flow times and noise as the current/baseline pair.
                    torch.manual_seed(seed + offset)
                    result[f"{domain}_stage1a_encoder_current_generator_reconstruction"] = float(
                        _reconstruction_loss(value, x, reference))
                    torch.manual_seed(seed + offset)
                    result[f"{domain}_current_encoder_stage1a_generator_reconstruction"] = float(
                        _reconstruction_loss(value, x, z, generator_parameters=generator_baselines[domain]))
            finally:
                encoder.train(was_training)
    return result


def _build_checkpoint_payload(
    *,
    step: int,
    encoders: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    ema: EncoderEMA | None,
    config: dict[str, Any],
    domains: dict[str, LoadedTrainingDomain],
    train_state: dict[str, Any],
    loader_generators: dict[str, torch.Generator],
    matching_heads: dict[str, torch.nn.Module] | None = None,
    matching_ema: EncoderEMA | None = None,
    decoder_training: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 3 if matching_heads else 2,
        "stage": str(config.get("stage", "stage1b_plain_infoot")),
        "step": int(step),
        "encoders": {domain: _cpu_state_dict(encoder) for domain, encoder in encoders.items()},
        "fixed_generators": {
            domain: _cpu_nested_state_dict(
                _fixed_generator_state_dict(value.branch)
            )
            for domain, value in domains.items()
        },
        "encoder_ema": ema.export() if ema is not None else None,
        "ema_state": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "config": config,
        "stage1a_provenance": {
            domain: {
                "checkpoint_path": str(value.checkpoint_path),
                "checkpoint_step": value.checkpoint_step,
                "weights": str(_nested(config, "stage1a").get("weights", "ema")),
                "architecture": value.stage1a_architecture,
            }
            for domain, value in domains.items()
        },
        "train_state": train_state,
        "rng_state": torch.get_rng_state(),
        "dataloader_generator_states": {
            domain: generator.get_state() for domain, generator in loader_generators.items()
        },
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    if matching_heads:
        payload["matching_heads"] = {d: _cpu_state_dict(head) for d, head in matching_heads.items()}
        payload["matching_head_ema"] = matching_ema.export() if matching_ema is not None else None
        payload["matching_head_ema_state"] = matching_ema.state_dict() if matching_ema is not None else None
    if decoder_training is not None:
        payload["format_version"] = 4
        payload.update(decoder_training.checkpoint_state())
    return payload


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("stage") not in {"stage1b_plain_infoot", "stage1b_fused_infoot"}:
        raise ValueError(f"Not a Stage 1B InfoOT checkpoint: {path}")
    return payload


def _resolve_resume(value: str | Path | None, root: Path, latest: Path) -> Path | None:
    if value is None:
        return None
    if str(value).lower() == "latest":
        return latest
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else resolve_project_local_path(
        candidate, root, field_name="resume_from"
    )


def _validate_resume_provenance(
    checkpoint: dict[str, Any],
    domains: dict[str, LoadedTrainingDomain],
    *,
    weights: str,
) -> None:
    provenance = checkpoint.get("stage1a_provenance") or {}
    fixed_generators = checkpoint.get("fixed_generators") or {}
    for domain, value in domains.items():
        record = provenance.get(domain)
        if not isinstance(record, dict):
            raise ValueError(f"Resume checkpoint has no stage1a_provenance.{domain} record.")
        expected_path = Path(str(record.get("checkpoint_path", ""))).resolve()
        actual_path = Path(value.checkpoint_path).resolve()
        mismatches = []
        if expected_path != actual_path:
            mismatches.append(f"path {actual_path} != {expected_path}")
        if int(record.get("checkpoint_step", -1)) != int(value.checkpoint_step):
            mismatches.append(
                f"step {value.checkpoint_step} != {record.get('checkpoint_step')}"
            )
        if str(record.get("weights")) != str(weights):
            mismatches.append(f"weights {weights} != {record.get('weights')}")
        if domain not in fixed_generators:
            mismatches.append("fixed generator state is missing")
        saved_architecture = record.get("architecture")
        if (
            saved_architecture is not None
            and saved_architecture != value.stage1a_architecture
        ):
            mismatches.append(
                f"architecture {value.stage1a_architecture} != {saved_architecture}"
            )
        if mismatches:
            raise ValueError(
                f"Configured Stage 1A {domain} state does not match the resume checkpoint: "
                + "; ".join(mismatches)
            )


@torch.no_grad()
def fixed_conditional_structure_probe(
    domains: dict[str, LoadedTrainingDomain],
    inputs: dict[str, dict[str, Any]],
    *,
    cost_scale: float,
    bandwidth: float,
    projection_bandwidth: float,
    teacher_temperature: float,
    solver_options: dict[str, Any],
    cross_cost_weight: float = 1.0,
    anchors: dict[str, torch.nn.Module] | None = None,
    anchor_variances: dict[str, float] | None = None,
    projection_support_options: dict[str, Any] | None = None,
    matching_heads: dict[str, torch.nn.Module] | None = None,
    matching_regularization_options: dict[str, float] | None = None,
    matching_contrastive_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fixed train references / validation queries; no updates or RNG consumption."""
    from diffusion_ot.losses.conditional_structure import conditional_structure_loss
    from diffusion_ot.losses.infoot import infoot_distance_scale, solve_infoot, transport_diagnostics
    from diffusion_ot.models.matching_head import matching_features, matching_geometry_diagnostics

    states = {domain: value.branch.encoder.training for domain, value in domains.items()}
    references, queries, reference_structure, query_structure, anchor_codes = {}, {}, {}, {}, {}
    alignment_device = domains["cat"].device
    try:
        for domain, value in domains.items():
            batch = inputs[domain]
            if set(batch["reference_ids"]) & set(batch["query_ids"]):
                raise ValueError("Projection probe reference/query IDs overlap.")
            value.branch.encoder.eval()
            for kind, codes in (("reference", references), ("query", queries)):
                latents = batch[f"{kind}_latents"]
                codes[domain] = torch.cat([
                    value.branch.encoder(part.to(value.device, dtype=value.dtype)).float().to(alignment_device)
                    for part in latents.split(32)
                ])
            reference_structure[domain] = batch["reference_structure"].to(alignment_device)
            query_structure[domain] = batch["query_structure"].to(alignment_device)
            if projection_support_options is not None:
                if anchors is None or anchor_variances is None:
                    raise ValueError("Projection-support validation requires fixed Stage 1A anchors.")
                anchor_codes[domain] = torch.cat([
                    anchors[domain](part.to(value.device, dtype=value.dtype)).float().to(alignment_device)
                    for part in batch["reference_latents"].split(32)
                ])
        features, query_features = {}, {}
        for domain, value in domains.items():
            head = (matching_heads or {}).get(domain)
            features[domain] = matching_features(references[domain].to(value.device), head).to(alignment_device)
            query_features[domain] = matching_features(queries[domain].to(value.device), head).to(alignment_device)
        solution = solve_infoot(
            features["cat"], features["dog"], bandwidth=bandwidth,
            distance_scale_x=infoot_distance_scale(features["cat"]),
            distance_scale_y=infoot_distance_scale(features["dog"]),
            cross_cost=torch.cdist(reference_structure["cat"], reference_structure["dog"]) / cost_scale,
            cross_cost_weight=cross_cost_weight, **solver_options,
        )
        result = conditional_structure_loss(
            references, queries, reference_structure, query_structure, solution.coupling,
            bandwidth=projection_bandwidth, cost_scale=cost_scale, teacher_temperature=teacher_temperature,
            reference_matching=features, query_matching=query_features,
        )
        metrics = {
            "conditional_structure_loss": float(result.loss),
            "conditional_structure": result.metrics,
            "projection_probe": {
                "reference_split": "train", "query_split": "val",
                "sample_ids": {domain: {kind: inputs[domain][f"{kind}_ids"] for kind in ("reference", "query")} for domain in domains},
                "fit_bandwidth": bandwidth, "projection_bandwidth": projection_bandwidth,
                "teacher_temperature": teacher_temperature, "structure_cost_scale": cost_scale,
                "sinkhorn_converged": solution.sinkhorn_converged,
                "row_residual": solution.row_residual, "column_residual": solution.column_residual,
                "iterations": solution.iterations, "outer_converged": solution.outer_converged,
                "iteration_budget": solver_options["inner_iterations"],
                "projection_iteration_budget": solver_options["projection_iterations"],
                "projection_tolerance": solver_options["projection_tolerance"],
                "plan_delta_l1": solution.plan_delta_l1,
                "transport": transport_diagnostics(solution.coupling),
                "matching_feature_variance": {
                    d: float(feature.var(0, unbiased=False).sum()) for d, feature in features.items()
                },
                "feature_geometry": {
                    d: matching_geometry_diagnostics(references[d], feature) for d, feature in features.items()
                },
            },
        }
        if projection_support_options is not None:
            from diffusion_ot.losses.projection_support import projection_support_loss
            support = projection_support_loss(
                result.weights, references, anchor_codes, anchor_variances,
                teacher_weights=result.teacher_weights, **projection_support_options
            )
            metrics["projection_support_loss"] = float(support.loss)
            metrics["projection_support"] = support.metrics
        if matching_regularization_options is not None:
            from diffusion_ot.losses.matching_regularization import matching_regularization_loss
            protection = matching_regularization_loss(features, **matching_regularization_options)
            metrics["matching_regularization_loss"] = float(protection.loss)
            metrics["matching_regularization"] = protection.metrics
        if matching_contrastive_options is not None:
            from diffusion_ot.losses.contrastive import matching_contrastive_loss
            contrastive, contrastive_metrics = matching_contrastive_loss(
                features, reference_structure, result.log_weights, result.teacher_weights,
                **matching_contrastive_options)
            metrics["matching_contrastive_loss"] = float(contrastive)
            metrics["matching_contrastive"] = contrastive_metrics
        return metrics
    finally:
        for domain, value in domains.items():
            value.branch.encoder.train(states[domain])


@torch.no_grad()
def fixed_decoded_translation_probe(decoder_training, domains, matching_heads, inputs, *,
                                    prior, bandwidth, projection_bandwidth, teacher_temperature,
                                    solver_options, seed, cross_cost_weight=1.0, image_dir=None):
    """Held-out decoded images; fixed noise, train-only OT references, no D update."""
    from diffusion_ot.losses.infoot import infoot_distance_scale, solve_infoot
    from diffusion_ot.losses.conditional_structure import conditional_structure_loss
    from diffusion_ot.models.matching_head import matching_features
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    modes = {d: v.branch.encoder.training for d, v in domains.items()}
    target_device = domains["cat"].device
    with torch.random.fork_rng(devices=devices):
        try:
            references, queries, ref_matching, query_matching = {}, {}, {}, {}
            reference_structure, query_structure = {}, {}
            for domain, value in domains.items():
                value.branch.encoder.eval()
                data = inputs[domain]
                references[domain] = value.branch.encoder(data["reference_latents"].to(value.device, dtype=value.dtype)).float().to(target_device)
                queries[domain] = value.branch.encoder(data["query_latents"].to(value.device, dtype=value.dtype)).float().to(target_device)
                ref_matching[domain] = matching_features(references[domain].to(value.device), matching_heads[domain]).to(target_device)
                query_matching[domain] = matching_features(queries[domain].to(value.device), matching_heads[domain]).to(target_device)
                reference_structure[domain] = data["reference_structure"].to(target_device)
                query_structure[domain] = data["query_structure"].to(target_device)
            solution = solve_infoot(ref_matching["cat"], ref_matching["dog"], bandwidth=bandwidth,
                                    distance_scale_x=float(infoot_distance_scale(ref_matching["cat"])),
                                    distance_scale_y=float(infoot_distance_scale(ref_matching["dog"])),
                                    cross_cost=prior.cost(reference_structure["cat"], reference_structure["dog"]),
                                    cross_cost_weight=cross_cost_weight, **solver_options)
            conditional = conditional_structure_loss(
                references, queries, reference_structure, query_structure, solution.coupling,
                cost_scale=prior.cost_scale, bandwidth=projection_bandwidth, teacher_temperature=teacher_temperature,
                reference_matching=ref_matching, query_matching=query_matching)
            _, metrics = decoder_training.loss(
                conditional.weights, references,
                {d: data["query_latents"] for d, data in inputs.items()},
                {d: data["reference_latents"] for d, data in inputs.items()},
                step=0, validation_seed=seed,
                source_reference_structure=reference_structure,
                source_query_structure=query_structure,
                source_query_metadata={d: data.get("query_metadata", []) for d, data in inputs.items()},
                **({"validation_image_dir": image_dir} if image_dir is not None else {}))
            metrics["seed"] = seed
            # This probe refits its own plan; do not borrow convergence from
            # the conditional probe, which can encode references in chunks.
            metrics["solver"] = {
                "iterations": solution.iterations,
                "iteration_budget": solver_options["inner_iterations"],
                "projection_iteration_budget": solver_options["projection_iterations"],
                "projection_tolerance": solver_options["projection_tolerance"],
                "sinkhorn_converged": solution.sinkhorn_converged,
                "outer_converged": solution.outer_converged,
                "plan_delta_l1": solution.plan_delta_l1,
                "row_residual": solution.row_residual,
                "column_residual": solution.column_residual,
            }
            metrics["reference_ids"] = {d: data["reference_ids"] for d, data in inputs.items()}
            count = int(decoder_training.options.get("batch_size", 4))
            metrics["query_ids"] = {d: data["query_ids"][:count] for d, data in inputs.items()}
            if (decoder_training.options.get("code_consistency_mode", "cosine") == "contrastive"
                    and float(decoder_training.options.get("code_consistency_weight", 0.0)) > 0):
                metrics["code_condition_bank_ids"] = {d: data["query_ids"] for d, data in inputs.items()}
            metrics["interpretation"] = "Decoded DINO, source-palette and optional code-recovery metrics are training signals, not independent semantic validation. RGB-uv measures whole-image palette, not spatial markings or exposure. Discriminator scores are not calibrated across checkpoints or critic kinds."
            return metrics
        finally:
            for domain, value in domains.items():
                value.branch.encoder.train(modes[domain])


def train_joint_infoot(
    config_path: str | Path,
    *,
    device_cat: str | None = None,
    device_dog: str | None = None,
    max_steps: int | None = None,
    resume_from: str | Path | None = None,
    dry_run: bool = False,
) -> JointInfoOTTrainReport:
    from torch.utils.data import DataLoader

    from diffusion_ot.data.latent_dataset import CachedLatentDataset, collate_latent_batch
    from diffusion_ot.losses.infoot import (
        infoot_distance_scale,
        kernel_offdiagonal_stats,
        normalize_matching_features,
        plain_infoot_feature_loss,
        solve_infoot,
        solver_kwargs,
        transport_diagnostics,
    )
    from diffusion_ot.losses.semantic_prior import (
        load_semantic_prior, neighborhood_distillation_loss, neighborhood_options, validate_prior_resume,
    )
    from diffusion_ot.losses.conditional_structure import conditional_structure_loss
    from diffusion_ot.losses.projection_support import projection_support_loss, support_options
    from diffusion_ot.losses.matching_regularization import (
        matching_regularization_loss, matching_regularization_options,
    )
    from diffusion_ot.training.gradient_guard import guarded_backward
    from diffusion_ot.losses.contrastive import matching_contrastive_loss, matching_contrastive_options
    from diffusion_ot.training.gradient_balance import (
        DecodedEncoderBalanceConfig, decoded_encoder_gradient_correction,
    )
    from diffusion_ot.training.gradient_diagnostics import GradientConflictMonitor, training_gradient_conflicts
    from diffusion_ot.training.pcgrad import PCGradConfig, pcgrad_backward
    from diffusion_ot.models.matching_head import make_matching_head, matching_head_spec, matching_features
    from diffusion_ot.models.generator_adaptation import generator_adaptation_enabled
    from diffusion_ot.training.decoded_translation import DecoderTraining, validate_decoder_config
    from diffusion_ot.training.stage1b_logging import Stage1BLogFormatter

    resolved_config_path = Path(config_path).resolve()
    config = load_yaml_config(resolved_config_path)
    if str(config.get("stage")) not in {"stage1b_plain_infoot", "stage1b_fused_infoot"}:
        raise ValueError("Expected a Stage 1B plain or fused InfoOT config.")
    root = effective_project_root(
        config, fallback=find_project_root(resolved_config_path.parent)
    )
    output_dir = resolve_project_local_path(
        config.get("output_dir", "outputs/stage1b_cat_dog_plain_infoot_sit_b2"),
        root,
        field_name="output_dir",
    )
    data_config = _nested(config, "data")
    train_config = _nested(config, "train")
    conflict_diagnostics_enabled = train_config.get("gradient_conflicts", False)
    if not isinstance(conflict_diagnostics_enabled, bool):
        raise ValueError("train.gradient_conflicts must be a boolean.")
    ema_config = _nested(config, "ema")
    loss_weights = _nested(config, "loss_weights")
    matching_config = _nested(config, "matching")
    scale_gradient = str(matching_config.get("distance_scale_gradient", "detached"))
    if scale_gradient not in {"detached", "full"}:
        raise ValueError("matching.distance_scale_gradient must be detached or full.")
    differentiate_distance_scale = scale_gradient == "full"
    if differentiate_distance_scale and matching_config.get("distance_scale", "infoot_rms") != "infoot_rms":
        raise ValueError("Full distance-scale gradients require matching.distance_scale=infoot_rms.")
    infoot_config = _nested(config, "infoot")
    conditional_config = _nested(config, "conditional_structure")
    conditional_enabled = bool(conditional_config.get("enabled", False))
    head_spec = matching_head_spec(config)
    head_config = _nested(config, "matching_head")
    matching_protection_options = matching_regularization_options(_nested(config, "matching_regularization"))
    contrastive_options = matching_contrastive_options(_nested(config, "matching_contrastive"))
    balance_config = DecodedEncoderBalanceConfig.from_mapping(_nested(config, "decoded_encoder_balance"))
    pcgrad_config = PCGradConfig.from_mapping(_nested(config, "pcgrad"))
    if head_spec["enabled"]:
        if not conditional_enabled:
            raise ValueError("Learned matching heads require conditional structure training.")
        if matching_config.get("distance_scale", "infoot_rms") != "infoot_rms":
            raise ValueError("Learned matching heads require matching.distance_scale=infoot_rms.")
        if not math.isfinite(float(head_config.get("lr", 2e-4))) or float(head_config.get("lr", 2e-4)) <= 0:
            raise ValueError("matching_head.lr must be finite and positive.")
    support_config = _nested(config, "projection_support")
    support_enabled = bool(support_config.get("enabled", False))
    gradient_guard_config = _nested(config, "gradient_guard")
    gradient_guard_enabled = bool(gradient_guard_config.get("enabled", False))
    if pcgrad_config.enabled and (balance_config.enabled or gradient_guard_enabled):
        raise ValueError("PCGrad cannot be combined with decoded encoder balance or the legacy gradient guard.")
    if balance_config.enabled:
        if gradient_guard_enabled:
            raise ValueError("Decoded encoder balance cannot be combined with the legacy gradient guard.")
        if not generator_adaptation_enabled(config) or not _nested(config, "decoded_translation").get("enabled", False):
            raise ValueError("Decoded encoder balance requires Experiment D decoded translation.")
    support_weight = float(loss_weights.get("projection_support", 0.0))
    if support_enabled:
        if not conditional_enabled or not math.isfinite(support_weight) or support_weight <= 0:
            raise ValueError("Projection support requires conditional structure training and a positive support loss weight.")
    elif support_weight != 0:
        raise ValueError("Enable projection_support for its nonzero loss weight.")
    expected_variant = "fused" if config["stage"] == "stage1b_fused_infoot" else "plain"
    if infoot_config.get("variant", "plain") != expected_variant:
        raise ValueError("stage and infoot.variant disagree.")
    prior = load_semantic_prior(config, root)
    if contrastive_options is not None and (prior is None or not conditional_enabled):
        raise ValueError("Matching contrastive supervision requires a frozen structure prior and conditional training.")
    if prior is not None:
        if float(data_config.get("random_horizontal_flip", 0.5)) != 0:
            raise ValueError("Cached structure descriptors require random_horizontal_flip: 0.0.")
        config["semantic_prior"]["fingerprint"] = prior.fingerprint
    trainable_config = _nested(config, "trainable")
    if not bool(trainable_config.get("encoders", True)):
        raise ValueError("Stage 1B-1 requires trainable encoders.")
    validate_decoder_config(config)

    seed = int(train_config.get("seed", 20260905))
    torch.manual_seed(seed)
    transport_batch_size = int(data_config.get("transport_batch_size", 64))
    reconstruction_batch_size = int(data_config.get("reconstruction_batch_size", 8))
    query_count = int(conditional_config.get("query_samples_per_domain", 32)) if conditional_enabled else 0
    if contrastive_options is not None:
        positives = contrastive_options["positive_count"]
        minimum_references = positives + (2 if contrastive_options["neighborhood_weight"] > 0 else 1)
        if transport_batch_size - query_count < minimum_references:
            raise ValueError("Matching contrastive supervision needs enough OT references for positives and negatives.")
        if int(train_config.get("validation_every", 0)) > 0 and int(
            conditional_config.get("validation_reference_samples", 96)
        ) < minimum_references:
            raise ValueError("Matching contrastive validation needs enough reference samples for positives and negatives.")
    conditional_weight = float(loss_weights.get("conditional_structure", 0.0))
    if conditional_enabled:
        if prior is None or str(data_config.get("split", "train")) != "train":
            raise ValueError("Conditional structure training requires a frozen fused prior and train split.")
        if not 1 <= query_count <= transport_batch_size - 3:
            raise ValueError("Conditional structure needs disjoint queries and at least three OT references.")
        if conditional_weight <= 0 or not math.isfinite(conditional_weight):
            raise ValueError("Enabled conditional structure requires a positive loss weight.")
        if matching_config.get("distance_scale", "infoot_rms") != "infoot_rms":
            raise ValueError("Conditional structure training uses InfoOT RMS distance scales.")
    elif conditional_weight != 0:
        raise ValueError("Set conditional_structure.enabled to use its nonzero loss weight.")
    if reconstruction_batch_size > transport_batch_size:
        raise ValueError("reconstruction_batch_size cannot exceed transport_batch_size.")
    final_step = int(max_steps or train_config.get("max_steps", 5000))
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    requested_resume = resume_from if resume_from is not None else train_config.get("resume_from")
    resume_path = _resolve_resume(requested_resume, root, checkpoint_path)
    if resume_path is None and checkpoint_path.exists() and not dry_run:
        raise FileExistsError(
            f"A training checkpoint already exists at {checkpoint_path}. "
            "Use --resume for the same experiment or a different output_dir for a fresh run."
        )

    dataset_kwargs = {
        "split": str(data_config.get("split", "train")),
        "project_root": root,
        "validate_exists": not dry_run,
        "random_horizontal_flip": float(data_config.get("random_horizontal_flip", 0.5)),
    }
    stage1a = _nested(config, "stage1a")
    datasets: dict[str, Any] = {}
    data_paths: dict[str, Path] = {}
    for domain in ("cat", "dog"):
        domain_stage1a = _nested(stage1a, domain)
        domain_train = load_yaml_config(
            resolve_project_local_path(
                domain_stage1a["config"], root, field_name=f"stage1a.{domain}.config"
            )
        )
        data_paths[domain] = resolve_project_local_path(
            domain_train["data_config"], root, field_name=f"stage1a.{domain}.data_config"
        )
        datasets[domain] = CachedLatentDataset(
            data_paths[domain], domain, **dataset_kwargs
        )
        if prior is not None:
            prior.lookup([row["sample_id"] for row in datasets[domain].records], domain, dataset_kwargs["split"])

    if dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = JointInfoOTTrainReport(
            config_path=str(resolved_config_path),
            output_dir=str(output_dir),
            initial_step=0,
            final_step=0,
            train_samples={domain: len(dataset) for domain, dataset in datasets.items()},
            transport_batch_size=transport_batch_size,
            reconstruction_batch_size=reconstruction_batch_size,
            checkpoint_path=None,
            resumed_from=str(resume_path) if resume_path else None,
            train_log_path=None,
            dry_run=True,
        )
        _write_json(output_dir / "dry_run_report.json", report.to_dict())
        return report

    domains = {
        "cat": _load_training_domain(config, root, "cat", device_override=device_cat),
        "dog": _load_training_domain(config, root, "dog", device_override=device_dog),
    }
    encoders = {domain: value.branch.encoder for domain, value in domains.items()}
    anchors = {
        domain: deepcopy(value.branch.encoder).eval().requires_grad_(False)
        for domain, value in domains.items()
    }

    calibration_count = int(matching_config.get("calibration_samples", 256))
    distance_scale_mode = str(matching_config.get("distance_scale", "infoot_rms"))
    anchor_variances: dict[str, float] = {}
    distance_scales: dict[str, float] = {}
    for domain, value in domains.items():
        calibration_dataset = CachedLatentDataset(
            data_paths[domain],
            domain,
            split=str(data_config.get("split", "train")),
            project_root=root,
            validate_exists=True,
            random_horizontal_flip=0.0,
        )
        anchor_variances[domain], distance_scales[domain] = _calibrate_encoder(
            value.branch.encoder,
            calibration_dataset,
            count=calibration_count,
            batch_size=min(transport_batch_size, 64),
            device=value.device,
            dtype=value.dtype,
            distance_scale_mode=distance_scale_mode,
        )

    loader_generators: dict[str, torch.Generator] = {}
    loaders: dict[str, Any] = {}
    for domain_index, domain in enumerate(("cat", "dog")):
        generator = torch.Generator().manual_seed(seed + domain_index)
        loader_generators[domain] = generator
        loader = DataLoader(
            datasets[domain],
            batch_size=transport_batch_size,
            shuffle=True,
            num_workers=int(data_config.get("num_workers", 4)),
            pin_memory=bool(data_config.get("pin_memory", True)),
            drop_last=bool(data_config.get("drop_last", True)),
            collate_fn=collate_latent_batch,
            generator=generator,
        )
        if len(loader) == 0:
            raise ValueError(f"{domain} DataLoader is empty; reduce transport_batch_size.")
        loaders[domain] = _cycle(loader)

    matching_heads = {
        domain: make_matching_head(head_spec, device=value.device, seed=seed + 700 + index)
        for index, (domain, value) in enumerate(domains.items())
    } if head_spec["enabled"] else {}
    head_parameters = [p for head in matching_heads.values() for p in head.parameters()]
    decoder_training = DecoderTraining(config, domains, prior, root, seed=seed) if generator_adaptation_enabled(config) else None
    generator_parameters = decoder_training.parameters if decoder_training is not None else []
    parameters = [parameter for encoder in encoders.values() for parameter in encoder.parameters()]
    conflict_monitor = GradientConflictMonitor()
    diagnostic_groups = {f"encoder.{d}": list(encoder.parameters()) for d, encoder in encoders.items()}
    diagnostic_groups.update({f"matching_head.{d}": list(head.parameters()) for d, head in matching_heads.items()})
    if decoder_training is not None:
        diagnostic_groups.update({f"generator.{d}": list(view.parameters()) for d, view in decoder_training.views.items()})
    parameter_groups = [{"params": parameters}]
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": float(head_config.get("lr", 2e-4))})
    if decoder_training is not None:
        parameter_groups.extend(decoder_training.parameter_groups())
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(train_config.get("lr_encoder", 2.0e-5)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
        betas=tuple(float(value) for value in train_config.get("betas", [0.9, 0.999])),
        foreach=False,
    )
    ema = (
        EncoderEMA(
            encoders,
            decay=float(ema_config.get("decay", 0.9999)),
            warmup_steps=int(ema_config.get("warmup_steps", 500)),
        )
        if bool(ema_config.get("enabled", True))
        else None
    )

    initial_step = 0
    matching_ema = EncoderEMA(
        matching_heads, decay=float(ema_config.get("decay", 0.995)),
        warmup_steps=int(ema_config.get("warmup_steps", 500)),
    ) if matching_heads and ema is not None else None
    training_seconds = 0.0
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = _load_checkpoint(resume_path)
        validate_prior_resume(checkpoint.get("config") or {}, config)
        _validate_resume_provenance(
            checkpoint,
            domains,
            weights=str(stage1a.get("weights", "ema")),
        )
        for domain, encoder in encoders.items():
            encoder.load_state_dict(checkpoint["encoders"][domain])
        if matching_heads:
            for domain, head in matching_heads.items():
                state = (checkpoint.get("matching_heads") or {}).get(domain)
                if state is None:
                    raise ValueError(f"Resume checkpoint has no matching_heads.{domain}.")
                head.load_state_dict(state, strict=True)
            if matching_ema is not None:
                if checkpoint.get("matching_head_ema_state") is None:
                    raise ValueError("Resume checkpoint has no matching head EMA state.")
                matching_ema.load_state_dict(checkpoint["matching_head_ema_state"], matching_heads)
        if decoder_training is not None:
            decoder_training.load_checkpoint(checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if ema is not None and checkpoint.get("ema_state") is not None:
            ema.load_state_dict(checkpoint["ema_state"], encoders)
        initial_step = int(checkpoint.get("step", 0))
        training_seconds = float((checkpoint.get("train_state") or {}).get("training_seconds", 0.0))
        if checkpoint.get("rng_state") is not None:
            torch.set_rng_state(checkpoint["rng_state"])
        for domain, state in (checkpoint.get("dataloader_generator_states") or {}).items():
            loader_generators[domain].set_state(state)
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            for index, state in enumerate(checkpoint["cuda_rng_state_all"][: torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state, device=index)
    if initial_step > final_step:
        raise ValueError(f"Checkpoint step {initial_step} exceeds max_steps {final_step}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)
    log_path = output_dir / "logs" / "train.jsonl"
    if resume_path is None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        # A failed initialization may leave a step-0 probe without a checkpoint.
        # Start its validation log afresh together with the training log.
        (log_path.parent / "validation.jsonl").write_text("", encoding="utf-8")

    alignment_device = domains["cat"].device
    bandwidth = float(matching_config.get("bandwidth_multiplier", 1.0))
    alignment_weight = float(loss_weights.get("infoot_alignment", 0.02))
    warmup_steps = int(loss_weights.get("alignment_warmup_steps", 1000))
    anchor_weight = float(loss_weights.get("latent_anchor", 0.01))
    if not math.isfinite(anchor_weight) or anchor_weight < 0:
        raise ValueError("latent_anchor weight must be finite and nonnegative.")
    needs_anchor_codes = anchor_weight > 0 or support_enabled or float(
        _nested(config, "generator_adaptation").get("conditioned_preservation_weight", 0.0)) > 0
    neighborhood_weight = float(loss_weights.get("semantic_neighborhood", 0.0))
    if not math.isfinite(neighborhood_weight) or neighborhood_weight < 0:
        raise ValueError("semantic_neighborhood weight must be finite and nonnegative.")
    if neighborhood_weight > 0 and prior is None:
        raise ValueError("semantic_neighborhood requires a frozen semantic prior.")
    prior_neighborhood_options = neighborhood_options(_nested(config, "semantic_prior"))
    rec_weights = {
        "cat": float(loss_weights.get("cat_reconstruction", 1.0)),
        "dog": float(loss_weights.get("dog_reconstruction", 1.0)),
    }
    log_formatter = Stage1BLogFormatter(config)
    log_every = int(train_config.get("log_every", 20))
    gradient_diagnostics_every = int(train_config.get("gradient_diagnostics_every", 200))
    save_every = int(train_config.get("save_every", 500))
    ema_update_every = int(ema_config.get("update_every", 1))
    solver_options = solver_kwargs(infoot_config)
    loss_window_size = int(train_config.get("loss_window", 100))
    if loss_window_size <= 0:
        raise ValueError("train.loss_window must be positive.")
    loss_window: deque[dict[str, float]] = deque(maxlen=loss_window_size)
    probe_every = int(train_config.get("validation_every", 0))
    probe_inputs = {}
    projection_probe_inputs = {}
    if probe_every > 0:
        for domain in domains:
            validation = CachedLatentDataset(data_paths[domain], domain, split="val", project_root=root)
            count = min(int(train_config.get("validation_samples", 8)), len(validation))
            if count <= 0:
                raise ValueError("Fixed validation needs nonempty validation samples.")
            indices = torch.randperm(len(validation), generator=torch.Generator().manual_seed(seed + 19))[:count]
            probe_inputs[domain] = torch.stack([validation[int(i)]["x0_latent"] for i in indices])
            if conditional_enabled:
                projection_probe_inputs[domain] = {}
                for kind, dataset, requested in (
                    ("reference", datasets[domain], int(conditional_config.get("validation_reference_samples", 96))),
                    ("query", validation, int(conditional_config.get("validation_query_samples", 32))),
                ):
                    selected = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed + 29))[:requested]
                    rows = [dataset[int(i)] for i in selected]
                    if len(rows) < (3 if kind == "reference" else 1):
                        raise ValueError("Projection validation needs at least three train references and one validation query.")
                    ids = [row["sample_id"] for row in rows]
                    projection_probe_inputs[domain][f"{kind}_ids"] = ids
                    projection_probe_inputs[domain][f"{kind}_metadata"] = [row.get("metadata", {}) for row in rows]
                    projection_probe_inputs[domain][f"{kind}_latents"] = torch.stack([row["x0_latent"] for row in rows])
                    projection_probe_inputs[domain][f"{kind}_structure"] = prior.lookup(ids, domain, "train" if kind == "reference" else "val")

    def validation_metrics(validation_step) -> dict[str, Any]:
        metrics = fixed_reconstruction_probe(domains, anchors, probe_inputs, seed=seed + 10000,
                                             generator_baselines=decoder_training.baselines if decoder_training else None)
        if conditional_enabled:
            metrics.update(fixed_conditional_structure_probe(
                domains, projection_probe_inputs, cost_scale=prior.cost_scale, bandwidth=bandwidth,
                projection_bandwidth=float(conditional_config.get("bandwidth_multiplier", 0.1)),
                teacher_temperature=float(conditional_config.get("teacher_temperature", 0.05)),
                solver_options=solver_options, cross_cost_weight=float(infoot_config.get("cross_cost_weight", 1.0)),
                anchors=anchors, anchor_variances=anchor_variances,
                projection_support_options=support_options(support_config) if support_enabled else None,
                matching_heads=matching_heads,
                matching_regularization_options=matching_protection_options,
                matching_contrastive_options=contrastive_options,
            ))
            # Fixed validation uses the full coefficient, independent of warmup.
            metrics["conditional_structure_weight"] = conditional_weight
            metrics["weighted_conditional_structure_loss"] = conditional_weight * metrics["conditional_structure_loss"]
        if decoder_training is not None:
            metrics["decoded_translation"] = fixed_decoded_translation_probe(
                decoder_training, domains, matching_heads, projection_probe_inputs,
                prior=prior, bandwidth=bandwidth,
                projection_bandwidth=float(conditional_config.get("bandwidth_multiplier", .1)),
                teacher_temperature=float(conditional_config.get("teacher_temperature", .05)),
                solver_options=solver_options, seed=seed + 11000,
                cross_cost_weight=float(infoot_config.get("cross_cost_weight", 1.0)),
                image_dir=(output_dir / "validation" / f"step_{validation_step:06d}"
                           if decoder_training.options.get("save_validation_images", False) else None),
            )
        return log_formatter.format(metrics)

    if probe_every > 0:
        _append_jsonl(output_dir / "logs" / "validation.jsonl", {
            "step": initial_step, "weights": "raw", "seed": seed + 10000,
            **validation_metrics(initial_step),
        })

    if resume_path is None:
        _save_checkpoint(
            checkpoint_path,
            _build_checkpoint_payload(
                step=0,
                encoders=encoders,
                optimizer=optimizer,
                ema=ema,
                config=config,
                domains=domains,
                train_state={
                    "training_seconds": training_seconds,
                    "anchor_variances": anchor_variances,
                    "distance_scales": distance_scales,
                },
                loader_generators=loader_generators,
                matching_heads=matching_heads, matching_ema=matching_ema,
                decoder_training=decoder_training,
            ),
        )

    for step in range(initial_step + 1, final_step + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        x0: dict[str, torch.Tensor] = {}
        z: dict[str, torch.Tensor] = {}
        reference_z: dict[str, torch.Tensor] = {}
        rec_losses: dict[str, torch.Tensor] = {}
        anchor_losses: dict[str, torch.Tensor] = {}
        teacher_features: dict[str, torch.Tensor] = {}
        neighborhood_losses: dict[str, torch.Tensor] = {}
        matching_z: dict[str, torch.Tensor] = {}
        reconstruction_diagnostics: dict[str, dict[str, float]] = {}
        query_metadata = {}
        for domain, value in domains.items():
            batch = next(loaders[domain])
            if query_count:
                query_metadata[domain] = batch["metadata"][-query_count:]
            x0[domain] = batch["x0_latent"].to(
                value.device, dtype=value.dtype, non_blocking=True
            )
            z[domain] = value.branch.encoder(x0[domain])
            matching_z[domain] = matching_features(z[domain], matching_heads.get(domain)).to(alignment_device)
            if prior is not None:
                teacher_features[domain] = prior.lookup(
                    batch["sample_id"], domain, dataset_kwargs["split"]
                ).to(alignment_device)
                neighborhood_losses[domain] = neighborhood_distillation_loss(
                    matching_z[domain] if matching_heads else z[domain].to(alignment_device), teacher_features[domain],
                    **prior_neighborhood_options,
                ) if neighborhood_weight > 0 else torch.zeros((), device=alignment_device)
            if needs_anchor_codes:
                with torch.no_grad():
                    reference_z[domain] = anchors[domain](x0[domain])
            reconstruction_diagnostics[domain] = {}
            reconstruction_args = ({"semantic_dropout": True, "diagnostics": reconstruction_diagnostics[domain]}
                                   if decoder_training is not None else {})
            rec_losses[domain] = _reconstruction_loss(
                value, x0[domain][:reconstruction_batch_size], z[domain][:reconstruction_batch_size],
                **reconstruction_args,
            )
            anchor_losses[domain] = encoder_anchor_loss(
                z[domain], reference_z[domain], anchor_variances[domain]
            ) if anchor_weight > 0 else torch.zeros((), device=value.device)

        # DataLoaders shuffle each domain independently. Reserve the last Q
        # examples as queries; they must never enter this update's OT fit.
        references = {
            domain: (code[:-query_count] if query_count else code).float().to(alignment_device)
            for domain, code in z.items()
        }
        if conditional_enabled and any(len(code) < 3 for code in references.values()):
            raise ValueError("Short training batch leaves too few OT references; enable drop_last.")
        reference_structure = {
            domain: teacher[:-query_count] if query_count else teacher
            for domain, teacher in teacher_features.items()
        }
        reference_matching = {
            domain: code[:-query_count] if query_count else code for domain, code in matching_z.items()
        }
        matching_protection = None
        matching_protection_loss = torch.zeros((), device=alignment_device)
        if matching_protection_options is not None:
            matching_protection = matching_regularization_loss(reference_matching, **matching_protection_options)
            matching_protection_loss = matching_protection.loss
        cat_features = reference_matching["cat"]
        dog_features = reference_matching["dog"]
        if distance_scale_mode == "infoot_rms":
            solve_distance_scales = {
                "cat": float(infoot_distance_scale(cat_features.detach())),
                "dog": float(infoot_distance_scale(dog_features.detach())),
            }
        else:
            solve_distance_scales = distance_scales
        cross_cost = prior.cost(reference_structure["cat"], reference_structure["dog"]) if prior is not None else None
        solution = solve_infoot(
            cat_features.detach(),
            dog_features.detach(),
            bandwidth=bandwidth,
            distance_scale_x=solve_distance_scales["cat"],
            distance_scale_y=solve_distance_scales["dog"],
            cross_cost=cross_cost,
            cross_cost_weight=float(infoot_config.get("cross_cost_weight", 1.0)),
            **solver_options,
        )
        # The solve is detached, but the neural objective must differentiate
        # the RMS estimate as well as distances when full gradients are enabled.
        feature_distance_scales = ({
            "cat": infoot_distance_scale(cat_features, detach=False),
            "dog": infoot_distance_scale(dog_features, detach=False),
        } if differentiate_distance_scale else solve_distance_scales)
        infoot_loss = plain_infoot_feature_loss(
            cat_features,
            dog_features,
            solution.coupling.detach(),
            bandwidth=bandwidth,
            distance_scale_x=feature_distance_scales["cat"],
            distance_scale_y=feature_distance_scales["dog"],
            mi_weight=float(infoot_config.get("mi_weight", 1.0)),
            eps=float(infoot_config.get("numerical_epsilon", 1.0e-8)),
        )
        warmup = 1.0 if warmup_steps <= 0 else min(step / warmup_steps, 1.0)
        beta = alignment_weight * warmup
        neighborhood_loss = sum(neighborhood_losses.values(), torch.zeros((), device=alignment_device))
        conditional_result = None
        projection_loss = torch.zeros((), device=alignment_device)
        if conditional_enabled:
            conditional_result = conditional_structure_loss(
                references, {domain: code[-query_count:].float().to(alignment_device) for domain, code in z.items()},
                reference_structure, {domain: teacher[-query_count:] for domain, teacher in teacher_features.items()},
                solution.coupling, cost_scale=prior.cost_scale,
                bandwidth=float(conditional_config.get("bandwidth_multiplier", 0.1)),
                teacher_temperature=float(conditional_config.get("teacher_temperature", 0.05)),
                reference_matching=reference_matching,
                query_matching={domain: code[-query_count:] for domain, code in matching_z.items()},
                differentiate_distance_scale=differentiate_distance_scale,
            )
            projection_loss = conditional_result.loss
        contrastive_loss = torch.zeros((), device=alignment_device)
        contrastive_metrics = None
        if contrastive_options is not None:
            contrastive_loss, contrastive_metrics = matching_contrastive_loss(
                reference_matching, reference_structure, conditional_result.log_weights,
                conditional_result.teacher_weights, **contrastive_options)
        support_result = None
        support_loss = torch.zeros((), device=alignment_device)
        if support_enabled:
            support_result = projection_support_loss(
                conditional_result.weights, references,
                {domain: code[:-query_count].float().to(alignment_device) for domain, code in reference_z.items()},
                anchor_variances, **support_options(support_config),
                teacher_weights=conditional_result.teacher_weights,
            )
            support_loss = support_result.loss
        decoded_loss = torch.zeros((), device=alignment_device)
        code_consistency_loss = torch.zeros((), device=alignment_device)
        decoded_metrics: dict[str, Any] = {"active": False}
        if decoder_training is not None and decoder_training.active(step):
            decoded_loss, decoded_metrics = decoder_training.loss(
                conditional_result.weights, references,
                {d: latent[-query_count:] for d, latent in x0.items()},
                {d: latent[:-query_count] for d, latent in x0.items()}, step=step,
                source_reference_structure=reference_structure,
                source_query_structure={d: teacher[-query_count:] for d, teacher in teacher_features.items()},
                source_query_metadata=query_metadata,
            )
            code_consistency_loss = decoder_training.code_consistency_objective
            decoded_metrics["active"] = True
        null_preservation = decoder_training.null_preservation_loss(
            {d: latent[:reconstruction_batch_size] for d, latent in x0.items()},
            {d: code[:reconstruction_batch_size] for d, code in z.items()},
        ) if decoder_training is not None else torch.zeros((), device=alignment_device)
        null_preservation_weight = float(_nested(config, "generator_adaptation").get("null_preservation_weight", .1))
        conditioned_preservation = decoder_training.conditioned_preservation_loss(
            {d: latent[:reconstruction_batch_size] for d, latent in x0.items()},
            # Fixed Stage 1A E codes; this is a G-only preservation objective.
            {d: code[:reconstruction_batch_size] for d, code in reference_z.items()},
        ) if decoder_training is not None else torch.zeros((), device=alignment_device)
        conditioned_preservation_weight = float(_nested(config, "generator_adaptation").get("conditioned_preservation_weight", 0.0))
        reconstruction_objective = (
            rec_weights["cat"] * rec_losses["cat"].to(alignment_device)
            + rec_weights["dog"] * rec_losses["dog"].to(alignment_device)
        )
        primary_objective = (
            reconstruction_objective
            + anchor_weight
            * (anchor_losses["cat"].to(alignment_device) + anchor_losses["dog"].to(alignment_device))
            + null_preservation_weight * null_preservation
            + conditioned_preservation_weight * conditioned_preservation
        )
        transport_objective = (
            beta * infoot_loss
            + neighborhood_weight * warmup * neighborhood_loss
            + conditional_weight * warmup * projection_loss
            + warmup * contrastive_loss
            + support_weight * warmup * support_loss
        )
        # Keep protection a separate PCGrad task: it addresses the observed
        # spread/rank failure independently of relational transport fitting.
        auxiliary_objective = transport_objective + decoded_loss + matching_protection_loss
        total = primary_objective + auxiliary_objective
        # Code recovery is an additional generator-only objective. Its scalar
        # contributes to reporting, while selective autograd below prevents
        # moving encoder/head targets through the live conditioning path.
        reported_total = total.detach() + code_consistency_loss.detach()
        if not torch.isfinite(reported_total):
            raise FloatingPointError(f"Non-finite Stage 1B loss at step {step}.")
        encoder_correction, balance_metrics = (), {}
        if balance_config.enabled and decoded_metrics["active"]:
            encoder_correction, balance_metrics = decoded_encoder_gradient_correction(
                reconstruction_objective, decoded_loss, parameters,
                config=balance_config, ramp=decoder_training.ramp(step))
        code_generator_gradients = ()
        if code_consistency_loss.requires_grad:
            code_generator_gradients = torch.autograd.grad(
                code_consistency_loss, generator_parameters, retain_graph=True, allow_unused=True)
            code_generator_gradients = tuple(
                None if g is None else g.detach() for g in code_generator_gradients)
        # No remaining autograd consumer needs the code-recovery readout graph.
        code_consistency_loss = code_consistency_loss.detach()
        if decoder_training is not None:
            decoder_training.code_consistency_objective = code_consistency_loss
        gradient_diagnostics: dict[str, Any] = {}
        diagnostics_due = gradient_diagnostics_every > 0 and step % gradient_diagnostics_every == 0
        conflict_report, conflict_norms = None, None
        if diagnostics_due and conflict_diagnostics_enabled and decoder_training is not None and decoded_metrics["active"]:
            conflict_monitor.set_phase("full_weight" if warmup >= 1 and decoder_training.ramp(step) >= 1 else "warmup")
            conflict_report, conflict_norms = training_gradient_conflicts(
                {
                    "reconstruction": reconstruction_objective,
                    "decoded": decoded_loss,
                    "perceptual": decoder_training.image_objectives.get("perceptual", decoded_loss.new_zeros(())),
                    "adversarial": decoder_training.image_objectives["adversarial"],
                    "structure": decoder_training.image_objectives["structure"],
                    "color": decoder_training.image_objectives.get("color", decoded_loss.new_zeros(())),
                    "matching": (beta * infoot_loss + neighborhood_weight * warmup * neighborhood_loss
                                 + conditional_weight * warmup * projection_loss + warmup * contrastive_loss
                                 + support_weight * warmup * support_loss + matching_protection_loss),
                    **({"conditional": conditional_weight * warmup * projection_loss,
                        "infoot": beta * infoot_loss, "protection": matching_protection_loss}
                       if conditional_enabled else {}),
                }, diagnostic_groups,
                code_gradients={id(p): g for p, g in zip(generator_parameters, code_generator_gradients)},
                encoder_scale=float(balance_metrics.get("scale", 1.0)), monitor=conflict_monitor)
            conflict_report["encoder_guard_applied_after_measurement"] = gradient_guard_enabled
            conflict_report["pcgrad_applied_after_measurement"] = pcgrad_config.enabled
        if diagnostics_due:
            gradient_diagnostics = {
                "reconstruction_gradient_norm": (conflict_norms["reconstruction"]["encoder.all"] if conflict_norms else _autograd_norm(
                    reconstruction_objective, parameters
                )),
                "weighted_alignment_gradient_norm": (conflict_norms["infoot"]["encoder.all"] if conflict_norms and "infoot" in conflict_norms else _autograd_norm(
                    beta * infoot_loss, parameters
                )),
            }
            if decoder_training is not None:
                gradient_diagnostics["reconstruction_generator_gradient_norm"] = (
                    conflict_norms["reconstruction"]["generator.all"] if conflict_norms else
                    _autograd_norm(reconstruction_objective, generator_parameters))
                gradient_diagnostics["weighted_null_preservation_generator_gradient_norm"] = _autograd_norm(
                    null_preservation_weight * null_preservation, generator_parameters)
                if conditioned_preservation_weight > 0:
                    gradient_diagnostics["weighted_conditioned_preservation_generator_gradient_norm"] = _autograd_norm(
                        conditioned_preservation_weight * conditioned_preservation, generator_parameters)
            gradient_diagnostics["alignment_to_reconstruction_gradient_ratio"] = (
                gradient_diagnostics["weighted_alignment_gradient_norm"]
                / max(gradient_diagnostics["reconstruction_gradient_norm"], 1e-12)
            )
            if prior is not None:
                neighborhood_norms = _autograd_group_norms(neighborhood_weight * warmup * neighborhood_loss, {
                    "encoder": parameters, "matching_head": head_parameters,
                })
                gradient_diagnostics["weighted_neighborhood_gradient_norm"] = neighborhood_norms["encoder"]
                gradient_diagnostics["weighted_neighborhood_matching_head_gradient_norm"] = neighborhood_norms["matching_head"]
            if head_parameters:
                gradient_diagnostics["weighted_alignment_matching_head_gradient_norm"] = (
                    conflict_norms["infoot"]["matching_head.all"] if conflict_norms and "infoot" in conflict_norms else
                    _autograd_norm(beta * infoot_loss, head_parameters))
            if conditional_enabled:
                gradient_diagnostics["weighted_conditional_structure_gradient_norm"] = (
                    conflict_norms["conditional"]["encoder.all"] if conflict_norms else
                    _autograd_norm(conditional_weight * warmup * projection_loss, parameters))
                gradient_diagnostics["conditional_structure_to_reconstruction_gradient_ratio"] = (
                    gradient_diagnostics["weighted_conditional_structure_gradient_norm"]
                    / max(gradient_diagnostics["reconstruction_gradient_norm"], 1e-12)
                )
                if head_parameters:
                    gradient_diagnostics["weighted_conditional_structure_matching_head_gradient_norm"] = (
                        conflict_norms["conditional"]["matching_head.all"] if conflict_norms else
                        _autograd_norm(conditional_weight * warmup * projection_loss, head_parameters))
            if support_enabled:
                gradient_diagnostics["weighted_projection_support_gradient_norm"] = _autograd_norm(
                    support_weight * warmup * support_loss, parameters
                )
            if matching_protection is not None:
                protection_norms = ({group: conflict_norms["protection"][f"{group}.all"]
                                    for group in ("encoder", "matching_head")}
                                   if conflict_norms and "protection" in conflict_norms else _autograd_group_norms(matching_protection_loss, {
                    "encoder": parameters, "matching_head": head_parameters,
                }))
                gradient_diagnostics.update({
                    f"weighted_matching_regularization_{group}_gradient_norm": norm
                    for group, norm in protection_norms.items()
                })
                gradient_diagnostics["matching_regularization_to_reconstruction_encoder_gradient_ratio"] = (
                    protection_norms["encoder"] / max(gradient_diagnostics["reconstruction_gradient_norm"], 1e-12)
                )
                # A small aggregate can conceal opposing V/C gradients. Keep
                # the already weighted components visible before retuning them.
                for name, objective in (("variance", matching_protection.weighted_variance_loss),
                                        ("covariance", matching_protection.weighted_covariance_loss)):
                    norms = _autograd_group_norms(objective, {"encoder": parameters, "matching_head": head_parameters})
                    gradient_diagnostics.update({f"weighted_matching_{name}_{group}_gradient_norm": norm
                                                 for group, norm in norms.items()})
            if decoder_training is not None and decoded_metrics["active"]:
                image_norms = ({group: conflict_norms["decoded"][f"{group}.all"]
                                for group in ("encoder", "matching_head", "generator")} if conflict_norms else
                              _autograd_group_norms(decoded_loss, {
                    "encoder": parameters, "matching_head": head_parameters, "generator": generator_parameters,
                }))
                gradient_diagnostics.update({
                    f"weighted_decoded_{group}_gradient_norm": norm for group, norm in image_norms.items()
                })
                gradient_diagnostics["decoded_to_reconstruction_encoder_gradient_ratio"] = (
                    image_norms["encoder"] / max(gradient_diagnostics["reconstruction_gradient_norm"], 1e-12)
                )
            if contrastive_options is not None:
                norms = _autograd_group_norms(warmup * contrastive_loss, {
                    "encoder": parameters, "matching_head": head_parameters})
                gradient_diagnostics.update({
                    f"weighted_matching_contrastive_{group}_gradient_norm": norm
                    for group, norm in norms.items()})
            if code_generator_gradients:
                gradient_diagnostics["weighted_code_consistency_generator_gradient_norm"] = math.sqrt(
                    sum(float(g.float().square().sum()) for g in code_generator_gradients if g is not None))
                gradient_diagnostics["applied_code_consistency_encoder_gradient_norm"] = 0.0
                gradient_diagnostics["applied_code_consistency_matching_head_gradient_norm"] = 0.0
            if conflict_report is not None:
                gradient_diagnostics["gradient_conflicts"] = conflict_report
        if balance_metrics:
            # Keep the familiar ratio key equal to the applied decoded encoder
            # contribution, and retain explicit before/after values for audit.
            gradient_diagnostics.update({
                "reconstruction_gradient_norm": balance_metrics["reconstruction_gradient_norm"],
                "weighted_decoded_encoder_gradient_norm_pre_balance": balance_metrics["decoded_gradient_norm_before"],
                "weighted_decoded_encoder_gradient_norm": balance_metrics["decoded_gradient_norm_after"],
                "decoded_to_reconstruction_encoder_gradient_ratio_pre_balance": balance_metrics["prebalanced_ratio"],
                "decoded_to_reconstruction_encoder_gradient_ratio": balance_metrics["effective_ratio"],
                "decoded_reconstruction_encoder_gradient_cosine": balance_metrics["cosine"],
            })
        guard_metrics, pcgrad_metrics = {}, None
        if pcgrad_config.enabled:
            tasks = {"native": primary_objective, "transport": transport_objective,
                     "protection": matching_protection_loss}
            if decoder_training is not None and decoded_metrics["active"]:
                tasks.update(decoder_training.image_objectives)
            routed = ({"code": {id(p): g for p, g in zip(generator_parameters, code_generator_gradients)}}
                      if code_generator_gradients else None)
            pcgrad_metrics = pcgrad_backward(
                tasks, {"encoder": parameters, "matching_head": head_parameters,
                        "generator": generator_parameters}, seed=seed, step=step,
                eps=pcgrad_config.eps, routed_gradients=routed)
        elif gradient_guard_enabled:
            if generator_parameters:
                generator_gradients = torch.autograd.grad(total, generator_parameters, retain_graph=True, allow_unused=True)
                for parameter, gradient in zip(generator_parameters, generator_gradients):
                    parameter.grad = None if gradient is None else gradient.detach()
            if head_parameters:
                # Alignment can change matching geometry without consuming the
                # encoder's reconstruction-protection budget. Preserve the graph
                # for the separate guarded encoder gradients below.
                head_gradients = torch.autograd.grad(auxiliary_objective, head_parameters, retain_graph=True)
                for parameter, gradient in zip(head_parameters, head_gradients):
                    parameter.grad = gradient.detach()
            guard_metrics = guarded_backward(
                primary_objective, auxiliary_objective,
                {domain: list(encoder.parameters()) for domain, encoder in encoders.items()},
                max_auxiliary_ratio=float(gradient_guard_config.get("max_auxiliary_ratio", .25)),
                project_conflicts=bool(gradient_guard_config.get("project_conflicts", True)),
            )
        else:
            total.backward()
        with torch.no_grad():
            for parameter, correction in zip(parameters, encoder_correction):
                if correction is not None:
                    if parameter.grad is None:
                        parameter.grad = correction.clone()
                    else:
                        parameter.grad.add_(correction)
            for parameter, gradient in zip(generator_parameters, () if pcgrad_config.enabled else code_generator_gradients):
                if gradient is not None:
                    if parameter.grad is None:
                        parameter.grad = gradient.clone()
                    else:
                        parameter.grad.add_(gradient)
        del encoder_correction, code_generator_gradients
        if pcgrad_config.enabled:
            del tasks, routed
        if decoder_training is not None:
            # Diagnostics have consumed these live losses; do not retain their
            # graphs into the next image rollout.
            decoder_training.image_objectives = {}
        grad_norms = {
            domain: _gradient_norm(list(encoder.parameters()))
            for domain, encoder in encoders.items()
        }
        total_grad_norm = _clip_gradient_norm(
            parameters, train_config.get("grad_clip_norm", 1.0)
        )
        head_grad_norm = _clip_gradient_norm(
            head_parameters, head_config.get("grad_clip_norm", 1.0)
        ) if head_parameters else 0.0
        if not math.isfinite(head_grad_norm):
            raise FloatingPointError(f"Non-finite matching head gradients at step {step}.")
        generator_grad_norm = _clip_gradient_norm(
            generator_parameters, _nested(config, "generator_adaptation").get("grad_clip_norm", 1.0)
        ) if generator_parameters else 0.0
        if not all(math.isfinite(v) for v in (total_grad_norm, generator_grad_norm)):
            raise FloatingPointError(f"Non-finite encoder/generator gradients at step {step}.")
        optimizer.step()
        if ema is not None and step % ema_update_every == 0:
            ema.update(encoders)
            if matching_ema is not None:
                matching_ema.update(matching_heads)
            if decoder_training is not None and decoder_training.ema is not None:
                decoder_training.ema.update(decoder_training.views)

        elapsed = time.perf_counter() - started
        training_seconds += elapsed
        loss_window.append({
            "loss": float(reported_total),
            "cat_reconstruction_loss": float(rec_losses["cat"].detach()),
            "dog_reconstruction_loss": float(rec_losses["dog"].detach()),
            "infoot_mutual_information": solution.mutual_information,
            "semantic_neighborhood_loss": float(neighborhood_loss.detach()),
            "conditional_structure_loss": float(projection_loss.detach()),
            "weighted_conditional_structure_loss": float(conditional_weight * warmup * projection_loss.detach()),
            "projection_support_loss": float(support_loss.detach()),
            "primary_objective": float(primary_objective.detach()),
            "auxiliary_objective": float(auxiliary_objective.detach()),
            **({"matching_contrastive_loss": float(contrastive_loss.detach())} if contrastive_options is not None else {}),
            **({"code_consistency_loss": float(code_consistency_loss.detach())} if decoder_training is not None else {}),
            **({"decoded_translation_loss": float(decoded_loss.detach())} if decoder_training is not None else {}),
            **({"null_preservation_loss": float(null_preservation.detach())} if decoder_training is not None else {}),
            **({"conditioned_preservation_loss": float(conditioned_preservation.detach()),
                "weighted_conditioned_preservation_loss": float(conditioned_preservation_weight * conditioned_preservation.detach())}
               if decoder_training is not None else {}),
            **({"matching_regularization_loss": float(matching_protection_loss.detach()),
                "matching_variance_loss": matching_protection.metrics["variance_loss"],
                "matching_covariance_loss": matching_protection.metrics["covariance_loss"]}
               if matching_protection is not None else {}),
        })
        if step == initial_step + 1 or step % log_every == 0 or diagnostics_due:
            metrics = {
                "event": "train",
                "optimizer_gradient_mode": ("pcgrad" if pcgrad_config.enabled else "primary_guarded" if gradient_guard_enabled else
                                            "decoded_encoder_balanced" if balance_config.enabled else "weighted_sum"),
                **({"pcgrad": pcgrad_metrics} if pcgrad_metrics is not None else {}),
                "step": step,
                "loss": float(reported_total.cpu()),
                "cat_reconstruction_loss": float(rec_losses["cat"].detach().cpu()),
                "dog_reconstruction_loss": float(rec_losses["dog"].detach().cpu()),
                "cat_anchor_loss": float(anchor_losses["cat"].detach().cpu()),
                "dog_anchor_loss": float(anchor_losses["dog"].detach().cpu()),
                "latent_anchor_weight": anchor_weight,
                "infoot_feature_loss": float(infoot_loss.detach().cpu()),
                "infoot_mutual_information": solution.mutual_information,
                "infoot_entropy": solution.entropy,
                "infoot_objective": solution.objective,
                "infoot_row_residual": solution.row_residual,
                "infoot_column_residual": solution.column_residual,
                "infoot_restart": solution.restart,
                "infoot_sinkhorn_converged": solution.sinkhorn_converged,
                "infoot_unconverged_inner_steps": solution.unconverged_inner_steps,
                "infoot_outer_converged": solution.outer_converged,
                "infoot_plan_delta_l1": solution.plan_delta_l1,
                "infoot_iterations": solution.iterations,
                "infoot_iteration_budget": solver_options["inner_iterations"],
                "infoot_projection_iteration_budget": solver_options["projection_iterations"],
                "infoot_projection_tolerance": solver_options["projection_tolerance"],
                "infoot_reference_counts": {domain: len(code) for domain, code in references.items()},
                "conditional_query_samples_per_domain": query_count,
                "infoot_distance_scale_mode": distance_scale_mode,
                "infoot_distance_scale_gradient": scale_gradient,
                "cat_distance_scale": solve_distance_scales["cat"],
                "dog_distance_scale": solve_distance_scales["dog"],
                "alignment_weight": beta,
                "cat_encoder_grad_norm_pre_clip": grad_norms["cat"],
                "dog_encoder_grad_norm_pre_clip": grad_norms["dog"],
                "total_grad_norm_pre_clip": total_grad_norm,
                "ema_decay": ema.effective_decay if ema is not None else None,
                "transport_batch_size": transport_batch_size,
                "reconstruction_batch_size": reconstruction_batch_size,
                "seconds_per_step": elapsed,
                "training_seconds_total": training_seconds,
                "cat_kernel": kernel_offdiagonal_stats(
                    cat_features.detach(),
                    bandwidth=bandwidth,
                    distance_scale=solve_distance_scales["cat"],
                ),
                "dog_kernel": kernel_offdiagonal_stats(
                    dog_features.detach(),
                    bandwidth=bandwidth,
                    distance_scale=solve_distance_scales["dog"],
                ),
            }
            metrics.update(gradient_diagnostics)
            metrics["transport"] = transport_diagnostics(solution.coupling)
            metrics["semantic_neighborhood_loss"] = float(neighborhood_loss.detach())
            metrics["semantic_neighborhood_weight"] = neighborhood_weight * warmup
            metrics["conditional_structure_loss"] = float(projection_loss.detach())
            metrics["conditional_structure_weight"] = conditional_weight * warmup
            metrics["weighted_conditional_structure_loss"] = float(conditional_weight * warmup * projection_loss.detach())
            metrics["projection_support_loss"] = float(support_loss.detach())
            metrics["projection_support_weight"] = support_weight * warmup
            metrics["gradient_guard"] = guard_metrics
            if balance_config.enabled:
                metrics["decoded_encoder_balance"] = balance_metrics
            if contrastive_metrics is not None:
                metrics["matching_contrastive_loss"] = float(contrastive_loss.detach())
                metrics["matching_contrastive"] = {**contrastive_metrics, "ramp": warmup,
                                                   "effective_weighted_loss": float(warmup * contrastive_loss.detach())}
            if matching_protection is not None:
                metrics["matching_regularization_loss"] = float(matching_protection_loss.detach())
                metrics["matching_regularization"] = matching_protection.metrics
            if decoder_training is not None:
                metrics["decoded_translation"] = decoded_metrics
                metrics["code_consistency_loss"] = float(code_consistency_loss.detach())
                metrics["generator_gradient_norm_pre_clip"] = generator_grad_norm
                metrics["generator_learning_rates"] = {g["name"]: g["lr"] for g in optimizer.param_groups if "name" in g}
                metrics["reconstruction_diagnostics"] = reconstruction_diagnostics
                metrics["null_preservation_loss"] = float(null_preservation.detach())
                metrics["conditioned_preservation_loss"] = float(conditioned_preservation.detach())
                metrics["conditioned_preservation_weight"] = conditioned_preservation_weight
                metrics["conditioned_preservation_samples"] = min(
                    int(_nested(config, "generator_adaptation").get("conditioned_preservation_samples", 4)),
                    reconstruction_batch_size) if conditioned_preservation_weight > 0 else 0
            if matching_heads:
                from diffusion_ot.models.matching_head import matching_geometry_diagnostics
                metrics["matching_head_gradient_norm_pre_clip"] = head_grad_norm
                metrics["matching_head_learning_rate"] = optimizer.param_groups[1]["lr"]
                metrics["matching_feature_variance"] = {
                    d: float(code.detach().var(0, unbiased=False).sum()) for d, code in reference_matching.items()
                }
                metrics["feature_geometry"] = {
                    d: matching_geometry_diagnostics(references[d], code) for d, code in reference_matching.items()
                }
            if prior is not None:
                metrics["semantic_neighborhood_geometry"] = prior_neighborhood_options["geometry"]
                metrics["semantic_neighborhood_temperature"] = prior_neighborhood_options["temperature"]
            if support_result is not None:
                metrics["projection_support"] = support_result.metrics
            if conditional_result is not None:
                metrics["conditional_structure"] = conditional_result.metrics
            metrics["window_mean"] = {
                key: sum(row[key] for row in loss_window) / len(loss_window)
                for key in loss_window[-1]
            }
            metrics["window_updates"] = len(loss_window)
            if cross_cost is not None:
                metrics["transport_structure_cost"] = float((solution.coupling * cross_cost).sum())
                metrics["independent_structure_cost"] = float(cross_cost.mean())
            metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            metrics = log_formatter.format(metrics)
            print(json.dumps(metrics, sort_keys=True))
            _append_jsonl(log_path, metrics)

        if pcgrad_config.enabled:
            # Earlier task traversals retained their private forward branches.
            # Drop loss roots before validation/the next expensive rollout.
            del total, primary_objective, auxiliary_objective, reconstruction_objective
            del transport_objective, decoded_loss, null_preservation, conditioned_preservation
            rec_losses.clear()
            anchor_losses.clear()

        if probe_every > 0 and (step % probe_every == 0 or step == final_step):
            _append_jsonl(output_dir / "logs" / "validation.jsonl", {
                "step": step, "weights": "raw", "seed": seed + 10000,
                **validation_metrics(step),
            })
        if step % save_every == 0 or step == final_step:
            _save_checkpoint(
                checkpoint_path,
                _build_checkpoint_payload(
                    step=step,
                    encoders=encoders,
                    optimizer=optimizer,
                    ema=ema,
                    config=config,
                    domains=domains,
                    train_state={
                        "training_seconds": training_seconds,
                        "anchor_variances": anchor_variances,
                        "distance_scales": distance_scales,
                    },
                    loader_generators=loader_generators,
                    matching_heads=matching_heads, matching_ema=matching_ema,
                    decoder_training=decoder_training,
                ),
            )
            if bool(train_config.get("keep_step_checkpoints", False)):
                import shutil
                shutil.copyfile(checkpoint_path, checkpoint_path.with_name(f"step_{step:06d}.pt"))

    return JointInfoOTTrainReport(
        config_path=str(resolved_config_path),
        output_dir=str(output_dir),
        initial_step=initial_step,
        final_step=final_step,
        train_samples={domain: len(dataset) for domain, dataset in datasets.items()},
        transport_batch_size=transport_batch_size,
        reconstruction_batch_size=reconstruction_batch_size,
        checkpoint_path=str(checkpoint_path) if checkpoint_path.is_file() else None,
        resumed_from=str(resume_path) if resume_path else None,
        train_log_path=str(log_path),
        dry_run=False,
    )
