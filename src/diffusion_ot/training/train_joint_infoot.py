from __future__ import annotations

from copy import deepcopy
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
) -> tuple[float, float]:
    from diffusion_ot.losses.infoot import median_distance_scale, normalize_matching_features

    values: list[torch.Tensor] = []
    count = min(int(count), len(dataset))
    for offset in range(0, count, int(batch_size)):
        items = [dataset[index] for index in range(offset, min(offset + int(batch_size), count))]
        x0 = torch.stack([item["x0_latent"] for item in items]).to(device=device, dtype=dtype)
        values.append(encoder(x0).float().cpu())
    codes = torch.cat(values, dim=0)
    variance = float(codes.var(dim=0, unbiased=False).mean().clamp_min(1.0e-8))
    distance_scale = float(median_distance_scale(normalize_matching_features(codes)))
    return variance, distance_scale


def _reconstruction_loss(
    domain: LoadedTrainingDomain,
    x0: torch.Tensor,
    z: torch.Tensor,
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
    output = domain.branch.predict_with_z(
        target.x_t, target.t, z, class_labels=labels
    )
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
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 2,
        "stage": "stage1b_plain_infoot",
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
    if not isinstance(payload, dict) or payload.get("stage") != "stage1b_plain_infoot":
        raise ValueError(f"Not a Stage 1B-1 checkpoint: {path}")
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
        kernel_offdiagonal_stats,
        normalize_matching_features,
        plain_infoot_feature_loss,
        solve_plain_infoot,
        solver_kwargs,
    )

    resolved_config_path = Path(config_path).resolve()
    config = load_yaml_config(resolved_config_path)
    if str(config.get("stage")) != "stage1b_plain_infoot":
        raise ValueError("Expected a stage1b_plain_infoot config.")
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
    ema_config = _nested(config, "ema")
    loss_weights = _nested(config, "loss_weights")
    matching_config = _nested(config, "matching")
    infoot_config = _nested(config, "infoot")
    trainable_config = _nested(config, "trainable")
    if not bool(trainable_config.get("encoders", True)):
        raise ValueError("Stage 1B-1 requires trainable encoders.")
    if (
        bool(trainable_config.get("adapters", False))
        or bool(trainable_config.get("attention_lora", False))
        or bool(trainable_config.get("base_transformers", False))
    ):
        raise ValueError("The first Stage 1B-1 experiment trains encoders only.")

    seed = int(train_config.get("seed", 20260905))
    torch.manual_seed(seed)
    transport_batch_size = int(data_config.get("transport_batch_size", 64))
    reconstruction_batch_size = int(data_config.get("reconstruction_batch_size", 8))
    if reconstruction_batch_size > transport_batch_size:
        raise ValueError("reconstruction_batch_size cannot exceed transport_batch_size.")
    final_step = int(max_steps or train_config.get("max_steps", 5000))
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    requested_resume = resume_from if resume_from is not None else train_config.get("resume_from")
    resume_path = _resolve_resume(requested_resume, root, checkpoint_path)

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

    parameters = [parameter for encoder in encoders.values() for parameter in encoder.parameters()]
    optimizer = torch.optim.AdamW(
        parameters,
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
    training_seconds = 0.0
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = _load_checkpoint(resume_path)
        _validate_resume_provenance(
            checkpoint,
            domains,
            weights=str(stage1a.get("weights", "ema")),
        )
        for domain, encoder in encoders.items():
            encoder.load_state_dict(checkpoint["encoders"][domain])
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

    alignment_device = domains["cat"].device
    bandwidth = float(matching_config.get("bandwidth_multiplier", 1.0))
    alignment_weight = float(loss_weights.get("infoot_alignment", 0.02))
    warmup_steps = int(loss_weights.get("alignment_warmup_steps", 1000))
    anchor_weight = float(loss_weights.get("latent_anchor", 0.01))
    rec_weights = {
        "cat": float(loss_weights.get("cat_reconstruction", 1.0)),
        "dog": float(loss_weights.get("dog_reconstruction", 1.0)),
    }
    log_every = int(train_config.get("log_every", 20))
    gradient_diagnostics_every = int(train_config.get("gradient_diagnostics_every", 200))
    save_every = int(train_config.get("save_every", 500))
    ema_update_every = int(ema_config.get("update_every", 1))
    solver_options = solver_kwargs(infoot_config)

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
        for domain, value in domains.items():
            batch = next(loaders[domain])
            x0[domain] = batch["x0_latent"].to(
                value.device, dtype=value.dtype, non_blocking=True
            )
            z[domain] = value.branch.encoder(x0[domain])
            with torch.no_grad():
                reference_z[domain] = anchors[domain](x0[domain])
            rec_losses[domain] = _reconstruction_loss(
                value,
                x0[domain][:reconstruction_batch_size],
                z[domain][:reconstruction_batch_size],
            )
            anchor_losses[domain] = encoder_anchor_loss(
                z[domain], reference_z[domain], anchor_variances[domain]
            )

        cat_features = normalize_matching_features(z["cat"].float()).to(alignment_device)
        dog_features = normalize_matching_features(z["dog"].float()).to(alignment_device)
        solution = solve_plain_infoot(
            cat_features.detach(),
            dog_features.detach(),
            bandwidth=bandwidth,
            distance_scale_x=distance_scales["cat"],
            distance_scale_y=distance_scales["dog"],
            seed=seed + step,
            **solver_options,
        )
        infoot_loss = plain_infoot_feature_loss(
            cat_features,
            dog_features,
            solution.coupling.detach(),
            bandwidth=bandwidth,
            distance_scale_x=distance_scales["cat"],
            distance_scale_y=distance_scales["dog"],
            mi_weight=float(infoot_config.get("mi_weight", 1.0)),
            eps=float(infoot_config.get("numerical_epsilon", 1.0e-8)),
        )
        warmup = 1.0 if warmup_steps <= 0 else min(step / warmup_steps, 1.0)
        beta = alignment_weight * warmup
        total = (
            rec_weights["cat"] * rec_losses["cat"].to(alignment_device)
            + rec_weights["dog"] * rec_losses["dog"].to(alignment_device)
            + beta * infoot_loss
            + anchor_weight
            * (anchor_losses["cat"].to(alignment_device) + anchor_losses["dog"].to(alignment_device))
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite Stage 1B loss at step {step}.")
        gradient_diagnostics: dict[str, float] = {}
        if gradient_diagnostics_every > 0 and step % gradient_diagnostics_every == 0:
            reconstruction_objective = (
                rec_weights["cat"] * rec_losses["cat"].to(alignment_device)
                + rec_weights["dog"] * rec_losses["dog"].to(alignment_device)
            )
            gradient_diagnostics = {
                "reconstruction_gradient_norm": _autograd_norm(
                    reconstruction_objective, parameters
                ),
                "weighted_alignment_gradient_norm": _autograd_norm(
                    beta * infoot_loss, parameters
                ),
            }
        total.backward()
        grad_norms = {
            domain: _gradient_norm(list(encoder.parameters()))
            for domain, encoder in encoders.items()
        }
        total_grad_norm = _clip_gradient_norm(
            parameters, train_config.get("grad_clip_norm", 1.0)
        )
        optimizer.step()
        if ema is not None and step % ema_update_every == 0:
            ema.update(encoders)

        elapsed = time.perf_counter() - started
        training_seconds += elapsed
        if step == initial_step + 1 or step % log_every == 0:
            metrics = {
                "event": "train",
                "step": step,
                "loss": float(total.detach().cpu()),
                "cat_reconstruction_loss": float(rec_losses["cat"].detach().cpu()),
                "dog_reconstruction_loss": float(rec_losses["dog"].detach().cpu()),
                "cat_anchor_loss": float(anchor_losses["cat"].detach().cpu()),
                "dog_anchor_loss": float(anchor_losses["dog"].detach().cpu()),
                "infoot_feature_loss": float(infoot_loss.detach().cpu()),
                "infoot_mutual_information": solution.mutual_information,
                "infoot_entropy": solution.entropy,
                "infoot_objective": solution.objective,
                "infoot_row_residual": solution.row_residual,
                "infoot_column_residual": solution.column_residual,
                "infoot_restart": solution.restart,
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
                    distance_scale=distance_scales["cat"],
                ),
                "dog_kernel": kernel_offdiagonal_stats(
                    dog_features.detach(),
                    bandwidth=bandwidth,
                    distance_scale=distance_scales["dog"],
                ),
            }
            metrics.update(gradient_diagnostics)
            print(json.dumps(metrics, sort_keys=True))
            _append_jsonl(log_path, metrics)

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
                ),
            )

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
