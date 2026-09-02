from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from diffusion_ot.integrations.hf_snapshot import (
    effective_project_root,
    find_project_root,
    load_yaml_config,
    resolve_project_local_path,
)


@dataclass
class PDAETrainReport:
    config_path: str
    domain: str
    split: str
    output_dir: str
    initial_step: int
    final_step: int
    train_samples: int
    effective_batch_size: int
    checkpoint_path: str | None
    resumed_from: str | None
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainableEMA:
    """EMA over E and G only; the frozen SiT weights are never duplicated."""

    def __init__(
        self,
        model,
        decay: float = 0.9999,
        warmup_steps: int = 1000,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        if warmup_steps < 0:
            raise ValueError("EMA warmup_steps must be non-negative.")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.num_updates = 0
        self.shadow: dict[str, Any] = {}
        self.reset(model)

    @staticmethod
    def _parameters(model) -> dict[str, Any]:
        return {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @property
    def effective_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        progress = min(1.0, self.num_updates / self.warmup_steps)
        return self.decay * progress

    def reset(self, model) -> None:
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in self._parameters(model).items()
        }
        self.num_updates = 0

    def update(self, model) -> None:
        import torch

        parameters = self._parameters(model)
        if set(parameters) != set(self.shadow):
            raise ValueError("Trainable parameter names changed after EMA initialization.")
        self.num_updates += 1
        decay = self.effective_decay
        with torch.no_grad():
            for name, parameter in parameters.items():
                self.shadow[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "num_updates": self.num_updates,
            "shadow": {
                name: value.detach().cpu()
                for name, value in self.shadow.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any], model) -> None:
        parameters = self._parameters(model)
        saved_shadow = state_dict.get("shadow") or {}
        if set(parameters) != set(saved_shadow):
            raise ValueError("EMA checkpoint does not match the trainable model parameters.")
        self.shadow = {
            name: saved_shadow[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            for name, parameter in parameters.items()
        }
        self.num_updates = int(state_dict.get("num_updates", 0))

    @contextmanager
    def average_parameters(self, model):
        import torch

        parameters = self._parameters(model)
        backup = {
            name: parameter.detach().clone()
            for name, parameter in parameters.items()
        }
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(backup[name])


def _nested(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _load_stage_config(config_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    path = Path(config_path).resolve()
    config = load_yaml_config(path)
    root = effective_project_root(config, fallback=find_project_root(path.parent))
    return config, path, root


def _resolve_config_path(config: dict[str, Any], root: Path, key: str) -> Path:
    value = config.get(key)
    if not value:
        raise ValueError(f"Stage config is missing {key}.")
    return resolve_project_local_path(value, root, field_name=key)


def _torch_dtype_from_model(transformer, fallback):
    for parameter in transformer.parameters():
        return parameter.dtype
    return fallback


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _optimizer_groups(branch, train_config: dict[str, Any]):
    encoder_params = [
        parameter
        for name, parameter in branch.named_parameters()
        if parameter.requires_grad and name.startswith("encoder.")
    ]
    adapter_params = [
        parameter
        for name, parameter in branch.named_parameters()
        if parameter.requires_grad and not name.startswith("encoder.")
    ]
    groups = []
    if encoder_params:
        groups.append(
            {
                "group_name": "encoder",
                "params": encoder_params,
                "lr": float(train_config.get("lr_encoder", train_config.get("lr", 1.0e-4))),
            }
        )
    if adapter_params:
        groups.append(
            {
                "group_name": "adapter",
                "params": adapter_params,
                "lr": float(train_config.get("lr_adapter", train_config.get("lr", 1.0e-4))),
            }
        )
    return groups


def _apply_optimizer_hyperparameters(optimizer, train_config: dict[str, Any]) -> None:
    weight_decay = float(train_config.get("weight_decay", 0.0))
    betas = tuple(float(value) for value in train_config.get("betas", [0.9, 0.999]))
    for index, group in enumerate(optimizer.param_groups):
        group_name = group.get("group_name") or ("encoder" if index == 0 else "adapter")
        group["group_name"] = group_name
        group["lr"] = float(
            train_config.get(
                f"lr_{group_name}",
                train_config.get("lr", 1.0e-4),
            )
        )
        group["weight_decay"] = weight_decay
        group["betas"] = betas


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _gradient_norm(parameters) -> float:
    import torch

    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared_norm = value if squared_norm is None else squared_norm + value
    if squared_norm is None:
        return 0.0
    return float(torch.sqrt(squared_norm).cpu())


def _residual_diagnostics(pred_delta, target_v, base_v) -> dict[str, float]:
    import torch

    from diffusion_ot.losses.pdae_flow import velocity_gap_target

    with torch.no_grad():
        target_delta = velocity_gap_target(target_v.float(), base_v.float(), detach=True)
        pred_delta = pred_delta.detach().float()
        error = pred_delta - target_delta
        target_delta_rms = target_delta.square().mean().sqrt()
        pred_delta_rms = pred_delta.square().mean().sqrt()
        error_rms = error.square().mean().sqrt()
        base_rms = base_v.detach().float().square().mean().sqrt()
        relative_error = error_rms / target_delta_rms.clamp_min(1.0e-8)
        delta_to_base = pred_delta_rms / base_rms.clamp_min(1.0e-8)
        cosine = torch.nn.functional.cosine_similarity(
            pred_delta.flatten(start_dim=1),
            target_delta.flatten(start_dim=1),
            dim=1,
        ).mean()
    return {
        "base_rms": float(base_rms.cpu()),
        "delta_target_cosine": float(cosine.cpu()),
        "delta_to_base_rms": float(delta_to_base.cpu()),
        "error_rms": float(error_rms.cpu()),
        "pred_delta_rms": float(pred_delta_rms.cpu()),
        "relative_error": float(relative_error.cpu()),
        "target_delta_rms": float(target_delta_rms.cpu()),
    }


def _empty_variant_stats() -> dict[str, float]:
    return {
        "count": 0.0,
        "error_mse_sum": 0.0,
        "target_mse_sum": 0.0,
        "pred_mse_sum": 0.0,
        "cosine_sum": 0.0,
    }


def _update_variant_stats(stats: dict[str, float], pred_delta, target_delta) -> None:
    import torch

    pred_flat = pred_delta.detach().float().flatten(start_dim=1)
    target_flat = target_delta.detach().float().flatten(start_dim=1)
    if not bool(torch.isfinite(pred_flat).all() and torch.isfinite(target_flat).all()):
        raise FloatingPointError("Non-finite value encountered during z-dependence validation.")
    error_mse = (pred_flat - target_flat).square().mean(dim=1)
    target_mse = target_flat.square().mean(dim=1)
    pred_mse = pred_flat.square().mean(dim=1)
    cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1)
    stats["count"] += float(pred_flat.shape[0])
    stats["error_mse_sum"] += float(error_mse.sum().cpu())
    stats["target_mse_sum"] += float(target_mse.sum().cpu())
    stats["pred_mse_sum"] += float(pred_mse.sum().cpu())
    stats["cosine_sum"] += float(cosine.sum().cpu())


def _finalize_variant_stats(stats: dict[str, float]) -> dict[str, float | int]:
    count = max(stats["count"], 1.0)
    mse = stats["error_mse_sum"] / count
    target_mse = stats["target_mse_sum"] / count
    pred_mse = stats["pred_mse_sum"] / count
    return {
        "count": int(stats["count"]),
        "mse": mse,
        "error_rms": math.sqrt(max(mse, 0.0)),
        "target_rms": math.sqrt(max(target_mse, 0.0)),
        "pred_rms": math.sqrt(max(pred_mse, 0.0)),
        "relative_error": math.sqrt(max(mse, 0.0) / max(target_mse, 1.0e-12)),
        "cosine": stats["cosine_sum"] / count,
    }


def _z_statistics(z_values) -> dict[str, float | int]:
    import torch

    z = torch.cat(z_values, dim=0).float()
    if not bool(torch.isfinite(z).all()):
        raise FloatingPointError("Non-finite semantic latent encountered during validation.")
    centered = z - z.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    if float(energy.sum()) > 0.0:
        probabilities = energy / energy.sum()
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
        )
    else:
        effective_rank = energy.new_tensor(0.0)
    norms = z.norm(dim=1)
    return {
        "count": int(z.shape[0]),
        "dim": int(z.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std(unbiased=False)),
        "feature_std_mean": float(z.std(dim=0, unbiased=False).mean()),
        "effective_rank": float(effective_rank),
    }


def _evaluate_z_dependence(
    branch,
    transformer,
    loader,
    *,
    device: str,
    model_dtype,
    flow_direction: str,
    time_eps: float,
    null_label: int | None,
    step: int,
    num_batches: int,
    num_time_bins: int,
    seed: int,
    ema: TrainableEMA | None,
    use_ema: bool,
) -> dict[str, Any]:
    import torch

    from diffusion_ot.losses.pdae_flow import make_linear_flow_target, velocity_gap_target
    from diffusion_ot.models.pdae_sit import make_null_class_labels

    if num_batches <= 0:
        raise ValueError("evaluation.num_batches must be positive.")
    if num_time_bins <= 0:
        raise ValueError("evaluation.num_time_bins must be positive.")

    variants = ("correct_z", "shuffled_z", "zero_z")
    global_stats = {name: _empty_variant_stats() for name in variants}
    binned_stats = [
        {name: _empty_variant_stats() for name in variants}
        for _ in range(num_time_bins)
    ]
    z_values = []

    device_object = torch.device(device)
    noise_generator = (
        torch.Generator(device=device_object)
        if device_object.type == "cuda"
        else torch.Generator()
    )
    noise_generator.manual_seed(seed)

    was_training = branch.training
    parameter_context = (
        ema.average_parameters(branch)
        if use_ema and ema is not None
        else nullcontext()
    )
    with parameter_context:
        branch.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if batch_index >= num_batches:
                    break
                x0 = batch["x0_latent"].to(
                    device=device,
                    dtype=model_dtype,
                    non_blocking=True,
                )
                noise = torch.randn(
                    x0.shape,
                    generator=noise_generator,
                    device=x0.device,
                    dtype=x0.dtype,
                )
                t = torch.rand(
                    x0.shape[0],
                    generator=noise_generator,
                    device=x0.device,
                    dtype=x0.dtype,
                ).clamp(time_eps, 1.0 - time_eps)
                target = make_linear_flow_target(
                    x0,
                    noise=noise,
                    t=t,
                    eps=time_eps,
                    direction=flow_direction,
                )
                class_labels = make_null_class_labels(
                    transformer,
                    batch_size=x0.shape[0],
                    device=x0.device,
                    null_label=null_label,
                )
                z = branch.encode(x0)
                z_values.append(z.detach().float().cpu())

                correct_output = branch.predict_with_z(
                    x_t=target.x_t,
                    timestep=target.t,
                    z=z,
                    class_labels=class_labels,
                )
                base_v = correct_output.base_sample.detach().float()
                target_delta = velocity_gap_target(
                    target.target_v.float(),
                    base_v,
                    detach=True,
                )
                predictions = {
                    "correct_z": correct_output.delta_sample.detach().float(),
                    "shuffled_z": branch.predict_with_z(
                        x_t=target.x_t,
                        timestep=target.t,
                        z=torch.roll(z, shifts=1, dims=0),
                        class_labels=class_labels,
                    ).delta_sample.detach().float(),
                    "zero_z": branch.predict_with_z(
                        x_t=target.x_t,
                        timestep=target.t,
                        z=torch.zeros_like(z),
                        class_labels=class_labels,
                    ).delta_sample.detach().float(),
                }
                time_bin = (target.t.detach().float() * num_time_bins).long()
                time_bin = time_bin.clamp(0, num_time_bins - 1)
                for name, prediction in predictions.items():
                    _update_variant_stats(global_stats[name], prediction, target_delta)
                    for bin_index in range(num_time_bins):
                        mask = time_bin == bin_index
                        if bool(mask.any()):
                            _update_variant_stats(
                                binned_stats[bin_index][name],
                                prediction[mask],
                                target_delta[mask],
                            )
        branch.train(was_training)

    if not z_values:
        raise ValueError("Validation loader produced no batches.")

    finalized = {
        name: _finalize_variant_stats(stats)
        for name, stats in global_stats.items()
    }
    correct_mse = float(finalized["correct_z"]["mse"])
    shuffled_mse = float(finalized["shuffled_z"]["mse"])
    zero_mse = float(finalized["zero_z"]["mse"])
    time_bins = []
    for bin_index, bin_stats in enumerate(binned_stats):
        values = {
            name: _finalize_variant_stats(stats)
            for name, stats in bin_stats.items()
        }
        bin_correct_mse = float(values["correct_z"]["mse"])
        time_bins.append(
            {
                "index": bin_index,
                "t_min": bin_index / num_time_bins,
                "t_max": (bin_index + 1) / num_time_bins,
                "count": values["correct_z"]["count"],
                "z_gain": (
                    float(values["shuffled_z"]["mse"]) / max(bin_correct_mse, 1.0e-12) - 1.0
                    if values["correct_z"]["count"]
                    else None
                ),
                "correct_z": values["correct_z"],
                "shuffled_z": values["shuffled_z"],
                "zero_z": values["zero_z"],
            }
        )

    return {
        "event": "validation",
        "step": int(step),
        "use_ema": bool(use_ema and ema is not None),
        "num_samples": int(finalized["correct_z"]["count"]),
        "z_gain": shuffled_mse / max(correct_mse, 1.0e-12) - 1.0,
        "zero_z_gain": zero_mse / max(correct_mse, 1.0e-12) - 1.0,
        "correct_z": finalized["correct_z"],
        "shuffled_z": finalized["shuffled_z"],
        "zero_z": finalized["zero_z"],
        "z_statistics": _z_statistics(z_values),
        "time_bins": time_bins,
    }


def _resolve_resume_path(
    resume_from: str | Path | None,
    root: Path,
    checkpoint_path: Path,
) -> Path | None:
    if resume_from is None or resume_from is False:
        return None
    value = str(resume_from).strip()
    if not value:
        return None
    if value.lower() == "latest":
        return checkpoint_path
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _load_torch_checkpoint(path: Path, device: str):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    step: int,
    domain: str,
    branch,
    optimizer,
    ema: TrainableEMA | None,
    config: dict[str, Any],
    train_state: dict[str, Any],
    loader_generator,
) -> None:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "step": int(step),
        "domain": domain,
        "model": branch.pdae_state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "config": config,
        "train_state": train_state,
        "rng_state": torch.get_rng_state(),
        "dataloader_generator_state": loader_generator.get_state(),
    }
    model_device = next(branch.parameters()).device
    if model_device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(model_device).cpu()
    temporary_path = checkpoint_path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)


def train_pdae_domain(
    config_path: str | Path,
    device: str | None = None,
    max_steps: int | None = None,
    resume_from: str | Path | None = None,
    dry_run: bool = False,
) -> PDAETrainReport:
    import torch
    from torch.utils.data import DataLoader

    from diffusion_ot.data.latent_dataset import CachedLatentDataset, collate_latent_batch
    from diffusion_ot.integrations.sit_diffusers import load_sit_components, validate_transformer_config
    from diffusion_ot.losses.pdae_flow import (
        make_linear_flow_target,
        pdae_flow_snr_weight,
        pdae_velocity_gap_loss,
    )
    from diffusion_ot.models.pdae_sit import build_pdae_sit_branch, make_null_class_labels

    config, resolved_config_path, root = _load_stage_config(config_path)
    train_config = _nested(config, "train")
    dataloader_config = _nested(config, "dataloader")
    flow_config = _nested(config, "flow")
    class_config = _nested(config, "class_conditioning")
    loss_weight_config = _nested(config, "loss_weighting")
    ema_config = _nested(config, "ema")
    evaluation_config = _nested(config, "evaluation")

    domain = str(config.get("domain", "")).lower()
    if not domain:
        raise ValueError("Stage config is missing domain.")
    split = str(config.get("split", "train")).lower()

    data_config_path = _resolve_config_path(config, root, "data_config")
    model_config_path = _resolve_config_path(config, root, "model_config")
    model_config = load_yaml_config(model_config_path)
    pretrained_config_path = (
        resolve_project_local_path(config["pretrained_config"], root, field_name="pretrained_config")
        if config.get("pretrained_config")
        else resolve_project_local_path(model_config["pretrained"], root, field_name="model.pretrained")
    )
    output_dir = resolve_project_local_path(
        config.get("output_dir", f"outputs/stage1a_{domain}_sit_b2"),
        root,
        field_name="output_dir",
    )

    seed = int(train_config.get("seed", 20260901))
    torch.manual_seed(seed)
    device = device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    limit = train_config.get("limit")
    random_horizontal_flip = float(dataloader_config.get("random_horizontal_flip", 0.0))
    dataset = CachedLatentDataset(
        data_config_path=data_config_path,
        domain=domain,
        split=split,
        project_root=root,
        limit=int(limit) if limit is not None else None,
        validate_exists=not dry_run,
        random_horizontal_flip=random_horizontal_flip,
    )

    batch_size = int(dataloader_config.get("batch_size", 16))
    accumulation_steps = int(dataloader_config.get("gradient_accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError("dataloader.gradient_accumulation_steps must be positive.")
    effective_batch_size = batch_size * accumulation_steps
    final_step = int(max_steps if max_steps is not None else train_config.get("max_steps", 1000))
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    requested_resume = resume_from if resume_from is not None else train_config.get("resume_from")
    resolved_resume_path = _resolve_resume_path(requested_resume, root, checkpoint_path)

    if dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = PDAETrainReport(
            config_path=str(resolved_config_path),
            domain=domain,
            split=split,
            output_dir=str(output_dir),
            initial_step=0,
            final_step=0,
            train_samples=len(dataset),
            effective_batch_size=effective_batch_size,
            checkpoint_path=None,
            resumed_from=str(resolved_resume_path) if resolved_resume_path is not None else None,
            dry_run=True,
        )
        _write_json(output_dir / "dry_run_report.json", report.to_dict())
        return report

    components = load_sit_components(
        pretrained_config_path,
        project_root=root,
        device=device,
        torch_dtype=str(config.get("torch_dtype", "float32")),
    )
    transformer = components.transformer
    transformer.eval()

    mismatches = validate_transformer_config(
        transformer,
        model_config.get("expected_transformer_config") or {},
    )
    if mismatches:
        raise ValueError(
            "Unexpected SiT transformer config:\n"
            + "\n".join(f"  - {item}" for item in mismatches)
        )

    branch = build_pdae_sit_branch(transformer, model_config=model_config, stage_config=config)
    model_dtype = _torch_dtype_from_model(transformer, torch.float32)
    branch.to(device=device, dtype=model_dtype)
    branch.train()

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(dataloader_config.get("num_workers", 4)),
        pin_memory=bool(dataloader_config.get("pin_memory", True)),
        drop_last=bool(dataloader_config.get("drop_last", True)),
        collate_fn=collate_latent_batch,
        generator=loader_generator,
    )
    if len(loader) == 0:
        raise ValueError("Training DataLoader is empty; reduce batch_size or disable drop_last.")
    batch_iter = _cycle(loader)

    optimizer = torch.optim.AdamW(
        _optimizer_groups(branch, train_config),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
        betas=tuple(float(value) for value in train_config.get("betas", [0.9, 0.999])),
    )
    ema = (
        TrainableEMA(
            branch,
            decay=float(ema_config.get("decay", 0.9999)),
            warmup_steps=int(ema_config.get("warmup_steps", 1000)),
        )
        if bool(ema_config.get("enabled", True))
        else None
    )

    initial_step = 0
    loss_ema: float | None = None
    clip_events = 0
    clip_checks = 0
    if resolved_resume_path is not None:
        if not resolved_resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resolved_resume_path}")
        checkpoint = _load_torch_checkpoint(resolved_resume_path, device=device)
        branch.load_pdae_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _apply_optimizer_hyperparameters(optimizer, train_config)
        if ema is not None:
            if checkpoint.get("ema") is not None:
                ema.load_state_dict(checkpoint["ema"], branch)
            else:
                ema.reset(branch)
        initial_step = int(checkpoint.get("step", 0))
        saved_train_state = checkpoint.get("train_state") or {}
        loss_ema = saved_train_state.get("loss_ema")
        clip_events = int(saved_train_state.get("clip_events", 0))
        clip_checks = int(saved_train_state.get("clip_checks", 0))
        if checkpoint.get("rng_state") is not None:
            torch.set_rng_state(checkpoint["rng_state"].cpu())
        if checkpoint.get("dataloader_generator_state") is not None:
            loader_generator.set_state(checkpoint["dataloader_generator_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device=device)
        elif torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            saved_states = checkpoint["cuda_rng_state_all"]
            for device_index, state in enumerate(saved_states[: torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state.cpu(), device=device_index)
    if initial_step > final_step:
        raise ValueError(
            f"Checkpoint step {initial_step} exceeds requested max_steps {final_step}."
        )

    grad_clip_norm = train_config.get("grad_clip_norm", 1.0)
    flow_direction = str(flow_config.get("direction", "noise_to_data"))
    time_eps = float(flow_config.get("time_eps", 1.0e-5))
    null_label_value = class_config.get("null_label")
    null_label = int(null_label_value) if null_label_value is not None else None
    log_every = int(train_config.get("log_every", 20))
    save_every = int(train_config.get("save_every", 1000))
    if log_every <= 0 or save_every <= 0:
        raise ValueError("train.log_every and train.save_every must be positive.")
    ema_update_every = int(ema_config.get("update_every", 1))
    if ema_update_every <= 0:
        raise ValueError("ema.update_every must be positive.")

    evaluation_enabled = bool(evaluation_config.get("enabled", True))
    evaluation_every = int(evaluation_config.get("every_steps", 1000))
    validation_loader = None
    if evaluation_enabled:
        validation_dataset = CachedLatentDataset(
            data_config_path=data_config_path,
            domain=domain,
            split=str(evaluation_config.get("split", "val")),
            project_root=root,
            limit=(
                int(evaluation_config["limit"])
                if evaluation_config.get("limit") is not None
                else None
            ),
            validate_exists=True,
            random_horizontal_flip=0.0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(evaluation_config.get("batch_size", batch_size)),
            shuffle=False,
            num_workers=int(evaluation_config.get("num_workers", dataloader_config.get("num_workers", 4))),
            pin_memory=bool(dataloader_config.get("pin_memory", True)),
            drop_last=False,
            collate_fn=collate_latent_batch,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)
    metrics_path = output_dir / "metrics.jsonl"
    if resolved_resume_path is None:
        metrics_path.write_text("", encoding="utf-8")
        _save_checkpoint(
            checkpoint_path,
            step=0,
            domain=domain,
            branch=branch,
            optimizer=optimizer,
            ema=ema,
            config=config,
            train_state={
                "loss_ema": loss_ema,
                "clip_events": clip_events,
                "clip_checks": clip_checks,
            },
            loader_generator=loader_generator,
        )

    def run_validation(step: int) -> None:
        if validation_loader is None:
            return
        metrics = _evaluate_z_dependence(
            branch,
            transformer,
            validation_loader,
            device=device,
            model_dtype=model_dtype,
            flow_direction=flow_direction,
            time_eps=time_eps,
            null_label=null_label,
            step=step,
            num_batches=int(evaluation_config.get("num_batches", 4)),
            num_time_bins=int(evaluation_config.get("num_time_bins", 10)),
            seed=int(evaluation_config.get("seed", seed + 1)),
            ema=ema,
            use_ema=bool(evaluation_config.get("use_ema", True)),
        )
        _append_jsonl(metrics_path, metrics)
        print(
            json.dumps(
                {
                    "event": "validation",
                    "step": step,
                    "correct_z_relative_error": metrics["correct_z"]["relative_error"],
                    "correct_z_cosine": metrics["correct_z"]["cosine"],
                    "z_gain": metrics["z_gain"],
                    "zero_z_gain": metrics["zero_z_gain"],
                    "z_effective_rank": metrics["z_statistics"]["effective_rank"],
                },
                sort_keys=True,
            )
        )

    if evaluation_enabled and bool(evaluation_config.get("evaluate_at_start", True)):
        run_validation(initial_step)

    trainable_parameters = [
        parameter for parameter in branch.parameters() if parameter.requires_grad
    ]
    encoder_parameters = [
        parameter
        for name, parameter in branch.named_parameters()
        if parameter.requires_grad and name.startswith("encoder.")
    ]
    adapter_parameters = [
        parameter
        for name, parameter in branch.named_parameters()
        if parameter.requires_grad and not name.startswith("encoder.")
    ]

    for step in range(initial_step + 1, final_step + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        weight_mean_sum = 0.0
        diagnostic_sums: dict[str, float] = {}

        for _ in range(accumulation_steps):
            batch = next(batch_iter)
            x0 = batch["x0_latent"].to(
                device=device,
                dtype=model_dtype,
                non_blocking=True,
            )
            target = make_linear_flow_target(
                x0,
                eps=time_eps,
                direction=flow_direction,
            )
            class_labels = make_null_class_labels(
                transformer,
                batch_size=x0.shape[0],
                device=x0.device,
                null_label=null_label,
            )
            output = branch(
                x0_latent=x0,
                x_t=target.x_t,
                timestep=target.t,
                class_labels=class_labels,
            )
            pred_delta_v = output.delta_sample
            base_v = output.base_sample
            weight = pdae_flow_snr_weight(
                t=target.t.float(),
                direction=flow_direction,
                gamma=float(loss_weight_config.get("gamma", 0.1)),
                normalize_mean_to=loss_weight_config.get("normalize_mean_to", 1.0),
                normalization_mode=str(loss_weight_config.get("normalization_mode", "fixed_uniform")),
                normalization_samples=int(loss_weight_config.get("normalization_samples", 65536)),
                clamp_min=loss_weight_config.get("clamp_min", 0.001),
                clamp_max=loss_weight_config.get("clamp_max"),
            )
            loss = pdae_velocity_gap_loss(
                pred_delta_v.float(),
                target.target_v.float(),
                base_v.float(),
                weight=weight.to(device=pred_delta_v.device, dtype=torch.float32),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite Stage 1A loss at step {step}.")
            (loss / accumulation_steps).backward()

            loss_sum += float(loss.detach().cpu())
            weight_mean_sum += float(weight.detach().mean().cpu())
            diagnostics = _residual_diagnostics(
                pred_delta_v,
                target.target_v,
                base_v,
            )
            for name, value in diagnostics.items():
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + value

        should_log = step == initial_step + 1 or step % log_every == 0
        encoder_grad_norm = _gradient_norm(encoder_parameters) if should_log else None
        adapter_grad_norm = _gradient_norm(adapter_parameters) if should_log else None
        if grad_clip_norm is not None:
            total_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                float(grad_clip_norm),
            )
            total_grad_norm = float(total_grad_norm_tensor.detach().cpu())
            clip_checks += 1
            if math.isfinite(total_grad_norm) and total_grad_norm > float(grad_clip_norm):
                clip_events += 1
        else:
            total_grad_norm = _gradient_norm(trainable_parameters)
        if not math.isfinite(total_grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}.")
        optimizer.step()
        if ema is not None and step % ema_update_every == 0:
            ema.update(branch)

        loss_value = loss_sum / accumulation_steps
        loss_ema = loss_value if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_value
        if should_log:
            metrics: dict[str, Any] = {
                "event": "train",
                "step": step,
                "loss": loss_value,
                "loss_ema": loss_ema,
                "weight_mean": weight_mean_sum / accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "total_grad_norm_pre_clip": total_grad_norm,
                "encoder_grad_norm_pre_clip": encoder_grad_norm,
                "adapter_grad_norm_pre_clip": adapter_grad_norm,
                "gradient_clip_fraction": clip_events / max(clip_checks, 1),
                "ema_decay": ema.effective_decay if ema is not None else None,
            }
            metrics.update(
                {
                    name: value / accumulation_steps
                    for name, value in diagnostic_sums.items()
                }
            )
            print(json.dumps(metrics, sort_keys=True))
            _append_jsonl(metrics_path, metrics)

        if step % save_every == 0 or step == final_step:
            _save_checkpoint(
                checkpoint_path,
                step=step,
                domain=domain,
                branch=branch,
                optimizer=optimizer,
                ema=ema,
                config=config,
                train_state={
                    "loss_ema": loss_ema,
                    "clip_events": clip_events,
                    "clip_checks": clip_checks,
                },
                loader_generator=loader_generator,
            )

        if evaluation_enabled and evaluation_every > 0 and step % evaluation_every == 0:
            run_validation(step)

    return PDAETrainReport(
        config_path=str(resolved_config_path),
        domain=domain,
        split=split,
        output_dir=str(output_dir),
        initial_step=initial_step,
        final_step=final_step,
        train_samples=len(dataset),
        effective_batch_size=effective_batch_size,
        checkpoint_path=str(checkpoint_path) if checkpoint_path.is_file() else None,
        resumed_from=str(resolved_resume_path) if resolved_resume_path is not None else None,
        dry_run=False,
    )
