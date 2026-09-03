from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch

from diffusion_ot.integrations.hf_snapshot import (
    effective_project_root,
    find_project_root,
    load_yaml_config,
    resolve_project_local_path,
)


_ALLOWED_Z_VARIANTS = {"correct_z", "shuffled_z", "zero_z"}


@dataclass
class Stage1ASmokeReport:
    protocol: str
    training_config_path: str
    evaluation_config_path: str
    checkpoint_path: str
    output_dir: str
    domain: str
    split: str
    checkpoint_step: int
    weights: str
    seed: int
    num_samples: int
    num_steps: int
    sample_ids: list[str | None]
    row_order: list[str]
    metrics: dict[str, dict[str, float]]
    grid_path: str
    extra_reports: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage1ARoundTripReport:
    protocol: str
    training_config_path: str
    evaluation_config_path: str
    checkpoint_path: str
    output_dir: str
    domain: str
    split: str
    checkpoint_step: int
    weights: str
    seed: int
    num_samples: int
    backward_num_steps: int
    forward_num_steps: int
    sample_ids: list[str | None]
    row_order: list[str]
    metrics: dict[str, dict[str, float]]
    inferred_noise_stats: dict[str, float]
    grid_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedStage1AEvaluator:
    branch: Any
    transformer: Any
    vae: Any
    dataset: Any
    training_config: dict[str, Any]
    evaluation_config: dict[str, Any]
    project_root: Path
    training_config_path: Path
    evaluation_config_path: Path
    checkpoint_path: Path
    checkpoint_step: int
    domain: str
    split: str
    device: str
    model_dtype: torch.dtype
    weights: str


def _nested(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _resolve_config_path(config: dict[str, Any], root: Path, key: str) -> Path:
    value = config.get(key)
    if not value:
        raise ValueError(f"Stage 1A training config is missing {key}.")
    return resolve_project_local_path(value, root, field_name=key)


def _torch_dtype_from_model(model: Any, fallback: torch.dtype = torch.float32) -> torch.dtype:
    for parameter in model.parameters():
        return parameter.dtype
    return fallback


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Not a Stage 1A PDAE checkpoint: {path}")
    return checkpoint


def _apply_ema_weights(branch: Any, checkpoint: dict[str, Any]) -> None:
    ema = checkpoint.get("ema") or {}
    shadow = ema.get("shadow") or {}
    parameters = {
        name: parameter
        for name, parameter in branch.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(set(parameters) - set(shadow))
    unexpected = sorted(set(shadow) - set(parameters))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing EMA parameters: {missing[:8]}")
        if unexpected:
            details.append(f"unexpected EMA parameters: {unexpected[:8]}")
        raise ValueError("EMA checkpoint does not match E/G: " + "; ".join(details))

    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(shadow[name].to(device=parameter.device, dtype=parameter.dtype))


def _validate_eval_config(config: dict[str, Any]) -> None:
    sampling = _nested(config, "sampling")
    if str(sampling.get("solver", "euler")) != "euler":
        raise ValueError("The first Stage 1A evaluator supports sampling.solver=euler only.")
    if str(sampling.get("direction", "noise_to_data")) != "noise_to_data":
        raise ValueError("Stage 1A evaluation must match the trained noise_to_data field.")
    if str(sampling.get("time_evaluation", "midpoint")) != "midpoint":
        raise ValueError("The first Stage 1A evaluator supports midpoint time evaluation only.")
    if not bool(sampling.get("fixed_starting_noise", True)):
        raise ValueError("Fixed starting noise is required for fair z-variant comparisons.")

    variants = list(sampling.get("variants") or [])
    if not variants or set(variants) - _ALLOWED_Z_VARIANTS:
        raise ValueError(
            "sampling.variants must be a non-empty subset of "
            "[correct_z, shuffled_z, zero_z]."
        )
    inferred_config = _nested(sampling, "inferred_noise")
    if bool(inferred_config.get("enabled", False)):
        inferred_variants = list(inferred_config.get("variants") or ["correct_z"])
        if not inferred_variants or set(inferred_variants) - _ALLOWED_Z_VARIANTS:
            raise ValueError(
                "sampling.inferred_noise.variants must be a non-empty subset of "
                "[correct_z, shuffled_z, zero_z]."
            )
        if "correct_z" not in inferred_variants:
            raise ValueError("sampling.inferred_noise.variants must include correct_z.")


def load_stage1a_evaluator(
    training_config_path: str | Path,
    evaluation_config_path: str | Path,
    *,
    device: str | None = None,
    weights: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> LoadedStage1AEvaluator:
    """Load one domain's frozen SiT, trained E/G, VAE, and validation latents."""
    from diffusion_ot.data.latent_dataset import CachedLatentDataset
    from diffusion_ot.integrations.sit_diffusers import (
        load_sit_components,
        validate_transformer_config,
    )
    from diffusion_ot.models.pdae_sit import build_pdae_sit_branch

    train_path = Path(training_config_path).expanduser().resolve()
    eval_path = Path(evaluation_config_path).expanduser().resolve()
    train_config = load_yaml_config(train_path)
    eval_config = load_yaml_config(eval_path)
    _validate_eval_config(eval_config)

    root = effective_project_root(
        train_config,
        fallback=find_project_root(train_path.parent),
    )
    eval_root = effective_project_root(eval_config, fallback=root)
    if eval_root != root:
        raise ValueError(
            "Training and evaluation configs resolve to different project roots: "
            f"{root} and {eval_root}."
        )

    domain = str(train_config.get("domain", "")).lower()
    if not domain:
        raise ValueError("Stage 1A training config is missing domain.")
    split = str(eval_config.get("split", "val")).lower()
    selected_weights = str(weights or eval_config.get("weights", "ema")).lower()
    if selected_weights not in {"ema", "raw"}:
        raise ValueError("weights must be either 'ema' or 'raw'.")
    selected_device = str(device or train_config.get("device", "cuda:0"))

    model_config_path = _resolve_config_path(train_config, root, "model_config")
    data_config_path = _resolve_config_path(train_config, root, "data_config")
    model_config = load_yaml_config(model_config_path)
    pretrained_config_path = _resolve_config_path(model_config, root, "pretrained")

    if checkpoint_path is None:
        output_dir = resolve_project_local_path(
            train_config.get("output_dir", f"outputs/stage1a_{domain}_sit_b2"),
            root,
            field_name="output_dir",
        )
        resolved_checkpoint = output_dir / "checkpoints" / "latest.pt"
    else:
        candidate = Path(checkpoint_path).expanduser()
        resolved_checkpoint = (
            candidate.resolve()
            if candidate.is_absolute()
            else resolve_project_local_path(candidate, root, field_name="checkpoint_path")
        )
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Stage 1A checkpoint not found: {resolved_checkpoint}")

    checkpoint = _load_checkpoint(resolved_checkpoint)
    checkpoint_domain = str(checkpoint.get("domain", domain)).lower()
    if checkpoint_domain != domain:
        raise ValueError(
            f"Checkpoint domain is {checkpoint_domain}, but training config domain is {domain}."
        )

    components = load_sit_components(
        pretrained_config_path,
        project_root=root,
        device=selected_device,
        torch_dtype=str(train_config.get("torch_dtype", "float32")),
    )
    mismatches = validate_transformer_config(
        components.transformer,
        model_config.get("expected_transformer_config") or {},
    )
    if mismatches:
        raise ValueError(
            "Unexpected SiT transformer config:\n"
            + "\n".join(f"  - {item}" for item in mismatches)
        )

    branch = build_pdae_sit_branch(
        components.transformer,
        model_config=model_config,
        stage_config=train_config,
    )
    model_dtype = _torch_dtype_from_model(components.transformer)
    branch.to(device=selected_device, dtype=model_dtype)
    branch.load_pdae_state_dict(checkpoint["model"])
    if selected_weights == "ema":
        if checkpoint.get("ema") is None:
            raise ValueError("EMA evaluation requested, but this checkpoint has no EMA state.")
        _apply_ema_weights(branch, checkpoint)
    branch.eval()
    components.vae.eval()

    dataset = CachedLatentDataset(
        data_config_path=data_config_path,
        domain=domain,
        split=split,
        project_root=root,
        validate_exists=True,
        random_horizontal_flip=0.0,
    )
    return LoadedStage1AEvaluator(
        branch=branch,
        transformer=components.transformer,
        vae=components.vae,
        dataset=dataset,
        training_config=train_config,
        evaluation_config=eval_config,
        project_root=root,
        training_config_path=train_path,
        evaluation_config_path=eval_path,
        checkpoint_path=resolved_checkpoint,
        checkpoint_step=int(checkpoint.get("step", 0)),
        domain=domain,
        split=split,
        device=selected_device,
        model_dtype=model_dtype,
        weights=selected_weights,
    )


@torch.inference_mode()
def integrate_pdae_flow(
    branch: Any,
    transformer: Any,
    initial_state: torch.Tensor,
    z: torch.Tensor,
    *,
    num_steps: int,
    start_time: float = 0.0,
    end_time: float = 1.0,
    guidance_scale: float = 1.0,
    null_label: int | None = None,
) -> torch.Tensor:
    """Euler-integrate the Stage 1A velocity field using midpoint time samples."""
    from diffusion_ot.models.pdae_sit import make_null_class_labels

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance_scale must be finite.")

    state = initial_state.clone()
    dt = (float(end_time) - float(start_time)) / int(num_steps)
    class_labels = make_null_class_labels(
        transformer,
        batch_size=state.shape[0],
        device=state.device,
        null_label=null_label,
    )
    for index in range(int(num_steps)):
        time_value = float(start_time) + (index + 0.5) * dt
        timestep = torch.full(
            (state.shape[0],),
            time_value,
            device=state.device,
            dtype=torch.float32,
        )
        output = branch.predict_with_z(
            x_t=state,
            timestep=timestep,
            z=z,
            class_labels=class_labels,
        )
        velocity = output.base_sample + float(guidance_scale) * output.delta_sample
        state = state + dt * velocity
    return state


@torch.inference_mode()
def infer_starting_noise(
    branch: Any,
    transformer: Any,
    x0_latent: torch.Tensor,
    z: torch.Tensor,
    *,
    num_steps: int,
    guidance_scale: float = 1.0,
    null_label: int | None = None,
) -> torch.Tensor:
    """Invert data to t=0; this protocol is reported separately from fixed noise."""
    return integrate_pdae_flow(
        branch,
        transformer,
        x0_latent,
        z,
        num_steps=num_steps,
        start_time=1.0,
        end_time=0.0,
        guidance_scale=guidance_scale,
        null_label=null_label,
    )


@torch.inference_mode()
def decode_vae_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    vae_dtype = _torch_dtype_from_model(vae)
    decoded = vae.decode((latents / scaling_factor).to(dtype=vae_dtype))
    images = decoded.sample if hasattr(decoded, "sample") else decoded[0]
    return ((images.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def _deterministic_subset(dataset: Any, count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("Requested sample count must be positive.")
    if len(dataset) < count:
        raise ValueError(f"Requested {count} samples, but the dataset has only {len(dataset)}.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return [dataset[index] for index in indices]


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    from diffusion_ot.data.latent_dataset import collate_latent_batch

    return collate_latent_batch(items)


def _noise_like(value: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=value.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        value.shape,
        generator=generator,
        device=value.device,
        dtype=value.dtype,
    )


def _variant_z(z: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "correct_z":
        return z
    if variant == "zero_z":
        return torch.zeros_like(z)
    if variant == "shuffled_z":
        if z.shape[0] < 2:
            raise ValueError("shuffled_z evaluation needs at least two samples.")
        return torch.roll(z, shifts=1, dims=0)
    raise ValueError(f"Unknown z variant: {variant}")


@torch.inference_mode()
def reconstruct_z_variants(
    branch: Any,
    transformer: Any,
    starting_noise: torch.Tensor,
    z: torch.Tensor,
    variants: list[str],
    *,
    num_steps: int,
    guidance_scale: float = 1.0,
    null_label: int | None = None,
) -> dict[str, torch.Tensor]:
    """Reconstruct z variants from one immutable starting-noise batch."""
    outputs: dict[str, torch.Tensor] = {}
    for variant in variants:
        outputs[variant] = integrate_pdae_flow(
            branch,
            transformer,
            starting_noise,
            _variant_z(z, variant),
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            null_label=null_label,
        )
    return outputs


def nearest_neighbor_indices(
    z: torch.Tensor,
    *,
    k: int = 5,
    metric: str = "cosine",
    exclude_self: bool = True,
) -> torch.Tensor:
    """Return within-bank neighbor indices, optionally masking each query itself."""
    if z.ndim != 2:
        raise ValueError(f"Expected z with shape [N, D], got {tuple(z.shape)}.")
    available = z.shape[0] - int(exclude_self)
    if k <= 0 or k > available:
        raise ValueError(f"k must be in [1, {available}] for this latent bank.")

    values = z.float()
    if metric == "cosine":
        values = torch.nn.functional.normalize(values, dim=1)
        distances = 1.0 - values @ values.transpose(0, 1)
    elif metric in {"euclidean", "l2"}:
        distances = torch.cdist(values, values, p=2)
    else:
        raise ValueError("metric must be 'cosine', 'euclidean', or 'l2'.")
    if exclude_self:
        distances.fill_diagonal_(float("inf"))
    return torch.topk(distances, k=int(k), dim=1, largest=False).indices


def interpolate_latents(
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    *,
    num_steps: int,
    mode: str = "linear",
) -> torch.Tensor:
    """Build endpoint-inclusive latent interpolations for one or more pairs."""
    if z_start.shape != z_end.shape:
        raise ValueError("Interpolation endpoints must have identical shapes.")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2 to include both endpoints.")
    if mode != "linear":
        raise ValueError("The first Stage 1A evaluator supports linear interpolation only.")

    alpha_shape = (int(num_steps),) + (1,) * z_start.ndim
    alpha = torch.linspace(
        0.0,
        1.0,
        int(num_steps),
        device=z_start.device,
        dtype=z_start.dtype,
    ).reshape(alpha_shape)
    return (1.0 - alpha) * z_start.unsqueeze(0) + alpha * z_end.unsqueeze(0)


def _mse_and_psnr(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = float(torch.mean((prediction.float() - target.float()) ** 2).cpu())
    psnr = float("inf") if mse == 0.0 else -10.0 * math.log10(mse)
    return {"mse": mse, "psnr": psnr}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _save_grid(path: Path, rows: list[torch.Tensor], samples_per_row: int) -> None:
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        torch.cat([row.detach().cpu() for row in rows], dim=0),
        str(path),
        nrow=int(samples_per_row),
        padding=2,
        pad_value=1.0,
    )


def _evaluation_output_dir(evaluator: LoadedStage1AEvaluator, protocol: str) -> Path:
    output_config = _nested(evaluator.evaluation_config, "output")
    training_output = resolve_project_local_path(
        evaluator.training_config.get(
            "output_dir",
            f"outputs/stage1a_{evaluator.domain}_sit_b2",
        ),
        evaluator.project_root,
        field_name="output_dir",
    )
    subdir = str(output_config.get("subdir", "stage1a_eval"))
    return (
        training_output
        / subdir
        / f"step_{evaluator.checkpoint_step:06d}"
        / evaluator.weights
        / protocol
    )


def _smoke_output_dir(evaluator: LoadedStage1AEvaluator) -> Path:
    return _evaluation_output_dir(evaluator, "smoke")


def _roundtrip_output_dir(evaluator: LoadedStage1AEvaluator) -> Path:
    return _evaluation_output_dir(evaluator, "roundtrip")


def _image_and_latent_metrics(
    latent_outputs: dict[str, torch.Tensor],
    decoded_outputs: dict[str, torch.Tensor],
    target_latents: torch.Tensor,
    target_images: torch.Tensor,
) -> dict[str, dict[str, float]]:
    metrics = {}
    for name, latent in latent_outputs.items():
        latent_mse = float(torch.mean((latent.float() - target_latents.float()) ** 2).cpu())
        pixel_values = _mse_and_psnr(decoded_outputs[name], target_images)
        metrics[name] = {
            "latent_mse": latent_mse,
            "pixel_mse": pixel_values["mse"],
            "pixel_psnr": pixel_values["psnr"],
        }
    return metrics


def _inferred_noise_stats(inferred_noise: torch.Tensor) -> dict[str, float]:
    value = inferred_noise.float()
    return {
        "mean": float(value.mean().cpu()),
        "std": float(value.std(unbiased=False).cpu()),
        "rms": float(torch.sqrt(torch.mean(value**2)).cpu()),
    }


@torch.inference_mode()
def _run_inferred_noise_roundtrip(
    evaluator: LoadedStage1AEvaluator,
    batch: dict[str, Any],
    x0: torch.Tensor,
    z: torch.Tensor,
    original_images: torch.Tensor,
    *,
    seed: int,
    guidance_scale: float,
    null_label: int | None,
) -> Stage1ARoundTripReport:
    sampling_config = _nested(evaluator.evaluation_config, "sampling")
    inferred_config = _nested(sampling_config, "inferred_noise")
    shared_num_steps = int(inferred_config.get("num_steps", sampling_config.get("num_steps", 100)))
    backward_num_steps = int(inferred_config.get("backward_num_steps", shared_num_steps))
    forward_num_steps = int(inferred_config.get("forward_num_steps", shared_num_steps))
    variants = list(inferred_config.get("variants") or ["correct_z"])

    inferred_noise = infer_starting_noise(
        evaluator.branch,
        evaluator.transformer,
        x0,
        z,
        num_steps=backward_num_steps,
        guidance_scale=guidance_scale,
        null_label=null_label,
    )
    latent_outputs = reconstruct_z_variants(
        evaluator.branch,
        evaluator.transformer,
        inferred_noise,
        z,
        variants,
        num_steps=forward_num_steps,
        guidance_scale=guidance_scale,
        null_label=null_label,
    )
    decoded_outputs = {
        name: decode_vae_latents(evaluator.vae, value)
        for name, value in latent_outputs.items()
    }
    metrics = _image_and_latent_metrics(
        latent_outputs,
        decoded_outputs,
        x0,
        original_images,
    )

    row_order = ["original", *latent_outputs.keys()]
    rows = [original_images, *[decoded_outputs[name] for name in latent_outputs]]
    output_dir = _roundtrip_output_dir(evaluator)
    grid_path = output_dir / "roundtrip_grid.png"
    _save_grid(grid_path, rows, samples_per_row=x0.shape[0])

    report = Stage1ARoundTripReport(
        protocol="inferred_noise_roundtrip",
        training_config_path=str(evaluator.training_config_path),
        evaluation_config_path=str(evaluator.evaluation_config_path),
        checkpoint_path=str(evaluator.checkpoint_path),
        output_dir=str(output_dir),
        domain=evaluator.domain,
        split=evaluator.split,
        checkpoint_step=evaluator.checkpoint_step,
        weights=evaluator.weights,
        seed=seed,
        num_samples=x0.shape[0],
        backward_num_steps=backward_num_steps,
        forward_num_steps=forward_num_steps,
        sample_ids=list(batch["sample_id"]),
        row_order=row_order,
        metrics=metrics,
        inferred_noise_stats=_inferred_noise_stats(inferred_noise),
        grid_path=str(grid_path),
    )
    _write_json(output_dir / "roundtrip_report.json", report.to_dict())
    return report


@torch.inference_mode()
def run_stage1a_smoke_test(
    training_config_path: str | Path,
    evaluation_config_path: str | Path,
    *,
    device: str | None = None,
    weights: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> Stage1ASmokeReport:
    """Run the first eight-image reconstruction gate and save a comparison grid."""
    evaluator = load_stage1a_evaluator(
        training_config_path,
        evaluation_config_path,
        device=device,
        weights=weights,
        checkpoint_path=checkpoint_path,
    )
    dataset_config = _nested(evaluator.evaluation_config, "dataset")
    sampling_config = _nested(evaluator.evaluation_config, "sampling")
    seed = int(evaluator.evaluation_config.get("seed", 20260902))
    num_samples = int(dataset_config.get("smoke_samples", 8))
    num_steps = int(sampling_config.get("smoke_num_steps", 50))
    guidance_scale = float(sampling_config.get("guidance_scale", 1.0))
    variants = list(
        sampling_config.get("variants")
        or ["correct_z", "shuffled_z", "zero_z"]
    )
    null_label_value = _nested(
        evaluator.training_config,
        "class_conditioning",
    ).get("null_label")
    null_label = int(null_label_value) if null_label_value is not None else None

    batch = _collate(_deterministic_subset(evaluator.dataset, num_samples, seed))
    x0 = batch["x0_latent"].to(
        device=evaluator.device,
        dtype=evaluator.model_dtype,
    )
    z = evaluator.branch.encode(x0)
    starting_noise = _noise_like(x0, seed + 1)

    latent_outputs = reconstruct_z_variants(
        evaluator.branch,
        evaluator.transformer,
        starting_noise,
        z,
        variants,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        null_label=null_label,
    )

    original_images = decode_vae_latents(evaluator.vae, x0)
    decoded_outputs = {
        name: decode_vae_latents(evaluator.vae, value)
        for name, value in latent_outputs.items()
    }
    metrics = _image_and_latent_metrics(
        latent_outputs,
        decoded_outputs,
        x0,
        original_images,
    )

    row_order = ["original", *latent_outputs.keys()]
    rows = [original_images, *[decoded_outputs[name] for name in latent_outputs]]
    output_dir = _smoke_output_dir(evaluator)
    grid_path = output_dir / "reconstruction_grid.png"
    _save_grid(grid_path, rows, samples_per_row=num_samples)

    extra_reports: dict[str, str] = {}
    inferred_config = _nested(sampling_config, "inferred_noise")
    if bool(inferred_config.get("enabled", False)):
        roundtrip_report = _run_inferred_noise_roundtrip(
            evaluator,
            batch,
            x0,
            z,
            original_images,
            seed=seed,
            guidance_scale=guidance_scale,
            null_label=null_label,
        )
        extra_reports["inferred_noise_roundtrip"] = str(
            Path(roundtrip_report.output_dir) / "roundtrip_report.json"
        )

    report = Stage1ASmokeReport(
        protocol="fixed_noise_smoke",
        training_config_path=str(evaluator.training_config_path),
        evaluation_config_path=str(evaluator.evaluation_config_path),
        checkpoint_path=str(evaluator.checkpoint_path),
        output_dir=str(output_dir),
        domain=evaluator.domain,
        split=evaluator.split,
        checkpoint_step=evaluator.checkpoint_step,
        weights=evaluator.weights,
        seed=seed,
        num_samples=num_samples,
        num_steps=num_steps,
        sample_ids=list(batch["sample_id"]),
        row_order=row_order,
        metrics=metrics,
        grid_path=str(grid_path),
        extra_reports=extra_reports,
    )
    _write_json(output_dir / "smoke_report.json", report.to_dict())
    return report
