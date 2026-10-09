from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from diffusion_ot.training.native_flow import native_flow_objective, validate_stage1a_objective

from diffusion_ot.integrations.hf_snapshot import (
    effective_project_root,
    find_project_root,
    load_yaml_config,
    resolve_project_local_path,
)


_ALLOWED_Z_VARIANTS = {"correct_z", "shuffled_z", "null_z", "zero_z"}


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
    semantic_cfg_enabled: bool
    attention_lora: dict[str, Any]
    seed: int
    num_samples: int
    num_steps: int
    guidance_scales: list[float]
    sample_ids: list[str | None]
    row_order: list[str]
    metrics: dict[str, dict[str, float]]
    grid_path: str
    extra_reports: dict[str, str]
    encoder_input_space: str = "latent"
    image_reference: str = "vae_reconstruction"
    encoder_architecture: dict[str, Any] | None = None
    native_flow_objective: dict[str, Any] | None = None
    checkpoint_ema: dict[str, Any] | None = None
    roundtrip_enabled: bool = False
    sampling: dict[str, Any] = field(default_factory=dict)
    per_image_metrics: list[dict[str, Any]] = field(default_factory=list)

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
    semantic_cfg_enabled: bool
    attention_lora: dict[str, Any]
    seed: int
    num_samples: int
    backward_num_steps: int
    forward_num_steps: int
    guidance_scale: float
    sample_ids: list[str | None]
    row_order: list[str]
    metrics: dict[str, dict[str, float]]
    inferred_noise_stats: dict[str, float]
    grid_path: str
    encoder_input_space: str = "latent"
    image_reference: str = "vae_reconstruction"
    encoder_architecture: dict[str, Any] | None = None
    native_flow_objective: dict[str, Any] | None = None
    checkpoint_ema: dict[str, Any] | None = None
    sampling: dict[str, Any] = field(default_factory=dict)

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
    checkpoint_ema: dict[str, Any] | None = None


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


def _attention_lora_metadata(branch: Any) -> dict[str, Any]:
    wrapper = getattr(branch, "semantic_transformer", None)
    enabled = bool(getattr(wrapper, "attention_lora_enabled", False))
    metadata: dict[str, Any] = {"enabled": enabled}
    if enabled:
        metadata.update(
            {
                "rank": int(wrapper.lora_rank),
                "alpha": float(wrapper.lora_alpha),
                "dropout": float(wrapper.lora_dropout),
                "layers": list(wrapper.lora_layers),
                "targets": list(wrapper.lora_targets),
            }
        )
    return metadata


def stage1a_architecture_metadata(branch: Any) -> dict[str, Any]:
    metadata = {
        "semantic_cfg_enabled": bool(getattr(branch, "semantic_cfg_enabled", False)),
        "attention_lora": _attention_lora_metadata(branch),
    }
    # Stage 1B compares this dictionary with saved checkpoint provenance exactly.
    # Preserve the legacy latent shape; absence of input-space metadata means latent.
    if getattr(branch, "encoder_input_space", "latent") == "rgb":
        metadata["encoder_input_space"] = "rgb"
        metadata["encoder_image_size"] = getattr(branch, "encoder_image_size", None)
    encoder_architecture = getattr(branch, "encoder_architecture", {"kind": "plain_cnn_v1"})
    if encoder_architecture["kind"] != "plain_cnn_v1":
        metadata["encoder_architecture"] = encoder_architecture
        if hasattr(branch.encoder, "spatial_feature_spec"):
            metadata["spatial_features"] = branch.encoder.spatial_feature_spec
    return metadata


def validate_stage1a_architecture(
    branch: Any,
    stage1a_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the Stage 1A architecture required by an alignment protocol."""
    metadata = stage1a_architecture_metadata(branch)
    expected_cfg = stage1a_config.get("require_semantic_cfg")
    expected_lora = stage1a_config.get("require_attention_lora")
    expected_lora_rank = stage1a_config.get("require_attention_lora_rank")
    expected_lora_alpha = stage1a_config.get("require_attention_lora_alpha")
    mismatches = []
    expected_encoder = stage1a_config.get("require_encoder_kind")
    actual_encoder = metadata.get("encoder_architecture", {"kind": "plain_cnn_v1"})["kind"]
    if expected_encoder is not None and expected_encoder != actual_encoder:
        mismatches.append(f"encoder kind is {actual_encoder}, expected {expected_encoder}")
    expected_input = stage1a_config.get("require_encoder_input_space")
    actual_input = getattr(branch, "encoder_input_space", "latent")
    if expected_input is not None and expected_input != actual_input:
        mismatches.append(f"encoder input is {actual_input}, expected {expected_input}")
    if expected_cfg is not None and metadata["semantic_cfg_enabled"] != bool(expected_cfg):
        mismatches.append(
            "semantic CFG "
            f"is {metadata['semantic_cfg_enabled']}, expected {bool(expected_cfg)}"
        )
    lora_enabled = bool(metadata["attention_lora"]["enabled"])
    if expected_lora is not None and lora_enabled != bool(expected_lora):
        mismatches.append(
            f"attention LoRA is {lora_enabled}, expected {bool(expected_lora)}"
        )
    actual_lora_rank = metadata["attention_lora"].get("rank")
    if (
        expected_lora_rank is not None
        and actual_lora_rank != int(expected_lora_rank)
    ):
        mismatches.append(
            f"attention LoRA rank is {actual_lora_rank}, expected {int(expected_lora_rank)}"
        )
    actual_lora_alpha = metadata["attention_lora"].get("alpha")
    if (
        expected_lora_alpha is not None
        and actual_lora_alpha != float(expected_lora_alpha)
    ):
        mismatches.append(
            "attention LoRA alpha is "
            f"{actual_lora_alpha}, expected {float(expected_lora_alpha)}"
        )
    if mismatches:
        raise ValueError(
            "Stage 1A architecture does not match the alignment protocol: "
            + "; ".join(mismatches)
        )
    return metadata


def _configured_guidance_scales(sampling: dict[str, Any]) -> list[float]:
    raw_scales = sampling.get("guidance_scales")
    if raw_scales is None:
        raw_scales = [sampling.get("guidance_scale", 1.0)]
    elif isinstance(raw_scales, (int, float)):
        raw_scales = [raw_scales]
    if not isinstance(raw_scales, (list, tuple)) or not raw_scales:
        raise ValueError("sampling.guidance_scales must be a non-empty list.")
    scales = [float(value) for value in raw_scales]
    if any(not math.isfinite(value) or value < 0.0 for value in scales):
        raise ValueError("sampling.guidance_scales must contain finite, non-negative values.")
    if len(set(scales)) != len(scales):
        raise ValueError("sampling.guidance_scales must not contain duplicates.")
    return scales


def _validate_eval_config(config: dict[str, Any]) -> None:
    _evaluation_image_reference(None, config)
    if not isinstance(_nested(config, "output").get("show_vae_reconstruction", True), bool):
        raise ValueError("output.show_vae_reconstruction must be a boolean.")
    sampling = _nested(config, "sampling")
    _configured_guidance_scales(sampling)
    solver = str(sampling.get("solver", "euler"))
    if solver not in {"euler", "heun"}:
        raise ValueError("sampling.solver must be euler or heun.")
    if str(sampling.get("direction", "noise_to_data")) != "noise_to_data":
        raise ValueError("Stage 1A evaluation must match the trained noise_to_data field.")
    expected_time = "midpoint" if solver == "euler" else "endpoints"
    if str(sampling.get("time_evaluation", expected_time)) != expected_time:
        raise ValueError(f"sampling.time_evaluation must be {expected_time} for {solver}.")
    if not bool(sampling.get("fixed_starting_noise", True)):
        raise ValueError("Fixed starting noise is required for fair z-variant comparisons.")

    variants = list(sampling.get("variants") or [])
    if not variants or set(variants) - _ALLOWED_Z_VARIANTS:
        raise ValueError(
            "sampling.variants must be a non-empty subset of "
            "[correct_z, shuffled_z, null_z, zero_z]."
        )
    inferred_config = _nested(sampling, "inferred_noise")
    if bool(inferred_config.get("enabled", False)):
        inferred_guidance_scale = float(inferred_config.get("guidance_scale", 1.0))
        if not math.isfinite(inferred_guidance_scale) or inferred_guidance_scale < 0.0:
            raise ValueError(
                "sampling.inferred_noise.guidance_scale must be finite and non-negative."
            )
        inferred_variants = list(inferred_config.get("variants") or ["correct_z"])
        if not inferred_variants or set(inferred_variants) - _ALLOWED_Z_VARIANTS:
            raise ValueError(
                "sampling.inferred_noise.variants must be a non-empty subset of "
                "[correct_z, shuffled_z, null_z, zero_z]."
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
    # E/G checkpoints omit the frozen backbone. Match the trainer's override
    # precedence so evaluation cannot silently load a different SiT/VAE pair.
    pretrained_config_path = (
        _resolve_config_path(train_config, root, "pretrained_config")
        if train_config.get("pretrained_config")
        else _resolve_config_path(model_config, root, "pretrained")
    )

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
    validate_stage1a_objective(train_config, checkpoint)
    checkpoint_domain = str(checkpoint.get("domain", domain)).lower()
    if checkpoint_domain != domain:
        raise ValueError(
            f"Checkpoint domain is {checkpoint_domain}, but training config domain is {domain}."
        )

    encoder_config = _nested(train_config, "encoder")
    if encoder_config.get("kind") == "siglip2_vit_b16":
        from diffusion_ot.models.pdae_v2.encoder import MANIFEST_NAME, restore_snapshot_manifest

        directory = resolve_project_local_path(encoder_config["local_dir"], root, field_name="encoder.local_dir")
        if not (directory / MANIFEST_NAME).is_file():
            try:
                restore_snapshot_manifest(directory, checkpoint["model"].get("frozen_encoder"))
            except (ValueError, FileNotFoundError) as exc:
                raise ValueError(
                    f"Cannot restore SigLIP metadata at {directory}: {exc}\n"
                    "If SigLIP is stored elsewhere, correct encoder.local_dir in the training YAML. "
                    "Otherwise recover the checkpoint's exact encoder with:\n"
                    f'python scripts/download_siglip2.py --checkpoint "{resolved_checkpoint}" '
                    f'--output-dir "{directory}"'
                ) from exc
            print(json.dumps({"event": "siglip_manifest_restored_from_checkpoint",
                              "manifest": str(directory / MANIFEST_NAME),
                              "checkpoint": str(resolved_checkpoint)}))

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
        project_root=root,
    )
    validate_stage1a_architecture(branch, _nested(eval_config, "architecture"))
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
        include_original_images=(
            getattr(branch, "encoder_input_space", "latent") == "rgb"
            or _evaluation_image_reference(branch, eval_config) == "original_rgb"
        ),
    )
    # Describe the saved average, not the current YAML's future EMA schedule.
    saved_ema = checkpoint.get("ema")
    checkpoint_ema = None
    if saved_ema is not None:
        checkpoint_ema = {key: saved_ema.get(key) for key in ("decay", "warmup_steps", "num_updates")}
        checkpoint_ema["last_reset"] = (checkpoint.get("train_state") or {}).get("ema_reset")
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
        checkpoint_ema=checkpoint_ema,
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
    solver: str = "euler",
    time_eps: float = 1.0e-5,
    batch_size: int | None = None,
    sampling_stats: dict | None = None,
) -> torch.Tensor:
    """Integrate velocity: legacy Euler at midpoint times, or endpoint Heun.

    Heun clamps network times to the training interval while integrating the
    full requested interval. Chunking never changes conditions or random noise.
    """
    from diffusion_ot.models.pdae_sit import make_null_class_labels

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance_scale must be finite.")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be euler or heun.")
    if not 0 <= time_eps < .5:
        raise ValueError("time_eps must be in [0,0.5).")
    if not all(math.isfinite(t) and 0 <= t <= 1 for t in (start_time, end_time)):
        raise ValueError("Integration times must be finite and in [0,1].")
    if z.shape[0] != initial_state.shape[0]:
        raise ValueError("Conditions and states must have matching batch dimensions.")
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(initial_state) > batch_size:
            return torch.cat([integrate_pdae_flow(
                branch, transformer, initial_state[i:i + batch_size], z[i:i + batch_size],
                num_steps=num_steps, start_time=start_time, end_time=end_time,
                guidance_scale=guidance_scale, null_label=null_label, solver=solver,
                time_eps=time_eps, sampling_stats=sampling_stats)
                for i in range(0, len(initial_state), batch_size)])

    state = initial_state.clone()
    dt = (float(end_time) - float(start_time)) / int(num_steps)
    class_labels = make_null_class_labels(
        transformer,
        batch_size=state.shape[0],
        device=state.device,
        null_label=null_label,
    )
    def velocity_at(value, time_value):
        if solver == "heun":
            time_value = min(1 - time_eps, max(time_eps, time_value))
        timestep = torch.full(
            (state.shape[0],),
            time_value,
            device=state.device,
            dtype=torch.float32,
        )
        if bool(getattr(branch, "semantic_cfg_enabled", False)):
            output = branch.predict_cfg_with_z(
                x_t=value,
                timestep=timestep,
                z=z,
                guidance_scale=guidance_scale,
                class_labels=class_labels,
            )
            velocity = output.sample
        else:
            # Compatibility path for Stage 1A checkpoints trained before learned-null CFG.
            output = branch.predict_with_z(
                x_t=value,
                timestep=timestep,
                z=z,
                class_labels=class_labels,
            )
            velocity = output.base_sample + float(guidance_scale) * output.delta_sample
        if sampling_stats is not None:
            semantic = bool(getattr(branch, "semantic_cfg_enabled", False))
            counts = {"velocity_batch_calls": 1, "sample_velocity_evaluations": len(value),
                      "conditional_prediction_batch_calls": int(not semantic or guidance_scale != 0),
                      "null_prediction_batch_calls": int(semantic and guidance_scale != 1)}
            for key, count in counts.items():
                sampling_stats[key] = sampling_stats.get(key, 0) + count
        return velocity

    for index in range(int(num_steps)):
        t = float(start_time) + index * dt
        if solver == "euler":
            state = state + dt * velocity_at(state, float(start_time) + (index + .5) * dt)
        else:
            first = velocity_at(state, t)
            second = velocity_at(state + dt * first, t + dt)
            state = state + .5 * dt * (first + second)
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
    **sampling_kwargs,
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
        **sampling_kwargs,
    )


@torch.inference_mode()
def decode_vae_latents(vae: Any, latents: torch.Tensor, *, batch_size: int | None = None) -> torch.Tensor:
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(latents) > batch_size:
            return torch.cat([decode_vae_latents(vae, chunk) for chunk in latents.split(batch_size)])
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


def _evaluation_image_reference(branch: Any, config: dict[str, Any]) -> str:
    reference = _nested(config, "metrics").get("image_reference")
    if reference is None:
        return "original_rgb" if getattr(branch, "encoder_input_space", "latent") == "rgb" else "vae_reconstruction"
    if not isinstance(reference, str) or reference not in {"original_rgb", "vae_reconstruction"}:
        raise ValueError("metrics.image_reference must be original_rgb or vae_reconstruction.")
    return reference


def _evaluation_encoder_image(
    evaluator: LoadedStage1AEvaluator,
    batch: dict[str, Any],
) -> torch.Tensor | None:
    """Load original RGB for conditioning or metrics, without a decoded fallback."""
    if (getattr(evaluator.branch, "encoder_input_space", "latent") != "rgb"
            and _evaluation_image_reference(evaluator.branch, evaluator.evaluation_config) != "original_rgb"):
        return None
    image = batch.get("encoder_image")
    if not isinstance(image, torch.Tensor):
        raise ValueError("Stage 1A original-RGB evaluation requires original encoder_image tensors.")
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("encoder_image must have shape [B, 3, H, W].")
    if image.shape[0] != batch["x0_latent"].shape[0]:
        raise ValueError("encoder_image and x0_latent must contain the same number of samples.")
    return image.to(device=evaluator.device, dtype=torch.float32)


def _noise_like(value: torch.Tensor, seed: int, *, block_size: int | None = None) -> torch.Tensor:
    generator = torch.Generator(device=value.device)
    generator.manual_seed(int(seed))
    if block_size is not None and block_size <= 0:
        raise ValueError("noise_batch_size must be positive.")
    # Fixed-size RNG draws retain the old eight-image CUDA noise when expanding
    # the cohort; CUDA randn can change its prefix when the draw shape changes.
    chunks = value.split(block_size) if block_size is not None else [value]
    return torch.cat([torch.randn(chunk.shape, generator=generator,
                                 device=value.device, dtype=value.dtype) for chunk in chunks])


def _variant_z(branch: Any, z: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "correct_z":
        return z
    if variant == "null_z":
        if not bool(getattr(branch, "semantic_cfg_enabled", False)):
            raise ValueError("null_z evaluation requires a learned-null semantic-CFG checkpoint.")
        return branch.semantic_null_like(z)
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
    **sampling_kwargs,
) -> dict[str, torch.Tensor]:
    """Reconstruct z variants from one immutable starting-noise batch."""
    outputs: dict[str, torch.Tensor] = {}
    for variant in variants:
        outputs[variant] = integrate_pdae_flow(
            branch,
            transformer,
            starting_noise,
            _variant_z(branch, z, variant),
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            null_label=null_label,
            **sampling_kwargs,
        )
    return outputs


def _cfg_output_key(variant: str, guidance_scale: float) -> str:
    return f"{variant}_cfg_{float(guidance_scale):g}"


@torch.inference_mode()
def reconstruct_cfg_sweep(
    branch: Any,
    transformer: Any,
    starting_noise: torch.Tensor,
    z: torch.Tensor,
    variants: list[str],
    guidance_scales: list[float],
    *,
    num_steps: int,
    null_label: int | None = None,
    **sampling_kwargs,
) -> dict[str, torch.Tensor]:
    """Evaluate every z variant and semantic-CFG scale from identical noise."""
    outputs: dict[str, torch.Tensor] = {}
    for guidance_scale in guidance_scales:
        scale_zero_output = None
        for variant in variants:
            variant_z = _variant_z(branch, z, variant)
            key = _cfg_output_key(variant, guidance_scale)
            if guidance_scale == 0.0 and scale_zero_output is not None:
                outputs[key] = scale_zero_output.clone()
                continue
            outputs[key] = integrate_pdae_flow(
                branch,
                transformer,
                starting_noise,
                variant_z,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                null_label=null_label,
                **sampling_kwargs,
            )
            if guidance_scale == 0.0:
                scale_zero_output = outputs[key]
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


def _sampling_kwargs(evaluator, batch_size: int, stats: dict) -> dict:
    sampling = _nested(evaluator.evaluation_config, "sampling")
    return {"solver": str(sampling.get("solver", "euler")), "batch_size": batch_size,
            "time_eps": float(_nested(evaluator.training_config, "flow").get("time_eps", 1e-5)),
            "sampling_stats": stats}


def _synchronize(value: torch.Tensor) -> None:
    if value.device.type == "cuda":
        torch.cuda.synchronize(value.device)


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
    vae_reconstruction: torch.Tensor | None = None,
) -> Stage1ARoundTripReport:
    sampling_config = _nested(evaluator.evaluation_config, "sampling")
    inferred_config = _nested(sampling_config, "inferred_noise")
    shared_num_steps = int(inferred_config.get("num_steps", sampling_config.get("num_steps", 100)))
    backward_num_steps = int(inferred_config.get("backward_num_steps", shared_num_steps))
    forward_num_steps = int(inferred_config.get("forward_num_steps", shared_num_steps))
    variants = list(inferred_config.get("variants") or ["correct_z"])
    batch_size = int(_nested(evaluator.evaluation_config, "dataset").get("batch_size", len(x0)))
    stats = {}
    sampling_kwargs = _sampling_kwargs(evaluator, batch_size, stats)
    _synchronize(x0)
    started = time.perf_counter()

    inferred_noise = infer_starting_noise(
        evaluator.branch,
        evaluator.transformer,
        x0,
        z,
        num_steps=backward_num_steps,
        **sampling_kwargs,
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
        **sampling_kwargs,
        guidance_scale=guidance_scale,
        null_label=null_label,
    )
    decoded_outputs = {
        name: decode_vae_latents(evaluator.vae, value, batch_size=batch_size)
        for name, value in latent_outputs.items()
    }
    metrics = _image_and_latent_metrics(
        latent_outputs,
        decoded_outputs,
        x0,
        original_images,
    )
    _synchronize(x0)
    stats.update(solver=sampling_kwargs["solver"], time_eps=sampling_kwargs["time_eps"],
                 seconds_including_decode=time.perf_counter() - started)

    row_order = ["original", *latent_outputs.keys()]
    rows = [original_images, *[decoded_outputs[name] for name in latent_outputs]]
    if vae_reconstruction is not None:
        if _nested(evaluator.evaluation_config, "output").get("show_vae_reconstruction", True):
            row_order.insert(1, "vae_reconstruction")
            rows.insert(1, vae_reconstruction)
        vae_metrics = _mse_and_psnr(vae_reconstruction, original_images)
        metrics["vae_reconstruction"] = {
            "pixel_mse": vae_metrics["mse"], "pixel_psnr": vae_metrics["psnr"],
        }
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
        semantic_cfg_enabled=bool(evaluator.branch.semantic_cfg_enabled),
        attention_lora=_attention_lora_metadata(evaluator.branch),
        seed=seed,
        num_samples=x0.shape[0],
        backward_num_steps=backward_num_steps,
        forward_num_steps=forward_num_steps,
        guidance_scale=guidance_scale,
        sample_ids=list(batch["sample_id"]),
        row_order=row_order,
        metrics=metrics,
        inferred_noise_stats=_inferred_noise_stats(inferred_noise),
        grid_path=str(grid_path),
        encoder_input_space=str(getattr(evaluator.branch, "encoder_input_space", "latent")),
        image_reference="original_rgb" if vae_reconstruction is not None else "vae_reconstruction",
        encoder_architecture=getattr(evaluator.branch, "encoder_architecture", None),
        native_flow_objective=native_flow_objective(evaluator.training_config),
        checkpoint_ema=evaluator.checkpoint_ema,
        sampling=stats,
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
    roundtrip: bool | None = None,
    solver: str | None = None,
    num_steps: int | None = None,
    noise_seed: int | None = None,
    output_subdir: str | None = None,
) -> Stage1ASmokeReport:
    """Generate fixed-noise images; roundtrip overrides the opt-in YAML setting."""
    evaluator = load_stage1a_evaluator(
        training_config_path,
        evaluation_config_path,
        device=device,
        weights=weights,
        checkpoint_path=checkpoint_path,
    )
    dataset_config = _nested(evaluator.evaluation_config, "dataset")
    sampling_config = _nested(evaluator.evaluation_config, "sampling")
    if solver is not None:
        sampling_config.update(solver=solver, time_evaluation="midpoint" if solver == "euler" else "endpoints")
    if num_steps is not None:
        sampling_config["smoke_num_steps"] = num_steps
    if noise_seed is not None:
        sampling_config["noise_seed"] = noise_seed
    if output_subdir is not None:
        evaluator.evaluation_config.setdefault("output", {})["subdir"] = output_subdir
    elif any(value is not None for value in (solver, num_steps, noise_seed)):
        output = evaluator.evaluation_config.setdefault("output", {})
        output["subdir"] = (str(output.get("subdir", "stage1a_eval"))
                            + f"_{sampling_config.get('solver', 'euler')}{sampling_config.get('smoke_num_steps', 50)}"
                            + f"_noise{sampling_config.get('noise_seed', int(evaluator.evaluation_config.get('seed', 20260902)) + 1)}")
    _validate_eval_config(evaluator.evaluation_config)
    seed = int(evaluator.evaluation_config.get("seed", 20260902))
    num_samples = int(dataset_config.get("smoke_samples", 8))
    batch_size = int(dataset_config.get("batch_size", num_samples))
    if batch_size <= 0:
        raise ValueError("dataset.batch_size must be positive.")
    num_steps = int(sampling_config.get("smoke_num_steps", 50))
    guidance_scales = _configured_guidance_scales(sampling_config)
    variants = list(
        sampling_config.get("variants")
        or ["correct_z", "shuffled_z"]
    )
    null_label_value = _nested(
        evaluator.training_config,
        "class_conditioning",
    ).get("null_label")
    null_label = int(null_label_value) if null_label_value is not None else None

    batch = _collate(_deterministic_subset(evaluator.dataset, num_samples, seed))
    num_samples = len(batch["sample_id"])
    x0 = batch["x0_latent"].to(
        device=evaluator.device,
        dtype=evaluator.model_dtype,
    )
    encoder_image = _evaluation_encoder_image(evaluator, batch)
    image_reference = _evaluation_image_reference(evaluator.branch, evaluator.evaluation_config)
    uses_rgb_encoder = getattr(evaluator.branch, "encoder_input_space", "latent") == "rgb"
    z = torch.cat([evaluator.branch.encode(
        x0[i:i + batch_size], **({"encoder_image": encoder_image[i:i + batch_size].to(dtype=evaluator.model_dtype)}
                                 if uses_rgb_encoder else {}))
        for i in range(0, len(x0), batch_size)])
    noise_seed = int(sampling_config.get("noise_seed", seed + 1))
    noise_batch_size = sampling_config.get("noise_batch_size")
    starting_noise = _noise_like(x0, noise_seed, block_size=noise_batch_size)
    stats = {}
    sampling_kwargs = _sampling_kwargs(evaluator, batch_size, stats)
    _synchronize(x0)
    started = time.perf_counter()

    latent_outputs = reconstruct_cfg_sweep(
        evaluator.branch,
        evaluator.transformer,
        starting_noise,
        z,
        variants,
        guidance_scales,
        num_steps=num_steps,
        null_label=null_label,
        **sampling_kwargs,
    )
    _synchronize(x0)
    stats.update(solver=sampling_kwargs["solver"], time_eps=sampling_kwargs["time_eps"],
                 time_evaluation="midpoint" if sampling_kwargs["solver"] == "euler" else "endpoints",
                 generation_seconds=time.perf_counter() - started, batch_size=batch_size,
                 noise_seed=noise_seed, noise_batch_size=noise_batch_size,
                 velocity_evaluations_per_sample_per_output=num_steps * (2 if sampling_kwargs["solver"] == "heun" else 1),
                 counting="Prediction calls include the branch's base/residual computation; CFG 1 uses one conditioned prediction, CFG >1 uses conditioned and null predictions.")

    vae_images = decode_vae_latents(evaluator.vae, x0, batch_size=batch_size)
    original_images = (
        ((encoder_image + 1.0) / 2.0).clamp(0.0, 1.0)
        if image_reference == "original_rgb" else vae_images
    )
    decoded_outputs = {
        name: decode_vae_latents(evaluator.vae, value, batch_size=batch_size)
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
    if image_reference == "original_rgb":
        if _nested(evaluator.evaluation_config, "output").get("show_vae_reconstruction", True):
            row_order.insert(1, "vae_reconstruction")
            rows.insert(1, vae_images)
        vae_metrics = _mse_and_psnr(vae_images, original_images)
        metrics["vae_reconstruction"] = {
            "pixel_mse": vae_metrics["mse"], "pixel_psnr": vae_metrics["psnr"],
        }
    output_dir = _smoke_output_dir(evaluator)
    grid_path = output_dir / "reconstruction_grid.png"
    _save_grid(grid_path, rows, samples_per_row=num_samples)

    extra_reports: dict[str, str] = {}
    if z.ndim == 3:
        from diffusion_ot.models.pdae_v2.encoder import patch_token_statistics

        token_report = output_dir / "condition_tokens.json"
        token_report.write_text(json.dumps(patch_token_statistics(z.detach().cpu()), indent=2) + "\n", encoding="utf-8")
        extra_reports["condition_tokens"] = str(token_report)
    from diffusion_ot.evaluation.input_statistics import probe_options, run_input_statistics_probe
    input_options = probe_options(evaluator.evaluation_config)
    if input_options is not None:
        if z.ndim != 2:
            raise ValueError("input_statistics requires vector codes; disable it for PDAE v2 patch tokens.")
        extra_reports["input_statistics"] = run_input_statistics_probe(
            evaluator.branch, data_config_path=_resolve_config_path(evaluator.training_config, evaluator.project_root, "data_config"),
            domain=evaluator.domain, project_root=evaluator.project_root, device=evaluator.device,
            dtype=evaluator.model_dtype, options=input_options, output_dir=output_dir / "input_statistics",
            provenance=dict(stage="stage1a", checkpoint=str(evaluator.checkpoint_path),
                            checkpoint_step=evaluator.checkpoint_step, weights=evaluator.weights,
                            training_config_path=str(evaluator.training_config_path),
                            native_flow_objective=native_flow_objective(evaluator.training_config)))
    inferred_config = _nested(sampling_config, "inferred_noise")
    roundtrip_enabled = bool(inferred_config.get("enabled", False)) if roundtrip is None else roundtrip
    if roundtrip_enabled:
        roundtrip_guidance_scale = float(inferred_config.get("guidance_scale", 1.0))
        roundtrip_report = _run_inferred_noise_roundtrip(
            evaluator,
            batch,
            x0,
            z,
            original_images,
            seed=seed,
            guidance_scale=roundtrip_guidance_scale,
            null_label=null_label,
            vae_reconstruction=vae_images if image_reference == "original_rgb" else None,
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
        semantic_cfg_enabled=bool(evaluator.branch.semantic_cfg_enabled),
        attention_lora=_attention_lora_metadata(evaluator.branch),
        seed=seed,
        num_samples=num_samples,
        num_steps=num_steps,
        guidance_scales=guidance_scales,
        sample_ids=list(batch["sample_id"]),
        row_order=row_order,
        metrics=metrics,
        grid_path=str(grid_path),
        extra_reports=extra_reports,
        encoder_input_space=str(getattr(evaluator.branch, "encoder_input_space", "latent")),
        image_reference=image_reference,
        encoder_architecture=getattr(evaluator.branch, "encoder_architecture", None),
        native_flow_objective=native_flow_objective(evaluator.training_config),
        checkpoint_ema=evaluator.checkpoint_ema,
        roundtrip_enabled=roundtrip_enabled,
        sampling=stats,
        per_image_metrics=[{
            "sample_id": sample_id,
            "metrics": _image_and_latent_metrics(
                {key: value[i:i + 1] for key, value in latent_outputs.items()},
                {key: value[i:i + 1] for key, value in decoded_outputs.items()},
                x0[i:i + 1], original_images[i:i + 1]),
        } for i, sample_id in enumerate(batch["sample_id"])],
    )
    _write_json(output_dir / "smoke_report.json", report.to_dict())
    return report


def run_stage1a_weight_comparison(
    training_config_path: str | Path,
    evaluation_config_path: str | Path,
    *,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    roundtrip: bool | None = None,
    **sampling_overrides,
) -> dict[str, Any]:
    """Compare raw/EMA with identical samples, starting noise and CFG settings."""
    raw = run_stage1a_smoke_test(training_config_path, evaluation_config_path,
                                device=device, weights="raw", checkpoint_path=checkpoint_path, roundtrip=roundtrip,
                                **sampling_overrides)
    averaged = run_stage1a_smoke_test(training_config_path, evaluation_config_path,
                                     device=device, weights="ema", checkpoint_path=raw.checkpoint_path, roundtrip=roundtrip,
                                     **sampling_overrides)
    for field in ("checkpoint_path", "checkpoint_step", "domain", "split", "seed", "sample_ids",
                  "num_samples", "num_steps", "guidance_scales", "row_order", "image_reference",
                  "checkpoint_ema", "roundtrip_enabled"):
        if getattr(raw, field) != getattr(averaged, field):
            raise ValueError(f"Raw/EMA smoke comparison changed {field}; use a fixed step checkpoint.")
    for key in ("solver", "time_eps", "time_evaluation", "batch_size", "noise_seed", "noise_batch_size"):
        if raw.sampling.get(key) != averaged.sampling.get(key):
            raise ValueError(f"Raw/EMA smoke comparison changed sampling.{key}.")
    comparison_path = Path(averaged.output_dir).parent.parent / "raw_ema_comparison.json"
    result = {
        "protocol": "matched_raw_ema_smoke",
        "reports": {"raw": raw.to_dict(), "ema": averaged.to_dict()},
        "ema_minus_raw": {
            variant: {key: value - raw.metrics[variant][key] for key, value in values.items()}
            for variant, values in averaged.metrics.items()
        },
        "interpretation": "Positive EMA-minus-raw MSE or negative PSNR favors raw on reconstruction metrics; inspect both grids for realism and detail.",
        "comparison_path": str(comparison_path),
    }
    _write_json(comparison_path, result)
    return result
