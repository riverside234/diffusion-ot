"""Build the existing CNN PDAE model, optimizer, and EMA."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diffusion_ot.data.pdae_v2_augmentation import MatchedRGBAugmentation, augmentation_options
from diffusion_ot.integrations.hf_snapshot import load_yaml_config, resolve_project_local_path
from diffusion_ot.integrations.sit_diffusers import load_sit_components, validate_transformer_config
from diffusion_ot.models.pdae_sit import PDAESiTBranch, build_pdae_sit_branch
from diffusion_ot.training.train_pdae_domain import TrainableEMA


@dataclass
class TrainingState:
    branch: PDAESiTBranch
    optimizer: torch.optim.Optimizer
    ema: TrainableEMA | None
    device: torch.device
    dtype: torch.dtype
    vae: torch.nn.Module | None = None
    augmentation: MatchedRGBAugmentation | None = None


def build_training_state(config):
    """Initialize a fresh run from the resolved configuration."""
    train = config.get("train") or {}
    ema_config = config.get("ema") or {}
    augmentation_config = augmentation_options(config)
    torch.manual_seed(int(train.get("seed", 20260901)))
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = config["project_root"]
    model_config = load_yaml_config(config["model_config"])
    augmentation = (
        MatchedRGBAugmentation(augmentation_config, use_posterior_mean=bool(
            (model_config.get("latent_cache") or {}).get("use_posterior_mean", True)))
        if augmentation_config is not None else None
    )
    pretrained = resolve_project_local_path(
        config.get("pretrained_config") or model_config["pretrained"],
        root, field_name="pretrained_config",
    )
    components = load_sit_components(
        pretrained, project_root=root, device=str(device),
        torch_dtype=config.get("torch_dtype", "float32"),
    )
    transformer = components.transformer.eval()
    mismatches = validate_transformer_config(
        transformer, model_config.get("expected_transformer_config") or {},
    )
    if mismatches:
        raise ValueError("Unexpected SiT transformer config:\n" + "\n".join(mismatches))

    dtype = next(transformer.parameters()).dtype
    branch = build_pdae_sit_branch(
        transformer, model_config=model_config, stage_config=config, project_root=root,
    ).to(device=device, dtype=dtype).train()
    optimizer = build_optimizer(branch, train)
    ema = (
        TrainableEMA(branch, decay=float(ema_config.get("decay", 0.9999)),
                     warmup_steps=int(ema_config.get("warmup_steps", 1000)))
        if ema_config.get("enabled", True) else None
    )
    vae = components.vae.eval().requires_grad_(False)
    return TrainingState(branch, optimizer, ema, device, dtype, vae, augmentation)


def build_optimizer(branch, train):
    """Group trainable parameters using the existing trainer's learning rates."""
    groups = {name: [] for name in ("encoder", "adapter", "lora")}
    for name, parameter in branch.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            group = "encoder"
        elif ".lora_down." in name or ".lora_up." in name:
            group = "lora"
        else:
            group = "adapter"
        groups[group].append(parameter)

    base_lr = train.get("lr", 1e-4)
    defaults = {"encoder": base_lr, "adapter": base_lr, "lora": 5e-5}
    return torch.optim.AdamW(
        [{"group_name": name, "params": parameters,
          "lr": float(train.get(f"lr_{name}", defaults[name]))}
         for name, parameters in groups.items() if parameters],
        weight_decay=float(train.get("weight_decay", 0.0)),
        betas=tuple(float(value) for value in train.get("betas", (0.9, 0.999))),
    )
