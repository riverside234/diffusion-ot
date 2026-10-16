"""Save and restore training state in the existing Stage 1A checkpoint format."""

from pathlib import Path

import torch

from diffusion_ot.training.learning_rate import resolve_lr_schedule
from diffusion_ot.training.native_flow import native_flow_objective, validate_stage1a_objective


def save_checkpoint(path, state, config, *, step, loader):
    """Atomically save model, optimizer, EMA, step, and Torch/loader random states."""
    path = Path(path).expanduser()
    objective = native_flow_objective(config)
    payload = {
        "format_version": 3,
        "step": int(step),
        "domain": config["domain"],
        "model": state.branch.pdae_state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "ema": state.ema.state_dict() if state.ema is not None else None,
        "config": config,
        "native_flow_objective": objective,
        "train_state": {
            "lr_schedule": resolve_lr_schedule((config.get("train") or {}).get("lr_schedule")),
            "loss_weighting": objective["loss_weighting"],
        },
        "rng_state": torch.get_rng_state(),
        "dataloader_generator_state": loader.generator.get_state(),
    }
    if state.device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(state.device).cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path, state, config, *, loader):
    """Restore into a freshly built training state and return the completed step."""
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if checkpoint.get("domain") != config["domain"]:
        raise ValueError("Checkpoint domain does not match the training config.")
    validate_stage1a_objective(config, checkpoint)
    train = config.get("train") or {}
    saved_train = (checkpoint.get("config") or {}).get("train") or {}
    saved_schedule = (checkpoint.get("train_state") or {}).get("lr_schedule")
    if saved_schedule is None:
        saved_schedule = saved_train.get("lr_schedule")
    if resolve_lr_schedule(saved_schedule) != resolve_lr_schedule(train.get("lr_schedule")):
        raise ValueError("Learning-rate schedule changed; use the checkpoint's training recipe.")
    saved_ema = checkpoint.get("ema")
    if (saved_ema is None) != (state.ema is None):
        raise ValueError("EMA must be enabled or disabled consistently when resuming.")
    if saved_ema is not None:
        for key in ("decay", "warmup_steps"):
            if saved_ema.get(key, getattr(state.ema, key)) != getattr(state.ema, key):
                raise ValueError(f"EMA {key} changed; use the checkpoint's EMA settings.")

    # Keep configured peak rates; saved rates may be partway through warmup.
    optimizer_settings = [
        {key: group[key] for key in ("group_name", "lr", "weight_decay", "betas")}
        for group in state.optimizer.param_groups
    ]
    state.branch.load_pdae_state_dict(checkpoint["model"])
    state.optimizer.load_state_dict(checkpoint["optimizer"])
    for group, settings in zip(state.optimizer.param_groups, optimizer_settings):
        group.update(settings)
    if state.ema is not None:
        state.ema.load_state_dict(saved_ema, state.branch)
    if checkpoint.get("rng_state") is not None:
        torch.set_rng_state(checkpoint["rng_state"].cpu())
    if checkpoint.get("dataloader_generator_state") is not None:
        loader.generator.set_state(checkpoint["dataloader_generator_state"].cpu())
    if state.device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device=state.device)
    return int(checkpoint["step"])
