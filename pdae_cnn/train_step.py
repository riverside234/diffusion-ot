"""One optimizer update, including gradient accumulation and EMA."""

import torch

from pdae_cnn.data import prepare_batch
from pdae_cnn.objective import compute_flow_loss


def train_step(state, batches, config, *, step):
    """Return detached metrics after one complete optimizer update."""
    options = config.get("train") or {}
    accumulation = int((config.get("dataloader") or {}).get("gradient_accumulation_steps", 1))
    ema_every = int((config.get("ema") or {}).get("update_every", 1))
    if accumulation <= 0 or (state.ema is not None and ema_every <= 0):
        raise ValueError("Gradient accumulation and enabled EMA update intervals must be positive.")
    clip_norm = options.get("grad_clip_norm", 1.0)
    if clip_norm is not None and not (float(clip_norm) > 0):
        raise ValueError("train.grad_clip_norm must be positive or null.")

    state.branch.train()
    state.optimizer.zero_grad(set_to_none=True)
    metrics = {}
    for _ in range(accumulation):
        batch = prepare_batch(
            next(batches), device=state.device, dtype=state.dtype,
            augmentation=state.augmentation, vae=state.vae,
        )
        loss, values = compute_flow_loss(state.branch, batch, config)
        (loss / accumulation).backward()
        for name, value in values.items():
            metrics[name] = metrics.get(name, 0) + value / accumulation
    if not torch.isfinite(metrics["loss"]):
        raise FloatingPointError(f"Non-finite training loss at step {step}.")
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in state.branch.parameters() if parameter.requires_grad],
        float(clip_norm) if clip_norm is not None else float("inf"), error_if_nonfinite=True,
    )
    state.optimizer.step()
    if state.ema is not None and step % ema_every == 0:
        state.ema.update(state.branch)
    return metrics
