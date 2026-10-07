"""Optional Stage 1A LR warmup, indexed by completed optimizer updates on resume."""
from __future__ import annotations


def resolve_lr_schedule(config=None) -> dict:
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config) - {"type", "warmup_steps"}:
        raise ValueError("train.lr_schedule accepts only type and warmup_steps.")
    kind = config.get("type", "constant")
    warmup = config.get("warmup_steps", 0)
    if kind not in {"constant", "constant_with_warmup"}:
        raise ValueError("train.lr_schedule.type must be constant or constant_with_warmup.")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("train.lr_schedule.warmup_steps must be a non-negative integer.")
    if (kind == "constant" and warmup != 0) or (kind == "constant_with_warmup" and warmup == 0):
        raise ValueError("constant requires zero warmup_steps; constant_with_warmup requires positive warmup_steps.")
    return {"type": kind, "warmup_steps": warmup}


def apply_step_learning_rates(optimizer, base_lrs, schedule: dict, step: int) -> float:
    """Set rates BEFORE the one-based optimizer update, never per microbatch.

    Deriving the factor from the saved global step prevents warmup restarting
    after resume and keeps --max-steps smoke runs on the same LR trajectory.
    """
    if step < 1 or len(base_lrs) != len(optimizer.param_groups):
        raise ValueError("LR scheduling requires a positive step and one base LR per optimizer group.")
    warmup = schedule["warmup_steps"]
    factor = min(1.0, step / warmup) if warmup else 1.0
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr) * factor
    return factor
