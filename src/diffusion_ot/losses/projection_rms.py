"""Reference-only running distance scales for conditional InfoOT projection.

This is the main self-supervised projection recipe; legacy controls remain
available through their query_batch configuration. Fitting and the neural MI objective
continue to use live reference RMS (including its full derivative). Running
statistics are detached: no straight-through surrogate denominator gradients.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
import torch.nn.functional as F


DOMAINS = ("cat", "dog")


def projection_rms_options(config: dict) -> dict | None:
    options = config.get("projection_rms") or {}
    mode = options.get("mode", "query_batch")
    if mode == "query_batch":
        if set(options) - {"mode"}:
            raise ValueError("projection_rms decay/eps require mode: reference_ema.")
        return None
    if mode != "reference_ema" or set(options) - {"mode", "decay", "eps"}:
        raise ValueError("projection_rms supports query_batch or reference_ema with decay/eps.")
    decay, eps = float(options.get("decay", .99)), float(options.get("eps", 1e-8))
    if not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("projection_rms.decay must be finite and in [0, 1).")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("projection_rms.eps must be finite and positive.")
    return {"mode": mode, "decay": decay, "eps": eps}


def validate_projection_scales(scales: dict[str, float]) -> dict[str, float]:
    if set(scales) != set(DOMAINS):
        raise ValueError("Projection RMS scales must contain exactly cat and dog.")
    result = {d: float(scales[d]) for d in DOMAINS}
    if any(not math.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError("Projection RMS scales must be finite and positive.")
    return result


@torch.no_grad()
def reference_variances(features: dict[str, torch.Tensor]) -> dict[str, float]:
    """trace Cov_biased(m) = mean_ij ||m_i-m_j||^2 / 2, for L2 rows m.

    Re-center in FP64 rather than subtracting nearly equal squared norms.
    Only training references belong here, never queries or generated images.
    """
    if set(features) != set(DOMAINS):
        raise ValueError("Reference features must contain exactly cat and dog.")
    result = {}
    for domain, value in features.items():
        if value.ndim != 2 or len(value) < 2 or value.shape[1] == 0:
            raise ValueError("Projection RMS needs at least two reference feature rows.")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("Projection RMS needs finite floating-point features.")
        value = value.detach().double()
        if (value.norm(dim=1) < 1e-8).any():
            raise ValueError("Projection RMS needs nonzero matching features.")
        value = F.normalize(value, dim=1)
        result[domain] = float((value - value.mean(0)).square().sum(1).mean())
    return result


class ReferenceRMSEMA:
    """One variance EMA per domain, with explicit once-per-step updates."""

    def __init__(self, *, decay: float = .99, eps: float = 1e-8):
        options = projection_rms_options({"projection_rms": {
            "mode": "reference_ema", "decay": decay, "eps": eps}})
        self.decay, self.eps = options["decay"], options["eps"]
        self.variances: dict[str, float] = {}
        self.batch_variances: dict[str, float] = {}
        self.num_updates = 0

    def initialize(self, references: dict[str, torch.Tensor]) -> None:
        if self.variances:
            raise ValueError("Projection RMS is already initialized.")
        self.variances = reference_variances(references)
        self.batch_variances = dict(self.variances)

    def update(self, references: dict[str, torch.Tensor], *, step: int) -> None:
        if not self.variances or step != self.num_updates + 1:
            raise ValueError("Initialize projection RMS, then update exactly once per training step.")
        batch = reference_variances(references)
        self.variances = {d: self.decay * self.variances[d] + (1 - self.decay) * batch[d]
                          for d in DOMAINS}
        self.batch_variances = batch
        self.num_updates = step

    def scales(self) -> dict[str, float]:
        if not self.variances:
            raise ValueError("Projection RMS has not been initialized.")
        return {d: math.sqrt(max(v, self.eps ** 2)) for d, v in self.variances.items()}

    def diagnostics(self, *, bandwidth: float) -> dict:
        scales = self.scales()
        batch = {d: math.sqrt(max(v, self.eps ** 2)) for d, v in self.batch_variances.items()}
        return {"mode": "reference_ema", "decay": self.decay,
                "num_updates": self.num_updates, "scales": scales, "batch_scales": batch,
                "running_to_batch_ratio": {d: scales[d] / batch[d] for d in DOMAINS},
                "effective_kernel_width": {d: bandwidth * scales[d] for d in DOMAINS},
                "scale_gradient": "detached_running_statistic"}

    def state_dict(self) -> dict:
        self.scales()  # Do not save an uninitialized state.
        return deepcopy({"version": 1, "decay": self.decay, "eps": self.eps,
                         "num_updates": self.num_updates, "variances": self.variances,
                         "batch_variances": self.batch_variances})

    def load_state_dict(self, state: dict, *, step: int) -> None:
        if (state.get("version") != 1 or state.get("decay") != self.decay
                or state.get("eps") != self.eps or state.get("num_updates") != step or step < 0):
            raise ValueError("Projection RMS checkpoint protocol or update count does not match.")
        for key in ("variances", "batch_variances"):
            values = state.get(key) or {}
            if set(values) != set(DOMAINS) or any(
                    not math.isfinite(float(v)) or float(v) < 0 for v in values.values()):
                raise ValueError("Invalid projection RMS checkpoint variances.")
        self.variances = {d: float(v) for d, v in state["variances"].items()}
        self.batch_variances = {d: float(v) for d, v in state["batch_variances"].items()}
        self.num_updates = step


def checkpoint_projection_rms(checkpoint: dict, config: dict, *, weights: str) -> ReferenceRMSEMA | None:
    """Strictly select statistics paired with raw or model-EMA E/head weights."""
    options = projection_rms_options(config)
    if options != projection_rms_options(checkpoint.get("config") or {}):
        raise ValueError("Checkpoint and config projection_rms protocols disagree.")
    if options is None:
        return None
    state = (checkpoint.get("projection_rms_state") or {}).get(weights)
    if state is None:
        raise ValueError(f"Checkpoint has no projection_rms_state.{weights}; start the new RMS experiment fresh.")
    tracker = ReferenceRMSEMA(decay=options["decay"], eps=options["eps"])
    tracker.load_state_dict(state, step=int(checkpoint["step"]))
    return tracker
