"""Per-domain MLPs for global source-code InfoNCE, separate from InfoOT heads."""
from __future__ import annotations

import math

from diffusion_ot.models.patch_sampler import PatchSampleMLP


def code_projector_options(config):
    """Absent options preserve the original raw-code InfoNCE experiment."""
    if config is None:
        return None
    if not isinstance(config, dict) or set(config) - {"kind", "projection_dim", "lr", "grad_clip_norm"}:
        raise ValueError("Unknown source_contrastive_projector option.")
    if config.get("kind", "mlp") != "mlp":
        raise ValueError("source_contrastive_projector.kind must be mlp.")
    width = config.get("projection_dim", 256)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("source_contrastive_projector.projection_dim must be a positive integer.")
    options = {"kind": "mlp", "projection_dim": width}
    for name, default in (("lr", 2e-4), ("grad_clip_norm", 1.)):
        value = float(config.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"source_contrastive_projector.{name} must be finite and positive.")
        options[name] = value
    return options


class CodeProjectionMLP(PatchSampleMLP):
    """Reuse CUT's two-linear-layer transform for one global vector per image.

    Input width -> projection_dim -> projection_dim, with ReLU between layers.
    InfoNCE normalizes the output. There is no spatial sampling or InfoOT change.
    """

    def __init__(self, input_dim, projection_dim=256, *, seed=0):
        super().__init__((input_dim,), projection_dim, seed=seed)

    def forward(self, codes):
        if codes.ndim != 2 or codes.shape[1] != self.channels[0]:
            raise ValueError(f"Global projector requires [batch, {self.channels[0]}] codes.")
        return super().forward(codes, 0)


def load_code_projector(input_dim, options, checkpoint, domain, *, weights, device):
    """Restore the selected raw/EMA head; never silently use random weights."""
    if weights not in {"raw", "ema"}:
        raise ValueError("Code projector weights must be raw or ema.")
    if options is None:
        return None
    state_key = "code_projector_ema" if weights == "ema" else "code_projectors"
    state = (checkpoint.get(state_key) or {}).get(domain)
    if state is None:
        raise ValueError(f"Checkpoint has no {state_key}.{domain}; MLP InfoNCE requires a fresh run.")
    head = CodeProjectionMLP(input_dim, options["projection_dim"]).to(device)
    head.load_state_dict(state, strict=True)
    return head.eval().requires_grad_(False)
