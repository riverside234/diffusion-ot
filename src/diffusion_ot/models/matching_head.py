"""Learn InfoOT geometry separately from the raw codes used by frozen decoders.

This is a project extension, not part of the official InfoOT solver. Heads
start at normalized identity, and remain part of inference/checkpoint state.
"""
from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ResidualMatchingHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if input_dim < 2 or hidden_dim < 1:
            raise ValueError("Matching head dimensions must be positive, with input_dim >= 2.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual = nn.Sequential(
            nn.LayerNorm(self.input_dim, elementwise_affine=False),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 2 or codes.shape[1] != self.input_dim:
            raise ValueError(f"Matching head expects [N, {self.input_dim}] raw codes.")
        codes = codes.float()
        return F.normalize(F.normalize(codes, dim=-1) + self.residual(codes), dim=-1)


def matching_head_spec(config: dict[str, Any]) -> dict[str, Any]:
    options = config.get("matching_head") or {}
    if not options.get("enabled", False):
        return {"enabled": False}
    kind = str(options.get("kind", "residual_mlp_v1"))
    if kind != "residual_mlp_v1":
        raise ValueError(f"Unsupported matching_head.kind: {kind}")
    spec = {"enabled": True, "kind": kind,
            "input_dim": int(options.get("input_dim", 512)),
            "hidden_dim": int(options.get("hidden_dim", 256))}
    if spec["input_dim"] < 2 or spec["hidden_dim"] < 1:
        raise ValueError("Invalid matching head dimensions.")
    return spec


def make_matching_head(spec: dict[str, Any], *, device: str, seed: int = 0) -> ResidualMatchingHead:
    # CPU initialization in a local RNG scope keeps data/noise sampling identical
    # to the no-head control. The zero last layer makes all initial outputs equal.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        head = ResidualMatchingHead(spec["input_dim"], spec["hidden_dim"])
    return head.to(device=device, dtype=torch.float32)


def matching_features(codes: torch.Tensor, head: nn.Module | None = None) -> torch.Tensor:
    return F.normalize(codes.float(), dim=-1) if head is None else head(codes.float())


def matching_head_id(head: nn.Module | None) -> str:
    if head is None:
        return "l2_raw_v1"
    digest = hashlib.sha256(type(head).__name__.encode())
    for name, tensor in sorted(head.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return "residual_mlp_v1_" + digest.hexdigest()[:16]


def load_matching_head(
    checkpoint: dict[str, Any], domain: str, *, weights: str, device: str,
) -> ResidualMatchingHead | None:
    if weights not in {"ema", "raw"}:
        raise ValueError("Matching head weights must be raw or ema.")
    spec = matching_head_spec(checkpoint.get("config") or {})
    if not spec["enabled"]:
        if checkpoint.get("matching_heads") or checkpoint.get("matching_head_ema"):
            raise ValueError("Checkpoint has matching heads but its config disables them.")
        return None
    state_key = "matching_head_ema" if weights == "ema" else "matching_heads"
    state = (checkpoint.get(state_key) or {}).get(domain)
    if state is None:
        raise ValueError(f"Checkpoint has no {state_key}.{domain}; cannot silently use raw-code matching.")
    head = make_matching_head(spec, device=device)
    head.load_state_dict(state, strict=True)
    return head.eval().requires_grad_(False)
