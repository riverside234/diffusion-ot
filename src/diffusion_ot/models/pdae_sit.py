from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn


def _as_list(value: Any, default: list[int] | None = None) -> list[int]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(item.strip()) for item in value.split(",")]
    return [int(item) for item in value]


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _module_config_value(module: Any, key: str, default: Any = None) -> Any:
    config = getattr(module, "config", None)
    value = _config_value(config, key, None)
    if value is not None:
        return value

    # Diffusers emits deprecation warnings when config fields are read as direct
    # model attributes, so only inspect real instance attributes as a fallback.
    return getattr(module, "__dict__", {}).get(key, default)


def _valid_group_count(channels: int, preferred_groups: int) -> int:
    groups = min(int(preferred_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def _patch_size_from_transformer(transformer: Any) -> int:
    patch_size = getattr(getattr(transformer, "x_embedder", None), "patch_size", None)
    if isinstance(patch_size, Iterable):
        return int(tuple(patch_size)[0])
    if patch_size is not None:
        return int(patch_size)
    return int(_module_config_value(transformer, "patch_size", 2))


def _final_patch_dim(transformer: Any) -> int:
    final_layer = getattr(transformer, "final_layer", None)
    linear = getattr(final_layer, "linear", None)
    if linear is not None and hasattr(linear, "out_features"):
        return int(linear.out_features)

    patch_size = _patch_size_from_transformer(transformer)
    in_channels = int(_module_config_value(transformer, "in_channels", 4))
    learn_sigma = bool(_module_config_value(transformer, "learn_sigma", False))
    out_channels = _module_config_value(transformer, "out_channels", None)
    if out_channels is None:
        out_channels = in_channels * 2 if learn_sigma else in_channels
    return patch_size * patch_size * int(out_channels)


@dataclass
class PDAESiTOutput:
    sample: torch.Tensor
    z: torch.Tensor
    base_sample: torch.Tensor | None = None
    delta_sample: torch.Tensor | None = None
    conditioned_z: torch.Tensor | None = None
    semantic_drop_mask: torch.Tensor | None = None
    conditional_delta_sample: torch.Tensor | None = None
    null_delta_sample: torch.Tensor | None = None


class LearnedSemanticCondition(nn.Module):
    """Per-sample semantic dropout backed by a learned null latent."""

    def __init__(self, z_dim: int, dropout_probability: float = 0.1) -> None:
        super().__init__()
        if z_dim <= 0:
            raise ValueError("z_dim must be positive.")
        if not 0.0 <= dropout_probability <= 1.0:
            raise ValueError("semantic CFG dropout_probability must be in [0, 1].")
        self.z_dim = int(z_dim)
        self.dropout_probability = float(dropout_probability)
        self.null_token = nn.Parameter(torch.zeros(self.z_dim))

    def null_like(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected z with shape [B, {self.z_dim}], got {tuple(z.shape)}."
            )
        return self.null_token.to(device=z.device, dtype=z.dtype).unsqueeze(0).expand_as(z)

    def forward(
        self,
        z: torch.Tensor,
        *,
        apply_dropout: bool,
        force_drop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        null_z = self.null_like(z)
        if force_drop_mask is not None:
            mask = force_drop_mask.to(device=z.device, dtype=torch.bool)
            if mask.ndim != 1 or mask.shape[0] != z.shape[0]:
                raise ValueError(
                    "force_drop_mask must be a Boolean tensor with shape [batch_size]."
                )
        elif apply_dropout and self.dropout_probability > 0.0:
            mask = torch.rand(z.shape[0], device=z.device) < self.dropout_probability
        else:
            mask = torch.zeros(z.shape[0], device=z.device, dtype=torch.bool)
        return torch.where(mask[:, None], null_z, z), mask


class PDAELatentEncoder(nn.Module):
    """Small PDAE-style encoder: latent image -> stacked GN-SiLU-Conv -> linear z."""

    def __init__(
        self,
        input_channels: int = 4,
        z_dim: int = 512,
        channels: Iterable[int] = (64, 128, 256),
        spatial_size: int = 4,
        num_groups: int = 32,
        normalize_z: bool = True,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = int(input_channels)
        for out_channels in channels:
            groups = _valid_group_count(in_channels, num_groups)
            blocks.extend(
                [
                    nn.GroupNorm(groups, in_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels, int(out_channels), kernel_size=3, stride=2, padding=1),
                ]
            )
            in_channels = int(out_channels)

        blocks.extend(
            [
                nn.GroupNorm(_valid_group_count(in_channels, num_groups), in_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((int(spatial_size), int(spatial_size))),
            ]
        )
        self.conv = nn.Sequential(*blocks)
        self.proj = nn.Linear(in_channels * int(spatial_size) * int(spatial_size), int(z_dim))
        self.z_norm = nn.LayerNorm(int(z_dim)) if normalize_z else nn.Identity()

    def forward(self, x0_latent: torch.Tensor) -> torch.Tensor:
        h = self.conv(x0_latent)
        z = self.proj(h.flatten(start_dim=1))
        return self.z_norm(z)


class AdaLNZeroResidualAdapter(nn.Module):
    """AdaLN-Zero residual branch used as the flow-matching analogue of PDAE's G."""

    def __init__(
        self,
        hidden_size: int,
        cond_dim: int | None = None,
        out_dim: int | None = None,
        bottleneck_dim: int = 64,
        norm_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.out_dim = int(out_dim or hidden_size)
        cond_dim = int(cond_dim or hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=norm_eps)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * self.hidden_size))
        self.in_proj = nn.Linear(self.hidden_size, int(bottleneck_dim))
        self.act = nn.GELU()
        self.hidden_proj = nn.Linear(int(bottleneck_dim), self.hidden_size)
        self.out_proj = (
            nn.Identity()
            if self.out_dim == self.hidden_size
            else nn.Linear(self.hidden_size, self.out_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear = self.cond_proj[-1]
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
        if isinstance(self.out_proj, nn.Linear):
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.cond_proj(cond).chunk(3, dim=-1)
        h = self.norm(tokens)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]
        h = self.hidden_proj(self.act(self.in_proj(h)))
        h = gate[:, None, :].to(dtype=h.dtype) * h
        return self.out_proj(h)


class SemanticLoRALinear(nn.Module):
    """A frozen linear layer with a LoRA update enabled only on semantic passes."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int = 4,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("SemanticLoRALinear requires an nn.Linear base layer.")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1).")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(self.rank if alpha is None else alpha)
        if self.alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive.")
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.lora_down = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base_layer.out_features, bias=False)
        self.lora_down.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_up.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.active = False

        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(inputs)
        if not self.active:
            return output
        delta = self.lora_up(self.dropout(self.lora_down(inputs)))
        return output + delta.to(dtype=output.dtype) * self.scaling

    def lora_state_dict(self) -> dict[str, Any]:
        return {
            "lora_down": self.lora_down.state_dict(),
            "lora_up": self.lora_up.state_dict(),
        }

    def load_lora_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        self.lora_down.load_state_dict(state_dict["lora_down"], strict=strict)
        self.lora_up.load_state_dict(state_dict["lora_up"], strict=strict)


class SemanticSiTWrapper(nn.Module):
    """Frozen SiT transformer plus trainable z-conditioned residual adapters."""

    def __init__(
        self,
        transformer: nn.Module,
        z_dim: int = 512,
        injection_layers: Iterable[int] | None = None,
        bottleneck_dim: int = 64,
        freeze_base: bool = True,
        attention_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.0,
        lora_layers: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        if attention_lora and not freeze_base:
            raise ValueError("Semantic attention LoRA requires adapter.freeze_base=true.")
        self.base = transformer
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            self.base.eval()

        hidden_size = int(_module_config_value(transformer, "hidden_size", 768))
        depth = int(_module_config_value(transformer, "depth", len(getattr(transformer, "blocks", []))))
        default_layers = list(range(max(depth // 3, 0), depth))
        layers = _as_list(injection_layers, default=default_layers)
        self.injection_layers = [layer for layer in layers if 0 <= layer < depth]
        self.z_proj = nn.Sequential(
            nn.LayerNorm(int(z_dim)),
            nn.Linear(int(z_dim), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.adapters = nn.ModuleDict(
            {
                str(layer): AdaLNZeroResidualAdapter(
                    hidden_size=hidden_size,
                    cond_dim=hidden_size,
                    out_dim=hidden_size,
                    bottleneck_dim=bottleneck_dim,
                )
                for layer in self.injection_layers
            }
        )
        self.final_adapter = AdaLNZeroResidualAdapter(
            hidden_size=hidden_size,
            cond_dim=hidden_size,
            out_dim=_final_patch_dim(transformer),
            bottleneck_dim=bottleneck_dim,
        )
        self.attention_lora_enabled = bool(attention_lora)
        default_lora_layers = self.injection_layers
        requested_lora_layers = _as_list(lora_layers, default=default_lora_layers)
        self.lora_layers = sorted(set(requested_lora_layers))
        invalid_lora_layers = [
            layer for layer in self.lora_layers if not 0 <= layer < depth
        ]
        if invalid_lora_layers:
            raise ValueError(f"LoRA layers are outside the SiT depth: {invalid_lora_layers}")
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(self.lora_rank if lora_alpha is None else lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_targets = ("qkv", "proj")
        self._attention_lora_modules: dict[int, dict[str, SemanticLoRALinear]] = {}
        if self.attention_lora_enabled:
            if not self.lora_layers:
                raise ValueError("At least one LoRA layer is required when LoRA is enabled.")
            self._install_attention_lora()

    def _install_attention_lora(self) -> None:
        for layer_index in self.lora_layers:
            block = self.base.blocks[layer_index]
            attention = getattr(block, "attn", None)
            if attention is None:
                raise ValueError(f"SiT block {layer_index} has no attn module for LoRA.")
            layer_modules: dict[str, SemanticLoRALinear] = {}
            for target in self.lora_targets:
                base_layer = getattr(attention, target, None)
                if not isinstance(base_layer, nn.Linear):
                    raise ValueError(
                        f"SiT block {layer_index} attention target {target!r} is not nn.Linear."
                    )
                wrapped = SemanticLoRALinear(
                    base_layer,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                )
                setattr(attention, target, wrapped)
                layer_modules[target] = wrapped
            self._attention_lora_modules[layer_index] = layer_modules

    @contextmanager
    def _semantic_lora(self, layer_index: int):
        modules = self._attention_lora_modules.get(layer_index, {}).values()
        for module in modules:
            module.active = True
        try:
            yield
        finally:
            for module in modules:
                module.active = False

    def _attention_lora_state_dict(self) -> dict[str, Any]:
        return {
            str(layer): {
                target: module.lora_state_dict()
                for target, module in modules.items()
            }
            for layer, modules in self._attention_lora_modules.items()
        }

    def _load_attention_lora_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool,
    ) -> None:
        expected_layers = {str(layer) for layer in self._attention_lora_modules}
        received_layers = set(state_dict)
        if strict and received_layers != expected_layers:
            raise ValueError(
                "Attention LoRA checkpoint layers do not match the model: "
                f"expected {sorted(expected_layers)}, received {sorted(received_layers)}."
            )
        for layer, modules in self._attention_lora_modules.items():
            saved_modules = state_dict.get(str(layer))
            if saved_modules is None:
                if strict:
                    raise ValueError(f"Attention LoRA checkpoint is missing layer {layer}.")
                continue
            if strict and set(saved_modules) != set(modules):
                raise ValueError(
                    f"Attention LoRA targets differ at layer {layer}: "
                    f"expected {sorted(modules)}, received {sorted(saved_modules)}."
                )
            for target, module in modules.items():
                if target in saved_modules:
                    module.load_lora_state_dict(saved_modules[target], strict=strict)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        for modules in self._attention_lora_modules.values():
            for module in modules.values():
                module.train(mode)
        return self

    def trainable_state_dict(self) -> dict[str, Any]:
        state = {
            "z_proj": self.z_proj.state_dict(),
            "adapters": self.adapters.state_dict(),
            "final_adapter": self.final_adapter.state_dict(),
            "attention_lora_enabled": self.attention_lora_enabled,
        }
        if self.attention_lora_enabled:
            state["attention_lora_config"] = {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "layers": self.lora_layers,
                "targets": list(self.lora_targets),
            }
            state["attention_lora"] = self._attention_lora_state_dict()
        return state

    def load_trainable_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> None:
        saved_lora_enabled = bool(
            state_dict.get("attention_lora_enabled", "attention_lora" in state_dict)
        )
        if saved_lora_enabled != self.attention_lora_enabled:
            raise ValueError(
                "Checkpoint attention-LoRA setting does not match the Stage 1A config."
            )
        if self.attention_lora_enabled and strict:
            saved_config = state_dict.get("attention_lora_config") or {}
            expected_config = {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "layers": self.lora_layers,
                "targets": list(self.lora_targets),
            }
            if saved_config != expected_config:
                raise ValueError(
                    "Checkpoint attention-LoRA configuration does not match the model: "
                    f"expected {expected_config}, received {saved_config}."
                )
        self.z_proj.load_state_dict(state_dict["z_proj"], strict=strict)
        self.adapters.load_state_dict(state_dict["adapters"], strict=strict)
        self.final_adapter.load_state_dict(state_dict["final_adapter"], strict=strict)
        if self.attention_lora_enabled:
            saved_lora = state_dict.get("attention_lora")
            if saved_lora is None:
                raise ValueError("Attention-LoRA checkpoint is missing its LoRA weights.")
            self._load_attention_lora_state_dict(saved_lora, strict=strict)

    def _label_embedding(self, class_labels, batch_size: int, device, force_drop_ids=None):
        y_embedder = getattr(self.base, "y_embedder", None)
        if y_embedder is None:
            hidden_size = int(_module_config_value(self.base, "hidden_size", 768))
            return torch.zeros(batch_size, hidden_size, device=device)
        if class_labels is None:
            class_labels = make_null_class_labels(self.base, batch_size, device)
        return y_embedder(class_labels, False, force_drop_ids=force_drop_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        z: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        force_drop_ids=None,
        return_dict: bool = True,
    ):
        x = self.base.x_embedder(hidden_states)
        pos_embed = self.base.pos_embed
        if pos_embed.ndim == 2:
            pos_embed = pos_embed.unsqueeze(0)
        x = x + pos_embed.to(device=x.device, dtype=x.dtype)

        c_base = self.base.t_embedder(timestep)
        c_base = c_base + self._label_embedding(
            class_labels,
            batch_size=hidden_states.shape[0],
            device=hidden_states.device,
            force_drop_ids=force_drop_ids,
        ).to(dtype=c_base.dtype)
        c_sem = c_base + self.z_proj(z.to(device=c_base.device, dtype=c_base.dtype))

        x_base = x
        x_sem = x
        for layer_index, block in enumerate(self.base.blocks):
            x_base = block(x_base, c_base)
            with self._semantic_lora(layer_index):
                x_sem = block(x_sem, c_base)
            adapter = self.adapters[str(layer_index)] if str(layer_index) in self.adapters else None
            if adapter is not None:
                x_sem = x_sem + adapter(x_sem, c_sem)

        base_patch = self.base.final_layer(x_base, c_base)
        sample_patch = self.base.final_layer(x_sem, c_base)
        sample_patch = sample_patch + self.final_adapter(x_sem, c_sem).to(dtype=sample_patch.dtype)
        base_sample = self.base.unpatchify(base_patch)
        sample = self.base.unpatchify(sample_patch)
        if bool(_module_config_value(self.base, "learn_sigma", False)):
            sample, _ = sample.chunk(2, dim=1)
            base_sample, _ = base_sample.chunk(2, dim=1)
        delta_sample = sample - base_sample

        output = PDAESiTOutput(
            sample=sample,
            z=z,
            base_sample=base_sample,
            delta_sample=delta_sample,
        )
        if return_dict:
            return output
        return (output.sample, output.z, output.base_sample, output.delta_sample)


class PDAESiTBranch(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        semantic_transformer: nn.Module,
        *,
        z_dim: int | None = None,
        semantic_cfg_enabled: bool = False,
        semantic_dropout_probability: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.semantic_transformer = semantic_transformer
        if semantic_cfg_enabled and z_dim is None:
            raise ValueError("z_dim is required when semantic CFG is enabled.")
        self.semantic_conditioner = (
            LearnedSemanticCondition(
                z_dim=int(z_dim),
                dropout_probability=semantic_dropout_probability,
            )
            if semantic_cfg_enabled
            else None
        )

    @property
    def semantic_cfg_enabled(self) -> bool:
        return self.semantic_conditioner is not None

    def encode(self, x0_latent: torch.Tensor) -> torch.Tensor:
        return self.encoder(x0_latent)

    def semantic_null_like(self, z: torch.Tensor) -> torch.Tensor:
        if self.semantic_conditioner is None:
            raise RuntimeError("This PDAE branch was not configured with semantic CFG.")
        return self.semantic_conditioner.null_like(z)

    def condition_semantic_z(
        self,
        z: torch.Tensor,
        *,
        apply_dropout: bool,
        force_drop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.semantic_conditioner is None:
            if force_drop_mask is not None and bool(force_drop_mask.any()):
                raise RuntimeError("Cannot force semantic dropout when semantic CFG is disabled.")
            return z, torch.zeros(z.shape[0], device=z.device, dtype=torch.bool)
        return self.semantic_conditioner(
            z,
            apply_dropout=apply_dropout,
            force_drop_mask=force_drop_mask,
        )

    def predict_with_z(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        z: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        force_drop_ids=None,
        return_dict: bool = True,
    ):
        return self.semantic_transformer(
            hidden_states=x_t,
            timestep=timestep,
            z=z,
            class_labels=class_labels,
            force_drop_ids=force_drop_ids,
            return_dict=return_dict,
        )

    def predict_cfg_with_z(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        z: torch.Tensor,
        *,
        guidance_scale: float = 1.0,
        class_labels: torch.Tensor | None = None,
        force_drop_ids=None,
        return_dict: bool = True,
    ):
        """Predict base + G(null) + scale * (G(z) - G(null))."""
        if not self.semantic_cfg_enabled:
            output = self.predict_with_z(
                x_t=x_t,
                timestep=timestep,
                z=z,
                class_labels=class_labels,
                force_drop_ids=force_drop_ids,
            )
            guided_delta = float(guidance_scale) * output.delta_sample
            guided = PDAESiTOutput(
                sample=output.base_sample + guided_delta,
                z=z,
                base_sample=output.base_sample,
                delta_sample=guided_delta,
                conditioned_z=z,
                conditional_delta_sample=output.delta_sample,
            )
        else:
            if float(guidance_scale) == 0.0:
                null_z = self.semantic_null_like(z)
                unconditional = self.predict_with_z(
                    x_t=x_t,
                    timestep=timestep,
                    z=null_z,
                    class_labels=class_labels,
                    force_drop_ids=force_drop_ids,
                )
                unconditional.z = z
                unconditional.conditioned_z = null_z
                unconditional.null_delta_sample = unconditional.delta_sample
                if return_dict:
                    return unconditional
                return (
                    unconditional.sample,
                    unconditional.z,
                    unconditional.base_sample,
                    unconditional.delta_sample,
                )
            conditional = self.predict_with_z(
                x_t=x_t,
                timestep=timestep,
                z=z,
                class_labels=class_labels,
                force_drop_ids=force_drop_ids,
            )
            if float(guidance_scale) == 1.0:
                conditional.conditioned_z = z
                conditional.conditional_delta_sample = conditional.delta_sample
                if return_dict:
                    return conditional
                return (
                    conditional.sample,
                    conditional.z,
                    conditional.base_sample,
                    conditional.delta_sample,
                )
            null_z = self.semantic_null_like(z)
            unconditional = self.predict_with_z(
                x_t=x_t,
                timestep=timestep,
                z=null_z,
                class_labels=class_labels,
                force_drop_ids=force_drop_ids,
            )
            guided_delta = unconditional.delta_sample + float(guidance_scale) * (
                conditional.delta_sample - unconditional.delta_sample
            )
            guided = PDAESiTOutput(
                sample=conditional.base_sample + guided_delta,
                z=z,
                base_sample=conditional.base_sample,
                delta_sample=guided_delta,
                conditioned_z=z,
                conditional_delta_sample=conditional.delta_sample,
                null_delta_sample=unconditional.delta_sample,
            )
        if return_dict:
            return guided
        return (guided.sample, guided.z, guided.base_sample, guided.delta_sample)

    def forward(
        self,
        x0_latent: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        force_drop_ids=None,
        semantic_drop_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        z = self.encode(x0_latent)
        conditioned_z, drop_mask = self.condition_semantic_z(
            z,
            apply_dropout=self.training,
            force_drop_mask=semantic_drop_mask,
        )
        output = self.predict_with_z(
            x_t=x_t,
            timestep=timestep,
            z=conditioned_z,
            class_labels=class_labels,
            force_drop_ids=force_drop_ids,
            return_dict=True,
        )
        output.z = z
        output.conditioned_z = conditioned_z
        output.semantic_drop_mask = drop_mask
        if return_dict:
            return output
        return (output.sample, output.z, output.base_sample, output.delta_sample)

    def generator_state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "semantic_transformer": self.semantic_transformer.trainable_state_dict(),
            "semantic_cfg_enabled": self.semantic_cfg_enabled,
        }
        if self.semantic_conditioner is not None:
            state["semantic_conditioner"] = self.semantic_conditioner.state_dict()
            state["semantic_dropout_probability"] = (
                self.semantic_conditioner.dropout_probability
            )
        return state

    def load_generator_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> None:
        # Stage 1B checkpoints created before semantic CFG stored G directly.
        if "semantic_transformer" not in state_dict:
            if self.semantic_cfg_enabled:
                raise ValueError(
                    "The generator checkpoint predates learned-null semantic CFG. "
                    "Retrain Stage 1A or use a non-CFG Stage 1A config."
                )
            self.semantic_transformer.load_trainable_state_dict(state_dict, strict=strict)
            return

        saved_cfg_enabled = bool(state_dict.get("semantic_cfg_enabled", False))
        if saved_cfg_enabled != self.semantic_cfg_enabled:
            raise ValueError(
                "Checkpoint semantic CFG setting does not match the current Stage 1A config."
            )
        self.semantic_transformer.load_trainable_state_dict(
            state_dict["semantic_transformer"], strict=strict
        )
        if self.semantic_conditioner is not None:
            conditioner_state = state_dict.get("semantic_conditioner")
            if conditioner_state is None:
                raise ValueError("Learned-null checkpoint is missing semantic_conditioner.")
            self.semantic_conditioner.load_state_dict(conditioner_state, strict=strict)

    def pdae_state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 3,
            "encoder": self.encoder.state_dict(),
            "generator": self.generator_state_dict(),
        }

    def load_pdae_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> None:
        self.encoder.load_state_dict(state_dict["encoder"], strict=strict)
        if "generator" in state_dict:
            self.load_generator_state_dict(state_dict["generator"], strict=strict)
            return
        if self.semantic_cfg_enabled:
            raise ValueError(
                "This Stage 1A checkpoint has no learned null token. Start a new semantic-CFG "
                "run instead of resuming the older non-CFG checkpoint."
            )
        self.semantic_transformer.load_trainable_state_dict(
            state_dict["semantic_transformer"], strict=strict
        )


def build_pdae_sit_branch(
    transformer: Any,
    model_config: dict[str, Any] | None = None,
    stage_config: dict[str, Any] | None = None,
) -> PDAESiTBranch:
    encoder_config = dict((stage_config or {}).get("encoder") or {})
    adapter_config = dict((stage_config or {}).get("adapter") or {})
    semantic_cfg_config = dict((stage_config or {}).get("semantic_cfg") or {})
    z_dim = int(encoder_config.get("z_dim", 512))
    input_channels = int(
        encoder_config.get(
            "input_channels",
            _config_value(model_config, "latent_channels", _module_config_value(transformer, "in_channels", 4)),
        )
    )

    encoder = PDAELatentEncoder(
        input_channels=input_channels,
        z_dim=z_dim,
        channels=encoder_config.get("channels") or [64, 128, 256],
        spatial_size=int(encoder_config.get("spatial_size", 4)),
        num_groups=int(encoder_config.get("num_groups", 32)),
        normalize_z=bool(encoder_config.get("normalize_z", True)),
    )
    semantic_transformer = SemanticSiTWrapper(
        transformer=transformer,
        z_dim=z_dim,
        injection_layers=adapter_config.get("injection_layers"),
        bottleneck_dim=int(adapter_config.get("bottleneck_dim", 64)),
        freeze_base=bool(adapter_config.get("freeze_base", True)),
        attention_lora=bool(adapter_config.get("lora", False)),
        lora_rank=int(adapter_config.get("lora_rank", 4)),
        lora_alpha=float(
            adapter_config.get("lora_alpha", adapter_config.get("lora_rank", 4))
        ),
        lora_dropout=float(adapter_config.get("lora_dropout", 0.0)),
        lora_layers=adapter_config.get("lora_layers"),
    )
    semantic_cfg_enabled = bool(semantic_cfg_config.get("enabled", False))
    null_condition = str(semantic_cfg_config.get("null_condition", "learned_token"))
    drop_granularity = str(semantic_cfg_config.get("drop_granularity", "sample"))
    if semantic_cfg_enabled and null_condition != "learned_token":
        raise ValueError("Stage 1A semantic_cfg.null_condition must be learned_token.")
    if semantic_cfg_enabled and drop_granularity != "sample":
        raise ValueError("Stage 1A semantic_cfg.drop_granularity must be sample.")
    return PDAESiTBranch(
        encoder=encoder,
        semantic_transformer=semantic_transformer,
        z_dim=z_dim,
        semantic_cfg_enabled=semantic_cfg_enabled,
        semantic_dropout_probability=float(
            semantic_cfg_config.get("dropout_probability", 0.1)
        ),
    )


def make_null_class_labels(transformer: Any, batch_size: int, device, null_label: int | None = None):
    if null_label is None:
        null_label = _module_config_value(transformer, "num_classes", None)
    if null_label is None:
        y_embedder = getattr(transformer, "y_embedder", None)
        embedding_table = getattr(y_embedder, "embedding_table", None)
        if embedding_table is not None:
            null_label = int(embedding_table.num_embeddings) - 1
    if null_label is None:
        raise ValueError("Cannot infer null class label; pass null_label in the Stage 1A config.")
    return torch.full((int(batch_size),), int(null_label), device=device, dtype=torch.long)


def predict_base_velocity(
    transformer: Any,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    force_drop_ids=None,
) -> torch.Tensor:
    output = transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        class_labels=class_labels,
        force_drop_ids=force_drop_ids,
    )
    if hasattr(output, "sample"):
        return output.sample
    return output[0]
