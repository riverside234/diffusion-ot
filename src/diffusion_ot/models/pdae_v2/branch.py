from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from diffusion_ot.models.pdae_sit import (
    PDAESiTBranch, PDAESiTOutput, SemanticSiTWrapper, _module_config_value,
)
from .cross_attention import ImageCrossAttention, validate_padding_mask
from .encoder import FrozenSiglipPatchEncoder


class LearnedTokenCondition(nn.Module):
    def __init__(self, dropout_probability: float = 0.1):
        super().__init__()
        if not 0 <= dropout_probability <= 1:
            raise ValueError("Semantic dropout probability must be in [0,1].")
        self.dropout_probability = float(dropout_probability)
        self.null_token = nn.Parameter(torch.empty(1, 1, 768))
        nn.init.normal_(self.null_token, std=0.02)

    def null_like(self, z):
        if z.ndim != 3 or z.shape[-1] != 768:
            raise ValueError("PDAE v2 conditions must be patch tokens [B,T,768].")
        return self.null_token.to(z).expand_as(z)

    def forward(self, z, *, apply_dropout: bool, force_drop_mask=None):
        null = self.null_like(z)
        if force_drop_mask is not None:
            if force_drop_mask.shape != (z.shape[0],):
                raise ValueError("Semantic drop mask must have shape [B].")
            mask = force_drop_mask.to(device=z.device, dtype=torch.bool)
        else:
            mask = (torch.rand(z.shape[0], device=z.device) < self.dropout_probability
                    if apply_dropout else torch.zeros(z.shape[0], dtype=torch.bool, device=z.device))
        return torch.where(mask[:, None, None], null, z), mask


class TokenConditionedSiT(SemanticSiTWrapper):
    """Reuse SiT and LoRA helpers; replace vector adapters only in this experiment."""

    def __init__(self, transformer, *, num_heads=12, attention_lora=True,
                 lora_rank=64, lora_alpha=64., lora_dropout=0., lora_layers=None):
        # Do not construct the legacy z_proj/AdaLN adapters or alter native AdaLN.
        nn.Module.__init__(self)
        self.base = transformer.requires_grad_(False).eval()
        hidden = int(_module_config_value(transformer, "hidden_size", 768))
        depth = len(self.base.blocks)
        self.injection_layers = list(range(depth))
        for block in self.base.blocks:
            if not all(hasattr(block, name) for name in ("norm1", "attn", "norm2", "mlp", "adaLN_modulation")):
                raise ValueError("PDAE v2 requires the repository's SiT AdaLN transformer blocks.")
        self.feature_projector = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768),
        )
        self.image_attention = nn.ModuleList([
            ImageCrossAttention(hidden, 768, num_heads) for _ in range(depth)
        ])
        self._configure_attention_lora(
            depth, attention_lora, lora_rank, lora_alpha, lora_dropout, lora_layers,
        )
        self.architecture_spec = {
            "kind": "pdae_v2_cross_attention", "token_dim": 768, "hidden_size": hidden,
            "num_heads": num_heads, "layers": self.injection_layers,
            "projector": "layernorm_linear_gelu_linear_768",
            "padding_mask": "true_is_padding_empty_row_zero_residual",
            "lora": {"enabled": self.attention_lora_enabled, "rank": self.lora_rank,
                     "alpha": self.lora_alpha, "dropout": self.lora_dropout,
                     "layers": self.lora_layers, "targets": list(self.lora_targets)},
        }

    def forward(self, hidden_states, timestep, z, class_labels=None, force_drop_ids=None,
                return_dict=True, condition_padding_mask=None):
        validate_padding_mask(z, condition_padding_mask)
        if z.shape[0] != hidden_states.shape[0] or z.shape[-1] != 768:
            raise ValueError("PDAE v2 requires [B,T,768] image conditions matching the latent batch.")
        # Sanitize BEFORE projection so padding NaNs cannot contaminate weights/gradients.
        tokens = z if condition_padding_mask is None else z.masked_fill(condition_padding_mask[..., None], 0)
        x = self.base.x_embedder(hidden_states)
        pos = self.base.pos_embed
        x = x + (pos.unsqueeze(0) if pos.ndim == 2 else pos).to(x)
        tokens = self.feature_projector(tokens.to(x))  # shared, evaluated once per forward
        mask = condition_padding_mask.to(x.device) if condition_padding_mask is not None else None
        c = self.base.t_embedder(timestep)
        c = c + self._label_embedding(class_labels, len(x), x.device, force_drop_ids).to(c)
        x_base, x_sem = x, x

        def modulate(value, shift, scale):
            return value * (1 + scale[:, None]) + shift[:, None]

        for index, block in enumerate(self.base.blocks):
            x_base = block(x_base, c)
            # Preserve the native SiT modulation, self-attention, and FFN exactly.
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = block.adaLN_modulation(c).chunk(6, dim=1)
            with self._semantic_lora(index):
                x_sem = x_sem + gate_a[:, None] * block.attn(modulate(block.norm1(x_sem), shift_a, scale_a))
            x_sem = x_sem + self.image_attention[index](x_sem, tokens, mask)
            x_sem = x_sem + gate_m[:, None] * block.mlp(modulate(block.norm2(x_sem), shift_m, scale_m))
        base = self.base.unpatchify(self.base.final_layer(x_base, c))
        sample = self.base.unpatchify(self.base.final_layer(x_sem, c))
        if bool(_module_config_value(self.base, "learn_sigma", False)):
            base, _ = base.chunk(2, dim=1)
            sample, _ = sample.chunk(2, dim=1)
        output = PDAESiTOutput(sample=sample, z=z, base_sample=base, delta_sample=sample - base)
        return output if return_dict else (sample, z, base, output.delta_sample)

    def trainable_state_dict(self):
        return {
            "architecture": self.architecture_spec,
            "feature_projector": self.feature_projector.state_dict(),
            "image_attention": self.image_attention.state_dict(),
            "attention_lora": self._attention_lora_state_dict(),
        }

    def load_trainable_state_dict(self, state_dict, strict=True):
        if state_dict.get("architecture") != self.architecture_spec:
            raise ValueError("PDAE v2 conditioning/LoRA architecture mismatch.")
        self.feature_projector.load_state_dict(state_dict["feature_projector"], strict=strict)
        self.image_attention.load_state_dict(state_dict["image_attention"], strict=strict)
        self._load_attention_lora_state_dict(state_dict["attention_lora"], strict=strict)


class PDAEV2Branch(PDAESiTBranch):
    def __init__(self, encoder, semantic_transformer, *, image_size=256,
                 semantic_cfg_enabled=True, semantic_dropout_probability=.1):
        super().__init__(encoder, semantic_transformer, encoder_input_space="rgb",
                         encoder_image_size=image_size)
        if semantic_cfg_enabled:
            self.semantic_conditioner = LearnedTokenCondition(semantic_dropout_probability)

    def pdae_state_dict(self):
        # Encoder weights are immutable and large; save their verified identity.
        return {
            "format_version": 6, "encoder_architecture": self.encoder_architecture,
            "encoder_input": {"space": "rgb", "image_size": self.encoder_image_size},
            "frozen_encoder": self.encoder.snapshot_identity,
            "generator": self.generator_state_dict(),
        }

    def load_pdae_state_dict(self, state_dict, strict=True):
        expected = self.pdae_state_dict()
        for key in ("format_version", "encoder_architecture", "encoder_input", "frozen_encoder"):
            if state_dict.get(key) != expected[key]:
                raise ValueError(f"PDAE v2 checkpoint mismatch: {key}. Use the matching config and encoder snapshot.")
        self.load_generator_state_dict(state_dict["generator"], strict=strict)


def build_pdae_v2_branch(transformer, stage_config, *, project_root=None):
    from diffusion_ot.integrations.hf_snapshot import effective_project_root, resolve_project_local_path

    encoder = stage_config["encoder"]
    adapter = stage_config.get("adapter", {})
    semantic = stage_config.get("semantic_cfg", {})
    if encoder.get("input_space") != "rgb" or not encoder.get("frozen", True):
        raise ValueError("PDAE v2 requires original RGB and frozen SigLIP 2.")
    if int(encoder.get("input_channels", 3)) != 3 or int(encoder.get("token_dim", 768)) != 768:
        raise ValueError("PDAE v2 uses RGB input and native 768-D patch tokens.")
    if adapter.get("kind") != "pdae_v2_cross_attention" or not adapter.get("freeze_base", True):
        raise ValueError("PDAE v2 requires cross-attention and a frozen SiT base.")
    all_layers = list(range(len(transformer.blocks)))
    if adapter.get("injection_layers", all_layers) != all_layers:
        raise ValueError("PDAE v2 injects image cross-attention in every SiT block.")
    if semantic.get("null_condition", "learned_token") != "learned_token" or semantic.get("drop_granularity", "sample") != "sample":
        raise ValueError("PDAE v2 CFG requires sample dropout with a learned null token.")
    if (stage_config.get("refinement") or {}).get("enabled", False):
        raise ValueError("PDAE v2 currently supports the native flow objective only.")
    root = project_root or effective_project_root(stage_config, fallback=Path.cwd())
    directory = resolve_project_local_path(encoder["local_dir"], root, field_name="encoder.local_dir")
    return PDAEV2Branch(
        FrozenSiglipPatchEncoder.from_local(directory),
        TokenConditionedSiT(transformer, num_heads=int(adapter.get("cross_attention_heads", 12)),
            attention_lora=bool(adapter.get("lora", True)), lora_rank=int(adapter.get("lora_rank", 64)),
            lora_alpha=float(adapter.get("lora_alpha", 64)), lora_dropout=float(adapter.get("lora_dropout", 0)),
            lora_layers=adapter.get("lora_layers")),
        image_size=int(encoder.get("image_size", 256)),
        semantic_cfg_enabled=bool(semantic.get("enabled", True)),
        semantic_dropout_probability=float(semantic.get("dropout_probability", .1)),
    )
