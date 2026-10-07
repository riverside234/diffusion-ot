from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def validate_padding_mask(tokens: torch.Tensor, mask: torch.Tensor | None) -> None:
    if tokens.ndim != 3 or tokens.shape[1] == 0 or not tokens.is_floating_point():
        raise ValueError("Image conditions must be floating-point [B,T,D] with T > 0.")
    if mask is not None and (
        mask.dtype != torch.bool or mask.shape != tokens.shape[:2] or mask.device != tokens.device
    ):
        raise ValueError("condition_padding_mask must be bool [B,T] on the token device; True means padding.")


class ImageCrossAttention(nn.Module):
    """Independent pre-norm residual; DiT queries attend to unordered image tokens."""

    def __init__(self, hidden_size: int, token_dim: int = 768, num_heads: int = 12):
        super().__init__()
        if num_heads < 1 or hidden_size % num_heads:
            raise ValueError("Cross-attention hidden size must be divisible by num_heads.")
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(token_dim, hidden_size)
        self.v = nn.Linear(token_dim, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden, tokens, condition_padding_mask=None):
        validate_padding_mask(tokens, condition_padding_mask)
        if hidden.ndim != 3 or hidden.shape[0] != tokens.shape[0]:
            raise ValueError("DiT hidden states and image tokens must have matching batch sizes.")
        batch, queries, width = hidden.shape
        has_tokens = torch.ones(batch, dtype=torch.bool, device=hidden.device)
        attention_mask = None
        if condition_padding_mask is not None:
            tokens = tokens.masked_fill(condition_padding_mask[..., None], 0)
            valid = ~condition_padding_mask
            has_tokens = valid.any(-1)
            # Give empty rows one finite dummy key; zero their entire residual
            # below, including a possibly trained output bias.
            valid = valid.clone()
            valid[:, 0] |= ~has_tokens
            attention_mask = valid[:, None, None, :]

        def heads(value):
            return value.reshape(batch, -1, self.num_heads, width // self.num_heads).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            heads(self.q(self.norm(hidden))), heads(self.k(tokens)), heads(self.v(tokens)),
            attn_mask=attention_mask, dropout_p=0.0,
        ).transpose(1, 2).reshape(batch, queries, width)
        return self.out(attended).masked_fill(~has_tokens[:, None, None], 0)
