from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


class FakePatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class FakeTEmbedder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, timestep):
        return self.proj(timestep.float().view(-1, 1))


class FakeYEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels, train: bool, force_drop_ids=None):
        return self.embedding_table(labels)


class FakeBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.token_proj = nn.Linear(hidden_size, hidden_size)
        self.cond_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, c):
        return torch.tanh(self.token_proj(x) + self.cond_proj(c)[:, None, :])


class FakeFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_dim)
        self.cond_proj = nn.Linear(hidden_size, patch_dim, bias=False)

    def forward(self, x, c):
        return self.linear(x) + self.cond_proj(c)[:, None, :]


class FakeSiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            depth=2,
            in_channels=4,
            num_classes=10,
            patch_size=2,
            learn_sigma=True,
        )
        self.learn_sigma = True
        self.in_channels = 4
        self.out_channels = 8
        self.x_embedder = FakePatchEmbed(in_channels=4, hidden_size=8, patch_size=2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, 8))
        self.t_embedder = FakeTEmbedder(hidden_size=8)
        self.y_embedder = FakeYEmbedder(num_classes=10, hidden_size=8)
        self.blocks = nn.ModuleList([FakeBlock(8), FakeBlock(8)])
        self.final_layer = FakeFinalLayer(hidden_size=8, patch_dim=2 * 2 * self.out_channels)

    def unpatchify(self, x):
        batch, tokens, channels = x.shape
        patch = 2
        side = int(tokens**0.5)
        x = x.reshape(batch, side, side, patch, patch, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(batch, self.out_channels, side * patch, side * patch)

    def forward(self, hidden_states, timestep, class_labels, force_drop_ids=None, return_dict=True):
        x = self.x_embedder(hidden_states) + self.pos_embed
        c = self.t_embedder(timestep)
        c = c + self.y_embedder(class_labels, False, force_drop_ids=force_drop_ids)
        for block in self.blocks:
            x = block(x, c)
        x = self.unpatchify(self.final_layer(x, c))
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return SimpleNamespace(sample=x)


def test_lightweight_encoder_outputs_z_shape():
    from diffusion_ot.models.pdae_sit import PDAELatentEncoder

    encoder = PDAELatentEncoder(input_channels=4, channels=[16, 32, 64], z_dim=32, num_groups=4)
    z = encoder(torch.randn(2, 4, 32, 32))

    assert z.shape == (2, 32)


def test_zero_initialized_semantic_wrapper_matches_base_at_start():
    from diffusion_ot.models.pdae_sit import SemanticSiTWrapper, make_null_class_labels

    torch.manual_seed(0)
    base = FakeSiT()
    wrapper = SemanticSiTWrapper(base, z_dim=32, injection_layers=[0, 1], bottleneck_dim=4)
    x_t = torch.randn(2, 4, 8, 8)
    timestep = torch.rand(2)
    labels = make_null_class_labels(base, batch_size=2, device=x_t.device)
    z = torch.randn(2, 32)

    with torch.no_grad():
        expected = base(hidden_states=x_t, timestep=timestep, class_labels=labels).sample
    output = wrapper(hidden_states=x_t, timestep=timestep, z=z, class_labels=labels)

    torch.testing.assert_close(output.sample, expected, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(output.delta_sample, torch.zeros_like(output.delta_sample), atol=1.0e-6, rtol=0.0)
    assert not any(parameter.requires_grad for parameter in base.parameters())
