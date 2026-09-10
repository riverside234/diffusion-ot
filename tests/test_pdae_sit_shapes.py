from __future__ import annotations

from copy import deepcopy
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


class FakeAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj((q + k + v) / 3.0)


class FakeBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = FakeAttention(hidden_size)
        self.token_proj = nn.Linear(hidden_size, hidden_size)
        self.cond_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, c):
        return torch.tanh(
            self.token_proj(x) + self.attn(x) + self.cond_proj(c)[:, None, :]
        )


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


def test_branch_can_predict_from_supplied_z_and_reload_trainable_state():
    from diffusion_ot.models.pdae_sit import (
        PDAELatentEncoder,
        PDAESiTBranch,
        SemanticSiTWrapper,
        make_null_class_labels,
    )

    torch.manual_seed(1)
    base = FakeSiT()
    encoder = PDAELatentEncoder(
        input_channels=4,
        channels=[8, 16],
        z_dim=16,
        spatial_size=2,
        num_groups=4,
    )
    wrapper = SemanticSiTWrapper(base, z_dim=16, injection_layers=[0, 1], bottleneck_dim=4)
    branch = PDAESiTBranch(encoder, wrapper)
    saved_state = deepcopy(branch.pdae_state_dict())

    x0 = torch.randn(2, 4, 8, 8)
    x_t = torch.randn_like(x0)
    timestep = torch.rand(2)
    labels = make_null_class_labels(base, batch_size=2, device=x0.device)
    z = branch.encode(x0)
    output = branch.predict_with_z(x_t, timestep, z, class_labels=labels)
    assert output.sample.shape == x0.shape

    with torch.no_grad():
        next(branch.encoder.parameters()).add_(1.0)
    branch.load_pdae_state_dict(saved_state)

    for name, value in branch.pdae_state_dict()["encoder"].items():
        torch.testing.assert_close(value, saved_state["encoder"][name])


def test_attention_lora_is_zero_initialized_and_only_changes_semantic_path():
    from diffusion_ot.models.pdae_sit import (
        SemanticLoRALinear,
        SemanticSiTWrapper,
        make_null_class_labels,
    )

    torch.manual_seed(2)
    base = FakeSiT()
    reference = deepcopy(base)
    wrapper = SemanticSiTWrapper(
        base,
        z_dim=16,
        injection_layers=[0, 1],
        bottleneck_dim=4,
        attention_lora=True,
        lora_rank=2,
        lora_alpha=2,
        lora_layers=[0, 1],
    )
    x_t = torch.randn(2, 4, 8, 8)
    timestep = torch.rand(2)
    labels = make_null_class_labels(base, batch_size=2, device=x_t.device)
    z = torch.randn(2, 16)

    with torch.no_grad():
        expected = reference(
            hidden_states=x_t,
            timestep=timestep,
            class_labels=labels,
        ).sample
        initial = wrapper(x_t, timestep, z, class_labels=labels)

    torch.testing.assert_close(initial.base_sample, expected, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(initial.sample, expected, atol=1.0e-6, rtol=1.0e-6)
    qkv_lora = base.blocks[0].attn.qkv
    assert isinstance(qkv_lora, SemanticLoRALinear)
    assert not any(parameter.requires_grad for parameter in qkv_lora.base_layer.parameters())
    assert all(parameter.requires_grad for parameter in qkv_lora.lora_down.parameters())
    assert all(parameter.requires_grad for parameter in qkv_lora.lora_up.parameters())

    with torch.no_grad():
        qkv_lora.lora_up.weight.fill_(0.05)
    updated = wrapper(x_t, timestep, z, class_labels=labels)

    torch.testing.assert_close(updated.base_sample, expected, atol=1.0e-6, rtol=1.0e-6)
    assert not torch.allclose(updated.sample, expected)
    updated.delta_sample.square().mean().backward()
    assert qkv_lora.lora_up.weight.grad is not None
    assert torch.isfinite(qkv_lora.lora_up.weight.grad).all()


def test_attention_lora_checkpoint_roundtrip_and_configuration_guard():
    from diffusion_ot.models.pdae_sit import SemanticSiTWrapper

    source = SemanticSiTWrapper(
        FakeSiT(),
        z_dim=16,
        injection_layers=[0, 1],
        bottleneck_dim=4,
        attention_lora=True,
        lora_rank=2,
        lora_alpha=4,
        lora_layers=[0, 1],
    )
    with torch.no_grad():
        source.base.blocks[1].attn.proj.lora_up.weight.fill_(0.125)
    state = deepcopy(source.trainable_state_dict())

    restored = SemanticSiTWrapper(
        FakeSiT(),
        z_dim=16,
        injection_layers=[0, 1],
        bottleneck_dim=4,
        attention_lora=True,
        lora_rank=2,
        lora_alpha=4,
        lora_layers=[0, 1],
    )
    restored.load_trainable_state_dict(state)
    torch.testing.assert_close(
        restored.base.blocks[1].attn.proj.lora_up.weight,
        source.base.blocks[1].attn.proj.lora_up.weight,
    )

    without_lora = SemanticSiTWrapper(
        FakeSiT(),
        z_dim=16,
        injection_layers=[0, 1],
        bottleneck_dim=4,
    )
    with pytest.raises(ValueError, match="attention-LoRA setting"):
        without_lora.load_trainable_state_dict(state)


def test_stage1a_optimizer_keeps_lora_in_a_separate_learning_rate_group():
    from diffusion_ot.models.pdae_sit import (
        PDAELatentEncoder,
        PDAESiTBranch,
        SemanticSiTWrapper,
    )
    from diffusion_ot.training.train_pdae_domain import _optimizer_groups

    encoder = PDAELatentEncoder(
        input_channels=4,
        channels=[8, 16],
        z_dim=16,
        spatial_size=2,
        num_groups=4,
    )
    wrapper = SemanticSiTWrapper(
        FakeSiT(),
        z_dim=16,
        injection_layers=[0, 1],
        bottleneck_dim=4,
        attention_lora=True,
        lora_rank=2,
        lora_layers=[0, 1],
    )
    branch = PDAESiTBranch(encoder, wrapper)
    groups = _optimizer_groups(
        branch,
        {"lr_encoder": 1.0e-4, "lr_adapter": 1.0e-4, "lr_lora": 5.0e-5},
    )

    assert [group["group_name"] for group in groups] == ["encoder", "adapter", "lora"]
    assert [group["lr"] for group in groups] == [1.0e-4, 1.0e-4, 5.0e-5]
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    trainable_ids = [id(parameter) for parameter in branch.parameters() if parameter.requires_grad]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)


def test_build_branch_enables_attention_lora_from_stage_config():
    from diffusion_ot.models.pdae_sit import (
        SemanticLoRALinear,
        build_pdae_sit_branch,
    )

    branch = build_pdae_sit_branch(
        FakeSiT(),
        model_config={"latent_channels": 4},
        stage_config={
            "encoder": {
                "channels": [8, 16],
                "spatial_size": 2,
                "z_dim": 16,
                "num_groups": 4,
            },
            "adapter": {
                "injection_layers": [0, 1],
                "bottleneck_dim": 4,
                "lora": True,
                "lora_rank": 2,
                "lora_alpha": 2,
                "lora_dropout": 0.0,
                "lora_layers": [0, 1],
            },
        },
    )

    assert branch.semantic_transformer.attention_lora_enabled
    assert isinstance(branch.semantic_transformer.base.blocks[0].attn.qkv, SemanticLoRALinear)


def test_attention_lora_rejects_an_unfrozen_base():
    from diffusion_ot.models.pdae_sit import SemanticSiTWrapper

    with pytest.raises(ValueError, match="freeze_base=true"):
        SemanticSiTWrapper(
            FakeSiT(),
            z_dim=16,
            injection_layers=[0, 1],
            attention_lora=True,
            freeze_base=False,
        )


def test_learned_semantic_condition_drops_individual_samples_and_gets_gradients():
    from diffusion_ot.models.pdae_sit import LearnedSemanticCondition

    condition = LearnedSemanticCondition(z_dim=3, dropout_probability=0.1)
    with torch.no_grad():
        condition.null_token.copy_(torch.tensor([2.0, 3.0, 4.0]))
    z = torch.ones(3, 3, requires_grad=True)
    dropped, mask = condition(
        z,
        apply_dropout=True,
        force_drop_mask=torch.tensor([True, False, True]),
    )

    assert mask.tolist() == [True, False, True]
    torch.testing.assert_close(dropped[0], condition.null_token)
    torch.testing.assert_close(dropped[1], z[1])
    torch.testing.assert_close(dropped[2], condition.null_token)
    dropped.sum().backward()
    torch.testing.assert_close(z.grad[0], torch.zeros(3))
    torch.testing.assert_close(z.grad[1], torch.ones(3))
    torch.testing.assert_close(condition.null_token.grad, torch.full((3,), 2.0))


class LinearSemanticTransformer(nn.Module):
    def forward(
        self,
        hidden_states,
        timestep,
        z,
        class_labels=None,
        force_drop_ids=None,
        return_dict=True,
    ):
        from diffusion_ot.models.pdae_sit import PDAESiTOutput

        base = torch.full_like(hidden_states, 2.0)
        delta = z[:, :1, None, None].expand_as(hidden_states)
        return PDAESiTOutput(
            sample=base + delta,
            z=z,
            base_sample=base,
            delta_sample=delta,
        )

    def trainable_state_dict(self):
        return self.state_dict()

    def load_trainable_state_dict(self, state_dict, strict=True):
        self.load_state_dict(state_dict, strict=strict)


def test_semantic_cfg_scale_zero_one_and_two_follow_learned_null_formula():
    from diffusion_ot.models.pdae_sit import PDAESiTBranch

    branch = PDAESiTBranch(
        nn.Identity(),
        LinearSemanticTransformer(),
        z_dim=2,
        semantic_cfg_enabled=True,
    )
    with torch.no_grad():
        branch.semantic_conditioner.null_token.copy_(torch.tensor([1.0, 0.0]))
    x_t = torch.zeros(2, 1, 2, 2)
    timestep = torch.zeros(2)
    z = torch.tensor([[4.0, 0.0], [4.0, 0.0]])

    scale_zero = branch.predict_cfg_with_z(x_t, timestep, z, guidance_scale=0.0)
    scale_one = branch.predict_cfg_with_z(x_t, timestep, z, guidance_scale=1.0)
    scale_two = branch.predict_cfg_with_z(x_t, timestep, z, guidance_scale=2.0)

    torch.testing.assert_close(scale_zero.sample, torch.full_like(x_t, 3.0))
    torch.testing.assert_close(scale_one.sample, torch.full_like(x_t, 6.0))
    torch.testing.assert_close(scale_two.sample, torch.full_like(x_t, 9.0))


def test_cfg_checkpoint_roundtrip_includes_null_token_and_rejects_legacy_state():
    from diffusion_ot.models.pdae_sit import PDAESiTBranch

    branch = PDAESiTBranch(
        nn.Linear(2, 2),
        LinearSemanticTransformer(),
        z_dim=2,
        semantic_cfg_enabled=True,
    )
    with torch.no_grad():
        branch.semantic_conditioner.null_token.fill_(3.5)
    state = deepcopy(branch.pdae_state_dict())
    assert "semantic_conditioner" in state["generator"]

    restored = PDAESiTBranch(
        nn.Linear(2, 2),
        LinearSemanticTransformer(),
        z_dim=2,
        semantic_cfg_enabled=True,
    )
    restored.load_pdae_state_dict(state)
    torch.testing.assert_close(
        restored.semantic_conditioner.null_token,
        branch.semantic_conditioner.null_token,
    )

    legacy = {
        "encoder": branch.encoder.state_dict(),
        "semantic_transformer": branch.semantic_transformer.trainable_state_dict(),
    }
    with pytest.raises(ValueError, match="no learned null token"):
        restored.load_pdae_state_dict(legacy)
