"""Geometry, attention, gradients and architecture-safe encoder checkpoints."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from diffusion_ot.models.pdae_sit import build_pdae_sit_branch
from diffusion_ot.models.residual_encoder import PDAEResidualLatentEncoder, EncoderSpatialAttention
from test_pdae_sit_shapes import FakeSiT


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return {
        "encoder": {"kind": "residual_cnn_v1", "input_space": "latent", "input_channels": 4,
                    "input_size": 8, "channels": [8, 16, 32, 32], "blocks_per_stage": [2, 2, 2, 2],
                    "spatial_size": 1, "num_groups": 4, "z_dim": 12,
                    "attention_resolutions": [4], "attention_heads": 4, "dropout": 0.0},
        "adapter": {"injection_layers": [0, 1], "bottleneck_dim": 4, "freeze_base": True,
                    "lora": True, "lora_rank": 2, "lora_alpha": 2, "lora_layers": [0, 1]},
        "semantic_cfg": {"enabled": True, "dropout_probability": 0.1},
    }


def test_proposed_encoder_geometry_and_gradients_and_batch_independence():
    torch.manual_seed(941)
    encoder = PDAEResidualLatentEncoder().eval()
    value = torch.randn(2, 4, 32, 32, requires_grad=True)
    maps = encoder.forward_spatial_features(value, [0, 1, 2, 3])
    assert [tuple(x.shape[1:]) for x in maps] == [
        (64, 32, 32), (128, 16, 16), (256, 8, 8), (256, 4, 4),
    ]
    attended_shapes = []
    handle = encoder.stages[1][-1].register_forward_pre_hook(
        lambda _, args: attended_shapes.append(tuple(args[0].shape)))
    codes = encoder(value)
    handle.remove()
    assert attended_shapes == [(2, 128, 16, 16)]
    assert codes.shape == (2, 512)
    torch.testing.assert_close(encoder(value[:1]), codes[:1], rtol=1e-4, atol=1e-5)
    (codes * torch.randn_like(codes)).mean().backward()
    assert value.grad is not None and value.grad.abs().sum() > 0 and torch.isfinite(value.grad).all()
    for parameter in encoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert encoder.stem.weight.grad.abs().sum() > 0
    assert encoder.stages[1][-1].qkv.weight.grad.abs().sum() > 0
    assert encoder.stages[1][0].shortcut.weight.grad.abs().sum() > 0


def test_latent_stem_receives_per_image_channel_offsets_without_normalization():
    encoder = PDAEResidualLatentEncoder()
    seen = []
    handle = encoder.stem.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    original = torch.randn(2, 4, 32, 32)
    offsets = torch.tensor([[1., 2., -3., 4.], [-2., 3., 4., -1.]]).view(2, 4, 1, 1)
    with torch.no_grad():
        encoder(original + offsets)
    handle.remove()
    torch.testing.assert_close(seen[0], original + offsets, rtol=0, atol=0)
    assert encoder.stem.stride == (1, 1)


def test_spatial_attention_matches_explicit_multihead_reference():
    torch.manual_seed(943)
    block = EncoderSpatialAttention(8, num_groups=4, num_heads=2).double()
    value = torch.randn(2, 8, 3, 3, dtype=torch.float64)
    q, k, v = block.qkv(block.norm(value)).chunk(3, dim=1)
    heads = []
    for head in range(2):
        query = q[:, head * 4:(head + 1) * 4].flatten(2).transpose(1, 2)
        key = k[:, head * 4:(head + 1) * 4].flatten(2)
        content = v[:, head * 4:(head + 1) * 4].flatten(2).transpose(1, 2)
        weights = (query @ key / 2).softmax(dim=-1)
        heads.append((weights @ content).transpose(1, 2).reshape(2, 4, 3, 3))
    expected = value + block.proj(torch.cat(heads, dim=1))
    torch.testing.assert_close(block(value), expected, rtol=1e-10, atol=1e-10)


def test_spatial_features_return_requested_order_and_propagate_input_gradients():
    config = tiny_config()["encoder"]
    encoder = build_pdae_sit_branch(FakeSiT(), stage_config={"encoder": config}).encoder
    value = torch.randn(2, 4, 8, 8, requires_grad=True)
    maps = encoder.forward_spatial_features(value, [2, 0])
    assert [tuple(x.shape[1:]) for x in maps] == [(32, 2, 2), (8, 8, 8)]
    sum(x.square().mean() for x in maps).backward()
    assert value.grad.abs().sum() > 0
    for layers in [[], [0, 0], [-1], [4], [True]]:
        with pytest.raises(ValueError, match="residual-stage"):
            encoder.forward_spatial_features(value, layers)
    with pytest.raises(ValueError, match="expects"):
        encoder(torch.randn(1, 4, 16, 16))


def test_residual_checkpoint_roundtrip_preserves_codes_and_architecture():
    torch.manual_seed(944)
    branch = build_pdae_sit_branch(FakeSiT(), stage_config=tiny_config()).eval()
    saved = deepcopy(branch.pdae_state_dict())
    assert saved["format_version"] == 5
    assert saved["encoder_architecture"] == branch.encoder_architecture
    restored = build_pdae_sit_branch(FakeSiT(), stage_config=tiny_config()).eval()
    restored.load_pdae_state_dict(saved)
    value = torch.randn(2, 4, 8, 8)
    torch.testing.assert_close(restored.encode(value), branch.encode(value), rtol=0, atol=0)
    for name, parameter in restored.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, dict(branch.named_parameters())[name], rtol=0, atol=0)


@pytest.mark.parametrize("change", [{"num_groups": 2}, {"attention_heads": 2}, {"dropout": .1}])
def test_equal_shape_architecture_changes_are_rejected_before_loading(change):
    config = tiny_config()
    original = build_pdae_sit_branch(FakeSiT(), stage_config=config)
    saved = deepcopy(original.pdae_state_dict())
    config["encoder"].update(change)
    target = build_pdae_sit_branch(FakeSiT(), stage_config=config)
    # These changes deliberately have identical state-dict keys and tensor sizes.
    assert {k: v.shape for k, v in saved["encoder"].items()} == {
        k: v.shape for k, v in target.encoder.state_dict().items()}
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="architecture mismatch"):
        target.load_pdae_state_dict(saved)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_plain_and_residual_checkpoints_are_not_interchangeable():
    config = tiny_config()
    residual = build_pdae_sit_branch(FakeSiT(), stage_config=config)
    config["encoder"].pop("kind")
    plain = build_pdae_sit_branch(FakeSiT(), stage_config=config)
    assert plain.pdae_state_dict()["format_version"] == 4
    for target, source in [(plain, residual), (residual, plain)]:
        with pytest.raises(ValueError, match="architecture mismatch"):
            target.load_pdae_state_dict(source.pdae_state_dict())
    missing = deepcopy(residual.pdae_state_dict())
    missing.pop("encoder_architecture")
    with pytest.raises(ValueError, match="architecture mismatch"):
        residual.load_pdae_state_dict(missing)


@pytest.mark.parametrize("options", [
    {"blocks_per_stage": [2, 2]}, {"blocks_per_stage": [2, 0, 2, 2]},
    {"input_size": 31}, {"spatial_size": 8}, {"attention_resolutions": [12]},
    {"attention_resolutions": [16, 16]}, {"attention_heads": 3},
    {"num_groups": 0}, {"dropout": float("nan")}, {"dropout": 1.0},
])
def test_invalid_residual_architecture_fails_early(options):
    with pytest.raises(ValueError):
        PDAEResidualLatentEncoder(**options)


def test_builder_rejects_unknown_architecture_and_rgb_residual_selection():
    for encoder in [{"kind": "typo"}, {"kind": "residual_cnn_v1", "input_space": "rgb"}]:
        with pytest.raises(ValueError, match="encoder.kind"):
            build_pdae_sit_branch(FakeSiT(), stage_config={"encoder": encoder})


@pytest.mark.parametrize("domain", ["cat", "dog"])
def test_residual_recipes_build_proposed_model_and_preserve_flow_training_settings(domain):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / f"configs/stage1a_pdae/{domain}_sit_b2_lora_residual.yaml").read_text())
    old = yaml.safe_load((root / f"configs/stage1a_pdae/{domain}_sit_b2_lora.yaml").read_text())
    for key in old.keys() - {"encoder", "output_dir"}:
        assert config[key] == old[key]
    assert config["output_dir"] == f"outputs/stage1a_{domain}_rescnn"
    assert not config["train"].get("initialize_from") and not config["train"].get("resume_from")
    assert not config.get("refinement", {}).get("enabled", False)
    config["adapter"].update(injection_layers=[0, 1], lora_layers=[0, 1])
    branch = build_pdae_sit_branch(FakeSiT(), stage_config=config)
    assert isinstance(branch.encoder, PDAEResidualLatentEncoder)
    assert branch.encoder_architecture == PDAEResidualLatentEncoder().architecture_spec
    assert branch.encoder_input_space == "latent"
    assert isinstance(branch.encoder.z_norm, nn.LayerNorm)
