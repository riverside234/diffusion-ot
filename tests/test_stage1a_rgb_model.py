"""RGB semantic conditioning must remain separate from the latent SiT flow."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from diffusion_ot.models.pdae_sit import build_pdae_sit_branch
from test_pdae_sit_shapes import FakeSiT


def _branch(space="rgb", image_size=32):
    encoder = dict(channels=[8, 16, 32], spatial_size=2, z_dim=12, num_groups=4)
    if space is not None:
        encoder.update(input_space=space, input_channels=3 if space == "rgb" else 4)
    if space == "rgb":
        encoder["image_size"] = image_size
    return build_pdae_sit_branch(FakeSiT(), stage_config={
        "encoder": encoder,
        "adapter": {"injection_layers": [0, 1], "bottleneck_dim": 4, "freeze_base": True},
    })


def test_rgb_forward_conditions_on_original_image_while_sit_receives_latents():
    torch.manual_seed(711)
    branch = _branch().eval()
    latent = torch.randn(2, 4, 8, 8)
    noisy_latent = torch.randn_like(latent)
    rgb = torch.rand(2, 3, 32, 32) * 2 - 1
    encoder_inputs, sit_inputs = [], []
    encoder_hook = branch.encoder.register_forward_pre_hook(lambda _, args: encoder_inputs.append(args[0]))
    sit_hook = branch.semantic_transformer.base.x_embedder.register_forward_pre_hook(
        lambda _, args: sit_inputs.append(args[0]))
    try:
        result = branch(latent, noisy_latent, torch.rand(2), class_labels=torch.full((2,), 10),
                        encoder_image=rgb)
    finally:
        encoder_hook.remove()
        sit_hook.remove()
    assert encoder_inputs == [rgb]
    assert sit_inputs
    for actual in sit_inputs:
        torch.testing.assert_close(actual, noisy_latent)
        assert actual.shape[1] == 4
    assert result.sample.shape == latent.shape
    assert result.z.shape == (2, 12)
    torch.testing.assert_close(result.z, branch.encoder(rgb))
    # Changing the diffusion target alone must not change the RGB semantic code.
    torch.testing.assert_close(branch.encode(latent, encoder_image=rgb),
                               branch.encode(latent + 10, encoder_image=rgb), rtol=0, atol=0)


def test_rgb_stem_preserves_color_channel_means_before_learned_feature_processing():
    encoder = _branch().encoder
    stem = encoder.conv[0]
    assert isinstance(stem, nn.Conv2d)
    assert stem.in_channels == 3
    # Select each input channel with the central kernel coefficient. This exposes
    # whether input normalization or an activation erased RGB offsets beforehand.
    with torch.no_grad():
        stem.weight.zero_()
        stem.bias.zero_()
        for channel in range(3):
            stem.weight[channel, channel, 1, 1] = 1
    colors = torch.tensor([-.75, .125, .5]).view(1, 3, 1, 1)
    rgb = colors.expand(2, 3, 32, 32).clone()
    features = encoder.forward_spatial_features(rgb, [0])[0]
    torch.testing.assert_close(features[:, :3], colors.expand(2, 3, 16, 16), rtol=0, atol=0)
    shifted = encoder.forward_spatial_features(rgb + .125, [0])[0]
    torch.testing.assert_close(shifted[:, :3] - features[:, :3],
                               torch.full_like(features[:, :3], .125), rtol=0, atol=0)


def test_flow_gradient_reaches_original_rgb_and_encoder_after_adapters_activate():
    torch.manual_seed(712)
    branch = _branch().eval()
    # AdaLN-Zero intentionally blocks conditioning gradients at initialization.
    # Activate its conditioning maps to test the path used after learning starts.
    with torch.no_grad():
        for adapter in branch.semantic_transformer.adapters.values():
            adapter.cond_proj[-1].weight.normal_(std=.03)
        branch.semantic_transformer.final_adapter.cond_proj[-1].weight.normal_(std=.03)
    latent = torch.randn(2, 4, 8, 8, requires_grad=True)
    rgb = (torch.rand(2, 3, 32, 32) * 2 - 1).requires_grad_()
    output = branch(latent, torch.randn_like(latent), torch.rand(2),
                    class_labels=torch.full((2,), 10), encoder_image=rgb)
    output.sample.square().mean().backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all() and rgb.grad.abs().sum() > 0
    assert latent.grad is None  # x0 serves only as the latent-batch contract here.
    for parameter in branch.encoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert sum(p.grad.abs().sum() for p in branch.encoder.parameters()) > 0
    assert all(not p.requires_grad and p.grad is None for p in branch.semantic_transformer.base.parameters())


@pytest.mark.parametrize("bad_image", [None, torch.randn(1, 3, 32, 32),
    torch.randn(2, 4, 32, 32), torch.randn(2, 3, 16, 16),
    torch.zeros(2, 3, 32, 32, dtype=torch.uint8)])
def test_rgb_encoder_rejects_missing_or_incompatible_image_inputs(bad_image):
    with pytest.raises(ValueError, match="RGB|encoder_image"):
        _branch().encode(torch.randn(2, 4, 8, 8), encoder_image=bad_image)


def test_default_latent_encoder_preserves_legacy_stem_and_input_contract():
    branch = _branch(space=None)
    assert branch.encoder_input_space == "latent" and branch.encoder_image_size is None
    assert isinstance(branch.encoder.conv[0], nn.GroupNorm)
    assert isinstance(branch.encoder.conv[1], nn.SiLU)
    assert isinstance(branch.encoder.conv[2], nn.Conv2d)
    assert branch.encoder.conv[2].in_channels == 4
    assert "conv.2.weight" in branch.encoder.state_dict()
    latent = torch.randn(2, 4, 8, 8)
    torch.testing.assert_close(branch.encode(latent), branch.encoder(latent), rtol=0, atol=0)
    with pytest.raises(ValueError, match="latent semantic encoder does not accept"):
        branch.encode(latent, encoder_image=torch.randn(2, 3, 32, 32))


@pytest.mark.parametrize("space", ["rgb", "latent"])
def test_checkpoint_format_four_records_input_contract_and_roundtrips(space):
    branch = _branch(space)
    saved = deepcopy(branch.pdae_state_dict())
    assert saved["format_version"] == 4
    assert saved["encoder_input"] == {"space": space, "image_size": 32 if space == "rgb" else None}
    restored = _branch(space)
    restored.load_pdae_state_dict(saved)
    for key, parameter in branch.state_dict().items():
        # Frozen pretrained backbone weights are intentionally external.
        if key.startswith("encoder.") or key.startswith("semantic_transformer.z_proj."):
            torch.testing.assert_close(restored.state_dict()[key], parameter, rtol=0, atol=0)


def test_old_checkpoint_without_input_metadata_still_loads_into_latent_branch():
    original = _branch("latent")
    saved = deepcopy(original.pdae_state_dict())
    saved.pop("encoder_input")
    saved["format_version"] = 3
    restored = _branch(space=None)
    restored.load_pdae_state_dict(saved)
    latent = torch.randn(2, 4, 8, 8)
    torch.testing.assert_close(restored.encode(latent), original.encode(latent), rtol=0, atol=0)


@pytest.mark.parametrize("change", ["input_space", "image_size", "legacy_missing_metadata"])
def test_incompatible_checkpoint_input_contract_fails_before_parameter_mutation(change):
    branch = _branch("rgb")
    original = {key: value.clone() for key, value in branch.state_dict().items()}
    saved = deepcopy(branch.pdae_state_dict())
    # A load occurring before validation would overwrite this live parameter.
    saved["encoder"]["proj.bias"].add_(123)
    if change == "input_space":
        saved["encoder_input"] = {"space": "latent", "image_size": None}
    elif change == "image_size":
        saved["encoder_input"]["image_size"] = 64
    else:
        saved.pop("encoder_input")
        saved["format_version"] = 3
    with pytest.raises(ValueError, match="Semantic encoder input mismatch"):
        branch.load_pdae_state_dict(saved)
    for key, value in branch.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


@pytest.mark.parametrize("domain", ["cat", "dog"])
def test_separate_flow_only_rgb_recipes_have_full_resolution_encoder_and_fresh_outputs(domain):
    root = Path(__file__).resolve().parents[1]
    recipe = yaml.safe_load((root / f"configs/stage1a_pdae/{domain}_sit_b2_lora_rgb.yaml").read_text())
    assert recipe["domain"] == domain
    assert recipe["encoder"]["input_space"] == "rgb"
    assert recipe["encoder"]["input_channels"] == 3
    assert recipe["encoder"]["image_size"] == 256
    assert recipe["encoder"]["channels"] == [64, 128, 256, 256, 256, 256]
    assert recipe["encoder"]["spatial_size"] == 4
    assert recipe["encoder"]["z_dim"] == 512
    assert recipe["output_dir"] == f"outputs/stage1a_{domain}_rgb_lora"
    assert recipe["train"]["lr_encoder"] == pytest.approx(1e-4)
    assert recipe["train"]["lr_adapter"] == pytest.approx(1e-4)
    assert recipe["train"]["lr_lora"] == pytest.approx(2.5e-5)
    assert recipe["train"]["max_steps"] == 50000
    assert not recipe["train"].get("initialize_from") and not recipe["train"].get("resume_from")
    assert not recipe.get("refinement", {}).get("enabled", False)
    # Build the actual encoder with a tiny frozen SiT; only its domain encoder
    # is executed here, so this checks real 256->4 geometry without GPU weights.
    config = deepcopy(recipe)
    config["adapter"].update(injection_layers=[0, 1], lora_layers=[0, 1])
    branch = build_pdae_sit_branch(FakeSiT(), stage_config=config)
    with torch.no_grad():
        maps = branch.encoder.forward_spatial_features(torch.randn(1, 3, 256, 256), range(6))
        code = branch.encoder(torch.randn(1, 3, 256, 256))
    assert [tuple(value.shape[-2:]) for value in maps] == [(128, 128), (64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]
    assert code.shape == (1, 512)


@pytest.mark.parametrize("domain", ["cat", "dog"])
def test_original_main_recipes_retain_the_latent_encoder_and_legacy_outputs(domain):
    root = Path(__file__).resolve().parents[1]
    recipe = yaml.safe_load((root / f"configs/stage1a_pdae/{domain}_sit_b2_lora.yaml").read_text())
    assert recipe["encoder"].get("input_space", "latent") == "latent"
    assert recipe["encoder"]["input_channels"] == 4
    assert recipe["encoder"]["channels"] == [64, 128, 256]
    assert recipe["output_dir"] == f"outputs/stage1a_{domain}_sit_b2_cfg_adaln_all_lora_r64"
