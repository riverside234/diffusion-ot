"""PatchNCE configuration and checkpoint-compatible PDAE spatial extraction."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from diffusion_ot.models.pdae_sit import PDAELatentEncoder
from diffusion_ot.training.decoded_translation import self_supervised_translation_options


def options():
    return {"objective": "patchnce", "source_contrastive_weight": 0.,
            "patchnce": {"weight": .15, "temperature": .2, "num_patches": 64, "layers": [0, 1, 2]}}


@pytest.mark.parametrize("field,value", [
    ("weight", 0), ("weight", float("nan")), ("temperature", -1),
    ("temperature", float("inf")), ("num_patches", True), ("num_patches", 1),
    ("num_patches", 3.5), ("layers", []), ("layers", [0, 0]),
    ("layers", [True]), ("layers", [-1]), ("layers", [0, "1"]),
    ("unknown_option", 1),
])
def test_invalid_patch_options_are_rejected(field, value):
    image = options()
    image["patchnce"][field] = value
    with pytest.raises(ValueError):
        self_supervised_translation_options(image)


def test_objectives_are_mutually_exclusive_and_old_default_is_unchanged():
    image = options()
    before = deepcopy(image)
    resolved = self_supervised_translation_options(image)
    assert image == before
    assert resolved["name"] == "patchnce" and resolved["weight"] == .15
    image["source_contrastive_weight"] = .15
    with pytest.raises(ValueError, match="replaces"):
        self_supervised_translation_options(image)
    image["objective"] = "source_infonce"
    with pytest.raises(ValueError, match="PatchNCE settings require"):
        self_supervised_translation_options(image)
    image["objective"] = "typo"
    with pytest.raises(ValueError, match="objective"):
        self_supervised_translation_options(image)
    assert self_supervised_translation_options({"source_contrastive_weight": .15}) == {
        "objective": "source_infonce", "name": "source_contrastive", "readout": "target",
        "weight": .15, "temperature": .2, "negative_similarity_threshold": .95}


def test_spatial_features_match_convolution_stages_without_changing_native_encoder():
    torch.manual_seed(8)
    encoder = PDAELatentEncoder(channels=(8, 16, 32), z_dim=12, spatial_size=4, num_groups=2)
    state = deepcopy(encoder.state_dict())
    values = torch.randn(2, 4, 32, 32, requires_grad=True)
    native = encoder(values)
    expected, hooks = [], []
    for module in encoder.conv:
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(lambda module, inputs, output: expected.append(output)))
    encoder(values)
    for hook in hooks:
        hook.remove()
    spatial = encoder.forward_spatial_features(values, [2, 0, 1])
    assert encoder.spatial_feature_channels == (8, 16, 32)
    for actual, index in zip(spatial, (2, 0, 1)):
        torch.testing.assert_close(actual, expected[index], rtol=0, atol=0)
    assert [x.shape[-2:] for x in spatial] == [(4, 4), (16, 16), (8, 8)]
    torch.testing.assert_close(encoder(values), native, rtol=0, atol=0)
    assert set(encoder.state_dict()) == set(state)
    for key, value in state.items():
        torch.testing.assert_close(encoder.state_dict()[key], value, rtol=0, atol=0)
    sum(x.square().mean() for x in spatial).backward()
    assert values.grad.norm() > 0
    assert all(p.grad is None for p in encoder.proj.parameters())
    assert any(p.grad is not None and p.grad.norm() > 0 for p in encoder.conv.parameters())


@pytest.mark.parametrize("layers", [[], [0, 0], [-1], [3], [True], [0, .5]])
def test_spatial_extractor_rejects_invalid_stages(layers):
    encoder = PDAELatentEncoder(channels=(8, 16), z_dim=6)
    with pytest.raises(ValueError):
        encoder.forward_spatial_features(torch.randn(2, 4, 8, 8), layers)
