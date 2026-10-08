from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusion_ot.data.pdae_v2_augmentation import MatchedRGBAugmentation, augmentation_options


def options(**updates):
    return augmentation_options({"encoder": {"kind": "siglip2_vit_b16", "input_space": "rgb"},
                                 "augmentation": {"enabled": True, **updates}})


class EncoderVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(scaling_factor=.25)
        self.inputs = []

    def encode(self, image):
        self.inputs.append(image.detach().clone())
        mean = torch.cat([image, image.mean(1, keepdim=True)], dim=1) * self.weight
        return SimpleNamespace(latent_dist=SimpleNamespace(
            mean=mean, sample=lambda generator=None: mean + .2))


def batch():
    return {"encoder_image": torch.rand(5, 3, 16, 16) * 2 - 1,
            "x0_latent": torch.zeros(5, 4, 16, 16), "horizontal_flip": [False] * 5,
            "metadata": [{}] * 5}


@pytest.mark.parametrize("posterior_mean", [True, False])
def test_transformed_condition_is_exact_vae_input_and_target(posterior_mean):
    source = batch()
    value = MatchedRGBAugmentation(options(horizontal_flip_probability=1., vae_batch_size=2,
        color_jitter={"probability": 1., "hue": .02}, affine={"probability": 1.}),
        use_posterior_mean=posterior_mean)
    vae = EncoderVAE().train()
    result, stats = value.prepare_batch(source, vae, device="cpu", dtype=torch.float32)
    assert [len(v) for v in vae.inputs] == [2, 2, 1]
    torch.testing.assert_close(torch.cat(vae.inputs), result["encoder_image"], atol=0, rtol=0)
    rgb = result["encoder_image"]
    expected = torch.cat([rgb, rgb.mean(1, keepdim=True)], 1)
    expected = (expected + (0 if posterior_mean else .2)) * .25
    torch.testing.assert_close(result["x0_latent"], expected)
    assert stats == {"samples": 5, "horizontal_flip": 5, "color_jitter": 5, "affine": 5}
    assert not vae.training and not vae.weight.requires_grad and vae.weight.grad is None
    assert not result["x0_latent"].requires_grad
    assert -1 <= rgb.min() <= rgb.max() <= 1 and torch.isfinite(rgb).all()
    assert not torch.equal(source["encoder_image"], result["encoder_image"])
    assert not source["x0_latent"].any()  # Input dictionary was not mutated.


def test_flip_is_exact_and_rng_restore_repeats_transforms():
    rgb = batch()["encoder_image"]
    flip = MatchedRGBAugmentation(options(horizontal_flip_probability=1.,
        color_jitter={"probability": 0.}, affine={"probability": 0.}), use_posterior_mean=True)
    torch.testing.assert_close(flip.transform(rgb)[0], rgb.flip(-1), atol=1e-7, rtol=0)
    random = MatchedRGBAugmentation(options(), use_posterior_mean=True)
    state = torch.get_rng_state()
    first = random.transform(rgb)
    torch.set_rng_state(state)
    second = random.transform(rgb)
    torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
    assert first[1] == second[1]


def test_affine_reflection_does_not_introduce_black_borders():
    value = MatchedRGBAugmentation(options(horizontal_flip_probability=0.,
        color_jitter={"probability": 0.}, affine={"probability": 1., "translate": [.05, .05],
                                                  "scale": [.95, 1.05]}), use_posterior_mean=True)
    result, _ = value.transform(torch.full((32, 3, 32, 32), .6))
    torch.testing.assert_close(result, torch.full_like(result, .6), atol=1e-6, rtol=0)


def test_invalid_inputs_and_posterior_mismatch_fail_clearly():
    assert augmentation_options({}) is None
    with pytest.raises(ValueError, match="PDAE v2"):
        augmentation_options({"augmentation": {"enabled": True}})
    with pytest.raises(ValueError, match="probabilities"):
        options(horizontal_flip_probability=float("nan"))
    value = MatchedRGBAugmentation(options(), use_posterior_mean=True)
    source = batch()
    source["horizontal_flip"][0] = True
    with pytest.raises(ValueError, match="cached-latent flips"):
        value.prepare_batch(source, EncoderVAE(), device="cpu", dtype=torch.float32)
    source["horizontal_flip"][0] = False
    source["metadata"] = [{"posterior_statistic": "sample"}]
    with pytest.raises(ValueError, match="posterior policy"):
        value.prepare_batch(source, EncoderVAE(), device="cpu", dtype=torch.float32)
