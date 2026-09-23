from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from test_decoded_translation import config as external_config, domains as base_domains
from diffusion_ot.training.self_supervised_translation import (
    SelfSupervisedDecoderTraining,
    encode_generated_images,
    source_code_contrastive_loss,
)


class RoundtripVAE(nn.Module):
    def __init__(self, scale=.3):
        super().__init__()
        self.decoder = nn.Conv2d(4, 3, 1)
        self.encoder = nn.Conv2d(3, 4, 1)
        self.config = SimpleNamespace(scaling_factor=scale)
        self.encoded_rgb = []

    def decode(self, values):
        return SimpleNamespace(sample=self.decoder(values).tanh())

    def encode(self, values):
        self.encoded_rgb.append(values.detach().clone())
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=self.encoder(values)))


def config():
    result = external_config()
    result["generator_adaptation"]["null_preservation_weight"] = 0
    result["decoded_translation"] = {
        "enabled": True, "supervision": "self_supervised", "batch_size": 2,
        "num_steps": 2, "validation_num_steps": 3, "warmup_steps": 2,
        "source_contrastive_weight": .1, "source_contrastive_temperature": .2,
        "source_contrastive_negative_similarity_threshold": .95,
        "source_contrastive_readout": "target",
    }
    return result


def domains():
    contexts = base_domains()
    for context in contexts.values():
        context.vae = RoundtripVAE()
    return contexts


def batch():
    references = {d: torch.randn(5, 6, requires_grad=True) for d in ("cat", "dog")}
    query_codes = {d: torch.randn(3, 6, requires_grad=True) for d in references}
    logits = {f"{s}_to_{t}": torch.randn(3, 5, requires_grad=True)
              for s, t in (("cat", "dog"), ("dog", "cat"))}
    weights = {d: F.softmax(value, -1) for d, value in logits.items()}
    latents = {d: torch.randn(3, 4, 8, 8) for d in references}
    return weights, references, query_codes, logits, latents


def test_vae_rgb_roundtrip_uses_scaled_posterior_mean_with_live_input_gradient():
    torch.manual_seed(1)
    vae = RoundtripVAE().requires_grad_(False)
    rgb = torch.rand(2, 3, 8, 8, requires_grad=True)
    output = encode_generated_images(vae, rgb)
    expected = vae.encoder(rgb * 2 - 1) * vae.config.scaling_factor
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(vae.encoded_rgb[0], rgb.detach() * 2 - 1)
    output.square().mean().backward()
    assert rgb.grad.norm() > 0
    assert all(p.grad is None for p in vae.parameters())


def test_source_retrieval_detaches_keys_masks_duplicates_and_reports_ties():
    keys = torch.eye(3, requires_grad=True)
    bank = torch.cat((keys, keys[:1]), 0)
    recovered = torch.eye(3).requires_grad_()
    loss, metrics = source_code_contrastive_loss(recovered, keys, bank)
    loss.backward()
    assert recovered.grad.norm() > 0
    assert keys.grad is None
    assert metrics["usable_samples"] == 3
    assert metrics["retrieval_top1"] == 1
    assert metrics["matched_cosine"] == pytest.approx(1)
    assert metrics["negative_bank_size"] == 4
    shuffled, _ = source_code_contrastive_loss(recovered.roll(1, 0), keys, bank)
    assert shuffled > loss
    _, tied = source_code_contrastive_loss(torch.ones(3, 3), keys, keys)
    assert tied["retrieval_top1"] == pytest.approx(1 / 3)
    assert tied["retrieval_top1"] == pytest.approx(tied["retrieval_chance"])
    empty, skipped = source_code_contrastive_loss(recovered[:1], keys[:1], keys[:1])
    assert empty == 0
    assert skipped["usable_samples"] == 0
    assert skipped["retrieval_top1"] is None


def test_full_translation_uses_own_live_target_encoder_and_no_external_supervision(monkeypatch, tmp_path):
    import diffusion_ot.training.decoded_translation as original

    def forbidden(*args, **kwargs):
        raise AssertionError("External supervision must never be loaded")

    monkeypatch.setattr(original, "load_image_features", forbidden)
    monkeypatch.setattr(original, "RGBPatchDiscriminator", forbidden)
    monkeypatch.setattr(original, "FeatureDiscriminator", forbidden)
    torch.manual_seed(5)
    ctx = domains()
    runtime = SelfSupervisedDecoderTraining(config(), ctx, None, tmp_path, seed=5)
    monkeypatch.setattr(runtime, "original_source_images", forbidden)
    weights, refs, keys, logits, latents = batch()
    loss, metrics = runtime.loss(weights, refs, latents, latents, step=2, source_query_codes=keys)
    loss.backward()
    assert metrics["supervision"] == "self_supervised"
    assert metrics["cat_to_dog"]["readout_domain"] == "dog"
    assert metrics["dog_to_cat"]["readout_domain"] == "cat"
    assert metrics["cat_to_dog"]["source_contrastive"]["negative_bank_size"] == 8
    assert all(value.grad is None for value in keys.values())
    assert all(value.grad is not None and value.grad.norm() > 0 for value in logits.values())
    assert all(value.grad is not None and value.grad.norm() > 0 for value in refs.values())
    for domain, context in ctx.items():
        assert any(p.grad is not None and p.grad.norm() > 0 for p in context.branch.encoder.parameters())
        assert any(p.grad is not None and p.grad.norm() > 0 for p in runtime.views[domain]["adapters"].parameters())
        assert any(p.grad is not None and p.grad.norm() > 0 for p in runtime.views[domain]["lora"].parameters())
        assert all(p.grad is None for p in context.vae.parameters())
        assert context.vae.encoded_rgb and context.vae.encoded_rgb[0].shape == (2, 3, 8, 8)
    assert set(runtime.image_objectives) == {"source_contrastive"}
    assert runtime.code_consistency_objective == 0
    assert not hasattr(runtime, "features") and not hasattr(runtime, "discriminators")


def test_fixed_validation_is_repeatable_and_does_not_consume_training_noise(tmp_path):
    torch.manual_seed(7)
    runtime = SelfSupervisedDecoderTraining(config(), domains(), None, tmp_path, seed=7)
    weights, refs, keys, _, latents = batch()
    rng = torch.get_rng_state().clone()
    noise = {d: value.get_state().clone() for d, value in runtime.noise_generators.items()}
    with torch.no_grad():
        first = runtime.loss(weights, refs, latents, latents, step=0, validation_seed=8, source_query_codes=keys)
        second = runtime.loss(weights, refs, latents, latents, step=2, validation_seed=8, source_query_codes=keys)
    torch.testing.assert_close(first[0], second[0])
    assert first[1] == second[1]
    assert first[1]["ramp"] == 1
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for d, value in runtime.noise_generators.items():
        torch.testing.assert_close(value.get_state(), noise[d])


def test_resume_restores_generator_ema_and_noise_without_discriminator_state(tmp_path):
    torch.manual_seed(9)
    runtime = SelfSupervisedDecoderTraining(config(), domains(), None, tmp_path, seed=9)
    for value in runtime.noise_generators.values():
        torch.randn(11, generator=value)
    with torch.no_grad():
        for value in runtime.parameters:
            value.add_(.05)
    runtime.ema.update(runtime.views)
    state = deepcopy(runtime.checkpoint_state())
    state.update(config=config(), format_version=4)
    assert not any("discriminator" in key or "augmentation" in key for key in state)
    restored = SelfSupervisedDecoderTraining(config(), domains(), None, tmp_path, seed=90)
    restored.load_checkpoint(state)
    for d in runtime.domains:
        for actual, expected in zip(restored.views[d].parameters(), runtime.views[d].parameters()):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.randn(11, generator=restored.noise_generators[d]),
                                   torch.randn(11, generator=runtime.noise_generators[d]))
    assert restored.ema.num_updates == runtime.ema.num_updates
    for field, value in (("decoded_supervision", "dino"), ("decoded_source_contrastive_readout", "source")):
        invalid = {**state, field: value}
        with pytest.raises(ValueError, match="different decoded supervision"):
            restored.load_checkpoint(invalid)


@pytest.mark.parametrize("option", ["perceptual_weight", "adversarial_weight", "structure_weight", "code_consistency_weight"])
def test_self_supervised_mode_rejects_external_or_old_recovery_objectives(option, tmp_path):
    cfg = config()
    cfg["decoded_translation"][option] = .01
    with pytest.raises(ValueError, match=option):
        SelfSupervisedDecoderTraining(cfg, domains(), None, tmp_path, seed=1)


def test_source_readout_control_and_missing_query_code_fallback(tmp_path):
    torch.manual_seed(11)
    cfg = config()
    cfg["decoded_translation"]["source_contrastive_readout"] = "source"
    runtime = SelfSupervisedDecoderTraining(cfg, domains(), None, tmp_path, seed=1)
    weights, refs, _, _, latents = batch()
    loss, metrics = runtime.loss(weights, refs, latents, latents, step=2)
    assert loss.isfinite()
    assert metrics["cat_to_dog"]["readout_domain"] == "cat"
    assert metrics["dog_to_cat"]["readout_domain"] == "dog"


def test_validation_grid_pairs_original_rgb_and_generated_target(monkeypatch, tmp_path):
    torch.manual_seed(13)
    cfg = config()
    cfg["decoded_translation"]["save_validation_images"] = True
    runtime = SelfSupervisedDecoderTraining(cfg, domains(), None, tmp_path, seed=1)
    weights, refs, keys, _, latents = batch()
    metadata = {d: [{"sample_id": f"{d}_{i}"} for i in range(3)] for d in ("cat", "dog")}
    seen = []
    def originals(domain, records):
        seen.append((domain, records))
        return torch.zeros(len(records), 3, 8, 8)
    monkeypatch.setattr(runtime, "original_source_images", originals)
    with torch.no_grad():
        _, metrics = runtime.loss(weights, refs, latents, latents, step=0, validation_seed=12,
            source_query_codes=keys, source_query_metadata=metadata, validation_image_dir=tmp_path)
    assert seen == [("cat", metadata["cat"][:2]), ("dog", metadata["dog"][:2])]
    for direction in ("cat_to_dog", "dog_to_cat"):
        assert (tmp_path / f"{direction}.png").is_file()
        assert metrics[direction]["validation_grid"] == str(tmp_path / f"{direction}.png")
