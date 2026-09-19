from copy import deepcopy

import pytest
import torch

from diffusion_ot.losses.semantic_prior import validate_prior_resume
from diffusion_ot.training.decoded_translation import DecoderTraining, validate_decoder_config
from diffusion_ot.training.diffaugment import random_translation
from test_decoded_translation import config, domains, tiny_features


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_translation_is_zero_filled_differentiable_and_uses_private_rng(dtype):
    # A non-square image with distinct pixels exposes wraparound/axis mistakes.
    images = torch.arange(1, 1 + 4 * 3 * 5 * 7, dtype=torch.float32).reshape(4, 3, 5, 7).to(dtype)
    images.requires_grad_(True)
    generator = torch.Generator().manual_seed(73)
    offsets = torch.Generator().manual_seed(73)
    dy = torch.randint(-1, 2, (4,), generator=offsets)
    dx = torch.randint(-2, 3, (4,), generator=offsets)
    expected, expected_grad = torch.zeros_like(images), torch.zeros_like(images)
    for b in range(4):
        for y in range(5):
            for x in range(7):
                sy, sx = y + int(dy[b]), x + int(dx[b])
                if 0 <= sy < 5 and 0 <= sx < 7:
                    expected[b, :, y, x] = images[b, :, sy, sx].detach()
                    expected_grad[b, :, sy, sx] = 1
    global_rng = torch.get_rng_state().clone()
    translated = random_translation(images, ratio=.25, generator=generator)
    torch.testing.assert_close(translated, expected, atol=0, rtol=0)
    assert translated.dtype == dtype and translated.is_contiguous()
    assert (translated == 0).any()
    state = generator.get_state().clone()
    translated.float().sum().backward()
    torch.testing.assert_close(images.grad, expected_grad, atol=0, rtol=0)
    assert torch.equal(generator.get_state(), state)
    assert torch.equal(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("ratio", [0, .001])
def test_zero_pixel_shift_is_identity_without_random_draws(ratio):
    images = torch.ones(2, 3, 8, 8, requires_grad=True)
    generator = torch.Generator().manual_seed(3)
    state = generator.get_state().clone()
    assert random_translation(images, ratio=ratio, generator=generator) is images
    assert torch.equal(generator.get_state(), state)


@pytest.mark.parametrize("options", [
    "translation", {"enabled": "false"}, {"policy": "color"},
    {"translation_ratio": -1}, {"translation_ratio": .51},
    {"translation_ratio": float("nan")}, {"translation_ratio": float("inf")},
    {"translation_ratio": True},
])
def test_invalid_augmentation_config_fails_early(options):
    cfg = config()
    cfg["decoded_translation"]["diffaugment"] = options
    with pytest.raises(ValueError, match="decoded_translation.diffaugment"):
        validate_decoder_config(cfg)


def make_runtime(monkeypatch, tmp_path, *, enabled=True):
    import diffusion_ot.training.decoded_translation as module
    monkeypatch.setattr(module, "load_image_features", tiny_features)
    torch.manual_seed(29)
    cfg = config()
    cfg["decoded_translation"].update(
        structure_contrastive_weight=.01, code_consistency_weight=.02,
        code_consistency_mode="contrastive",
        diffaugment={"enabled": enabled, "policy": "translation", "translation_ratio": .125})
    validate_decoder_config(cfg)
    return DecoderTraining(cfg, domains(), None, tmp_path, seed=18)


def inputs():
    generator = torch.Generator().manual_seed(5)
    refs = {d: torch.randn(5, 6, generator=generator, requires_grad=True) for d in ("cat", "dog")}
    weights = {d: torch.softmax(torch.randn(4, 5, generator=generator), -1)
               for d in ("cat_to_dog", "dog_to_cat")}
    latents = {d: torch.randn(4, 4, 8, 8, generator=generator) for d in refs}
    structures = {"source_query_structure": {d: torch.randn(4, 12, generator=generator) for d in refs},
                  "source_reference_structure": {d: torch.randn(7, 12, generator=generator) for d in refs}}
    return (weights, refs, latents, latents), structures


def test_adversarial_only_augmentation_keeps_structure_and_code_targets_and_gradients(monkeypatch, tmp_path):
    augmented = make_runtime(monkeypatch, tmp_path)
    control = make_runtime(monkeypatch, tmp_path, enabled=False)
    args, kwargs = inputs()
    seen = []
    def capture(module, values):
        seen.append(values[0])
        if values[0].requires_grad:
            values[0].retain_grad()
    hook = augmented.features.register_forward_pre_hook(capture)
    loss, metrics = augmented.loss(*args, step=2, **kwargs)
    hook.remove()
    # Feature calls per direction: original fake, augmented fake, original
    # source, augmented real. Only fake paths have image gradients.
    assert len(seen) == 8
    for i in (0, 4):
        assert seen[i].requires_grad and seen[i + 1].requires_grad
        assert not seen[i + 2].requires_grad and not seen[i + 3].requires_grad
        assert (seen[i] > 0).all() and (seen[i + 2] > 0).all()
        assert (seen[i + 1] == 0).any() and (seen[i + 3] == 0).any()
    _, baseline = control.loss(*args, step=2, **kwargs)
    for name in ("structure_loss", "structure_contrastive_loss", "code_consistency_loss"):
        assert metrics[name] == pytest.approx(baseline[name], abs=1e-7)
    assert metrics["adversarial_augmentation"]["applied"]
    assert not baseline["adversarial_augmentation"]["applied"]
    states = {d: g.get_state().clone() for d, g in augmented.augmentation_generators.items()}
    discriminator = deepcopy(augmented.discriminators.state_dict())
    loss.backward()
    # These two tensors participate exclusively in adversarial supervision.
    assert all(seen[i].grad is not None and seen[i].grad.norm() > 0 for i in (1, 5))
    assert all(z.grad is not None and z.grad.norm() > 0 for z in args[1].values())
    assert all(sum(p.grad.norm().item() for p in view.parameters() if p.grad is not None) > 0
               for view in augmented.views.values())
    assert all(p.grad is None for p in augmented.features.parameters())
    assert all(p.grad is None for p in augmented.discriminators.parameters())
    assert all(torch.equal(g.get_state(), states[d]) for d, g in augmented.augmentation_generators.items())
    for name, value in discriminator.items():
        torch.testing.assert_close(augmented.discriminators.state_dict()[name], value, atol=0, rtol=0)
    assert all(torch.equal(g.get_state(), control.noise_generators[d].get_state())
               for d, g in augmented.noise_generators.items())


def test_fixed_validation_is_unaugmented_and_preserves_all_rng_and_discriminator_state(monkeypatch, tmp_path):
    runtime = make_runtime(monkeypatch, tmp_path)
    args, kwargs = inputs()
    saved = deepcopy(runtime.checkpoint_state())
    global_rng = torch.get_rng_state().clone()
    calls = []
    hook = runtime.features.register_forward_pre_hook(lambda module, values: calls.append(values[0].detach()))
    with torch.no_grad():
        first = runtime.loss(*args, step=2, validation_seed=21, **kwargs)[1]
        second = runtime.loss(*args, step=2, validation_seed=21, **kwargs)[1]
    hook.remove()
    assert first == second
    assert not first["adversarial_augmentation"]["applied"]
    assert first["adversarial_augmentation"]["enabled"]
    assert len(calls) == 12 and all((x > 0).all() for x in calls)
    assert runtime.discriminator_updates == 0
    assert torch.equal(torch.get_rng_state(), global_rng)
    current = runtime.checkpoint_state()
    for key in ("decoded_noise_states", "decoded_augmentation_states", "decoded_discriminators"):
        for domain, state in saved[key].items():
            torch.testing.assert_close(current[key][domain], state, atol=0, rtol=0)


def test_resume_replays_next_augmented_update_exactly_and_requires_saved_rng(monkeypatch, tmp_path):
    runtime = make_runtime(monkeypatch, tmp_path)
    args, kwargs = inputs()
    runtime.loss(*args, step=2, **kwargs)
    saved = deepcopy(runtime.checkpoint_state())
    saved.update(config=deepcopy(runtime.config), format_version=4)
    expected_loss, expected_metrics = runtime.loss(*args, step=3, **kwargs)
    expected_state = deepcopy(runtime.checkpoint_state())
    resumed = make_runtime(monkeypatch, tmp_path)
    resumed.load_checkpoint(saved)
    loss, metrics = resumed.loss(*args, step=3, **kwargs)
    torch.testing.assert_close(loss, expected_loss, atol=0, rtol=0)
    assert metrics == expected_metrics
    for key in ("decoded_noise_states", "decoded_augmentation_states", "decoded_discriminators"):
        for domain, state in expected_state[key].items():
            torch.testing.assert_close(resumed.checkpoint_state()[key][domain], state, atol=0, rtol=0)
    del saved["decoded_augmentation_states"]
    with pytest.raises(ValueError, match="DiffAugment resume requires"):
        resumed.load_checkpoint(saved)
    # Legacy disabled configurations do not require augmentation state.
    legacy = make_runtime(monkeypatch, tmp_path, enabled=False)
    del saved["config"]["decoded_translation"]["diffaugment"]
    legacy.load_checkpoint(saved)
    assert "decoded_augmentation_states" not in legacy.checkpoint_state()


def test_resume_rejects_a_changed_augmentation_objective():
    cfg = config()
    cfg["decoded_translation"]["diffaugment"] = {"enabled": True, "translation_ratio": .0625}
    validate_prior_resume(cfg, deepcopy(cfg))
    changed = deepcopy(cfg)
    changed["decoded_translation"]["diffaugment"]["translation_ratio"] = .125
    with pytest.raises(ValueError, match="Resume cannot change decoded_translation"):
        validate_prior_resume(cfg, changed)
