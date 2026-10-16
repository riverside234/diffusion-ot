"""Augmented RGB must produce the same latent for the CNN and flow target."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from diffusion_ot.data.pdae_v2_augmentation import augmentation_options
from pdae_cnn import data, initialize, objective
from pdae_cnn.eval.validation import validate
from pdae_cnn.train_step import train_step
from test_pdae_sit_shapes import FakeSiT
from test_pdae_v2_augmentation import EncoderVAE
from test_residual_encoder import tiny_config, small_cpu_thread_pool


def config():
    value = tiny_config()
    value.update(
        project_root=".", data_config="unused.yaml", domain="cat", device="cpu",
        dataloader={"batch_size": 2, "gradient_accumulation_steps": 2,
                    "num_workers": 0, "pin_memory": False, "random_horizontal_flip": 1.},
        train={"seed": 91, "lr": .003}, ema={"decay": .9, "warmup_steps": 0},
        evaluation={"enabled": True, "num_batches": 1, "save_images": False},
        augmentation={"enabled": True, "horizontal_flip_probability": 1., "vae_batch_size": 2,
                      "color_jitter": {"probability": 0.}, "affine": {"probability": 0.}},
    )
    value["semantic_cfg"]["dropout_probability"] = 0.
    return value


@pytest.mark.parametrize("kind", ["plain_cnn_v1", "residual_cnn_v1"])
@pytest.mark.parametrize("posterior_mean", [True, False])
def test_training_matches_augmented_vae_target_and_keeps_validation_cached(
        tmp_path, monkeypatch, kind, posterior_mean):
    options = config()
    options["encoder"]["kind"] = kind
    options["project_root"] = str(tmp_path)
    model = tmp_path / "model.yaml"
    model.write_text("pretrained: pretrained.yaml\nlatent_cache:\n  use_posterior_mean: "
                     + str(posterior_mean).lower() + "\n", encoding="utf-8")
    options["model_config"] = str(model)
    vae = EncoderVAE()
    monkeypatch.setattr(initialize, "load_sit_components", lambda *args, **kwargs:
                        SimpleNamespace(transformer=FakeSiT(), vae=vae))
    state = initialize.build_training_state(options)
    assert state.augmentation.use_posterior_mean is posterior_mean
    source = {"encoder_image": torch.linspace(-.9, .9, 4 * 3 * 8 * 8).reshape(4, 3, 8, 8),
              "x0_latent": torch.zeros(4, 4, 8, 8), "horizontal_flip": [False] * 4,
              "metadata": [{"posterior_statistic": "mean" if posterior_mean else "sample"}] * 4}
    rgb = source["encoder_image"].flip(-1)
    expected = torch.cat([rgb, rgb.mean(1, keepdim=True)], dim=1)
    expected = (expected + (0. if posterior_mean else .2)) * vae.config.scaling_factor
    seen_cnn, seen_target = [], []
    handle = state.branch.encoder.register_forward_pre_hook(
        lambda _, args: seen_cnn.append(args[0].detach().clone()))
    original_target = objective.prepare_flow_target

    def record_target(x0, *args, **kwargs):
        seen_target.append(x0.detach().clone())
        return original_target(x0, *args, **kwargs)

    monkeypatch.setattr(objective, "prepare_flow_target", record_target)
    before = {name: value.detach().clone() for name, value in state.branch.encoder.named_parameters()}
    batches = iter([source] * 4)
    for step in (1, 2):
        assert torch.isfinite(train_step(state, batches, options, step=step)["loss"])
    handle.remove()
    assert len(seen_cnn) == len(seen_target) == 4  # Two accumulated batches per update.
    for cnn_input, flow_target in zip(seen_cnn, seen_target):
        torch.testing.assert_close(cnn_input, expected)
        torch.testing.assert_close(flow_target, expected)
    torch.testing.assert_close(torch.cat(vae.inputs), rgb.repeat(4, 1, 1, 1))
    assert not source["x0_latent"].any()
    assert not vae.training and all(not p.requires_grad and p.grad is None for p in vae.parameters())
    assert any(not torch.equal(before[name], p) for name, p in state.branch.encoder.named_parameters())
    assert state.ema.num_updates == 2

    vae.inputs.clear()
    loader = torch.utils.data.DataLoader([{"x0_latent": x} for x in source["x0_latent"]], batch_size=2)
    metrics = validate(state, loader, options)
    assert metrics["num_samples"] == 2 and not vae.inputs
    assert state.branch.training


@pytest.mark.parametrize("enabled", [True, False])
def test_loaders_use_source_rgb_only_for_augmented_training(monkeypatch, enabled):
    class Dataset:
        def __init__(self, **kwargs):
            self.options = kwargs

        def __len__(self):
            return 4

    monkeypatch.setattr(data, "CachedLatentDataset", Dataset)
    options = config()
    options["augmentation"]["enabled"] = enabled
    training, validation = data.build_loaders(options)
    assert training.dataset.options["include_original_images"] is enabled
    assert training.dataset.options["random_horizontal_flip"] == (0. if enabled else 1.)
    assert validation.dataset.options["include_original_images"] is False
    assert validation.dataset.options["random_horizontal_flip"] == 0.


def test_cached_fallback_and_missing_vae():
    batch = {"x0_latent": torch.randn(2, 4, 8, 8)}
    torch.testing.assert_close(data.prepare_batch(batch, device="cpu", dtype=torch.float32)["x0_latent"],
                               batch["x0_latent"], atol=0, rtol=0)
    with pytest.raises(ValueError, match="pretrained VAE"):
        data.prepare_batch(batch, device="cpu", dtype=torch.float32, augmentation=object())
    options = config()
    options["augmentation"]["enabled"] = False
    assert augmentation_options(options) is None
