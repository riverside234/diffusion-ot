"""RGB conditions in the real Stage 1A flow loop, with a tiny SiT substitute."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_decoded_translation import branch as latent_branch, TinyVAE
from test_stage1a_refinement import setup as latent_training_setup


def rgb_branch():
    value = latent_branch()
    value.encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 6), nn.LayerNorm(6))
    value.encoder_input_space = "rgb"
    value.encoder_image_size = 8
    return value


@pytest.fixture
def rgb_training_setup(latent_training_setup, monkeypatch):
    import diffusion_ot.data.ground_truth as ground_truth
    import diffusion_ot.integrations.sit_diffusers as sit
    import diffusion_ot.losses.pdae_flow as flow

    run, config, latest, root, _, _, source = latent_training_setup
    monkeypatch.setattr(ground_truth, "load_afhq_dataset", lambda *args, **kwargs: source)
    config["encoder"] = {"input_space": "rgb", "input_channels": 3, "image_size": 8}
    config["train"]["initialize_from"] = None
    config["refinement"] = {"enabled": False}
    torch.manual_seed(19)
    initial = rgb_branch()
    observed = {"encoder_inputs": [], "training_calls": [], "flow_targets": [], "flow_losses": []}

    def components(*args, **kwargs):
        value = deepcopy(initial)

        def encoder_input(module, args):
            observed["encoder_inputs"].append((value.training, args[0].detach().clone()))

        def training_input(module, args, kwargs):
            observed["training_calls"].append({key: kwargs[key].detach().clone()
                                                for key in ("x0_latent", "encoder_image", "x_t")})

        value.encoder.register_forward_pre_hook(encoder_input)
        value.register_forward_pre_hook(training_input, with_kwargs=True)
        vae = TinyVAE()

        def unexpected_vae(*args, **kwargs):
            pytest.fail("Flow-only RGB training must not substitute VAE reconstructions for original RGB.")

        vae.decode = unexpected_vae
        vae.encode = unexpected_vae
        latest.update(branch=value, vae=vae)
        return SimpleNamespace(transformer=value.semantic_transformer.base, vae=vae)

    monkeypatch.setattr(sit, "load_sit_components", components)
    original_target = flow.make_linear_flow_target

    def record_target(x0, **kwargs):
        result = original_target(x0, **kwargs)
        observed["flow_targets"].append((x0.detach().clone(), result))
        return result

    monkeypatch.setattr(flow, "make_linear_flow_target", record_target)
    original_loss = flow.pdae_velocity_gap_loss

    def record_loss(*args, **kwargs):
        loss = original_loss(*args, **kwargs)
        observed["flow_losses"].append(float(loss.detach()))
        return loss

    monkeypatch.setattr(flow, "pdae_velocity_gap_loss", record_loss)
    return run, config, latest, root, initial, observed


def test_rgb_flow_training_uses_original_images_and_latent_targets_and_resumes(rgb_training_setup):
    run, _, latest, _, initial, observed = rgb_training_setup
    first, first_logs, report = run("rgb_flow", steps=1)
    assert report.initial_step == 0 and report.final_step == 1
    assert first["train_state"]["initialization"] is None
    assert first["train_state"]["refinement"] is None
    assert first_logs["refinement"] == []
    assert len(observed["training_calls"]) == 2  # gradient accumulation, both use RGB
    for call in observed["training_calls"]:
        assert call["encoder_image"].shape == (4, 3, 8, 8)
        assert call["x0_latent"].shape == call["x_t"].shape == (4, 4, 8, 8)
        assert -1 <= call["encoder_image"].min() <= call["encoder_image"].max() <= 1
    assert {training for training, _ in observed["encoder_inputs"]} == {True, False}
    assert all(image.shape[1] == 3 for _, image in observed["encoder_inputs"])
    for x0, target in observed["flow_targets"]:
        assert x0.shape[1] == 4
        torch.testing.assert_close(target.target_v, x0 - target.noise)

    assert any(not torch.equal(value, initial.encoder.state_dict()[name])
               for name, value in first["model"]["encoder"].items())
    assert first_logs["train"][0]["encoder_grad_norm_pre_clip"] > 0
    assert first_logs["train"][0]["adapter_grad_norm_pre_clip"] > 0
    assert len(observed["flow_losses"]) == 2
    assert first_logs["train"][0]["loss"] == pytest.approx(sum(observed["flow_losses"]) / 2)
    assert "refinement" not in first_logs["train"][0]
    assert first_logs["validation"][0]["step"] == 0
    assert "refinement" not in first_logs["validation"][0]

    resumed, logs, report = run("rgb_flow", steps=3, resume=True)
    assert report.initial_step == 1 and report.final_step == 3
    assert resumed["ema"]["num_updates"] == 3
    assert [row["step"] for row in logs["train"]] == [1, 2, 3]
    assert [row["step"] for row in logs["validation"]] == [0, 1, 2, 3]
    assert latest["branch"].training


def test_rgb_training_validation_does_not_change_model_updates(rgb_training_setup):
    run, _, _, _, _, _ = rgb_training_setup
    with_val, _, _ = run("rgb_with_val", steps=2)
    without_val, _, _ = run("rgb_without_val", steps=2,
                            modify=lambda cfg: cfg["evaluation"].update(enabled=False))

    def identical(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                identical(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                identical(a, b)
        else:
            assert left == right

    for key in ("model", "optimizer", "ema", "rng_state", "dataloader_generator_state"):
        identical(with_val[key], without_val[key])


def test_rgb_z_dependence_encodes_rgb_but_scores_latent_velocity():
    from diffusion_ot.losses.pdae_flow import make_linear_flow_target
    from diffusion_ot.models.pdae_sit import make_null_class_labels
    from diffusion_ot.training.train_pdae_domain import _evaluate_z_dependence

    torch.manual_seed(25)
    value = rgb_branch()
    image = torch.rand(3, 3, 8, 8) * 2 - 1
    x0 = torch.randn(3, 4, 8, 8)
    seen = []
    handle = value.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    result = _evaluate_z_dependence(
        value, value.semantic_transformer.base, [{"x0_latent": x0, "encoder_image": image}],
        device="cpu", model_dtype=torch.float32, flow_direction="noise_to_data",
        time_eps=1e-5, null_label=None, step=0, num_batches=1, num_time_bins=2,
        seed=71, ema=None, use_ema=False,
    )
    handle.remove()
    assert value.training
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], image)
    rng = torch.Generator().manual_seed(71)
    noise = torch.randn(x0.shape, generator=rng)
    t = torch.rand(x0.shape[0], generator=rng).clamp(1e-5, 1 - 1e-5)
    target = make_linear_flow_target(x0, noise=noise, t=t)
    value.eval()
    with torch.no_grad():
        output = value.predict_with_z(
            x_t=target.x_t, timestep=t, z=value.encode(x0, encoder_image=image),
            class_labels=make_null_class_labels(value.semantic_transformer.base,
                                               batch_size=3, device=x0.device),
        )
        expected = (output.delta_sample - (x0 - noise - output.base_sample)).square().mean()
    assert result["correct_z"]["mse"] == pytest.approx(float(expected))
    assert result["num_samples"] == 3


def test_rgb_training_rejects_refinement_and_unknown_input_space_before_loading(rgb_training_setup):
    run, _, _, _, _, observed = rgb_training_setup
    with pytest.raises(ValueError, match="refinement"):
        run("rgb_refinement_rejected", steps=1, modify=lambda cfg: cfg["refinement"].update(enabled=True))
    with pytest.raises(ValueError, match="input_space"):
        run("unknown_input_rejected", steps=1, modify=lambda cfg: cfg["encoder"].update(input_space="pixels"))
    assert observed["training_calls"] == []
