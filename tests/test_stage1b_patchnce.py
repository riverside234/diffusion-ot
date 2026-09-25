"""CPU integration coverage for the isolated own-encoder PatchNCE experiment."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from diffusion_ot.models.generator_adaptation import generator_parameter_view
from diffusion_ot.models.pdae_sit import PDAELatentEncoder
from diffusion_ot.training.self_supervised_translation import SelfSupervisedDecoderTraining
from test_self_supervised_translation import batch, config as runtime_config, domains
from test_stage1b_extensions import assert_finite_numbers, assert_tensor_tree_equal, experiment
from test_stage1b_self_supervised import (
    assert_no_external_metrics,
    own_encoder_run,
    self_supervised_recipe,
)


def patchnce_options(options):
    options.update(objective="patchnce", source_contrastive_weight=0.,
                   source_contrastive_readout="target",
                   patchnce={"weight": .15, "temperature": .2, "num_patches": 8, "layers": [0]})


def patchnce_recipe(config):
    self_supervised_recipe(config)
    patchnce_options(config["decoded_translation"])


def patchnce_runtime(tmp_path, seed=11, *, mlp=False):
    config = runtime_config()
    patchnce_options(config["decoded_translation"])
    if mlp:
        config["decoded_translation"]["patchnce"].update(sampler="mlp_sample", projection_dim=16)
    contexts = domains()
    for context in contexts.values():
        context.branch.encoder = PDAELatentEncoder(channels=(8,), z_dim=6, spatial_size=2, num_groups=2)
    return SelfSupervisedDecoderTraining(config, contexts, None, tmp_path, seed=seed)


@pytest.fixture
def patchnce_training(own_encoder_run):
    # The existing fixture uses a flattened linear encoder. Use the production
    # convolutional encoder so the spatial path and its gradients are real.
    _, originals, _, _ = own_encoder_run
    for context in originals.values():
        context.branch.encoder = PDAELatentEncoder(channels=(8,), z_dim=6, spatial_size=2, num_groups=2)
    return own_encoder_run


def assert_no_global_retrieval_metrics(record):
    if isinstance(record, dict):
        assert not {"source_contrastive_loss", "source_contrastive", "source_negative_bank_ids"}.intersection(record)
        for value in record.values():
            assert_no_global_retrieval_metrics(value)
    elif isinstance(record, list):
        for value in record:
            assert_no_global_retrieval_metrics(value)


@pytest.mark.parametrize("mlp", [False, True])
def test_patchnce_runtime_has_live_translation_path_and_detached_source_spatial_keys(monkeypatch, tmp_path, mlp):
    import diffusion_ot.training.self_supervised_translation as translation

    def forbidden(*args, **kwargs):
        raise AssertionError("PatchNCE must not call the replaced global source InfoNCE.")

    monkeypatch.setattr(translation, "source_code_contrastive_loss", forbidden)
    torch.manual_seed(15)
    runtime = patchnce_runtime(tmp_path, mlp=mlp)
    calls = {domain: [] for domain in runtime.domains}
    for domain, context in runtime.domains.items():
        original = context.branch.encoder.forward_spatial_features

        def observe(latents, layers, *, domain=domain, original=original):
            calls[domain].append((torch.is_grad_enabled(), latents.requires_grad))
            return original(latents, layers)

        monkeypatch.setattr(context.branch.encoder, "forward_spatial_features", observe)
    weights, references, source_codes, logits, latents = batch()
    for value in latents.values():
        value.requires_grad_(True)
    loss, metrics = runtime.loss(weights, references, latents, latents, step=2,
                                 source_query_codes=source_codes)
    loss.backward()
    assert runtime.objective_name == "patchnce"
    assert set(runtime.image_objectives) == {"patchnce"}
    assert metrics["patchnce_loss"] > 0
    assert metrics["weighted_loss"] == pytest.approx(.15 * metrics["patchnce_loss"])
    assert all(value.grad is None for value in source_codes.values())
    assert all(value.grad is None for value in latents.values())
    assert all(value.grad is not None and value.grad.norm() > 0 for value in logits.values())
    assert all(value.grad is not None and value.grad.norm() > 0 for value in references.values())
    for source, target in (("cat", "dog"), ("dog", "cat")):
        assert metrics[f"{source}_to_{target}"]["readout_domain"] == target
        assert "patchnce" in metrics[f"{source}_to_{target}"]
        assert any(not enabled for enabled, _ in calls[source])
        assert any(enabled and live for enabled, live in calls[source])
        context = runtime.domains[source]
        assert any(p.grad is not None and p.grad.norm() > 0 for p in context.branch.encoder.parameters())
        for group in ("adapters", "lora"):
            assert any(p.grad is not None and p.grad.norm() > 0 for p in runtime.views[source][group].parameters())
        assert all(p.grad is None for p in context.vae.parameters())
        if mlp:
            assert all(p.grad is not None and p.grad.norm() > 0 for p in runtime.patch_projectors[source].parameters())
            retrieval = metrics[f"{source}_to_{target}"]["patchnce"]
            assert retrieval["query_projector_domain"] == target and retrieval["key_projector_domain"] == source
            assert retrieval["key_projector_detached"]
    assert_no_global_retrieval_metrics(metrics)
    assert_no_external_metrics(metrics)


@pytest.mark.parametrize("mlp", [False, True])
def test_patchnce_validation_is_repeatable_and_preserves_both_private_random_streams(tmp_path, mlp):
    torch.manual_seed(17)
    runtime = patchnce_runtime(tmp_path, mlp=mlp)
    weights, references, source_codes, _, latents = batch()
    global_rng = torch.get_rng_state().clone()
    before = deepcopy(runtime.checkpoint_state())
    with torch.no_grad():
        first = runtime.loss(weights, references, latents, latents, step=0,
                             validation_seed=19, source_query_codes=source_codes)
        second = runtime.loss(weights, references, latents, latents, step=2,
                              validation_seed=19, source_query_codes=source_codes)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    assert first[1] == second[1]
    assert first[1]["ramp"] == 1
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    after = runtime.checkpoint_state()
    for key in ("decoded_noise_states", "decoded_patch_sampling_states"):
        assert_tensor_tree_equal(before[key], after[key])
    if mlp:
        for key in ("patch_projectors", "patch_projector_ema", "patch_projector_ema_state"):
            assert_tensor_tree_equal(before[key], after[key])


@pytest.mark.parametrize("mlp", [False, True])
def test_patchnce_resume_restores_next_patches_and_diffusion_noise(tmp_path, mlp):
    torch.manual_seed(23)
    runtime = patchnce_runtime(tmp_path, mlp=mlp)
    weights, references, source_codes, _, latents = batch()
    optimizer = torch.optim.AdamW(runtime.parameter_groups())
    loss, _ = runtime.loss(weights, references, latents, latents, step=2, source_query_codes=source_codes)
    loss.backward()
    optimizer.step()
    if runtime.patch_projector_ema is not None:
        runtime.patch_projector_ema.update(runtime.patch_projectors)
    state = deepcopy(runtime.checkpoint_state())
    state.update(config=runtime.config, format_version=4)
    restored = SelfSupervisedDecoderTraining(runtime.config, deepcopy(runtime.domains), None, tmp_path, seed=901)
    restored.load_checkpoint(state)
    with torch.no_grad():
        expected = runtime.loss(weights, references, latents, latents, step=2, source_query_codes=source_codes)
        actual = restored.loss(weights, references, latents, latents, step=2, source_query_codes=source_codes)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert actual[1] == expected[1]
    for key in ("decoded_noise_states", "decoded_patch_sampling_states"):
        assert_tensor_tree_equal(runtime.checkpoint_state()[key], restored.checkpoint_state()[key])
    if mlp:
        for key in ("patch_projectors", "patch_projector_ema", "patch_projector_ema_state"):
            assert_tensor_tree_equal(state[key], restored.checkpoint_state()[key])
        missing = deepcopy(state)
        missing.pop("patch_projectors")
        with pytest.raises(ValueError, match="no patch_projectors"):
            restored.load_checkpoint(missing)
    else:
        legacy = deepcopy(state)
        legacy["decoded_patchnce_options"].pop("sampler")
        restored.load_checkpoint(legacy)


@pytest.mark.parametrize("rms_enabled", [False, True])
@pytest.mark.parametrize("mlp", [False, True])
def test_patchnce_real_training_retains_flow_transport_protection_and_cleans_logs(
        patchnce_training, monkeypatch, rms_enabled, mlp):
    import diffusion_ot.training.self_supervised_translation as translation

    def forbidden(*args, **kwargs):
        raise AssertionError("PatchNCE training executed the replaced global source InfoNCE.")

    monkeypatch.setattr(translation, "source_code_contrastive_loss", forbidden)
    run, originals, latest, encode_calls = patchnce_training
    def recipe(config):
        patchnce_recipe(config)
        if mlp:
            config["decoded_translation"]["patchnce"].update(sampler="mlp_sample", projection_dim=16)
        if rms_enabled:
            config["projection_rms"] = {"mode": "reference_ema", "decay": .99, "eps": 1e-8}
    complete, logs = run("patchnce_complete", modify=recipe)
    assert complete["step"] == 2
    if rms_enabled:
        assert all(s["num_updates"] == 2 for s in complete["projection_rms_state"].values())
    assert encode_calls
    expected_tasks = {
        "encoder": {"native", "transport", "protection", "patchnce"},
        "matching_head": {"transport", "protection", "patchnce"},
        "generator": {"native", "patchnce"},
    }
    if mlp:
        expected_tasks["patch_projector"] = {"patchnce"}
        from diffusion_ot.models.patch_sampler import PatchSampleMLP
        for i, domain in enumerate(("cat", "dog")):
            initial = PatchSampleMLP((8,), 16, seed=complete["config"]["train"].get("seed", 20260905) + 2901 + i)
            assert any(not torch.equal(value, initial.state_dict()[name])
                       for name, value in complete["patch_projectors"][domain].items())
        assert complete["patch_projector_ema_state"]["num_updates"] == 2
        group = next(g for g in complete["optimizer"]["param_groups"] if g.get("name") == "patch_projectors")
        assert group["lr"] == .0002
        assert all(complete["optimizer"]["state"][p]["step"] == 2 for p in group["params"])
    expected_losses = {"cat_reconstruction", "dog_reconstruction", "infoot_alignment",
                       "matching_variance", "matching_covariance", "patchnce"}
    for row in logs["train"] + logs["validation"]:
        assert_finite_numbers(row)
        assert_no_external_metrics(row)
        assert_no_global_retrieval_metrics(row)
        assert set(row["enabled_losses"]) == expected_losses
        decoded = row["decoded_translation"]
        assert decoded["patchnce_loss"] > 0
        assert decoded["weighted_loss"] == pytest.approx(.15 * decoded["patchnce_loss"])
        assert "conditional_projection" in row
        if rms_enabled:
            assert row["projection_rms"]["num_updates"] == row["step"]
    for row in logs["train"]:
        for group, tasks in expected_tasks.items():
            assert set(row["pcgrad"]["groups"][group]["active_tasks"]) == tasks
        assert row["weighted_decoded_encoder_gradient_norm"] > 0
        assert row["weighted_decoded_generator_gradient_norm"] > 0
        assert row["weighted_decoded_matching_head_gradient_norm"] > 0
        assert "patchnce_loss" in row["window_mean"]
        if mlp:
            assert row["weighted_decoded_patch_projector_gradient_norm"] > 0
            assert row["patch_projector_gradient_norm_pre_clip"] > 0
            assert row["patch_projector_learning_rate"] == .0002
            assert set(row["generator_learning_rates"]) == {"adapters", "lora"}
    for row in logs["validation"]:
        assert row["loss"] == pytest.approx(row["cat_raw_reconstruction"] + row["dog_raw_reconstruction"]
            + row["weighted_infoot_alignment_loss"] + row["matching_regularization_loss"]
            + row["decoded_translation"]["weighted_loss"])
        assert set(row["decoded_translation"]["native_reconstruction"]) == {"cat", "dog"}
    for domain, original in originals.items():
        current = latest[domain].branch
        assert any(not torch.equal(value, original.branch.encoder.state_dict()[key])
                   for key, value in current.encoder.state_dict().items())
        for group in ("adapters", "lora"):
            before = generator_parameter_view(original.branch)[group].state_dict()
            assert any(not torch.equal(value, before[key])
                       for key, value in generator_parameter_view(current)[group].state_dict().items())


@pytest.mark.parametrize("mlp", [False, True])
def test_patchnce_real_resume_preserves_streams_and_rejects_objective_changes(patchnce_training, mlp):
    run, _, _, _ = patchnce_training
    def recipe(config):
        patchnce_recipe(config)
        if mlp:
            config["decoded_translation"]["patchnce"].update(sampler="mlp_sample", projection_dim=16)
    complete, _ = run("patchnce_uninterrupted", modify=recipe)
    run("patchnce_resumed", steps=1, modify=recipe)
    resumed, logs = run("patchnce_resumed", resume=True, modify=recipe)
    assert [row["step"] for row in logs["train"]] == [1, 2]
    # As with the global-code control, the shuffled DataLoader cursor is not
    # restored. The guaranteed continuation here is the two private streams.
    for key in ("decoded_noise_states", "decoded_patch_sampling_states", "rng_state"):
        assert_tensor_tree_equal(complete[key], resumed[key])
    if mlp:
        assert resumed["patch_projector_ema_state"]["num_updates"] == 2
        group = next(g for g in resumed["optimizer"]["param_groups"] if g.get("name") == "patch_projectors")
        assert all(resumed["optimizer"]["state"][p]["step"] == 2 for p in group["params"])
    for update in ({"temperature": .1}, {"weight": .1}, {"num_patches": 4},
                   {"sampler": "sample" if mlp else "mlp_sample"}):
        def modify(config):
            recipe(config)
            config["decoded_translation"]["patchnce"].update(update)
            if update.get("sampler") == "sample":
                config["decoded_translation"]["patchnce"].pop("projection_dim", None)
        with pytest.raises(ValueError, match="Resume cannot change decoded_translation"):
            run("patchnce_resumed", resume=True, modify=modify)
    with pytest.raises(ValueError, match="Resume cannot change decoded_translation"):
        run("patchnce_resumed", resume=True, modify=self_supervised_recipe)


def test_patchnce_configs_change_only_translation_objective_and_output_destinations():
    root = Path(__file__).resolve().parents[1]
    directory = root / "configs/stage1b_infoot"
    control = yaml.safe_load((directory / "self_supervised_sit_b2.yaml").read_text())
    experiment_config = yaml.safe_load((directory / "self_supervised_patchnce_sit_b2.yaml").read_text())
    assert experiment_config["output_dir"] != control["output_dir"]
    assert experiment_config["decoded_translation"]["objective"] == "patchnce"
    assert experiment_config["decoded_translation"]["source_contrastive_weight"] == 0
    assert experiment_config["decoded_translation"]["patchnce"]["weight"] == .15
    for key in ("stage1a", "data", "train", "ema", "self_supervised_diagnostics", "matching", "infoot",
                "loss_weights", "matching_regularization", "matching_head", "generator_adaptation",
                "pcgrad", "conditional_projection"):
        assert experiment_config[key] == control[key]
    evaluate = yaml.safe_load((root / experiment_config["quick_evaluation"]["config"]).read_text())
    assert evaluate["infoot"] == experiment_config["infoot"]
    assert evaluate["matching"]["bandwidth_multiplier"] == experiment_config["matching"]["bandwidth_multiplier"]
    assert not evaluate["translation"]["decoded_structure_metrics"]
    from diffusion_ot.training.decoded_translation import validate_decoder_config
    validate_decoder_config(experiment_config)


@pytest.mark.parametrize("suffix", ["", "_rms_ema"])
def test_mlp_configs_change_only_sampler_and_destinations(suffix):
    root = Path(__file__).resolve().parents[1]
    old_stem = f"self_supervised_patchnce{suffix}_sit_b2"
    new_stem = f"self_supervised_patchnce_mlp{suffix}_sit_b2"
    for directory in ("stage1b_infoot", "stage1b_eval"):
        old = yaml.safe_load((root / f"configs/{directory}/{old_stem}.yaml").read_text())
        new = yaml.safe_load((root / f"configs/{directory}/{new_stem}.yaml").read_text())
        assert new.pop("output_dir") != old.pop("output_dir")
        if directory == "stage1b_infoot":
            from diffusion_ot.training.decoded_translation import validate_decoder_config
            validate_decoder_config(new)
            assert Path(root / new["quick_evaluation"]["config"]).is_file()
            new["quick_evaluation"]["config"] = old["quick_evaluation"]["config"]
            patch = new["decoded_translation"]["patchnce"]
            assert patch.pop("sampler") == "mlp_sample"
            assert patch.pop("projection_dim") == 256
            assert patch.pop("lr") == .0002
            assert patch.pop("grad_clip_norm") == 1.
        else:
            assert new["translation"].pop("patchnce_metrics") is True
        assert new == old


@pytest.mark.parametrize("mode", ["weighted_sum", "guarded"])
def test_mlp_heads_also_update_without_pcgrad(patchnce_training, mode):
    run, _, _, _ = patchnce_training
    def recipe(config):
        patchnce_recipe(config)
        config["decoded_translation"]["patchnce"].update(sampler="mlp_sample", projection_dim=16)
        config["pcgrad"]["enabled"] = False
        config["gradient_guard"]["enabled"] = mode == "guarded"
    checkpoint, logs = run("mlp_without_pcgrad", steps=1, modify=recipe)
    assert logs["train"][0]["patch_projector_gradient_norm_pre_clip"] > 0
    group = next(g for g in checkpoint["optimizer"]["param_groups"] if g.get("name") == "patch_projectors")
    assert all(checkpoint["optimizer"]["state"][p]["step"] == 1 for p in group["params"])
