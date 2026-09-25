from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


class TinyBranch(nn.Module):
    semantic_cfg_enabled = False

    def __init__(self, input_space):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.encoder_input_space = input_space
        self.observed_images = []

    def encode(self, x0, *, encoder_image=None):
        self.observed_images.append(encoder_image)
        if self.encoder_input_space == "rgb":
            assert encoder_image is not None
            return encoder_image.mean((2, 3))
        assert encoder_image is None
        return x0.mean((2, 3))

    def load_pdae_state_dict(self, state):
        self.loaded_state = state

    def predict_with_z(self, x_t, **kwargs):
        return SimpleNamespace(base_sample=torch.zeros_like(x_t), delta_sample=torch.zeros_like(x_t))


class TinyVAE(nn.Module):
    config = SimpleNamespace(scaling_factor=1.0)

    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()), requires_grad=False)

    def decode(self, value):
        return SimpleNamespace(sample=value[:, :3])


def _evaluator(tmp_path, input_space="rgb", include_images=True, roundtrip=False):
    from diffusion_ot.evaluation.stage1a_eval import LoadedStage1AEvaluator

    items = []
    for index in range(2):
        item = {
            "x0_latent": torch.ones(4, 4, 4),
            "sample_id": f"cat-{index}",
            "domain": "cat", "split": "val", "latent_path": str(index), "metadata": {},
        }
        if include_images:
            item["encoder_image"] = torch.full((3, 4, 4), -0.6)
        items.append(item)
    return LoadedStage1AEvaluator(
        branch=TinyBranch(input_space),
        transformer=SimpleNamespace(config=SimpleNamespace(num_classes=3)),
        vae=TinyVAE(), dataset=items,
        training_config={"output_dir": "outputs/test_rgb"},
        evaluation_config={
            "dataset": {"smoke_samples": 2},
            "sampling": {
                "smoke_num_steps": 2, "variants": ["correct_z"], "guidance_scales": [1.0],
                "inferred_noise": {"enabled": roundtrip, "num_steps": 2},
            },
        },
        project_root=tmp_path,
        training_config_path=tmp_path / "train.yaml",
        evaluation_config_path=tmp_path / "eval.yaml",
        checkpoint_path=tmp_path / "latest.pt", checkpoint_step=100,
        domain="cat", split="val", device="cpu", model_dtype=torch.float32, weights="raw",
    )


def _install_smoke_mocks(monkeypatch, evaluator):
    import diffusion_ot.evaluation.stage1a_eval as evaluation

    grids = {}
    monkeypatch.setattr(evaluation, "load_stage1a_evaluator", lambda *args, **kwargs: evaluator)
    monkeypatch.setattr(
        evaluation, "reconstruct_cfg_sweep",
        lambda branch, transformer, starting_noise, *args, **kwargs: {
            "correct_z_cfg_1": torch.zeros_like(starting_noise),
        },
    )
    monkeypatch.setattr(
        evaluation, "_save_grid",
        lambda path, rows, samples_per_row: grids.update({path.name: rows}),
    )
    return grids


def test_rgb_smoke_and_roundtrip_use_original_rgb_for_condition_and_metrics(tmp_path, monkeypatch):
    from diffusion_ot.evaluation.stage1a_eval import run_stage1a_smoke_test

    evaluator = _evaluator(tmp_path, roundtrip=True)
    grids = _install_smoke_mocks(monkeypatch, evaluator)
    report = run_stage1a_smoke_test("train.yaml", "eval.yaml")

    torch.testing.assert_close(
        evaluator.branch.observed_images[0], torch.full((2, 3, 4, 4), -0.6),
    )
    assert report.encoder_input_space == "rgb"
    assert report.image_reference == "original_rgb"
    assert report.row_order == ["original", "vae_reconstruction", "correct_z_cfg_1"]
    # Generated RGB=0.5 is compared with real RGB=0.2, not VAE reconstruction=1.0.
    assert report.metrics["correct_z_cfg_1"]["pixel_mse"] == pytest.approx(0.09)
    assert report.metrics["correct_z_cfg_1"]["latent_mse"] == pytest.approx(1.0)
    assert report.metrics["vae_reconstruction"]["pixel_mse"] == pytest.approx(0.64)
    torch.testing.assert_close(grids["reconstruction_grid.png"][0], torch.full((2, 3, 4, 4), 0.2))
    torch.testing.assert_close(grids["reconstruction_grid.png"][1], torch.ones(2, 3, 4, 4))

    roundtrip = json.loads(Path(report.extra_reports["inferred_noise_roundtrip"]).read_text(encoding="utf-8"))
    assert roundtrip["image_reference"] == "original_rgb"
    assert roundtrip["row_order"] == ["original", "vae_reconstruction", "correct_z"]
    assert roundtrip["metrics"]["correct_z"]["pixel_mse"] == pytest.approx(0.64)
    torch.testing.assert_close(grids["roundtrip_grid.png"][0], torch.full((2, 3, 4, 4), 0.2))


def test_latent_smoke_keeps_legacy_condition_and_metric_reference(tmp_path, monkeypatch):
    from diffusion_ot.evaluation.stage1a_eval import run_stage1a_smoke_test

    evaluator = _evaluator(tmp_path, input_space="latent", include_images=False)
    grids = _install_smoke_mocks(monkeypatch, evaluator)
    report = run_stage1a_smoke_test("train.yaml", "eval.yaml")

    assert evaluator.branch.observed_images == [None]
    assert report.encoder_input_space == "latent"
    assert report.image_reference == "vae_reconstruction"
    assert report.row_order == ["original", "correct_z_cfg_1"]
    assert report.metrics["correct_z_cfg_1"]["pixel_mse"] == pytest.approx(0.25)
    torch.testing.assert_close(grids["reconstruction_grid.png"][0], torch.ones(2, 3, 4, 4))


def test_latent_comparison_conditions_on_latents_but_scores_original_rgb(tmp_path, monkeypatch):
    from diffusion_ot.evaluation.stage1a_eval import run_stage1a_smoke_test

    evaluator = _evaluator(tmp_path, input_space="latent", include_images=True, roundtrip=True)
    evaluator.evaluation_config["metrics"] = {"image_reference": "original_rgb"}
    grids = _install_smoke_mocks(monkeypatch, evaluator)
    report = run_stage1a_smoke_test("train.yaml", "eval.yaml")

    assert evaluator.branch.observed_images == [None]
    assert report.encoder_input_space == "latent"
    assert report.image_reference == "original_rgb"
    assert report.row_order == ["original", "vae_reconstruction", "correct_z_cfg_1"]
    assert report.metrics["correct_z_cfg_1"]["pixel_mse"] == pytest.approx(0.09)
    assert report.metrics["vae_reconstruction"]["pixel_mse"] == pytest.approx(0.64)
    torch.testing.assert_close(grids["reconstruction_grid.png"][0], torch.full((2, 3, 4, 4), 0.2))
    roundtrip = json.loads(Path(report.extra_reports["inferred_noise_roundtrip"]).read_text(encoding="utf-8"))
    assert roundtrip["encoder_input_space"] == "latent"
    assert roundtrip["image_reference"] == "original_rgb"
    assert roundtrip["row_order"] == ["original", "vae_reconstruction", "correct_z"]
    assert roundtrip["metrics"]["correct_z"]["pixel_mse"] == pytest.approx(0.64)


def test_rgb_encoder_still_gets_original_images_when_metric_reference_is_vae(tmp_path, monkeypatch):
    from diffusion_ot.evaluation.stage1a_eval import run_stage1a_smoke_test

    evaluator = _evaluator(tmp_path)
    evaluator.evaluation_config["metrics"] = {"image_reference": "vae_reconstruction"}
    _install_smoke_mocks(monkeypatch, evaluator)
    report = run_stage1a_smoke_test("train.yaml", "eval.yaml")
    torch.testing.assert_close(evaluator.branch.observed_images[0], torch.full((2, 3, 4, 4), -0.6))
    assert report.image_reference == "vae_reconstruction"
    assert report.row_order == ["original", "correct_z_cfg_1"]
    assert report.metrics["correct_z_cfg_1"]["pixel_mse"] == pytest.approx(0.25)


def test_rgb_smoke_rejects_missing_original_images_without_vae_fallback(tmp_path, monkeypatch):
    import diffusion_ot.evaluation.stage1a_eval as evaluation

    evaluator = _evaluator(tmp_path, include_images=False)
    _install_smoke_mocks(monkeypatch, evaluator)
    monkeypatch.setattr(
        evaluation, "decode_vae_latents",
        lambda *args: pytest.fail("Must reject missing originals before any VAE decoding."),
    )
    with pytest.raises(ValueError, match="requires original encoder_image"):
        evaluation.run_stage1a_smoke_test("train.yaml", "eval.yaml")


@pytest.mark.parametrize("shape", [(2, 4, 4, 4), (3, 3, 4, 4), (3, 4, 4)])
def test_rgb_eval_rejects_invalid_image_shape(tmp_path, shape):
    from diffusion_ot.evaluation.stage1a_eval import _evaluation_encoder_image

    with pytest.raises(ValueError, match="encoder_image"):
        _evaluation_encoder_image(
            _evaluator(tmp_path),
            {"x0_latent": torch.zeros(2, 4, 4, 4), "encoder_image": torch.zeros(shape)},
        )


@pytest.mark.parametrize("input_space", ["rgb", "latent"])
@pytest.mark.parametrize("reference", [None, "original_rgb", "vae_reconstruction"])
def test_evaluator_loads_originals_for_rgb_encoder_or_original_reference(tmp_path, monkeypatch, input_space, reference):
    import diffusion_ot.data.latent_dataset as data
    import diffusion_ot.evaluation.stage1a_eval as evaluation
    import diffusion_ot.integrations.sit_diffusers as sit
    import diffusion_ot.models.pdae_sit as models

    train_path, eval_path = tmp_path / "train.yaml", tmp_path / "eval.yaml"
    checkpoint_path = tmp_path / "latest.pt"
    torch.save({"domain": "cat", "model": {"test_state": 1}}, checkpoint_path)
    configs = {
        "train.yaml": {
            "domain": "cat", "model_config": "model.yaml", "data_config": "data.yaml",
        },
        "eval.yaml": {"sampling": {"variants": ["correct_z"]}, "metrics": {"image_reference": reference}},
        "model.yaml": {"pretrained": "pretrained.yaml"},
    }
    branch = TinyBranch(input_space)
    transformer = nn.Linear(1, 1)
    captured = {}
    monkeypatch.setattr(evaluation, "load_yaml_config", lambda path: configs[path.name])
    monkeypatch.setattr(evaluation, "effective_project_root", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(evaluation, "find_project_root", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(sit, "load_sit_components", lambda *args, **kwargs: SimpleNamespace(transformer=transformer, vae=TinyVAE()))
    monkeypatch.setattr(sit, "validate_transformer_config", lambda *args, **kwargs: [])
    monkeypatch.setattr(models, "build_pdae_sit_branch", lambda *args, **kwargs: branch)

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(data, "CachedLatentDataset", fake_dataset)
    evaluator = evaluation.load_stage1a_evaluator(
        train_path, eval_path, device="cpu", weights="raw", checkpoint_path=checkpoint_path,
    )
    assert captured["include_original_images"] is (input_space == "rgb" or reference == "original_rgb")
    assert captured["random_horizontal_flip"] == 0.0
    assert evaluator.branch.loaded_state == {"test_state": 1}
    assert evaluation.stage1a_architecture_metadata(evaluator.branch).get("encoder_input_space", "latent") == input_space


def test_latent_architecture_metadata_preserves_exact_legacy_provenance_shape():
    from diffusion_ot.evaluation.stage1a_eval import stage1a_architecture_metadata

    legacy = {"semantic_cfg_enabled": False, "attention_lora": {"enabled": False}}
    # Both old branches without the attribute and newly built latent branches must
    # compare equal with Stage 1B checkpoint provenance created before RGB support.
    assert stage1a_architecture_metadata(SimpleNamespace()) == legacy
    assert stage1a_architecture_metadata(TinyBranch("latent")) == legacy
    rgb = TinyBranch("rgb")
    rgb.encoder_image_size = 256
    assert stage1a_architecture_metadata(rgb) == {
        **legacy, "encoder_input_space": "rgb", "encoder_image_size": 256,
    }


@pytest.mark.parametrize("reference", ["original", "encoder", ["original_rgb"], 1])
def test_evaluation_rejects_unknown_image_reference(reference):
    from diffusion_ot.evaluation.stage1a_eval import _validate_eval_config

    with pytest.raises(ValueError, match="metrics.image_reference"):
        _validate_eval_config({"metrics": {"image_reference": reference}})


def test_comparison_evaluation_config_changes_only_reference_and_output():
    import yaml
    from diffusion_ot.evaluation.stage1a_eval import _validate_eval_config

    root = Path(__file__).resolve().parents[1] / "configs" / "stage1a_eval"
    baseline = yaml.safe_load((root / "sit_b2_256.yaml").read_text())
    comparison = yaml.safe_load((root / "rgb_vs_latent_sit_b2_256.yaml").read_text())
    _validate_eval_config(comparison)
    assert comparison["metrics"].pop("image_reference") == "original_rgb"
    assert comparison["output"]["subdir"] == "stage1a_rgb_comparison"
    comparison["output"]["subdir"] = baseline["output"]["subdir"]
    assert comparison == baseline
