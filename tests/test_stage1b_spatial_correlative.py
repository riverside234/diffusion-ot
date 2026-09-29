"""v4.5: tiny real trainer/evaluator runs, calibration provenance and v4 parity."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from diffusion_ot.losses.spatial_correlative import (
    checkpoint_spatial_cost, encode_spatial_descriptors, spatial_correlative_options,
    spatial_descriptor_id,
)
from diffusion_ot.models.pdae_sit import PDAELatentEncoder
from test_decoded_translation import TinyVAE
from test_infonce_projection_gradient import v3_tiny_recipe
from test_stage1b_extensions import experiment, assert_finite_numbers, assert_tensor_tree_equal
from test_translation_image import image_options

ROOT = Path(__file__).resolve().parents[1]


def recipe(cfg, alpha=.2):
    v3_tiny_recipe(cfg)
    cfg["decoded_translation"].update(image_options())
    cfg["spatial_correlative_cost"] = dict(enabled=True, layers=[0], grid_size=2, mixing_weight=alpha)


@pytest.fixture
def spatial_run(experiment, monkeypatch):
    import diffusion_ot.training.decoded_translation as external
    import diffusion_ot.losses.semantic_prior as semantic
    run, originals, latest = experiment
    for context in originals.values():
        context.branch.encoder = PDAELatentEncoder(channels=(8,), z_dim=6, spatial_size=2, num_groups=2)
    def forbidden(*args, **kwargs):
        pytest.fail("Spatial experiment loaded an external teacher or GAN")
    for name in ("load_image_features", "RGBPatchDiscriminator", "FeatureDiscriminator"):
        monkeypatch.setattr(external, name, forbidden)
    monkeypatch.setattr(semantic, "SemanticPriorBank", forbidden)
    monkeypatch.setattr(TinyVAE, "encode", lambda self, rgb: SimpleNamespace(
        latent_dist=SimpleNamespace(mean=F.conv2d(rgb, self.conv.weight.transpose(0, 1)))), raising=False)
    return run, originals, latest


def test_actual_training_validation_and_resume_preserve_frozen_calibration(spatial_run, tmp_path):
    import diffusion_ot.training.train_joint_infoot as train
    from diffusion_ot.evaluation.stage1b_eval import _validate_self_supervised_checkpoint
    run, _, _ = spatial_run
    complete, logs = run("spatial", modify=recipe)
    state = complete["spatial_correlative_cost_state"]
    initial = train._load_checkpoint(tmp_path / "spatial/checkpoints/step_000000.pt")
    assert state["calibration"] == initial["spatial_correlative_cost_state"]["calibration"]
    assert state["calibration"]["status"] == "calibrated"
    assert_tensor_tree_equal(state["probe_baseline"], initial["spatial_correlative_cost_state"]["probe_baseline"])
    for d in ("cat", "dog"):
        assert all(i.startswith(d + "_train") for i in state["calibration"]["sample_ids"][d])
    for row in logs["train"] + logs["validation"]:
        assert_finite_numbers(row)
        diagnostic = row["spatial_correlative"]
        assert set(("encoder", "spatial", "mixed")) <= diagnostic.keys()
        assert diagnostic["mixing_weight"] == .2
        assert diagnostic["effective_sinkhorn_centered_rms_over_epsilon"] > 0
        assert row["decoded_translation"]["source_contrastive_projection_gradient_scale"] == .1
        assert set(row["decoded_translation"]["image_losses"]) == set(image_options())
    for row in logs["validation"]:
        diagnostic = row["spatial_correlative"]
        assert set(diagnostic["conditional"]) == {"cat_to_dog", "dog_to_cat"}
        for ids in row["projection_probe"]["sample_ids"].values():
            assert not set(ids["reference"]) & set(ids["query"])
        # Both fixed validation solves use the same calibrated cost mixture.
        assert row["decoded_translation"]["spatial_correlative"]["mixed"]["expected_cost"] == pytest.approx(
            diagnostic["mixed"]["expected_cost"], rel=1e-5)
    assert logs["validation"][0]["spatial_correlative"]["fixed_reference_descriptor_drift_rms"] == dict(cat=0., dog=0.)
    assert all(v > 0 for v in logs["validation"][-1]["spatial_correlative"]["fixed_reference_descriptor_drift_rms"].values())
    for row in logs["train"]:
        assert row["optimizer_gradient_mode"] == "pcgrad"
        assert row["transport_encoder_cost"] == pytest.approx(row["spatial_correlative"]["encoder"]["expected_cost"])
        assert "spatial_correlative" not in row["enabled_losses"]  # A fitting cost, not a new neural task.
    _validate_self_supervised_checkpoint(complete["config"], complete)
    resumed, _ = run("spatial", steps=3, resume=True, modify=recipe)
    assert_tensor_tree_equal(state, resumed["spatial_correlative_cost_state"])
    with pytest.raises(ValueError, match="disagree"):
        run("spatial", steps=4, resume=True, modify=lambda c: recipe(c, .1))


def test_alpha_zero_has_exact_v4_parameters_rng_and_logs(spatial_run, monkeypatch):
    import diffusion_ot.training.train_joint_infoot as train
    run, _, _ = spatial_run
    def baseline(cfg):
        recipe(cfg)
        cfg.pop("spatial_correlative_cost")
    control, control_logs = run("control", steps=1, modify=baseline)
    def forbidden(*args, **kwargs):
        pytest.fail("Alpha-zero control computed spatial descriptors")
    monkeypatch.setattr(train, "encode_spatial_descriptors", forbidden)
    zero, zero_logs = run("zero", steps=1, modify=lambda c: recipe(c, 0))
    for key in ("encoders", "encoder_ema", "matching_heads", "matching_head_ema", "generators", "optimizer",
                "decoded_code_projectors", "decoded_noise_states", "decoded_image_sampling_states", "rng_state"):
        if key in control:
            assert_tensor_tree_equal(control[key], zero[key])
    for rows in (zero_logs["train"], zero_logs["validation"]):
        assert all("spatial_correlative" not in r for r in rows)
    assert "spatial_correlative_cost_state" not in zero
    for key in ("loss", "infoot_feature_loss"):
        assert control_logs["train"][0][key] == zero_logs["train"][0][key]


def test_v4_5_configs_only_change_reference_cost_and_output():
    for directory in ("stage1b_infoot", "stage1b_eval"):
        base = yaml.safe_load((ROOT / f"configs/{directory}/self_supervised_infonce_v4_sit_b2.yaml").read_text())
        new = yaml.safe_load((ROOT / f"configs/{directory}/self_supervised_infonce_v4_5_sit_b2.yaml").read_text())
        assert new.pop("output_dir") == base.pop("output_dir") + "_5"
        options = spatial_correlative_options(new)
        assert options["mixing_weight"] == .2 and options["layers"] == [0, 1] and options["grid_size"] == 8
        new.pop("spatial_correlative_cost")
        if "quick_evaluation" in new:
            assert new["quick_evaluation"]["config"] == base["quick_evaluation"]["config"].replace("_v4_", "_v4_5_")
            new["quick_evaluation"]["config"] = base["quick_evaluation"]["config"]
        assert new == base


def test_full_bank_export_cannot_silently_drop_spatial_cost(tmp_path, monkeypatch):
    from diffusion_ot.evaluation import offline_pipeline
    monkeypatch.setattr(offline_pipeline, "load_yaml_config", lambda path: {
        "alignment_config": "alignment.yaml", "spatial_correlative_cost": {"enabled": True}})
    with pytest.raises(ValueError, match="full-bank export does not yet support"):
        offline_pipeline.run_stage23(tmp_path / "offline.yaml", "unused.pt", root=tmp_path)


@pytest.mark.parametrize("spatial", [False, True])
def test_quick_eval_cli_uses_calibrated_initial_checkpoint_for_spatial_only(tmp_path, monkeypatch, spatial):
    import importlib.util
    import diffusion_ot.integrations.hf_snapshot as configs
    import diffusion_ot.training.train_joint_infoot as trainer
    import diffusion_ot.evaluation.stage1b_eval as evaluator
    spec = importlib.util.spec_from_file_location("train_joint_infoot_cli", ROOT / "scripts/train_joint_infoot.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cfg = {"spatial_correlative_cost": {"enabled": True}} if spatial else {}
    monkeypatch.setattr(configs, "load_yaml_config", lambda path: cfg)
    args = SimpleNamespace(config=str(tmp_path / "config.yaml"), smoke=False, max_steps=1,
        quick_eval=str(tmp_path / "eval.yaml"), eval_weights="raw", dry_run=False, resume=None,
        device_cat=None, device_dog=None, max_reference=None, max_projection=None, max_query=None)
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    calls = []
    def train(*args, **kwargs):
        calls.append("train")
        return SimpleNamespace(checkpoint_path=str(tmp_path / "checkpoints/latest.pt"), to_dict=lambda: {})
    def evaluate(*args, **kwargs):
        calls.append(Path(kwargs["checkpoint_path"]).name if kwargs.get("checkpoint_path") else "stage1a")
        return SimpleNamespace(output_dir="eval", baseline_comparison={"status": "compared"})
    monkeypatch.setattr(trainer, "train_joint_infoot", train)
    monkeypatch.setattr(evaluator, "run_stage1b_evaluation", evaluate)
    assert cli.main() == 0
    assert calls == (["train", "step_000000.pt", "latest.pt"] if spatial else ["stage1a", "train", "latest.pt"])


@pytest.mark.parametrize("weights", ["raw", "ema"])
@pytest.mark.parametrize("v6", [False, True])
def test_standalone_uses_selected_encoder_and_shared_calibration_with_reference_grids(spatial_run, monkeypatch, tmp_path, weights, v6):
    import diffusion_ot.data.ground_truth as gt
    import diffusion_ot.evaluation.stage1b_eval as evaluation
    from diffusion_ot.models.matching_head import make_matching_head, matching_head_spec
    from diffusion_ot.models.generator_adaptation import load_joint_generator
    from diffusion_ot.losses.projection_rms import checkpoint_projection_rms
    from diffusion_ot.losses.source_selection import checkpoint_source_selection
    from test_stage1b_v6 import v6_recipe
    run, _, latest = spatial_run
    checkpoint, _ = run("eval_training", steps=1, modify=v6_recipe if v6 else recipe)
    config = checkpoint["config"]
    checkpoint_path = tmp_path / "eval_training/checkpoints/latest.pt"
    eval_config = yaml.safe_load((ROOT / "configs/stage1b_eval/self_supervised_infonce_v4_5_sit_b2.yaml").read_text())
    eval_config.update(project_root=str(tmp_path), output_dir="eval", alignment_device="cpu",
                       spatial_correlative_cost=config["spatial_correlative_cost"], infoot=config["infoot"])
    if v6:
        eval_config["source_aware_selection"] = config["source_aware_selection"]
        eval_config["matching"]["projection_bandwidth_multiplier"] = .25
    eval_config["data"].update(reference_samples_per_domain=6, projection_samples_per_domain=8,
                               query_samples_per_domain=2, batch_size=4, num_workers=0)
    eval_config["translation"].update(samples_per_direction=2, num_steps=3)
    eval_config["reconstruction"].update(samples_per_domain=2, num_steps=3)
    eval_config["visualization"]["enabled"] = False
    eval_path = tmp_path / "evaluation.yaml"
    eval_path.write_text(yaml.safe_dump(eval_config))

    def latent(sample_id):
        domain, split, index = sample_id.split("_")
        rng = torch.Generator().manual_seed(int(index) + (100 if domain == "dog" else 0) + (20 if split == "val" else 0))
        return torch.randn(4, 8, 8, generator=rng)

    class Dataset:
        def __init__(self, domain, split):
            self.domain, self.split = domain, split
        def __len__(self):
            return 8
        def __getitem__(self, index):
            sample_id = f"{self.domain}_{self.split}_{index}"
            return dict(x0_latent=latent(sample_id), sample_id=sample_id,
                        metadata=dict(sample_id=sample_id, latent_path="unused"))

    def load_context(alignment, project_root, domain, **kwargs):
        selected = torch.load(kwargs["checkpoint_path"], weights_only=False)
        evaluation._validate_self_supervised_checkpoint(alignment, selected)
        value = latest[domain]
        value.domain = domain
        value.stage1a_checkpoint_path, value.stage1a_checkpoint_step = tmp_path / f"{domain}.pt", 100
        value.stage1a_weights, value.weights = "ema", weights
        value.checkpoint_path, value.checkpoint_step = kwargs["checkpoint_path"], selected["step"]
        value.branch.encoder.load_state_dict(selected["encoder_ema" if weights == "ema" else "encoders"][domain])
        load_joint_generator(value.branch, selected, domain, weights=weights)
        value.matching_head = make_matching_head(matching_head_spec(config), device="cpu")
        value.matching_head.load_state_dict(selected["matching_head_ema" if weights == "ema" else "matching_heads"][domain])
        value.projection_rms = checkpoint_projection_rms(selected, config, weights=weights)
        value.spatial_cost = checkpoint_spatial_cost(selected, config)
        value.source_selection = checkpoint_source_selection(selected, config)
        value.branch.eval()
        value.matching_head.eval()
        return value

    monkeypatch.setattr(evaluation, "_load_domain_context", load_context)
    monkeypatch.setattr(evaluation, "_dataset", lambda context, split, root: Dataset(context.domain, split))
    monkeypatch.setattr(evaluation, "_load_latents_from_bank", lambda bank, count: torch.stack([latent(i) for i in bank.sample_ids[:count]]))
    monkeypatch.setattr(gt, "load_ground_truth_images", lambda path, rows: torch.stack([latent(row["sample_id"])[:3].sigmoid() for row in rows]))
    initial_report = evaluation.run_stage1b_evaluation(tmp_path / "eval_training.yaml", eval_path,
                                              checkpoint_path=checkpoint_path.with_name("step_000000.pt"), weights=weights)
    assert initial_report.baseline_comparison["status"] == "baseline"
    report = evaluation.run_stage1b_evaluation(tmp_path / "eval_training.yaml", eval_path,
                                              checkpoint_path=checkpoint_path, weights=weights)
    assert report.baseline_comparison["status"] == "compared"
    assert Path(report.baseline_comparison["baseline_report"]).parent == Path(initial_report.output_dir)
    assert report.generation_protocol["spatial_correlative"]["calibration"] == checkpoint["spatial_correlative_cost_state"]["calibration"]
    if v6:
        assert report.generation_protocol["source_aware_selection"] == checkpoint["source_aware_selection_state"]
    assert report.solver["spatial_correlative"]["mixed"]["centered_rms"] > 0
    banks = {d: evaluation.load_latent_bank(Path(report.output_dir) / "banks" / f"{d}_reference.pt") for d in latest}
    runtime = checkpoint_spatial_cost(checkpoint, config)
    for d, bank in banks.items():
        assert bank.to_payload()["format_version"] == 3
        expected = encode_spatial_descriptors(latest[d].branch.encoder, torch.stack([latent(i) for i in bank.sample_ids]),
                                              runtime.options, device="cpu", dtype=torch.float32)
        torch.testing.assert_close(bank.spatial_descriptors, expected, rtol=0, atol=0)
        assert bank.spatial_id == spatial_descriptor_id(runtime.metadata())
        query = evaluation.load_latent_bank(Path(report.output_dir) / "banks" / f"{d}_query.pt")
        evaluation.validate_bank_compatibility(bank, query)
        query.spatial_id = "wrong-grid-or-calibration"
        with pytest.raises(ValueError, match="spatial descriptor protocols"):
            evaluation.validate_bank_compatibility(bank, query)
        query.spatial_id = bank.spatial_id
        query.checkpoint_id += "wrong-weights"
        with pytest.raises(ValueError, match="different encoders"):
            evaluation.validate_bank_compatibility(bank, query)
    costs = runtime.costs({d: b.matching_features for d, b in banks.items()}, {d: b.spatial_descriptors for d, b in banks.items()})
    coupling = torch.load(Path(report.output_dir) / "coupling.pt", weights_only=False)
    assert float((costs["mixed"] * coupling["coupling"]).sum()) == pytest.approx(report.solver["spatial_correlative"]["mixed"]["expected_cost"])
    for direction in ("cat_to_dog", "dog_to_cat"):
        diagnostic = report.projections[direction]
        assert "spatial_correlative" in diagnostic
        if v6:
            assert diagnostic["source_aware_selection"]["mean_weight_l1_change"] > 0
            decoded = diagnostic["decoded_image_diagnostics"]
            # The same generated row supplies both active image losses and held-out texture.
            row = next(v for v in decoded.values() if isinstance(v, dict) and "image_losses" in v)
            assert "source_lab_swd" in row["image_losses"] and "target_patch_swd" not in row["image_losses"]
            assert row["diagnostics"]["target_patch_swd"]["real_to_real_distance"] > 0
        grid = diagnostic["decoded_image_diagnostics"]["transport_reference_grid"]
        assert Path(grid["path"]).is_file() and len(grid["rows"]) == 4
        assert len(grid["target_ids"]) == 2 and len(grid["target_ids"][0]) == 2
    with pytest.raises(ValueError, match="calibrated checkpoint" if v6 else "checkpoint with frozen calibration"):
        evaluation.run_stage1b_evaluation(tmp_path / "eval_training.yaml", eval_path)
    eval_config["spatial_correlative_cost"]["grid_size"] = 4
    eval_path.write_text(yaml.safe_dump(eval_config))
    with pytest.raises(ValueError, match="match the training spatial"):
        evaluation.run_stage1b_evaluation(tmp_path / "eval_training.yaml", eval_path, checkpoint_path=checkpoint_path)
