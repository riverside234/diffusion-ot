"""Stopped factor previews preserve fit artifacts, validation, and resume."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from test_infoot_vit_mapping import banks, cpu_threads, mapped
from test_infoot_vit_lowrank import tiny_config
from infoot_vit import infoot_test
from infoot_vit.infoot_helper.conditional import BalancedModel, normalize_rows
from infoot_vit.infoot_helper.feature_bank import file_hash, save_tensor, write_json
from infoot_vit.infoot_helper.lowrank_preview import LowRankCheckpointPreview, load_preview_snapshot
from infoot_vit.infoot_helper.mapping import FeatureMapper, load_mapped
from infoot_vit.lowrank import experiment, kernels, solver
from infoot_vit.lowrank.storage import checkpoint_digest


@pytest.fixture
def stopped_fit(tmp_path, banks):
    c = tiny_config()
    c["optimizer"].update(max_steps=2, stationarity_tolerance=1e-15)
    with pytest.raises(RuntimeError, match="max_steps"):
        experiment.fit(c, root=tmp_path)
    return next((tmp_path / "outputs/infoot_vit").iterdir()), c


def hashes(directory):
    return {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob("*") if p.is_file()}


def forbid_fitting(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Preview must not fit or repair factors/kernels/routers")
    monkeypatch.setattr(BalancedModel, "fit", unexpected)
    monkeypatch.setattr(solver, "solve", unexpected)
    monkeypatch.setattr(solver, "project", unexpected)
    monkeypatch.setattr(kernels, "fit_features", unexpected)


@pytest.mark.parametrize("multiplier", [1., .5])
def test_preview_matches_dense_saved_factor_projection_without_refit(stopped_fit, banks, monkeypatch, multiplier):
    directory, c = stopped_fit
    before = hashes(directory)
    with pytest.raises(ValueError, match="incomplete"):
        FeatureMapper.load(directory)
    forbid_fitting(monkeypatch)
    snapshot, state = load_preview_snapshot(directory)
    projection = deepcopy(snapshot["config"]["projection"])
    projection["bandwidth_multiplier"] = multiplier
    mapper = LowRankCheckpointPreview.load(directory, projection=projection, device="cpu")
    assert mapper.manifest["status"] == "failed"
    assert not mapper.manifest["checkpoint_preview"]["converged"]
    assert mapper.manifest["checkpoint_preview"]["step"] == 2
    assert mapper.device.type == mapper.cross.device.type == "cpu"
    q, r, g = (state[k].double() for k in ("q", "r", "g"))
    saved = torch.load(directory / "kernels.pt", weights_only=True)
    fx = (saved["fx"].double() if multiplier == 1 else
          kernels.features(mapper.x.flatten(0, 1), kernels.projection_state(saved["source"], multiplier)))
    fy = mapper.target_kernel_features.flatten(0, 1)
    plan = (q / g) @ r.T
    ky = fy @ fy.T
    actual = mapped(mapper, banks[2], return_metadata=True, chunk_size=1)
    again = mapped(mapper, banks[2], return_metadata=True, chunk_size=2)
    torch.testing.assert_close(actual.mapped_features, again.mapped_features, rtol=0, atol=0)
    for query, output in zip(banks[2].features, actual.mapped_features):
        kq = kernels.features(query, mapper.kernel_source) @ fx.T
        scores = (kq @ plan @ ky.T) / ky.mean(0)
        weights = mapper.image.conditional_weights(query.flatten()[None])[0]
        expected = sum(weights[j] * (normalize_rows(scores[:, j*4:(j+1)*4]) @ mapper.y[j])
                       for j in range(len(mapper.y)))
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    assert actual.mapped_features.shape == banks[2].features.shape
    assert actual.valid_mask.all() and torch.isfinite(actual.mapped_features).all()
    assert hashes(directory) == before


@pytest.mark.parametrize("kind", ["checksum", "identity", "negative", "nonfinite", "capacity", "zero_g", "shape", "step"])
def test_corrupt_or_infeasible_checkpoint_is_rejected(stopped_fit, kind):
    directory, _ = stopped_fit
    path = directory / "factors/latest.pt"
    s = torch.load(path, weights_only=True)
    if kind == "identity":
        s["fit_fingerprint"] = "different fit"
    elif kind == "step":
        s["step"] += 1
    elif kind == "zero_g":
        s["g"][0] = 0
    elif kind == "shape":
        s["q"] = s["q"][:-1]
    else:
        s["q"][0, 0] = {"checksum": .5, "negative": -1., "nonfinite": float("nan"), "capacity": .9}[kind]
    if kind != "checksum":
        # Even when metadata and checksum are internally consistent, the
        # original constraints/finiteness and fit identity must still gate use.
        s["q_rows"], s["r_rows"] = s["q"].double().sum(1), s["r"].double().sum(1)
        s["q_columns"], s["r_columns"] = s["q"].double().sum(0), s["r"].double().sum(0)
        s["checkpoint_sha256"] = checkpoint_digest(s)
    save_tensor(path, s)
    before = hashes(directory)
    with pytest.raises(ValueError):
        LowRankCheckpointPreview.load(directory)
    assert hashes(directory) == before


@pytest.mark.parametrize("kind", ["active", "complete", "partial", "config", "ids", "missing", "router", "kernels"])
def test_invalid_fit_or_dependencies_are_rejected(stopped_fit, kind):
    directory, _ = stopped_fit
    path = directory / "manifest.json"
    m = json.loads(path.read_text())
    if kind in {"active", "complete"}:
        m["status"] = "fitting" if kind == "active" else "complete"
    elif kind == "partial":
        m["config"]["mode"] = "grouped_partial_lowrank"
    elif kind == "config":
        m["config"]["optimizer"]["lam"] += .1
    elif kind == "ids":
        m["source_ids"].reverse()
    elif kind == "missing":
        (directory / "factors/latest.pt").unlink()
    else:
        entry = m["files"]["image" if kind == "router" else "kernels"]
        (directory / entry["file"]).write_bytes(b"corrupt dependency")
    write_json(path, m)
    before = hashes(directory)
    with pytest.raises(ValueError):
        LowRankCheckpointPreview.load(directory)
    assert hashes(directory) == before


def test_interrupted_preview_and_resume_keep_original_fit_identity(stopped_fit, tmp_path, banks):
    directory, c = stopped_fit
    m = json.loads((directory / "manifest.json").read_text())
    m["status"] = "interrupted"
    write_json(directory / "manifest.json", m)
    before = hashes(directory)
    mapper = LowRankCheckpointPreview.load(directory)
    result, metadata = mapper.project_bank(banks[2], tmp_path / "preview", count=16)
    assert metadata["checkpoint_preview"]["original_status"] == "interrupted"
    assert metadata["ids"] == banks[2].ids
    assert hashes(directory) == before
    # Increase ONLY the allowed budget. A second max_steps is fine here: it
    # proves resume accepted the same identity and advanced the saved factors.
    c["optimizer"]["max_steps"] = 4
    with pytest.raises(RuntimeError, match="Low-rank InfoOT max_steps"):
        experiment.fit(c, root=tmp_path, resume=directory)
    checkpoint = torch.load(directory / "factors/latest.pt", weights_only=True)
    assert checkpoint["step"] == 4 and checkpoint["fit_fingerprint"] == m["fit_fingerprint"]
    assert file_hash(directory / m["files"]["image"]["file"]) == m["files"]["image"]["sha256"]


def test_cli_preview_generation_records_fixed_queries_noise_and_visible_label(stopped_fit, tmp_path, banks, monkeypatch, capsys):
    from diffusion_ot.evaluation import stage1a_eval
    from diffusion_ot.data import ground_truth
    directory, _ = stopped_fit
    before = hashes(directory)
    forbid_fitting(monkeypatch)
    checkpoint = tmp_path / "pdae.pt"
    checkpoint.write_bytes(b"fixed test PDAE checkpoint")
    cfg = tmp_path / "dog.yaml"
    cfg.write_text(yaml.safe_dump(dict(domain="dog", data_config="unused.yaml", class_conditioning=dict(null_label=1000))))
    encoder = SimpleNamespace(snapshot_identity=banks[1].representation["encoder"],
        architecture_spec=dict(features=banks[1].representation["layer"]))
    evaluator = SimpleNamespace(branch=SimpleNamespace(encoder=encoder), transformer=None, vae=None,
        device="cpu", model_dtype=torch.float32, checkpoint_path=checkpoint)
    monkeypatch.setattr(stage1a_eval, "load_stage1a_evaluator", lambda *a, **kw: evaluator)
    observations = []
    def sample(branch, transformer, noise, tokens, **kwargs):
        observations.append((noise.clone(), tokens.clone(), kwargs))
        return noise
    monkeypatch.setattr(stage1a_eval, "integrate_pdae_flow", sample)
    monkeypatch.setattr(stage1a_eval, "decode_vae_latents", lambda vae, z, **kw: torch.zeros(len(z), 3, 8, 8))
    monkeypatch.setattr(ground_truth, "load_ground_truth_images", lambda cfg, records: torch.full((len(records), 3, 8, 8), .5))
    args = ["--mapping", str(directory), "--query-bank", str(banks[2].path), "--preview-checkpoint",
            "--generate", "--train-config", str(cfg), "--device", "cpu", "--threads", "1"]
    output = tmp_path / "cli-preview"
    capsys.readouterr()
    assert infoot_test.main(args + ["--output-dir", str(output), "--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["checkpoint_preview"]["step"] == 2 and not output.exists()
    assert infoot_test.main(args + ["--output-dir", str(output), "--chunk-size", "1"]) == 0
    result, _, mapped_manifest = load_mapped(output)
    report = json.loads((output / "generation_report.json").read_text())
    snapshot = json.loads(next((output / "logs").glob("*/checkpoint_preview_snapshot.json")).read_text())
    assert report["checkpoint_preview"] == mapped_manifest["checkpoint_preview"] == dry["checkpoint_preview"]
    assert snapshot["status"] == "failed" and snapshot["artifact_id"] == report["mapper_id"] == dry["mapper_id"]
    assert json.loads((output / "checkpoint_preview.json").read_text()) == report["checkpoint_preview"]
    assert report["query_ids"] == banks[2].ids
    assert report["row_order"] == ["original_source", "top1_routed_target_reference_not_ground_truth", "translated"]
    assert "UNCONVERGED CHECKPOINT PREVIEW | step 2" in report["checkpoint_preview"]["label"]
    with Image.open(output / "translation_grid.png") as image:
        assert image.height > 3*8 + 4*4  # Three rows plus the preview header.
    torch.testing.assert_close(observations[0][1], result.mapped_features.float())
    assert torch.equal(observations[0][2]["condition_padding_mask"], ~result.valid_mask)
    second = tmp_path / "cli-preview-repeat"
    assert infoot_test.main(args + ["--output-dir", str(second), "--chunk-size", "2"]) == 0
    second_report = json.loads((second / "generation_report.json").read_text())
    assert second_report["per_image_noise_seeds"] == report["per_image_noise_seeds"]
    torch.testing.assert_close(observations[0][0], observations[1][0], rtol=0, atol=0)
    assert hashes(directory) == before


def test_preview_cannot_write_into_fit_or_accept_training_queries(stopped_fit, banks):
    directory, _ = stopped_fit
    before = hashes(directory)
    args = ["--mapping", str(directory), "--query-bank", str(banks[2].path), "--preview-checkpoint"]
    for dry in ([], ["--dry-run"]):
        with pytest.raises(ValueError, match="outside the original"):
            infoot_test.main(args + ["--output-dir", str(directory / "preview")] + dry)
    mapper = LowRankCheckpointPreview.load(directory)
    with pytest.raises(ValueError, match="outside the original"):
        mapper.project_bank(banks[2], directory / "preview")
    with pytest.raises(ValueError, match="held-out"):
        infoot_test.main(["--mapping", str(directory), "--query-bank", str(banks[0].path),
                         "--preview-checkpoint", "--dry-run"])
    with pytest.raises(SystemExit):
        infoot_test.main(args + ["--allow-failed-pairs"])
    assert hashes(directory) == before
