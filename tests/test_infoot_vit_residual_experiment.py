"""Matched residual endpoint, exact v7 reuse, and required compact UMAP output."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_infoot_vit_mapping import banks, cpu_threads
from test_infoot_vit_displacement_runner import experiment, stub_decoder
from test_infoot_vit_projection_overrides import hashes, forbid_fitting
from test_infoot_vit_umap import fake_umap
from infoot_vit import compare_displacement as runner
from infoot_vit.infoot_helper import comparison_umap
from infoot_vit.infoot_helper.feature_bank import digest, file_hash, write_json
from infoot_vit.infoot_helper.mapping import load_mapped


@pytest.fixture
def completed_v7(experiment, banks, monkeypatch, tmp_path):
    args, fit, config = experiment
    evaluator, loaded, sampled = stub_decoder(monkeypatch, args, banks[0].representation)
    report = runner.run_experiment(args)
    args.reference_run = args.output_dir
    args.output_dir, args.work_dir = tmp_path / "residual", tmp_path / "residual_work"
    args.experiment = "residual_ablation"
    loaded.clear()
    sampled.clear()
    return args, report, loaded, sampled


def test_residual_reuses_exact_fit_masks_noise_and_writes_only_three_files(completed_v7, banks, monkeypatch, fake_umap):
    args, previous, loaded, sampled = completed_v7
    before, reference_before = hashes(args.mapping), hashes(args.reference_run)
    forbid_fitting(monkeypatch)
    monkeypatch.setattr(runner, "fit_mapping", lambda *a, **k: pytest.fail("Residual experiment must not fit"))
    report = runner.run_experiment(args)
    assert report["status"] == "complete" and report["schema"] == "infoot_residual_ablation_v1"
    assert report["mapper_id"] == previous["mapper_id"]
    assert report["initial_noise_sha256"] == previous["initial_noise_sha256"]
    assert report["confidence_sha256"] == previous["confidence_sha256"]
    assert report["valid_mask_sha256"] == previous["valid_mask_sha256"]
    assert len(loaded) == 1 and len(sampled) == 6
    assert all(torch.equal(sampled[0][0], row[0]) for row in sampled)
    assert all(torch.equal(sampled[1][2]["condition_padding_mask"], row[2]["condition_padding_mask"]) for row in sampled[2:])
    base, _, _ = load_mapped(args.work_dir / "barycentric")
    for mode, index in (("conditional_displacement", 4), ("conditional_edge_mean", 5)):
        result, _, manifest = load_mapped(args.work_dir / mode)
        runner.verify_paired_mapping(base, result)
        assert manifest["projection"]["patch_value_mode"] == mode
        torch.testing.assert_close(sampled[index][1], result.mapped_features.float())
    assert len({x["manifest"]["artifact_id"] for x in report["mapping_outputs"].values()}) == 3
    assert report["primary_comparison"] == ["conditional_displacement", "conditional_edge_mean"]
    assert len(report["row_order"]) == 8 and report["row_order"][-1] == "mapped_conditional_edge_mean"
    assert report["quality_review"]["automatic_quality_pass"] is False
    assert {p.name for p in args.output_dir.rglob("*")} == {"experiment_report.json", "comparison_grid.png", "umap.png"}
    assert set(report["analysis_files"]) == {p.name for p in args.output_dir.iterdir()}
    assert report["umap"]["sha256"] == file_hash(args.output_dir / "umap.png")
    assert len(fake_umap) == 2
    assert hashes(args.mapping) == before and hashes(args.reference_run) == reference_before
    assert not (args.work_dir / "fit").exists()


@pytest.mark.parametrize("change", ["refit", "legacy_reference", "different_complete_fit", "decoder_config", "masks", "noise"])
def test_residual_contract_rejects_changed_reference(completed_v7, monkeypatch, fake_umap, change):
    args, previous, loaded, sampled = completed_v7
    reference_path = args.reference_run / "experiment_report.json"
    if change == "refit":
        args.fit_config, args.mapping = args.output_dir / "new.yaml", None
    elif change == "legacy_reference":
        args.reference_run = args.reference_run.parent / "reference"
    elif change == "different_complete_fit":
        path = args.mapping / "manifest.json"
        fit = json.loads(path.read_text())
        fit["local_revision"] = "different-fit"
        fit["artifact_id"] = digest({k: v for k, v in fit.items() if k != "artifact_id"})
        write_json(path, fit)
    elif change == "decoder_config":
        args.eval_config.write_text("changed: true\n")
    else:
        if change == "masks":
            previous["valid_mask_sha256"] = "0" * 64
        else:
            previous["initial_noise_sha256"] = "0" * 64
            previous["mapping_outputs"]["barycentric"]["generation"]["initial_noise_sha256"] = "0" * 64
        write_json(reference_path, previous)
    with pytest.raises(ValueError):
        runner.run_experiment(args)
    assert not sampled  # Even a changed instantiated-noise hash stops before sampling.
    if (args.output_dir / "experiment_report.json").exists():
        assert json.loads((args.output_dir / "experiment_report.json").read_text())["status"] == "failed"
    else:
        assert not args.output_dir.exists()


def test_missing_required_umap_fails_before_creating_output(completed_v7, monkeypatch):
    args, _, loaded, sampled = completed_v7
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(RuntimeError, match="pip install umap-learn matplotlib"):
        runner.run_experiment(args)
    assert not args.output_dir.exists() and not args.work_dir.exists() and not loaded and not sampled


def test_umap_failure_preserves_mapping_and_never_marks_experiment_complete(completed_v7, monkeypatch, fake_umap):
    args, _, loaded, sampled = completed_v7
    def fail(*a, **k):
        raise RuntimeError("UMAP display failed")
    monkeypatch.setattr(comparison_umap, "save_comparison_umap", fail)
    with pytest.raises(RuntimeError, match="UMAP display failed"):
        runner.run_experiment(args)
    report = json.loads((args.output_dir / "experiment_report.json").read_text())
    assert report["status"] == "failed" and report["quality_acceptance"] == "unreviewed"
    assert not loaded and not sampled
    assert all((args.work_dir / mode / "mapped.pt").exists() for mode in comparison_umap.MODES)


def umap_inputs(banks):
    source, target, query = banks
    mapper = SimpleNamespace(source=source, target=target, manifest={"artifact_id": "fit"})
    mask = torch.tensor([[True, False, True, False], [False, False, False, False]])
    results = {}
    for i, mode in enumerate(comparison_umap.MODES):
        features = query.features.clone() + i + 1
        features[~mask] = float("nan")  # Masked entries never enter either readout.
        results[mode] = SimpleNamespace(mapped_features=features, valid_mask=mask.clone(),
            diagnostics={"queries": [{"query_id": sid} for sid in query.ids]})
    return mapper, query, results, {mode: mode + "_id" for mode in results}


def test_comparison_umap_training_only_shared_tokens_coordinates_and_no_extra_json(tmp_path, banks, fake_umap):
    mapper, query, results, ids = umap_inputs(banks)
    originals = {mode: x.mapped_features.clone() for mode, x in results.items()}
    report = comparison_umap.save_comparison_umap(mapper, query, results, tmp_path, mapped_ids=ids)
    assert len(fake_umap) == 2 and set(report["panels"]) == {"image", "patch"}
    assert report["all_invalid_query_ids"] == [query.ids[1]]
    for level, reducer in zip(("image", "patch"), fake_umap):
        expected = torch.cat([mapper.source.features.mean(1), mapper.target.features.mean(1)]) if level == "image" else torch.cat([mapper.source.features.flatten(0, 1), mapper.target.features.flatten(0, 1)])
        np.testing.assert_allclose(reducer.training, expected.float().numpy(), rtol=1e-6, atol=1e-7)
        vectors = [query.features[0, [0, 2]].float(), *[r.mapped_features[0, [0, 2]].float() for r in results.values()]]
        expected_queries = torch.cat([v.mean(0, keepdim=True) if level == "image" else v for v in vectors])
        np.testing.assert_allclose(reducer.queries, expected_queries.numpy(), rtol=1e-6, atol=1e-7)
        groups = report["panels"][level]["groups"]
        source_records = [{k:v for k,v in point.items() if k != "xy"} for point in groups["source_query"]["points"]]
        for mode in results:
            assert source_records == [{k:v for k,v in point.items() if k != "xy"} for point in groups[mode]["points"]]
            np.testing.assert_allclose([p["xy"] for p in groups[mode]["points"]],
                (results[mode].mapped_features[0, [0, 2]].float().mean(0, keepdim=True) if level == "image" else results[mode].mapped_features[0, [0, 2]].float())[:, :2].numpy())
        assert reducer.kwargs["n_jobs"] == 1 and reducer.kwargs["random_state"] == reducer.kwargs["transform_seed"] == 42
    assert {p.name for p in tmp_path.iterdir() if p.is_file()} == {"umap.png"}
    assert (tmp_path / "umap.png").stat().st_size > 1000
    for mode, result in results.items():
        torch.testing.assert_close(result.mapped_features, originals[mode], rtol=0, atol=0, equal_nan=True)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("change", ["mask", "query_ids", "training_split", "heldout_split", "finite", "coordinates"])
def test_comparison_umap_rejects_mismatched_or_invalid_inputs(tmp_path, banks, fake_umap, monkeypatch, change):
    mapper, query, results, ids = umap_inputs(banks)
    if change == "mask":
        results["conditional_edge_mean"].valid_mask[0, 0] = False
    elif change == "query_ids":
        results["conditional_edge_mean"].diagnostics["queries"].reverse()
    elif change == "training_split":
        mapper.source.manifest["split"] = "val"
    elif change == "heldout_split":
        query.manifest["split"] = "train"
    elif change == "finite":
        results["conditional_edge_mean"].mapped_features[0, 0, 0] = float("nan")
    else:
        import umap
        monkeypatch.setattr(umap.UMAP, "transform", lambda self, values: np.full((len(values), 2), np.nan))
    with pytest.raises((ValueError, RuntimeError)):
        comparison_umap.save_comparison_umap(mapper, query, results, tmp_path, mapped_ids=ids)
    assert not (tmp_path / "umap.png").exists()


def test_actual_comparison_umap_finite_plot(tmp_path, banks):
    pytest.importorskip("umap")
    mapper, query, results, ids = umap_inputs(banks)
    report = comparison_umap.save_comparison_umap(mapper, query, results, tmp_path,
        mapped_ids=ids, max_points=8, n_neighbors=3)
    assert report["sha256"] == file_hash(tmp_path / "umap.png")
    for panel in report["panels"].values():
        assert all(np.isfinite([p["xy"] for p in group["points"]]).all() for group in panel["groups"].values())
