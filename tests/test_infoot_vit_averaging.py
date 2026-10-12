"""Independent averaging decomposition and matched mapping-only experiment."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from test_infoot_vit_mapping import banks, cpu_threads
from test_infoot_vit_displacement import example
from test_infoot_vit_displacement_runner import experiment
from test_infoot_vit_residual_experiment import completed_v7
from test_infoot_vit_umap import fake_umap
from test_infoot_vit_projection_overrides import hashes, forbid_fitting
from infoot_vit import compare_displacement as runner
from infoot_vit import averaging_experiment
from infoot_vit.infoot_helper.averaging_diagnostic import AveragingObserver, METHOD_TITLES
from infoot_vit.infoot_helper.conditional_displacement import conditional_displacement
from infoot_vit.infoot_helper.feature_bank import file_hash, write_json
from infoot_vit.infoot_helper.mapping import load_mapped


def toy_observer(**kwargs):
    y = torch.tensor([[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [2., 2., 0.]],
                      [[10., -3., 1.], [12., -3., 1.], [10., -1., 1.], [12., -1., 1.]]], dtype=torch.float64)
    mapper = SimpleNamespace(y=y, target=SimpleNamespace(ids=["dog_b", "dog_a"], representation={"grid": [2, 2]}),
        config={"projection": {"confidence": {"threshold": .05, "mass_floor": 1e-12}}})
    return AveragingObserver(mapper, **kwargs)


def toy_mass():
    beta = torch.tensor([[[.6, .4, 0., 0.], [0., 0., .5, .5], [.25, .25, .25, .25], [0., 0., 0., 1.]],
                         [[0., .2, .8, 0.], [1., 0., 0., 0.], [.1, .2, .3, .4], [.5, 0., .5, 0.]]], dtype=torch.float64)
    mass = torch.tensor([[.56, .35, 0., .02], [.15, .15, .21, .12]], dtype=torch.float64)
    return mass, beta


def test_exported_edge_weights_match_independent_marginal_without_changing_values():
    q, x, y, plan, logs, ky, b = example()
    before, stats = conditional_displacement(q, x, y, plan, logs, ky, b, retain_source_residual=False)
    actual, diagnostic = conditional_displacement(q, x, y, plan, logs, ky, b,
        retain_source_residual=False, return_edge_weights=True)
    density = ky @ b[..., None]
    importance = (ky.transpose(-1, -2) @ (b[..., None] / density)).squeeze(-1)
    joint = logs.exp()[..., None] * plan[:, None] * importance[:, None, None]
    expected = joint.sum(-2) / joint.sum((-1, -2))[..., None]
    assert torch.equal(actual, before) and all(torch.equal(stats[k], diagnostic[k]) for k in stats)
    torch.testing.assert_close(diagnostic["edge_weights"], expected, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(diagnostic["edge_weights"] @ y, actual, atol=2e-14, rtol=2e-14)
    plan[1] = 0
    _, d = conditional_displacement(q, x, y, plan, logs - 10000, ky, b,
        retain_source_residual=False, return_edge_weights=True)
    assert not d["edge_weights"][1].any()
    torch.testing.assert_close(d["edge_weights"][0], expected[0], atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("block", [1, 2, 8])
def test_variance_oracle_reconstruction_common_mask_and_target_argmax(block):
    observer = toy_observer(target_block_size=block)
    mass, beta = toy_mass()
    alpha = torch.tensor([.7, .3], dtype=torch.float64)
    confidence = mass.sum(0)
    mapped = torch.einsum('jq,jqk,jkd->qd', mass, beta, observer.target) / confidence[:, None]
    before = [v.clone() for v in (mass, beta, mapped, confidence, observer.target)]
    observer.begin("cat", mapped, alpha)
    # Two contributions to the same dog, including repeated targets in a block.
    observer.add([0, 1, 0], torch.stack([mass[0] * .25, mass[1], mass[0] * .75]), beta[[0, 1, 0]])
    observer.finish(mapped, confidence, torch.ones(4, dtype=torch.bool))
    row = observer.records[0]
    expected_within, expected_between, expected_total = [], [], []
    for q in range(4):
        mu = mapped[q]
        within = between = total = 0.
        for j in range(2):
            mean_j = sum(beta[j, q, k] * observer.target[j, k] for k in range(4))
            pi = mass[j, q] / confidence[q]
            between += pi * ((mean_j - mu)**2).mean()
            for k in range(4):
                within += pi * beta[j, q, k] * ((observer.target[j, k] - mean_j)**2).mean()
                total += pi * beta[j, q, k] * ((observer.target[j, k] - mu)**2).mean()
        expected_within.append(float(within)); expected_between.append(float(between)); expected_total.append(float(total))
    for name, expected in (("within_target", expected_within), ("between_targets", expected_between), ("total", expected_total)):
        torch.testing.assert_close(torch.tensor(row["per_patch"][name]), torch.tensor(expected))
    assert row["common_valid_mask"] == [True, True, False, False]
    assert row["removed_by_common_mask"] == 2
    assert row["selected_patch_indices"] == [0, 2, -1, -1]  # Stable row-major tie.
    torch.testing.assert_close(observer.argmax_maps[0][:2], observer.target[0, [0, 2]])
    torch.testing.assert_close(observer.means[0][:2], (beta[0] @ observer.target[0])[:2])
    assert not observer.argmax_maps[0][2:].any() and not observer.means[0][2:].any()
    assert row["reconstruction_max_abs"] < 1e-12 and row["variance_identity_max_abs"] < 1e-12
    assert observer.edge_mass is None
    for old, new in zip(before, (mass, beta, mapped, confidence, observer.target)):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    json.dumps(observer.report(), allow_nan=False)


@pytest.mark.parametrize("case", ["one_dog_one_patch", "identical_dogs", "all_rejected", "dominant_unsupported"])
def test_diagnostic_endpoints_and_no_reference_or_source_fallback(case):
    observer = toy_observer()
    mass, beta = toy_mass()
    alpha = torch.tensor([.7, .3], dtype=torch.float64)
    if case == "one_dog_one_patch":
        mass[0], mass[1] = .7, 0
        beta.zero_(); beta[:, :, 2] = 1
    elif case == "identical_dogs":
        observer.target[1] = observer.target[0]
        beta[1] = beta[0]
    elif case == "all_rejected":
        mass.zero_()
    else:
        mass[0] = 0
    confidence = mass.sum(0)
    valid = confidence >= .05
    mapped = torch.einsum('jq,jqk,jkd->qd', mass, beta, observer.target) / confidence.masked_fill(confidence == 0, 1)[:, None]
    observer.begin("cat", mapped, alpha)
    observer.add([0, 1], mass, beta)
    observer.finish(mapped, confidence, valid)
    row = observer.records[0]
    if case == "one_dog_one_patch":
        assert row["conditional_dispersion_common_mask"]["total"]["mean"] == 0
        assert row["hard_selection_coherence"]["unique_target_patches"] == 1
        torch.testing.assert_close(observer.means[0], observer.argmax_maps[0])
    elif case == "identical_dogs":
        assert row["conditional_dispersion_original_mask"]["between_targets"]["mean"] < 1e-12
    else:
        assert row["all_invalid_common"] and not observer.masks[0].any()
        assert not observer.means[0].any() and not observer.argmax_maps[0].any()
    json.dumps(observer.report(), allow_nan=False)


def test_stable_dog_tie_and_resource_guard():
    observer = toy_observer(max_host_bytes=1)
    with pytest.raises(ValueError, match="CPU scratch"):
        observer.begin("cat", observer.target[0], torch.tensor([.5, .5], dtype=torch.float64))
    assert observer.edge_mass is None
    observer = toy_observer()
    observer.begin("cat", observer.target[0], torch.tensor([.5, .5], dtype=torch.float64))
    assert observer.dominant == 1  # dog_a, regardless of physical support order.


@pytest.mark.parametrize("broken", ["weights", "confidence", "mapped"])
def test_corrupted_distribution_or_reconstruction_fails(broken):
    observer = toy_observer()
    mass, beta = toy_mass()
    confidence = mass.sum(0)
    mapped = torch.einsum('jq,jqk,jkd->qd', mass, beta, observer.target) / confidence[:, None]
    observer.begin("cat", mapped, torch.tensor([.7, .3], dtype=torch.float64))
    with pytest.raises(ValueError):
        if broken == "weights":
            beta *= .5
        observer.add([0, 1], mass, beta)
        observer.finish(mapped + (broken == "mapped"), confidence + (broken == "confidence"), torch.ones(4, dtype=torch.bool))


@pytest.fixture
def completed_v8(completed_v7, fake_umap):
    args, _, loaded, sampled = completed_v7
    previous = runner.run_experiment(args)
    args.reference_run = args.output_dir
    args.output_dir, args.work_dir = args.output_dir.parent / "averaging", args.work_dir.parent / "averaging_work"
    args.experiment = "averaging_ablation"
    loaded.clear(); sampled.clear(); fake_umap.clear()
    return args, previous, loaded, sampled


def test_full_experiment_exact_replay_no_refit_shared_noise_masks_and_three_files(completed_v8, fake_umap, monkeypatch):
    args, previous, loaded, sampled = completed_v8
    before, reference_before = hashes(args.mapping), hashes(args.reference_run)
    forbid_fitting(monkeypatch)
    monkeypatch.setattr(runner, "fit_mapping", lambda *a, **k: pytest.fail("No refitting"))
    report = runner.run_experiment(args)
    assert report["status"] == "complete" and report["quality_acceptance"] == "unreviewed"
    assert report["mapper_id"] == previous["mapper_id"] and report["initial_noise_sha256"] == previous["initial_noise_sha256"]
    assert report["original_valid_mask_sha256"] == previous["valid_mask_sha256"]
    assert report["original_confidence_sha256"] == previous["confidence_sha256"]
    original, _, _ = load_mapped(args.work_dir / "conditional_edge_mean")
    old, _, _ = load_mapped(Path(previous["work_directory"]) / "conditional_edge_mean")
    assert torch.equal(original.mapped_features, old.mapped_features)
    assert torch.equal(original.match_confidence, old.match_confidence) and torch.equal(original.valid_mask, old.valid_mask)
    assert len(loaded) == 1 and len(sampled) == 7 and len(report["row_order"]) == 9
    assert all(torch.equal(s[0], sampled[0][0]) for s in sampled)
    for i in (1, 2, 4, 5, 6):
        assert torch.equal(sampled[i][2]["condition_padding_mask"], sampled[4][2]["condition_padding_mask"])
    assert torch.equal(sampled[3][2]["condition_padding_mask"], ~original.valid_mask)
    cache = torch.load(args.work_dir / "averaging_variants.pt", weights_only=True)
    assert not (cache["common_valid_mask"] & ~cache["original_valid_mask"]).any()
    for i, name in enumerate(METHOD_TITLES):
        torch.testing.assert_close(sampled[4+i][1], cache["variants"][name].float())
    assert set(report["diagnostic_variants"]) == set(METHOD_TITLES) == set(report["umap"]["mapped_artifact_ids"])
    assert len(fake_umap) == 2 and report["umap"]["method_titles"] == METHOD_TITLES
    assert report["umap"]["sha256"] == file_hash(args.output_dir / "umap.png")
    assert {p.name for p in args.output_dir.rglob('*')} == {"experiment_report.json", "comparison_grid.png", "umap.png"}
    assert not report["quality_review"]["automatic_quality_pass"]
    assert not (args.work_dir / "fit").exists()
    assert hashes(args.mapping) == before and hashes(args.reference_run) == reference_before


@pytest.mark.parametrize("change", ["refit", "v7", "decoder_config", "mask", "noise", "umap", "empty_common_mask"])
def test_averaging_contract_stops_before_sampling(completed_v8, monkeypatch, change):
    args, previous, loaded, sampled = completed_v8
    if change == "refit":
        args.fit_config = args.mapping / "anything.yaml"
    elif change == "v7":
        args.reference_run = args.reference_run.parent / "comparison"
    elif change == "decoder_config":
        args.eval_config.write_text("changed: true\n")
    elif change in {"mask", "noise"}:
        if change == "mask":
            previous["valid_mask_sha256"] = '0' * 64
        else:
            previous["initial_noise_sha256"] = '0' * 64
            previous["mapping_outputs"]["barycentric"]["generation"]["initial_noise_sha256"] = '0' * 64
        write_json(args.reference_run / "experiment_report.json", previous)
    elif change == "umap":
        monkeypatch.setitem(sys.modules, "umap", None)
    else:
        original = AveragingObserver.finish
        def empty(self, *a):
            original(self, *a)
            self.masks[-1].zero_()
            self.records[-1]["all_invalid_common"] = True
        monkeypatch.setattr(AveragingObserver, "finish", empty)
    with pytest.raises((ValueError, RuntimeError)):
        runner.run_experiment(args)
    assert not sampled
    if args.output_dir.exists():
        report = json.loads((args.output_dir / "experiment_report.json").read_text())
        assert report["status"] == "failed"
        if change == "empty_common_mask":
            assert report["averaging_diagnostic"]["queries"][0]["all_invalid_common"]
    else:
        assert not args.work_dir.exists()


def test_plot_failure_keeps_diagnostic_cache_and_failed_report(completed_v8, monkeypatch):
    args, _, loaded, sampled = completed_v8
    def fail(*a, **k):
        raise RuntimeError("synthetic UMAP failure")
    monkeypatch.setattr(averaging_experiment.comparison_umap, "save_comparison_umap", fail)
    with pytest.raises(RuntimeError, match="synthetic UMAP failure"):
        runner.run_experiment(args)
    report = json.loads((args.output_dir / "experiment_report.json").read_text())
    assert report["status"] == "failed" and len(report["averaging_diagnostic"]["queries"]) == 2
    assert (args.work_dir / "averaging_variants.pt").is_file() and not loaded and not sampled
