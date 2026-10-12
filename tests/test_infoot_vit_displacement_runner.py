"""End-to-end fitted-artifact/mapping checks with a stubbed diffusion decoder."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from test_infoot_vit_mapping import banks, config, cpu_threads
from test_infoot_vit_projection_overrides import hashes
from infoot_vit import compare_displacement as runner
from infoot_vit.infoot_helper.feature_bank import file_hash, write_json
from infoot_vit.infoot_helper.fit_mapping import fit_mapping
from infoot_vit.infoot_helper.mapping import FeatureMapper, load_mapped
from infoot_vit.infoot_helper.experiment_output import create_work_directory, fit_diagnostics


@pytest.fixture
def experiment(tmp_path, banks):
    c = config("grouped_partial")
    c.update(pair_batch_size=3, fit_pair_top_k=2)
    c["projection"].update(bandwidth_multiplier=.2/.7, patch_bandwidth=.2)
    directory = fit_mapping(c, root=tmp_path)
    fit = json.loads((directory / "manifest.json").read_text())
    mapper = FeatureMapper.load(directory)
    reference_dir = tmp_path / "reference"
    result, manifest = mapper.project_bank(banks[2], reference_dir)
    write_json(reference_dir / "fitting_logs/manifest.json", fit)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fake-frozen-dog-checkpoint")
    ids = banks[2].ids
    seeds = [int.from_bytes(hashlib.sha256(f"20260903:{sid}".encode()).digest()[:8], "big") % (2**63 - 1) for sid in ids]
    generation = dict(mapper_id=fit["artifact_id"], query_ids=ids,
        projection=manifest["projection"], confidence_policy=manifest["projection"]["confidence"],
        target_domain="dog", checkpoint_sha256=file_hash(checkpoint), weights="raw", num_steps=40,
        guidance_scale=1.5, solver="euler", seed=20260903, per_image_noise_seeds=seeds)
    write_json(reference_dir / "generation_report.json", generation)
    train = tmp_path / "dog.yaml"
    train.write_text("domain: dog\ndata_config: unused.yaml\nclass_conditioning:\n  null_label: 1000\n")
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text("{}\n")
    args = SimpleNamespace(mapping=directory, fit_config=None, source_bank=None, target_bank=None,
        reference_run=reference_dir, query_bank=banks[2].path, checkpoint=checkpoint, output_dir=tmp_path / "comparison",
        work_dir=tmp_path / "work",
        train_config=train, eval_config=eval_config, device="cpu", batch_size=1, pair_cache_mib=1,
        projection_pair_batch_size=2)
    return args, fit, c


def stub_decoder(monkeypatch, args, representation):
    from diffusion_ot.evaluation import stage1a_eval
    from diffusion_ot.data import ground_truth
    branch = torch.nn.Module()
    branch.encoder = SimpleNamespace(snapshot_identity=representation["encoder"],
        architecture_spec=dict(features=representation["layer"]))
    branch.semantic_transformer = torch.nn.Module()
    branch.semantic_transformer.feature_projector = torch.nn.Sequential(
        torch.nn.LayerNorm(3), torch.nn.Linear(3, 3), torch.nn.GELU(), torch.nn.Linear(3, 3))
    evaluator = SimpleNamespace(branch=branch, transformer=torch.nn.Linear(2, 2), vae=torch.nn.Identity(),
        device="cpu", model_dtype=torch.float32, checkpoint_path=args.checkpoint)
    loaded, sampled = [], []
    def load(*params, **kwargs):
        loaded.append(kwargs)
        return evaluator
    def integrate(branch, transformer, noise, features, **kwargs):
        sampled.append((noise.clone(), features.clone(), kwargs))
        noise.add_(71)
        return noise
    monkeypatch.setattr(stage1a_eval, "load_stage1a_evaluator", load)
    monkeypatch.setattr(stage1a_eval, "integrate_pdae_flow", integrate)
    monkeypatch.setattr(stage1a_eval, "decode_vae_latents", lambda vae, z, **kwargs: torch.zeros(len(z), 3, 8, 8))
    monkeypatch.setattr(ground_truth, "load_ground_truth_images", lambda path, records: torch.ones(len(records), 3, 8, 8))
    return evaluator, loaded, sampled


@pytest.mark.parametrize("fresh_fit", [False, True])
def test_fit_then_map_pair_preserves_noise_checkpoint_and_mask(experiment, banks, tmp_path, monkeypatch, fresh_fit):
    args, fit, c = experiment
    before = hashes(args.mapping)
    reference_before = hashes(args.reference_run)
    evaluator, loaded, sampled = stub_decoder(monkeypatch, args, banks[0].representation)
    fit_calls = []
    original_fit = runner.fit_mapping
    def observe_fit(*params, **kwargs):
        assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        fit_calls.append(params[0])
        return original_fit(*params, **kwargs)
    monkeypatch.setattr(runner, "fit_mapping", observe_fit)
    old_directory = args.mapping
    if fresh_fit:
        c["source_bank"], c["target_bank"] = str(banks[0].path), str(banks[1].path)
        args.fit_config = tmp_path / "candidate.yaml"
        args.fit_config.write_text(yaml.safe_dump(c))
        args.mapping = None
    report = runner.run_experiment(args)
    assert len(fit_calls) == int(fresh_fit)
    assert report["status"] == "complete" and report["quality_acceptance"] == "unreviewed"
    assert report["successful_pairs"] == 4 and report["failed_pairs"] == 0
    assert len(loaded) == 1 and loaded[0]["weights"] == "raw" and len(sampled) == 5
    assert not torch.equal(sampled[0][0][0], sampled[0][0][1])
    assert all(torch.equal(row[0], sampled[0][0]) for row in sampled)
    assert all(row[2]["num_steps"] == 40 and row[2]["guidance_scale"] == 1.5 for row in sampled)
    baseline, _, bm = load_mapped(args.work_dir / "barycentric")
    candidate, _, cm = load_mapped(args.work_dir / "conditional_displacement")
    runner.verify_paired_mapping(baseline, candidate)
    assert bm["mapper_id"] == cm["mapper_id"] == report["mapper_id"]
    torch.testing.assert_close(sampled[3][1], baseline.mapped_features.float())
    torch.testing.assert_close(sampled[4][1], candidate.mapped_features.float())
    torch.testing.assert_close(sampled[2][1], banks[2].features.float())
    assert all(torch.equal(row[2]["condition_padding_mask"], ~baseline.valid_mask) for row in sampled[1:])
    assert all(not module.training for module in (evaluator.branch, evaluator.transformer, evaluator.vae))
    assert not any(p.requires_grad for p in evaluator.branch.parameters())
    assert hashes(old_directory) == before and hashes(args.reference_run) == reference_before
    review = report["quality_review"]
    assert review["status"] == "unreviewed" and review["automatic_quality_pass"] is False
    assert all(row["verdict"] is None for row in review["per_query"])
    assert {p.name for p in args.output_dir.rglob("*")} == {"experiment_report.json", "comparison_grid.png"}
    assert report["mapping_outputs"]["barycentric"]["manifest"] == bm
    assert report["mapping_outputs"]["conditional_displacement"]["manifest"] == cm
    assert report["mapping_outputs"]["barycentric"]["summary"]["mode"] == "grouped_partial"
    assert report["fit_diagnostics"]["files"]["manifest.json"]["data"]["pair_count"] == 4
    assert not report["fit_diagnostics"]["export_errors"]
    assert report["comparison_grid"]["sha256"] == file_hash(args.output_dir / "comparison_grid.png")
    saved = torch.load(args.work_dir / "conditioning_variants.pt", weights_only=True)
    assert torch.equal(saved["initial_noise"], sampled[0][0])
    with pytest.raises(FileExistsError):
        runner.run_experiment(args)


@pytest.mark.parametrize("change", ["checkpoint", "query", "sampling", "partial", "projection", "inventory", "failed", "noise"])
def test_contract_rejects_changed_controls_before_decode(experiment, tmp_path, monkeypatch, change):
    args, fit, _ = experiment
    ref = runner.read_reference(args.reference_run)
    if change in {"checkpoint", "query", "noise"}:
        if change == "checkpoint":
            args.checkpoint.write_bytes(b"different-checkpoint")
        elif change == "query":
            args.query_bank = args.query_bank.parent / "source"
        else:
            path = args.reference_run / "generation_report.json"
            report = json.loads(path.read_text())
            report["per_image_noise_seeds"][0] += 1
            write_json(path, report)
        with pytest.raises(ValueError):
            runner.run_experiment(args)
        assert not args.output_dir.exists()
    else:
        changed = deepcopy(fit)
        if change == "sampling":
            changed["config"]["sampling"]["seed"] += 1
        elif change == "partial":
            changed["config"]["partial"]["solver"]["lam"] = .01
        elif change == "projection":
            changed["config"]["projection"]["patch_bandwidth"] = .15
        elif change == "inventory":
            changed["pair_count"] -= 1
        else:
            changed["status"] = "failed"
        with pytest.raises(ValueError):
            runner.validate_complete_fit(changed, ref)


def test_failed_generation_leaves_failure_report_and_no_quality_pass(experiment, banks, monkeypatch):
    args, _, _ = experiment
    stub_decoder(monkeypatch, args, banks[0].representation)
    from diffusion_ot.evaluation import stage1a_eval
    def fail(*params, **kwargs):
        raise RuntimeError("synthetic decoder failure")
    monkeypatch.setattr(stage1a_eval, "integrate_pdae_flow", fail)
    with pytest.raises(RuntimeError, match="synthetic decoder failure"):
        runner.run_experiment(args)
    report = json.loads((args.output_dir / "experiment_report.json").read_text())
    assert report["status"] == "failed" and report["quality_acceptance"] == "unreviewed"
    assert (args.work_dir / "conditional_displacement/mapped.pt").exists()
    assert [p.name for p in args.output_dir.iterdir()] == ["experiment_report.json"]


def test_fit_failure_exports_histories_without_copying_success_logs(experiment, banks, tmp_path, monkeypatch):
    args, _, c = experiment
    c["source_bank"], c["target_bank"] = str(banks[0].path), str(banks[1].path)
    args.fit_config = tmp_path / "candidate.yaml"
    args.fit_config.write_text(yaml.safe_dump(c))
    args.mapping = None
    failure = dict(source_id="source", target_id="target", status="line_search_stalled",
                   report=dict(history=[dict(iteration=1, objective=.123)]))
    def fail(config, *, root, output_root):
        directory = output_root / "failed_fit"
        write_json(directory / "manifest.json", dict(status="failed", pair_count=3, failed_pair_count=1))
        (directory / "pair_failures.jsonl").write_text(json.dumps(failure) + "\n")
        write_json(directory / "plans/pairs/success/iterations.jsonl", dict(retained="locally"))
        raise RuntimeError("one failed pair")
    monkeypatch.setattr(runner, "fit_mapping", fail)
    with pytest.raises(RuntimeError, match="one failed pair"):
        runner.run_experiment(args)
    report = json.loads((args.output_dir / "experiment_report.json").read_text())
    assert report["status"] == "failed"
    assert report["fit_diagnostics"]["files"]["pair_failures.jsonl"]["data"] == [failure]
    assert (args.work_dir / "fit/failed_fit/plans/pairs/success/iterations.jsonl").exists()
    assert [p.name for p in args.output_dir.iterdir()] == ["experiment_report.json"]


def test_work_files_cannot_leak_into_results_or_existing_directories(tmp_path):
    output = tmp_path / "results/infoot_vit/run"
    for work in (output, output / "fit", output.parent, tmp_path / "results/elsewhere"):
        with pytest.raises(ValueError, match="Working files"):
            create_work_directory(tmp_path, output, work)
    work = create_work_directory(tmp_path, output)
    assert work.is_relative_to(tmp_path / "outputs/infoot_vit")
    assert not output.exists()
    (work / "keep.txt").write_text("preserved")
    with pytest.raises(FileExistsError):
        create_work_directory(tmp_path, output, work)
    assert (work / "keep.txt").read_text() == "preserved"


def test_diagnostic_export_preserves_original_bytes_and_reports_partial_json(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_bytes(b'{"status":"failed"}\r\n')
    original_hash = file_hash(path)
    (tmp_path / "pair_failures.jsonl").write_text('{"incomplete":')
    report = fit_diagnostics(tmp_path)
    assert report["files"]["manifest.json"]["sha256"] == original_hash == file_hash(path)
    assert "pair_failures.jsonl" in report["export_errors"]


def test_mapping_comparison_rejects_mask_or_routing_change(experiment):
    args, _, _ = experiment
    original, _, _ = load_mapped(args.reference_run)
    changed = deepcopy(original)
    changed.valid_mask[0, 0] = ~changed.valid_mask[0, 0]
    with pytest.raises(ValueError, match="confidence and masks"):
        runner.verify_paired_mapping(original, changed)
    changed = deepcopy(original)
    changed.diagnostics["queries"][0]["image_weights"][0] += .01
    with pytest.raises(ValueError, match="routing"):
        runner.verify_paired_mapping(original, changed)
