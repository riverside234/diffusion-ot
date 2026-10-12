"""One dense-partial fit, then a matched baseline/displacement translation test.

Use --fit-config for a fresh fit or --mapping for a certified complete fit.
The reviewed run supplies query, bank, solver and generation identities.
Execution completion is explicitly separate from visual quality acceptance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from infoot_vit.diagnose_blur import decode_variants, decoder_conditioning_diagnostics, tensor_hash
from infoot_vit.infoot_test import projection_settings
from infoot_vit.infoot_helper.device import resolve_device
from infoot_vit.infoot_helper.experiment_output import create_work_directory, fit_diagnostics, mapping_diagnostics
from infoot_vit.infoot_helper.feature_bank import FeatureBank, digest, file_hash, save_tensor, write_json
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, validate_config
from infoot_vit.infoot_helper.mapping import FeatureMapper
from infoot_vit.infoot_helper.mapping_manifest import load_mapping_manifest
from infoot_vit.infoot_helper.sampling import sample_ids


def read_reference(directory):
    """Metadata only: do not require the old multi-GB plans or mapped tensors."""
    paths = {"mapped": directory / "manifest.json", "generation": directory / "generation_report.json",
             "fit": directory / "fitting_logs/manifest.json"}
    data = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    mapped, generation, fit = (data[key] for key in ("mapped", "generation", "fit"))
    ids = mapped["ids"]
    if (digest({k: v for k, v in mapped.items() if k != "artifact_id"}) != mapped["artifact_id"]
            or mapped["mode"] != "grouped_partial" or not ids or len(set(ids)) != len(ids)
            or generation["mapper_id"] != mapped["mapper_id"] or generation["query_ids"] != ids
            or generation["projection"] != mapped["projection"]
            or generation["confidence_policy"] != mapped["projection"]["confidence"]
            or generation.get("incomplete_fit") != mapped.get("incomplete_fit")
            or generation["target_domain"] != fit["target_domain"]
            or mapped["representation"] != fit["representation"]):
        raise ValueError("Reference run metadata, generation controls or identities disagree.")
    snapshot = deepcopy(fit)
    if fit.get("status") != "complete":
        if fit.get("status") != "failed" or not mapped.get("incomplete_fit"):
            raise ValueError("Reference fit must be complete or the recorded stopped failed-pair snapshot.")
        snapshot["incomplete_fit"] = mapped["incomplete_fit"]
    expected = digest({k: v for k, v in snapshot.items() if k != "artifact_id"})
    if expected != mapped["mapper_id"]:
        raise ValueError("Reference fit snapshot differs from the reviewed mapper.")
    c = validate_config(fit["config"])
    projection = mapped["projection"]
    if (c["mode"] != "grouped_partial" or projection.get("patch_value_mode", "barycentric") != "barycentric"
            or not math.isclose(c["solver"]["h"] * projection["bandwidth_multiplier"], .2, abs_tol=1e-12, rel_tol=0)
            or not math.isclose(projection.get("patch_bandwidth", 0.), .2, abs_tol=1e-12, rel_tol=0)
            or projection["selection"] != "mean" or projection["top_k_images"] is not None
            or generation["weights"] not in {"raw", "ema"}
            or generation["solver"] not in {"euler", "heun"}
            or type(generation["num_steps"]) is not int or generation["num_steps"] < 1
            or not math.isfinite(generation["guidance_scale"]) or generation["guidance_scale"] < 0):
        raise ValueError("Require the reviewed dense .20/.20 barycentric baseline and valid saved sampling controls.")
    seeds = [int.from_bytes(hashlib.sha256(f"{generation['seed']}:{sid}".encode()).digest()[:8], "big")
             % (2**63 - 1) for sid in ids]
    if seeds != generation["per_image_noise_seeds"]:
        raise ValueError("Reference per-query noise identity differs from its seed and ordered IDs.")
    data["file_sha256"] = {key: file_hash(path) for key, path in paths.items()}
    return data


def validate_fit_config(config, reference):
    candidate, original = validate_config(config), validate_config(reference["fit"]["config"])
    for key in ("mode", "sampling", "fit_pair_top_k", "solver", "partial"):
        if candidate.get(key) != original.get(key):
            raise ValueError(f"Matched displacement experiment cannot change fitting {key}.")
    # Pair selection depends on image projection; require the same policy at fit time too.
    for key in ("bandwidth_multiplier", "patch_bandwidth", "selection", "top_k_images", "seed", "confidence"):
        if candidate["projection"].get(key) != reference["mapped"]["projection"].get(key):
            raise ValueError(f"Matched displacement experiment cannot change projection {key}.")
    return candidate


def validate_complete_fit(fit, reference):
    validate_fit_config(fit["config"], reference)
    expected = len(fit["source_ids"]) * min(fit["config"]["fit_pair_top_k"] or len(fit["target_ids"]), len(fit["target_ids"]))
    if (fit.get("status") != "complete" or fit.get("incomplete_fit")
            or fit.get("failed_pair_count", 0) != 0 or fit["pair_count"] != expected):
        raise ValueError("A certified complete pair inventory is required; failed pairs cannot be skipped here.")
    old = reference["fit"]
    for key in ("source_ids", "target_ids", "source_domain", "target_domain", "representation", "precision"):
        if fit[key] != old[key]:
            raise ValueError(f"Fit {key} differs from the reviewed experiment.")
    for key in ("source_bank", "target_bank"):
        if fit[key]["artifact_id"] != old[key]["artifact_id"]:
            raise ValueError(f"Fit {key} identity differs from the reviewed experiment.")


def verify_paired_mapping(baseline, candidate):
    if (not torch.equal(baseline.match_confidence, candidate.match_confidence)
            or not torch.equal(baseline.valid_mask, candidate.valid_mask)):
        raise ValueError("Matched baseline and displacement must have identical confidence and masks.")
    keys = ("query_id", "image_weights", "target_ids", "image_projection_h", "patch_projection_h",
            "fit_pair_retained_routing_mass", "fit_pair_discarded_routing_mass",
            "ot_rejected_mass", "support_invalid_retained_mass")
    before, after = baseline.diagnostics["queries"], candidate.diagnostics["queries"]
    if len(before) != len(after) or any(a[key] != b[key] for a, b in zip(before, after) for key in keys):
        raise ValueError("Matched baseline and displacement changed queries, routing, bandwidth or rejection.")


def review_template(ids, reference_ids):
    return dict(status="unreviewed", automatic_quality_pass=False,
        rule="Inspect every column at native resolution. Accept only if clarity AND dog realism AND source "
             "pose/expression/color improve without greater reference copying or facial artifacts. "
             "A sharper cat-like image, or a retrieved dog, is not successful translation.",
        per_query=[dict(query_id=sid, reference_id=rid, clarity_vs_baseline=None,
            dog_realism=None, source_pose_expression_color=None, facial_artifacts=None,
            reference_copying=None, verdict=None, notes="") for sid, rid in zip(ids, reference_ids)])


def run_experiment(args):
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Comparison output must be new: {output}")
    if args.mapping and args.mapping.resolve() in output.parents:
        raise ValueError("Write comparison outputs outside the immutable fitted mapper.")
    reference = read_reference(args.reference_run)
    checkpoint_hash = file_hash(args.checkpoint)
    if checkpoint_hash != reference["generation"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the reviewed dog decoder; stop before fitting.")
    device = resolve_device(args.device)
    query = FeatureBank.load(args.query_bank)
    if (query.artifact_id != reference["mapped"]["query_bank_id"]
            or query.manifest["split"] not in {"val", "test"}
            or query.representation != reference["fit"]["representation"]):
        raise ValueError("Require the original compatible held-out query bank.")
    query = query.subset(reference["generation"]["query_ids"])
    from diffusion_ot.integrations.hf_snapshot import load_yaml_config
    train_path, eval_path = ROOT / args.train_config, ROOT / args.eval_config
    train = load_yaml_config(train_path)
    # Check both config paths before committing to the expensive fit.
    file_hash(eval_path)
    if train["domain"] != reference["fit"]["target_domain"]:
        raise ValueError("Training config must select the original target-domain decoder.")
    fit_config = None
    if args.fit_config:
        fit_config = load_yaml_config(args.fit_config)
        fit_config["device"] = str(device)
        for key in ("source_bank", "target_bank"):
            if getattr(args, key) is not None:
                fit_config[key] = str(getattr(args, key))
        fit_config = validate_fit_config(fit_config, reference)
        for key in ("source_bank", "target_bank"):
            bank = json.loads((ROOT / fit_config[key] / "manifest.json").read_text(encoding="utf-8"))
            selected = sample_ids(bank["ids"], fit_config["sampling"]["images_per_domain"], fit_config["sampling"]["seed"])
            if (digest({k: v for k, v in bank.items() if k != "artifact_id"}) != bank["artifact_id"]
                    or bank["artifact_id"] != reference["fit"][key]["artifact_id"]
                    or selected != reference["fit"][key.replace("_bank", "_ids")]):
                raise ValueError("Fresh fitting must use the original training banks and selected IDs.")
    else:
        validate_complete_fit(load_mapping_manifest(args.mapping), reference)
    work = create_work_directory(ROOT, output, getattr(args, "work_dir", None))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="infoot_displacement_comparison_v1", status="running", quality_acceptance="unreviewed",
        reference_run=str(args.reference_run), reference_files_sha256=reference["file_sha256"],
        work_directory=str(work), analysis_files=["experiment_report.json", "comparison_grid.png"],
        checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
        hypothesis="Transport local source-to-target displacements while preserving query residuals; "
                   "may reduce conditioning blur but can retain cat attributes or wrong semantic matches.",
        config_sha256=dict(train=file_hash(train_path), evaluation=file_hash(eval_path)),
        code_sha256={str(path.relative_to(ROOT)): file_hash(path) for path in (
            Path(__file__).resolve(), ROOT / "infoot_vit/diagnose_blur.py",
            ROOT / "infoot_vit/infoot_helper/experiment_output.py",
            ROOT / "infoot_vit/infoot_helper/conditional_displacement.py",
            ROOT / "infoot_vit/infoot_helper/dense_partial_projection.py",
            ROOT / "src/diffusion_ot/evaluation/stage1a_eval.py",
            ROOT / "src/diffusion_ot/models/pdae_v2/branch.py",
            ROOT / "src/diffusion_ot/models/pdae_v2/cross_attention.py")})
    report_path = output / "experiment_report.json"
    write_json(report_path, report)
    try:
        directory = args.mapping
        if fit_config is not None:
            report["status"] = "fitting"
            write_json(report_path, report)
            directory = fit_mapping(fit_config, root=ROOT, output_root=work / "fit")
        fit = load_mapping_manifest(directory)
        validate_complete_fit(fit, reference)
        report.update(status="mapping", mapping=str(directory), mapper_id=fit["artifact_id"],
                      successful_pairs=fit["pair_count"], failed_pairs=0,
                      fit_diagnostics=fit_diagnostics(directory))
        write_json(report_path, report)
        projection = deepcopy(reference["mapped"]["projection"])
        projection["patch_value_mode"] = "barycentric"
        mapper = FeatureMapper.load(directory, device=str(device), projection=projection,
            pair_cache_bytes=args.pair_cache_mib * 1024**2, pair_projection_batch_size=args.projection_pair_batch_size)
        results, manifests = {}, {}
        for mode in ("barycentric", "conditional_displacement"):
            # Only the value estimator changes; reuse the same validated kernels,
            # support thresholds, mapper and bounded read-only plan cache.
            mapper.config["projection"] = projection_settings(mapper.config, None, patch_value_mode=mode)
            results[mode], manifests[mode] = mapper.project_bank(query, work / mode, chunk_size=args.batch_size)
        report["mapping_outputs"] = {mode: mapping_diagnostics(work / mode, manifest)
                                     for mode, manifest in manifests.items()}
        verify_paired_mapping(results["barycentric"], results["conditional_displacement"])
        baseline, candidate = results["barycentric"], results["conditional_displacement"]
        valid = baseline.valid_mask.cpu()
        refs = [min(range(len(mapper.target.ids)), key=lambda j: (-row["image_weights"][j], mapper.target.ids[j]))
                for row in baseline.diagnostics["queries"]]
        native = mapper.target.features[refs].cpu()
        variants = dict(native_reference_full=(native, torch.ones_like(valid)),
            native_reference_mapped_mask=(native, valid),
            source_tokens_dog_decoder_control=(query.features.cpu(), valid),
            mapped_barycentric=(baseline.mapped_features.cpu(), valid),
            mapped_conditional_displacement=(candidate.mapped_features.cpu(), valid))
        generation = {key: deepcopy(reference["generation"][key]) for key in
            ("weights", "num_steps", "guidance_scale", "solver", "seed", "per_image_noise_seeds", "query_ids")}
        reference_ids = [mapper.target.ids[j] for j in refs]
        generation["target_reference_ids"] = reference_ids
        # Release multi-GB mapping caches/support copies before loading the DiT.
        target_representation = mapper.target.representation
        reference_records = [mapper.target.records[j] for j in refs]
        del mapper._dense_projector, mapper, results, baseline, candidate
        if device.type == "cuda":
            torch.cuda.empty_cache()
        from diffusion_ot.evaluation.stage1a_eval import load_stage1a_evaluator, integrate_pdae_flow, decode_vae_latents
        from diffusion_ot.data.ground_truth import load_ground_truth_images
        from torchvision.utils import save_image
        evaluator = load_stage1a_evaluator(train_path, eval_path, device=str(device),
            weights=generation["weights"], checkpoint_path=args.checkpoint)
        if (file_hash(evaluator.checkpoint_path) != checkpoint_hash
                or evaluator.branch.encoder.snapshot_identity != target_representation["encoder"]
                or evaluator.branch.encoder.architecture_spec["features"] != target_representation["layer"]):
            raise ValueError("Loaded checkpoint or frozen encoder identity changed.")
        for module in (evaluator.branch, evaluator.transformer, evaluator.vae):
            module.requires_grad_(False).eval()
        report.update(status="decoding", generation=generation, device=str(evaluator.device),
            model_dtype=str(evaluator.model_dtype), batch_size=args.batch_size,
            mapped_cache_ids={key: m["artifact_id"] for key, m in manifests.items()},
            confidence_sha256=tensor_hash(torch.load(work / "barycentric/mapped.pt", weights_only=True)["match_confidence"]),
            valid_mask_sha256=tensor_hash(valid),
            conditioning_diagnostics=decoder_conditioning_diagnostics(evaluator, variants),
            interpretation="Only the two mapped rows isolate the estimator. Native references are decoder controls, "
                "not paired targets. The source control detects apparent improvements caused by leaving cats untranslated. "
                "The new fit may have repaired pairs, so comparison with the previous grid also changes pair inventory.")
        report["quality_review"] = review_template(query.ids, reference_ids)
        write_json(report_path, report)
        images, noise = decode_variants(evaluator, variants, generation,
            null_label=train["class_conditioning"]["null_label"], batch_size=args.batch_size,
            integrate=integrate_pdae_flow, decode=decode_vae_latents)
        rows = dict(original_source=load_ground_truth_images(ROOT / train["data_config"], query.records),
            top1_reference_rgb_not_ground_truth=load_ground_truth_images(ROOT / train["data_config"], reference_records), **images)
        save_image(torch.cat(list(rows.values())).clamp(0, 1), output / "comparison_grid.png", nrow=len(query.ids), padding=4)
        tensor_path = work / "conditioning_variants.pt"
        save_tensor(tensor_path, dict(ids=query.ids, initial_noise=noise,
            variants={name: dict(features=x.cpu(), valid_mask=mask.cpu()) for name, (x, mask) in variants.items()}))
        for mode in ("barycentric", "conditional_displacement"):
            generated = dict(generation,
                protocol="fixed_checkpoint_compatibility_not_retraining", mapper_id=fit["artifact_id"],
                checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                projection=manifests[mode]["projection"], confidence_policy=manifests[mode]["projection"]["confidence"],
                target_domain=train["domain"], row_order=["original_source", "top1_routed_target_reference_not_ground_truth", "translated"],
                initial_noise_sha256=tensor_hash(noise), quality_acceptance="unreviewed")
            write_json(work / mode / "generation_report.json", generated)
            report["mapping_outputs"][mode]["generation"] = generated
        report.update(status="complete", row_order=list(rows), initial_noise_sha256=tensor_hash(noise),
            output=dict(file=str(tensor_path), sha256=file_hash(tensor_path)),
            comparison_grid=dict(file="comparison_grid.png", sha256=file_hash(output / "comparison_grid.png")))
        write_json(report_path, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        # A fit may raise before returning its newly created directory. Preserve
        # its failure diagnostics in this one shareable report as well.
        candidates = sorted((work / "fit").glob("*")) if directory is None else [Path(directory)]
        if candidates:
            report.update(mapping=str(candidates[-1]), fit_diagnostics=fit_diagnostics(candidates[-1]))
        write_json(report_path, report)
        raise
    print(f"Saved matched displacement experiment: {output}; image quality still requires review.", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    fitting = parser.add_mutually_exclusive_group(required=True)
    fitting.add_argument("--fit-config", type=Path, help="Fit once in the work directory, then compare both estimators.")
    fitting.add_argument("--mapping", type=Path, help="Reuse an already complete, identity-matched fit.")
    parser.add_argument("--reference-run", required=True, type=Path, help="Reviewed result with manifest, generation report and fitting_logs/manifest.json.")
    parser.add_argument("--query-bank", required=True, type=Path)
    parser.add_argument("--source-bank", type=Path, help="Fresh fit only: original source bank path.")
    parser.add_argument("--target-bank", type=Path, help="Fresh fit only: original target bank path.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New analysis directory: report + comparison grid only.")
    parser.add_argument("--work-dir", type=Path, help="New directory outside results/ for resumable state; defaults to ignored outputs/infoot_vit/.")
    parser.add_argument("--train-config", default="configs/stage1a_pdae_v2_l/dog.yaml")
    parser.add_argument("--eval-config", default="configs/stage1a_eval/pdae_v2_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pair-cache-mib", type=int, default=3072)
    parser.add_argument("--projection-pair-batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads, args.projection_pair_batch_size) < 1 or args.pair_cache_mib < 0:
        parser.error("Positive batch/thread counts and nonnegative pair-cache MiB are required.")
    if args.mapping and (args.source_bank is not None or args.target_bank is not None):
        parser.error("Bank overrides apply only to --fit-config.")
    torch.set_num_threads(args.threads)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
