"""Mapping-only averaging diagnostic, dispatched by compare_displacement.py."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from .compare_displacement import ROOT, read_reference, validate_complete_fit, review_template
from .diagnose_blur import decode_variants, decoder_conditioning_diagnostics, tensor_hash
from .infoot_helper.averaging_diagnostic import AveragingObserver, METHOD_TITLES
from .infoot_helper import comparison_umap
from .infoot_helper.device import resolve_device
from .infoot_helper.experiment_output import create_work_directory, fit_diagnostics, mapping_diagnostics
from .infoot_helper.feature_bank import FeatureBank, digest, file_hash, save_tensor, write_json
from .infoot_helper.mapping import FeatureMapper, MappingResult
from .infoot_helper.mapping_manifest import load_mapping_manifest


def read_v8_reference(directory):
    reference = read_reference(directory)
    report = json.loads((directory / "experiment_report.json").read_text(encoding="utf-8"))
    if report.get("schema") != "infoot_residual_ablation_v1" or not reference.get("complete_mapper_id"):
        raise ValueError("Averaging ablation requires the completed residual_ablation_v8 reference.")
    edge = report["mapping_outputs"]["conditional_edge_mean"]["manifest"]
    if (edge["artifact_id"] != digest({k: v for k, v in edge.items() if k != "artifact_id"})
            or edge["mapper_id"] != reference["complete_mapper_id"]
            or edge["query_bank_id"] != reference["mapped"]["query_bank_id"]
            or edge["ids"] != reference["generation"]["query_ids"]
            or edge["representation"] != reference["fit"]["representation"]
            or edge["projection"] != dict(reference["mapped"]["projection"], patch_value_mode="conditional_edge_mean")):
        raise ValueError("Completed v8 edge-mean reference identity or projection differs.")
    reference["edge_manifest"] = edge
    return reference


def verify_v8_replay(result, reference):
    if (tensor_hash(result.match_confidence) != reference["confidence_sha256"]
            or tensor_hash(result.valid_mask) != reference["valid_mask_sha256"]):
        raise ValueError("The original mapping confidence or masks changed from v8.")
    old = reference["edge_manifest"]["diagnostics"]["queries"]
    rows = result.diagnostics["queries"]
    keys = ("query_id", "image_weights", "target_ids", "image_projection_h", "patch_projection_h",
            "fit_pair_retained_routing_mass", "fit_pair_discarded_routing_mass",
            "ot_rejected_mass", "support_invalid_retained_mass")
    if len(old) != len(rows) or any(a[k] != b[k] for a, b in zip(old, rows) for k in keys):
        raise ValueError("Original routing, bandwidths or rejection changed from v8.")


@torch.inference_mode()
def run_experiment(args):
    if (args.mapping is None or args.fit_config is not None
            or args.source_bank is not None or args.target_bank is not None):
        raise ValueError("Averaging ablation requires --mapping with the original banks; no refitting.")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Comparison output must be new: {output}")
    if args.mapping.resolve() in output.parents:
        raise ValueError("Write comparison outputs outside the immutable fitted mapper.")
    reference = read_v8_reference(args.reference_run)
    comparison_umap.require_dependencies()
    checkpoint_hash = file_hash(args.checkpoint)
    if checkpoint_hash != reference["generation"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the reviewed v8 dog decoder.")
    device = resolve_device(args.device)
    query = FeatureBank.load(args.query_bank)
    if (query.artifact_id != reference["mapped"]["query_bank_id"]
            or query.manifest["split"] not in {"val", "test"}
            or query.representation != reference["fit"]["representation"]):
        raise ValueError("Require the original compatible held-out query bank.")
    query = query.subset(reference["generation"]["query_ids"])
    fit = load_mapping_manifest(args.mapping)
    validate_complete_fit(fit, reference)
    from diffusion_ot.integrations.hf_snapshot import load_yaml_config
    train_path, eval_path = ROOT / args.train_config, ROOT / args.eval_config
    train = load_yaml_config(train_path)
    config_hashes = dict(train=file_hash(train_path), evaluation=file_hash(eval_path))
    if config_hashes != reference["config_sha256"] or train["domain"] != fit["target_domain"]:
        raise ValueError("Decoder configurations differ from the completed v8 reference.")
    work = create_work_directory(ROOT, output, getattr(args, "work_dir", None))
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "experiment_report.json"
    report = dict(schema="infoot_averaging_ablation_v1", experiment="averaging_ablation", status="mapping",
        quality_acceptance="unreviewed", reference_run=str(args.reference_run),
        reference_files_sha256=reference["file_sha256"], mapping=str(args.mapping), mapper_id=fit["artifact_id"],
        work_directory=str(work), successful_pairs=fit["pair_count"], failed_pairs=0,
        fit_diagnostics=fit_diagnostics(args.mapping), checkpoint=str(args.checkpoint),
        checkpoint_sha256=checkpoint_hash, config_sha256=config_hashes,
        analysis_files=["experiment_report.json", "comparison_grid.png", "umap.png"],
        comparisons=dict(across_dogs=["mixture_edge_mean", "dominant_target_mean"],
            within_dog_patches=["dominant_target_mean", "dominant_target_argmax"],
            mask_control=["mapped_edge_mean_original_mask", "mixture_edge_mean"]),
        hypothesis="Diagnose averaging within a dog's transported patch map separately from averaging across dogs. "
            "Patch argmax is an experimental diagnostic, with no source residual or automatic quality acceptance.",
        code_sha256={name: file_hash(ROOT / name) for name in (
            "infoot_vit/averaging_experiment.py", "infoot_vit/compare_displacement.py", "infoot_vit/diagnose_blur.py",
            "infoot_vit/infoot_helper/averaging_diagnostic.py", "infoot_vit/infoot_helper/mapping.py",
            "infoot_vit/infoot_helper/conditional_displacement.py", "infoot_vit/infoot_helper/dense_partial_projection.py",
            "infoot_vit/infoot_helper/comparison_umap.py", "infoot_vit/infoot_helper/umap_plot.py",
            "infoot_vit/infoot_helper/experiment_output.py", "src/diffusion_ot/evaluation/stage1a_eval.py",
            "src/diffusion_ot/models/pdae_v2/branch.py", "src/diffusion_ot/models/pdae_v2/cross_attention.py")})
    write_json(report_path, report)
    observer = None
    try:
        projection = deepcopy(reference["edge_manifest"]["projection"])
        mapper = FeatureMapper.load(args.mapping, device=str(device), projection=projection,
            pair_cache_bytes=args.pair_cache_mib * 1024**2, pair_projection_batch_size=args.projection_pair_batch_size)
        observer = AveragingObserver(mapper)
        original, manifest = mapper.project_bank(query, work / "conditional_edge_mean", chunk_size=args.batch_size,
                                                 edge_observer=observer)
        report["mapping_outputs"] = dict(conditional_edge_mean=mapping_diagnostics(work / "conditional_edge_mean", manifest))
        report["averaging_diagnostic"] = observer.report()
        write_json(report_path, report)
        verify_v8_replay(original, reference)
        if [r["query_id"] for r in observer.records] != query.ids:
            raise ValueError("Averaging diagnostic query order changed.")
        common = torch.stack(observer.masks)
        bad = [sid for sid, active in zip(query.ids, common.any(1).tolist()) if not active]
        if bad:
            raise ValueError(f"No common valid tokens for {bad}; diagnostic recorded, no fallback or decoder bypass.")
        refs = [r["dominant_target_index"] for r in observer.records]
        reference_ids = [mapper.target.ids[j] for j in refs]
        if reference_ids != reference["generation"]["target_reference_ids"]:
            raise ValueError("Dominant dogs differ from the reviewed v8 routed references.")
        original_features = original.mapped_features.detach().cpu()
        original_valid = original.valid_mask.cpu()
        selected_confidence = torch.stack(observer.selected_confidences).to(original_features)
        values = dict(mixture_edge_mean=original_features.masked_fill(~common[..., None], 0.),
            dominant_target_mean=torch.stack(observer.means).to(original_features),
            dominant_target_argmax=torch.stack(observer.argmax_maps).to(original_features))
        results, variant_ids = {}, {}
        report["diagnostic_variants"] = {}
        for name, features in values.items():
            confidence = original.match_confidence.cpu() if name == "mixture_edge_mean" else selected_confidence
            result = MappingResult(features, confidence, common, dict(queries=[dict(query_id=sid) for sid in query.ids]))
            result.conditioning(query.ids)
            identity = dict(schema="infoot_averaging_variant_v1", method=name, parent_mapped_id=manifest["artifact_id"],
                mapper_id=fit["artifact_id"], query_bank_id=query.artifact_id, query_ids=query.ids,
                target_ids=reference_ids, features_sha256=tensor_hash(features),
                valid_mask_sha256=tensor_hash(common), confidence_sha256=tensor_hash(confidence),
                diagnostic_implementation_sha256=report["code_sha256"]["infoot_vit/infoot_helper/averaging_diagnostic.py"])
            identity["artifact_id"] = digest(identity)
            report["diagnostic_variants"][name] = identity
            variant_ids[name], results[name] = identity["artifact_id"], result
        cache_path = work / "averaging_variants.pt"
        save_tensor(cache_path, dict(ids=query.ids, reference_ids=reference_ids, common_valid_mask=common,
            original_valid_mask=original_valid, dominant_patch_probabilities=torch.stack(observer.selected_probabilities),
            dominant_conditional_confidence=selected_confidence, variants=values))
        report.update(status="umap", diagnostic_cache=dict(file=str(cache_path), sha256=file_hash(cache_path)),
            original_confidence_sha256=tensor_hash(original.match_confidence),
            original_valid_mask_sha256=tensor_hash(original_valid), common_valid_mask_sha256=tensor_hash(common))
        write_json(report_path, report)
        report["umap"] = comparison_umap.save_comparison_umap(mapper, query, results, output,
            mapped_ids=variant_ids, method_titles=METHOD_TITLES,
            title="Averaging diagnostic: one training-fitted UMAP per row")
        native = mapper.target.features[refs].cpu()
        variants = dict(native_reference_full=(native, torch.ones_like(common)),
            native_reference_common_mask=(native, common),
            source_tokens_dog_decoder_control=(query.features.cpu(), common),
            mapped_edge_mean_original_mask=(original_features, original_valid),
            **{name: (features, common) for name, features in values.items()})
        generation = {key: deepcopy(reference["generation"][key]) for key in
            ("weights", "num_steps", "guidance_scale", "solver", "seed", "per_image_noise_seeds", "query_ids")}
        generation["target_reference_ids"] = reference_ids
        target_representation = mapper.target.representation
        reference_records = [mapper.target.records[j] for j in refs]
        del mapper._dense_projector, mapper, original, observer, results
        observer = None
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
            conditioning_diagnostics=decoder_conditioning_diagnostics(evaluator, variants),
            quality_review=review_template(query.ids, reference_ids),
            interpretation="Compare the final three rows on the same mask. Full mixture vs dominant dog isolates dog "
                "averaging; dominant mean vs patch argmax replaces within-dog averaging with a hard readout. "
                "The original-mask mixture row isolates mask changes. Hard readout can break coherence or copy the "
                "reference; native controls and UMAP do not certify successful translation.")
        report["quality_review"]["comparisons"] = report["comparisons"]
        write_json(report_path, report)
        images, noise = decode_variants(evaluator, variants, generation,
            null_label=train["class_conditioning"]["null_label"], batch_size=args.batch_size,
            integrate=integrate_pdae_flow, decode=decode_vae_latents,
            expected_noise_sha256=reference["initial_noise_sha256"])
        rows = dict(original_source=load_ground_truth_images(ROOT / train["data_config"], query.records),
            top1_reference_rgb_not_ground_truth=load_ground_truth_images(ROOT / train["data_config"], reference_records), **images)
        save_image(torch.cat(list(rows.values())).clamp(0, 1), output / "comparison_grid.png", nrow=len(query.ids), padding=4)
        tensor_path = work / "conditioning_variants.pt"
        save_tensor(tensor_path, dict(ids=query.ids, initial_noise=noise,
            variants={name: dict(features=x.cpu(), valid_mask=mask.cpu()) for name, (x, mask) in variants.items()}))
        report.update(status="complete", row_order=list(rows), initial_noise_sha256=tensor_hash(noise),
            output=dict(file=str(tensor_path), sha256=file_hash(tensor_path)),
            comparison_grid=dict(file="comparison_grid.png", sha256=file_hash(output / "comparison_grid.png")))
        write_json(report_path, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if observer is not None:
            report["averaging_diagnostic"] = observer.report()
        write_json(report_path, report)
        raise
    print(f"Saved averaging diagnostic: {output}; image quality still requires review.", flush=True)
    return report
