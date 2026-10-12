"""Reuse v9's sharp dog tokens while testing source-latent structure preservation.

No fitting or remapping. Three result files; tensors stay in ignored outputs/.
Run this on the lab machine with the preserved v9 conditioning cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from infoot_vit.compare_displacement import review_template
from infoot_vit.diagnose_blur import decode_variants, tensor_hash, decoder_conditioning_diagnostics
from infoot_vit.infoot_helper import comparison_umap
from infoot_vit.infoot_helper.device import resolve_device
from infoot_vit.infoot_helper.experiment_output import create_work_directory
from infoot_vit.infoot_helper.feature_bank import FeatureBank, digest, file_hash, save_tensor, write_json
from infoot_vit.infoot_helper.mapping import MappingResult
from infoot_vit.infoot_helper.mapping_manifest import load_mapping_manifest
from infoot_vit.infoot_helper.source_edit import edit_schedule, encode_source, flow_edit, make_velocity


METHOD_TITLES = dict(argmax_noise_baseline="V9: dog tokens + noise",
                     argmax_source_denoise="Dog tokens + noised source",
                     argmax_flow_edit="Dog tokens + source velocity edit")
START_TIME = .5  # One prespecified setting; no per-image selection or sweep.


def _canonical(value):
    return value.get("artifact_id") == digest({k: v for k, v in value.items() if k != "artifact_id"})


def read_reference(directory):
    """Validate v9 metadata without loading its mapper or re-evaluating any pairs."""
    report = json.loads((Path(directory) / "experiment_report.json").read_text(encoding="utf-8"))
    if (report.get("schema") != "infoot_averaging_ablation_v1" or report.get("status") != "complete"
            or report.get("failed_pairs") != 0):
        raise ValueError("Source editing requires the completed averaging_ablation_v9 reference.")
    fit = report["fit_diagnostics"]["files"]["manifest.json"]["data"]
    mapped = report["mapping_outputs"]["conditional_edge_mean"]["manifest"]
    generation = report["generation"]
    ids = generation["query_ids"]
    if (not _canonical(fit) or not _canonical(mapped) or fit.get("status") != "complete"
            or fit.get("incomplete_fit") or fit.get("failed_pair_count", 0) != 0
            or fit["config"]["mode"] != "grouped_partial" or fit["target_domain"] != "dog"
            or fit["artifact_id"] != report["mapper_id"] or mapped["mapper_id"] != report["mapper_id"]
            or mapped["ids"] != ids or not ids or len(set(ids)) != len(ids)
            or fit["pair_count"] != report["successful_pairs"]
            or fit["pair_count"] != len(fit["source_ids"]) * min(fit["config"]["fit_pair_top_k"] or
                                                               len(fit["target_ids"]), len(fit["target_ids"]))
            or mapped["representation"] != fit["representation"]
            or mapped["projection"]["patch_value_mode"] != "conditional_edge_mean"):
        raise ValueError("V9 fit, mapped identity or query ordering changed.")
    seeds = [int.from_bytes(hashlib.sha256(f"{generation['seed']}:{sid}".encode()).digest()[:8], "big")
             % (2**63 - 1) for sid in ids]
    if (generation["per_image_noise_seeds"] != seeds or generation["solver"] != "euler"
            or generation["weights"] not in {"raw", "ema"}
            or not math.isfinite(generation["guidance_scale"]) or generation["guidance_scale"] < 0):
        raise ValueError("Require v9's per-image noise and Euler/CFG generation controls.")
    edit_schedule(generation["num_steps"], START_TIME)
    diagnostic = report["averaging_diagnostic"]["queries"]
    if ([q["query_id"] for q in diagnostic] != ids
            or [q["dominant_target_id"] for q in diagnostic] != generation["target_reference_ids"]):
        raise ValueError("V9 dominant dog identities changed.")
    for name, identity in report["diagnostic_variants"].items():
        if (not _canonical(identity) or identity["method"] != name
                or identity["mapper_id"] != fit["artifact_id"] or identity["query_ids"] != ids
                or identity["query_bank_id"] != mapped["query_bank_id"]
                or identity["parent_mapped_id"] != mapped["artifact_id"]
                or identity["target_ids"] != generation["target_reference_ids"]
                or identity["valid_mask_sha256"] != report["common_valid_mask_sha256"]):
            raise ValueError("V9 diagnostic variant identity changed.")
    if set(report["diagnostic_variants"]) != {"mixture_edge_mean", "dominant_target_mean", "dominant_target_argmax"}:
        raise ValueError("V9 reference is missing its three diagnostic variants.")
    return report, fit, mapped


def load_inputs(args):
    """Bind the old tensors, training banks and current originals before sampling."""
    reference, fit, mapped = read_reference(args.reference_run)
    if load_mapping_manifest(args.mapping) != fit:
        raise ValueError("Use the exact completed v7 mapper recorded in v9.")
    if file_hash(args.checkpoint) != reference["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the reviewed v9 dog decoder.")
    query = FeatureBank.load(args.query_bank)
    if (query.artifact_id != mapped["query_bank_id"] or query.manifest["split"] not in {"val", "test"}
            or query.manifest["domain"] != fit["source_domain"] or query.representation != fit["representation"]):
        raise ValueError("Use the original compatible held-out query bank.")
    query = query.subset(reference["generation"]["query_ids"])
    banks = {}
    for name in ("source", "target"):
        bank = FeatureBank.load(args.mapping / fit[f"{name}_bank"]["path"])
        if (bank.artifact_id != fit[f"{name}_bank"]["artifact_id"] or bank.manifest["split"] != "train"
                or bank.representation != query.representation or bank.manifest["domain"] != fit[f"{name}_domain"]
                or set(query.ids) & set(bank.ids)):
            raise ValueError("Original training bank identity or held-out split changed.")
        banks[name] = bank.subset(fit[f"{name}_ids"])
    path = (args.reference_work_dir / "conditioning_variants.pt" if args.reference_work_dir else
            Path(reference["output"]["file"]))
    if not path.is_file():
        raise FileNotFoundError(f"Preserve the v9 conditioning cache on the lab machine: {path}. "
                                "Use --reference-work-dir if it was relocated; no remapping fallback.")
    if file_hash(path) != reference["output"]["sha256"]:
        raise ValueError("V9 conditioning cache hash changed.")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if cache["ids"] != query.ids or tensor_hash(cache["initial_noise"]) != reference["initial_noise_sha256"]:
        raise ValueError("Cached v9 query order or instantiated noise changed.")
    variants = cache["variants"]
    for name, identity in reference["diagnostic_variants"].items():
        value = variants[name]
        if (tensor_hash(value["features"]) != identity["features_sha256"]
                or tensor_hash(value["valid_mask"]) != identity["valid_mask_sha256"]):
            raise ValueError("Cached diagnostic condition differs from v9.")
    target = variants["dominant_target_argmax"]
    mask = target["valid_mask"]
    if (mask.dtype != torch.bool or mask.shape != query.features.shape[:2] or not mask.any(1).all()
            or target["features"].shape != query.features.shape or not torch.isfinite(target["features"]).all()
            or not torch.equal(variants["source_tokens_dog_decoder_control"]["features"], query.features)
            or not torch.equal(mask, variants["source_tokens_dog_decoder_control"]["valid_mask"])):
        raise ValueError("Source/target tokens or the shared v9 confidence mask changed.")
    refs = reference["generation"]["target_reference_ids"]
    # A routed dog may be reused by several queries; FeatureBank subsets require unique IDs.
    indices = [banks["target"].ids.index(sid) for sid in refs]
    native_features = banks["target"].features[indices]
    if (not torch.equal(variants["native_reference_common_mask"]["features"], native_features)
            or not torch.equal(variants["native_reference_common_mask"]["valid_mask"], mask)):
        raise ValueError("Native reference features or mask changed.")
    for row, diag in zip(mask, reference["averaging_diagnostic"]["queries"]):
        if row.tolist() != diag["common_valid_mask"] or int(row.sum()) != diag["common_valid_tokens"]:
            raise ValueError("Cached common mask disagrees with the v9 diagnostic.")
    return reference, fit, query, banks, cache, path, [banks["target"].records[i] for i in indices]


@torch.inference_mode()
def run_experiment(args):
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Source-edit output must be new: {output}")
    if output.is_relative_to(args.mapping.resolve()) or output.is_relative_to(args.reference_run.resolve()):
        raise ValueError("Write outputs outside the immutable fit and reference result.")
    if args.work_dir is not None and any(args.work_dir.resolve().is_relative_to(p.resolve())
                                        for p in (args.mapping, args.reference_run)):
        raise ValueError("Write work files outside the immutable fit and reference result.")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    reference, fit, query, banks, cache, cache_path, reference_records = load_inputs(args)
    comparison_umap.require_dependencies()
    from diffusion_ot.integrations.hf_snapshot import load_yaml_config
    train_path, eval_path = ROOT / args.train_config, ROOT / args.eval_config
    config_hashes = dict(train=file_hash(train_path), evaluation=file_hash(eval_path))
    train = load_yaml_config(train_path)
    if (config_hashes != reference["config_sha256"] or train["domain"] != fit["target_domain"]
            or train.get("flow", {}).get("direction", "noise_to_data") != "noise_to_data"):
        raise ValueError("Source editing requires unchanged v9 decoder configs and noise-to-data flow.")
    device = resolve_device(args.device)
    work = create_work_directory(ROOT, output, args.work_dir)
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "experiment_report.json"
    generation = reference["generation"]
    schedule = edit_schedule(generation["num_steps"], START_TIME)
    report = dict(schema="infoot_source_edit_v1", status="loading_decoder", quality_acceptance="unreviewed",
        reference_run=str(args.reference_run), reference_report_sha256=file_hash(args.reference_run / "experiment_report.json"),
        reference_cache=dict(file=str(cache_path), sha256=reference["output"]["sha256"]),
        mapper_id=fit["artifact_id"], query_bank_id=query.artifact_id,
        checkpoint_sha256=reference["checkpoint_sha256"], config_sha256=config_hashes,
        target_condition=reference["diagnostic_variants"]["dominant_target_argmax"],
        common_valid_mask_sha256=reference["common_valid_mask_sha256"],
        generation=generation, work_directory=str(work), batch_size=args.batch_size,
        analysis_files=["experiment_report.json", "comparison_grid.png", "umap.png"],
        noise_protocol=dict(baseline="Original v9 per-query noise, hash-checked before decoding.",
            source_denoise="Same original noise mixed with the deterministic source VAE latent.",
            flow_edit="Fresh independent step noises; each draw shared between source/target fields. "
                      "This is a different stochastic protocol from noise generation and source denoising."),
        editing=dict(start_time=START_TIME, end_time=1., steps=len(schedule), dt=1/generation["num_steps"],
            time_direction="noise_to_data", network_times="same reference-grid midpoints",
            n_avg=1, target_only_finishing_steps=0, source_and_target_cfg=generation["guidance_scale"],
            source_velocity_model="same frozen dog decoder conditioned on source cat tokens; experimental",
            noise_seed_rule="sha256(source_edit_v1:seed:query_id:global_step), first 8 bytes big-endian modulo 2**63-1",
            equation="s=(1-t)*eps+t*source; target_state=s+(edit-source); edit+=dt*(v_target-v_source)",
            source="https://arxiv.org/abs/2412.08629", adaptation="Image-token conditions, increasing time and midpoint Euler; not a reproduced paper result."),
        quality_review=review_template(query.ids, generation["target_reference_ids"]),
        interpretation="The three main methods share exact v9 dog tokens and conditioning masks. "
            "Source RGB supplies spatial structure through VAE latents. The source field uses cat tokens in the dog "
            "decoder, which is outside its training domain. Watch for unchanged cats, reference copying, lost pose "
            "and malformed anatomy. No automatic quality pass; generated-image UMAP is descriptive only.",
        code_sha256={name: file_hash(ROOT / name) for name in (
            "infoot_vit/edit_source.py", "infoot_vit/infoot_helper/source_edit.py", "infoot_vit/diagnose_blur.py",
            "infoot_vit/infoot_helper/comparison_umap.py", "infoot_vit/infoot_helper/umap_plot.py",
            "src/diffusion_ot/evaluation/stage1a_eval.py", "src/diffusion_ot/models/pdae_v2/encoder.py",
            "src/diffusion_ot/models/pdae_v2/branch.py", "src/diffusion_ot/models/pdae_v2/cross_attention.py")})
    write_json(report_path, report)
    try:
        from diffusion_ot.evaluation.stage1a_eval import load_stage1a_evaluator, integrate_pdae_flow, decode_vae_latents
        from diffusion_ot.data.ground_truth import load_ground_truth_images
        from torchvision.utils import save_image
        evaluator = load_stage1a_evaluator(train_path, eval_path, device=str(device),
            weights=generation["weights"], checkpoint_path=args.checkpoint)
        if (file_hash(evaluator.checkpoint_path) != reference["checkpoint_sha256"]
                or evaluator.branch.encoder.snapshot_identity != query.representation["encoder"]
                or evaluator.branch.encoder.architecture_spec["features"] != query.representation["layer"]
                or str(evaluator.model_dtype) != reference["model_dtype"]):
            raise ValueError("Loaded decoder, encoder or precision differs from v9.")
        for module in (evaluator.branch, evaluator.transformer, evaluator.vae):
            module.requires_grad_(False).eval()
        original = load_ground_truth_images(ROOT / train["data_config"], query.records)
        reference_rgb = load_ground_truth_images(ROOT / train["data_config"], reference_records)
        source = encode_source(evaluator.vae, original, device=evaluator.device,
                               dtype=evaluator.model_dtype, batch_size=args.batch_size)
        if source.shape != cache["initial_noise"].shape:
            raise ValueError("Source VAE latent shape differs from v9 noise.")
        old = cache["variants"]
        common = old["dominant_target_argmax"]["valid_mask"]
        target = old["dominant_target_argmax"]["features"]
        baseline_variants = dict(native_reference_common_mask=(old["native_reference_common_mask"]["features"], common),
                                 argmax_noise_baseline=(target, common))
        report.update(status="decoding", source_rgb_sha256=tensor_hash(original), source_latent_sha256=tensor_hash(source),
            vae_encoding=dict(posterior="mode", scaling_factor=float(evaluator.vae.config.scaling_factor),
                              input_range=[-1, 1]),
            conditioning_diagnostics=decoder_conditioning_diagnostics(evaluator, baseline_variants))
        write_json(report_path, report)
        images, noise = decode_variants(evaluator, baseline_variants, generation,
            null_label=train["class_conditioning"]["null_label"], batch_size=args.batch_size,
            integrate=integrate_pdae_flow, decode=decode_vae_latents,
            expected_noise_sha256=reference["initial_noise_sha256"])
        mask = common.to(evaluator.device)
        tokens = target.to(device=evaluator.device, dtype=evaluator.model_dtype)
        source_tokens = query.features.to(tokens)
        initial = START_TIME * source + (1 - START_TIME) * noise.to(source)
        latents = {}
        for name, condition in (("source_condition_denoise_control", source_tokens), ("argmax_source_denoise", tokens)):
            print(f"Source editing: {name}, t={START_TIME}..1, {len(schedule)} steps", flush=True)
            latent = integrate_pdae_flow(evaluator.branch, evaluator.transformer, initial.clone(), condition,
                num_steps=len(schedule), start_time=START_TIME, end_time=1., solver=generation["solver"],
                guidance_scale=generation["guidance_scale"], null_label=train["class_conditioning"]["null_label"],
                batch_size=args.batch_size, condition_padding_mask=~mask)
            latents[name] = latent.cpu()
            images[name] = decode_vae_latents(evaluator.vae, latent, batch_size=args.batch_size).cpu()
        print("Source editing: paired source/target velocity differences", flush=True)
        velocity = make_velocity(evaluator, guidance_scale=generation["guidance_scale"],
            null_label=train["class_conditioning"]["null_label"], batch_size=args.batch_size)
        edited, trace = flow_edit(source, source_tokens, tokens, mask, query.ids,
            seed=generation["seed"], num_steps=generation["num_steps"], start_time=START_TIME, velocity=velocity)
        latents["argmax_flow_edit"] = edited.cpu()
        images["argmax_flow_edit"] = decode_vae_latents(evaluator.vae, edited, batch_size=args.batch_size).cpu()
        reconstruction = decode_vae_latents(evaluator.vae, source, batch_size=args.batch_size).cpu()
        if any(not torch.isfinite(x).all() for x in [reconstruction, *images.values(), *latents.values()]):
            raise ValueError("Nonfinite latent or generated image in source editing.")
        rows = dict(original_source=original, source_vae_reconstruction_control=reconstruction,
            top1_reference_rgb_not_ground_truth=reference_rgb, **images)
        save_image(torch.cat(list(rows.values())).clamp(0, 1), output / "comparison_grid.png", nrow=len(query.ids), padding=4)
        tensor_path = work / "source_edit.pt"
        save_tensor(tensor_path, dict(ids=query.ids, initial_noise=noise, source_latents=source.cpu(),
            noised_source_initial_state=initial.cpu(), common_valid_mask=common, target_tokens=target,
            images=rows, edited_latents=latents))
        report.update(status="umap", row_order=list(rows), initial_noise_sha256=tensor_hash(noise), flow_edit_trace=trace,
            output=dict(file=str(tensor_path), sha256=file_hash(tensor_path)),
            comparison_grid=dict(file="comparison_grid.png", sha256=file_hash(output / "comparison_grid.png")),
            pixel_diagnostics=dict(interpretation="RGB MAE is a change/copy diagnostic, not a translation score.",
                methods={name: [dict(query_id=sid, source_mae=float((img - src).abs().mean()),
                                    reference_mae=float((img - ref).abs().mean()))
                    for sid, img, src, ref in zip(query.ids, images[name], original, reference_rgb)] for name in METHOD_TITLES}))
        write_json(report_path, report)
        results, identities = {}, {}
        full_mask = torch.ones_like(common)
        for name in METHOD_TITLES:
            features = torch.cat([evaluator.branch.encoder(chunk.to(evaluator.device) * 2 - 1).cpu()
                                  for chunk in images[name].split(args.batch_size)])
            if features.shape != query.features.shape or not torch.isfinite(features).all():
                raise ValueError("Re-encoded output features do not match the frozen bank representation.")
            identities[name] = dict(method=name, rgb_sha256=tensor_hash(images[name]), features_sha256=tensor_hash(features),
                encoder=query.representation, ids=query.ids, target_condition_id=report["target_condition"]["artifact_id"])
            identities[name]["artifact_id"] = digest(identities[name])
            results[name] = MappingResult(features, full_mask.float(), full_mask,
                dict(queries=[dict(query_id=sid) for sid in query.ids]))
        feature_path = work / "generated_features.pt"
        save_tensor(feature_path, dict(ids=query.ids, features={name: result.mapped_features for name, result in results.items()}))
        report["generated_feature_cache"] = dict(file=str(feature_path), sha256=file_hash(feature_path))
        report["generated_readouts"] = identities
        report["umap"] = comparison_umap.save_comparison_umap(SimpleNamespace(**banks, manifest=fit), query, results,
            output, mapped_ids={name: item["artifact_id"] for name, item in identities.items()}, method_titles=METHOD_TITLES,
            readout="generated_images", title="Source editing: generated RGB re-encoded, training-fitted UMAP")
        report["status"] = "complete"
        write_json(report_path, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(report_path, report)
        raise
    print(f"Saved source editing comparison: {output}; visual acceptance remains unreviewed.", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True, help="Completed averaging_ablation_v9 results.")
    parser.add_argument("--reference-work-dir", type=Path, help="Only if v9's conditioning_variants.pt was relocated.")
    parser.add_argument("--query-bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; exactly report, grid and UMAP.")
    parser.add_argument("--work-dir", type=Path, help="New cache directory outside results/, default ignored outputs/.")
    parser.add_argument("--train-config", default="configs/stage1a_pdae_v2_l/dog.yaml")
    parser.add_argument("--eval-config", default="configs/stage1a_eval/pdae_v2_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) < 1:
        parser.error("Batch size and threads must be positive.")
    torch.set_num_threads(args.threads)
    run_experiment(args)


if __name__ == "__main__":
    main()
