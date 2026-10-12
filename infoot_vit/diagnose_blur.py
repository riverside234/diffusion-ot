"""Test one conditioning-spread intervention with matched frozen-decoder controls.

Reads an existing mapped cache and its generation report. Does not fit, remap,
change sampling settings, or replace any existing artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from infoot_vit.infoot_helper.conditioning_spread import (
    calibrate_native_spread, conditioning_variants, token_moments,
)
from infoot_vit.infoot_helper.feature_bank import FeatureBank, digest, file_hash, save_tensor, write_json
from infoot_vit.infoot_helper.mapping import load_mapped
from infoot_vit.infoot_helper.mapping_manifest import load_mapping_manifest
from infoot_vit.infoot_helper.experiment_output import create_work_directory


def validate_comparison(fit, mapped_manifest, generation, query, target, state):
    """Bind the comparison to exactly the saved queries, fit, routing and noise."""
    ids = state["ids"]
    projection = mapped_manifest["projection"]
    if (fit["config"]["mode"] != "grouped_partial" or fit["config"]["fit_pair_top_k"] != 8
            or fit["artifact_id"] != mapped_manifest["mapper_id"]
            or generation["mapper_id"] != mapped_manifest["mapper_id"]
            or generation["query_ids"] != ids or mapped_manifest["ids"] != ids
            or generation["projection"] != projection
            or generation["confidence_policy"] != projection["confidence"]
            or generation.get("incomplete_fit") != mapped_manifest.get("incomplete_fit")
            or fit.get("incomplete_fit") != mapped_manifest.get("incomplete_fit")):
        raise ValueError("Comparison requires the same dense top-8 fit, ordered queries, projection and failed-pair snapshot.")
    image_h = fit["config"]["solver"]["h"] * projection["bandwidth_multiplier"]
    if (not math.isclose(image_h, .20, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(projection["patch_bandwidth"], .20, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("This comparison requires an existing .20/.20 mapped cache.")
    if (query.artifact_id != mapped_manifest["query_bank_id"]
            or target.artifact_id != fit["target_bank"]["artifact_id"]
            or query.manifest["split"] not in {"val", "test"} or target.manifest["split"] != "train"
            or query.manifest["domain"] != fit["source_domain"]
            or target.manifest["domain"] != fit["target_domain"]
            or generation["target_domain"] != fit["target_domain"]
            or query.representation != target.representation
            or target.representation != fit["representation"]
            or mapped_manifest["representation"] != fit["representation"]
            or set(ids) & (set(fit["source_ids"]) | set(target.manifest["ids"]))):
        raise ValueError("Require the original compatible held-out queries and selected target training bank.")
    selected_query = query.subset(ids)
    if selected_query.records != state["records"]:
        raise ValueError("Cached query records differ from the original bank.")
    rows = mapped_manifest["diagnostics"]["queries"]
    if [row["query_id"] for row in rows] != ids or target.ids != fit["target_ids"]:
        raise ValueError("Query diagnostics or target support ordering changed.")
    references = []
    for row in rows:
        weights = row["image_weights"]
        if (len(weights) != len(target.ids) or not all(math.isfinite(w) and w >= 0 for w in weights)
                or not math.isclose(sum(weights), 1, rel_tol=0, abs_tol=1e-7)):
            raise ValueError("Invalid saved post-pruning image weights.")
        references.append(min(range(len(target.ids)), key=lambda j: (-weights[j], target.ids[j])))
    if generation["target_reference_ids"] != [target.ids[j] for j in references]:
        raise ValueError("Recorded native references differ from saved post-pruning routing.")
    seeds = [int.from_bytes(hashlib.sha256(f"{generation['seed']}:{sid}".encode()).digest()[:8], "big")
             % (2**63 - 1) for sid in ids]
    if (generation["per_image_noise_seeds"] != seeds or generation["weights"] not in {"raw", "ema"}
            or type(generation["num_steps"]) is not int or generation["num_steps"] < 1
            or generation["solver"] not in {"euler", "heun"}
            or not math.isfinite(generation["guidance_scale"]) or generation["guidance_scale"] < 0):
        raise ValueError("Invalid recorded generation controls or per-query noise identity.")
    return selected_query, references


def tensor_hash(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def spread_summary(features, valid):
    _, variances = token_moments(features, valid)
    return dict(mean_within_image_variance=float(variances.mean()),
                per_image_within_variance=variances.tolist(),
                valid_token_counts=valid.sum(1).tolist(),
                mean_token_norm=float(features.detach().double()[valid].norm(dim=-1).mean()))


@torch.inference_mode()
def decoder_conditioning_diagnostics(evaluator, variants):
    """Measure the actual learned LayerNorm/projector, not only input norms."""
    projector = evaluator.branch.semantic_transformer.feature_projector
    if not isinstance(projector[0], torch.nn.LayerNorm):
        raise ValueError("Expected the PDAE v2 LayerNorm-first token projector.")
    diagnostics = {}
    for name, (features, valid) in variants.items():
        mask = valid.to(evaluator.device)
        tokens = features.to(device=evaluator.device, dtype=evaluator.model_dtype)
        tokens = tokens.masked_fill(~mask[..., None], 0)
        diagnostics[name] = dict(input=spread_summary(features, valid),
            after_decoder_layernorm=spread_summary(projector[0](tokens), mask),
            after_decoder_projector=spread_summary(projector(tokens), mask))
    return diagnostics


@torch.inference_mode()
def decode_variants(evaluator, variants, generation, *, null_label, batch_size, integrate, decode,
                    expected_noise_sha256=None):
    """All conditions receive clones of exactly the same per-query noise tensor."""
    noises = [torch.randn((1, 4, 32, 32),
        generator=torch.Generator(device=evaluator.device).manual_seed(seed),
        device=evaluator.device, dtype=evaluator.model_dtype) for seed in generation["per_image_noise_seeds"]]
    initial_noise = torch.cat(noises)
    if expected_noise_sha256 is not None and tensor_hash(initial_noise) != expected_noise_sha256:
        raise ValueError("Instantiated noise differs from the completed reference experiment; stop before decoding.")
    images = {}
    for name, (features, valid) in variants.items():
        if len(features) != len(initial_noise):
            raise ValueError("Noise and condition batches differ.")
        print(f"Decoding {name}: {len(features)} images, identical starting noise", flush=True)
        latents = integrate(evaluator.branch, evaluator.transformer, initial_noise.clone(),
            features.to(device=evaluator.device, dtype=evaluator.model_dtype),
            condition_padding_mask=(~valid).to(evaluator.device), num_steps=generation["num_steps"],
            guidance_scale=generation["guidance_scale"], solver=generation["solver"],
            null_label=null_label, batch_size=batch_size)
        images[name] = decode(evaluator.vae, latents, batch_size=batch_size).cpu()
        if not torch.isfinite(images[name]).all():
            raise ValueError(f"Nonfinite decoded image in {name}.")
    return images, initial_noise.cpu()


@torch.inference_mode()
def run_comparison(args):
    # Import the decoder only after the cache/checkpoint identity checks below.
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    generation_path = args.mapped / "generation_report.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = file_hash(checkpoint)
    if checkpoint_hash != generation["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the reviewed generation; use the preserved checkpoint.")
    fit = load_mapping_manifest(args.mapping, allow_failed_pairs=args.allow_failed_pairs)
    result, state, mapped_manifest = load_mapped(args.mapped, mapper_id=fit["artifact_id"])
    target_path = args.target_bank or (args.mapping / fit["target_bank"]["path"])
    target = FeatureBank.load(target_path).subset(fit["target_ids"])
    query = FeatureBank.load(args.query_bank)
    query, references = validate_comparison(fit, mapped_manifest, generation, query, target, state)
    calibration = calibrate_native_spread(target.features)
    calibration.update(bank_id=target.artifact_id, split="train", selected_ids=target.ids,
                       selected_ids_sha256=digest(target.ids))
    native_features = target.features[references].to(result.mapped_features)
    variants, intervention = conditioning_variants(result.mapped_features, result.valid_mask,
        native_features, calibration["target_within_image_variance"])

    from diffusion_ot.evaluation.stage1a_eval import (
        load_stage1a_evaluator, integrate_pdae_flow, decode_vae_latents,
    )
    from diffusion_ot.data.ground_truth import load_ground_truth_images
    from diffusion_ot.integrations.hf_snapshot import load_yaml_config
    from torchvision.utils import save_image

    train_path, eval_path = ROOT / args.train_config, ROOT / args.eval_config
    config = load_yaml_config(train_path)
    if config["domain"] != target.manifest["domain"]:
        raise ValueError("Training config must select the fitted target domain's decoder.")
    evaluator = load_stage1a_evaluator(train_path, eval_path, device=args.device,
        weights=generation["weights"], checkpoint_path=checkpoint)
    if (file_hash(evaluator.checkpoint_path) != checkpoint_hash
            or evaluator.branch.encoder.snapshot_identity != target.representation["encoder"]
            or evaluator.branch.encoder.architecture_spec["features"] != target.representation["layer"]):
        raise ValueError("Loaded checkpoint or frozen encoder identity differs from the saved experiment.")
    for module in (evaluator.branch, evaluator.transformer, evaluator.vae):
        module.requires_grad_(False).eval()
    diagnostics = decoder_conditioning_diagnostics(evaluator, variants)
    work = create_work_directory(ROOT, args.output_dir, getattr(args, "work_dir", None))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(schema="infoot_conditioning_spread_comparison_v1", status="running",
        hypothesis="Averaged mapped conditions lack within-image token spread; scalar restoration may help.",
        quality_conclusion="Untested visually; requires inspection of the generated matched grid.",
        mapper_id=fit["artifact_id"], mapped_cache_id=mapped_manifest["artifact_id"],
        work_directory=str(work), analysis_files=["blur_report.json", "comparison_grid.png"],
        original_generation_report_sha256=file_hash(generation_path),
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        generation={k: generation[k] for k in ("weights", "num_steps", "guidance_scale", "solver", "seed",
            "per_image_noise_seeds", "query_ids", "target_reference_ids", "projection", "confidence_policy")},
        device=str(evaluator.device), model_dtype=str(evaluator.model_dtype), batch_size=args.batch_size,
        calibration=calibration, intervention=intervention, conditioning_diagnostics=diagnostics,
        incomplete_fit=fit.get("incomplete_fit"),
        code_sha256={str(path.relative_to(ROOT)): file_hash(path) for path in (
            Path(__file__).resolve(), ROOT / "infoot_vit/infoot_helper/conditioning_spread.py",
            ROOT / "infoot_vit/infoot_helper/experiment_output.py",
            ROOT / "src/diffusion_ot/evaluation/stage1a_eval.py",
            ROOT / "src/diffusion_ot/models/pdae_v2/branch.py",
            ROOT / "src/diffusion_ot/models/pdae_v2/cross_attention.py")},
        config_sha256={"train": file_hash(train_path), "eval": file_hash(eval_path)},
        interpretation="Native references test the decoder; they are not paired targets or translation scores. "
            "Changing feature spread cannot recover lost semantic correspondences. "
            "Matching old seeds does not promise bitwise images across hardware/software changes.")
    report_path = args.output_dir / "blur_report.json"
    write_json(report_path, report)
    try:
        originals = load_ground_truth_images(ROOT / config["data_config"], query.records)
        reference_images = load_ground_truth_images(ROOT / config["data_config"], [target.records[j] for j in references])
        images, noise = decode_variants(evaluator, variants, generation,
            null_label=config["class_conditioning"]["null_label"], batch_size=args.batch_size,
            integrate=integrate_pdae_flow, decode=decode_vae_latents)
        rows = dict(original_source=originals, top1_reference_rgb_not_ground_truth=reference_images, **images)
        save_image(torch.cat(list(rows.values())).clamp(0, 1), args.output_dir / "comparison_grid.png",
                   nrow=len(query.ids), padding=4)
        # Preserve confidence/mask atomically. This is explicitly NOT an ordinary
        # mapped cache: the corrected tensors are an unvalidated decoder input.
        tensor_path = work / "conditioning_variants.pt"
        save_tensor(tensor_path, dict(ids=query.ids, initial_noise=noise,
            match_confidence=result.match_confidence.cpu(),
            variants={name: dict(features=x.cpu(), valid_mask=valid.cpu()) for name, (x, valid) in variants.items()}))
        report.update(status="complete", row_order=list(rows), initial_noise_sha256=tensor_hash(noise),
            output=dict(file=str(tensor_path), sha256=file_hash(tensor_path)),
            comparison_grid=dict(file="comparison_grid.png", sha256=file_hash(args.output_dir / "comparison_grid.png")),
            unchanged_match_confidence_sha256=tensor_hash(result.match_confidence),
            unchanged_valid_mask_sha256=tensor_hash(result.valid_mask))
        write_json(report_path, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(report_path, report)
        raise
    print(f"Saved matched conditioning experiment: {args.output_dir}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, type=Path, help="Original full fit matching the mapped cache.")
    parser.add_argument("--mapped", required=True, type=Path, help="Existing result with mapped.pt and generation_report.json.")
    parser.add_argument("--query-bank", required=True, type=Path)
    parser.add_argument("--target-bank", type=Path, help="Optional relocated ORIGINAL bank; its hash must still match.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New analysis directory: report + comparison grid only.")
    parser.add_argument("--work-dir", type=Path, help="New directory outside results/ for tensors; defaults to ignored outputs/infoot_vit/.")
    parser.add_argument("--train-config", default="configs/stage1a_pdae_v2_l/dog.yaml")
    parser.add_argument("--eval-config", default="configs/stage1a_eval/pdae_v2_l.yaml")
    parser.add_argument("--allow-failed-pairs", action="store_true",
                        help="Explicitly allow the original failed-pair snapshot; never relabel it as repaired.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) < 1:
        parser.error("Batch size and threads must be positive.")
    torch.set_num_threads(args.threads)
    run_comparison(args)


if __name__ == "__main__":
    main()
