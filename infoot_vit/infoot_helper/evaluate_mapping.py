"""Fixed-checkpoint inference using the existing PDAE sampler and mask interface."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from .feature_bank import file_hash, write_json


@torch.inference_mode()
def generate(mapper, query_bank, result, output, *, root, train_config, eval_config,
             checkpoint=None, weights="ema", steps=50, guidance=1.5, solver="euler", seed=20260903,
             batch_size=4, device="cuda"):
    from diffusion_ot.evaluation.stage1a_eval import load_stage1a_evaluator, integrate_pdae_flow, decode_vae_latents
    from diffusion_ot.data.ground_truth import load_ground_truth_images
    from diffusion_ot.integrations.hf_snapshot import load_yaml_config
    from torchvision.utils import save_image

    root, output = Path(root), Path(output)
    if (not isinstance(steps, int) or steps < 1 or not isinstance(batch_size, int) or batch_size < 1
            or not math.isfinite(guidance) or guidance < 0 or solver not in {"euler", "heun"}):
        raise ValueError("Generation requires positive step/batch counts, finite nonnegative guidance, and Euler/Heun.")
    count = len(result.mapped_features)
    ids = query_bank.ids[:count]
    if (count > len(query_bank.ids) or query_bank.manifest["domain"] != mapper.source.manifest["domain"]
            or query_bank.representation != mapper.source.representation
            or [item.get("query_id") for item in result.diagnostics.get("queries", [])] != ids):
        raise ValueError("Mapped results must match the query bank's ordered stable IDs and representation.")
    policy = mapper.config["projection"]["confidence"]["all_invalid_policy"]
    features, padding = result.conditioning(ids, all_invalid_policy=policy)
    config = load_yaml_config(root / train_config)
    if config["domain"] != mapper.target.manifest["domain"]:
        raise ValueError("Generation needs the fitted target domain's PDAE checkpoint/config.")
    evaluator = load_stage1a_evaluator(root / train_config, root / eval_config, device=device,
                                     weights=weights, checkpoint_path=checkpoint)
    representation = mapper.target.representation
    if evaluator.branch.encoder.snapshot_identity != representation["encoder"]:
        raise ValueError("PDAE checkpoint SigLIP snapshot differs from the feature bank.")
    if evaluator.branch.encoder.architecture_spec["features"] != representation["layer"]:
        raise ValueError("PDAE feature layer differs from the cached bank.")
    features = features.to(device=evaluator.device, dtype=evaluator.model_dtype)
    padding = padding.to(evaluator.device)
    # Per-ID randomness keeps cross-mapper and chunk comparisons matched.
    noises, seeds = [], []
    for query_id in ids:
        item_seed = int.from_bytes(hashlib.sha256(f"{seed}:{query_id}".encode()).digest()[:8], "big") % (2**63 - 1)
        seeds.append(item_seed)
        noises.append(torch.randn((1, 4, 32, 32), generator=torch.Generator(device=evaluator.device).manual_seed(item_seed),
                                  device=evaluator.device, dtype=evaluator.model_dtype))
    generated = integrate_pdae_flow(evaluator.branch, evaluator.transformer, torch.cat(noises), features,
        condition_padding_mask=padding, num_steps=steps, guidance_scale=guidance, solver=solver,
        null_label=config["class_conditioning"]["null_label"], batch_size=batch_size)
    images = decode_vae_latents(evaluator.vae, generated, batch_size=batch_size).cpu()
    originals = load_ground_truth_images(root / config["data_config"], query_bank.records[:count])
    rows, row_names = [originals], ["original_source"]
    targets = []
    if mapper.image is not None:
        for diag in result.diagnostics["queries"]:
            idx = min(range(len(mapper.target.ids)), key=lambda j: (-diag["image_weights"][j], mapper.target.ids[j]))
            targets.append(mapper.target.records[idx])
        rows.append(load_ground_truth_images(root / config["data_config"], targets))
        row_names.append("top1_routed_target_reference_not_ground_truth")
    rows.append(images); row_names.append("translated")
    save_image(torch.cat(rows).clamp(0, 1), output / "translation_grid.png", nrow=count, padding=4)
    # Confidence is separate from image colors; never multiplied through LayerNorm.
    gh, gw = representation["grid"]
    heat = result.match_confidence.cpu().reshape(count, 1, gh, gw)
    heat = torch.nn.functional.interpolate(heat, size=originals.shape[-2:], mode="nearest")
    save_image(heat.expand(-1, 3, -1, -1), output / "confidence_grid.png", nrow=count, padding=4)
    used_checkpoint = evaluator.checkpoint_path
    report = dict(protocol="fixed_checkpoint_compatibility_not_retraining", mapper_id=mapper.manifest["artifact_id"],
        checkpoint=str(used_checkpoint), checkpoint_sha256=file_hash(used_checkpoint), weights=weights,
        query_ids=ids, target_reference_ids=[r["sample_id"] for r in targets], row_order=row_names,
        solver=solver, num_steps=steps, guidance_scale=guidance, seed=seed, per_image_noise_seeds=seeds,
        projection=mapper.config["projection"],
        confidence_policy=mapper.config["projection"]["confidence"], target_domain=config["domain"],
        interpretation="No paired target ground truth. Feature metrics are not independent semantic validation.")
    write_json(output / "generation_report.json", report)
    return report
