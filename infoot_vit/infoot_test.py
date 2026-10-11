"""Project held-out SigLIP banks, optionally generate with a fixed PDAE checkpoint."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
import math
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from infoot_vit.infoot_helper.feature_bank import FeatureBank, digest, write_json
from infoot_vit.infoot_helper.mapping import FeatureMapper
from infoot_vit.infoot_helper.mapping_manifest import load_mapping_manifest
from infoot_vit.infoot_helper.run_logging import RunLog


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping", required=True, type=Path)
    p.add_argument("--query-bank", type=Path, default=ROOT / "data/infoot_vit/cat_val")
    p.add_argument("--output-dir", type=Path, help="New directory for mapped tensors, metadata and optional images.")
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--generate", action="store_true", help="Also run fixed-checkpoint diffusion inference; never train.")
    p.add_argument("--train-config", default="configs/stage1a_pdae_v2_l/dog.yaml")
    p.add_argument("--eval-config", default="configs/stage1a_eval/pdae_v2_l.yaml")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--weights", choices=["raw", "ema"], default="ema")
    p.add_argument("--steps", type=int, help="Generation steps; defaults to 40 for grouped_partial, 50 for other modes.")
    p.add_argument("--guidance", type=float, help="Generation guidance; defaults to 2.0 for grouped_partial, 1.5 for other modes.")
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--device",help="Mapping AND generation device; defaults to the saved experiment device.")
    p.add_argument("--confidence-threshold", type=float,
                   help="Partial mapping only: override token validity threshold without refitting. Recorded in mapped metadata.")
    p.add_argument("--projection-bandwidth", type=float,
                   help="Absolute image-router projection h (patch h for patch_global). Also scales patch kernels unless patch_bandwidth is set; saved plans stay fixed.")
    p.add_argument("--patch-projection-bandwidth", type=float,
                   help="Dense grouped_partial only: absolute patch projection h, independent of image routing. No refitting.")
    p.add_argument("--top-k-images", type=int,
                   help="Retain and renormalize K target-image weights; 0 keeps all. Partial modes still use saved pairs only.")
    p.add_argument("--allow-failed-pairs", action="store_true",
                   help="Test successful pairs from a failed batched grouped_partial fit; record skipped routing mass without modifying the fit.")
    p.add_argument("--preview-checkpoint", action="store_true",
                   help="Preview factors/latest.pt from a stopped failed/interrupted grouped_patch_lowrank fit. Never refit or mark it complete.")
    a = p.parse_args(argv)
    if a.preview_checkpoint and a.allow_failed_pairs:
        p.error("--preview-checkpoint cannot be combined with --allow-failed-pairs")
    if min(a.count, a.chunk_size, a.threads) < 1 or (a.steps is not None and a.steps < 1):
        p.error("Counts, chunk size, threads and steps must be positive")
    if a.confidence_threshold is not None and not 0 <= a.confidence_threshold <= 1:
        p.error("--confidence-threshold must be finite and in [0,1]")
    if a.projection_bandwidth is not None and (not math.isfinite(a.projection_bandwidth) or a.projection_bandwidth <= 0):
        p.error("--projection-bandwidth must be finite and positive")
    if a.patch_projection_bandwidth is not None and (not math.isfinite(a.patch_projection_bandwidth) or a.patch_projection_bandwidth <= 0):
        p.error("--patch-projection-bandwidth must be finite and positive")
    if a.top_k_images is not None and a.top_k_images < 0:
        p.error("--top-k-images must be nonnegative (0 keeps all)")
    torch.set_num_threads(a.threads)
    if a.dry_run:
        # Validate small metadata before allocating supports/kernels or the DiT.
        if a.preview_checkpoint:
            from infoot_vit.infoot_helper.lowrank_preview import load_preview_snapshot, check_preview_output
            if a.output_dir is not None:
                check_preview_output(a.mapping, a.output_dir)
            m, _ = load_preview_snapshot(a.mapping)
        else:
            m = load_mapping_manifest(a.mapping, allow_failed_pairs=a.allow_failed_pairs)
        a.steps, a.guidance = generation_settings(m["config"]["mode"], a.steps, a.guidance)
        projection = projection_settings(m["config"], a.confidence_threshold,
            bandwidth=a.projection_bandwidth, top_k_images=a.top_k_images, patch_bandwidth=a.patch_projection_bandwidth)
        q = json.loads((a.query_bank / "manifest.json").read_text(encoding="utf-8"))
        training_ids = set(m["source_ids"]) | set(m["target_ids"])
        if m.get("schema") in {"siglip_infoot_mapping_v3", "siglip_lowrank_grouped_patch_v1", "siglip_lowrank_grouped_partial_v1"}:
            for name in ("source_bank", "target_bank"):
                fit_bank = json.loads((a.mapping / m[name]["path"] / "manifest.json").read_text(encoding="utf-8"))
                if (fit_bank["artifact_id"] != m[name]["artifact_id"]
                        or digest({k: v for k, v in fit_bank.items() if k != "artifact_id"}) != fit_bank["artifact_id"]):
                    raise ValueError("Fit bank identity changed.")
                training_ids.update(fit_bank["ids"])
        if (m["representation"] != q["representation"] or q["split"] not in {"val", "test"}
                or m["source_domain"] != q["domain"] or set(q["ids"]) & training_ids
                or digest({k:v for k,v in m.items() if k != "artifact_id"}) != m["artifact_id"]
                or digest({k:v for k,v in q.items() if k != "artifact_id"}) != q["artifact_id"]):
            raise ValueError("Need completed mapper and compatible held-out query bank")
        print(json.dumps(dict(mapper_id=m["artifact_id"], mode=m["config"]["mode"], fit_resources=m["resources"],
            query_count=min(a.count, len(q["ids"])), mask="existing condition_padding_mask; True is padding",
            generate=a.generate, projection=projection, incomplete_fit=m.get("incomplete_fit"),
            checkpoint_preview=m.get("checkpoint_preview"),
            sampling=dict(num_steps=a.steps, guidance_scale=a.guidance)), indent=2))
        return 0
    output = a.output_dir or ROOT / "results/infoot_vit" / f"{a.mapping.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    if a.preview_checkpoint:
        from infoot_vit.infoot_helper.lowrank_preview import LowRankCheckpointPreview, check_preview_output
        check_preview_output(a.mapping, output)
    output.mkdir(parents=True, exist_ok=False)
    with RunLog(output, "mapping_test", vars(a)) as log:
        m = json.loads((a.mapping / "manifest.json").read_text(encoding="utf-8"))
        projection = projection_settings(m["config"], a.confidence_threshold,
            bandwidth=a.projection_bandwidth, top_k_images=a.top_k_images, patch_bandwidth=a.patch_projection_bandwidth)
        # Set overrides BEFORE loading cached KDEs, factors and support thresholds.
        if a.preview_checkpoint:
            mapper = LowRankCheckpointPreview.load(a.mapping, device=a.device, projection=projection, run_log=log)
            log.write_json("checkpoint_preview_snapshot.json", mapper.manifest)
            write_json(output / "checkpoint_preview.json", mapper.manifest["checkpoint_preview"])
            print(mapper.manifest["checkpoint_preview"]["label"], flush=True)
        else:
            mapper = FeatureMapper.load(a.mapping, device=a.device, projection=projection, run_log=log,
                                        allow_failed_pairs=a.allow_failed_pairs)
        bank = FeatureBank.load(a.query_bank)
        a.steps, a.guidance = generation_settings(mapper.mode, a.steps, a.guidance)
        log.event("artifacts_loaded", mapper_id=mapper.manifest["artifact_id"], query_bank_id=bank.artifact_id)
        result, manifest = mapper.project_bank(bank, output, count=a.count, chunk_size=a.chunk_size, run_log=log)
        if a.generate:
            from infoot_vit.infoot_helper.evaluate_mapping import generate
            log.event("generation_started", num_steps=a.steps, guidance_scale=a.guidance, weights=a.weights)
            generate(mapper, bank, result, output, root=ROOT, train_config=a.train_config, eval_config=a.eval_config,
                     checkpoint=a.checkpoint, weights=a.weights, steps=a.steps, guidance=a.guidance, solver=a.solver,
                     seed=a.seed, batch_size=a.chunk_size, device=str(mapper.device))
            log.event("generation_completed", report="generation_report.json")
        print(f"Saved {len(manifest['ids'])} projections to {output}; mode={mapper.mode}")
    return 0


def generation_settings(mode, steps, guidance):
    """Use the dense partial recipe unless explicit sampling overrides are supplied."""
    partial = mode == "grouped_partial"
    return ((40 if partial else 50) if steps is None else steps,
            (2.0 if partial else 1.5) if guidance is None else guidance)


def projection_settings(config, threshold, *, bandwidth=None, top_k_images=None, patch_bandwidth=None):
    """Projection-only comparison; never mutate fit metadata or refit a plan."""
    projection = deepcopy(config["projection"])
    if patch_bandwidth is not None:
        if config["mode"] != "grouped_partial":
            raise ValueError("--patch-projection-bandwidth applies only to dense grouped_partial.")
        if isinstance(patch_bandwidth, bool) or not math.isfinite(patch_bandwidth) or patch_bandwidth <= 0:
            raise ValueError("Patch projection bandwidth must be finite and positive.")
        projection["patch_bandwidth"] = patch_bandwidth
    if threshold is not None:
        if config["mode"] not in {"grouped_partial", "grouped_partial_lowrank"}:
            raise ValueError("--confidence-threshold applies only to partial mappings.")
        projection["confidence"]["threshold"] = threshold
    if bandwidth is not None or top_k_images is not None:
        if bandwidth is not None:
            if not math.isfinite(bandwidth) or bandwidth <= 0:
                raise ValueError("Projection bandwidth must be finite and positive.")
            solver = "image_solver" if config["mode"].endswith("_lowrank") else "solver"
            projection["bandwidth_multiplier"] = bandwidth / config[solver]["h"]
        if top_k_images is not None:
            if config["mode"] == "patch_global":
                raise ValueError("patch_global has no image router; --top-k-images is not applicable.")
            if type(top_k_images) is not int or top_k_images < 0:
                raise ValueError("top_k_images must be a nonnegative integer (0 keeps all).")
            projection["top_k_images"] = top_k_images or None
    return projection


if __name__ == "__main__":
    raise SystemExit(main())
