"""Fit immutable SigLIP feature-map transports; never load a diffusion model."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, inspect_fit, display_inspection


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--source-bank", default=None)
    p.add_argument("--target-bank", default=None)
    p.add_argument("--mode", choices=["patch_global", "whole_map", "grouped_patch", "grouped_partial"])
    p.add_argument("--h", type=float, help="solver.h: balanced router/global-patch bandwidth (not partial.solver).")
    p.add_argument("--reg", type=float, help="solver.reg: balanced entropy regularization (not partial.solver).")
    p.add_argument("--lam", type=float, help="solver.lam: balanced MI weight (not partial.solver).")
    p.add_argument("--max-outer-steps", type=int, help="solver.max_outer_steps; changed settings require a new fit.")
    p.add_argument("--partial-h", type=float, help="grouped_partial: patch-pair KDE bandwidth.")
    p.add_argument("--partial-reg", type=float, help="grouped_partial: patch-pair entropy regularization.")
    p.add_argument("--partial-lam", type=float, help="grouped_partial: patch-pair MI weight.")
    p.add_argument("--partial-max-steps", type=int, help="grouped_partial: patch-pair outer iteration budget.")
    p.add_argument("--keep-mass", type=float, help="Partial patch mass; use 1.0 for the matched balanced-pair baseline.")
    p.add_argument("--sample-images", type=int, help="grouped_partial only: training images per domain, sampled without replacement.")
    p.add_argument("--sample-seed", type=int, help="grouped_partial only: sampling seed (default 42).")
    pairs = p.add_mutually_exclusive_group()
    pairs.add_argument("--fit-pair-top-k", type=int, help="grouped_partial only: target pairs fitted per source image.")
    pairs.add_argument("--full-pairs", action="store_true", help="grouped_partial only: fit every image pair.")
    batch = p.add_mutually_exclusive_group()
    batch.add_argument("--pair-batch-size", type=int, help="grouped_partial: independent dense pair plans per GPU/CPU batch (including 1).")
    batch.add_argument("--serial-pairs", action="store_true", help="grouped_partial: use the original serial POT reference solver.")
    p.add_argument("--output-root", type=Path, help="Parent of a NEW timestamped fit directory.")
    p.add_argument("--resume", type=Path, help="Exact failed/interrupted fit directory; matching fingerprints required.")
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device",help="Device for image router, kernels and transport solvers; overrides YAML.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tune", action="store_true", help="Fit only the image router; terminal output only, no files/plans saved.")
    p.add_argument("--tune-log-every", type=int, default=25, help="Router tuning progress interval (default 25).")
    args = p.parse_args(argv)
    if args.tune and (args.resume or args.output_root or args.dry_run):
        p.error("--tune cannot be combined with --resume, --output-root or --dry-run")
    if args.tune_log_every < 1:
        p.error("--tune-log-every must be positive")
    if args.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config else dict(
        mode="patch_global", source_bank="data/infoot_vit/cat_train", target_bank="data/infoot_vit/dog_train",
        solver=dict(h=.4, reg=.02, lam=1., cost_scale=1.),device="cuda")
    for key in ("source_bank", "target_bank", "mode", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ("h", "reg", "lam", "max_outer_steps"):
        if getattr(args, key) is not None:
            config.setdefault("solver", {})[key] = getattr(args, key)
    for argument, field in ((args.partial_h, "h"), (args.partial_reg, "reg"),
                            (args.partial_lam, "lam"), (args.partial_max_steps, "max_outer_steps")):
        if argument is not None:
            config.setdefault("partial", {}).setdefault("solver", {})[field] = argument
    if args.keep_mass is not None:
        config.setdefault("partial", {})["keep_mass"] = args.keep_mass
    for argument, field in ((args.sample_images, "images_per_domain"), (args.sample_seed, "seed")):
        if argument is not None:
            config.setdefault("sampling", {})[field] = argument
    if args.full_pairs or args.fit_pair_top_k is not None:
        config["fit_pair_top_k"] = None if args.full_pairs else args.fit_pair_top_k
    if args.serial_pairs or args.pair_batch_size is not None:
        config["pair_batch_size"] = None if args.serial_pairs else args.pair_batch_size
    if args.tune:
        from infoot_vit.infoot_helper.tuning import tune_image_router
        report = tune_image_router(config, root=args.project_root, log_every=args.tune_log_every)
        return 0 if report["status"] == "converged" else 2
    if args.dry_run:
        print(json.dumps(display_inspection(inspect_fit(config, args.project_root)), indent=2))
    else:
        directory = fit_mapping(config, root=args.project_root, output_root=args.output_root, resume=args.resume)
        print(f"Saved fitted mapping: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
