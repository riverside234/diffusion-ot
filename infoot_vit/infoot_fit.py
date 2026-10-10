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
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, inspect_fit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--source-bank", default=None)
    p.add_argument("--target-bank", default=None)
    p.add_argument("--mode", choices=["patch_global", "whole_map", "grouped_patch", "grouped_partial"])
    p.add_argument("--h", type=float)
    p.add_argument("--reg", type=float)
    p.add_argument("--lam", type=float)
    p.add_argument("--keep-mass", type=float, help="Partial patch mass; use 1.0 for the matched balanced-pair baseline.")
    p.add_argument("--output-root", type=Path, help="Parent of a NEW timestamped fit directory.")
    p.add_argument("--resume", type=Path, help="Exact failed/interrupted fit directory; matching fingerprints required.")
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config else dict(
        mode="patch_global", source_bank="data/infoot_vit/cat_train", target_bank="data/infoot_vit/dog_train",
        solver=dict(h=.4, reg=.02, lam=1., cost_scale=1.))
    for key in ("source_bank", "target_bank", "mode"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ("h", "reg", "lam"):
        if getattr(args, key) is not None:
            config.setdefault("solver", {})[key] = getattr(args, key)
    if args.keep_mass is not None:
        config.setdefault("partial", {})["keep_mass"] = args.keep_mass
    if args.dry_run:
        print(json.dumps(inspect_fit(config, args.project_root), indent=2))
    else:
        directory = fit_mapping(config, root=args.project_root, output_root=args.output_root, resume=args.resume)
        print(f"Saved fitted mapping: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
