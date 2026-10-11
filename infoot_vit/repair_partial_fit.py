"""Repair only failed dense partial pairs in a NEW fit, preserving the source."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from infoot_vit.infoot_helper.fit_mapping import fit_mapping
from infoot_vit.infoot_helper.partial_repair import repair_partial_fit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--source", type=Path, help="Stopped failed fit; uses its exact saved config.")
    action.add_argument("--resume", type=Path, help="Existing repair directory; exact fingerprint required.")
    parser.add_argument("--output", type=Path, help="New directory required with --source; never overwritten.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.source is not None) != (args.output is not None):
        parser.error("Use --source with --output, or --resume without --output; threads must be positive.")
    torch.set_num_threads(args.threads)
    if args.source:
        directory = repair_partial_fit(args.source, args.output, root=args.project_root)
    else:
        manifest = json.loads((args.resume / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("repair", {}).get("policy") != "strict_failed_pairs_new_artifact_v1":
            parser.error("--resume requires a directory prepared by this repair tool.")
        directory = fit_mapping(manifest["config"], root=args.project_root, resume=args.resume)
    print(f"Saved repaired mapping: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
