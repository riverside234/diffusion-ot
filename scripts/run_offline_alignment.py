from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2-3: Frozen Bank Construction and Global InfoOT Fitting.")
    parser.add_argument("--config", default="configs/stage23_offline/full_sit_b2.yaml")
    parser.add_argument("--checkpoint", required=True, help="Selected format-4 Stage 1B checkpoint.")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="Override both encoding and fitting device.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-iterations", type=int, default=None,
                        help="Full-size preflight: save progress after this many outer updates.")
    args = parser.parse_args()
    from diffusion_ot.evaluation.offline_pipeline import run_stage23
    root = args.project_root.resolve()
    report = run_stage23(root / args.config, args.checkpoint, root=root, output_dir=args.output_dir,
                        weights=args.weights, device=args.device, resume=args.resume,
                        max_new_iterations=args.max_new_iterations)
    print(json.dumps(report, indent=2))
    return 2 if report["status"] == "iteration_limit" else 0


if __name__ == "__main__":
    raise SystemExit(main())
