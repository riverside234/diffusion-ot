from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description="Stage 4: all held-out translations, FID, source SSIM and 16 pairs/direction.")
    parser.add_argument("--config", default="configs/stage4_eval/fid_ssim_sit_b2.yaml")
    parser.add_argument("--bundle", default="outputs/s23_rms", help="Completed Stage 2-3 offline alignment output directory.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="Override all evaluation devices.")
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from diffusion_ot.evaluation.final_eval import run_stage4
    root = args.project_root.resolve()
    report = run_stage4(root / args.config, args.bundle, root=root, output_dir=args.output_dir,
                        device=args.device, resume=args.resume,
                        generation_batch_size=args.generation_batch_size)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
