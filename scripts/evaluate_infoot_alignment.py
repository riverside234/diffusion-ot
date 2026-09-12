from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def repo_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Stage 1A offline-InfoOT baseline or a plain/fused Stage 1B checkpoint."
    )
    parser.add_argument("--alignment-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Stage 1B checkpoint. Omit it to evaluate frozen Stage 1A encoders plus offline InfoOT.",
    )
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--device-cat", default=None)
    parser.add_argument("--device-dog", default=None)
    parser.add_argument("--max-reference", type=int, default=None)
    parser.add_argument(
        "--max-projection",
        type=int,
        default=None,
        help="Bound the Eq. (7) target projection bank; the config default uses all samples.",
    )
    parser.add_argument("--max-query", type=int, default=None)
    parser.add_argument(
        "--projection-bandwidth", type=float, default=None,
        help="Override the conditional retrieval/projection bandwidth multiplier; "
        "the InfoOT fitting bandwidth is unchanged.",
    )
    parser.add_argument(
        "--require-stage1a-baseline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require a protocol-matched Stage 1A report before evaluating a Stage 1B "
        "checkpoint. The config value is used when this option is omitted; pass "
        "--no-require-stage1a-baseline for a standalone checkpoint evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    from diffusion_ot.evaluation.stage1b_eval import run_stage1b_evaluation

    args = parse_args()
    report = run_stage1b_evaluation(
        repo_path(args.alignment_config),
        repo_path(args.eval_config),
        checkpoint_path=repo_path(args.checkpoint) if args.checkpoint else None,
        weights=args.weights,
        device_cat=args.device_cat,
        device_dog=args.device_dog,
        max_reference=args.max_reference,
        max_projection=args.max_projection,
        max_query=args.max_query,
        projection_bandwidth=args.projection_bandwidth,
        require_stage1a_baseline=args.require_stage1a_baseline,
    )
    print("stage1b_evaluation_report:")
    print(f"  mode: {report.mode}")
    print(f"  output_dir: {report.output_dir}")
    print(f"  stage1a_architectures: {report.stage1a_architectures}")
    print(f"  generation_protocol: {report.generation_protocol}")
    print(f"  mutual_information: {report.solver['mutual_information']}")
    print(f"  row_residual: {report.solver['row_residual']}")
    print(f"  column_residual: {report.solver['column_residual']}")
    print(f"  baseline_comparison: {report.baseline_comparison['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
