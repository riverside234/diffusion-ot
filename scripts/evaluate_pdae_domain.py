from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def repo_path(path: str) -> Path:
    raw_path = Path(path)
    return raw_path if raw_path.is_absolute() else ROOT / raw_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stage 1A PDAE reconstruction smoke gate for one domain."
    )
    parser.add_argument(
        "--train-config",
        required=True,
        help="Cat or Dog Stage 1A training config.",
    )
    parser.add_argument(
        "--eval-config",
        required=True,
        help="Shared Stage 1A evaluation config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--weights",
        choices=["ema", "raw"],
        default=None,
        help="Checkpoint weights override; the evaluation config defaults to EMA.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint override; defaults to the training run's latest.pt.",
    )
    return parser.parse_args()


def main() -> int:
    from diffusion_ot.evaluation.stage1a_eval import run_stage1a_smoke_test

    args = parse_args()
    report = run_stage1a_smoke_test(
        repo_path(args.train_config),
        repo_path(args.eval_config),
        device=args.device,
        weights=args.weights,
        checkpoint_path=args.checkpoint,
    )
    print("stage1a_smoke_report:")
    for key, value in sorted(report.to_dict().items()):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

