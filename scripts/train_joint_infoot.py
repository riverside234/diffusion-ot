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
        description="Jointly fine-tune Cat and Dog PDAE encoders with mini-batch plain InfoOT."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device-cat", default=None)
    parser.add_argument("--device-dog", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from a Stage 1B checkpoint, or latest.pt when no path is supplied.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    from diffusion_ot.training.train_joint_infoot import train_joint_infoot

    args = parse_args()
    report = train_joint_infoot(
        repo_path(args.config),
        device_cat=args.device_cat,
        device_dog=args.device_dog,
        max_steps=args.max_steps,
        resume_from=args.resume,
        dry_run=args.dry_run,
    )
    print("stage1b_train_report:")
    for key, value in sorted(report.to_dict().items()):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
