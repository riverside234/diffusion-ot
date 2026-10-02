from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_ot.config_defaults import stage1a_training_config


def repo_path(path: str) -> Path:
    raw_path = Path(path)
    return raw_path if raw_path.is_absolute() else ROOT / raw_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Stage 1A PDAE branch with RGB or latent semantic input and cached SiT latent flow targets.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", help="Explicit Stage 1A recipe (including historical experiments).")
    selection.add_argument("--domain", choices=["cat", "dog"], help="Use this domain's default fresh residual-cosmap recipe.")
    parser.add_argument("--device", default=None, help="Torch device override using process-visible indices. With one GPU in CUDA_VISIBLE_DEVICES, use cuda:0; cpu is also supported.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional optimizer-update limit override.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from a checkpoint path, or from this run's latest.pt when passed without a path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve config/dataset paths without loading SiT.")
    return parser.parse_args()


def main() -> int:
    from diffusion_ot.training.train_pdae_domain import train_pdae_domain

    args = parse_args()
    report = train_pdae_domain(
        repo_path(args.config or stage1a_training_config(args.domain)),
        device=args.device,
        max_steps=args.max_steps,
        resume_from=args.resume,
        dry_run=args.dry_run,
    )
    print("pdae_train_report:")
    for key, value in sorted(report.to_dict().items()):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
