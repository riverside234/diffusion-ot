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
    parser = argparse.ArgumentParser(description="Train one Stage 1A PDAE domain branch on cached SiT latents.")
    parser.add_argument("--config", required=True, help="Path to a Stage 1A PDAE config.")
    parser.add_argument("--device", default=None, help="Torch device override, for example cuda or cpu.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max-step override.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config/dataset paths without loading SiT.")
    return parser.parse_args()


def main() -> int:
    from diffusion_ot.training.train_pdae_domain import train_pdae_domain

    args = parse_args()
    report = train_pdae_domain(
        repo_path(args.config),
        device=args.device,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
    )
    print("pdae_train_report:")
    for key, value in sorted(report.to_dict().items()):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
