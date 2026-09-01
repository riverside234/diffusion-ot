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
        description="Download/cache huggan/AFHQ under the project root for Stage 0."
    )
    parser.add_argument(
        "--config",
        default="configs/data/afhq_huggan.yaml",
        help="Path to the AFHQ data config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the saved local dataset copy under download_dir.",
    )
    parser.add_argument(
        "--no-save-to-disk",
        action="store_true",
        help="Only populate the Hugging Face cache_dir; do not save a local Dataset copy.",
    )
    return parser.parse_args()


def main() -> int:
    from diffusion_ot.data.afhq import download_afhq_dataset

    args = parse_args()
    report = download_afhq_dataset(
        repo_path(args.config),
        force=args.force,
        save_to_disk=not args.no_save_to_disk,
    )
    print("afhq_download_report:")
    for key, value in sorted(report.items()):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
