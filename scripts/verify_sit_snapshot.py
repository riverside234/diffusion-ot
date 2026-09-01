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
    parser = argparse.ArgumentParser(description="Verify the local SiT-B-2-256 Diffusers snapshot.")
    parser.add_argument(
        "--config",
        default="configs/pretrained/bilisakura_sit_b2_256.yaml",
        help="Path to the pretrained snapshot config.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Optionally download the HF snapshot if required files are missing.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write snapshot_report.json.",
    )
    return parser.parse_args()


def main() -> int:
    from diffusion_ot.integrations.hf_snapshot import format_snapshot_report, verify_sit_snapshot

    args = parse_args()
    report = verify_sit_snapshot(
        repo_path(args.config),
        project_root=ROOT,
        allow_download=args.download_if_missing,
        write_report=not args.no_report,
    )
    print(format_snapshot_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
