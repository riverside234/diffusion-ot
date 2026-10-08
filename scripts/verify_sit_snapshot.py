from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def repo_path(path: str) -> Path:
    raw_path = Path(path)
    return raw_path if raw_path.is_absolute() else ROOT / raw_path


def parse_args(argv=None, *, default_config="configs/pretrained/bilisakura_sit_b2_256.yaml",
               download_by_default=False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify or download a local SiT Diffusers snapshot (transformer + VAE).")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the pretrained snapshot config.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        default=download_by_default,
        help="Optionally download the HF snapshot if required files are missing.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write snapshot_report.json.",
    )
    return parser.parse_args(argv)


def main(argv=None, *, default_config="configs/pretrained/bilisakura_sit_b2_256.yaml",
         download_by_default=False) -> int:
    from diffusion_ot.integrations.hf_snapshot import format_snapshot_report, verify_sit_snapshot

    args = parse_args(argv, default_config=default_config, download_by_default=download_by_default)
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
