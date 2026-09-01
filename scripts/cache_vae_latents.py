from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def repo_path(path: str | None) -> Path | None:
    if path is None:
        return None
    raw_path = Path(path)
    return raw_path if raw_path.is_absolute() else ROOT / raw_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache VAE latents for AFHQ manifests using SiT-B-2-256 VAE.")
    parser.add_argument(
        "--data-config",
        default="configs/data/afhq_huggan.yaml",
        help="Path to AFHQ data config.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/sit_b2_256.yaml",
        help="Path to SiT model config.",
    )
    parser.add_argument(
        "--pretrained-config",
        default=None,
        help="Optional override for pretrained snapshot config.",
    )
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override VAE batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-manifest record limit.")
    parser.add_argument("--dry-run", action="store_true", help="Build latent manifest without encoding latents.")
    return parser.parse_args()


def main() -> int:
    from diffusion_ot.data.latent_cache import cache_vae_latents

    args = parse_args()
    report = cache_vae_latents(
        data_config_path=repo_path(args.data_config),
        model_config_path=repo_path(args.model_config),
        pretrained_config_path=repo_path(args.pretrained_config),
        project_root=ROOT,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print("latent_cache_report:")
    for key, value in sorted(report.items()):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
