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
    parser = argparse.ArgumentParser(description="Create deterministic AFHQ Cat/Dog JSONL manifests.")
    parser.add_argument(
        "--config",
        default="configs/data/afhq_huggan.yaml",
        help="Path to the AFHQ data config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional total record limit after active-domain filtering.",
    )
    return parser.parse_args()


def main() -> int:
    from diffusion_ot.data.afhq import build_afhq_records, load_afhq_dataset
    from diffusion_ot.data.manifests import write_split_manifests
    from diffusion_ot.integrations.hf_snapshot import load_yaml_config

    args = parse_args()
    config_path = repo_path(args.config)
    config = load_yaml_config(config_path)
    dataset = load_afhq_dataset(config_path)
    records = build_afhq_records(dataset, config)
    if args.limit is not None:
        records = records[: args.limit]
    counts = write_split_manifests(records, config, project_root=ROOT)

    print("wrote_manifests:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
