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
    parser = argparse.ArgumentParser(description="Inspect huggan/AFHQ labels and first-pass Cat/Dog counts.")
    parser.add_argument(
        "--config",
        default="configs/data/afhq_huggan.yaml",
        help="Path to the AFHQ data config.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a quick smoke check.")
    return parser.parse_args()


def main() -> int:
    from collections import Counter

    from diffusion_ot.data.afhq import build_afhq_records, label_name, load_afhq_dataset
    from diffusion_ot.integrations.hf_snapshot import load_yaml_config

    args = parse_args()
    config_path = repo_path(args.config)
    config = load_yaml_config(config_path)
    dataset = load_afhq_dataset(config_path)
    if args.limit is not None:
        dataset_for_counts = dataset.select(range(min(args.limit, len(dataset))))
    else:
        dataset_for_counts = dataset

    label_column = config.get("label_column", "label")
    label_feature = dataset.features.get(label_column)
    domains = [str(item).lower() for item in config.get("domains", [])]
    raw_counts = Counter(
        label_name(row[label_column], label_feature, domains) for row in dataset_for_counts
    )
    active_records = build_afhq_records(dataset_for_counts, config)
    active_counts = Counter(record["domain"] for record in active_records)

    print(f"dataset_id: {config['dataset_id']}")
    print(f"split: {config.get('split', 'train')}")
    print(f"rows_checked: {len(dataset_for_counts)}")
    print("raw_label_counts:")
    for domain, count in sorted(raw_counts.items()):
        print(f"  {domain}: {count}")
    print("active_domain_counts:")
    for domain, count in sorted(active_counts.items()):
        print(f"  {domain}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
