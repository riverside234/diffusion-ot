from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from diffusion_ot.integrations.hf_snapshot import effective_project_root, resolve_project_local_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def deterministic_score(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def split_by_domain(
    records: Iterable[dict[str, Any]],
    seed: int,
    val_fraction: float,
    max_samples_per_domain: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["domain"]].append(record)

    output: dict[str, list[dict[str, Any]]] = {}
    for domain, domain_records in sorted(grouped.items()):
        ordered = sorted(domain_records, key=lambda item: deterministic_score(seed, item["sample_id"]))
        if max_samples_per_domain is not None:
            ordered = ordered[:max_samples_per_domain]
        val_count = max(1, round(len(ordered) * val_fraction)) if ordered else 0
        val_records = ordered[:val_count]
        train_records = ordered[val_count:]
        output[f"{domain}_train"] = train_records
        output[f"{domain}_val"] = val_records
    return output


def write_split_manifests(
    records: Iterable[dict[str, Any]],
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, int]:
    root = effective_project_root(config, fallback=project_root)
    manifest_dir = resolve_project_local_path(
        config.get("manifest_dir", "data/manifests"),
        root,
        field_name="manifest_dir",
    )
    splits = split_by_domain(
        records,
        seed=int(config.get("seed", 0)),
        val_fraction=float(config.get("val_fraction", 0.1)),
        max_samples_per_domain=config.get("max_samples_per_domain"),
    )

    counts: dict[str, int] = {}
    for split_name, split_records in splits.items():
        counts[split_name] = write_jsonl(manifest_dir / f"{split_name}.jsonl", split_records)

    label_map = {
        "domains": config.get("domains", []),
        "active_domains": config.get("active_domains", []),
        "heldout_domains": config.get("heldout_domains", []),
        "label_column": config.get("label_column", "label"),
    }
    (manifest_dir / "afhq_label_map.json").write_text(
        json.dumps(label_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return counts


def manifest_paths(config: dict[str, Any], project_root: str | Path | None = None) -> list[Path]:
    root = effective_project_root(config, fallback=project_root)
    manifest_dir = resolve_project_local_path(
        config.get("manifest_dir", "data/manifests"),
        root,
        field_name="manifest_dir",
    )
    active_domains = [str(item).lower() for item in config.get("active_domains", [])]
    paths = []
    for domain in active_domains:
        paths.append(manifest_dir / f"{domain}_train.jsonl")
        paths.append(manifest_dir / f"{domain}_val.jsonl")
    return paths
