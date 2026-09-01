from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from diffusion_ot.integrations.hf_snapshot import (
    effective_project_root,
    find_project_root,
    load_yaml_config,
    resolve_project_local_path,
)


def load_afhq_dataset(config_path: str | Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load huggan/AFHQ. Install datasets.") from exc

    config = load_yaml_config(config_path)
    cache_dir = config.get("cache_dir")
    download_dir = config.get("download_dir")
    cache_path = None
    download_path = None
    project_root = effective_project_root(config, fallback=find_project_root(Path(config_path).resolve()))
    if cache_dir:
        cache_path = resolve_project_local_path(cache_dir, project_root, field_name="cache_dir")
        cache_path.mkdir(parents=True, exist_ok=True)
    if download_dir:
        download_path = resolve_project_local_path(download_dir, project_root, field_name="download_dir")
        download_path.mkdir(parents=True, exist_ok=True)
    return load_dataset(
        config["dataset_id"],
        split=config.get("split", "train"),
        cache_dir=str(cache_path) if cache_path is not None else None,
    )


def label_name(label_value: Any, label_feature: Any, domains: list[str]) -> str:
    if isinstance(label_value, str):
        return label_value.lower()
    if hasattr(label_feature, "int2str"):
        return str(label_feature.int2str(int(label_value))).lower()
    if isinstance(label_value, int) and 0 <= label_value < len(domains):
        return domains[label_value].lower()
    return str(label_value).lower()


def dataset_label_feature(dataset: Any, label_column: str) -> Any:
    features = getattr(dataset, "features", {})
    return features.get(label_column) if hasattr(features, "get") else None


def jsonable_label(label_value: Any) -> Any:
    if hasattr(label_value, "item"):
        return label_value.item()
    return label_value


def record_from_row(
    dataset_id: str,
    split: str,
    index: int,
    row: dict[str, Any],
    label_column: str,
    image_column: str,
    label_feature: Any,
    domains: list[str],
) -> dict[str, Any]:
    domain = label_name(row[label_column], label_feature, domains)
    return {
        "sample_id": f"afhq_{domain}_{index:06d}",
        "dataset_id": dataset_id,
        "dataset_split": split,
        "hf_index": index,
        "domain": domain,
        "label": jsonable_label(row[label_column]),
        "image_column": image_column,
        "label_column": label_column,
    }


def build_afhq_records(dataset: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_id = config["dataset_id"]
    split = config.get("split", "train")
    label_column = config.get("label_column", "label")
    image_column = config.get("image_column", "image")
    domains = [str(item).lower() for item in config.get("domains", [])]
    active_domains = {str(item).lower() for item in config.get("active_domains", domains)}
    label_feature = dataset_label_feature(dataset, label_column)

    records = []
    for index, row in enumerate(dataset):
        record = record_from_row(
            dataset_id=dataset_id,
            split=split,
            index=index,
            row=row,
            label_column=label_column,
            image_column=image_column,
            label_feature=label_feature,
            domains=domains,
        )
        if record["domain"] in active_domains:
            records.append(record)
    return records


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(record["domain"] for record in records))


def load_config_and_records(config_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_yaml_config(config_path)
    dataset = load_afhq_dataset(config_path)
    return config, build_afhq_records(dataset, config)
