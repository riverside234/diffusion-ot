from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from diffusion_ot.integrations.hf_snapshot import (
    effective_project_root,
    find_project_root,
    load_yaml_config,
    resolve_project_local_path,
)


def afhq_storage_paths(config_path: str | Path) -> tuple[dict[str, Any], Path | None, Path | None]:
    config = load_yaml_config(config_path)
    project_root = effective_project_root(config, fallback=find_project_root(Path(config_path).resolve()))
    cache_path = None
    download_path = None
    if config.get("cache_dir"):
        cache_path = resolve_project_local_path(config["cache_dir"], project_root, field_name="cache_dir")
    if config.get("download_dir"):
        download_path = resolve_project_local_path(
            config["download_dir"],
            project_root,
            field_name="download_dir",
        )
    return config, cache_path, download_path


def is_saved_dataset_dir(path: Path) -> bool:
    return (path / "dataset_info.json").exists() or (path / "state.json").exists()


def load_afhq_dataset(config_path: str | Path):
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required to load huggan/AFHQ. Install datasets.") from exc

    config, cache_path, download_path = afhq_storage_paths(config_path)
    if download_path is not None and is_saved_dataset_dir(download_path):
        return load_from_disk(str(download_path))
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
    return load_dataset(
        config["dataset_id"],
        split=config.get("split", "train"),
        cache_dir=str(cache_path) if cache_path is not None else None,
    )


def download_afhq_dataset(
    config_path: str | Path,
    force: bool = False,
    save_to_disk: bool = True,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to download huggan/AFHQ. Install datasets.") from exc

    config, cache_path, download_path = afhq_storage_paths(config_path)
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    if save_to_disk and download_path is None:
        raise ValueError("save_to_disk requires download_dir in the data config.")
    if save_to_disk and download_path is not None and is_saved_dataset_dir(download_path) and not force:
        return {
            "dataset_id": config["dataset_id"],
            "split": config.get("split", "train"),
            "cache_dir": str(cache_path) if cache_path else None,
            "download_dir": str(download_path),
            "saved_to_disk": False,
            "already_exists": True,
            "num_rows": None,
        }

    dataset = load_dataset(
        config["dataset_id"],
        split=config.get("split", "train"),
        cache_dir=str(cache_path) if cache_path is not None else None,
    )

    saved_to_disk = False
    if save_to_disk and download_path is not None:
        download_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(download_path))
        saved_to_disk = True
        report_path = download_path / "download_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "dataset_id": config["dataset_id"],
                    "split": config.get("split", "train"),
                    "cache_dir": str(cache_path) if cache_path else None,
                    "download_dir": str(download_path),
                    "num_rows": len(dataset),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "dataset_id": config["dataset_id"],
        "split": config.get("split", "train"),
        "cache_dir": str(cache_path) if cache_path else None,
        "download_dir": str(download_path) if download_path else None,
        "saved_to_disk": saved_to_disk,
        "already_exists": False,
        "num_rows": len(dataset),
    }


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
