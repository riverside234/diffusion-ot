from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from diffusion_ot.data.manifests import read_jsonl
from diffusion_ot.integrations.hf_snapshot import (
    effective_project_root,
    load_yaml_config,
    resolve_project_local_path,
)


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_latent_tensor(path: str | Path):
    value = _torch_load(Path(path))
    if hasattr(value, "shape"):
        return value
    if isinstance(value, dict):
        for key in ("x0_latent", "latent", "latents", "tensor"):
            if key in value:
                return value[key]
    raise TypeError(f"Expected tensor or tensor dict in latent file: {path}")


def latent_root_from_config(
    data_config: dict[str, Any],
    project_root: str | Path | None = None,
) -> Path:
    root = effective_project_root(data_config, fallback=project_root)
    return resolve_project_local_path(
        data_config.get("latent_dir", "data/latents"),
        root,
        field_name="latent_dir",
    )


def latent_manifest_path(
    data_config: dict[str, Any],
    project_root: str | Path | None = None,
) -> Path:
    manifest_name = data_config.get("latent_manifest_name", "latent_manifest.jsonl")
    return latent_root_from_config(data_config, project_root) / manifest_name


def split_manifest_path(
    data_config: dict[str, Any],
    domain: str,
    split: str,
    project_root: str | Path | None = None,
) -> Path:
    root = effective_project_root(data_config, fallback=project_root)
    manifest_dir = resolve_project_local_path(
        data_config.get("manifest_dir", "data/manifests"),
        root,
        field_name="manifest_dir",
    )
    return manifest_dir / f"{domain}_{split}.jsonl"


def _normalize_record_paths(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        item = dict(record)
        if "latent_path" in item:
            item["latent_path"] = str(Path(item["latent_path"]))
        normalized.append(item)
    return normalized


def records_from_latent_manifest(
    data_config: dict[str, Any],
    domain: str,
    split: str,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    path = latent_manifest_path(data_config, project_root)
    if not path.exists():
        return []
    split_name = f"{domain}_{split}"
    records = [
        record
        for record in read_jsonl(path)
        if record.get("domain") == domain and Path(str(record.get("latent_path", ""))).parent.name == split_name
    ]
    return _normalize_record_paths(records)


def records_from_split_manifest(
    data_config: dict[str, Any],
    domain: str,
    split: str,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    path = split_manifest_path(data_config, domain, split, project_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing split manifest: {path}")

    root = latent_root_from_config(data_config, project_root)
    split_name = f"{domain}_{split}"
    records = []
    for record in read_jsonl(path):
        item = dict(record)
        item["latent_path"] = str(root / split_name / f"{item['sample_id']}.pt")
        item.setdefault("latent_shape", [4, 32, 32])
        records.append(item)
    return records


def load_latent_records(
    data_config_path: str | Path,
    domain: str,
    split: str,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    data_config = load_yaml_config(data_config_path)
    domain = domain.lower()
    split = split.lower()
    records = records_from_latent_manifest(data_config, domain, split, project_root)
    if records:
        return records
    return records_from_split_manifest(data_config, domain, split, project_root)


class CachedLatentDataset:
    def __init__(
        self,
        data_config_path: str | Path,
        domain: str,
        split: str = "train",
        project_root: str | Path | None = None,
        limit: int | None = None,
        validate_exists: bool = True,
        validate_shape: bool = True,
        random_horizontal_flip: float = 0.0,
    ) -> None:
        self.data_config_path = Path(data_config_path)
        self.domain = domain.lower()
        self.split = split.lower()
        self.records = load_latent_records(self.data_config_path, self.domain, self.split, project_root)
        if limit is not None:
            self.records = self.records[:limit]
        self.validate_shape = validate_shape
        self.random_horizontal_flip = float(random_horizontal_flip)
        if not 0.0 <= self.random_horizontal_flip <= 1.0:
            raise ValueError("random_horizontal_flip must be between 0 and 1.")

        if validate_exists:
            missing = [record["latent_path"] for record in self.records if not Path(record["latent_path"]).exists()]
            if missing:
                preview = "\n".join(f"  - {path}" for path in missing[:10])
                raise FileNotFoundError(
                    f"Missing {len(missing)} cached latent files for {self.domain}_{self.split}.\n{preview}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        record = self.records[index]
        latent = load_latent_tensor(record["latent_path"])
        expected_shape = record.get("latent_shape")
        if self.validate_shape and expected_shape and list(latent.shape) != list(expected_shape):
            raise ValueError(
                f"Latent shape mismatch for {record['latent_path']}: "
                f"expected {expected_shape}, got {list(latent.shape)}"
            )
        if self.random_horizontal_flip > 0.0:
            if self.random_horizontal_flip >= 1.0 or torch.rand(()) < self.random_horizontal_flip:
                latent = torch.flip(latent, dims=(-1,))
        return {
            "x0_latent": latent,
            "sample_id": record.get("sample_id"),
            "domain": record.get("domain", self.domain),
            "split": self.split,
            "latent_path": record["latent_path"],
            "metadata": record,
        }


def collate_latent_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "x0_latent": torch.stack([item["x0_latent"] for item in items], dim=0),
        "sample_id": [item["sample_id"] for item in items],
        "domain": [item["domain"] for item in items],
        "split": [item["split"] for item in items],
        "latent_path": [item["latent_path"] for item in items],
        "metadata": [item["metadata"] for item in items],
    }
