"""Recover the original RGB targets using the latent cache's sample metadata."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from diffusion_ot.data.afhq import (
    dataset_label_feature, load_afhq_dataset, record_from_row,
)
from diffusion_ot.data.latent_cache import image_to_tensor
from diffusion_ot.integrations.hf_snapshot import load_yaml_config


def validate_original_image_records(config: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Reject missing or mismatched source identities before loading any images."""
    dataset_id, split = config["dataset_id"], config.get("split", "train")
    for record in records:
        required = {"sample_id", "hf_index", "image_column", "domain", "dataset_id", "dataset_split"}
        if missing := required - record.keys():
            raise ValueError(f"Original-image metadata missing {sorted(missing)} for {record.get('sample_id')}; "
                             "rebuild the latent bank from the source manifest.")
        if record["dataset_id"] != dataset_id or record["dataset_split"] != split:
            raise ValueError(f"Original-image dataset mismatch for {record['sample_id']}.")
        if config.get("image_column", record["image_column"]) != record["image_column"]:
            raise ValueError(f"Original-image column mismatch for {record['sample_id']}.")


def _original_image_tensors(config: dict[str, Any], records: list[dict[str, Any]], dataset):
    """Load metadata-checked source RGB using the latent cache preprocessing."""
    validate_original_image_records(config, records)
    dataset_id, split = config["dataset_id"], config.get("split", "train")
    label_column = config.get("label_column", "label")
    label_feature = dataset_label_feature(dataset, label_column)
    domains = [str(d).lower() for d in config.get("domains", [])]
    images = []
    for record in records:
        index = int(record["hf_index"])
        if index < 0 or index >= len(dataset):
            raise ValueError(f"Original-image index out of range for {record['sample_id']}: {index}.")
        row = dataset[index]
        missing_columns = {label_column, record["image_column"]} - row.keys()
        if missing_columns:
            raise ValueError(f"Original-image source columns missing {sorted(missing_columns)} "
                             f"for {record['sample_id']}.")
        actual = record_from_row(dataset_id, split, index, row, label_column,
                                 record["image_column"], label_feature, domains)
        if actual["sample_id"] != record["sample_id"] or actual["domain"] != record["domain"]:
            raise ValueError(f"Original-image identity mismatch for {record['sample_id']}.")
        pixels = image_to_tensor(row[record["image_column"]], int(config.get("image_size", 256)),
                                 bool(config.get("center_crop", True)))
        images.append(pixels)
    return images


class OriginalImageLoader:
    """Reusable original-RGB reader with one lazily opened HF dataset per process.

    Configuration is parsed once, not per sample. Spawned DataLoader workers
    receive no open dataset; forked workers also reopen it on their first read.
    Returned tensors are float32 CHW in [-1,1], exactly as for latent caching.
    """

    def __init__(self, data_config_path: str | Path, records: list[dict[str, Any]]) -> None:
        self.data_config_path = Path(data_config_path)
        self.config = load_yaml_config(self.data_config_path)
        validate_original_image_records(self.config, records)
        self._dataset = None
        self._dataset_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dataset"] = None
        state["_dataset_pid"] = None
        return state

    def load(self, record: dict[str, Any]):
        pid = os.getpid()
        if self._dataset is None or self._dataset_pid != pid:
            self._dataset = load_afhq_dataset(self.data_config_path)
            self._dataset_pid = pid
        return _original_image_tensors(self.config, [record], self._dataset)[0]


def load_ground_truth_images(data_config_path: str | Path, records: list[dict[str, Any]], *, dataset=None):
    """Return preprocessed original images in [0,1], in the supplied order.

    Use exactly the crop/resize used when caching x0. Missing source metadata
    is an error: decoding x0 with the VAE is not an original-image fallback.
    """
    import torch

    if not records:
        raise ValueError("Ground-truth image evaluation requires at least one sample.")
    config = load_yaml_config(data_config_path)
    validate_original_image_records(config, records)
    # Offline evaluation can share one Arrow-backed dataset across image chunks.
    if dataset is None:
        dataset = load_afhq_dataset(data_config_path)
    return (torch.stack(_original_image_tensors(config, records, dataset)) + 1) / 2
