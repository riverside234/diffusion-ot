"""Recover the original RGB targets using the latent cache's sample metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from diffusion_ot.data.afhq import (
    dataset_label_feature, load_afhq_dataset, record_from_row,
)
from diffusion_ot.data.latent_cache import image_to_tensor
from diffusion_ot.integrations.hf_snapshot import load_yaml_config


def load_ground_truth_images(data_config_path: str | Path, records: list[dict[str, Any]]):
    """Return preprocessed original images in [0,1], in the supplied order.

    Use exactly the crop/resize used when caching x0. Missing source metadata
    is an error: decoding x0 with the VAE is not an original-image fallback.
    """
    import torch

    if not records:
        raise ValueError("Ground-truth image evaluation requires at least one sample.")
    config = load_yaml_config(data_config_path)
    dataset_id, split = config["dataset_id"], config.get("split", "train")
    for record in records:
        required = {"sample_id", "hf_index", "image_column", "domain", "dataset_id", "dataset_split"}
        if missing := required - record.keys():
            raise ValueError(f"Original-image metadata missing {sorted(missing)} for {record.get('sample_id')}; "
                             "rebuild the latent bank from the source manifest.")
        if record["dataset_id"] != dataset_id or record["dataset_split"] != split:
            raise ValueError(f"Original-image dataset mismatch for {record['sample_id']}.")
    dataset = load_afhq_dataset(data_config_path)
    label_column = config.get("label_column", "label")
    label_feature = dataset_label_feature(dataset, label_column)
    domains = [str(d).lower() for d in config.get("domains", [])]
    images = []
    for record in records:
        index = int(record["hf_index"])
        if index < 0 or index >= len(dataset):
            raise ValueError(f"Original-image index out of range for {record['sample_id']}: {index}.")
        row = dataset[index]
        actual = record_from_row(dataset_id, split, index, row, label_column,
                                 record["image_column"], label_feature, domains)
        if actual["sample_id"] != record["sample_id"] or actual["domain"] != record["domain"]:
            raise ValueError(f"Original-image identity mismatch for {record['sample_id']}.")
        pixels = image_to_tensor(row[record["image_column"]], int(config.get("image_size", 256)),
                                 bool(config.get("center_crop", True)))
        images.append((pixels + 1) / 2)
    return torch.stack(images)
