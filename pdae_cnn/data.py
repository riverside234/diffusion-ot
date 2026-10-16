"""Cached-latent data loaders and batch preparation for CNN PDAE training."""

import torch
from torch.utils.data import DataLoader

from diffusion_ot.data.latent_dataset import CachedLatentDataset, collate_latent_batch


def build_loaders(config):
    """Return (train_loader, validation_loader); disabled validation returns None."""
    data = config.get("dataloader") or {}
    train = config.get("train") or {}
    validation = config.get("evaluation") or {}
    seed = int(train.get("seed", 20260901))

    def make_loader(*, training):
        options = train if training else validation
        loader_options = data if training else {**data, **validation}
        split = config.get("split", "train") if training else validation.get("split", "val")
        limit = options.get("limit")
        dataset = CachedLatentDataset(
            data_config_path=config["data_config"], domain=config["domain"],
            split=split, project_root=config["project_root"],
            limit=int(limit) if limit is not None else None,
            random_horizontal_flip=float(data.get("random_horizontal_flip", 0.0)) if training else 0.0,
        )
        generator = torch.Generator().manual_seed(
            seed if training else int(validation.get("seed", seed + 1))
        )
        loader = DataLoader(
            dataset, batch_size=int(loader_options.get("batch_size", 16)),
            shuffle=training, drop_last=bool(data.get("drop_last", True)) if training else False,
            num_workers=int(loader_options.get("num_workers", 4)),
            pin_memory=bool(data.get("pin_memory", True)),
            collate_fn=collate_latent_batch, generator=generator,
        )
        if not len(loader):
            raise ValueError(f"No batches for {split}; check dataset size, batch_size, and drop_last.")
        return loader

    train_loader = make_loader(training=True)
    validation_loader = make_loader(training=False) if validation.get("enabled", True) else None
    return train_loader, validation_loader


def prepare_batch(batch, *, device, dtype):
    """Move cached latents to the model's device and dtype, preserving metadata."""
    return {
        **batch,
        "x0_latent": batch["x0_latent"].to(device=device, dtype=dtype, non_blocking=True),
    }


def cycle_batches(loader):
    """Start a fresh loader iterator each epoch without caching batches."""
    if not len(loader):
        raise ValueError("Training loader produced no batches.")
    while True:
        yield from loader
