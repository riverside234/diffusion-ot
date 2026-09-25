from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from diffusion_ot.data import ground_truth
from diffusion_ot.data.latent_cache import image_to_tensor
from diffusion_ot.data.latent_dataset import CachedLatentDataset, collate_latent_batch


@pytest.fixture
def rgb_data(tmp_path, monkeypatch):
    pixels = np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3)
    source = [
        {"image": Image.fromarray(pixels), "label": "cat"},
        {"image": Image.fromarray(np.flip(pixels, axis=0)), "label": "cat"},
    ]
    records = []
    latents = []
    latent_dir = tmp_path / "latents" / "cat_train"
    latent_dir.mkdir(parents=True)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    for index in range(2):
        sample_id = f"afhq_cat_{index:06d}"
        records.append({"sample_id": sample_id, "domain": "cat", "hf_index": index,
                        "dataset_id": "test/AFHQ", "dataset_split": "train",
                        "image_column": "image"})
        latent = torch.arange(4 * 32 * 32, dtype=torch.float32).reshape(4, 32, 32) + index
        latents.append(latent)
        torch.save(latent, latent_dir / f"{sample_id}.pt")
    manifest = manifest_dir / "cat_train.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump({
        "dataset_id": "test/AFHQ", "split": "train", "image_column": "image",
        "image_size": 8, "center_crop": True,
        "manifest_dir": "manifests", "latent_dir": "latents",
    }), encoding="utf-8")
    loads = []

    def load(path):
        loads.append(path)
        return source

    monkeypatch.setattr(ground_truth, "load_afhq_dataset", load)
    return config, source, records, latents, loads, manifest


def make_dataset(rgb_data, **kwargs):
    config, *_ = rgb_data
    return CachedLatentDataset(config, "cat", project_root=config.parent,
                               include_original_images=True, **kwargs)


@pytest.mark.parametrize("center_crop", [True, False])
def test_original_rgb_exactly_matches_latent_cache_preprocessing(rgb_data, center_crop):
    config, source, _, latents, _, _ = rgb_data
    values = yaml.safe_load(config.read_text())
    values["center_crop"] = center_crop
    config.write_text(yaml.safe_dump(values))
    dataset = make_dataset(rgb_data)
    item = dataset[1]
    expected = image_to_tensor(source[1]["image"], 8, center_crop)
    assert item["encoder_image"].shape == (3, 8, 8)
    assert item["encoder_image"].dtype == torch.float32
    torch.testing.assert_close(item["encoder_image"], expected, rtol=0, atol=0)
    torch.testing.assert_close(item["x0_latent"], latents[1])
    assert -1 <= item["encoder_image"].min() <= item["encoder_image"].max() <= 1


def test_source_dataset_is_lazy_reused_and_config_not_reparsed_per_sample(rgb_data, monkeypatch):
    config, _, _, _, loads, _ = rgb_data
    config_reads = []
    original_load = ground_truth.load_yaml_config

    def load_config(path):
        config_reads.append(path)
        return original_load(path)

    monkeypatch.setattr(ground_truth, "load_yaml_config", load_config)
    dataset = make_dataset(rgb_data)
    assert loads == []
    for index in [0, 1, 0]:
        dataset[index]
    assert loads == [config]
    assert config_reads == [config]


@pytest.mark.parametrize("flip", [False, True])
def test_one_random_flip_is_shared_by_rgb_and_latent(rgb_data, monkeypatch, flip):
    _, source, _, latents, _, _ = rgb_data
    dataset = make_dataset(rgb_data, random_horizontal_flip=0.5)
    draws = []

    def random_value(*args, **kwargs):
        draws.append(True)
        return torch.tensor(0.25 if flip else 0.75)

    monkeypatch.setattr(torch, "rand", random_value)
    item = dataset[0]
    expected_image = image_to_tensor(source[0]["image"], 8, True)
    expected_latent = latents[0]
    if flip:
        expected_image = expected_image.flip(-1)
        expected_latent = expected_latent.flip(-1)
    assert item["horizontal_flip"] is flip
    assert len(draws) == 1
    torch.testing.assert_close(item["encoder_image"], expected_image)
    torch.testing.assert_close(item["x0_latent"], expected_latent)


def test_collation_stacks_rgb_and_legacy_default_does_not_load_source(rgb_data):
    config, _, _, _, loads, _ = rgb_data
    legacy = CachedLatentDataset(config, "cat", project_root=config.parent)
    legacy_item = legacy[0]
    assert "encoder_image" not in legacy_item
    assert "encoder_image" not in collate_latent_batch([legacy_item])
    assert loads == []
    dataset = make_dataset(rgb_data)
    items = [dataset[1], dataset[0]]
    batch = collate_latent_batch(items)
    assert batch["encoder_image"].shape == (2, 3, 8, 8)
    torch.testing.assert_close(batch["encoder_image"], torch.stack([x["encoder_image"] for x in items]))
    with pytest.raises(ValueError, match="with and without encoder_image"):
        collate_latent_batch([items[0], legacy_item])


@pytest.mark.parametrize("change,missing,pattern", [
    ({}, "hf_index", "metadata missing"),
    ({"dataset_id": "wrong"}, None, "dataset mismatch"),
    ({"dataset_split": "test"}, None, "dataset mismatch"),
    ({"image_column": "another_image"}, None, "column mismatch"),
])
def test_missing_or_mismatched_metadata_fails_before_source_is_opened(rgb_data, change, missing, pattern):
    _, _, records, _, loads, manifest = rgb_data
    records[0].update(change)
    if missing:
        records[0].pop(missing)
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match=pattern):
        make_dataset(rgb_data)
    assert loads == []


@pytest.mark.parametrize("change,pattern", [
    ({"hf_index": -1}, "index out of range"),
    ({"hf_index": 2}, "index out of range"),
    ({"hf_index": 1}, "identity mismatch"),
])
def test_wrong_index_or_sample_identity_never_silently_loads_wrong_rgb(rgb_data, change, pattern):
    dataset = make_dataset(rgb_data)
    dataset.records[0].update(change)
    with pytest.raises(ValueError, match=pattern):
        dataset[0]


def test_changed_source_domain_is_rejected(rgb_data):
    _, source, *_ = rgb_data
    source[0]["label"] = "dog"
    dataset = make_dataset(rgb_data)
    with pytest.raises(ValueError, match="identity mismatch"):
        dataset[0]


def test_missing_source_column_has_sample_context(rgb_data):
    _, source, *_ = rgb_data
    del source[0]["image"]
    dataset = make_dataset(rgb_data)
    with pytest.raises(ValueError, match="source columns missing.*afhq_cat_000000"):
        dataset[0]


def test_original_reader_reopens_for_fork_and_spawn_workers(rgb_data):
    _, _, _, _, loads, _ = rgb_data
    dataset = make_dataset(rgb_data)
    expected = dataset[0]["encoder_image"]
    reader = dataset._original_image_loader
    reader._dataset_pid = -1  # A different process cannot reuse this handle.
    torch.testing.assert_close(dataset[0]["encoder_image"], expected)
    assert len(loads) == 2
    reader._dataset = lambda: None  # Deliberately unpicklable: must be discarded.
    spawned = pickle.loads(pickle.dumps(dataset))
    assert spawned._original_image_loader._dataset is None
    torch.testing.assert_close(spawned[0]["encoder_image"], expected)
    assert len(loads) == 3
