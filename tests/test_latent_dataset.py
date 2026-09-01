from __future__ import annotations

import json

import pytest


torch = pytest.importorskip("torch")


def test_cached_latent_dataset_loads_split_manifest(tmp_path):
    from diffusion_ot.data.latent_dataset import CachedLatentDataset, collate_latent_batch

    config_path = tmp_path / "afhq_test.yaml"
    manifest_dir = tmp_path / "manifests"
    latent_dir = tmp_path / "latents"
    split_dir = latent_dir / "cat_train"
    manifest_dir.mkdir()
    split_dir.mkdir(parents=True)

    sample_id = "cat_000000"
    latent_path = split_dir / f"{sample_id}.pt"
    torch.save({"x0_latent": torch.zeros(4, 32, 32)}, latent_path)
    (manifest_dir / "cat_train.jsonl").write_text(
        json.dumps({"sample_id": sample_id, "domain": "cat", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                "manifest_dir: manifests",
                "latent_dir: latents",
                "latent_manifest_name: latent_manifest.jsonl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = CachedLatentDataset(
        data_config_path=config_path,
        domain="cat",
        split="train",
        project_root=tmp_path,
    )
    item = dataset[0]
    batch = collate_latent_batch([item, item])

    assert len(dataset) == 1
    assert item["sample_id"] == sample_id
    assert item["x0_latent"].shape == (4, 32, 32)
    assert batch["x0_latent"].shape == (2, 4, 32, 32)
