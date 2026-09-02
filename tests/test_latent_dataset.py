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


def test_cached_latent_dataset_can_flip_latent_horizontally(tmp_path):
    from diffusion_ot.data.latent_dataset import CachedLatentDataset

    config_path = tmp_path / "afhq_test.yaml"
    manifest_dir = tmp_path / "manifests"
    latent_dir = tmp_path / "latents"
    split_dir = latent_dir / "cat_train"
    manifest_dir.mkdir()
    split_dir.mkdir(parents=True)

    sample_id = "cat_000000"
    latent = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    latent_path = split_dir / f"{sample_id}.pt"
    torch.save({"x0_latent": latent}, latent_path)
    (manifest_dir / "cat_train.jsonl").write_text(
        json.dumps({"sample_id": sample_id, "domain": "cat", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        "manifest_dir: manifests\nlatent_dir: latents\nlatent_manifest_name: latent_manifest.jsonl\n",
        encoding="utf-8",
    )

    dataset = CachedLatentDataset(
        data_config_path=config_path,
        domain="cat",
        split="train",
        project_root=tmp_path,
        random_horizontal_flip=1.0,
    )

    torch.testing.assert_close(dataset[0]["x0_latent"], torch.flip(latent, dims=(-1,)))


def test_latent_cache_selects_configured_posterior_statistic():
    from diffusion_ot.data.latent_cache import select_posterior_latent

    class FakePosterior:
        def __init__(self):
            self.mean = torch.tensor([1.0])
            self.sample_value = torch.tensor([2.0])
            self.received_generator = None

        def sample(self, generator=None):
            self.received_generator = generator
            return self.sample_value

    posterior = FakePosterior()
    marker = object()

    torch.testing.assert_close(
        select_posterior_latent(posterior, use_posterior_mean=True),
        posterior.mean,
    )
    torch.testing.assert_close(
        select_posterior_latent(
            posterior,
            use_posterior_mean=False,
            generator=marker,
        ),
        posterior.sample_value,
    )
    assert posterior.received_generator is marker
