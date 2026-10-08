"""SiT-L configuration, Hub download, and existing VAE-cache integration (offline)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from diffusion_ot.integrations.hf_snapshot import load_yaml_config
from diffusion_ot.models.pdae_v2.branch import TokenConditionedSiT
from test_pdae_v2 import NativeSiTBlock


ROOT = Path(__file__).resolve().parents[1]


def test_l_config_uses_all_24_layers_and_native_siglip_width():
    for domain in ("cat", "dog"):
        stage = load_yaml_config(ROOT / f"configs/stage1a_pdae_v2_l/{domain}.yaml")
        model = load_yaml_config(ROOT / stage["model_config"])
        data = load_yaml_config(ROOT / stage["data_config"])
        pretrained = load_yaml_config(ROOT / model["pretrained"])
        assert stage["domain"] == domain
        assert stage["train"]["initialize_from"] is stage["train"]["resume_from"] is None
        assert stage["output_dir"] == f"outputs/pdae_v2_l_{domain}"
        assert stage["dataloader"]["batch_size"] * stage["dataloader"]["gradient_accumulation_steps"] == 64
        assert stage["evaluation"]["batch_size"] * stage["evaluation"]["num_batches"] == 32
        assert model["hidden_size"] == 1024 and model["depth"] == 24
        assert model["num_heads"] == stage["adapter"]["cross_attention_heads"] == 16
        assert stage["encoder"]["token_dim"] == 768
        assert data["latent_dir"] == "data/latents/afhq_sit_l2_256"
        assert data["manifest_dir"] == "data/manifests"
        assert pretrained["subfolder"] == "SiT-L-2-256"
        assert len(pretrained["revision"]) == 40 and not pretrained["download_if_missing"]

        # Instantiate all L-sized new layers without allocating the full pretrained model.
        with torch.device("meta"):
            base = nn.Module()
            base.config = SimpleNamespace(**model["expected_transformer_config"])
            base.blocks = nn.ModuleList([NativeSiTBlock(1024) for _ in range(24)])
            wrapper = TokenConditionedSiT(base, num_heads=16,
                                         lora_layers=stage["adapter"]["lora_layers"])
        assert len(wrapper.image_attention) == 24
        assert wrapper.lora_layers == list(range(24))
        assert wrapper.image_attention[-1].q.weight.shape == (1024, 1024)
        assert wrapper.image_attention[-1].k.weight.shape == (1024, 768)
        assert wrapper.image_attention[-1].v.weight.shape == (1024, 768)
        assert wrapper._attention_lora_modules[23]["qkv"].lora_down.weight.shape == (64, 1024)
        assert wrapper.feature_projector[-1].weight.shape == (768, 768)


def test_l_download_fetches_diffusion_and_vae_once(tmp_path, monkeypatch, capsys):
    import huggingface_hub
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import download_sit_l
    import verify_sit_snapshot

    config = load_yaml_config(ROOT / "configs/pretrained/bilisakura_sit_l2_256.yaml")
    config["project_root"] = str(tmp_path)
    path = tmp_path / "pretrained.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = []
    sentinel = tmp_path / "artifacts/pretrained/SiT-B-2-256/keep.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("existing experiment")

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        directory = Path(kwargs["local_dir"]) / config["subfolder"]
        for relative in config["expected_files"]:
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"offline Hub fixture")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setenv("HF_TOKEN", "test-token-not-for-logs")
    assert download_sit_l.main(["--config", str(path)]) == 0
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "BiliSakura/SiT-diffusers"
    assert calls[0]["revision"] == config["revision"]
    assert calls[0]["allow_patterns"] == ["SiT-L-2-256/**"]
    assert "**/__pycache__/**" in calls[0]["ignore_patterns"]
    report_path = tmp_path / config["metadata_file"]
    report = json.loads(report_path.read_text())
    assert report["ok"] and report["downloaded"] and report["revision"] == config["revision"]
    assert download_sit_l.main(["--config", str(path)]) == 0
    assert len(calls) == 1 and sentinel.read_text() == "existing experiment"
    assert "test-token-not-for-logs" not in capsys.readouterr().out + report_path.read_text()
    # The old verifier remains read-only by default and still selects B.
    old = verify_sit_snapshot.parse_args([])
    assert not old.download_if_missing and old.config.endswith("bilisakura_sit_b2_256.yaml")


def test_l_cache_launcher_encodes_all_existing_splits_with_bundled_vae(tmp_path, monkeypatch):
    from PIL import Image
    import diffusion_ot.data.latent_cache as cache
    from diffusion_ot.data.latent_dataset import CachedLatentDataset
    from diffusion_ot.data.manifests import read_jsonl, write_jsonl
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import cache_pdae_v2_l_latents
    import cache_vae_latents as cli

    data = load_yaml_config(ROOT / "configs/data/afhq_pdae_v2_l.yaml")
    model = load_yaml_config(ROOT / "configs/model/sit_l2_256.yaml")
    data["project_root"] = model["project_root"] = str(tmp_path)
    data_path, model_path = tmp_path / "data.yaml", tmp_path / "model.yaml"
    data_path.write_text(yaml.safe_dump(data))
    model_path.write_text(yaml.safe_dump(model))
    rows = []
    for domain in data["active_domains"]:
        for split in ("train", "val"):
            record = dict(sample_id=f"{domain}_{split}", domain=domain, split=split,
                          hf_index=len(rows), image_column="image")
            rows.append({"image": Image.new("RGB", (300, 256), (96, 128, 180))})
            write_jsonl(tmp_path / "data/manifests" / f"{domain}_{split}.jsonl", [record])
    monkeypatch.setattr(cache, "load_afhq_dataset", lambda path: rows)
    received = []

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(scaling_factor=0.18215)

        def encode(self, pixels):
            assert not torch.is_grad_enabled() and not self.training
            assert pixels.shape[1:] == (3, 256, 256) and pixels.dtype == torch.float32
            assert pixels.min() >= -1 and pixels.max() <= 1
            # A posterior without sample() also proves the mean is selected.
            return SimpleNamespace(latent_dist=SimpleNamespace(mean=torch.ones(len(pixels), 4, 32, 32)))

    def load_vae(pretrained, root, **kwargs):
        received.append(pretrained)
        return VAE()

    monkeypatch.setattr(cache, "load_sit_vae", load_vae)
    assert cache_pdae_v2_l_latents.main([
        "--data-config", str(data_path), "--model-config", str(model_path),
        "--device", "cpu", "--batch-size", "2",
    ]) == 0
    assert received == [tmp_path / model["pretrained"]]
    latent_root = tmp_path / data["latent_dir"]
    records = read_jsonl(latent_root / data["latent_manifest_name"])
    assert len(records) == 4
    for record in records:
        latent = torch.load(record["latent_path"], weights_only=True)
        assert latent.shape == (4, 32, 32) and latent.dtype == torch.float32
        torch.testing.assert_close(latent, torch.full_like(latent, 0.18215))
        assert record["posterior_statistic"] == "mean"
        assert Path(record["latent_path"]).is_relative_to(latent_root)
        # Exercise the training/eval dataset reader against the real saved manifest.
        dataset = CachedLatentDataset(data_path, domain=record["domain"], split=record["split"],
                                      project_root=tmp_path, validate_exists=True)
        torch.testing.assert_close(dataset[0]["x0_latent"], latent)
    assert not (tmp_path / "data/latents/afhq_sit_b2_256").exists()
    assert cli.parse_args([]).model_config.endswith("sit_b2_256.yaml")
