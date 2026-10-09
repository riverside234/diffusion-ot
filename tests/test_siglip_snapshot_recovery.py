"""Recover repository-only SigLIP metadata without replacing trained encoder bytes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from diffusion_ot.models.pdae_v2.encoder import (
    MANIFEST_NAME, MODEL_ID, file_sha256, restore_snapshot_manifest, snapshot_identity,
)


@pytest.fixture
def snapshot(tmp_path):
    directory = tmp_path / "siglip"
    directory.mkdir()
    content = {"config.json": b'{"model_type":"siglip"}',
               "preprocessor_config.json": b'{}', "model.safetensors": b'fixture weights'}
    for name, value in content.items():
        (directory / name).write_bytes(value)
    identity = dict(model_id=MODEL_ID, revision="a" * 40,
                    files={name: file_sha256(directory / name) for name in content})
    checkpoint = tmp_path / "latest.pt"
    torch.save({"model": {"frozen_encoder": identity}}, checkpoint)
    return directory, identity, content, checkpoint


def downloader():
    path = Path(__file__).resolve().parents[1] / "scripts/download_siglip2.py"
    spec = importlib.util.spec_from_file_location("siglip_recovery_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_recovery_is_offline_when_weights_already_exist(snapshot, monkeypatch):
    import huggingface_hub
    directory, identity, content, checkpoint = snapshot
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: pytest.fail("No download needed"))
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda *a, **k: pytest.fail("Do not resolve main"))
    assert downloader().recover_from_checkpoint(directory, checkpoint) == identity
    assert snapshot_identity(directory) == identity
    for name, value in content.items():
        assert (directory / name).read_bytes() == value
    # It is safe to run the same repair twice.
    assert downloader().recover_from_checkpoint(directory, checkpoint) == identity


@pytest.mark.parametrize("damage", ["missing", "changed", "outside", "wrong_model", "no_identity"])
def test_manifest_not_written_when_checkpoint_cannot_verify_local_files(snapshot, damage):
    directory, identity, _, _ = snapshot
    if damage == "missing":
        (directory / "model.safetensors").unlink()
    elif damage == "changed":
        (directory / "preprocessor_config.json").write_bytes(b'changed preprocessing')
    elif damage == "outside":
        identity["files"]["../outside"] = "not-a-hash"
    elif damage == "wrong_model":
        identity["model_id"] = "different/model"
    else:
        identity = None
    with pytest.raises((ValueError, FileNotFoundError)):
        restore_snapshot_manifest(directory, identity)
    assert not (directory / MANIFEST_NAME).exists()


@pytest.mark.parametrize("bad_download", [False, True])
def test_missing_weights_use_checkpoint_revision_and_verify_before_manifest(snapshot, monkeypatch, bad_download):
    import huggingface_hub
    directory, identity, content, checkpoint = snapshot
    (directory / "model.safetensors").unlink()
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        for name, value in content.items():
            (directory / name).write_bytes(value)
        if bad_download:
            (directory / "model.safetensors").write_bytes(b'different encoder')

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda *a, **k: pytest.fail("Do not resolve main"))
    monkeypatch.setenv("HF_TOKEN", "test-token")
    if bad_download:
        with pytest.raises(ValueError, match="file changed"):
            downloader().recover_from_checkpoint(directory, checkpoint)
        assert not (directory / MANIFEST_NAME).exists()
    else:
        assert downloader().recover_from_checkpoint(directory, checkpoint) == identity
        assert snapshot_identity(directory) == identity
        assert "test-token" not in (directory / MANIFEST_NAME).read_text()
    assert len(calls) == 1 and calls[0]["revision"] == identity["revision"]
    assert calls[0]["allow_patterns"] == list(identity["files"])
    assert calls[0]["token"] == "test-token"


def test_recovery_does_not_replace_existing_different_identity(snapshot, monkeypatch):
    import huggingface_hub
    directory, identity, _, checkpoint = snapshot
    other = deepcopy(identity)
    other["revision"] = "b" * 40
    restore_snapshot_manifest(directory, other)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: pytest.fail("Do not switch encoder"))
    with pytest.raises(ValueError, match="manifest differs"):
        downloader().recover_from_checkpoint(directory, checkpoint)
    assert snapshot_identity(directory) == other
