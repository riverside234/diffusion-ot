"""Pin the official FixRes SigLIP 2 snapshot for fully offline PDAE v2 runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_ot.integrations.hf_snapshot import resolve_project_local_path
from diffusion_ot.models.pdae_v2.encoder import (
    MANIFEST_NAME, MODEL_ID, file_sha256, snapshot_identity, restore_snapshot_manifest,
)


def recover_from_checkpoint(output_dir: Path, checkpoint_path: Path) -> dict:
    """Reuse matching local files, or ask HF Hub for the exact saved revision."""
    from diffusion_ot.evaluation.stage1a_eval import _load_checkpoint

    identity = _load_checkpoint(checkpoint_path)["model"].get("frozen_encoder")
    if not isinstance(identity, dict) or identity.get("model_id") != MODEL_ID or not identity.get("revision"):
        raise ValueError("The checkpoint does not contain a PDAE v2 SigLIP frozen_encoder identity.")
    # Do not silently switch an already registered encoder to another version.
    target = output_dir / MANIFEST_NAME
    if target.is_file() and json.loads(target.read_text(encoding="utf-8")) != identity:
        raise ValueError("Local SigLIP manifest differs from the checkpoint's frozen_encoder.")
    try:
        return restore_snapshot_manifest(output_dir, identity)
    except (ValueError, FileNotFoundError):
        # HF Hub reuses its cache. Never resolve a moving 'main' for evaluation.
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=MODEL_ID, revision=identity["revision"],
                          token=os.environ.get("HF_TOKEN") or None, local_dir=str(output_dir),
                          allow_patterns=list(identity.get("files", {})))
        return restore_snapshot_manifest(output_dir, identity)


def download(output_dir: Path, revision: str = "main") -> dict:
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    # Resolve a moving revision before downloading; record only the immutable SHA.
    commit = HfApi().model_info(MODEL_ID, revision=revision, token=token).sha
    snapshot_download(
        repo_id=MODEL_ID, revision=commit, token=token, local_dir=str(output_dir),
        allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors", "*.safetensors.index.json"],
    )
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "siglip":
        raise ValueError("Expected official FixRes model_type=siglip, not NaFlex.")
    files = [output_dir / "config.json", output_dir / "preprocessor_config.json"]
    files += sorted(output_dir.glob("*.safetensors")) + sorted(output_dir.glob("*.safetensors.index.json"))
    if not any(path.suffix == ".safetensors" for path in files):
        raise FileNotFoundError("The downloaded snapshot contains no safetensors weights.")
    manifest = {"model_id": MODEL_ID, "revision": commit,
                "files": {path.name: file_sha256(path) for path in files}}
    target = output_dir / MANIFEST_NAME
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return snapshot_identity(output_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/siglip2_base_patch16_224")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--revision", default=None, help="Hub revision to resolve and pin; defaults to main for a new download.")
    source.add_argument("--checkpoint", help="Recover missing metadata/files using this trained checkpoint's exact SigLIP identity.")
    args = parser.parse_args(argv)
    directory = resolve_project_local_path(args.output_dir, ROOT, field_name="output_dir")
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser()
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        manifest = recover_from_checkpoint(directory, checkpoint)
    else:
        manifest = download(directory, args.revision or "main")
    print(f"Verified {manifest['model_id']} @ {manifest['revision']} at {directory}")
    print("Training and evaluation will load this snapshot locally without a network connection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
