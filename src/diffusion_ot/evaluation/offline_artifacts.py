"""Shared artifact integrity and atomic writes for offline stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def read_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def bind_run(output, identity, *, resume):
    """Never mix outputs produced under different scientific protocols."""
    output = Path(output)
    marker = output / "run.json"
    if marker.exists():
        if not resume:
            raise ValueError(f"{output} already exists; use --resume or a new --output-dir")
        if json.loads(marker.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Output protocol mismatch; choose a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to reuse nonempty unrecognized output directory: {output}")
    else:
        atomic_json(marker, {"identity": identity})


def verify_files(root, hashes):
    for relative, expected in hashes.items():
        path = (Path(root) / relative).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("Artifact path escapes its root")
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Artifact missing or changed: {path}")
