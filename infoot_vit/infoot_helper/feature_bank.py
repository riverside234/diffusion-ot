"""Versioned immutable RGB patch banks: tensors and JSON, no pickled models."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def checked_file(directory, entry):
    directory = Path(directory).resolve()
    path = (directory / entry["file"]).resolve()
    if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != entry["sha256"]:
        raise ValueError(f"Missing/changed artifact file: {path}")
    return path


def validate_maps(features, ids, representation, valid_mask=None):
    if features.ndim != 3 or not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("Features must be finite unpooled floating-point [N,P,D].")
    n, p, d = features.shape
    if n < 1 or len(ids) != n or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != n:
        raise ValueError("Every feature map needs one unique, nonempty stable image ID.")
    required = {"grid", "dim", "patch_order", "encoder", "layer", "preprocessing", "normalization"}
    if missing := required - representation.keys():
        raise ValueError(f"Missing feature representation metadata: {sorted(missing)}")
    grid = representation["grid"]
    if (len(grid) != 2 or any(not isinstance(x, int) or x < 1 for x in grid)
            or grid[0] * grid[1] != p or d < 1 or representation["dim"] != d
            or representation["patch_order"] != "row_major_no_special_tokens"
            or representation["normalization"] != "none"):
        raise ValueError("Require a fixed patch grid, no CLS/register tokens, pooling, or extra normalization.")
    if valid_mask is None or valid_mask.dtype != torch.bool or valid_mask.shape != (n, p) or not valid_mask.all():
        raise ValueError("Input banks require an explicit fully valid fixed grid; padded/variable maps are unsupported.")


@dataclass
class FeatureBank:
    features: torch.Tensor
    ids: list[str]
    records: list[dict]
    manifest: dict
    path: Path

    @property
    def representation(self):
        return self.manifest["representation"]

    @property
    def artifact_id(self):
        return self.manifest["artifact_id"]

    @classmethod
    def load(cls, directory):
        directory = Path(directory).resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        identity = {k: v for k, v in manifest.items() if k != "artifact_id"}
        if manifest.get("schema") != "siglip_patch_bank_v1" or digest(identity) != manifest.get("artifact_id"):
            raise ValueError(f"Invalid bank manifest/fingerprint: {directory}")
        features, ids, records = [], [], []
        for entry in manifest["shards"]:
            state = torch.load(checked_file(directory, entry), map_location="cpu", weights_only=True, mmap=True)
            validate_maps(state["features"], state["ids"], manifest["representation"], state["valid_mask"])
            if len(state["ids"]) != entry["count"]:
                raise ValueError("Bank shard count mismatch.")
            features.append(state["features"])
            ids.extend(state["ids"])
            records.extend(state["records"])
        if not features or ids != manifest["ids"] or len(set(ids)) != len(ids):
            raise ValueError("Bank ordering/IDs do not match its manifest.")
        if len(records) != len(ids) or any(r.get("sample_id") != i or r.get("domain") != manifest["domain"]
                                          for r, i in zip(records, ids)):
            raise ValueError("Bank records must match stable IDs and domain.")
        return cls(torch.cat(features), ids, records, manifest, directory)


class BankWriter:
    def __init__(self, directory, *, domain, split, representation, provenance):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = dict(schema="siglip_patch_bank_v1", domain=domain, split=split,
                             representation=representation, provenance=provenance, ids=[], shards=[])

    def append(self, features, records):
        features = features.detach().cpu().contiguous()
        ids = [r["sample_id"] for r in records]
        mask = torch.ones(features.shape[:2], dtype=torch.bool)
        validate_maps(features, ids, self.manifest["representation"], mask)
        if set(ids) & set(self.manifest["ids"]) or any(r["domain"] != self.manifest["domain"] for r in records):
            raise ValueError("Duplicate bank ID or wrong domain.")
        name = f"features_{len(self.manifest['shards']):05d}.pt"
        path = self.directory / name
        save_tensor(path, dict(features=features, valid_mask=mask, ids=ids, records=records))
        self.manifest["ids"].extend(ids)
        self.manifest["shards"].append(dict(file=name, sha256=file_hash(path), count=len(ids)))

    def finish(self):
        if not self.manifest["ids"]:
            raise ValueError("Cannot save an empty feature bank.")
        self.manifest["artifact_id"] = digest(self.manifest)
        write_json(self.directory / "manifest.json", self.manifest)
        return self.manifest


def compatible_banks(source, target, *, fitting=True):
    if source.representation != target.representation:
        raise ValueError("Encoder, layer, preprocessing, normalization, or patch-grid mismatch between banks.")
    if fitting and (source.manifest["split"] != "train" or target.manifest["split"] != "train"):
        raise ValueError("Fitted supports must both use the train split; never validation/test.")
    if fitting and (source.manifest["domain"] == target.manifest["domain"] or set(source.ids) & set(target.ids)):
        raise ValueError("Fit requires distinct domains and disjoint stable image IDs.")
