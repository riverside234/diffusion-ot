import hashlib
from pathlib import Path

import torch


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def bank_metadata(path, bank):
    metadata = file_identity(path)
    checkpoint = bank.get("checkpoint_path")
    if checkpoint is not None:
        metadata["checkpoint_at_fit"] = file_identity(checkpoint)
    return metadata


def save_plan(path, P, h, reg, lam, *, optimization=None, banks=None, projection_scales=None):
    state = {"P": P.detach().cpu(), "feature_space": "raw",
             "h": h, "reg": reg, "lam": lam}
    if optimization is not None:
        state["solver"] = optimization["solver"]
        state["optimization"] = optimization
    if banks is not None:
        state["banks"] = banks
    if projection_scales is not None:
        state["projection_scales"] = projection_scales
    torch.save(state, path)
