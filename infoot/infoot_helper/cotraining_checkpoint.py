from pathlib import Path
import logging
import torch
import torch.distributed as dist

from .batchnorm_matching import batchnorm_checkpoint
from .batchnorm_matching.mapped import restore_mapped_batchnorms
from .distributed.checkpoint import restore_random_state


def latest_checkpoint(output_dir):
    paths = (path for path in Path(output_dir).glob("step_*.pt")
             if path.is_file() and path.stem[5:].isdigit())
    return max(paths, key=lambda path: int(path.stem[5:]), default=None)


def resume_latest(output_dir, domains, batch_norms, optimizer, device=None, mapped_batch_norms=None):
    path = latest_checkpoint(output_dir)
    if dist.is_initialized():
        chosen = [str(path) if path is not None and dist.get_rank() == 0 else None]
        dist.broadcast_object_list(chosen, src=0)
        path = Path(chosen[0]) if chosen[0] is not None else None
    if path is None:
        return 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restore_mapped_batchnorms(mapped_batch_norms, checkpoint)
    for name, context in domains.items():
        norm = batch_norms[name]
        saved = checkpoint["matching_batchnorm"][name]
        if saved["config"] != batchnorm_checkpoint(norm)["config"]:
            raise ValueError(f"Match the {name} BatchNorm settings to {path.name} before resuming.")
        context.branch.load_pdae_state_dict(checkpoint["models"][name])
        norm.load_state_dict(saved["state_dict"])
        norm.train()
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_random_state(checkpoint, device)
    step = int(checkpoint["step"])
    logging.info("Resumed %s; next training step: %s", path.name, step + 1)
    return step
