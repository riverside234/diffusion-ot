import logging

import torch
import torch.distributed as dist

from diffusion_ot.training.train_joint_infoot import _save_checkpoint
from ..batchnorm_matching import batchnorm_checkpoint
from ..infoot_cotraining_helper import save_domain_checkpoints


def restore_random_state(checkpoint, device=None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    states = checkpoint.get("rank_rng_states")
    if states is not None and len(states) == world_size:
        torch.set_rng_state(states[rank]["cpu"])
        if device.type == "cuda" and states[rank]["cuda"] is not None:
            torch.cuda.set_rng_state(states[rank]["cuda"], device=device)
    elif states is None and world_size == 1:
        if checkpoint.get("rng_state") is not None:
            torch.set_rng_state(checkpoint["rng_state"])
        for index, state in enumerate(checkpoint.get("cuda_rng_state_all", [])[:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(state, device=index)
    else:
        torch.manual_seed(42 + int(checkpoint["step"]) * world_size + rank)
        logging.info("GPU count changed or per-rank RNG was absent; reseeded training streams.")


def save_checkpoint(output_dir, step, domains, batch_norms, optimizer, settings, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    state = {"cpu": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None}
    states = [None] * world_size if rank == 0 else None
    if dist.is_initialized():
        dist.gather_object(state, states, dst=0)
    else:
        states = [state]
    if rank == 0:
        save_domain_checkpoints(domains, output_dir, step, batch_norms)
        _save_checkpoint(output_dir / f"step_{step:06d}.pt", {
            "step": step, "settings": settings, "world_size": world_size,
            "models": {name: context.branch.pdae_state_dict() for name, context in domains.items()},
            "training_configs": {name: context.training_config for name, context in domains.items()},
            "optimizer": optimizer.state_dict(), "rank_rng_states": states,
            "matching_batchnorm": {name: batchnorm_checkpoint(norm) for name, norm in batch_norms.items()},
            "batchnorm_policy": "local_references_averaged_running_statistics",
        })
        logging.info("Saved checkpoint at step %s", step)
    if dist.is_initialized():
        dist.barrier()
