import logging
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return rank, world_size, device


def close_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_logging(output_dir, rank):
    handlers = [logging.NullHandler()]
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers = [logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers, force=True)


def wrap_distributed(model, device):
    if not dist.is_initialized():
        return model
    return DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False, find_unused_parameters=True,
    )


@torch.no_grad()
def sync_batchnorm_buffers(batch_norms):
    """Keep local reference normalization; average running estimates for evaluation."""
    if dist.is_initialized():
        for norm in batch_norms.values():
            for buffer in (norm.running_mean, norm.running_var):
                dist.all_reduce(buffer)
                buffer.div_(dist.get_world_size())


def mean_metrics(metrics, device):
    values = torch.tensor(list(metrics.values()), device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(values)
        values.div_(dist.get_world_size())
    return dict(zip(metrics, values.cpu().tolist()))
