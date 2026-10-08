from itertools import islice

import torch
from torch.utils.data import DataLoader, DistributedSampler


def make_loader(records, batch_size, rank=0, world_size=1, workers=2, seed=42):
    sampler = DistributedSampler(records, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(
        records, batch_size=batch_size, sampler=sampler, drop_last=True, collate_fn=list,
        num_workers=workers, multiprocessing_context="spawn" if workers else None,
        persistent_workers=bool(workers), generator=torch.Generator().manual_seed(seed + rank),
    )
    if not len(loader):
        raise ValueError("batch_size exceeds the available training samples per GPU.")
    return loader


def cycle_batches(loader, start_step=0):
    epoch, offset = divmod(start_step, len(loader))
    while True:
        loader.sampler.set_epoch(epoch)
        yield from islice(loader, offset, None)
        epoch, offset = epoch + 1, 0
