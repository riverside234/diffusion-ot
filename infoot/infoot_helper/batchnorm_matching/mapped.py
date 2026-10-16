import logging

import torch

from .checkpoint import batchnorm_checkpoint


def make_mapped_batchnorms(batch_norms, settings):
    enabled = settings.get("mapped_batchnorm", False)
    if enabled and settings["query_count"] < 2:
        raise ValueError("Mapped BatchNorm training requires query_count >= 2.")
    return torch.nn.ModuleDict({
        name: torch.nn.BatchNorm1d(norm.num_features, eps=norm.eps,
                                  momentum=norm.momentum, affine=False).to(norm.running_mean)
        for name, norm in batch_norms.items() if enabled
    })


def restore_mapped_batchnorms(norms, checkpoint):
    saved = checkpoint.get("mapped_batchnorm", {})
    enabled = checkpoint.get("settings", {}).get("mapped_batchnorm", False)
    if enabled and not saved:
        raise ValueError("Checkpoint is missing enabled mapped BatchNorm statistics.")
    if norms is None:
        if enabled:
            raise ValueError("Pass mapped_batch_norms to restore this checkpoint.")
        return
    if not norms:
        return
    if not saved:
        logging.info("Initialized mapped BatchNorm; checkpoint has no previous mapped statistics.")
        return
    for name, norm in norms.items():
        if saved[name]["config"] != batchnorm_checkpoint(norm)["config"]:
            raise ValueError(f"Match the {name} mapped BatchNorm configuration before resuming.")
        norm.load_state_dict(saved[name]["state_dict"])
        norm.train()
