from pathlib import Path
import torch

from .batchnorm_matching import batchnorm_checkpoint


def latest_checkpoint(output_dir):
    paths = (path for path in Path(output_dir).glob("step_*.pt")
             if path.is_file() and path.stem[5:].isdigit())
    return max(paths, key=lambda path: int(path.stem[5:]), default=None)


def resume_latest(output_dir, domains, batch_norms, optimizer):
    path = latest_checkpoint(output_dir)
    if path is None:
        return 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for name, context in domains.items():
        norm = batch_norms[name]
        saved = checkpoint["matching_batchnorm"][name]
        if saved["config"] != batchnorm_checkpoint(norm)["config"]:
            raise ValueError(f"Match the {name} BatchNorm settings to {path.name} before resuming.")
        context.branch.load_pdae_state_dict(checkpoint["models"][name])
        norm.load_state_dict(saved["state_dict"])
        norm.train()
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("rng_state") is not None:
        torch.set_rng_state(checkpoint["rng_state"])
    for index, state in enumerate(checkpoint.get("cuda_rng_state_all", [])[:torch.cuda.device_count()]):
        torch.cuda.set_rng_state(state, device=index)
    step = int(checkpoint["step"])
    print(f"Resumed {path.name}; next training step: {step + 1}")
    return step
