import torch


def batchnorm_checkpoint(norm):
    return {
        "config": {key: getattr(norm, key) for key in (
            "num_features", "eps", "momentum", "affine", "track_running_stats",
        )},
        "state_dict": norm.state_dict(),
    }


def load_batchnorm(path, *, domain, step, device, key="matching_batchnorm"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("domain") != domain or checkpoint.get("step") != step:
        raise ValueError("BatchNorm checkpoint domain or training step does not match.")
    
    saved = checkpoint.get(key)
    if key == "mapped_batchnorm":
        enabled = checkpoint.get("mapped_batchnorm_enabled", False)
        if enabled != (saved is not None):
            raise ValueError("Checkpoint mapped BatchNorm setting and statistics do not match.")
        if not enabled:
            return torch.nn.Identity().eval()
        if saved["state_dict"]["num_batches_tracked"].item() == 0:
            raise ValueError("Mapped BatchNorm has not observed any training batches.")
   
    norm = torch.nn.BatchNorm1d(**saved["config"])
   
    norm = norm.to(device=device, dtype=saved["state_dict"]["running_mean"].dtype)
    norm.load_state_dict(saved["state_dict"])
 
    return norm.eval()
