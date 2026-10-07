import torch


def batchnorm_checkpoint(norm):
    return {
        "config": {key: getattr(norm, key) for key in (
            "num_features", "eps", "momentum", "affine", "track_running_stats",
        )},
        "state_dict": norm.state_dict(),
    }


def load_batchnorm(path, *, domain, step, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("domain") != domain or checkpoint.get("step") != step:
        raise ValueError("BatchNorm checkpoint domain or training step does not match.")
    saved = checkpoint.get("matching_batchnorm")
    if saved is None:
        raise ValueError(f"{path} has no matching BatchNorm; use a checkpoint from BatchNorm co-training.")
    norm = torch.nn.BatchNorm1d(**saved["config"])
    if not norm.track_running_stats:
        raise ValueError("Matching BatchNorm must have saved running statistics.")
    norm = norm.to(device=device, dtype=saved["state_dict"]["running_mean"].dtype)
    norm.load_state_dict(saved["state_dict"])
    if norm.num_batches_tracked < 1:
        raise ValueError("Matching BatchNorm has not observed training references.")
    return norm.eval()
