"""Explicit numerical device selection; CPU transfers belong at I/O boundaries."""
import torch


def resolve_device(value="cuda", *, check=True):
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("InfoOT supports device: cpu, cuda, or cuda:<index>.")
    if check and device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Select device: cpu (or --device cpu) explicitly.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device {device.index} is unavailable.")
    return device


def move(value, device):
    """Move an artifact's tensors once, without changing precision or metadata."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(item, device) for item in value)
    return value
