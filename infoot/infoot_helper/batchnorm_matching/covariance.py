import torch
from torch.nn.functional import mse_loss


def covariance_loss(v):
    """||Cov(v)-I||_F^2 / D, including diagonal and off-diagonal entries.

    Population covariance uses 1/N. Rank <= min(N-1, D), so the loss cannot
    reach zero when N <= D. Call separately per domain, then average.
    """
    if v.ndim != 2 or len(v) < 2 or not v.shape[1]:
        raise ValueError("Covariance needs at least two nonempty reference feature rows.")
    v = v if v.dtype == torch.float64 else v.float()
    dimension = v.shape[1]
    with torch.autocast(device_type=v.device.type, enabled=False):
        covariance = torch.cov(v.T, correction=0).reshape(dimension, dimension)
        identity = torch.eye(dimension, device=v.device, dtype=v.dtype)
        return mse_loss(covariance, identity, reduction="sum") / dimension
