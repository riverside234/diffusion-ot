import math

import torch


DOMAINS = ("cat", "dog")


class ReferenceRMSEMA(torch.nn.Module):
    """Detached EMA of raw reference variance; initialize from the first batch."""

    def __init__(self, decay=0.99, eps=1e-8):
        super().__init__()
        if not 0 <= decay < 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("Reference RMS requires 0 <= decay < 1 and finite eps > 0.")
        self.register_buffer("variance", torch.zeros(len(DOMAINS), dtype=torch.float64))
        self.register_buffer("num_updates", torch.tensor(0))
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float64))
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float64))

    @torch.no_grad()
    def update(self, references, *, step):
        if step != int(self.num_updates) + 1 or set(references) != set(DOMAINS):
            raise ValueError("Update cat/dog reference RMS exactly once per training step.")
        variances = []
        for name in DOMAINS:
            v = references[name].detach().double()
            if v.ndim != 2 or len(v) < 2 or not v.shape[1]:
                raise ValueError("Reference RMS needs at least two nonempty feature rows.")
            variances.append(v.var(dim=0, unbiased=False).sum().to(self.variance))
        batch = torch.stack(variances)
        if not torch.isfinite(batch).all():
            raise ValueError("Reference RMS needs finite reference variances.")
        weight = 1.0 if step == 1 else 1.0 - float(self.decay)
        self.variance.lerp_(batch, weight)
        self.num_updates.add_(1)
        return self.scales()

    def scales(self):
        if self.num_updates < 1 or not torch.isfinite(self.variance).all() or (self.variance < 0).any():
            raise ValueError("Reference RMS needs initialized, finite, nonnegative variances.")
        values = self.variance.clamp_min(self.eps.square()).sqrt().tolist()
        return dict(zip(DOMAINS, values))
