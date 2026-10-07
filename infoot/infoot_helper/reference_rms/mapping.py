import math

import torch

from ..infoot import projection, ratio


def rms_conditional_mapping(v_query, v_source, v_target, P, h, scales):
    source_scale, target_scale = map(float, scales)
    if any(not math.isfinite(x) or x <= 0 for x in (h, source_scale, target_scale)):
        raise ValueError("Projection bandwidth and RMS scales must be finite and positive.")
    kernels = []
    for x, y, scale in (
        (v_query, v_source, source_scale), (v_target, v_target, target_scale),
    ):
        distances = torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -0.5 * (distances / max(h * scale, 1e-8)).square()
        kernels.append(logits.softmax(dim=1))
    scores = ratio(P.detach(), *kernels)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Conditional density ratios are non-finite.")
    return projection(scores, v_target)
