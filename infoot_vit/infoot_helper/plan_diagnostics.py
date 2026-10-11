"""Descriptive transport concentration; these metrics never alter a plan."""
import math

import torch


def patch_plan_concentration(plan):
    """Retained-mass-weighted row statistics, ignoring zero-mass rows.

    Accepts a validated nonnegative [..., source_patch, target_patch] plan.
    Returns tensors on the input device; callers choose the logging boundary.
    """
    rows = plan.sum(-1)
    weights = plan / rows.clamp_min(torch.finfo(plan.dtype).tiny)[..., None]
    entropy = -torch.special.xlogy(weights, weights).sum(-1)
    row_measure = rows / rows.sum(-1, keepdim=True)
    return dict(
        fitted_pair_effective_patches=(row_measure * entropy.exp()).sum(-1),
        fitted_pair_top1_probability=(row_measure * weights.max(-1).values).sum(-1),
        fitted_pair_normalized_entropy=(row_measure * entropy).sum(-1) / (math.log(plan.shape[-1]) if plan.shape[-1] > 1 else 1.),
    )
