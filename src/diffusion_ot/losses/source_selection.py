"""Opt-in source-aware correction of InfoOT conditional probabilities.

The fitted coupling and MI objective are unchanged. Detached original-image
compatibility corrects query readout, not balanced OT marginals. No external
teacher is used. Fixed train-only scales and the protocol are checkpointed.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
import torch.nn.functional as F

from diffusion_ot.losses.lab_swd import coarse_lab_descriptor
from diffusion_ot.losses.spatial_correlative import spatial_correlative_cost

SELECTION_PROTOCOL = "source_spatial_coarse_lab_conditional_v1"


def source_selection_options(config):
    supplied = config.get("source_aware_selection") or {}
    if not isinstance(supplied, dict):
        raise ValueError("source_aware_selection must be a mapping.")
    if not supplied.get("enabled", False):
        return None
    options = dict(enabled=True, spatial_weight=.25, appearance_weight=.25,
                   appearance_size=8, calibration="fixed_train_row_centered_rms", eps=1e-8,
                   min_calibration_spread=1e-6)
    if set(supplied) - set(options):
        raise ValueError("Unknown source_aware_selection options.")
    options.update(supplied)
    if options["enabled"] is not True or options["calibration"] != "fixed_train_row_centered_rms":
        raise ValueError("Selection requires enabled: true and fixed train-only calibration.")
    if type(options["appearance_size"]) is not int or options["appearance_size"] < 1:
        raise ValueError("Selection appearance_size must be a positive integer.")
    for key in ("spatial_weight", "appearance_weight", "eps", "min_calibration_spread"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid selection {key}.")
        if key in ("eps", "min_calibration_spread") and value == 0:
            raise ValueError(f"Selection {key} must be positive.")
        options[key] = float(value)
    return options


class SourceAwareSelection:
    def __init__(self, options):
        self.options = deepcopy(options)
        self.calibration = None

    def appearance(self, rgb):
        return coarse_lab_descriptor(rgb, self.options["appearance_size"])

    @torch.no_grad()
    def costs(self, query_spatial, target_spatial, query_appearance, target_appearance):
        with torch.autocast(device_type=query_appearance.device.type, enabled=False):
            appearance = torch.cdist(query_appearance.detach().float(),
                                     target_appearance.detach().to(query_appearance.device).float(), p=1) / query_appearance.shape[1]
            spatial = spatial_correlative_cost(query_spatial, target_spatial).to(appearance.device)
        if spatial.shape != appearance.shape or not torch.isfinite(appearance).all():
            raise ValueError("Selection descriptors must be finite with identical ordered rows.")
        return dict(spatial=spatial, appearance=appearance)

    @torch.no_grad()
    def calibrate(self, spatial, appearance, sample_ids):
        if self.calibration is not None:
            raise ValueError("Selection calibration is frozen.")
        for d in ("cat", "dog"):
            ids = sample_ids[d]
            if len(ids) < 2 or len(ids) != len(set(ids)) or len(ids) != len(spatial[d]) or len(ids) != len(appearance[d]):
                raise ValueError("Selection calibration needs unique ordered training IDs matching both descriptors.")
        costs = self.costs(spatial["cat"], spatial["dog"], appearance["cat"], appearance["dog"])
        # Each direction has its own row-centering: target-popularity offsets
        # affect conditional selection and must not be removed as in balanced OT.
        directions = {}
        for source, target in (("cat", "dog"), ("dog", "cat")):
            scales = {}
            for key, value in costs.items():
                value = value.double() if source == "cat" else value.T.double()
                scale = float((value - value.mean(1, keepdim=True)).square().mean().sqrt())
                scales[key] = scale
            directions[f"{source}_to_{target}"] = scales
        self.calibration = dict(split="train", sample_ids=deepcopy(sample_ids), scales=directions)

    def apply(self, log_weights, costs, *, direction, target_codes):
        if self.calibration is None:
            raise ValueError("Selection requires saved train-only calibration.")
        if log_weights.ndim != 2 or not torch.isfinite(log_weights).all():
            raise ValueError("Selection requires finite log probabilities.")
        if target_codes.shape[0] != log_weights.shape[1]:
            raise ValueError("Selection target codes must match probability columns.")
        scales = self.calibration["scales"][direction]
        penalty = torch.zeros_like(log_weights)
        active = False
        for name in ("spatial", "appearance"):
            cost = costs[name].detach().to(log_weights)
            if cost.shape != log_weights.shape or not torch.isfinite(cost).all():
                raise ValueError("Selection costs must match finite ordered query/target probabilities.")
            scale = scales[name]
            if self.options[f"{name}_weight"] and scale > self.options["min_calibration_spread"]:
                penalty = penalty + self.options[f"{name}_weight"] * cost / scale
                active = True
        # Stable log(W + eps), with exact legacy forward/gradient if both
        # coefficients are zero (or both calibration costs are degenerate).
        corrected = F.log_softmax(torch.logaddexp(log_weights, log_weights.new_tensor(math.log(self.options["eps"])))
                                 - penalty, dim=1) if active else log_weights
        with torch.no_grad():
            def stats(logp):
                weights = logp.exp()
                codes = target_codes.detach().to(weights)
                projected = weights @ codes
                result = dict(effective_targets=float((-(weights * logp).sum(1)).exp().mean()),
                    mean_max_target_probability=float(weights.max(1).values.mean()),
                    projected_to_target_norm_ratio=float(projected.norm(dim=1).mean() / codes.norm(dim=1).mean().clamp_min(1e-8)),
                    projected_to_target_variance_ratio=float(projected.var(0, unbiased=False).mean() / codes.var(0, unbiased=False).mean().clamp_min(1e-12)))
                result.update({f"expected_{k}_cost": float((weights * v.to(weights)).sum(1).mean()) for k, v in costs.items()})
                return result
            before, after = stats(log_weights), stats(corrected)
            metrics = dict(protocol=SELECTION_PROTOCOL, before=before, after=after, scales=deepcopy(scales),
                mean_weight_l1_change=float((corrected.exp() - log_weights.exp()).abs().sum(1).mean()),
                calibration_fallback={k: scales[k] <= self.options["min_calibration_spread"] for k in scales})
            metrics["cost_gain"] = {k: before[f"expected_{k}_cost"] - after[f"expected_{k}_cost"] for k in costs}
        return corrected, metrics

    def state_dict(self):
        return deepcopy(dict(protocol=SELECTION_PROTOCOL, options=self.options, calibration=self.calibration))

    def load_state_dict(self, state):
        if state.get("protocol") != SELECTION_PROTOCOL or state.get("options") != self.options:
            raise ValueError("Source selection checkpoint protocol/options disagree.")
        calibration = state.get("calibration") or {}
        if calibration.get("split") != "train":
            raise ValueError("Selection checkpoint needs train-only calibration.")
        for direction in ("cat_to_dog", "dog_to_cat"):
            scales = (calibration.get("scales") or {}).get(direction, {})
            for name in ("spatial", "appearance"):
                scale = scales.get(name)
                if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
                    raise ValueError("Invalid selection calibration scale.")
        for d in ("cat", "dog"):
            ids = (calibration.get("sample_ids") or {}).get(d, [])
            if len(ids) < 2 or len(ids) != len(set(ids)):
                raise ValueError("Invalid selection calibration sample IDs.")
        self.calibration = deepcopy(calibration)


def checkpoint_source_selection(checkpoint, config):
    options = source_selection_options(config)
    if options != source_selection_options(checkpoint.get("config") or {}):
        raise ValueError("Source selection config and checkpoint disagree; start a separate experiment.")
    state = checkpoint.get("source_aware_selection_state")
    if options is None:
        if state is not None:
            raise ValueError("Unexpected source selection checkpoint state.")
        return None
    if state is None:
        raise ValueError("Missing source selection calibration; never calibrate from evaluation queries.")
    runtime = SourceAwareSelection(options)
    runtime.load_state_dict(state)
    return runtime
