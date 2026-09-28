"""Detached own-encoder spatial correlations for Stage 1B reference matching.

Inspired by PatchSim in https://github.com/lyndonzheng/F-LSeSim (CVPR 2021).
This independent implementation uses spatially centered, channel-normalized
own-encoder maps. Unlike that image loss, we remove the diagonal and compare
all rows symmetrically, without VGG, target-selected masking, or a new loss.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import warnings

import torch
import torch.nn.functional as F

from diffusion_ot.losses.encoder_transport import encoder_transport_cost


SPATIAL_COST_PROTOCOL = "own_encoder_spatial_correlative_v1"


def spatial_correlative_options(config: dict) -> dict | None:
    supplied = config.get("spatial_correlative_cost") or {}
    if not isinstance(supplied, dict):
        raise ValueError("spatial_correlative_cost must be a mapping.")
    if not supplied.get("enabled", False):
        return None
    options = dict(enabled=True, feature_source="own_encoder", layers=[0, 1], grid_size=8,
                   spatial_center=True, channel_normalize=True, exclude_diagonal=True,
                   comparison="symmetric_row_cosine", mixing_weight=0.20,
                   calibration="fixed_train_centered_rms", detach=True, eps=1e-6,
                   min_calibration_spread=1e-6)
    if set(supplied) - set(options):
        raise ValueError(f"Unknown spatial_correlative_cost options: {sorted(set(supplied) - set(options))}")
    options.update(supplied)
    for key in ("feature_source", "comparison", "calibration"):
        expected = {"feature_source": "own_encoder", "comparison": "symmetric_row_cosine",
                    "calibration": "fixed_train_centered_rms"}[key]
        if options[key] != expected:
            raise ValueError(f"spatial_correlative_cost.{key} must be {expected}.")
    for key in ("enabled", "spatial_center", "channel_normalize", "exclude_diagonal", "detach"):
        if options[key] is not True:
            raise ValueError(f"spatial_correlative_cost.{key} must be true for this protocol.")
    layers = options["layers"]
    if (not isinstance(layers, (list, tuple)) or not layers or len(set(layers)) != len(layers)
            or any(type(i) is not int or i < 0 for i in layers)):
        raise ValueError("Spatial cost layers must be distinct nonnegative integers.")
    options["layers"] = list(layers)
    if type(options["grid_size"]) is not int or options["grid_size"] < 2:
        raise ValueError("Spatial cost grid_size must be an integer >= 2.")
    for key in ("mixing_weight", "eps", "min_calibration_spread"):
        value = float(options[key])
        if not math.isfinite(value) or value < 0 or (key != "mixing_weight" and value == 0):
            raise ValueError(f"Invalid spatial_correlative_cost.{key}.")
        options[key] = value
    if options["mixing_weight"] >= 1:
        raise ValueError("Spatial mixing_weight must be less than 1 (retain encoder cost).")
    # Alpha zero is the exact legacy path: no descriptors, state, or extra work.
    return options if options["mixing_weight"] > 0 else None


def validate_spatial_training_config(config: dict) -> dict | None:
    options = spatial_correlative_options(config)
    if options is not None:
        infoot = config.get("infoot") or {}
        if (infoot.get("variant") != "fused" or infoot.get("cross_cost_source") != "encoder"
                or infoot.get("feature_objective", "mi") != "mi"
                or (config.get("matching") or {}).get("distance_scale", "infoot_rms") != "infoot_rms"
                or (config.get("decoded_translation") or {}).get("supervision") != "self_supervised"
                or (config.get("data") or {}).get("split", "train") != "train"):
            raise ValueError("Spatial cost requires self-supervised fused encoder transport, MI-only neural alignment, live InfoOT RMS, and train references.")
    return options


def spatial_descriptor_id(options: dict) -> str:
    payload = json.dumps(dict(protocol=SPATIAL_COST_PROTOCOL, options=options), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@torch.no_grad()
def spatial_descriptors(maps: list[torch.Tensor], *, grid_size: int = 8, eps: float = 1e-6) -> torch.Tensor:
    """[B,C,H,W] maps -> [B,L,P,P] row-normalized relations, in FP32."""
    if not maps or grid_size < 2:
        raise ValueError("Spatial descriptors require maps and grid_size >= 2.")
    rows = []
    for feature in maps:
        if (feature.ndim != 4 or feature.shape[0] != maps[0].shape[0]
                or min(feature.shape[-2:]) < grid_size):
            raise ValueError("Spatial feature maps must share a batch and be at least grid_size in H/W.")
        if not torch.isfinite(feature).all():
            raise FloatingPointError("Non-finite own-encoder spatial feature map.")
        # Explicitly disable autocast for correlations and normalization.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            feature = F.adaptive_avg_pool2d(feature.float(), (grid_size, grid_size)).flatten(2)
            feature = feature - feature.mean(dim=2, keepdim=True)
            feature = F.normalize(feature, dim=1, eps=eps)
            relation = feature.transpose(1, 2).bmm(feature)
            diagonal = torch.eye(relation.shape[-1], device=feature.device, dtype=torch.bool)
            relation = relation.masked_fill(diagonal, 0)
            rows.append(F.normalize(relation, dim=-1, eps=eps))
    return torch.stack(rows, dim=1)


@torch.no_grad()
def encode_spatial_descriptors(encoder, latents, options, *, device, dtype, batch_size=32):
    """Encode the supplied ordered references/queries; preserve all module modes."""
    if not callable(getattr(encoder, "forward_spatial_features", None)):
        raise ValueError("Spatial cost requires encoder.forward_spatial_features; global codes cannot be reshaped into patches.")
    modes = {module: module.training for module in encoder.modules()}
    try:
        encoder.eval()
        return torch.cat([
            spatial_descriptors(encoder.forward_spatial_features(part.to(device=device, dtype=dtype), options["layers"]),
                                grid_size=options["grid_size"], eps=options["eps"])
            for part in latents.split(batch_size)
        ])
    finally:
        for module, mode in modes.items():
            module.training = mode


@torch.no_grad()
def spatial_correlative_cost(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim != 4 or right.ndim != 4 or left.shape[1:] != right.shape[1:]:
        raise ValueError("Spatial descriptors must have compatible [N,L,P,P] shapes.")
    if left.shape[-1] != left.shape[-2] or not left.shape[0] or not right.shape[0]:
        raise ValueError("Spatial descriptors must contain nonempty square relation maps.")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise FloatingPointError("Non-finite spatial descriptor.")
    with torch.autocast(device_type=left.device.type, enabled=False):
        similarity = left.float().flatten(1) @ right.to(left.device).float().flatten(1).T
        return (1 - similarity / (left.shape[1] * left.shape[2])).clamp(0, 2)


@torch.no_grad()
def centered_cost_rms(cost):
    centered = cost.double() - cost.double().mean(0, keepdim=True) - cost.double().mean(1, keepdim=True) + cost.double().mean()
    return float(centered.square().mean().sqrt())


@torch.no_grad()
def descriptor_diagnostics(descriptor, *, eps=1e-6):
    return dict(near_zero_row_fraction=float((descriptor.norm(dim=-1) <= eps).float().mean()),
                spatial_variance=float(descriptor.var(dim=-2, unbiased=False).mean()),
                between_image_variance=float(descriptor.var(dim=0, unbiased=False).mean()))


class SpatialCorrelativeCost:
    """Fixed train-only scale calibration plus a detached per-batch cost builder."""
    def __init__(self, options):
        self.options = deepcopy(options)
        self.calibration = None
        self.probe_baseline = {}

    @torch.no_grad()
    def calibrate(self, matching, descriptors, sample_ids):
        if self.calibration is not None:
            raise ValueError("Spatial calibration is frozen; cannot recalibrate it.")
        for domain in ("cat", "dog"):
            if (len(sample_ids[domain]) < 2 or len(sample_ids[domain]) != matching[domain].shape[0]
                    or len(set(sample_ids[domain])) != len(sample_ids[domain])
                    or descriptors[domain].shape[0] != matching[domain].shape[0]):
                raise ValueError("Spatial calibration IDs must identify every ordered reference exactly once.")
        encoder = encoder_transport_cost(matching["cat"], matching["dog"]).detach().float()
        spatial = spatial_correlative_cost(descriptors["cat"], descriptors["dog"]).to(encoder.device)
        if not torch.isfinite(encoder).all():
            raise FloatingPointError("Non-finite encoder calibration cost.")
        s_enc, s_sc = centered_cost_rms(encoder), centered_cost_rms(spatial)
        floor, alpha = self.options["min_calibration_spread"], self.options["mixing_weight"]
        reason, ratio, gain = None, 0.0, 1.0
        if min(s_enc, s_sc) <= floor:
            reason = "near_constant_calibration_cost"
        else:
            ratio = s_enc / s_sc
            spread = centered_cost_rms((1 - alpha) * encoder + alpha * ratio * spatial)
            if spread <= floor:
                reason = "near_constant_mixed_calibration_cost"
            else:
                gain = s_enc / spread
        self.calibration = dict(status="fallback_encoder" if reason else "calibrated", reason=reason,
                                encoder_spread=s_enc, spatial_spread=s_sc, spatial_scale=ratio,
                                mixture_gain=gain, sample_ids=deepcopy(sample_ids), split="train")
        if reason:
            warnings.warn(f"Spatial cost falls back to encoder cost: {reason}.", RuntimeWarning)
        return self.metadata()

    @torch.no_grad()
    def costs(self, matching, descriptors):
        if self.calibration is None:
            raise ValueError("Spatial cost must be calibrated from the initial training bank or restored from a checkpoint.")
        encoder = encoder_transport_cost(matching["cat"], matching["dog"]).detach().float()
        spatial = spatial_correlative_cost(descriptors["cat"], descriptors["dog"]).to(encoder.device)
        if encoder.shape != spatial.shape or not torch.isfinite(encoder).all():
            raise ValueError("Encoder/spatial reference costs are incompatible or non-finite.")
        alpha, calibration = self.options["mixing_weight"], self.calibration
        mixed = encoder if calibration["status"] == "fallback_encoder" else calibration["mixture_gain"] * (
            (1 - alpha) * encoder + alpha * calibration["spatial_scale"] * spatial)
        if not torch.isfinite(mixed).all():
            raise FloatingPointError("Non-finite mixed spatial transport cost.")
        return dict(encoder=encoder, spatial=spatial, mixed=mixed)

    def metadata(self):
        return deepcopy(dict(protocol=SPATIAL_COST_PROTOCOL, options=self.options, calibration=self.calibration))

    def state_dict(self):
        return {**self.metadata(), "probe_baseline": deepcopy(self.probe_baseline)}

    def load_state_dict(self, state):
        if state.get("protocol") != SPATIAL_COST_PROTOCOL or state.get("options") != self.options:
            raise ValueError("Spatial cost checkpoint protocol/options disagree.")
        calibration = state.get("calibration") or {}
        if calibration.get("status") not in {"calibrated", "fallback_encoder"} or calibration.get("split") != "train":
            raise ValueError("Spatial cost checkpoint has no valid fixed train calibration.")
        for key in ("encoder_spread", "spatial_spread", "spatial_scale", "mixture_gain"):
            if key not in calibration or not math.isfinite(float(calibration[key])) or float(calibration[key]) < 0:
                raise ValueError(f"Invalid spatial calibration {key}.")
        if calibration["mixture_gain"] <= 0 or (calibration["status"] == "calibrated" and calibration["spatial_scale"] <= 0):
            raise ValueError("Invalid spatial calibration scale.")
        for domain in ("cat", "dog"):
            ids = (calibration.get("sample_ids") or {}).get(domain, [])
            if len(ids) < 2 or len(set(ids)) != len(ids):
                raise ValueError("Missing/duplicate spatial calibration sample IDs.")
        self.calibration = deepcopy(calibration)
        self.probe_baseline = deepcopy(state.get("probe_baseline", {}))

    @torch.no_grad()
    def probe_drift(self, descriptors, sample_ids):
        """First fixed-validation cohort is persisted, including across resume."""
        metrics = {}
        for domain, descriptor in descriptors.items():
            current, ids = descriptor[:32].float().cpu(), sample_ids[domain][:32]
            if domain not in self.probe_baseline:
                self.probe_baseline[domain] = dict(descriptors=current.clone(), sample_ids=list(ids))
            initial = self.probe_baseline[domain]
            if initial["sample_ids"] != list(ids) or initial["descriptors"].shape != current.shape:
                raise ValueError("Spatial fixed-validation drift cohort changed; use the original validation cohort.")
            metrics[domain] = float((current - initial["descriptors"]).square().mean().sqrt())
        return metrics


def checkpoint_spatial_cost(checkpoint, config):
    current = spatial_correlative_options(config)
    saved = spatial_correlative_options(checkpoint.get("config") or {})
    state = checkpoint.get("spatial_correlative_cost_state")
    if current != saved or (current is None and state is not None):
        raise ValueError("Spatial cost config and checkpoint disagree; keep v4 and v4.5 separate.")
    if current is None:
        return None
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is missing spatial cost calibration; do not recalibrate from evaluation data.")
    runtime = SpatialCorrelativeCost(current)
    runtime.load_state_dict(state)
    return runtime


@torch.no_grad()
def spatial_cost_diagnostics(runtime, costs, descriptors, coupling, matching, *, bandwidth, solver_options, cross_cost_weight=1.0):
    from diffusion_ot.losses.infoot import gaussian_kernel, infoot_distance_scale, infoot_plan_gradient
    metrics = dict(protocol=SPATIAL_COST_PROTOCOL, calibration_status=runtime.calibration["status"],
                   mixing_weight=runtime.options["mixing_weight"], spatial_scale=runtime.calibration["spatial_scale"],
                   mixture_gain=runtime.calibration["mixture_gain"],
                   descriptors={d: descriptor_diagnostics(v, eps=runtime.options["eps"]) for d, v in descriptors.items()})
    plan = coupling.detach()
    for name, cost in costs.items():
        cost = cost.to(plan)
        expected, uniform = float((plan * cost).sum()), float(cost.mean())
        metrics[name] = dict(expected_cost=expected, uniform_cost=uniform, gain_over_uniform=uniform - expected,
                             centered_rms=centered_cost_rms(cost))
    kernels = [gaussian_kernel(matching[d].detach().to(plan), bandwidth=bandwidth,
                               distance_scale=infoot_distance_scale(matching[d].detach().to(plan)),
                               eps=solver_options["eps"]) for d in ("cat", "dog")]
    effective = cross_cost_weight * costs["mixed"].to(plan) - solver_options["mi_weight"] * infoot_plan_gradient(
        plan, *kernels, eps=solver_options["eps"])
    metrics["effective_sinkhorn_centered_rms_over_epsilon"] = centered_cost_rms(effective) / solver_options["entropy_epsilon"]
    return metrics


@torch.no_grad()
def conditional_spatial_diagnostics(query_descriptor, target_descriptor, weights):
    cost = spatial_correlative_cost(query_descriptor, target_descriptor).to(weights)
    if cost.shape != weights.shape:
        raise ValueError("Conditional spatial diagnostic weights/descriptor ordering disagree.")
    expected, uniform = float((weights * cost).sum(-1).mean()), float(cost.mean())
    return dict(expected_cost=expected, uniform_cost=uniform, gain_over_uniform=uniform - expected,
                best_reference_cost=float(cost.min(-1).values.mean()))
