"""Resolved native velocity objective shared by Stage 1A/1B checkpoint paths."""
from copy import deepcopy
import math

from diffusion_ot.losses.pdae_flow import resolve_flow_loss_weighting


def native_flow_objective(config):
    flow = config.get("flow") or {}
    direction = str(flow.get("direction", "noise_to_data"))
    time_eps = float(flow.get("time_eps", 1e-5))
    if direction not in {"noise_to_data", "data_to_noise"}:
        raise ValueError(f"Unsupported native flow direction: {direction}")
    if not math.isfinite(time_eps) or not 0 <= time_eps < .5:
        raise ValueError("Native flow time_eps must be finite and in [0, 0.5).")
    return dict(protocol="native_velocity_uniform_time_v1", direction=direction,
                time_eps=time_eps, timestep_sampling="uniform",
                loss_weighting=resolve_flow_loss_weighting(config.get("loss_weighting")))


def validate_stage1a_objective(config, checkpoint, *, required_type=None):
    current = native_flow_objective(config)
    if required_type is not None and current["loss_weighting"]["type"] != required_type:
        raise ValueError(f"Stage 1A native flow weighting must be {required_type} for this protocol.")
    saved = checkpoint.get("native_flow_objective")
    if saved is None and "config" in checkpoint:
        saved_config = deepcopy(checkpoint["config"] or {})
        recorded_weighting = (checkpoint.get("train_state") or {}).get("loss_weighting")
        if recorded_weighting is not None:
            saved_config["loss_weighting"] = recorded_weighting
        saved = native_flow_objective(saved_config)
    if saved is None:
        # Historical SNR checkpoints can lack recipe metadata. Never relabel
        # such a checkpoint as a new cosmap/uniform objective.
        if current["loss_weighting"]["type"] != "pdae_flow_snr":
            raise ValueError("Stage 1A checkpoint lacks native flow objective provenance.")
    elif saved != current:
        raise ValueError("Stage 1A native flow weighting objective changed on resume/load; use the exact training recipe.")
    return current


def validate_joint_objective(checkpoint, domain, config):
    current = native_flow_objective(config)
    saved_all = checkpoint.get("native_flow_objectives")
    if saved_all is None:
        # Old Stage 1B used SNR unconditionally. A claimed cosmap/uniform
        # recipe is not evidence that the old implementation executed it.
        if current["loss_weighting"]["type"] != "pdae_flow_snr":
            raise ValueError("Stage 1B checkpoint lacks native flow objective provenance; start a fresh cosmap/uniform run.")
    elif saved_all.get(domain) != current:
        raise ValueError(f"Stage 1B {domain} native flow objective changed on resume/load.")
    return current
