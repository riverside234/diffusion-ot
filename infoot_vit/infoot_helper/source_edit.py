"""Source-latent experiments for the existing noise-to-data PDAE velocity field.

FlowEdit-inspired update (https://arxiv.org/abs/2412.08629, Algorithm 1),
adapted to image-token conditions and this repository's opposite time direction.
This is an experimental adaptation, not a reproduced paper result.
"""
from __future__ import annotations

import hashlib
import math

import torch


def edit_schedule(num_steps, start_time=.5):
    """Use a suffix of the reference grid, preserving its 1/num_steps step size."""
    if (type(num_steps) is not int or num_steps < 1 or not math.isfinite(start_time)
            or not 0 <= start_time < 1):
        raise ValueError("Need positive reference steps and a start time in [0,1).")
    first = round(start_time * num_steps)
    if not math.isclose(first / num_steps, start_time, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Edit start must lie on the reference integration grid.")
    return [(i, (i + .5) / num_steps, 1. / num_steps) for i in range(first, num_steps)]


def step_noises(like, ids, seed, step):
    """Independent per-step, per-image noise, invariant to query order/chunking."""
    if (len(ids) != len(like) or len(set(ids)) != len(ids) or type(seed) is not int
            or type(step) is not int or step < 0):
        raise ValueError("Noise requires unique matching IDs, integer seed and step.")
    seeds = [int.from_bytes(hashlib.sha256(f"source_edit_v1:{seed}:{sid}:{step}".encode()).digest()[:8],
                            "big") % (2**63 - 1) for sid in ids]
    return torch.stack([torch.randn(like.shape[1:], device=like.device, dtype=like.dtype,
        generator=torch.Generator(device=like.device).manual_seed(s)) for s in seeds])


def _finite(value, name):
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite or nonfloating {name} in source editing.")


@torch.inference_mode()
def flow_edit(source, source_tokens, target_tokens, valid, ids, *, seed,
              num_steps, velocity, start_time=.5):
    """Noise-free edit state; source/target velocity calls share each noise draw.

    t=0 is noise, t=1 is data. At midpoint t, s=(1-t)*eps+t*source,
    target_state=s+(edit-source), edit += dt*(v_target-v_source).
    n_avg=1, no target-only finishing steps, identical CFG for both fields.
    No raw source-feature residual is added to the target condition.
    """
    _finite(source, "source latent")
    for tokens in (source_tokens, target_tokens):
        _finite(tokens, "condition")
    if (source.ndim != 4 or len(source) == 0 or len(ids) != len(source) or source_tokens.ndim != 3
            or source_tokens.shape != target_tokens.shape or len(source_tokens) != len(source)
            or valid.dtype != torch.bool or valid.shape != source_tokens.shape[:2]
            or not valid.any(1).all()):
        raise ValueError("Source editing requires matched latents, tokens and nonempty shared masks.")
    schedule = edit_schedule(num_steps, start_time)
    edited = source.clone()
    trace = []
    for step, t, dt in schedule:
        noise = step_noises(source, ids, seed, step)
        noisy_source = (1 - t) * noise + t * source
        noisy_target = noisy_source + (edited - source)
        source_velocity = velocity(noisy_source, t, source_tokens, valid)
        target_velocity = velocity(noisy_target, t, target_tokens, valid)
        if source_velocity.shape != source.shape or target_velocity.shape != source.shape:
            raise ValueError("Velocity shape differs from the source latent.")
        _finite(source_velocity, "source velocity")
        _finite(target_velocity, "target velocity")
        delta = target_velocity - source_velocity
        edited = edited + dt * delta
        _finite(edited, "edited latent")
        trace.append(dict(step=step, time=t, dt=dt,
            delta_velocity_rms=float(delta.double().square().mean().sqrt()),
            source_displacement_rms=float((edited - source).double().square().mean().sqrt())))
    return edited, trace


def make_velocity(evaluator, *, guidance_scale, null_label, batch_size):
    """The same learned-null CFG used by integrate_pdae_flow; bounded batches."""
    from diffusion_ot.models.pdae_sit import make_null_class_labels
    if (type(batch_size) is not int or batch_size < 1 or not math.isfinite(guidance_scale)
            or guidance_scale < 0 or not evaluator.branch.semantic_cfg_enabled):
        raise ValueError("Source editing requires the reviewed learned-null CFG and valid batch size/scale.")

    @torch.inference_mode()
    def velocity(state, time, tokens, valid):
        outputs = []
        for i in range(0, len(state), batch_size):
            x = state[i:i + batch_size]
            labels = make_null_class_labels(evaluator.transformer, batch_size=len(x),
                device=x.device, null_label=null_label)
            outputs.append(evaluator.branch.predict_cfg_with_z(x_t=x,
                timestep=torch.full((len(x),), time, dtype=torch.float32, device=x.device),
                z=tokens[i:i + batch_size], condition_padding_mask=~valid[i:i + batch_size],
                guidance_scale=guidance_scale, class_labels=labels).sample)
        return torch.cat(outputs)
    return velocity


@torch.inference_mode()
def encode_source(vae, images, *, device, dtype, batch_size):
    """Use the deterministic VAE posterior mode and the training latent scale."""
    _finite(images, "source RGB")
    scale = float(vae.config.scaling_factor)
    if (images.ndim != 4 or images.shape[1] != 3 or len(images) == 0
            or (images < 0).any() or (images > 1).any() or batch_size < 1
            or not math.isfinite(scale) or scale <= 0
            or getattr(vae.config, "shift_factor", 0.) not in (None, 0.)):
        raise ValueError("Require [0,1] source RGB and the existing unshifted scaled VAE convention.")
    parameter = next(vae.parameters(), None)
    vae_dtype = parameter.dtype if parameter is not None else dtype
    latent = torch.cat([vae.encode((chunk * 2 - 1).to(device=device, dtype=vae_dtype)).latent_dist.mode()
                        for chunk in images.split(batch_size)]) * scale
    _finite(latent, "encoded source")
    return latent.to(device=device, dtype=dtype)
