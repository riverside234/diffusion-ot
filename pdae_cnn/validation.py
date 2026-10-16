"""Small, repeatable conditioning probes using unweighted velocity MSE."""

from contextlib import nullcontext
from itertools import islice

import torch

from diffusion_ot.losses.pdae_flow import pdae_velocity_gap_loss
from diffusion_ot.models.pdae_sit import make_null_class_labels
from pdae_cnn.data import prepare_batch
from pdae_cnn.objective import prepare_flow_target


@torch.no_grad()
def validate(state, loader, config):
    """Return sample-averaged MSEs while preserving training mode and RNG state.

    Singleton batches are omitted from the shuffled-code comparison.
    """
    options = config.get("evaluation") or {}
    if loader is None or not options.get("enabled", True):
        return {}
    num_batches = int(options.get("num_batches", 4))
    if num_batches <= 0:
        raise ValueError("evaluation.num_batches must be positive.")
    seed = int(options.get("seed", int((config.get("train") or {}).get("seed", 20260901)) + 1))
    time_eps = float((config.get("flow") or {}).get("time_eps", 1e-5))
    branch, was_training = state.branch, state.branch.training
    use_ema = bool(options.get("use_ema", True) and state.ema is not None)
    parameters = state.ema.average_parameters(branch) if use_ema else nullcontext()
    devices = [state.device] if state.device.type == "cuda" else []
    generator = torch.Generator(device=state.device).manual_seed(seed)
    loader_generator = getattr(loader, "generator", None)
    loader_rng = loader_generator.get_state() if loader_generator is not None else None
    totals, counts = {}, {}

    try:
        with torch.random.fork_rng(devices=devices), parameters:
            branch.eval()
            if loader_generator is not None:
                loader_generator.manual_seed(seed)
            for batch in islice(loader, num_batches):
                batch = prepare_batch(batch, device=state.device, dtype=state.dtype)
                x0 = batch["x0_latent"]
                noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
                t = torch.rand(len(x0), device=x0.device, dtype=x0.dtype, generator=generator)
                target = prepare_flow_target(x0, config, noise=noise, t=t.clamp(time_eps, 1 - time_eps))
                labels = make_null_class_labels(
                    branch.semantic_transformer.base, batch_size=len(x0), device=x0.device,
                    null_label=(config.get("class_conditioning") or {}).get("null_label"),
                )
                z = branch.encode(x0)
                conditions = {"correct_z": z}
                if len(x0) > 1:
                    conditions["shuffled_z"] = torch.roll(z, shifts=1, dims=0)
                if branch.semantic_cfg_enabled:
                    conditions["null_z"] = branch.semantic_null_like(z)
                for name, condition in conditions.items():
                    output = branch.predict_with_z(target.x_t, target.t, condition, class_labels=labels)
                    mse = pdae_velocity_gap_loss(
                        output.delta_sample.float(), target.target_v.float(), output.base_sample.float(),
                    )
                    totals[name] = totals.get(name, 0) + mse * len(x0)
                    counts[name] = counts.get(name, 0) + len(x0)
    finally:
        branch.train(was_training)
        if loader_generator is not None:
            loader_generator.set_state(loader_rng)

    if not counts:
        raise ValueError("Validation loader produced no batches.")
    metrics = {f"{name}_mse": float((total / counts[name]).cpu()) for name, total in totals.items()}
    return {**metrics, "num_samples": counts["correct_z"], "use_ema": use_ema}
