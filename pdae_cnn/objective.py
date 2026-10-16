"""Flow-matching objective using the existing PDAE model and loss functions."""

from diffusion_ot.losses.pdae_flow import (
    flow_loss_weight,
    make_linear_flow_target,
    pdae_velocity_gap_loss,
)
from diffusion_ot.models.pdae_sit import make_null_class_labels
from diffusion_ot.training.native_flow import native_flow_objective


def prepare_flow_target(x0, config, *, noise=None, t=None):
    """Sample a linear flow target, or reuse supplied noise and timesteps."""
    flow = native_flow_objective(config)
    return make_linear_flow_target(
        x0, noise=noise, t=t, direction=flow["direction"], eps=flow["time_eps"],
    )


def compute_flow_loss(branch, batch, config, *, noise=None, t=None):
    """Return loss and detached metrics for a batch already on the model device."""
    x0 = batch["x0_latent"]
    target = prepare_flow_target(x0, config, noise=noise, t=t)
    labels = make_null_class_labels(
        branch.semantic_transformer.base, batch_size=x0.shape[0], device=x0.device,
        null_label=(config.get("class_conditioning") or {}).get("null_label"),
    )
    output = branch(
        x0_latent=x0, x_t=target.x_t, timestep=target.t, class_labels=labels,
    )
    direction = (config.get("flow") or {}).get("direction", "noise_to_data")
    weight = flow_loss_weight(target.t.float(), config.get("loss_weighting"), direction=direction)
    pred = output.delta_sample.float()
    target_v = target.target_v.float()
    base = output.base_sample.float()
    loss = pdae_velocity_gap_loss(pred, target_v, base, weight=weight)
    metrics = {
        "loss": loss.detach(),
        "unweighted_flow_mse": pdae_velocity_gap_loss(pred.detach(), target_v, base).detach(),
        "weight_mean": weight.mean().detach(),
    }
    return loss, metrics
