"""Save an eight-column grid: original RGB, null token, CFG 1.0, and CFG 1.5."""

from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusion_ot.data.ground_truth import load_ground_truth_images
from diffusion_ot.evaluation.stage1a_eval import decode_vae_latents, integrate_pdae_flow
from diffusion_ot.training.native_flow import native_flow_objective


@torch.inference_mode()
def save_images(state, dataset, config, *, step):
    """Save four matched rows; call inside validation's eval/EMA context."""
    if len(dataset) < 8:
        raise ValueError("Image previews require at least eight validation samples.")
    if state.vae is None:
        raise ValueError("Image previews require the pretrained VAE.")
    options = config.get("evaluation") or {}
    seed = int(options.get("seed", int((config.get("train") or {}).get("seed", 20260901)) + 1))
    generator = torch.Generator(device=state.device).manual_seed(seed)
    items = [dataset[index] for index in range(8)]
    originals = load_ground_truth_images(config["data_config"], [item["metadata"] for item in items])
    rows = [originals.cpu()]
    x0 = torch.stack([item["x0_latent"] for item in items]).to(
        device=state.device, dtype=state.dtype,
    )
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    flow = native_flow_objective(config)
    start, end = (0.0, 1.0) if flow["direction"] == "noise_to_data" else (1.0, 0.0)
    branch = state.branch
    z = branch.encode(x0)
    for scale in (0.0, 1.0, 1.5):
        latents = integrate_pdae_flow(
            branch, branch.semantic_transformer.base, noise, z,
            num_steps=int(options.get("image_steps", 50)), start_time=start, end_time=end,
            guidance_scale=scale, time_eps=flow["time_eps"],
            null_label=(config.get("class_conditioning") or {}).get("null_label"),
        )
        rows.append(decode_vae_latents(state.vae, latents, batch_size=1).cpu())
    directory = Path(config["output_dir"]) / "results" / f"step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat(rows, dim=0), str(directory / "reconstruction_grid.png"), nrow=8)
    return directory
