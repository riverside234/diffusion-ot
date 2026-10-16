"""Save eight fixed validation previews using the existing flow sampler and VAE."""

from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusion_ot.evaluation.stage1a_eval import decode_vae_latents, integrate_pdae_flow
from diffusion_ot.training.native_flow import native_flow_objective


@torch.inference_mode()
def save_images(state, dataset, config, *, step):
    """Save eight PNGs; call inside validation's eval/EMA context."""
    if len(dataset) < 8:
        raise ValueError("Image previews require at least eight validation samples.")
    if state.vae is None:
        raise ValueError("Image previews require the pretrained VAE.")
    options = config.get("evaluation") or {}
    seed = int(options.get("seed", int((config.get("train") or {}).get("seed", 20260901)) + 1))
    generator = torch.Generator(device=state.device).manual_seed(seed)
    x0 = torch.stack([dataset[index]["x0_latent"] for index in range(8)]).to(
        device=state.device, dtype=state.dtype,
    )
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    flow = native_flow_objective(config)
    start, end = (0.0, 1.0) if flow["direction"] == "noise_to_data" else (1.0, 0.0)
    branch = state.branch
    latents = integrate_pdae_flow(
        branch, branch.semantic_transformer.base, noise, branch.encode(x0),
        num_steps=int(options.get("image_steps", 50)), start_time=start, end_time=end,
        time_eps=flow["time_eps"],
        null_label=(config.get("class_conditioning") or {}).get("null_label"),
    )
    images = decode_vae_latents(state.vae, latents, batch_size=1).cpu()
    directory = Path(config["output_dir"]) / "results" / f"step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images, start=1):
        save_image(image, str(directory / f"{index:02d}.png"))
    return directory
