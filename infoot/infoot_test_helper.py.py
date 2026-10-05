from pathlib import Path
import torch
from torchvision.utils import save_image

from diffusion_ot.evaluation.stage1a_eval import (
    integrate_pdae_flow,
    decode_vae_latents,
)


@torch.inference_mode()
def generate_and_save_grid(
    dog, v_dog, cat_images, output_path, steps=50, seed=0
):
    v_dog = v_dog.to(device=dog.device, dtype=dog.model_dtype)
    generator = torch.Generator(device=dog.device).manual_seed(seed)
    noise = torch.randn(
        (len(v_dog), 4, 32, 32),
        device=dog.device,
        dtype=dog.model_dtype,
        generator=generator,
    )

    latents = integrate_pdae_flow(
        dog.branch,
        dog.transformer,
        noise,
        v_dog,
        num_steps=steps,
        guidance_scale=1.0,
        null_label=dog.training_config["class_conditioning"]["null_label"],
    )
    dog_images = decode_vae_latents(dog.vae, latents)

    grid = torch.stack(
        [cat_images.cpu(), dog_images.cpu()], dim=1
    ).flatten(0, 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid.clamp(0, 1), str(output_path), nrow=4, padding=8)
