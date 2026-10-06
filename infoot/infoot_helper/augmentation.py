import torch
from torchvision import transforms

from diffusion_ot.training.self_supervised_translation import encode_generated_images


augment = transforms.RandomHorizontalFlip(p=0.5)


@torch.no_grad()
def augment_and_encode(domain, images):
    images = torch.stack([augment(image) for image in images])
    images = images.to(device=domain.device, dtype=torch.float32)

    return encode_generated_images(
        domain.vae, images
    ).to(dtype=domain.model_dtype)