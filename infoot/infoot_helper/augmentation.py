from torchvision.transforms import TrivialAugmentWide, InterpolationMode

from diffusion_ot.training.self_supervised_translation import encode_generated_images


augment = TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR)


@torch.no_grad()
def augment_and_encode(domain, images):
    images = (images * 255).round().to(torch.uint8)
    images = torch.stack([augment(image) for image in images])
    images = images.to(device=domain.device, dtype=torch.float32) / 255

    return encode_generated_images(
        domain.vae, images
    ).to(dtype=domain.model_dtype)