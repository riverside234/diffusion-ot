import torch

from diffusion_ot.training.decoded_translation import (
    integrate_training_flow,
    decode_training_images,
)
from diffusion_ot.training.self_supervised_translation import (
    encode_generated_images,
    source_code_contrastive_loss,
)


def translation_contrastive_loss(
    domain,
    source,
    mapped_v,
    steps=20,
    temperature=0.2,
    query_projector=None,
    key_projector=None,
):
    mapped_v = mapped_v.to(
        device=domain.device, dtype=domain.model_dtype
    )
    noise = torch.randn_like(
        source["queries"]["x0"],
        device=domain.device,
        dtype=domain.model_dtype,
    )

    generated_latent = integrate_training_flow(
        domain.branch,
        domain.transformer,
        noise,
        mapped_v,
        num_steps=steps,
        guidance_scale=1.0,
        null_label=domain.training_config[
            "class_conditioning"
        ]["null_label"],
    )

    generated_image = decode_training_images(
        domain.vae, generated_latent
    )
    recovered_latent = encode_generated_images(
        domain.vae, generated_image
    )
    recovered_v = domain.branch.encode(
        recovered_latent.to(dtype=domain.model_dtype)
    ).float()

    return source_code_contrastive_loss(
        recovered_v,
        source["queries"]["v"],
        source["v"],
        temperature=temperature,
        query_projector=query_projector,
        key_projector=key_projector,
    )