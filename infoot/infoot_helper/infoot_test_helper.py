from pathlib import Path
import torch
from torchvision.utils import save_image

from diffusion_ot.evaluation.stage1a_eval import (
    integrate_pdae_flow,
    decode_vae_latents,
)
from diffusion_ot.data.latent_dataset import load_latent_tensor
from diffusion_ot.evaluation.stage1a_eval import load_stage1a_evaluator
from . import infoot
from .infoot_cotraining_helper import fit_transport
from .transport.plan_io import file_identity

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

#functions for co-training test (load co-training checkpoints)
@torch.no_grad()
def encode_paths(domain, paths, batch_size=32):
    features = []
    for start in range(0, len(paths), batch_size):
        x0 = torch.stack([
            load_latent_tensor(path)
            for path in paths[start:start + batch_size]
        ]).to(device=domain.device, dtype=domain.model_dtype)
        features.append(domain.branch.encode(x0).float())
    return torch.cat(features)


def prepare_cotraining_test(root, banks, step, device, h=0.4, reg=0.02,
                           iterations=1200):
    models, features = {}, {}

    for name, bank in banks.items():
        models[name] = load_stage1a_evaluator(
            root / f"configs/stage1a_pdae/{name}_sit_b2_lora_residual_cosmap.yaml",
            root / "configs/stage1a_eval/residual_sit_b2_256.yaml",
            device=device,
            weights="raw",
            checkpoint_path=(
                root / "outputs/infoot_cotraining"
                / f"{name}_step_{step:06d}.pt"
            ),
        )
        features[name] = encode_paths(models[name], bank["latent_paths"])

    Xs, Xt = features["cat"], features["dog"]
    diagnostics = {}
    P = fit_transport(Xs, Xt, h=h, reg=reg, mi_weight=0.10,
                      iterations=iterations, diagnostics=diagnostics)
    references = {
        name: {"latent_paths": banks[name]["latent_paths"],
               "checkpoint_at_fit": file_identity(model.checkpoint_path)}
        for name, model in models.items()
    }
    infoot.save_plan(
        root / "outputs/infoot_cotraining" / f"cat_to_dog_step_{step:06d}_plan.pt",
        P, h, reg, lam=0.10,
        optimization=diagnostics, banks=references,
    )
    return models["cat"], models["dog"], Xs, Xt, P
