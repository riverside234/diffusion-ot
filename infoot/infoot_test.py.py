from infoot_helper import infoot
import torch
import sys
from pathlib import Path
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_ot.data.manifests import read_jsonl
from diffusion_ot.data.ground_truth import load_ground_truth_images
from diffusion_ot.evaluation.stage1a_eval import (
    load_stage1a_evaluator,
    integrate_pdae_flow,
    decode_vae_latents,
)

@torch.inference_mode()
def generate_and_save_grid(
    dog, v_dog, cat_image, output_path, steps=50, seed=0
):
    v_dog = v_dog.to(device=dog.device, dtype=dog.model_dtype)
    generator = torch.Generator(device=dog.device).manual_seed(seed)

    noise = torch.randn(
        (1, 4, 32, 32),
        device=dog.device,
        dtype=dog.model_dtype,
        generator=generator,
    )

    dog_latent = integrate_pdae_flow(
        dog.branch,
        dog.transformer,
        noise,
        v_dog,
        num_steps=steps,
        guidance_scale=1.0,
        null_label=dog.training_config["class_conditioning"]["null_label"],
    )
    dog_image = decode_vae_latents(dog.vae, dog_latent)

    grid = torch.cat(
        [cat_image.cpu(), dog_image.cpu()], dim=0
    ).clamp(0, 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(output_path), nrow=2, padding=8)


device = "cuda" if torch.cuda.is_available() else "cpu"
bank_dir = ROOT / "data/infoot_test"

cat_bank = torch.load(
    bank_dir / "cat_bank.pt", map_location="cpu", weights_only=True
)
dog_bank = torch.load(
    bank_dir / "dog_bank.pt", map_location="cpu", weights_only=True
)
P = torch.load(
    bank_dir / "cat_to_dog_plan.pt",
    map_location=device,
    weights_only=True,
)

Xs = cat_bank["v_bank"].to(device=device, dtype=torch.float32)
Xt = dog_bank["v_bank"].to(device=device, dtype=torch.float32)

solver = infoot.InfoOT(Xs, Xt, h=0.5, reg=0.05)
solver.P = P.to(dtype=Xs.dtype)
assert solver.P.shape == (len(Xs), len(Xt))

index = 0

with torch.no_grad():
    v_cat = Xs[index:index + 1]
    scores = solver.conditional_score(v_cat)
    v_dog = infoot.projection(scores, Xt)

#select a cat from bank
sample_id = Path(cat_bank["latent_paths"][index]).stem
records = read_jsonl(ROOT / "data/manifests/cat_train.jsonl")
record = next(r for r in records if r["sample_id"] == sample_id)

cat_image = load_ground_truth_images(
    ROOT / "configs/data/afhq_huggan.yaml",
    [record],
)

dog = load_stage1a_evaluator(
    ROOT / "configs/stage1a_pdae/dog_sit_b2_lora_residual_cosmap.yaml",
    ROOT / "configs/stage1a_eval/residual_sit_b2_256.yaml",
    device=device,
    weights="raw",
    checkpoint_path=dog_bank["checkpoint_path"],
)

output_path = ROOT / "results/infoot_test" / f"{sample_id}_cat_to_dog.png"
generate_and_save_grid(dog, v_dog, cat_image, output_path)

print("Saved:", output_path)
