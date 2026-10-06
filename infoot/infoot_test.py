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
from infoot_helper.infoot_test_helper import generate_and_save_grid
from diffusion_ot.data.latent_dataset import load_latent_tensor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--h", type=float)
parser.add_argument("--reg", type=float)
parser.add_argument("--save", type=str, default="1")
args = parser.parse_args()

@torch.inference_mode()
def generate_and_save_grid(
    dog, v_dog, cat_image, output_path, steps=20, seed=0
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
    save_image(grid, str(output_path), nrow=len(cat_image), padding=8)


device = "cuda" if torch.cuda.is_available() else "cpu"
bank_dir = ROOT / "data/infoot_test"

cat_bank = torch.load(
    bank_dir / "cat_bank.pt", map_location="cpu", weights_only=True
)
dog_bank = torch.load(
    bank_dir / "dog_bank.pt", map_location="cpu", weights_only=True
)
transport = torch.load(
    bank_dir / "cat_to_dog_plan.pt",
    map_location=device,
    weights_only=True,
)
if not isinstance(transport, dict) or transport.get("feature_space") != "raw":
    raise ValueError("Rerun infoot_fit.py to save a plan fitted on raw features.")

Xs = cat_bank["v_bank"].to(device=device, dtype=torch.float32)
Xt = dog_bank["v_bank"].to(device=device, dtype=torch.float32)

solver = infoot.InfoOT(
    Xs, Xt, h=args.h, reg=transport["reg"], lam=transport["lam"]
)
solver.P = transport["P"].to(dtype=Xs.dtype)
assert solver.P.shape == (len(Xs), len(Xt))

count = 16
latent_dir = ROOT / "data/latents/afhq_sit_b2_256/cat_val"
paths = sorted(latent_dir.glob("*.pt"))[:count]
if len(paths) < count:
    raise ValueError(f"Need {count} validation latents in {latent_dir}")

cat = load_stage1a_evaluator(
    ROOT / "configs/stage1a_pdae/cat_sit_b2_lora_residual_cosmap.yaml",
    ROOT / "configs/stage1a_eval/residual_sit_b2_256.yaml",
    device=device,
    weights="raw",
    checkpoint_path=cat_bank["checkpoint_path"],
)

with torch.no_grad():
    x0 = torch.stack([load_latent_tensor(path) for path in paths])
    x0 = x0.to(device=cat.device, dtype=cat.model_dtype)
    v_cat = cat.branch.encode(x0).to(Xs)

    scores = solver.conditional_score(v_cat)
    v_dog = infoot.projection(scores, Xt)

del cat

records = {
    record["sample_id"]: record
    for record in read_jsonl(ROOT / "data/manifests/cat_val.jsonl")
}
cat_images = load_ground_truth_images(
    ROOT / "configs/data/afhq_huggan.yaml",
    [records[path.stem] for path in paths],
)

dog = load_stage1a_evaluator(
    ROOT / "configs/stage1a_pdae/dog_sit_b2_lora_residual_cosmap.yaml",
    ROOT / "configs/stage1a_eval/residual_sit_b2_256.yaml",
    device=device,
    weights="raw",
    checkpoint_path=dog_bank["checkpoint_path"],
)

output_path = ROOT / f"results/infoot_test/cat_to_dog_test_{args.save}.png"
generate_and_save_grid(dog, v_dog, cat_images, output_path)
print("Saved:", output_path)
