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
from infoot_helper.infoot_test_helper import (
    encode_paths,
    prepare_cotraining_test,
    generate_and_save_grid,
)
from diffusion_ot.data.latent_dataset import load_latent_tensor

import argparse

from infoot_helper.infoot_cotraining_helper import (
    conditional_mapping,
)

parser = argparse.ArgumentParser()
parser.add_argument("--h", type=float, default=0.4)
parser.add_argument("--reg", type=float, default=0.02)
parser.add_argument("--save", type=str, default="1")
parser.add_argument("--step", type=int, default=2000)
parser.add_argument("--restarts", type=int, default=6)
parser.add_argument("--fit-iterations", type=int, default=1200)
parser.add_argument("--seed", type=int, default=0)

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

cat, dog, Xs, Xt, P = prepare_cotraining_test(
    ROOT,
    {"cat": cat_bank, "dog": dog_bank},
    step=args.step,
    device=device,
    h=args.h,
    reg=args.reg,
    restarts=args.restarts,
    iterations=args.fit_iterations,
    seed=args.seed,
)

count = 16
latent_dir = ROOT / "data/latents/afhq_sit_b2_256/cat_val"
paths = sorted(latent_dir.glob("*.pt"))[:count]
if len(paths) < count:
    raise ValueError(f"Need {count} validation latents in {latent_dir}")

with torch.no_grad():
    v_cat = encode_paths(cat, paths)
    v_dog = conditional_mapping(
        v_cat, Xs, Xt, P, h=args.h,
    )

records = {
    record["sample_id"]: record
    for record in read_jsonl(ROOT / "data/manifests/cat_val.jsonl")
}
cat_images = load_ground_truth_images(
    ROOT / "configs/data/afhq_huggan.yaml",
    [records[path.stem] for path in paths],
)

output_path = ROOT / (
    f"results/infoot_test/"
    f"cotraining_step_{args.step:06d}_{args.save}.png"
)
generate_and_save_grid(dog, v_dog, cat_images, output_path)
print("Saved:", output_path)
