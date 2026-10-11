import argparse
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_ot.data.manifests import read_jsonl
from diffusion_ot.data.ground_truth import load_ground_truth_images
from infoot_helper.infoot_test_helper import (
    encode_paths,
    prepare_cotraining_test,
    generate_and_save_grid,
)
from infoot_helper.infoot_cotraining_helper import (
    conditional_mapping,
)
from infoot_helper.cotraining_checkpoint import latest_checkpoint
from infoot_helper.umap_plot import save_umap

parser = argparse.ArgumentParser()
parser.add_argument("--h", type=float, default=0.2)
parser.add_argument("--reg", type=float, default=0.74)
parser.add_argument("--lam", type=float, default=1.95)
parser.add_argument("--save", type=str, default="1")
parser.add_argument("--step", type=int, help="Checkpoint step (default: latest).")

args = parser.parse_args()

if args.step is None:
    checkpoint = latest_checkpoint(ROOT / "outputs/infoot_cotraining")
    if checkpoint is None:
        raise FileNotFoundError(f"No co-training checkpoints in {ROOT / 'outputs/infoot_cotraining'}")
    args.step = int(checkpoint.stem[5:])
print(f"Evaluating co-training step {args.step}")

device = "cuda" if torch.cuda.is_available() else "cpu"
bank_dir = ROOT / "data/infoot_test"

cat_bank = torch.load(
    bank_dir / "cat_bank.pt", map_location="cpu", weights_only=True
)
dog_bank = torch.load(
    bank_dir / "dog_bank.pt", map_location="cpu", weights_only=True
)

domains, _, matching, batch_norms, P = prepare_cotraining_test(
    ROOT,
    {"cat": cat_bank, "dog": dog_bank},
    step=args.step,
    device=device,
    h=args.h,
    reg=args.reg,
    lam=args.lam
)
cat, dog = domains["cat"], domains["dog"]

count = 16
latent_dir = ROOT / "data/latents/afhq_sit_b2_256/cat_val"
paths = sorted(latent_dir.glob("*.pt"))[:count]
if len(paths) < count:
    raise ValueError(f"Need {count} validation latents in {latent_dir}")

with torch.no_grad():
    v_cat = encode_paths(cat, paths)
    m_cat = batch_norms["cat"](v_cat)
    m_dog = conditional_mapping(
        m_cat, matching["cat"], matching["dog"], P,
        h=args.h,
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
generate_and_save_grid(dog, m_dog, cat_images, output_path, steps=20)
print("Saved:", output_path)
print("Saved:", save_umap(
    matching["cat"], matching["dog"], m_cat, m_dog,
    output_path.with_name(f"{output_path.stem}_umap.png"),
    title=f"Co-training step {args.step}: BatchNorm features",
))
