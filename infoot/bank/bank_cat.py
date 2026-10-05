from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_ot.data.latent_dataset import load_latent_tensor
from diffusion_ot.models.residual_encoder import PDAEResidualLatentEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# check point for pdae
checkpoint_path = (
    ROOT / "outputs/stage1a_cat_rescnn_cosmap/checkpoints/latest.pt"
)

# Use the same encoder configuration as the checkpoint.
config_path = ROOT / "configs/stage1a_pdae/cat_sit_b2_lora_residual_cosmap.yaml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
encoder_config = dict(config["encoder"])

assert encoder_config.pop("kind") == "residual_cnn_v1"
assert encoder_config.pop("input_space") == "latent"

encoder = PDAEResidualLatentEncoder(**encoder_config)
checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=True
)
encoder.load_state_dict(checkpoint["model"]["encoder"], strict=True)
encoder = encoder.to(device).eval()
encoder.requires_grad_(False)


latent_dir = ROOT / "data/latents/afhq_sit_b2_256"
latent_paths = sorted(latent_dir.glob("*.pt"))
if not latent_paths:
    raise FileNotFoundError(f"No saved latents found in {latent_dir}")

features = []

with torch.inference_mode():
    for latent_path in latent_paths:
        x0 = load_latent_tensor(latent_path)

        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)     # [1, 4, 32, 32]

        x0 = x0.to(
            device=device,
            dtype=next(encoder.parameters()).dtype,
        )

        v = encoder(x0)             # [1, 512]
        features.append(v.cpu())

v_bank = torch.cat(features, dim=0)  # [N, 512]

output_dir = ROOT / "data/infoot_test"
output_dir.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "v_bank": v_bank,
        "latent_paths": [str(path) for path in latent_paths],
        "checkpoint_path": str(checkpoint_path),
    },
    output_dir / "cat_bank.pt",
)

print("Saved bank:", v_bank.shape)
