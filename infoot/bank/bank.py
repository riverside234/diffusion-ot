import argparse
from pathlib import Path

import torch
import yaml

from diffusion_ot.data.latent_dataset import load_latent_tensor
from diffusion_ot.models.residual_encoder import PDAEResidualLatentEncoder


def create_bank(root, domain, checkpoint_path, batch_size=32):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    root = Path(root).resolve()
    checkpoint_path = (root / Path(checkpoint_path).expanduser()).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["domain"] != domain:
        raise ValueError(f"Expected a {domain} checkpoint.")
    config = checkpoint["config"]
    encoder_config = dict(config["encoder"])
    if (encoder_config.pop("kind") != "residual_cnn_v1"
            or encoder_config.pop("input_space") != "latent"):
        raise ValueError("Bank creation requires a residual latent CNN encoder.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = checkpoint["model"]["encoder"]
    dtype = next(iter(state.values())).dtype
    encoder = PDAEResidualLatentEncoder(**encoder_config).to(device=device, dtype=dtype)
    encoder.load_state_dict(state, strict=True)
    encoder.eval().requires_grad_(False)

    latent_dir = root / "data/latents/afhq_sit_b2_256" / f"{domain}_train"
    paths = sorted(latent_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No saved latents found in {latent_dir}")
    features = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            x0 = torch.stack([
                load_latent_tensor(path).squeeze(0) for path in paths[start:start + batch_size]
            ]).to(device=device, dtype=dtype)
            features.append(encoder(x0).float().cpu())

    output_dir = root / "data/infoot_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{domain}_pdae.pt"
    config_path = output_dir / f"{domain}_config.yaml"
    torch.save({key: checkpoint[key] for key in (
        "model", "config", "step", "domain", "native_flow_objective",
    ) if key in checkpoint}, model_path)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    v_bank = torch.cat(features)
    bank_path = output_dir / f"{domain}_bank.pt"
    torch.save({
        "v_bank": v_bank, "latent_paths": [str(path) for path in paths],
        "checkpoint_path": str(model_path), "config_path": str(config_path),
        "source_checkpoint_path": str(checkpoint_path),
    }, bank_path)
    print("Saved bank:", bank_path, tuple(v_bank.shape))
    return bank_path


def main(root, domain):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=(
        root / f"outputs/stage1a_{domain}_rescnn_cosmap/checkpoints/latest.pt"
    ))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    create_bank(root, domain, args.checkpoint, args.batch_size)
