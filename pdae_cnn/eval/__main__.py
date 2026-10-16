"""Validate a saved CNN PDAE checkpoint and generate eight preview images."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from diffusion_ot.config_defaults import stage1a_training_config
from diffusion_ot.integrations.hf_snapshot import resolve_project_local_path
from pdae_cnn.helpers.config import load_config


def evaluate(config, *, checkpoint=None, weights="ema"):
    """Load model weights, then run the same validation/images used in training."""
    import torch

    from diffusion_ot.training.native_flow import validate_stage1a_objective
    from pdae_cnn.data import build_loaders
    from pdae_cnn.eval.validation import validate
    from pdae_cnn.initialize import build_training_state

    if weights not in ("raw", "ema"):
        raise ValueError("weights must be raw or ema.")
    path = resolve_project_local_path(
        checkpoint or Path(config["output_dir"]) / "checkpoints/latest.pt",
        config["project_root"], field_name="checkpoint",
    )
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("domain") != config["domain"]:
        raise ValueError("Checkpoint domain does not match the evaluation config.")
    validate_stage1a_objective(config, saved)
    if weights == "ema" and saved.get("ema") is None:
        raise ValueError("Checkpoint has no EMA weights; use --weights raw.")
    step = int(saved["step"])
    config = {
        **config,
        "augmentation": {**(config.get("augmentation") or {}), "enabled": False},
        "ema": {**(config.get("ema") or {}), "enabled": weights == "ema"},
        "evaluation": {**(config.get("evaluation") or {}), "enabled": True,
                       "save_images": True, "use_ema": weights == "ema"},
    }
    state = build_training_state(config)
    state.branch.load_pdae_state_dict(saved["model"])
    if weights == "ema":
        state.ema.load_state_dict(saved["ema"], state.branch)
    del saved
    _, loader = build_loaders(config)
    metrics = validate(state, loader, config, step=step)
    return {"checkpoint": str(path), "step": step, **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    recipe = parser.add_mutually_exclusive_group(required=True)
    recipe.add_argument("--domain", choices=("cat", "dog"))
    recipe.add_argument("--config", help="The training YAML used by the checkpoint.")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to the run's latest.pt.")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", help="Torch device override, such as cuda:0 or cpu.")
    parser.add_argument("--project-root", help="Override the training recipe's project root.")
    parser.add_argument("--output-dir", help="Run directory containing checkpoints and results.")
    args = parser.parse_args()
    path = Path(args.config or stage1a_training_config(args.domain)).expanduser()
    config = load_config(
        path if path.is_absolute() else ROOT / path,
        project_root=args.project_root, device=args.device, output_dir=args.output_dir,
    )
    print(json.dumps(evaluate(config, checkpoint=args.checkpoint, weights=args.weights), indent=2))


if __name__ == "__main__":
    main()
