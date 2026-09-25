"""Existing Stage 1B experiments must retain their latent-input checkpoints."""
from pathlib import Path

import pytest
import torch
import yaml

from diffusion_ot.evaluation.stage1b_eval import (
    _load_domain_context, require_latent_stage1a_encoder,
)
from diffusion_ot.training.train_joint_infoot import _load_training_domain


@pytest.mark.parametrize("loader", ["train", "evaluation"])
@pytest.mark.parametrize("origin", ["recipe_space", "recipe_channels", "checkpoint_config", "checkpoint_model"])
def test_stage1b_loaders_reject_rgb_before_loading_pretrained_models(tmp_path, loader, origin):
    # No model/data configs exist: the input-space error must happen first.
    recipe = {"domain": "cat"}
    checkpoint = {"model": {}}
    if origin == "recipe_space":
        recipe["encoder"] = {"input_space": "rgb", "input_channels": 3}
    elif origin == "recipe_channels":
        recipe["encoder"] = {"input_channels": 3}
    elif origin == "checkpoint_config":
        checkpoint["config"] = {"encoder": {"input_space": "rgb", "input_channels": 3}}
    else:
        checkpoint["model"]["encoder_input"] = {"space": "rgb", "image_size": 256}
    (tmp_path / "cat.yaml").write_text(yaml.safe_dump(recipe))
    torch.save(checkpoint, tmp_path / "cat.pt")
    config = {"stage1a": {"cat": {"config": "cat.yaml", "checkpoint": "cat.pt"}}}
    with pytest.raises(ValueError, match="Stage 1B currently requires.*VAE-latent"):
        if loader == "train":
            _load_training_domain(config, tmp_path, "cat", device_override="cpu")
        else:
            _load_domain_context(config, tmp_path, "cat", checkpoint_path=None,
                                 joint_weights="raw", device_override="cpu")


@pytest.mark.parametrize("recipe", [{}, {"encoder": {"input_channels": 4}},
    {"encoder": {"input_space": "latent", "input_channels": 4}}])
def test_stage1b_accepts_legacy_and_explicit_latent_encoder_metadata(recipe):
    require_latent_stage1a_encoder(recipe, source="test recipe")
    require_latent_stage1a_encoder(recipe, source="test checkpoint", checkpoint={
        "model": {"encoder_input": {"space": "latent", "image_size": None}}})


def test_existing_stage1b_recipes_pin_legacy_stage1a_architecture_and_checkpoints():
    root = Path(__file__).resolve().parents[1]
    pinned_count = 0
    for path in (root / "configs/stage1b_infoot").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        for domain in ("cat", "dog"):
            initialization = config.get("stage1a", {}).get(domain, {})
            current_rgb_path = f"configs/stage1a_pdae/{domain}_sit_b2_lora_rgb.yaml"
            assert initialization.get("config") != current_rgb_path, path.name
            if initialization.get("config") == f"configs/stage1a_pdae/{domain}_sit_b2_lora.yaml":
                pinned_count += 1
                legacy = yaml.safe_load((root / initialization["config"]).read_text())
                assert legacy["encoder"].get("input_space", "latent") == "latent"
                assert legacy["encoder"]["input_channels"] == 4
                assert initialization["checkpoint"] == legacy["output_dir"] + "/checkpoints/latest.pt"
    assert pinned_count >= 4  # Both global InfoNCE and PatchNCE, in both domains.
