from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn


MODEL_ID = "google/siglip2-base-patch16-224"
MANIFEST_NAME = "pdae_v2_snapshot.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_identity(directory: Path) -> dict:
    """Verify the locally pinned encoder/processor; never contact the Hub."""
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run scripts/download_siglip2.py first.")
    identity = json.loads(path.read_text(encoding="utf-8"))
    files = identity.get("files", {})
    if (identity.get("model_id") != MODEL_ID or not identity.get("revision")
            or not {"config.json", "preprocessor_config.json"}.issubset(files)
            or not any(name.endswith(".safetensors") for name in files)):
        raise ValueError("Invalid PDAE v2 SigLIP snapshot manifest.")
    for name, expected in files.items():
        candidate = (directory / name).resolve()
        if not candidate.is_relative_to(directory.resolve()):
            raise ValueError("Snapshot manifest contains a path outside its directory.")
        if not candidate.is_file() or file_sha256(candidate) != expected:
            raise ValueError(f"SigLIP snapshot file changed or is missing: {name}")
    return identity


class FrozenSiglipPatchEncoder(nn.Module):
    """FixRes SigLIP 2 vision tower; keep all post-LN patches, never pooled output."""

    def __init__(self, vision_model: nn.Module, processor, identity: dict) -> None:
        super().__init__()
        config = vision_model.config
        if (config.model_type != "siglip_vision_model" or config.hidden_size != 768
                or config.patch_size != 16 or config.image_size != 224):
            raise ValueError("PDAE v2 requires FixRes SigLIP 2 ViT-B/16, 224px, 768-D.")
        self.vision_model = vision_model.requires_grad_(False)
        self.processor = processor
        self.snapshot_identity = identity
        self.architecture_spec = {
            "kind": "siglip2_vit_b16", "model_id": MODEL_ID,
            "token_dim": 768, "tokens": 196, "image_size": 224, "frozen": True,
            "features": "last_hidden_state_post_layernorm",
        }
        self.train(False)

    @classmethod
    def from_local(cls, directory: str | Path):
        directory = Path(directory)
        identity = snapshot_identity(directory)
        from transformers import AutoImageProcessor, SiglipVisionModel

        # This FixRes checkpoint is model_type=siglip, not the NaFlex Siglip2 model.
        # The backend API arrived during Transformers 5.x. Keep PIL processing
        # on both APIs; allowing the new torchvision default changes the pixels.
        has_backend_api = "image_processor_classes" in inspect.signature(AutoImageProcessor.register).parameters
        processor_options = {"backend": "pil"} if has_backend_api else {"use_fast": False}
        processor = AutoImageProcessor.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, **processor_options,
        )

        class VisionOnlySiglipModel(SiglipVisionModel):
            # Official snapshots also contain the unused text tower and scoring
            # scalars. Scope these exceptions to our loader, not the global class.
            _keys_to_ignore_on_load_unexpected = [r"^text_model\.", r"^logit_(scale|bias)$"]

        vision, loading = VisionOnlySiglipModel.from_pretrained(
            directory, local_files_only=True, output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete or incompatible SigLIP vision weights in {directory}: {loading}")
        return cls(vision, processor, identity)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("SigLIP expects floating-point RGB [B,3,H,W] in [-1,1].")
        if not bool(torch.isfinite(images).all()) or bool((images.abs() > 1.0002).any()):
            raise ValueError("SigLIP RGB input must be finite and in [-1,1].")
        # Match the saved processor exactly. RGB already has the crop/flip of the
        # cached latent target. Do not rescale [0,1] a second time by 1/255.
        rgb = ((images.detach().float().cpu() + 1) / 2).clamp(0, 1)
        pixels = self.processor(
            images=list(rgb.numpy()), do_rescale=False, return_tensors="pt",
            input_data_format="channels_first",
        )["pixel_values"]
        parameter = next(self.vision_model.parameters())
        pixels = pixels.to(device=parameter.device, dtype=parameter.dtype)
        tokens = self.vision_model(pixel_values=pixels, return_dict=True).last_hidden_state
        if tokens.shape != (images.shape[0], 196, 768):
            raise ValueError(f"Expected SigLIP patch tokens [B,196,768], received {tuple(tokens.shape)}.")
        return tokens


def patch_token_statistics(tokens: torch.Tensor) -> dict:
    """Diagnostics on token populations, without replacing conditioning by a vector."""
    if tokens.ndim != 3 or not bool(torch.isfinite(tokens).all()):
        raise ValueError("Expected finite patch tokens [B,T,D].")
    tokens = tokens.float()
    population = tokens.reshape(-1, tokens.shape[-1])
    # Bound validation SVD cost; deterministic sampling includes every image.
    stride = max(1, (len(population) + 2047) // 2048)
    sample = population[::stride]
    energy = torch.linalg.svdvals(sample - sample.mean(0)).square()
    p = energy / energy.sum().clamp_min(1e-12)
    rank = torch.exp(-(p * p.clamp_min(1e-12).log()).sum()) if energy.sum() > 0 else energy.new_zeros(())
    return {
        "count": tokens.shape[0], "dim": tokens.shape[-1], "tokens_per_image": tokens.shape[1],
        "statistics_space": "patch_token_population", "rank_sample_count": len(sample),
        "norm_mean": float(population.norm(dim=-1).mean()),
        "norm_std": float(population.norm(dim=-1).std(unbiased=False)),
        "feature_std_mean": float(population.std(0, unbiased=False).mean()),
        "effective_rank": float(rank),
    }
