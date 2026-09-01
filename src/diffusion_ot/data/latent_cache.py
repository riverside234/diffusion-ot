from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diffusion_ot.data.afhq import load_afhq_dataset
from diffusion_ot.data.manifests import manifest_paths, read_jsonl, write_jsonl
from diffusion_ot.integrations.hf_snapshot import (
    effective_project_root,
    load_yaml_config,
    resolve_project_local_path,
    resolve_project_path,
)
from diffusion_ot.integrations.sit_diffusers import load_sit_components


def _center_crop_square(image: Any) -> Any:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def image_to_tensor(image: Any, image_size: int, center_crop: bool = True):
    import numpy as np
    import torch
    from PIL import Image

    if image.mode != "RGB":
        image = image.convert("RGB")
    if center_crop:
        image = _center_crop_square(image)
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype="float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor.mul(2.0).sub(1.0)


def _batched(records: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _resolve_pretrained_config(model_config: dict[str, Any], project_root: Path) -> Path:
    pretrained = model_config.get("pretrained")
    if not pretrained:
        raise ValueError("Model config is missing pretrained.")
    return resolve_project_path(pretrained, project_root)


def cache_vae_latents(
    data_config_path: str | Path,
    model_config_path: str | Path,
    pretrained_config_path: str | Path | None = None,
    project_root: str | Path | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    import torch

    fallback_root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    data_config = load_yaml_config(data_config_path)
    model_config = load_yaml_config(model_config_path)
    root = effective_project_root(data_config, fallback=effective_project_root(model_config, fallback_root))
    pretrained_config = (
        resolve_project_path(pretrained_config_path, root)
        if pretrained_config_path is not None
        else _resolve_pretrained_config(model_config, root)
    )

    dataset = load_afhq_dataset(data_config_path)
    dtype_name = model_config.get("latent_cache", {}).get("dtype", "float16")
    use_half_load = device is not None and str(device).startswith("cuda")
    torch_dtype = dtype_name if use_half_load else None
    components = None if dry_run else load_sit_components(pretrained_config, root, device=device, torch_dtype=torch_dtype)
    vae = None if components is None else components.vae
    if vae is not None:
        vae.eval()

    image_size = int(data_config.get("image_size", model_config.get("resolution", 256)))
    center_crop = bool(data_config.get("center_crop", True))
    batch_size = batch_size or int(model_config.get("latent_cache", {}).get("batch_size", 16))
    latent_root = resolve_project_local_path(
        data_config.get("latent_dir", "data/latents"),
        root,
        field_name="latent_dir",
    )
    latent_root.mkdir(parents=True, exist_ok=True)

    written_records: list[dict[str, Any]] = []
    for manifest_path in manifest_paths(data_config, root):
        records = read_jsonl(manifest_path)
        if limit is not None:
            records = records[:limit]
        split_name = manifest_path.stem
        split_output_dir = latent_root / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for batch_records in _batched(records, batch_size):
            images = []
            for record in batch_records:
                row = dataset[int(record["hf_index"])]
                images.append(image_to_tensor(row[record["image_column"]], image_size, center_crop))

            latent_paths = [split_output_dir / f"{record['sample_id']}.pt" for record in batch_records]
            if not dry_run:
                vae_param = next(vae.parameters())
                pixel_values = torch.stack(images).to(device=vae_param.device, dtype=vae_param.dtype)
                with torch.no_grad():
                    posterior = vae.encode(pixel_values).latent_dist
                    latents = posterior.mean
                    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 1.0)
                    latents = latents * scaling_factor
                save_dtype = torch.float16 if dtype_name in {"float16", "fp16"} else torch.float32
                for latent, latent_path in zip(latents.cpu().to(dtype=save_dtype), latent_paths):
                    torch.save(latent, latent_path)

            for record, latent_path in zip(batch_records, latent_paths):
                item = dict(record)
                item["latent_path"] = str(latent_path)
                item["latent_shape"] = [4, 32, 32]
                item["latent_dtype"] = dtype_name
                item["dry_run"] = dry_run
                written_records.append(item)

    manifest_name = data_config.get("latent_manifest_name", "latent_manifest.jsonl")
    latent_manifest_path = latent_root / manifest_name
    write_jsonl(latent_manifest_path, written_records)
    report = {
        "latent_manifest": str(latent_manifest_path),
        "num_records": len(written_records),
        "dry_run": dry_run,
    }
    (latent_root / "cache_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
