"""Cache fixed DINOv2 patch-structure descriptors in AFHQ manifest order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data/afhq_huggan.yaml")
    parser.add_argument("--output", default="data/semantic_priors/afhq_dinov2_structure.pt")
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=4)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModel
    from diffusion_ot.data.afhq import load_afhq_dataset
    from diffusion_ot.data.latent_cache import image_to_tensor
    from diffusion_ot.data.manifests import read_jsonl
    from diffusion_ot.data.latent_dataset import split_manifest_path
    from diffusion_ot.integrations.hf_snapshot import (
        effective_project_root, load_yaml_config, resolve_project_local_path,
    )
    from diffusion_ot.losses.semantic_prior import descriptor_digest, patch_structure_descriptor

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    config_path = Path(args.data_config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml_config(config_path)
    root = effective_project_root(config, fallback=ROOT)
    output = resolve_project_local_path(args.output, root, field_name="output")
    if output.exists():
        raise FileExistsError(f"Choose a new --output to preserve the existing descriptor bank: {output}")
    cache = root / "artifacts" / "pretrained" / "dinov2"
    processor = AutoImageProcessor.from_pretrained(args.model, revision=args.revision, cache_dir=cache)
    model = AutoModel.from_pretrained(args.model, revision=args.revision, cache_dir=cache)
    model.eval().requires_grad_(False).to(args.device)
    dataset = load_afhq_dataset(config_path)
    ids, domains, splits, features = [], [], [], []
    manifest_hashes = {}
    import hashlib
    for domain in ("cat", "dog"):
        for split in ("train", "val"):
            manifest = split_manifest_path(config, domain, split, root)
            manifest_hashes[f"{domain}_{split}"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            records = read_jsonl(manifest)
            for start in range(0, len(records), args.batch_size):
                batch = records[start:start + args.batch_size]
                # Exactly the Stage 1A crop/resize before DINO's own preprocessing.
                images = [image_to_tensor(
                    dataset[int(row["hf_index"])][row["image_column"]],
                    int(config.get("image_size", 256)), bool(config.get("center_crop", True)),
                ).add(1).div(2) for row in batch]
                inputs = processor(images=images, do_rescale=False, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    # dinov2-small has one CLS token followed by a square patch grid.
                    tokens = model(**inputs).last_hidden_state[:, 1:]
                    descriptors = patch_structure_descriptor(tokens, args.grid_size)
                features.append(descriptors.cpu())
                ids.extend(row["sample_id"] for row in batch)
                domains.extend([domain] * len(batch))
                splits.extend([split] * len(batch))
            print(f"cached {domain}_{split}: {len(records)}", flush=True)
    values = torch.cat(features)
    # Only training descriptors determine the cost scale; validation is read-only.
    calibration = {}
    for domain in ("cat", "dog"):
        indices = [i for i in range(len(ids)) if domains[i] == domain and splits[i] == "train"]
        if len(indices) < 3:
            raise ValueError(f"Not enough training descriptors for {domain}.")
        gen = torch.Generator().manual_seed(20260912)
        chosen = torch.randperm(len(indices), generator=gen)[:512].tolist()
        calibration[domain] = [indices[i] for i in chosen]
    scale = float(torch.cdist(values[calibration["cat"]], values[calibration["dog"]]).median())
    payload = {
        "format_version": 1, "sample_ids": ids, "domains": domains, "splits": splits,
        "features": values,
        "metadata": {
            "model": args.model, "requested_revision": args.revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            # Convert PIL resampling enums to plain JSON primitives so the bank
            # can be loaded with torch.load(weights_only=True).
            "processor": json.loads(json.dumps(processor.to_dict())),
            "descriptor": "pooled_patch_self_similarity_v1",
            "grid_size": args.grid_size, "image_size": int(config.get("image_size", 256)),
            "center_crop": bool(config.get("center_crop", True)),
            "manifest_hashes": manifest_hashes, "cost_scale": scale,
            "calibration_ids": {d: [ids[i] for i in indices] for d, indices in calibration.items()},
        },
    }
    payload["fingerprint"] = descriptor_digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(f"semantic_prior: {output}\nfingerprint: {payload['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
