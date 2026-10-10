"""Extract the exact frozen PDAE-v2 SigLIP representation from original RGB."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from diffusion_ot.data.ground_truth import OriginalImageLoader
from diffusion_ot.data.manifests import read_jsonl
from diffusion_ot.integrations.hf_snapshot import load_yaml_config, resolve_project_local_path
from diffusion_ot.models.pdae_v2.encoder import FrozenSiglipPatchEncoder
from infoot_vit.infoot_helper.feature_bank import BankWriter, file_hash


def main(domain, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", default=f"configs/stage1a_pdae_v2_l/{domain}.yaml")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, help="Explicit first-N stable-ID subset; recorded in provenance.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    resolve = lambda p: resolve_project_local_path(p, root, field_name="bank path")
    train_path = resolve(args.train_config)
    train = load_yaml_config(train_path)
    enc = train["encoder"]
    if train["domain"] != domain or enc["kind"] != "siglip2_vit_b16" or not enc.get("frozen"):
        raise ValueError("Choose a matching-domain frozen-SigLIP PDAE v2/v2-L config.")
    if args.batch_size < 1 or (args.max_images is not None and args.max_images < 1):
        raise ValueError("Batch size and any explicit image limit must be positive.")
    data_path = resolve(train["data_config"])
    data = load_yaml_config(data_path)
    manifest_path = resolve(data["manifest_dir"]) / f"{domain}_{args.split}.jsonl"
    records = sorted(read_jsonl(manifest_path), key=lambda r: r["sample_id"])
    records = records if args.max_images is None else records[:args.max_images]
    if not records:
        raise ValueError(f"No images in {manifest_path}")
    loader = OriginalImageLoader(data_path, records)
    encoder = FrozenSiglipPatchEncoder.from_local(resolve(enc["local_dir"])).to(args.device).eval()
    representation = dict(grid=[14, 14], dim=768, patch_order="row_major_no_special_tokens",
        encoder=encoder.snapshot_identity, layer="last_hidden_state_post_layernorm", normalization="none",
        preprocessing=dict(rgb_image_size=int(data["image_size"]), center_crop=bool(data.get("center_crop", True)),
                           rgb_range=[-1, 1], siglip_processor="snapshot_pil_do_rescale_false", augmentation="none"))
    output = args.output_dir or Path(f"data/infoot_vit/{domain}_{args.split}")
    writer = BankWriter(resolve(output), domain=domain, split=args.split, representation=representation,
        provenance=dict(manifest=str(manifest_path), manifest_sha256=file_hash(manifest_path),
                        data_config=str(data_path), train_config=str(train_path), max_images=args.max_images,
                        extraction_dtype="float32", saved_dtype="float32", selection="sorted_stable_ids"))
    with torch.inference_mode():
        for offset in range(0, len(records), args.batch_size):
            batch = records[offset:offset + args.batch_size]
            rgb = torch.stack([loader.load(record) for record in batch])
            writer.append(encoder(rgb).float(), batch)
            print(f"{domain}/{args.split}: {min(offset + len(batch), len(records))}/{len(records)}", flush=True)
    manifest = writer.finish()
    print(f"Saved [N,196,768] bank to {writer.directory}; ID={manifest['artifact_id']}")
    return 0
