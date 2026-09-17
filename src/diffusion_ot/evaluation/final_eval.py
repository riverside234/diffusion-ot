"""Stage 4: all held-out translations, directional FID/source SSIM, fixed pairs."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from diffusion_ot.evaluation import stage1b_eval as legacy
from diffusion_ot.evaluation.final_image_metrics import (
    CleanFID, exact_image_files, image_name, metric_versions, save_gallery, save_rgb, source_ssim,
)
from diffusion_ot.evaluation.full_infoot import project_full
from diffusion_ot.evaluation.offline_artifacts import (
    atomic_json, atomic_torch, bind_run, file_hash, fingerprint, read_torch, verify_files,
)
from diffusion_ot.evaluation.offline_pipeline import (
    DOMAINS, build_bank, complete_dataset, domain_data_path, load_bundle, local, release_models,
)
from diffusion_ot.integrations.hf_snapshot import load_yaml_config


def image_seed(seed, direction, sample_id):
    return int(fingerprint([int(seed), direction, str(sample_id)])[:16], 16) % (2**63 - 1)


def fixed_noise(ids, shape, seed, direction):
    return torch.stack([torch.randn(shape, generator=torch.Generator().manual_seed(
        image_seed(seed, direction, key)), dtype=torch.float32) for key in ids])


def gallery_ids(ids, seed, direction, count=16):
    # Ranking IDs is independent of their manifest order and output quality.
    return sorted(ids, key=lambda key: fingerprint([seed, direction, key]))[:count]


def cached_image_valid(path, receipt, identity):
    if not path.exists() or not receipt.exists():
        return False
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    return recorded.get("identity") == identity and recorded.get("sha256") == file_hash(path)


def record_image(path, receipt, pixels, identity, **metadata):
    save_rgb(path, pixels)
    atomic_json(receipt, {"identity": identity, "sha256": file_hash(path), **metadata})


def cached_bank(path, identity):
    receipt = path.with_suffix(".json")
    if not path.exists() or not receipt.exists():
        return None
    if json.loads(receipt.read_text(encoding="utf-8")) != {"identity": identity, "sha256": file_hash(path)}:
        raise ValueError(f"Bank receipt mismatch: {path}")
    return legacy.load_latent_bank(path)


def save_bank(path, bank, identity):
    legacy.save_latent_bank(path, bank)
    atomic_json(path.with_suffix(".json"), {"identity": identity, "sha256": file_hash(path)})


def pending_images(query, output, direction, identity):
    return [i for i, key in enumerate(query.sample_ids)
            if not cached_image_valid(output / "images" / direction / image_name(key),
                                      output / "receipts" / direction / (image_name(key) + ".json"), identity)]


@torch.inference_mode()
def generate_direction(context, query, codes, output, direction, *, identity, seed,
                       num_steps, guidance, batch_size, image_size):
    from diffusion_ot.data.latent_dataset import load_latent_tensor
    from diffusion_ot.evaluation.stage1a_eval import integrate_pdae_flow, decode_vae_latents
    shape = tuple(load_latent_tensor(query.metadata[0]["latent_path"]).shape)
    if len(shape) != 3:
        raise ValueError("Expected CHW latent shape")
    pending = pending_images(query, output, direction, identity)
    for start in range(0, len(pending), batch_size):
        indices = pending[start:start+batch_size]
        ids = [query.sample_ids[i] for i in indices]
        noise = fixed_noise(ids, shape, seed, direction).to(context.device, dtype=context.dtype)
        latent = integrate_pdae_flow(
            context.branch, context.transformer, noise, codes[indices].to(context.device, dtype=context.dtype),
            num_steps=num_steps, guidance_scale=guidance,
            null_label=(context.training_config.get("class_conditioning") or {}).get("null_label"))
        images = decode_vae_latents(context.vae, latent).cpu()
        if images.shape != (len(ids), 3, image_size, image_size):
            raise ValueError("Generated image dimensions disagree with the evaluation protocol")
        for key, pixels in zip(ids, images):
            name = image_name(key)
            record_image(output / "images" / direction / name,
                         output / "receipts" / direction / (name + ".json"), pixels, identity,
                         sample_id=key, seed=image_seed(seed, direction, key), direction=direction)
        print(f"{direction}: generated {min(start+batch_size, len(pending))}/{len(pending)} pending images", flush=True)


def run_stage4(config_path, bundle_path, *, root, output_dir=None, device=None, resume=False,
               generation_batch_size=None):
    from diffusion_ot.data.afhq import load_afhq_dataset
    from diffusion_ot.data.ground_truth import load_ground_truth_images
    root = Path(root).resolve()
    torch.set_float32_matmul_precision("highest")
    config = load_yaml_config(config_path)
    if (config.get("evaluation_split") != "val" or config.get("source_samples_per_domain") != "all"
            or config.get("real_samples_per_domain") != "all"):
        raise ValueError("Stage 4 requires complete held-out validation source and real-target sets")
    if config.get("metrics") != ["fid", "source_ssim"]:
        raise ValueError("Stage 4 metric protocol is [fid, source_ssim]")
    versions = metric_versions()  # Fail before expensive generation if packages are missing/wrong.
    bundle_path = local(root, bundle_path)
    manifest, references, plan = load_bundle(bundle_path)
    verify_files(root, manifest["dependencies"])
    alignment = manifest["alignment"]
    seed = int(config["seed"])
    image_size = int(config["image_size"])
    steps, guidance = int(config["num_steps"]), float(config["guidance_scale"])
    count = int(config["gallery_pairs_per_direction"])
    encode_batch = int(config["encoding_batch_size"])
    query_batch, target_block = int(config["projection_batch_size"]), int(config["projection_target_block_size"])
    generation_batch = int(config["generation_batch_size"] if generation_batch_size is None else generation_batch_size)
    fid_batch = int(config["fid_batch_size"])
    if min(image_size, steps, count, encode_batch, query_batch, target_block, generation_batch, fid_batch) < 1:
        raise ValueError("Stage 4 sizes/counts must be positive")
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance_scale must be finite and nonnegative")
    data_paths = {d: domain_data_path(alignment, root, d) for d in DOMAINS}
    datasets = {p: load_afhq_dataset(p) for p in set(data_paths.values())}
    for p in data_paths.values():
        if int(load_yaml_config(p).get("image_size", 256)) != image_size:
            raise ValueError("Image size differs from the data preprocessing protocol")
    evaluation_ids = {}
    for d in DOMAINS:
        evaluation_ids[d] = [str(r["sample_id"]) for r in complete_dataset(alignment, root, d, "val").records]
        if evaluation_ids[d] != manifest["validation_ids"][d]:
            raise ValueError("Held-out manifest changed since Stage 2-3 offline alignment")
    implementation = fingerprint({name: file_hash(Path(__file__).parent / name)
                                  for name in ("final_eval.py", "final_image_metrics.py", "full_infoot.py")})
    protocol = {"bundle_identity": manifest["identity"], "bundle_sha256": file_hash(bundle_path / "bundle.json"),
                "seed": seed, "image_size": image_size, "num_steps": steps, "guidance_scale": guidance,
                "readout": "conditional_mean", "metrics": config["metrics"], "versions": versions,
                "gallery_pairs_per_direction": count, "implementation": implementation,
                "datasets": {str(p): getattr(ds, "_fingerprint", None) for p, ds in datasets.items()}}
    identity = fingerprint(protocol)
    output = local(root, output_dir or config["output_dir"])
    bind_run(output, identity, resume=resume)
    atomic_json(output / "protocol.json", {**protocol, "identity": identity})
    query_banks = {}
    for domain in DOMAINS:
        path = output / f"queries/{domain}.pt"
        bank = cached_bank(path, identity)
        if bank is None:
            bank = build_bank(alignment, root, domain, "val", bundle_path / "models/paired_state.pt",
                              manifest["weights"], manifest["identity"],
                              device or config["generation_device"], encode_batch)
            save_bank(path, bank, identity)
        legacy.validate_bank_compatibility(references[domain], bank)
        if bank.sample_ids != evaluation_ids[domain]:
            raise ValueError("Incomplete held-out bank")
        query_banks[domain] = bank
        # Stream original RGB from the original dataset, never decode the VAE for references.
        for start in range(0, len(bank.sample_ids), encode_batch):
            selected = []
            for i in range(start, min(start+encode_batch, len(bank.sample_ids))):
                name = image_name(bank.sample_ids[i])
                if not cached_image_valid(output / "real" / domain / name,
                                          output / "receipts" / f"real_{domain}" / (name + ".json"), identity):
                    selected.append(i)
            if not selected:
                continue
            images = load_ground_truth_images(data_paths[domain], [bank.metadata[i] for i in selected],
                                               dataset=datasets[data_paths[domain]])
            for i, pixels in zip(selected, images):
                key = bank.sample_ids[i]
                name = image_name(key)
                record_image(output / "real" / domain / name,
                             output / "receipts" / f"real_{domain}" / (name + ".json"), pixels, identity,
                             sample_id=key, reference="original_dataset_rgb")
    del datasets
    solver_device = device or config["projection_device"]
    gallery_selection = {}
    for source, target in (("cat", "dog"), ("dog", "cat")):
        direction = f"{source}_to_{target}"
        query = query_banks[source]
        code_path = output / f"projections/{direction}.pt"
        code_receipt = code_path.with_suffix(".json")
        if code_path.exists() and code_receipt.exists():
            if json.loads(code_receipt.read_text(encoding="utf-8")) != {"identity": identity, "sha256": file_hash(code_path)}:
                raise ValueError("Projected-code cache mismatch")
            codes = read_torch(code_path)
        else:
            print(f"Projecting all {len(query.sample_ids)} {direction} queries against the full train bank...", flush=True)
            coupling = (plan if source == "cat" else plan.T).to(solver_device)
            codes = project_full(query.matching_features, references[source].matching_features.to(coupling),
                                 references[target].matching_features.to(coupling), coupling,
                                 references[target].raw_codes, source_scale=manifest["scales"][source],
                                 target_scale=manifest["scales"][target], bandwidth=manifest["projection_bandwidth"],
                                 query_batch_size=query_batch, target_block_size=target_block)
            atomic_torch(code_path, codes)
            atomic_json(code_receipt, {"identity": identity, "sha256": file_hash(code_path)})
            del coupling
            release_models()
        if codes.shape != (len(query.sample_ids), references[target].raw_codes.shape[1]) or not torch.isfinite(codes).all():
            raise ValueError("Invalid projected codes")
        if pending_images(query, output, direction, identity):
            context = legacy._load_domain_context(alignment, root, target,
                                                 checkpoint_path=bundle_path / "models/paired_state.pt",
                                                 joint_weights=manifest["weights"],
                                                 device_override=device or config["generation_device"])
            try:
                context.branch.eval().requires_grad_(False)
                context.vae.eval().requires_grad_(False)
                generate_direction(context, query, codes, output, direction, identity=identity, seed=seed,
                                   num_steps=steps, guidance=guidance, batch_size=generation_batch, image_size=image_size)
            finally:
                del context
                release_models()
        selected = gallery_ids(query.sample_ids, seed, direction, count)
        gallery_selection[direction] = selected
        save_gallery(output / "galleries" / direction, direction, selected,
                     output / "real" / source, output / "images" / direction)
    atomic_json(output / "gallery_ids.json", gallery_selection)
    fid = CleanFID(local(root, config["fid_cache_dir"]), device=device or config["fid_device"],
                   batch_size=fid_batch, num_workers=int(config["num_workers"]))
    metrics, per_image, generation_manifest = {}, [], []
    for source, target in (("cat", "dog"), ("dog", "cat")):
        direction = f"{source}_to_{target}"
        ids = query_banks[source].sample_ids
        generated = exact_image_files(output / "images" / direction, ids)
        originals = exact_image_files(output / "real" / source, ids)
        targets = exact_image_files(output / "real" / target, query_banks[target].sample_ids)
        values = []
        for key, fake, real in zip(ids, generated, originals):
            value = source_ssim(fake, real)
            values.append(value)
            per_image.append({"direction": direction, "source_id": key, "source_ssim": value})
            generation_manifest.append({"direction": direction, "source_id": key,
                                        "seed": image_seed(seed, direction, key),
                                        "translation": str(fake.relative_to(output)), "translation_sha256": file_hash(fake),
                                        "source": str(real.relative_to(output)), "source_sha256": file_hash(real)})
        metrics[direction] = {"fid": fid.compute(generated, targets),
                              "source_ssim": {"mean": float(np.mean(values)), "std": float(np.std(values)), "count": len(values)},
                              "generated_count": len(generated), "real_target_count": len(targets),
                              "gallery_pairs": len(gallery_selection[direction])}
    write_jsonl(output / "ssim_per_image.jsonl", per_image)
    write_jsonl(output / "generation_manifest.jsonl", generation_manifest)
    report = {"status": "complete", "protocol": protocol, "identity": identity,
              "fid_weights_sha256": fid.weight_hash, "metrics": metrics,
              "ssim_reference": "original_source_rgb", "evaluation_split": "val"}
    atomic_json(output / "metrics.json", report)
    lines = ["# Stage 4 evaluation", "", "All held-out sources; full training-bank InfoOT.", "",
             "| Direction | FID | Source SSIM (mean ± std) | Generated / real |",
             "|---|---:|---:|---:|"]
    for direction, result in metrics.items():
        ssim = result["source_ssim"]
        lines.append(f"| {direction} | {result['fid']:.4f} | {ssim['mean']:.4f} ± {ssim['std']:.4f} | "
                     f"{result['generated_count']} / {result['real_target_count']} |")
    lines += ["", "Source SSIM measures source preservation; a copied source can score highly without translating.",
              "This validation split has been used for tuning; it is not an untouched blind test.", ""]
    for direction, selected in gallery_selection.items():
        lines += [f"## {direction}: {len(selected)} inspection pairs", "",
                  f"![Source and translation](galleries/{direction}/contact_sheet.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": "complete", "output_dir": str(output), "metrics": metrics}


def write_jsonl(path, rows):
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)
