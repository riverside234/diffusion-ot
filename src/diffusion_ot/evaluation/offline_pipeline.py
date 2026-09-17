"""Stage 2-3: frozen bank construction and global InfoOT fitting."""
from __future__ import annotations

from dataclasses import asdict
import gc
import json
from pathlib import Path

import torch

from diffusion_ot.evaluation import stage1b_eval as legacy
from diffusion_ot.evaluation.full_infoot import FullFitSettings, feasibility, fit_global, reference_rms
from diffusion_ot.evaluation.offline_artifacts import (
    atomic_json, atomic_torch, bind_run, file_hash, fingerprint, read_torch, verify_files,
)
from diffusion_ot.integrations.hf_snapshot import load_yaml_config, resolve_project_local_path
from diffusion_ot.losses.infoot import uniform_marginals

PROTOCOL = "full_bank_fixed_rms_v1"
DOMAINS = ("cat", "dog")


def local(root, value):
    return resolve_project_local_path(value, root)


def release_models():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def domain_data_path(alignment, root, domain):
    training = load_yaml_config(local(root, alignment["stage1a"][domain]["config"]))
    return local(root, training["data_config"])


def complete_dataset(alignment, root, domain, split):
    """The latent manifest alone can be incomplete: check canonical IDs too."""
    from diffusion_ot.data.latent_dataset import CachedLatentDataset, split_manifest_path
    from diffusion_ot.data.manifests import read_jsonl
    path = domain_data_path(alignment, root, domain)
    dataset = CachedLatentDataset(path, domain, split=split, project_root=root,
                                  validate_exists=True, random_horizontal_flip=0)
    canonical = read_jsonl(split_manifest_path(load_yaml_config(path), domain, split, root))
    expected = [str(row["sample_id"]) for row in canonical]
    actual = [str(row["sample_id"]) for row in dataset.records]
    if (len(actual) < 2 or len(actual) != len(set(actual)) or len(expected) != len(set(expected))
            or set(actual) != set(expected)):
        raise ValueError(f"Incomplete/duplicate {domain}_{split} latent coverage: "
                         f"{len(actual)} cached versus {len(expected)} canonical IDs")
    if any(str(row.get("domain")) != domain for row in dataset.records):
        raise ValueError("Latent manifest domain mismatch")
    return dataset


def dependency_hashes(alignment, root):
    """Fingerprint external model/data inputs needed to reconstruct the bundle.

    Large frozen base weights remain external; their actual bytes are verified.
    """
    from diffusion_ot.data.latent_dataset import latent_manifest_path, split_manifest_path
    paths = set()
    for domain in DOMAINS:
        branch = alignment["stage1a"][domain]
        train_path = local(root, branch["config"])
        train = load_yaml_config(train_path)
        model_path, data_path = local(root, train["model_config"]), local(root, train["data_config"])
        model, data = load_yaml_config(model_path), load_yaml_config(data_path)
        pretrained_path = local(root, model["pretrained"])
        pretrained = load_yaml_config(pretrained_path)
        paths.update((train_path, model_path, data_path, pretrained_path, local(root, branch["checkpoint"])))
        snapshot = local(root, pretrained["local_dir"])
        # load_sit_pipeline rewrites this verification receipt on every load.
        # It is not model state and must not invalidate the frozen-input identity.
        receipt = local(root, pretrained["metadata_file"]) if pretrained.get("metadata_file") else snapshot / "snapshot_report.json"
        weights = [p for p in snapshot.rglob("*") if p.is_file()
                   and p != receipt and p.name != "snapshot_report.json"
                   and p.suffix in {".json", ".bin", ".safetensors", ".pt", ".py"}]
        if not weights:
            raise FileNotFoundError(f"Pretrained snapshot has no model files: {snapshot}")
        paths.update(weights)
        for split in ("train", "val"):
            paths.add(split_manifest_path(data, domain, split, root))
        latent_manifest = latent_manifest_path(data, root)
        if latent_manifest.is_file():
            paths.add(latent_manifest)
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(paths)}


def implementation_hash():
    # Changes to numerical/model-loading code cannot silently reuse old results.
    base = Path(__file__).parents[1]
    names = ["evaluation/full_infoot.py", "evaluation/offline_pipeline.py",
             "evaluation/stage1b_eval.py", "evaluation/stage1a_eval.py",
             "models/generator_adaptation.py", "models/matching_head.py", "models/pdae_sit.py",
             "losses/infoot.py", "losses/semantic_prior.py", "integrations/sit_diffusers.py",
             "data/ground_truth.py", "data/latent_cache.py", "data/latent_dataset.py"]
    return fingerprint({name: file_hash(base / name) for name in names})


def build_bank(alignment, root, domain, split, checkpoint, weights, identity, device, batch_size):
    dataset = complete_dataset(alignment, root, domain, split)
    context = legacy._load_domain_context(alignment, root, domain, checkpoint_path=checkpoint,
                                         joint_weights=weights, device_override=device)
    try:
        context.branch.requires_grad_(False)
        context.vae.requires_grad_(False)
        bank = legacy.build_latent_bank(
            context.branch.encoder, dataset, domain=domain, split=split, count=None,
            seed=0, batch_size=batch_size, device=context.device, dtype=context.dtype,
            checkpoint_id=identity, matching_head=context.matching_head)
    finally:
        del context
        release_models()
    return bank


def load_bundle(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or not manifest["solver"].get("converged"):
        raise ValueError("Bundle is unsupported or its fit is incomplete")
    scientific_inputs = {k: v for k, v in manifest.items() if k not in {"identity", "scales", "solver", "files"}}
    if fingerprint(scientific_inputs) != manifest["identity"]:
        raise ValueError("Bundle protocol fingerprint mismatch")
    if manifest["implementation"] != implementation_hash():
        raise ValueError("Model/InfoOT implementation changed since this bundle was built; rerun Stage 2-3")
    verify_files(path, manifest["files"])
    calibration = json.loads((path / "calibration.json").read_text(encoding="utf-8"))
    if calibration != {"protocol": PROTOCOL, "scales": manifest["scales"]}:
        raise ValueError("Bundle calibration mismatch")
    banks = {d: legacy.load_latent_bank(path / f"banks/{d}.pt") for d in DOMAINS}
    for domain, bank in banks.items():
        if (bank.checkpoint_id != manifest["identity"] or bank.domain != domain or bank.split != "train"
                or bank.sample_ids != manifest["train_ids"][domain]):
            raise ValueError("Bundle bank identity/order mismatch")
    transport = read_torch(path / "transport.pt")
    plan = transport["coupling"]
    if plan.shape != (len(banks["cat"].sample_ids), len(banks["dog"].sample_ids)):
        raise ValueError("Bundle coupling size mismatch")
    settings = FullFitSettings(**manifest["fit_settings"])
    a, b = uniform_marginals(*plan.shape, device=plan.device, dtype=plan.dtype)
    if not torch.equal(transport["a"], a) or not torch.equal(transport["b"], b):
        raise ValueError("Bundle marginals disagree with the full uniform-mass protocol")
    if not feasibility(plan, a, b, settings)["feasible"]:
        raise ValueError("Exported coupling fails feasibility checks")
    return manifest, banks, plan


def run_stage23(config_path, checkpoint_path, *, root, output_dir=None, weights="ema",
               device=None, resume=False, max_new_iterations=None):
    root = Path(root).resolve()
    torch.set_float32_matmul_precision("highest")
    config = load_yaml_config(config_path)
    alignment = load_yaml_config(local(root, config["alignment_config"]))
    if weights not in {"raw", "ema"}:
        raise ValueError("weights must be raw or ema")
    if config.get("reference_split") != "train" or config.get("reference_samples_per_domain") != "all":
        raise ValueError("Stage 2-3 offline alignment requires all training references")
    settings = FullFitSettings(**config["fit"])
    settings.validate()
    projection_bandwidth = float(config["projection_bandwidth"])
    if not 0 < projection_bandwidth < float("inf"):
        raise ValueError("projection_bandwidth must be positive and finite")
    batch_size = int(config["encoding_batch_size"])
    if batch_size < 1:
        raise ValueError("encoding_batch_size must be positive")
    checkpoint_path = local(root, checkpoint_path)
    checkpoint = legacy._load_joint_checkpoint(checkpoint_path)
    if checkpoint.get("format_version") != 4:
        raise ValueError("This offline pipeline requires a Stage 1B format-4 paired checkpoint")
    from diffusion_ot.losses.semantic_prior import load_semantic_prior
    prior = load_semantic_prior({**alignment, "infoot": {"variant": "fused"}}, root)
    print("Verifying complete manifests and model fingerprints...", flush=True)
    train_ids, validation_ids = {}, {}
    for domain in DOMAINS:
        for split, dest in (("train", train_ids), ("val", validation_ids)):
            dest[domain] = [str(r["sample_id"]) for r in complete_dataset(alignment, root, domain, split).records]
        if set(train_ids[domain]) & set(validation_ids[domain]):
            raise ValueError("Training references overlap held-out evaluation IDs")
    dependencies = dependency_hashes(alignment, root)
    dtype_name = config.get("solver_dtype", "float32")
    if dtype_name not in {"float32", "float64"}:
        raise ValueError("solver_dtype must be float32 or float64")
    protocol = {"protocol": PROTOCOL, "checkpoint_sha256": file_hash(checkpoint_path),
                "weights": weights, "alignment": alignment, "dependencies": dependencies,
                "train_ids": train_ids, "validation_ids": validation_ids,
                "prior_fingerprint": prior.fingerprint, "fit_settings": asdict(settings),
                "projection_bandwidth": projection_bandwidth, "solver_dtype": dtype_name,
                "implementation": implementation_hash(), "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda}
    identity = fingerprint(protocol)
    output = local(root, output_dir or config["output_dir"])
    bind_run(output, identity, resume=resume)
    if (output / "bundle.json").exists():
        load_bundle(output)
        return {"status": "complete", "output_dir": str(output)}
    paired_path = output / "models/paired_state.pt"
    paired_receipt = paired_path.with_suffix(".json")
    if paired_path.exists() and paired_receipt.exists():
        if json.loads(paired_receipt.read_text(encoding="utf-8")) != {"identity": identity, "sha256": file_hash(paired_path)}:
            raise ValueError("Cached paired model state changed")
    else:
        keys = ("stage", "step", "format_version", "config", "stage1a_provenance", "encoders",
                "encoder_ema", "matching_heads", "matching_head_ema", "generators", "generator_ema")
        atomic_torch(paired_path, {key: checkpoint[key] for key in keys if key in checkpoint})
        atomic_json(paired_receipt, {"identity": identity, "sha256": file_hash(paired_path)})
    del checkpoint
    banks = {}
    for domain in DOMAINS:
        path = output / f"banks/{domain}.pt"
        receipt = path.with_suffix(".json")
        if path.exists() and receipt.exists():
            expected = json.loads(receipt.read_text(encoding="utf-8"))
            if expected != {"identity": identity, "sha256": file_hash(path)}:
                raise ValueError("Cached bank changed")
            bank = legacy.load_latent_bank(path)
        else:
            print(f"Encoding all {len(train_ids[domain])} training {domain} images...", flush=True)
            bank = build_bank(alignment, root, domain, "train", paired_path, weights, identity,
                              device or config.get("encoding_device", "cuda:0"), batch_size)
            legacy.save_latent_bank(path, bank)
            atomic_json(receipt, {"identity": identity, "sha256": file_hash(path)})
        if bank.sample_ids != train_ids[domain] or bank.checkpoint_id != identity:
            raise ValueError("Bank does not cover the complete ordered training manifest")
        banks[domain] = bank
    scales = {d: reference_rms(banks[d].matching_features) for d in DOMAINS}
    dtype = getattr(torch, dtype_name)
    solver_device = device or config["solver_device"]
    x, y = [banks[d].matching_features.to(device=solver_device, dtype=dtype) for d in DOMAINS]
    descriptors = [prior.lookup(train_ids[d], d, "train").to(solver_device) for d in DOMAINS]
    cost = torch.empty((len(x), len(y)), device=solver_device, dtype=dtype)
    for start in range(0, len(x), settings.block_size):
        cost[start:start+settings.block_size] = prior.cost(descriptors[0][start:start+settings.block_size], descriptors[1])
    del descriptors, prior
    print(f"Global InfoOT: {len(x)} x {len(y)}; one plan matrix is {cost.numel()*cost.element_size()/2**20:.1f} MiB", flush=True)
    if x.is_cuda:
        torch.cuda.reset_peak_memory_stats(x.device)
    plan, report = fit_global(x, y, cost, scales=(scales["cat"], scales["dog"]), settings=settings,
                              output=output, identity=identity, resume=resume,
                              max_new_iterations=max_new_iterations)
    atomic_json(output / "checks.json", report)
    if plan is None:
        status = "iteration_limit" if report["iteration"] >= settings.outer_iterations else "preflight_complete"
        return {"status": status, "output_dir": str(output), "solver": report}
    a, b = uniform_marginals(*plan.shape, device="cpu", dtype=plan.dtype)
    atomic_torch(output / "transport.pt", {"coupling": plan.cpu(), "a": a, "b": b})
    atomic_json(output / "calibration.json", {"protocol": PROTOCOL, "scales": scales})
    files = ["models/paired_state.pt", "banks/cat.pt", "banks/dog.pt", "transport.pt", "calibration.json"]
    manifest = {**protocol, "identity": identity, "scales": scales, "solver": report,
                "files": {p: file_hash(output / p) for p in files}}
    atomic_json(output / "bundle.json", manifest)
    return {"status": "complete", "output_dir": str(output), "solver": report}
