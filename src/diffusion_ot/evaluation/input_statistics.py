"""N1 appearance accessibility probes, fitted/tuned on training samples only."""
from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch

PROTOCOL = "latent_input_appearance_probe_v1"
TARGET_NAMES = [f"{space}_{stat}_{channel}" for space, channels in
                (("rgb", "rgb"), ("lab", "Lab")) for stat in ("mean", "std") for channel in channels]


def probe_options(config):
    supplied = config.get("input_statistics") or {}
    if not isinstance(supplied, dict):
        raise ValueError("input_statistics must be a mapping.")
    if not supplied.get("enabled", False):
        return None
    options = dict(enabled=True, train_samples=1024, development_samples=256, batch_size=32,
                   development_split="val", seed=20261001, tuning_fraction=.2,
                   ridge_alphas=[.0001, .001, .01, .1, 1., 10.], sensitivity_samples=8)
    if set(supplied) - set(options):
        raise ValueError("Unknown input_statistics options: " + str(sorted(set(supplied) - set(options))))
    options.update(supplied)
    for key in ("train_samples", "development_samples", "batch_size", "sensitivity_samples"):
        if type(options[key]) is not int or options[key] < (8 if key == "train_samples" else 1):
            raise ValueError(f"Invalid input_statistics.{key}.")
    if options["development_split"] != "val":
        raise ValueError("N1 uses val as development only; never tune on held-out test data.")
    if not 0 < float(options["tuning_fraction"]) < .5:
        raise ValueError("N1 tuning_fraction must be in (0, 0.5).")
    alphas = options["ridge_alphas"]
    if not isinstance(alphas, list) or not alphas or any(not math.isfinite(float(x)) or float(x) <= 0 for x in alphas):
        raise ValueError("N1 ridge_alphas must be a nonempty list of finite positive values.")
    return options


@contextmanager
def evaluation_modes(branch):
    modes = {module: module.training for module in branch.modules()}
    try:
        branch.eval()
        yield
    finally:
        for module, mode in modes.items():
            module.training = mode


def validate_cohort_ids(train_ids, development_ids):
    for name, ids in (("train", train_ids), ("development", development_ids)):
        if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"N1 {name} needs nonempty unique sample IDs.")
    if set(train_ids) & set(development_ids):
        raise ValueError("N1 training and development cohorts overlap.")


def latent_statistics(x):
    if x.ndim != 4 or x.shape[1] != 4 or not torch.isfinite(x).all():
        raise ValueError("N1 requires finite four-channel VAE latents.")
    return torch.cat((x.float().mean((2, 3)), x.float().std((2, 3), unbiased=False)), 1)


def appearance_targets(original_rgb):
    from diffusion_ot.losses.lab_swd import normalized_lab
    if original_rgb.ndim != 4 or original_rgb.shape[1] != 3 or not torch.isfinite(original_rgb).all():
        raise ValueError("N1 requires original RGB images.")
    if original_rgb.min() < 0 or original_rgb.max() > 1:
        raise ValueError("N1 original RGB must be in [0, 1].")
    lab = normalized_lab(original_rgb.float())
    return torch.cat([v for image in (original_rgb.float(), lab)
                      for v in (image.mean((2, 3)), image.std((2, 3), unbiased=False))], 1)


@torch.inference_mode()
def representations(branch, x):
    encoder = branch.encoder
    if not hasattr(encoder, "z_norm"):
        raise ValueError("N1 needs the encoder z_norm boundary.")
    captured = []
    handle = encoder.z_norm.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        codes = encoder(x)
    finally:
        handle.remove()
    z_proj = branch.semantic_transformer.z_proj
    if not isinstance(z_proj[0], torch.nn.LayerNorm):
        raise ValueError("N1 expects generator z_proj to start with LayerNorm.")
    return dict(pre_encoder_layernorm=captured[0], codes=codes,
                post_layernorm=z_proj[0](codes), post_z_proj=z_proj(codes))


@torch.inference_mode()
def input_sensitivity(branch, x):
    """Latent-only interventions; never interpret these as RGB perturbations."""
    encoder = branch.encoder
    conv = next((m for m in encoder.modules() if isinstance(m, torch.nn.Conv2d)), None)
    if conv is None:
        raise ValueError("N1 cannot locate the first encoder convolution.")
    offsets = x.new_tensor([-.3, -.1, .1, .3])[None, :, None, None]
    scales = x.new_tensor([.75, .9, 1.1, 1.25])[None, :, None, None]
    cases = dict(original=x, channel_offset=x + offsets, channel_scale=x * scales)
    observed, encoded = {}, {}
    with evaluation_modes(branch):
        for name, value in cases.items():
            seen = []
            handle = conv.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
            try:
                encoded[name] = representations(branch, value)
            finally:
                handle.remove()
            observed[name] = seen[0]
    raw = getattr(encoder, "architecture_spec", {}).get("kind") == "residual_cnn_v1"
    errors = {name: float((observed[name] - value).abs().max()) for name, value in cases.items()}
    if raw and any(error != 0 for error in errors.values()):
        raise ValueError("Residual N1 raw-stem contract failed: first convolution did not receive unmodified input.")
    changes = {}
    for name in ("channel_offset", "channel_scale"):
        changes[name] = {key: float((value.float() - encoded["original"][key].float()).square().mean().sqrt())
                         for key, value in encoded[name].items()}
        changes[name]["stem_input_rms_change"] = float((observed[name] - observed["original"]).float().square().mean().sqrt())
    return dict(raw_residual_stem=raw, first_conv_input_max_abs_error=errors, changes=changes,
                offset=offsets.flatten().tolist(), scale=scales.flatten().tolist(),
                interpretation="Latent-only sensitivity; no RGB change or normalization invariance is assumed.")


def _ridge_fit(x, y, alpha):
    x, y = x.double(), y.double()
    xm, xs = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    ym, ys = y.mean(0), y.std(0, unbiased=False).clamp_min(1e-6)
    a, b = (x - xm) / xs, (y - ym) / ys
    # Penalize mean squared error + alpha * ||coef||^2. Fit scalers on
    # precisely this training subset, never on tuning/development samples.
    if a.shape[1] <= a.shape[0]:
        matrix = a.T @ a + len(a) * alpha * torch.eye(a.shape[1], dtype=a.dtype)
        coef = torch.linalg.solve(matrix, a.T @ b)
    else:
        matrix = a @ a.T + len(a) * alpha * torch.eye(len(a), dtype=a.dtype)
        coef = a.T @ torch.linalg.solve(matrix, b)
    return dict(x_mean=xm, x_scale=xs, y_mean=ym, y_scale=ys, coef=coef, alpha=float(alpha))


def _ridge_predict(x, model):
    return ((x.double() - model["x_mean"]) / model["x_scale"] @ model["coef"]) * model["y_scale"] + model["y_mean"]


def fit_appearance_probes(train, development, options):
    """Select ridge strength on an internal training holdout; refit all train."""
    validate_cohort_ids(train["ids"], development["ids"])
    if len(train["ids"]) < 8:
        raise ValueError("N1 needs at least eight training examples for fit/tuning separation.")
    order = torch.randperm(len(train["ids"]), generator=torch.Generator().manual_seed(int(options["seed"])))
    tune_count = max(2, int(len(order) * options["tuning_fraction"]))
    tune, fit = order[:tune_count], order[tune_count:]
    targets = train["targets"].double()
    y_dev = development["targets"].double()
    def features(cohort):
        result = {name: cohort[name] for name in ("pre_encoder_layernorm", "codes", "post_layernorm", "post_z_proj")}
        result["latent_stats_only"] = cohort["latent_stats"]
        for name in ("codes", "post_layernorm", "post_z_proj"):
            result[name + "_plus_latent_stats"] = torch.cat((cohort[name], cohort["latent_stats"]), 1)
        return result
    train_features, dev_features = features(train), features(development)
    reports, models, predictions = {}, {}, {}
    for name, x in train_features.items():
        candidates = []
        for alpha in options["ridge_alphas"]:
            model = _ridge_fit(x[fit], targets[fit], float(alpha))
            error = (_ridge_predict(x[tune], model) - targets[tune]) / model["y_scale"]
            candidates.append(dict(alpha=float(alpha), train_holdout_standardized_mse=float(error.square().mean())))
        winner = min(candidates, key=lambda c: c["train_holdout_standardized_mse"])
        model = _ridge_fit(x, targets, winner["alpha"])
        predicted = _ridge_predict(dev_features[name], model)
        per_target = (predicted - y_dev).square().mean(0)
        variance = y_dev.var(0, unbiased=False)
        reports[name] = dict(alpha=winner["alpha"], tuning=candidates, feature_dimensions=x.shape[1],
                             development_rgb_mse=float(per_target[:6].mean()), development_lab_mse=float(per_target[6:].mean()),
                             per_target_mse=dict(zip(TARGET_NAMES, per_target.tolist())),
                             per_target_r2={key: float(1 - e / v) if v > 1e-12 else None
                                            for key, e, v in zip(TARGET_NAMES, per_target, variance)})
        models[name], predictions[name] = model, predicted
    report = dict(probes=reports, fit_ids=[train["ids"][i] for i in fit],
                  tuning_ids=[train["ids"][i] for i in tune], selection_cohort="training_holdout_only",
                  final_fit_cohort="all_training", target_names=TARGET_NAMES)
    report["added_statistics_mse_reduction"] = {
        name: {space: reports[name][f"development_{space}_mse"] - reports[name + "_plus_latent_stats"][f"development_{space}_mse"]
               for space in ("rgb", "lab")} for name in ("codes", "post_layernorm", "post_z_proj")}
    return report, dict(models=models, development_predictions=predictions)


def run_input_statistics_probe(branch, *, data_config_path, domain, project_root, device, dtype,
                               options, output_dir, provenance):
    from diffusion_ot.data.latent_dataset import CachedLatentDataset
    datasets = {split: CachedLatentDataset(data_config_path, domain, split=split, project_root=project_root,
                                           random_horizontal_flip=0., include_original_images=True)
                for split in ("train", options["development_split"])}
    # Check whole manifests before subset selection, not only sampled rows.
    validate_cohort_ids([r.get("sample_id") for r in datasets["train"].records],
                        [r.get("sample_id") for r in datasets[options["development_split"]].records])
    cohorts = {}
    sensitivity = None
    with evaluation_modes(branch), torch.inference_mode():
        for index, (name, split, requested) in enumerate((("train", "train", options["train_samples"]),
                ("development", options["development_split"], options["development_samples"]))):
            dataset = datasets[split]
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(int(options["seed"]) + index))[:requested]
            chunks, ids = {}, []
            for selected in indices.split(options["batch_size"]):
                rows = [dataset[int(i)] for i in selected]
                x = torch.stack([r["x0_latent"] for r in rows]).to(device=device, dtype=dtype)
                original_rgb = (torch.stack([r["encoder_image"] for r in rows]).float() + 1) / 2
                values = {**representations(branch, x), "latent_stats": latent_statistics(x),
                          "targets": appearance_targets(original_rgb)}
                ids.extend(r["sample_id"] for r in rows)
                for key, value in values.items():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f"N1 nonfinite {key}.")
                    chunks.setdefault(key, []).append(value.float().cpu())
                if sensitivity is None:
                    sensitivity = input_sensitivity(branch, x[:options["sensitivity_samples"]])
            cohorts[name] = dict(ids=ids, **{k: torch.cat(v) for k, v in chunks.items()})
    report, fitted = fit_appearance_probes(cohorts["train"], cohorts["development"], options)
    report.update(protocol=PROTOCOL, domain=domain, options=options, provenance=provenance,
                  architecture=getattr(branch, "encoder_architecture", None),
                  counts={k: len(v["ids"]) for k, v in cohorts.items()}, stem_sensitivity=sensitivity,
                  limitations=["Development data is for reporting, never fitting/scaling/alpha selection.",
                               "A probe gain measures accessible appearance information, not proof of irreversible information loss.",
                               "RGB/Lab summaries use original images; VAE channels are not RGB channels.",
                               "Codes are after encoder z_norm; the pre-norm and actual generator LayerNorm/z_proj controls are also saved."])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / "appearance_probe.pt"
    torch.save(dict(protocol=PROTOCOL, provenance=provenance, cohorts=cohorts, **fitted), tensor_path)
    report["tensors"] = str(tensor_path)
    path = output / "appearance_probe.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return str(path)
