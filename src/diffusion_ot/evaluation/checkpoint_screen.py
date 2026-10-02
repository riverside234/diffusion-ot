"""P1/E0: controlled checkpoint/readout sweeps and portable report comparison."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from diffusion_ot.evaluation.checkpoint_cache import CheckpointEvaluationCache, fingerprint

SCREEN_VERSION = "checkpoint_screen_v1"
DEFAULT_BANDWIDTHS = (.15, .20, .25, .35)
READOUTS = ("conditional_mean", "conditional_map", "conditional_sample")


def screen_variants(config, *, bandwidths=DEFAULT_BANDWIDTHS, finalists=None,
                    steps=None, samples=None, draws=3, bank_seeds=None):
    """Keep cohort seed fixed; vary generation draws independently of banks."""
    if draws < 1:
        raise ValueError("draws must be positive (use >=3 for categorical readout comparisons).")
    steps = list(steps if steps is not None else ([20, 40] if finalists else [20]))
    samples = int(samples if samples is not None else (64 if finalists else 16))
    if not steps or any(int(n) != n or n < 1 for n in steps) or samples < 1:
        raise ValueError("Integration steps and generated sample count must be positive integers.")
    if len(set(steps)) != len(steps):
        raise ValueError("Duplicate integration steps.")
    groups = {}
    if finalists:
        for bandwidth, readout in finalists:
            if readout not in READOUTS:
                raise ValueError(f"Unsupported finalist readout: {readout}.")
            if readout in groups.setdefault(float(bandwidth), []):
                raise ValueError("Duplicate finalist pair.")
            groups[float(bandwidth)].append(readout)
    else:
        if not bandwidths or len(set(bandwidths)) != len(bandwidths):
            raise ValueError("Bandwidths must be nonempty and unique.")
        groups = {float(h): list(READOUTS) for h in bandwidths}
    if any(not math.isfinite(h) or h <= 0 for h in groups):
        raise ValueError("Projection bandwidths must be finite and positive.")
    seed = int(config.get("seed", 20260906))
    bank_seeds = list(bank_seeds if bank_seeds is not None else [int(config.get("data", {}).get("bank_seed", seed))])
    if not bank_seeds or len(set(bank_seeds)) != len(bank_seeds):
        raise ValueError("Bank seeds must be nonempty and unique.")
    variants = []
    for bank_seed in bank_seeds:
        for bandwidth, readouts in groups.items():
            for num_steps in steps:
                for draw in range(draws):
                    cfg = deepcopy(config)
                    cfg.setdefault("matching", {})["projection_bandwidth_multiplier"] = bandwidth
                    cfg.setdefault("data", {})["bank_seed"] = int(bank_seed)
                    if samples > int(cfg["data"].get("query_samples_per_domain", 256)):
                        raise ValueError("Generated sample count exceeds the configured query cohort.")
                    cfg.setdefault("translation", {}).update(enabled=True, readouts=readouts,
                        primary_readout=readouts[0], num_steps=int(num_steps), samples_per_direction=samples,
                        seed=seed + draw * 100000, save_generation_inputs=True)
                    # Numeric P0 controls remain on; repeated UMAP fits do not help this screen.
                    cfg.setdefault("projection_audit", {})["enabled"] = True
                    cfg.setdefault("visualization", {})["enabled"] = False
                    axes = dict(bandwidth=bandwidth, readouts=readouts, steps=int(num_steps),
                                samples=samples, draw=draw, bank_seed=int(bank_seed), generation_seed=seed + draw * 100000)
                    variants.append({"id": fingerprint(axes)[:12], "axes": axes, "config": cfg})
    return variants


def run_checkpoint_screen(alignment_path, evaluation_path, checkpoints, output_dir, *,
                          weights="ema", device_cat=None, device_dog=None, **options):
    """One checkpoint's models/banks/fit stay resident across all its variants."""
    import yaml
    from diffusion_ot.integrations.hf_snapshot import load_yaml_config
    from diffusion_ot.evaluation.stage1b_eval import _run_stage1b_evaluation
    paths = [Path(path).resolve() for path in checkpoints]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide unique checkpoints, with step 0 first when available.")
    for path in [Path(alignment_path), Path(evaluation_path), *paths]:
        if not path.is_file():
            raise FileNotFoundError(f"Screen input not found: {path}")
    config = load_yaml_config(evaluation_path)
    variants = screen_variants(config, **options)
    output_dir = Path(output_dir).resolve()
    protocol_dir = output_dir / "protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        variant["config"]["output_dir"] = str(output_dir / "runs")
        path = protocol_dir / f"{variant['id']}.yaml"
        path.write_text(yaml.safe_dump(variant["config"], sort_keys=False), encoding="utf-8")
        variant["path"] = str(path)
    manifest = dict(version=SCREEN_VERSION, status="running", weights=weights,
                    alignment_config=str(Path(alignment_path).resolve()), evaluation_config=str(Path(evaluation_path).resolve()),
                    checkpoints=list(map(str, paths)), variants=[{"id": v["id"], "axes": v["axes"], "config": v["path"]} for v in variants],
                    runs=[], cache_statistics=[], note="P1 only. No input-normalization changes or P1a probes.")
    def save_manifest():
        (output_dir / "screen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    save_manifest()
    baselines = {}
    try:
        for checkpoint in paths:
            cache = CheckpointEvaluationCache()
            try:
                for variant in variants:
                    print(f"Screen {checkpoint.name}: {variant['axes']}", flush=True)
                    report = _run_stage1b_evaluation(alignment_path, variant["path"], checkpoint_path=checkpoint,
                        weights=weights, device_cat=device_cat, device_dog=device_dog,
                        require_stage1a_baseline=False, initial_baseline_report=baselines.get(variant["id"]),
                        evaluation_cache=cache)
                    if report.evaluation_protocol["checkpoint_step"] == 0:
                        baselines[variant["id"]] = report
                    manifest["runs"].append({"checkpoint": str(checkpoint), "variant": variant["id"],
                        "report": str(Path(report.output_dir) / "evaluation_report.json"),
                        "baseline_status": report.baseline_comparison["status"]})
                    save_manifest()
            finally:
                manifest["cache_statistics"].append({"checkpoint": str(checkpoint),
                    "hits": dict(cache.hits), "misses": dict(cache.misses)})
                cache.clear()
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_manifest()
    summarize_reports([Path(run["report"]) for run in manifest["runs"]], output_dir)
    return manifest


def _differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [path for key in sorted(a.keys() | b.keys())
                for path in _differences(a.get(key), b.get(key), f"{prefix}.{key}")]
    return [] if a == b else [prefix]


def bandwidth_comparability(reference, current, *, axis="bandwidth"):
    """Require identical saved protocols except the chosen experimental axis."""
    if axis not in {"bandwidth", "integration_steps"}:
        raise ValueError("Unsupported comparison axis.")
    def protocol(report):
        value = deepcopy(report.get("evaluation_protocol", {}))
        if not all(k in value for k in ("ordered_sample_ids", "checkpoint", "weights", "seed", "effective_config")):
            raise ValueError("Missing saved protocol/IDs; cannot assert a paired comparison.")
        value.pop("identifier", None)
        cfg = value["effective_config"]
        cfg.pop("output_dir", None)
        if axis == "bandwidth":
            cfg.get("matching", {}).pop("projection_bandwidth_multiplier", None)
        else:
            cfg.get("translation", {}).pop("num_steps", None)
        return value
    try:
        differences = _differences(protocol(reference), protocol(current))
    except ValueError as error:
        return {"matched": False, "differences": [str(error)]}
    return {"matched": not differences, "differences": differences}


def _paired_delta(a, b, seed=20260906):
    import torch
    delta = torch.tensor(b, dtype=torch.float64) - torch.tensor(a, dtype=torch.float64)
    if delta.ndim != 1 or not len(delta) or not torch.isfinite(delta).all():
        raise ValueError("Expected finite paired per-image errors.")
    rng = torch.Generator().manual_seed(seed)
    draws = delta[torch.randint(len(delta), (2000, len(delta)), generator=rng)].mean(1)
    return {"delta": delta.mean().item(), "ci95": draws.quantile(torch.tensor([.025, .975], dtype=draws.dtype)).tolist(),
            "images": len(delta), "bootstrap_seed": seed, "interpretation": "Current minus reference; image-paired percentile interval."}


def report_rows(report):
    from diffusion_ot.evaluation.stage1b_eval import _TRANSLATION_ROW_NAMES
    rows = []
    proto = report.get("evaluation_protocol", {})
    cfg = proto.get("effective_config", {})
    for direction, projection in report["projections"].items():
        decoded = projection.get("decoded_image_diagnostics", {})
        for readout in cfg.get("translation", {}).get("readouts", ["conditional_mean"]):
            metrics = decoded.get(_TRANSLATION_ROW_NAMES[readout], {})
            row = dict(direction=direction, readout=readout, step=proto.get("checkpoint_step"),
                bandwidth=report["generation_protocol"]["projection_bandwidth_multiplier"],
                integration_steps=cfg.get("translation", {}).get("num_steps"),
                generation_seed=cfg.get("translation", {}).get("seed", report["seed"]),
                bank_seed=cfg.get("data", {}).get("bank_seed", report["seed"]),
                samples=metrics.get("samples", cfg.get("translation", {}).get("samples_per_direction")),
                conditional_effective_targets=projection.get("mean_conditional_effective_target_count"),
                effective_target_query_count=report.get("query_sizes", {}).get(direction.split("_to_")[0]),
                conditional_raw_variance_ratio=projection.get("projected_to_target_variance_ratio"))
            for loss, entry in metrics.get("image_losses", {}).items():
                row[loss] = entry.get("loss")
            row["target_texture"] = metrics.get("diagnostics", {}).get("target_patch_swd", {}).get("distance")
            for representation, stats in projection.get("audit", {}).get("representations", {}).items():
                for group in ("conditional", "real_target_query", "same_domain_conditional"):
                    group_stats = stats.get(group, {})
                    row[f"{representation}_{group}_nn_cosine"] = group_stats.get("nn_cosine", {}).get("mean")
            row["selected_target_coverage"] = decoded.get("readout_selection", {}).get(readout, {}).get("target_coverage")
            rows.append(row)
    return rows


def summarize_reports(paths, output_dir, *, reference_bandwidth=.25):
    from diffusion_ot.evaluation.stage1b_eval import _TRANSLATION_ROW_NAMES
    paths = sorted(set(Path(path).resolve() for path in paths))
    if not paths:
        raise ValueError("No evaluation reports found.")
    reports = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    summary = dict(version=SCREEN_VERSION, reports=[], comparisons=[],
        limitations=["Aggregate texture distances have no paired-image interval.",
                    "UMAP appearance and effective target count alone do not select a winner.",
                    "Choose finalists by source fidelity, texture/anatomy, target reuse and blinded image inspection."])
    references = [(p, r) for p, r in reports if r["generation_protocol"]["projection_bandwidth_multiplier"] == reference_bandwidth]
    pairs = []
    for path, report in reports:
        summary["reports"].append({"path": str(path), "rows": report_rows(report),
            "baseline_status": report.get("baseline_comparison", {}).get("status"),
            "has_local_projection_tensors": all((path.parent / "projections" / f"{direction}.pt").is_file() for direction in report["projections"])})
        if report["generation_protocol"]["projection_bandwidth_multiplier"] == reference_bandwidth:
            continue
        candidates = [(p, r, bandwidth_comparability(r, report)) for p, r in references if p != path]
        matched = [item for item in candidates if item[2]["matched"]]
        if not candidates:
            continue
        reference_path, reference, comparison = (matched or candidates)[0]
        pairs.append((path, report, reference_path, reference, comparison, "bandwidth"))
    # Finalist runs compare integration at fixed bandwidth, readouts, IDs and noise.
    for path, report in reports:
        steps = report.get("evaluation_protocol", {}).get("effective_config", {}).get("translation", {}).get("num_steps")
        if steps != 40:
            continue
        for reference_path, reference in reports:
            previous_steps = reference.get("evaluation_protocol", {}).get("effective_config", {}).get("translation", {}).get("num_steps")
            comparison = bandwidth_comparability(reference, report, axis="integration_steps")
            if previous_steps == 20 and comparison["matched"]:
                pairs.append((path, report, reference_path, reference, comparison, "integration_steps"))
                break
    for path, report, reference_path, reference, comparison, axis in pairs:
        item = {"axis": axis, "reference": str(reference_path), "current": str(path), **comparison, "paired_image_losses": {}}
        if comparison["matched"]:
            for direction, proj in report["projections"].items():
                current = proj.get("decoded_image_diagnostics", {})
                previous = reference["projections"][direction].get("decoded_image_diagnostics", {})
                ids_match = current.get("source_query_ids") == previous.get("source_query_ids") and bool(current.get("source_query_ids"))
                if not ids_match or current.get("image_loss_seed") != previous.get("image_loss_seed"):
                    continue
                for readout in READOUTS:
                    key = _TRANSLATION_ROW_NAMES[readout]
                    for name, metric in current.get(key, {}).get("image_losses", {}).items():
                        a = previous.get(key, {}).get("image_losses", {}).get(name, {}).get("per_image_loss")
                        b = metric.get("per_image_loss")
                        if a is not None and b is not None and len(a) == len(b) == len(current["source_query_ids"]):
                            item["paired_image_losses"][f"{direction}.{readout}.{name}"] = _paired_delta(a, b)
        summary["comparisons"].append(item)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screen_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# P1 cached-checkpoint screen", "", "Metrics below come from the supplied evaluation reports; this summary does not generate images.", "",
             "| Checkpoint step | h | Integration steps | Noise seed base | Bank seed | Direction | Readout | N images | Effective targets (all queries) | Lab-SWD | RGB | Texture |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def fmt(value):
        return "—" if value is None else (f"{value:.5g}" if isinstance(value, float) else str(value))
    for report in summary["reports"]:
        for row in report["rows"]:
            keys = ("step", "bandwidth", "integration_steps", "generation_seed", "bank_seed", "direction", "readout", "samples", "conditional_effective_targets", "source_lab_swd", "coarse_rgb", "target_texture")
            lines.append("| " + " | ".join(fmt(row.get(k)) for k in keys) + " |")
    lines.extend(["", "## Comparison checks", ""])
    for item in summary["comparisons"]:
        lines.append(f"- {'Matched' if item['matched'] else 'Unmatched'} {item['axis']}: `{item['current']}` versus `{item['reference']}`.")
        if not item["matched"]:
            lines.append("  Differences: " + ", ".join(item["differences"]) + ".")
        for name, value in item["paired_image_losses"].items():
            lines.append(f"  - {name}: Δ {value['delta']:.5g}, paired 95% interval [{value['ci95'][0]:.5g}, {value['ci95'][1]:.5g}], n={value['images']}.")
    lines.extend(["", "## Scope", "", *[f"- {x}" for x in summary["limitations"]],
        "- Full protocol, decoder-space controls, artifact availability and comparison details are in `screen_summary.json`.", ""])
    (output_dir / "screen_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
