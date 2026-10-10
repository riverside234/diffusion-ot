"""Fit/resume offline mappings. Every fit owns a new directory with raw plans."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

import torch

from .conditional import BalancedModel, calibrate_support
from .feature_bank import FeatureBank, compatible_banks, digest, file_hash, write_json, save_tensor, checked_file
from .partial import capabilities, distance, kernel_state, solver_config, solve_partial, OBJECTIVE
from .run_logging import RunLog

MODES = {"patch_global", "whole_map", "grouped_patch", "grouped_partial"}
CONFIDENCE = dict(support_calibration="fit_leave_one_out_log_density", support_quantile=.01,
                  support_log_threshold=-100., threshold=.05, mass_floor=1e-12, all_invalid_policy="error")
PROJECTION = dict(bandwidth_multiplier=1., query_chunk_size=16, selection="mean", top_k_images=None, seed=0)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Mapping configuration must be a YAML mapping.")
    if unknown := config.keys() - {"mode", "source_bank", "target_bank", "solver", "partial", "projection", "resources", "image_model", "patch_model"}:
        raise ValueError(f"Unknown mapping settings: {sorted(unknown)}")
    config = dict(config)
    mode = config.get("mode", "patch_global")
    if mode not in MODES:
        raise ValueError(f"Unknown mapping mode {mode}.")
    config["mode"] = mode
    config["solver"] = solver_config(config.get("solver"))
    projection = config.get("projection", {})
    if unknown := projection.keys() - (PROJECTION.keys() | {"confidence"}):
        raise ValueError(f"Unknown projection settings: {sorted(unknown)}")
    projection = PROJECTION | projection
    if (not math.isfinite(projection["bandwidth_multiplier"]) or projection["bandwidth_multiplier"] <= 0
            or type(projection["query_chunk_size"]) is not int or projection["query_chunk_size"] < 1
            or projection["selection"] not in {"mean", "argmax", "sample"}
            or (projection["top_k_images"] is not None and (type(projection["top_k_images"]) is not int or projection["top_k_images"] < 1))
            or type(projection["seed"]) is not int):
        raise ValueError("Invalid projection bandwidth, chunk size, selection or top_k_images.")
    if mode == "patch_global" and (projection["selection"] != "mean" or projection["top_k_images"] is not None):
        raise ValueError("Image selection requires an image router, unavailable in patch_global.")
    conf = projection.get("confidence", {})
    if unknown := conf.keys() - CONFIDENCE.keys():
        raise ValueError(f"Unknown confidence settings: {sorted(unknown)}")
    conf = CONFIDENCE | conf
    if (not 0 <= conf["threshold"] <= 1 or not 0 < conf["mass_floor"] < 1
            or not 0 <= conf["support_quantile"] <= 1 or not math.isfinite(conf["support_log_threshold"])
            or conf["all_invalid_policy"] not in {"error", "bypass"}
            or conf["support_calibration"] not in {"fixed", "disabled", "fit_leave_one_out_log_density"}):
        raise ValueError("Invalid confidence/support/all-invalid policy.")
    projection["confidence"] = conf
    config["projection"] = projection
    partial = config.get("partial", {})
    if partial.keys() - {"keep_mass", "solver"}:
        raise ValueError("Partial mode supports only fixed keep_mass and solver settings; no dustbins/sparse pairs.")
    mass = partial.get("keep_mass", .8)
    if not 0 < mass <= 1:
        raise ValueError("keep_mass must be in (0,1].")
    if partial and mode != "grouped_partial":
        raise ValueError("Partial constraints belong only to grouped_partial patch-pair fits.")
    config["partial"] = dict(keep_mass=mass, solver=solver_config(partial.get("solver"))) if mode == "grouped_partial" else {}
    resources = dict(max_working_gib=16., max_plan_gib=64.) | config.get("resources", {})
    if resources.keys() - {"max_working_gib", "max_plan_gib"} or any(not math.isfinite(v) or v <= 0 for v in resources.values()):
        raise ValueError("Invalid explicit resource limits.")
    config["resources"] = resources
    for name, modes in (("image_model", {"whole_map", "grouped_patch", "grouped_partial"}),
                        ("patch_model", {"patch_global", "grouped_patch"})):
        if config.get(name) and mode not in modes:
            raise ValueError(f"{name} cannot be used by {mode}.")
    return config


def resource_estimate(source_shape, target_shape, mode):
    n, p, d = source_shape
    m = target_shape[0]
    supports = (n + m) * p * d * 8
    image_arrays = (4 * n * m + 3 * (n * n + m * m)) * 8
    patch_arrays = (4 * n * m + 3 * (n * n + m * m)) * p * p * 8
    shared_pair_kernels = (n + m) * p * p * 8
    plans = ((n * m * p * p) if mode in {"patch_global", "grouped_patch", "grouped_partial"} else n * m) * 8
    # Every solver unconditionally keeps latest.pt as well as its final plan.
    final_plans = plans + (n * m * 8 if mode in {"grouped_patch", "grouped_partial"} else 0)
    required_storage = 2 * final_plans + (shared_pair_kernels if mode == "grouped_partial" else 0)
    working = supports + (patch_arrays if mode in {"patch_global", "grouped_patch"} else image_arrays)
    if mode == "grouped_partial":
        working += shared_pair_kernels + 24 * p * p * 8
    return dict(source_shape=list(source_shape), target_shape=list(target_shape), support_bytes=supports,
                image_arrays_bytes=image_arrays, global_patch_arrays_bytes=patch_arrays,
                shared_pair_kernels_bytes=shared_pair_kernels, plan_storage_bytes=plans,
                latest_plan_storage_bytes=final_plans, required_plan_storage_bytes=required_storage,
                required_plan_storage_gib=required_storage / 2**30,
                # Workspace/BLAS/autodiff and simultaneous cache dtype conversion allowance.
                estimated_working_gib=2 * working / 2**30, plan_storage_gib=plans / 2**30,
                pairs=n * m if mode == "grouped_partial" else 0,
                note="Exact full support. Required storage includes mandatory latest plans and pair kernels; excludes tensor/JSON metadata, logs and filesystem overhead.")


def inspect_fit(config, root):
    config = validate_config(config)
    root = Path(root)
    manifests = []
    for key in ("source_bank", "target_bank"):
        path = root / config[key] / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if digest({k: v for k, v in manifest.items() if k != "artifact_id"}) != manifest.get("artifact_id"):
            raise ValueError(f"Changed bank manifest {path}")
        manifests.append(manifest)
    source, target = manifests
    if source["split"] != "train" or target["split"] != "train" or source["representation"] != target["representation"]:
        raise ValueError("Need compatible fixed-grid train-split banks.")
    for name in ("image", "patch"):
        if config.get(f"{name}_model"):
            directory = root / config[f"{name}_model"]
            reused = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if (reused.get("status") != "complete" or name not in reused.get("models", {})
                    or reused["source_bank"]["artifact_id"] != source["artifact_id"]
                    or reused["target_bank"]["artifact_id"] != target["artifact_id"]
                    or reused["config"]["solver"] != config["solver"]):
                raise ValueError(f"Incompatible reused {name} model metadata.")
            checked_file(directory, reused["models"][name])
    r = source["representation"]
    p = r["grid"][0] * r["grid"][1]
    estimate = resource_estimate((len(source["ids"]), p, r["dim"]), (len(target["ids"]), p, r["dim"]), config["mode"])
    runtime = capabilities()
    if config["mode"] == "grouped_partial" and runtime["partial_method"] == "unavailable":
        raise RuntimeError("Log-domain partial OT requires POT>=0.9.7; install it explicitly for this experiment.")
    return dict(config=config, resources=estimate, runtime=runtime,
                consumer_mask="PDAEV2Branch condition_padding_mask: True is padding; all-invalid error by default",
                source_id=source["artifact_id"], target_id=target["artifact_id"])


def _file_entry(directory, path):
    return dict(file=Path(path).relative_to(directory).as_posix(), sha256=file_hash(path))


def _revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _read_pair_inventory(path, *, resume, log):
    """Recover only an unfinished last journal write; never mask interior corruption."""
    if not path.exists():
        return []
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    entries = []
    for i, line in enumerate(lines):
        try:
            entries.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not resume or i != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Corrupt pair inventory; only an interrupted final write can be recovered.") from None
            (log.directory / "pairs_truncated_tail.bin").write_bytes(line)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(b"".join(lines[:i]))
            temporary.replace(path)
            log.event("pair_inventory_tail_recovered", dropped_bytes=len(line))
            print("Recovered incomplete final pair inventory write; its pair will be refitted.", flush=True)
            return entries
    if content and not content.endswith(b"\n"):
        # A complete JSON entry may have reached disk just before the newline.
        with path.open("ab") as handle:
            handle.write(b"\n")
    return entries


def fit_mapping(config, *, root, output_root=None, resume=None):
    root = Path(root).resolve()
    if resume:
        directory = Path(resume).resolve()
        if not (directory / "manifest.json").is_file():
            raise FileNotFoundError(f"Resume needs an existing fit manifest: {directory}")
    else:
        parent = Path(output_root) if output_root else root / "outputs/infoot_vit"
        mode = config.get("mode", "patch_global") if isinstance(config, dict) else "invalid"
        mode = mode if mode in MODES else "invalid"
        directory = parent / f"{mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
        directory.mkdir(parents=True, exist_ok=False)
    with RunLog(directory, "fit", dict(root=root, resume=resume)) as log:
        (log.directory / "requested_config.txt").write_text(repr(config) + "\n", encoding="utf-8")
        return _fit_mapping(config, root=root, directory=directory, resume=resume, log=log)


def _fit_mapping(config, *, root, directory, resume, log):
    start = time.perf_counter()
    inspection = inspect_fit(config, root)
    log.write_json("inspection.json", inspection)
    print(json.dumps(inspection, indent=2), flush=True)
    config = inspection["config"]
    for name, cap in (("estimated_working_gib", "max_working_gib"), ("required_plan_storage_gib", "max_plan_gib")):
        if inspection["resources"][name] > config["resources"][cap]:
            raise MemoryError(f"Exact {name}={inspection['resources'][name]:.3f} exceeds {cap}. "
                              "Set an explicit larger limit or build an explicitly smaller image bank; no subsampling was applied.")
    source, target = [FeatureBank.load(root / config[key]) for key in ("source_bank", "target_bank")]
    compatible_banks(source, target)
    implementation = {name: file_hash(Path(__file__).with_name(name))
                      for name in ("infoot.py", "partial.py", "conditional.py", "fit_mapping.py")}
    fingerprint = digest(dict(config=config, source=source.artifact_id, target=target.artifact_id,
                              implementation=implementation, runtime=inspection["runtime"]))
    if resume:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["fit_fingerprint"] != fingerprint:
            raise ValueError("Resume fingerprint mismatch: configuration, bank bytes, or support order changed. Start a new fit.")
        if manifest["status"] == "complete":
            # Validate the completed artifact rather than silently accepting a stale fit.
            from .mapping import FeatureMapper
            FeatureMapper.load(directory)
            log.event("completed_fit_verified", artifact_id=manifest["artifact_id"])
            return directory
        manifest.update(status="fitting")
        manifest.pop("error", None)
    else:
        manifest = dict(schema="siglip_infoot_mapping_v2", status="fitting", fit_fingerprint=fingerprint,
            config=config, runtime=inspection["runtime"], resources=inspection["resources"], models={},
            source_bank=dict(path=os.path.relpath(source.path, directory), artifact_id=source.artifact_id),
            target_bank=dict(path=os.path.relpath(target.path, directory), artifact_id=target.artifact_id),
            source_ids=source.ids, target_ids=target.ids, representation=source.representation,
            source_domain=source.manifest["domain"], target_domain=target.manifest["domain"],
            precision=dict(fit="float64", projection="float64", output="query dtype/device"),
            solver_implementation_sha256=implementation,
            local_revision=_revision(), local_solver_sha256=file_hash(Path(__file__).with_name("infoot.py")),
            upstream_revision="352efd202f5b475dc170a8d08a99049689d5ee1a",
            distance="euclidean", bandwidth_policy="training_RMS_distance_fixed_for_all_queries",
            pair_coverage="all" if config["mode"] == "grouped_partial" else None)
    write_json(directory / "manifest.json", manifest)
    print(f"Fit directory: {directory}", flush=True)
    log.event("fit_started", fingerprint=fingerprint, resume=bool(resume), mode=config["mode"])
    prior_fit_seconds = manifest.get("fit_seconds", 0.)

    def progress(name):
        def callback(plan, report):
            path = directory / "plans" / name
            save_tensor(path / "latest.pt", dict(plan=plan.cpu(), report=report, fit_fingerprint=fingerprint))
            with (path / "iterations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(report, run_id=log.run_id, elapsed_seconds=time.perf_counter() - start), allow_nan=False) + "\n")
            log.event("solver_iteration", unit=name, **report)
        return callback

    try:
        mode = config["mode"]
        x, y = source.features.double(), target.features.double()
        units = []
        if mode != "patch_global":
            units.append(("image", x.reshape(len(x), -1), y.reshape(len(y), -1)))
        if mode in {"patch_global", "grouped_patch"}:
            units.append(("patch", x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])))
        for name, xs, ys in units:
            log.event("model_started", unit=name, source_count=len(xs), target_count=len(ys))
            if name in manifest["models"]:
                state = torch.load(checked_file(directory, manifest["models"][name]), weights_only=True)
                BalancedModel(xs, ys, state)
            else:
                reuse = config.get(f"{name}_model")
                if reuse:
                    from .mapping import FeatureMapper
                    existing = FeatureMapper.load(root / reuse)
                    model = getattr(existing, name)
                    if (model is None or existing.source.artifact_id != source.artifact_id
                            or existing.target.artifact_id != target.artifact_id or model.state["config"] != config["solver"]):
                        raise ValueError(f"Reused {name} model must have identical fit banks/settings; no silent refit.")
                    state = dict(model.state)
                    manifest.setdefault("reused_models", {})[name] = existing.manifest["artifact_id"]
                else:
                    model = BalancedModel.fit(xs, ys, config["solver"], on_step=progress(name))
                    state = model.state | dict(source_ids=source.ids, target_ids=target.ids)
                path = directory / "plans" / f"{name}.pt"
                save_tensor(path, state)
                if state["status"] != "converged":
                    raise RuntimeError(f"{name} FusedInfoOT {state['status']}; diagnostics/plan saved in {path}. "
                                       "Reassess scales or start a new configuration with a larger iteration budget.")
                manifest["models"][name] = _file_entry(directory, path)
                write_json(directory / "manifest.json", manifest)
            log.event("model_completed", unit=name, status=state["status"], last_iteration=state["history"][-1])
        if mode == "grouped_partial":
            pc = config["partial"]["solver"]
            h_projection = pc["h"] * config["projection"]["bandwidth_multiplier"]
            if "pair_kernels" in manifest:
                shared = torch.load(checked_file(directory, manifest["pair_kernels"]), weights_only=True)
            else:
                kernels_x, scales_x, support = [], [], []
                kernels_y, scales_y = [], []
                for patches in x:
                    k, scale = kernel_state(patches, pc["h"])
                    kernels_x.append(k); scales_x.append(scale)
                    support.append(calibrate_support(patches, scale, h_projection, config["projection"]["confidence"]))
                for patches in y:
                    k, scale = kernel_state(patches, pc["h"])
                    kernels_y.append(k); scales_y.append(scale)
                shared = dict(kx=torch.stack(kernels_x), ky=torch.stack(kernels_y), sx=scales_x, sy=scales_y,
                              support=support, h_projection=h_projection)
                path = directory / "plans/pair_kernels.pt"
                save_tensor(path, shared)
                manifest["pair_kernels"] = _file_entry(directory, path)
                manifest["partial_objective"] = OBJECTIVE
                write_json(directory / "manifest.json", manifest)
            index_path = directory / "pairs.jsonl"
            inventory = _read_pair_inventory(index_path, resume=bool(resume), log=log)
            done = {(entry["source_id"], entry["target_id"]): entry for entry in inventory}
            if len(done) != len(inventory):
                raise ValueError("Duplicate pair inventory entries.")
            if set(done) - {(s, t) for s in source.ids for t in target.ids}:
                raise ValueError("Pair inventory contains unknown support IDs.")
            for entry in inventory:
                checked_file(directory, entry)
            for i, sid in enumerate(source.ids):
                for j, tid in enumerate(target.ids):
                    if (sid, tid) in done:
                        continue
                    log.event("pair_started", source_id=sid, target_id=tid, completed_pairs=len(done), total_pairs=len(x)*len(y))
                    # Unique indices within this immutable support order plus an
                    # ID hash keep nested log paths usable on Windows. Identity
                    # comes from full stable IDs/fingerprints in the inventory.
                    pair_key = f"{i:06d}_{j:06d}_{digest([sid, tid])[:12]}"
                    plan, report = solve_partial(distance(x[i], y[j]), shared["kx"][i], shared["ky"][j],
                        keep_mass=config["partial"]["keep_mass"], config=pc, on_step=progress(f"pairs/{pair_key}"))
                    a, b = plan.new_full((len(x[i]),), 1 / len(x[i])), plan.new_full((len(y[j]),), 1 / len(y[j]))
                    path = directory / "plans/pairs" / f"{pair_key}.pt"
                    save_tensor(path, dict(plan=plan, a=a, b=b, r=plan.sum(1), c=plan.sum(0),
                        source_id=sid, target_id=tid, source_index=i, target_index=j,
                        fit_fingerprint=fingerprint, report=report))
                    if report["status"] != "converged":
                        raise RuntimeError(f"Pair {sid}/{tid}: {report['status']}; saved plan is not accepted as a successful fit.")
                    entry = _file_entry(directory, path) | dict(source_id=sid, target_id=tid, source_index=i, target_index=j)
                    with index_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(entry) + "\n")
                    done[sid, tid] = entry
                    log.event("pair_completed", source_id=sid, target_id=tid, completed_pairs=len(done),
                              total_pairs=len(x)*len(y), status=report["status"], last_iteration=report["history"][-1])
                    print(f"Pair {len(done)}/{len(x)*len(y)}: {sid} -> {tid}", flush=True)
            manifest["pair_inventory"] = _file_entry(directory, index_path)
            manifest["pair_count"] = len(done)
        manifest.update(status="complete", fit_seconds=prior_fit_seconds + time.perf_counter() - start)
        manifest.pop("error", None)
        manifest["artifact_id"] = digest({k: v for k, v in manifest.items() if k != "artifact_id"})
        report = summarize_fit(directory, manifest)
        write_json(directory / "fit_report.json", report)
        write_json(directory / "manifest.json", manifest)
        log.write_json("fit_report.json", report)
        log.event("fit_completed", artifact_id=manifest["artifact_id"], fit_seconds=manifest["fit_seconds"])
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error),
                        fit_seconds=prior_fit_seconds + time.perf_counter() - start)
        manifest.pop("artifact_id", None)
        write_json(directory / "manifest.json", manifest)
        raise
    return directory


def summarize_fit(directory, manifest):
    """Compact fit diagnostics; raw plans, marginals and full traces stay on disk."""
    report = dict(mode=manifest["config"]["mode"], artifact_id=manifest["artifact_id"],
                  resources=manifest["resources"], runtime=manifest["runtime"], models={})
    for name, entry in manifest["models"].items():
        state = torch.load(Path(directory) / entry["file"], weights_only=True)
        report["models"][name] = {k: state[k] for k in ("status", "cost_scale", "source_scale", "target_scale", "row_residual", "column_residual")}
        report["models"][name]["last_iteration"] = state["history"][-1]
    if manifest["config"]["mode"] == "grouped_partial":
        stats = dict(pair_count=0, fitted_mass_min=1., fitted_mass_max=0., row_cap_error_max=0., column_cap_error_max=0.)
        with (Path(directory) / "pairs.jsonl").open(encoding="utf-8") as index, (
                Path(directory) / "pair_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
            for line in index:
                entry = json.loads(line)
                state = torch.load(Path(directory) / entry["file"], weights_only=True)
                r, c = state["r"] / state["a"], state["c"] / state["b"]
                summary = dict(source_id=entry["source_id"], target_id=entry["target_id"],
                    source_retention_min=float(r.min()), source_retention_mean=float(r.mean()), source_retention_max=float(r.max()),
                    target_retention_min=float(c.min()), target_retention_mean=float(c.mean()), target_retention_max=float(c.max()),
                    source_mass_weighted_retention=float((r * state["a"]).sum()), target_mass_weighted_retention=float((c * state["b"]).sum()))
                summary.update({k: state["report"][k] for k in ("status", "mass", "mass_error", "row_cap_error", "column_cap_error", "cost_scale")})
                summary["last_iteration"] = state["report"]["history"][-1]
                handle.write(json.dumps(summary) + "\n")
                stats["pair_count"] += 1
                stats["fitted_mass_min"] = min(stats["fitted_mass_min"], summary["mass"])
                stats["fitted_mass_max"] = max(stats["fitted_mass_max"], summary["mass"])
                for key in ("row_cap_error", "column_cap_error"):
                    stats[f"{key}_max"] = max(stats[f"{key}_max"], summary[key])
        report["partial"] = dict(stats, keep_mass=manifest["config"]["partial"]["keep_mass"],
            diagnostics_file="pair_diagnostics.jsonl", confidence_note="Raw retention differs from smoothed query confidence and thresholded mask coverage.")
    return report
