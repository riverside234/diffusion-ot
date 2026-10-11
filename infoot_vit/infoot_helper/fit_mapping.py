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
from .sampling import sample_ids, sampling_record
from .storage import VERSION as STORAGE_VERSION, store_plan_state, validate_plan, store_kernels, load_kernels
from .pair_selection import select_pairs, selection_edges
from .device import resolve_device, move
from .plan_diagnostics import patch_plan_concentration

MODES = {"patch_global", "whole_map", "grouped_patch", "grouped_partial"}
CONFIDENCE = dict(support_calibration="fit_leave_one_out_log_density", support_quantile=.01,
                  support_log_threshold=-100., threshold=.05, mass_floor=1e-12, all_invalid_policy="error")
PROJECTION = dict(bandwidth_multiplier=1., query_chunk_size=16, selection="mean", top_k_images=None, seed=0)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Mapping configuration must be a YAML mapping.")
    if unknown := config.keys() - {"mode", "source_bank", "target_bank", "solver", "partial", "projection", "resources", "image_model", "patch_model", "sampling", "fit_pair_top_k", "device", "pair_batch_size", "pair_checkpoint_every"}:
        raise ValueError(f"Unknown mapping settings: {sorted(unknown)}")
    config = dict(config)
    # Legacy artifacts/config dictionaries without a device retain CPU behavior.
    # All current experiment YAMLs explicitly select CUDA.
    if "device" in config:
        resolve_device(config["device"],check=False)
    mode = config.get("mode", "patch_global")
    if mode not in MODES:
        raise ValueError(f"Unknown mapping mode {mode}.")
    config["mode"] = mode
    if mode == "grouped_partial":
        sampling = dict(images_per_domain=None, seed=42) | config.get("sampling", {})
        if (sampling.keys() - {"images_per_domain", "seed"} or type(sampling["seed"]) is not int
                or (sampling["images_per_domain"] is not None and
                    (type(sampling["images_per_domain"]) is not int or sampling["images_per_domain"] < 1))):
            raise ValueError("Invalid grouped_partial training sampling settings.")
        config["sampling"] = sampling
        k = config.get("fit_pair_top_k")
        if k is not None and (type(k) is not int or k < 1):
            raise ValueError("fit_pair_top_k must be a positive integer or null (full pairs).")
        config["fit_pair_top_k"] = k
        # Missing/null preserves the serial POT reference for old configs.
        size = config.get("pair_batch_size")
        if size is not None and (type(size) is not int or size < 1):
            raise ValueError("pair_batch_size must be positive, or null for the serial solver.")
        config["pair_batch_size"] = size
        every = config.get("pair_checkpoint_every", 10)
        if type(every) is not int or every < 1:
            raise ValueError("pair_checkpoint_every must be a positive integer.")
        config["pair_checkpoint_every"] = every
    elif any(k in config for k in ("sampling", "fit_pair_top_k", "pair_batch_size", "pair_checkpoint_every")):
        raise ValueError("Training sampling and pair fitting settings apply only to grouped_partial.")
    config["solver"] = solver_config(config.get("solver"))
    projection = config.get("projection", {})
    if unknown := projection.keys() - (PROJECTION.keys() | {"confidence", "patch_bandwidth"}):
        raise ValueError(f"Unknown projection settings: {sorted(unknown)}")
    projection = PROJECTION | projection
    if "patch_bandwidth" in projection:
        h_patch = projection["patch_bandwidth"]
        if (mode != "grouped_partial" or isinstance(h_patch, bool)
                or not isinstance(h_patch, (int, float)) or not math.isfinite(h_patch) or h_patch <= 0):
            raise ValueError("patch_bandwidth must be finite and positive, and applies only to dense grouped_partial.")
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


def resource_estimate(source_shape, target_shape, mode, fit_pair_top_k=None, pair_batch_size=None):
    n, p, d = source_shape
    m = target_shape[0]
    supports = (n + m) * p * d * 8
    image_arrays = (4 * n * m + 3 * (n * n + m * m)) * 8
    patch_arrays = (4 * n * m + 3 * (n * n + m * m)) * p * p * 8
    shared_pair_kernels = (n + m) * p * p * 8
    plans = ((n * m * p * p) if mode in {"patch_global", "grouped_patch", "grouped_partial"} else n * m) * 8
    # Baseline modes retain latest.pt; grouped_partial overrides this below.
    final_plans = plans + (n * m * 8 if mode in {"grouped_patch", "grouped_partial"} else 0)
    required_storage = 2 * final_plans + (shared_pair_kernels if mode == "grouped_partial" else 0)
    pairs = n * (m if fit_pair_top_k is None else min(m, fit_pair_top_k)) if mode == "grouped_partial" else 0
    active_pairs = min(pair_batch_size or 1, pairs) if pairs else 0
    latest_bytes = final_plans
    kernel_disk = shared_pair_kernels
    if mode == "grouped_partial":
        plans = pairs * p * p * 4
        final_plans = plans + n * m * 4
        kernel_disk = shared_pair_kernels // 2
        # Active batch checkpoints + atomic copies; failed checkpoints are extra.
        latest_bytes = 2 * max(n * m, active_pairs * p * p) * 4
        required_storage = final_plans + latest_bytes + 2 * kernel_disk
    working = supports + (patch_arrays if mode in {"patch_global", "grouped_patch"} else image_arrays)
    if mode == "grouped_partial":
        working += shared_pair_kernels + (32 if pair_batch_size else 24) * active_pairs * p * p * 8
        if pair_batch_size:
            working += 2 * active_pairs * p * d * 8  # Gathered feature pairs.
    return dict(source_shape=list(source_shape), target_shape=list(target_shape), support_bytes=supports,
                image_arrays_bytes=image_arrays, global_patch_arrays_bytes=patch_arrays,
                shared_pair_kernels_bytes=shared_pair_kernels, plan_storage_bytes=plans,
                latest_plan_storage_bytes=latest_bytes, final_plan_storage_bytes=final_plans,
                kernel_storage_bytes=kernel_disk if mode == "grouped_partial" else 0,
                storage_dtype="float32" if mode == "grouped_partial" else "float64",
                required_plan_storage_bytes=required_storage,
                required_plan_storage_gib=required_storage / 2**30,
                # Workspace/BLAS/autodiff and simultaneous cache dtype conversion allowance.
                estimated_working_gib=2 * working / 2**30, plan_storage_gib=plans / 2**30,
                pairs=pairs, full_pairs=n*m if mode == "grouped_partial" else 0,
                fit_pair_top_k=fit_pair_top_k, pair_batch_size=pair_batch_size, active_pairs=active_pairs,
                note="Grouped partial: float32 final plans/kernels + active-batch checkpoints/temps and kernel temp; completed latest files removed. Other methods retain float64 dense behavior. Excludes logs, failed checkpoints, tensor/JSON/filesystem overhead; memory is an estimate, not a measurement.")


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
    selected = {}
    for name, bank in (("source", source), ("target", target)):
        sampling = config.get("sampling", {})
        ids = sample_ids(bank["ids"], sampling.get("images_per_domain"), sampling.get("seed", 42))
        selected[name] = sampling_record(bank["ids"], ids, sampling.get("seed", 42))
    for name in ("image", "patch"):
        if config.get(f"{name}_model"):
            directory = root / config[f"{name}_model"]
            reused = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if (reused.get("status") != "complete" or name not in reused.get("models", {})
                    or reused["source_bank"]["artifact_id"] != source["artifact_id"]
                    or reused["target_bank"]["artifact_id"] != target["artifact_id"]
                    or reused["source_ids"] != selected["source"]["ordered_ids"]
                    or reused["target_ids"] != selected["target"]["ordered_ids"]
                    or reused["config"]["solver"] != config["solver"]):
                raise ValueError(f"Incompatible reused {name} model metadata.")
            checked_file(directory, reused["models"][name])
    r = source["representation"]
    p = r["grid"][0] * r["grid"][1]
    estimate = resource_estimate((selected["source"]["selected_count"], p, r["dim"]),
                                (selected["target"]["selected_count"], p, r["dim"]), config["mode"], config.get("fit_pair_top_k"), config.get("pair_batch_size"))
    runtime = capabilities()
    if config["mode"] == "grouped_partial" and runtime["partial_method"] == "unavailable":
        raise RuntimeError("Log-domain partial OT requires POT>=0.9.7; install it explicitly for this experiment.")
    return dict(config=config, sampling=selected, resources=estimate, runtime=runtime,
                consumer_mask="PDAEV2Branch condition_padding_mask: True is padding; all-invalid error by default",
                source_id=source["artifact_id"], target_id=target["artifact_id"])


def _file_entry(directory, path):
    return dict(file=Path(path).relative_to(directory).as_posix(), sha256=file_hash(path))


def display_inspection(report):
    """Keep terminal output compact; full ordered IDs are saved in artifacts/logs."""
    return dict(report, sampling={name: {**{k: v for k, v in row.items() if k != "ordered_ids"},
        "ordered_ids_sha256": digest(row["ordered_ids"]), "first_ids": row["ordered_ids"][:5]}
        for name, row in report["sampling"].items()})


def _verified_save(directory, path, state, validator):
    save_tensor(path, state)
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())
    entry = _file_entry(directory, path)
    loaded = torch.load(checked_file(directory, entry), map_location="cpu", weights_only=True)
    validator(loaded)
    _sync_directory(path.parent)
    return entry, loaded


def _sync_directory(path):
    # Persist new/renamed entries before deleting the redundant checkpoint.
    # Windows exposes file fsync, but not POSIX directory file descriptors.
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _registered_manifest(directory, manifest):
    write_json(directory / "manifest.json", manifest)
    if manifest["config"]["mode"] == "grouped_partial":
        with (directory / "manifest.json").open("r+b") as handle:
            os.fsync(handle.fileno())
        _sync_directory(directory)


def _cleanup_latest(directory, entry, log):
    # Called only AFTER the final file passed validation and durable registration.
    final = checked_file(directory, entry)
    latest = final.with_suffix("") / "latest.pt"
    if not latest.resolve().is_relative_to(Path(directory).resolve()):
        raise ValueError("Checkpoint cleanup escaped the fit directory.")
    if latest.exists():
        latest.unlink()
        log.event("redundant_checkpoint_removed", file=str(latest.relative_to(directory)))


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
    print(json.dumps(display_inspection(inspection), indent=2), flush=True)
    config = inspection["config"]
    device = resolve_device(config.get("device", "cpu"))
    for name, cap in (("estimated_working_gib", "max_working_gib"), ("required_plan_storage_gib", "max_plan_gib")):
        if inspection["resources"][name] > config["resources"][cap]:
            raise MemoryError(f"Exact {name}={inspection['resources'][name]:.3f} exceeds {cap}. "
                              "Set an explicit larger limit or configure a smaller training population; no automatic reduction was applied.")
    source, target = [FeatureBank.load(root / config[key]) for key in ("source_bank", "target_bank")]
    compatible_banks(source, target)
    compact = config["mode"] == "grouped_partial"
    source = source.subset(inspection["sampling"]["source"]["ordered_ids"])
    target = target.subset(inspection["sampling"]["target"]["ordered_ids"])
    implementation = {name: file_hash(Path(__file__).with_name(name))
                      for name in ("infoot.py", "partial.py", "conditional.py", "fit_mapping.py", "storage.py", "sampling.py", "pair_selection.py", "plan_diagnostics.py")}
    if config.get("pair_batch_size"):
        implementation.update({name: file_hash(Path(__file__).with_name(name))
                               for name in ("partial_batch.py", "pair_batch_fit.py")})
    fingerprint = digest(dict(config=config, source=source.artifact_id, target=target.artifact_id,
                              source_ids=source.ids, target_ids=target.ids, storage=STORAGE_VERSION if compact else "float64",
                              implementation=implementation, runtime=inspection["runtime"]))
    if resume:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["fit_fingerprint"] != fingerprint:
            raise ValueError("Resume fingerprint mismatch: configuration, bank bytes, or support order changed. Start a new fit.")
        if manifest["status"] == "complete":
            # Validate the completed artifact rather than silently accepting a stale fit.
            from .mapping import FeatureMapper
            completed = FeatureMapper.load(directory)
            if compact:
                for entry in list(manifest["models"].values()) + list(completed.pairs.values()):
                    _cleanup_latest(directory, entry, log)
            log.event("completed_fit_verified", artifact_id=manifest["artifact_id"])
            return directory
        manifest.update(status="fitting")
        manifest.pop("error", None)
    else:
        manifest = dict(schema="siglip_infoot_mapping_v3" if compact else "siglip_infoot_mapping_v2", status="fitting", fit_fingerprint=fingerprint,
            config=config, runtime=inspection["runtime"], resources=inspection["resources"], models={},
            source_bank=dict(path=os.path.relpath(source.path, directory), artifact_id=source.artifact_id),
            target_bank=dict(path=os.path.relpath(target.path, directory), artifact_id=target.artifact_id),
            source_ids=source.ids, target_ids=target.ids, representation=source.representation,
            source_domain=source.manifest["domain"], target_domain=target.manifest["domain"],
            precision=dict(fit="float64", projection="float64", output="query dtype/device",
                           plans="float32" if compact else "float64", kernels="float32" if compact else "float64",
                           marginals="float64", storage_version=STORAGE_VERSION if compact else None),
            sampling=inspection["sampling"],
            solver_implementation_sha256=implementation,
            local_revision=_revision(), local_solver_sha256=file_hash(Path(__file__).with_name("infoot.py")),
            upstream_revision="352efd202f5b475dc170a8d08a99049689d5ee1a",
            distance="euclidean", bandwidth_policy="training_RMS_distance_fixed_for_all_queries",
            pair_coverage=("all" if config["fit_pair_top_k"] is None or config["fit_pair_top_k"] >= len(target.ids)
                           else "router_top_k") if compact else None)
    write_json(directory / "manifest.json", manifest)
    print(f"Fit directory: {directory}", flush=True)
    log.event("fit_started", fingerprint=fingerprint, resume=bool(resume), mode=config["mode"])
    prior_fit_seconds = manifest.get("fit_seconds", 0.)

    def progress(name):
        def callback(plan, report):
            path = directory / "plans" / name
            snapshot = dict(plan=plan.cpu(), report=report, fit_fingerprint=fingerprint)
            save_tensor(path / "latest.pt", store_plan_state(snapshot) if compact else snapshot)
            with (path / "iterations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(report, run_id=log.run_id, elapsed_seconds=time.perf_counter() - start), allow_nan=False) + "\n")
            log.event("solver_iteration", unit=name, **report)
            if name in {"image", "patch"} and (report["iteration"] == 1 or report["iteration"] % 25 == 0
                                               or report["status"] != "running"):
                print(f"{name} FusedInfoOT {report['iteration']}/{config['solver']['max_outer_steps']}: "
                      f"objective={report['objective']:.8g}, cost={report['cost']:.8g}, "
                      f"mi_term={report['mi_term']:.8g}, entropy_term={report['entropy_term']:.8g}, "
                      f"lam={config['solver']['lam']:.6g}, reg={config['solver']['reg']:.6g}, "
                      f"delta={report['plan_delta_l1']:.3g}, "
                      f"effective_targets={report['mean_row_effective_targets']:.1f}, status={report['status']}", flush=True)
        return callback

    try:
        mode = config["mode"]
        x, y = source.features.to(device=device,dtype=torch.float64), target.features.to(device=device,dtype=torch.float64)
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
                    existing = FeatureMapper.load(root / reuse,device=device)
                    model = getattr(existing, name)
                    if (model is None or existing.source.artifact_id != source.artifact_id
                            or existing.target.artifact_id != target.artifact_id or existing.source.ids != source.ids
                            or existing.target.ids != target.ids or model.state["config"] != config["solver"]):
                        raise ValueError(f"Reused {name} model must have identical fit banks/settings; no silent refit.")
                    state = dict(model.state)
                    if not compact and state.get("storage"):
                        raise ValueError("Float32 grouped_partial routers cannot change a float64 control fit through model reuse; fit the control independently.")
                    manifest.setdefault("reused_models", {})[name] = existing.manifest["artifact_id"]
                else:
                    model = BalancedModel.fit(xs, ys, config["solver"], on_step=progress(name))
                    state = model.state | dict(source_ids=source.ids, target_ids=target.ids)
                report = {key: state[key] for key in ("status", "config", "solver", "cost_scale", "source_scale",
                    "target_scale", "source_count", "target_count", "kernel_diagnostics", "history",
                    "row_residual", "column_residual") if key in state}
                # Keep a readable report even if convergence/quantization validation fails.
                write_json(directory / f"{name}_report.json", report)
                log.write_json(f"{name}_report.json", report)
                path = directory / "plans" / f"{name}.pt"
                if compact:
                    # Solver has finished in double. Only disk state is quantized.
                    state = state if state.get("storage") else store_plan_state(state)
                    entry, state = _verified_save(directory, path, state, lambda saved: BalancedModel(xs, ys, saved))
                else:
                    save_tensor(path, state)
                    entry = _file_entry(directory, path)
                if state["status"] != "converged":
                    last = state["history"][-1]
                    stage = " Patch-pair fitting has not started." if name == "image" and mode == "grouped_partial" else ""
                    raise RuntimeError(f"{name} FusedInfoOT {state['status']} after {last['iteration']} outer iterations: "
                        f"plan_delta_l1={last['plan_delta_l1']:.3g}, tolerance={config['solver']['outer_tolerance']:.3g}, "
                        f"inner_error={last['inner']['error']:.3g}.{stage} "
                        f"Diagnostics: {directory / (name + '_report.json')}; plan: {path}. "
                        "For a budget stop, increase solver.max_outer_steps (--max-outer-steps) or reassess solver.h/reg/lam; "
                        "partial.solver and confidence.threshold do not control the image router. "
                        "Changed settings require a new fit, without --resume.")
                manifest["models"][name] = entry
                _registered_manifest(directory, manifest)
            if compact:
                _cleanup_latest(directory, manifest["models"][name], log)
            log.event("model_completed", unit=name, status=state["status"], last_iteration=state["history"][-1])
        if mode == "grouped_partial":
            multiplier = config["projection"]["bandwidth_multiplier"]
            router = BalancedModel(x.reshape(len(x), -1), y.reshape(len(y), -1), state, multiplier)
            if "pair_selection" in manifest:
                selection = json.loads(checked_file(directory, manifest["pair_selection"]).read_text(encoding="utf-8"))
            else:
                selection = select_pairs(router, source.ids, target.ids, config["fit_pair_top_k"])
                selection["router_sha256"] = manifest["models"]["image"]["sha256"]
                selection["projection_bandwidth_multiplier"] = multiplier
                selection["projection_h"] = router.h
                selection_path = directory / "pair_selection.json"
                write_json(selection_path, selection)
                manifest["pair_selection"] = _file_entry(directory, selection_path)
                _registered_manifest(directory, manifest)
            if selection["router_sha256"] != manifest["models"]["image"]["sha256"]:
                raise ValueError("Pair selection belongs to another router.")
            selected_pairs = selection_edges(selection, source.ids, target.ids, config["fit_pair_top_k"])
            log.event("pair_selection_loaded", selected_pairs=len(selected_pairs), full_pairs=len(x)*len(y),
                      fit_pair_top_k=config["fit_pair_top_k"])
            retained = torch.tensor([r["retained_probability"] for r in selection["rows"]], dtype=torch.float64)
            routing = dict(projection_h=router.h, projection_bandwidth_multiplier=multiplier,
                effective_k=selection["effective_k"], mean_retained_probability=float(retained.mean()),
                median_retained_probability=float(retained.median()), min_retained_probability=float(retained.min()),
                max_retained_probability=float(retained.max()), mean_discarded_probability=float(1-retained.mean()),
                interpretation="Training-source conditional routing before retained-edge renormalization; not partial OT rejection.")
            write_json(directory / "pair_selection_report.json", routing)
            log.write_json("pair_selection_report.json", routing)
            print(f"Top-{selection['effective_k']} router selection: mean retained mass={retained.mean():.2%}, "
                  f"discarded={1-retained.mean():.2%}, projection h={router.h:g}.", flush=True)
            if float(retained.mean()) < .05:
                print("Router diagnostic: selected pairs retain less than 5% of probability. "
                      "Inspect router h/reg/lam and mapping images; fit convergence alone does not establish useful routing.", flush=True)
            pc = config["partial"]["solver"]
            h_projection = config["projection"].get("patch_bandwidth", pc["h"] * config["projection"]["bandwidth_multiplier"])
            if "pair_kernels" in manifest:
                saved_shared = torch.load(checked_file(directory, manifest["pair_kernels"]), weights_only=True)
                shared = load_kernels(saved_shared)
                # Resume must not change solver arithmetic to quantized kernels.
                for key, features in (("kx", x), ("ky", y)):
                    exact = torch.stack([kernel_state(patches, pc["h"])[0] for patches in features])
                    if not torch.equal(exact.float(), saved_shared[key].to(device)):
                        raise ValueError("Rebuilt fit kernels do not match the saved float32 snapshot.")
                    shared[key] = exact
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
                entry, _ = _verified_save(directory, path, store_kernels(shared), load_kernels)
                manifest["pair_kernels"] = entry
                manifest["partial_objective"] = OBJECTIVE
                _registered_manifest(directory, manifest)
            index_path = directory / "pairs.jsonl"
            inventory = _read_pair_inventory(index_path, resume=bool(resume), log=log)
            done = {(entry["source_id"], entry["target_id"]): entry for entry in inventory}
            if len(done) != len(inventory):
                raise ValueError("Duplicate pair inventory entries.")
            if set(done) - selected_pairs:
                raise ValueError("Pair inventory contains an intentionally excluded or unknown pair.")
            for entry in inventory:
                saved = torch.load(checked_file(directory, entry), weights_only=True)
                validate_plan(saved, saved["a"], saved["b"], config["partial"]["keep_mass"], pc)
                if (saved["report"]["status"] != "converged" or saved["fit_fingerprint"] != fingerprint
                        or saved["source_id"] != entry["source_id"] or saved["target_id"] != entry["target_id"]):
                    raise ValueError("Registered pair identity/status mismatch.")
                _cleanup_latest(directory, entry, log)
            if config.get("pair_batch_size"):
                from .pair_batch_fit import fit_pairs_batched
                fit_pairs_batched(directory=directory, manifest=manifest, source=source, target=target,
                    x=x, y=y, shared=shared, selected_pairs=selected_pairs, done=done, log=log)
            # Keep the original POT serial path as an explicit reference.
            serial_sources = () if config.get("pair_batch_size") else enumerate(source.ids)
            for i, sid in serial_sources:
                for j, tid in enumerate(target.ids):
                    if (sid, tid) not in selected_pairs or (sid, tid) in done:
                        continue
                    log.event("pair_started", source_id=sid, target_id=tid, completed_pairs=len(done), total_pairs=len(selected_pairs))
                    # Unique indices within this immutable support order plus an
                    # ID hash keep nested log paths usable on Windows. Identity
                    # comes from full stable IDs/fingerprints in the inventory.
                    pair_key = f"{i:06d}_{j:06d}_{digest([sid, tid])[:12]}"
                    plan, report = solve_partial(distance(x[i], y[j]), shared["kx"][i], shared["ky"][j],
                        keep_mass=config["partial"]["keep_mass"], config=pc, on_step=progress(f"pairs/{pair_key}"))
                    a, b = plan.new_full((len(x[i]),), 1 / len(x[i])), plan.new_full((len(y[j]),), 1 / len(y[j]))
                    path = directory / "plans/pairs" / f"{pair_key}.pt"
                    saved = store_plan_state(dict(plan=plan, a=a, b=b,
                        source_id=sid, target_id=tid, source_index=i, target_index=j,
                        fit_fingerprint=fingerprint, report=report))
                    entry, saved = _verified_save(directory, path, saved,
                        lambda s: validate_plan(s, s["a"], s["b"], config["partial"]["keep_mass"], pc))
                    if report["status"] != "converged":
                        raise RuntimeError(f"Pair {sid}/{tid}: {report['status']}; saved plan is not accepted as a successful fit.")
                    entry.update(source_id=sid, target_id=tid, source_index=i, target_index=j)
                    with index_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(entry) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    _sync_directory(directory)
                    done[sid, tid] = entry
                    _cleanup_latest(directory, entry, log)
                    log.event("pair_completed", source_id=sid, target_id=tid, completed_pairs=len(done),
                              total_pairs=len(selected_pairs), status=report["status"], last_iteration=report["history"][-1],
                              storage=saved["storage"])
                    print(f"Pair {len(done)}/{len(selected_pairs)}: {sid} -> {tid}", flush=True)
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
                  resources=manifest["resources"], runtime=manifest["runtime"], precision=manifest["precision"],
                  sampling=manifest.get("sampling"), models={})
    for name, entry in manifest["models"].items():
        state = torch.load(Path(directory) / entry["file"], weights_only=True)
        report["models"][name] = {k: state[k] for k in ("status", "cost_scale", "source_scale", "target_scale", "row_residual", "column_residual")}
        report["models"][name]["last_iteration"] = state["history"][-1]
        if state.get("storage"):
            report["models"][name]["storage"] = state["storage"]
            p = state["plan"].double()
            report["models"][name]["storage_validation"] = validate_plan(state,
                p.new_full((len(p),), 1/len(p)), p.new_full((p.shape[1],), 1/p.shape[1]), 1., state["config"], balanced=True)
    if manifest["config"]["mode"] == "grouped_partial":
        stats = dict(pair_count=0, fitted_mass_min=1., fitted_mass_max=0., row_cap_error_max=0., column_cap_error_max=0.)
        geometry_rows = []
        stats.update(quantization_l1_error_max=0., quantization_underflow_entries=0, stored_mass_min=1., stored_mass_max=0.)
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
                geometry = {k: float(v) for k, v in patch_plan_concentration(state["plan"].double()).items()}
                summary["geometry"] = geometry
                geometry_rows.append(geometry)
                summary["storage"] = state.get("storage")
                stored = validate_plan(state, state["a"], state["b"], manifest["config"]["partial"]["keep_mass"], manifest["config"]["partial"]["solver"])
                summary["storage_validation"] = stored
                stats["stored_mass_min"] = min(stats["stored_mass_min"], stored["mass"])
                stats["stored_mass_max"] = max(stats["stored_mass_max"], stored["mass"])
                if state.get("storage"):
                    errors = state["storage"]["quantization"]
                    stats["quantization_l1_error_max"] = max(stats["quantization_l1_error_max"], errors["l1_error"])
                    stats["quantization_underflow_entries"] += errors["underflow_entries"]
                handle.write(json.dumps(summary) + "\n")
                stats["pair_count"] += 1
                stats["fitted_mass_min"] = min(stats["fitted_mass_min"], summary["mass"])
                stats["fitted_mass_max"] = max(stats["fitted_mass_max"], summary["mass"])
                for key in ("row_cap_error", "column_cap_error"):
                    stats[f"{key}_max"] = max(stats[f"{key}_max"], summary[key])
        report["partial"] = dict(stats, keep_mass=manifest["config"]["partial"]["keep_mass"],
            fit_pair_top_k=manifest["config"].get("fit_pair_top_k"), full_pair_count=len(manifest["source_ids"])*len(manifest["target_ids"]),
            diagnostics_file="pair_diagnostics.jsonl", confidence_note="Raw retention differs from smoothed query confidence and thresholded mask coverage.")
        if geometry_rows:
            report["partial"]["fitted_pair_geometry"] = {key: dict(min=min(r[key] for r in geometry_rows),
                mean=sum(r[key] for r in geometry_rows)/len(geometry_rows), max=max(r[key] for r in geometry_rows))
                for key in geometry_rows[0]}
        if "pair_selection" in manifest:
            selection = json.loads(checked_file(directory, manifest["pair_selection"]).read_text(encoding="utf-8"))
            retained = [r["retained_probability"] for r in selection["rows"]]
            report["partial"]["fit_selection_retained_probability"] = dict(min=min(retained), mean=sum(retained)/len(retained), max=max(retained))
            shared = torch.load(checked_file(directory, manifest["pair_kernels"]), weights_only=True)
            report["partial"]["kernel_storage"] = shared.get("storage")
    return report
