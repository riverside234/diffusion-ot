"""Small, reproducible partial-InfoOT trials using the production solvers.

No plans/checkpoints are written. An optional JSON report contains scalar
diagnostics only. Projection probes query TRAINING patches, not held-out images.
"""
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import torch

from .conditional import BalancedModel, calibrate_support, partial_projection
from .feature_bank import checked_file, digest, file_hash, write_json
from .fit_mapping import resource_estimate, validate_config
from .pair_selection import select_pairs, selection_edges
from .partial import OBJECTIVE, distance, feasibility, kernel_state, solve_partial
from .partial_batch import solve_partial_batch
from .plan_diagnostics import patch_plan_concentration
from .storage import VERSION as STORAGE_VERSION, validate_plan
from .tuning import load_tuning_support


def _stats(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return dict(count=0, mean=None, median=None, p90=None, min=None, max=None)
    v = torch.tensor(finite, dtype=torch.float64)
    return dict(count=len(v), mean=float(v.mean()), median=float(v.quantile(.5)),
                p90=float(v.quantile(.9)), min=float(v.min()), max=float(v.max()))


def _sample_pairs(selection, limit, seed):
    population = []
    for row in selection["rows"]:
        for rank, (tid, probability) in enumerate(zip(row["target_ids"], row["probabilities"]), 1):
            population.append(dict(source_id=row["source_id"], target_id=tid, router_rank=rank,
                routing_probability=probability, retained_routing_mass=row["retained_probability"],
                normalized_routing_probability=probability / row["retained_probability"]))
    population.sort(key=lambda r: (r["source_id"], r["target_id"]))
    indices = sorted(random.Random(seed).sample(range(len(population)), min(limit, len(population))))
    return [population[i] for i in indices]


def _selection(c, banks, x, y, from_fit, log_every):
    source, target = banks
    multiplier = c["projection"]["bandwidth_multiplier"]
    if from_fit is not None:
        manifest = json.loads((from_fit / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "siglip_infoot_mapping_v3":
            raise ValueError("--tune-from requires a dense grouped_partial fit directory, not a test-results directory.")
        if (manifest.get("status") == "complete" and
                digest({k:v for k,v in manifest.items() if k != "artifact_id"}) != manifest.get("artifact_id")):
            raise ValueError("Changed saved mapping manifest.")
        saved = validate_config(manifest["config"])
        if (saved["mode"] != "grouped_partial" or saved["solver"] != c["solver"]
                or saved["fit_pair_top_k"] != c["fit_pair_top_k"]
                or saved["projection"]["bandwidth_multiplier"] != multiplier
                or manifest["source_ids"] != source.ids or manifest["target_ids"] != target.ids
                or manifest["source_bank"]["artifact_id"] != source.artifact_id
                or manifest["target_bank"]["artifact_id"] != target.artifact_id
                or manifest["representation"] != source.representation):
            raise ValueError("Saved router/selection settings or bank support differ; hold image settings and sampling fixed.")
        if "image" not in manifest["models"] or "pair_selection" not in manifest:
            raise ValueError("Saved fit needs a registered image router and pair selection; no replacement router is fitted.")
        entry = manifest["models"]["image"]
        # Storage validation uses its original CPU reduction arithmetic. All
        # new patch costs/kernels/solver/projection computations use x.device.
        state = torch.load(checked_file(from_fit, entry), map_location="cpu", weights_only=True)
        if (state["status"] != "converged" or state["config"] != c["solver"]
                or state.get("storage", {}).get("version") != STORAGE_VERSION
                or state["source_ids"] != source.ids or state["target_ids"] != target.ids):
            raise ValueError("Saved image router is nonconverged or its identity/configuration is inconsistent.")
        a = torch.full((len(source.ids),), 1 / len(source.ids), dtype=torch.float64)
        b = torch.full((len(target.ids),), 1 / len(target.ids), dtype=torch.float64)
        validate_plan(state, a, b, 1., c["solver"], balanced=True)
        selected = json.loads(checked_file(from_fit, manifest["pair_selection"]).read_text(encoding="utf-8"))
        if (selected.get("router_sha256") != entry["sha256"]
                or selected.get("projection_bandwidth_multiplier") != multiplier
                or not math.isclose(selected.get("projection_h", -1), c["solver"]["h"] * multiplier)):
            raise ValueError("Saved pair selection has a different router or projection bandwidth.")
        provenance = dict(kind="saved_selection", directory=str(from_fit),
            router_sha256=entry["sha256"], selection_sha256=manifest["pair_selection"]["sha256"],
            router_fit_fingerprint=manifest["fit_fingerprint"], router_precision="stored_float32")
    else:
        def progress(plan, record):
            if record["iteration"] == 1 or record["iteration"] % log_every == 0 or record["status"] != "running":
                print(f"Tuning router {record['iteration']}: objective={record['objective']:.8g}, "
                      f"delta={record['plan_delta_l1']:.3g}, status={record['status']}", flush=True)
        model = BalancedModel.fit(x.flatten(1), y.flatten(1), c["solver"], on_step=progress)
        if model.state["status"] != "converged":
            raise RuntimeError(f"Tuning image router {model.state['status']}; partial trials have not started.")
        router = BalancedModel(x.flatten(1), y.flatten(1), model.state, multiplier)
        selected = select_pairs(router, source.ids, target.ids, c["fit_pair_top_k"])
        provenance = dict(kind="fresh_router", router_precision="float64_before_storage",
            router_last_iteration=model.state["history"][-1], selection_sha256=digest(selected))
    selection_edges(selected, source.ids, target.ids, c["fit_pair_top_k"])
    return selected, provenance


def _kernel_diagnostics(k, scale, h):
    w = k / k.sum(-1, keepdim=True)
    return dict(rms_scale=scale, bandwidth=scale*h,
        mean_self_probability=float(w.diagonal().mean()),
        mean_effective_neighbors=float((-torch.special.xlogy(w, w).sum(-1)).exp().mean()))


def _projection_probe(x, y, plan, source_kernel, target_kernel, projection):
    _, sx = source_kernel
    _, sy = target_kernel
    h, cf = projection["patch_bandwidth"], projection["confidence"]
    support = calibrate_support(x, sx, h, cf)
    a, b = x.new_full((len(x),), 1/len(x)), y.new_full((len(y),), 1/len(y))
    mapped, confidence, diag = partial_projection(x, x, y, plan, a, b, sx, sy, h, support)
    mask = (confidence > cf["mass_floor"]) & (confidence >= cf["threshold"])
    eff = (-torch.special.xlogy(diag["weights"], diag["weights"]).sum(1)).exp()
    total = float(confidence.sum())
    valid = mapped[mask]
    target_var = float(y.var(0, unbiased=False).mean())
    target_norm = float(y.norm(dim=1).mean())
    return dict(probe="source_training_patches_self_query", patch_projection_h=h,
        support_valid_fraction=float(diag["support_valid"].double().mean()),
        confidence_mean=float(confidence.mean()), confidence_min=float(confidence.min()),
        raw_confidence_mean=float(diag["raw_confidence"].mean()),
        valid_fraction=float(mask.double().mean()), valid_tokens=int(mask.sum()),
        all_invalid=not bool(mask.any()),
        projected_effective_patches=float((confidence*eff).sum())/total if total > 0 else None,
        mapped_to_target_variance_ratio=float(valid.var(0, unbiased=False).mean())/target_var
            if len(valid) >= 2 and target_var > 0 else None,
        mapped_to_target_norm_ratio=float(valid.norm(dim=1).mean())/target_norm
            if len(valid) and target_norm > 0 else None,
        threshold_sweep=[dict(threshold=t, valid_fraction=float(((confidence > cf["mass_floor"]) &
            (confidence >= t)).double().mean())) for t in sorted({.05, .10, .20, .30, cf["threshold"]})])


def _weighted_terms(record, pc):
    result = dict(record)
    result["mi_term"] = -pc["lam"]*record["information"] if record.get("information") is not None else None
    result["entropy_term"] = pc["reg"]*record["entropy"] if record.get("entropy") is not None else None
    return result


def _summary(rows):
    successes = [r for r in rows if r["status"] == "converged"]
    geometry = {k:_stats([r["geometry"][k] for r in successes])
                for k in successes[0]["geometry"]} if successes else {}
    projected = [r["projection"] for r in successes if "projection" in r]
    return dict(status_counts=dict(Counter(r["status"] for r in rows)),
        successful_pairs=len(successes), failed_pairs=len(rows)-len(successes),
        projection_failures=sum("projection_error" in r for r in successes),
        iterations_all_pairs=_stats([r["iterations"] for r in rows]),
        feasibility_all_pairs={k:_stats([r["feasibility"].get(k) for r in rows])
            for k in ("mass", "mass_error", "row_cap_error", "column_cap_error")},
        last_inner_diagnostics_all_pairs={k:_stats([r["last_iteration"].get("inner", {}).get(k) for r in rows])
            for k in ("iterations", "error", "relative_plan_delta_l1", "kkt_error", "relative_duality_gap")},
        successful_objectives={k:_stats([r["last_iteration"].get(k) for r in successes])
            for k in ("objective", "cost", "information", "entropy", "mi_term", "entropy_term")},
        successful_geometry=geometry,
        successful_projection={k:_stats([p[k] for p in projected]) for k in (
            "valid_fraction", "confidence_mean", "support_valid_fraction", "projected_effective_patches",
            "mapped_to_target_variance_ratio", "mapped_to_target_norm_ratio")},
        all_invalid_projection_pairs=sum(p["all_invalid"] for p in projected),
        note="Unweighted sampled-pair summaries; failures excluded from success metrics. No full routed image or held-out quality measurement.")


@torch.no_grad()
def tune_partial(raw, *, root, from_fit=None, pair_limit=64, seed=42, log_every=25, report_path=None):
    """Fit selected TRAINING pairs with exact production solvers, then discard plans."""
    if raw.get("mode") != "grouped_partial":
        raise ValueError("--tune-partial supports dense grouped_partial only.")
    if type(pair_limit) is not int or pair_limit < 1 or type(seed) is not int or type(log_every) is not int or log_every < 1:
        raise ValueError("Tuning requires positive pair/log limits and an integer seed.")
    root = Path(root).resolve()
    from_fit = (root / from_fit).resolve() if from_fit is not None else None
    report_path = (root / report_path).resolve() if report_path is not None else None
    if report_path is not None:
        if report_path.exists():
            raise FileExistsError(f"Tuning report already exists: {report_path}")
        if from_fit is not None and report_path.is_relative_to(from_fit):
            raise ValueError("Save the tuning report outside the reused fit directory.")
    started = time.perf_counter()
    c, banks, device, context = load_tuning_support(raw, root=root)
    pc, mass = c["partial"]["solver"], c["partial"]["keep_mass"]
    # Bound actual trial batch workspace, not the full 16,000-pair fit/storage.
    size = min(c["pair_batch_size"] or 1, pair_limit)
    estimate = resource_estimate(banks[0].features.shape, banks[1].features.shape,
                                 "grouped_partial", c["fit_pair_top_k"], size)
    if estimate["estimated_working_gib"] >= c["resources"]["max_working_gib"]:
        raise MemoryError("Partial tuning exceeds resources.max_working_gib; reduce --pair-batch-size or --tune-pairs.")
    x, y = [bank.features.to(device=device, dtype=torch.float64) for bank in banks]
    print("Partial tuning: exact sampled pair solves; no mapping, plan, or checkpoint files saved.", flush=True)
    selected, provenance = _selection(c, banks, x, y, from_fit, log_every)
    pairs = _sample_pairs(selected, pair_limit, seed)
    context.update(tuning_stage="partial", image_solver=context.pop("solver"), partial_solver=pc,
        keep_mass=mass, objective_version=OBJECTIVE,
        objective="cost - lam * information + reg * entropy",
        information_convention="transported-mass-weighted KDE MI; not the balanced image-router estimator",
        entropy_convention="sum(Gamma * (log(Gamma) - 1)); not positive Shannon entropy",
        image_projection_h=c["solver"]["h"]*c["projection"]["bandwidth_multiplier"],
        patch_projection_h=c["projection"].get("patch_bandwidth", pc["h"]*c["projection"]["bandwidth_multiplier"]),
        confidence=c["projection"]["confidence"], pair_batch_size=c["pair_batch_size"],
        actual_batch_limit=min(size, len(pairs)), estimated_working_gib=estimate["estimated_working_gib"],
        selection=provenance, pair_sampling=dict(algorithm="sorted_pair_ids_python_random_sample_v1", seed=seed,
            requested_pairs=pair_limit, selected_pairs=len(pairs), population_pairs=selected["pair_count"],
            replacement=False, effective_top_k=selected["effective_k"],
            ordered_pairs_sha256=digest([(p["source_id"], p["target_id"]) for p in pairs]),
            rank_counts=dict(Counter(p["router_rank"] for p in pairs))),
        implementation_sha256={n:file_hash(Path(__file__).with_name(n)) for n in (
            "partial_tuning.py", "partial.py", "partial_batch.py", "partial_newton.py", "conditional.py")},
        projection_probe_note="Self-query training patches, float64 before storage; confidence/variance are not semantic quality. No decoder, held-out queries, or image-router mixture is evaluated.")
    print(json.dumps(context, indent=2, allow_nan=False), flush=True)
    xi, yi = [{sid:i for i,sid in enumerate(bank.ids)} for bank in banks]
    cache_x, cache_y = {}, {}
    for row in pairs:
        for sid, index, data, cache in ((row["source_id"], xi, x, cache_x), (row["target_id"], yi, y, cache_y)):
            if sid not in cache:
                cache[sid] = kernel_state(data[index[sid]], pc["h"])
    context["kernel_diagnostics"] = {
        name: {sid:_kernel_diagnostics(*state, pc["h"]) for sid,state in cache.items()}
        for name, cache in (("source", cache_x), ("target", cache_y))}
    rows, batches = [], []
    solver_seconds = 0.
    for offset in range(0, len(pairs), size):
        block = pairs[offset:offset+size]
        xx = x[[xi[r["source_id"]] for r in block]]
        yy = y[[yi[r["target_id"]] for r in block]]
        costs = distance(xx, yy)
        kx = torch.stack([cache_x[r["source_id"]][0] for r in block])
        ky = torch.stack([cache_y[r["target_id"]][0] for r in block])
        observed = [None]*len(block)
        def progress(plans, reports):
            due = False
            for i,r in enumerate(reports):
                if r is None:
                    continue
                observed[i] = r
                last = r["history"][-1] if r["history"] else {}
                iteration = last.get("iteration", 0)
                due |= iteration == 1 or iteration % log_every == 0 or r["status"] != "running"
                if r["status"] not in {"running", "converged"}:
                    print("Partial pair failure: " + json.dumps(dict(block[i], status=r["status"],
                        error=r.get("error"), last_iteration=last), allow_nan=False), flush=True)
            if due:
                last = [r["history"][-1] for r in observed if r is not None and r["history"]]
                stats = {k:_stats([v.get(k) for v in last])["mean"] for k in ("objective", "cost", "information", "entropy", "plan_delta_l1")}
                print("Partial batch progress: " + json.dumps(dict(batch=offset//size, pairs=len(block),
                    outer_iteration=max((v["iteration"] for v in last), default=0),
                    statuses=dict(Counter(r["status"] for r in observed if r)), **_weighted_terms(stats, pc)),
                    allow_nan=False), flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        if c["pair_batch_size"] is not None:
            plans, reports = solve_partial_batch(costs, kx, ky, keep_mass=mass, config=pc, on_step=progress)
        else:
            try:
                def serial_progress(plan, record):
                    progress(plan[None], [dict(status=record["status"], history=[record])])
                plan, report = solve_partial(costs[0], kx[0], ky[0], keep_mass=mass, config=pc, on_step=serial_progress)
                plans, reports = plan[None], [report]
            except (RuntimeError, ValueError) as error:
                plans = None
                reports = [dict(status="solver_error", error=str(error), history=observed[0]["history"] if observed[0] else [])]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter()-begin
        solver_seconds += seconds
        batches.append(dict(batch=offset//size, pairs=len(block), solver_seconds=seconds,
                            pairs_per_second=len(block)/seconds if seconds else None))
        for i,(pair, report) in enumerate(zip(block, reports)):
            history = report["history"]
            row = dict(pair, status=report["status"], iterations=len(history), error=report.get("error"),
                cost_scale=report.get("cost_scale"), runtime=report.get("runtime"),
                last_iteration=_weighted_terms(history[-1], pc) if history else {},
                feasibility={k:report.get(k) for k in ("mass", "mass_error", "row_cap_error", "column_cap_error")},
                inner_iterations=_stats([r.get("inner", {}).get("iterations") for r in history]),
                backtracks=_stats([r.get("backtracks") for r in history]))
            if row["status"] == "converged":
                plan = plans[i]
                a, b = plan.new_full((len(xx[i]),), 1/len(xx[i])), plan.new_full((len(yy[i]),), 1/len(yy[i]))
                row["feasibility"] = feasibility(plan, a, b, mass, pc)
                spread = float(costs[i].std(-1, unbiased=False).mean()/report["cost_scale"])
                row["geometry"] = dict(mean_row_cost_std=spread,
                    reg_over_mean_row_cost_std=pc["reg"]/spread if spread else None,
                    **{k:float(v) for k,v in patch_plan_concentration(plan).items()})
                try:
                    row["projection"] = _projection_probe(xx[i], yy[i], plan, cache_x[pair["source_id"]],
                        cache_y[pair["target_id"]], dict(c["projection"], patch_bandwidth=context["patch_projection_h"]))
                except (RuntimeError, ValueError) as error:
                    row["projection_error"] = str(error)
            rows.append(row)
            print(f"Partial pair {len(rows)}/{len(pairs)}: {pair['source_id']} -> {pair['target_id']}, "
                  f"status={row['status']}, outer={row['iterations']}, "
                  f"effective_patches={row.get('geometry', {}).get('fitted_pair_effective_patches')}", flush=True)
    summary = _summary(rows)
    report = dict(context, status="converged" if not summary["failed_pairs"] and not summary["projection_failures"] else "failed",
        seconds=time.perf_counter()-started, solver_seconds=solver_seconds,
        summary=summary, pairs=rows, batches=batches,
        timing_note="Solver timings include progress/callback overhead and CUDA boundary synchronization; not a serial-vs-batch benchmark.")
    if report_path is not None:
        if report_path.exists():
            raise FileExistsError(f"Tuning report appeared during the run: {report_path}")
        write_json(report_path, report)
        print(f"Saved partial tuning diagnostics: {report_path}", flush=True)
    print("Partial tuning result (not a saved mapping):", flush=True)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report
