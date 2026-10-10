"""Benchmark identical deterministic dense partial problems; no bank/model changes."""
from pathlib import Path
import argparse
import json
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from infoot_vit.infoot_helper.device import resolve_device
from infoot_vit.infoot_helper.feature_bank import write_json
from infoot_vit.infoot_helper.partial import distance, kernel_state, solve_partial, solver_config, objective
from infoot_vit.infoot_helper.partial_batch import solve_partial_batch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "infoot_vit/configs/grouped_partial.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--pair-batch-size", type=int, default=256)
    p.add_argument("--patches", type=int, default=196)
    p.add_argument("--feature-dim", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if min(args.pairs, args.pair_batch_size, args.repeats, args.threads, args.feature_dim) < 1 or args.patches < 2:
        p.error("Use positive counts and at least two patches.")
    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        report = dict(status="unverified", reason="CUDA hardware/runtime unavailable; no CPU substitute benchmark.",
                      torch_version=str(torch.__version__), requested_device=args.device)
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
        return 0
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))["partial"]
    c, mass = solver_config(cfg.get("solver")), cfg.get("keep_mass", .8)
    rng = torch.Generator().manual_seed(args.seed)
    shape = (args.pairs, args.patches, args.feature_dim)
    x, y = (torch.randn(shape, generator=rng, dtype=torch.float64).to(device) for _ in range(2))
    costs = distance(x, y)
    kx, ky = (torch.stack([kernel_state(row, c["h"])[0] for row in values]) for values in (x, y))
    del x, y

    def run(kind, count):
        plans, reports = [], []
        for start in range(0, count, 1 if kind == "serial" else args.pair_batch_size):
            if kind == "serial":
                last = {}
                def track(plan, record):
                    last.update(plan=plan, record=record)
                try:
                    plan, report = solve_partial(costs[start], kx[start], ky[start], keep_mass=mass, config=c, on_step=track)
                except (RuntimeError, ValueError) as error:
                    # Preserve the last feasible serial iterate for comparison;
                    # a failed solve never counts as a valid speed comparison.
                    plan = last.get("plan", costs.new_full(costs[start].shape, mass / args.patches**2))
                    scale = float(costs[start].mean()) if c["cost_scale"] == "mean" else c["cost_scale"]
                    report = dict(status="serial_failed", error=str(error), history=[
                        objective(plan, costs[start] / scale, kx[start], ky[start], c)])
                plans.append(plan[None]); reports.append(report)
            else:
                end = min(start + args.pair_batch_size, count)
                plan, report = solve_partial_batch(costs[start:end], kx[start:end], ky[start:end], keep_mass=mass, config=c)
                plans.append(plan); reports.extend(report)
        return torch.cat(plans), reports

    # Both methods see exactly the same matrices; warm-up is outside timing.
    for kind in ("serial", "batched"):
        run(kind, min(2, args.pairs))
    results, timing = {}, {name: [] for name in ("serial", "batched")}
    for repeat in range(args.repeats):
        for kind in (("serial", "batched") if repeat % 2 == 0 else ("batched", "serial")):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            plans, reports = run(kind, args.pairs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            timing[kind].append(dict(seconds=elapsed, pairs_per_second=args.pairs / elapsed,
                converged_pairs=sum(r["status"] == "converged" for r in reports),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None))
            results[kind] = (plans.cpu(), reports)
            del plans
    serial, batched = results["serial"], results["batched"]
    all_converged = all(r["status"] == "converged" for result in results.values() for r in result[1])
    plan_error = float((serial[0] - batched[0]).abs().max())
    objective_error = max(abs(a["history"][-1]["objective"]-b["history"][-1]["objective"])
                          for a, b in zip(serial[1], batched[1]))
    finite = bool(torch.isfinite(serial[0]).all() & torch.isfinite(batched[0]).all())
    agreement = finite and plan_error <= 1e-8 and objective_error <= 1e-6
    report = dict(status="solver_failed" if not all_converged else "measured" if agreement else "reference_mismatch",
        valid_speed_comparison=all_converged and agreement,
        device=str(device), torch_version=str(torch.__version__),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        workload="seeded synthetic features; not AFHQ image-quality evidence",
        seed=args.seed, pairs=args.pairs, patches=args.patches, feature_dim=args.feature_dim,
        pair_batch_size=args.pair_batch_size, solver=c, keep_mass=mass,
        timing_scope="solver including diagnostics; excludes cost/kernel construction and file I/O; peak includes cached inputs",
        runs=timing, summary={name: dict(median_seconds=statistics.median(r["seconds"] for r in rows),
            median_pairs_per_second=statistics.median(r["pairs_per_second"] for r in rows)) for name, rows in timing.items()},
        agreement=dict(max_plan_abs_error=plan_error, max_objective_abs_error=objective_error, finite=finite,
            plan_comparison_atol=1e-8, objective_comparison_atol=1e-6,
            serial_statuses=[r["status"] for r in serial[1]], batched_statuses=[r["status"] for r in batched[1]],
            failures={name: [dict(pair=i, status=r["status"], error=r.get("error"), last_iteration=r["history"][-1])
                            for i, r in enumerate(result[1]) if r["status"] != "converged"]
                      for name, result in results.items()}))
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
