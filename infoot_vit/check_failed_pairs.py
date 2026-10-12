"""Replay the exact failed training pairs before committing to a full fit.

Reads the compact analysis retained after deleting displacement_next. No router,
mapped cache, decoder, per-pair files or tensor checkpoints are created. Exit 0
requires every listed pair to pass the original strict convergence checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from infoot_vit.infoot_helper.device import resolve_device
from infoot_vit.infoot_helper.feature_bank import FeatureBank, compatible_banks, file_hash, write_json
from infoot_vit.infoot_helper.fit_mapping import validate_config
from infoot_vit.infoot_helper.partial import distance, kernel_state, feasibility
from infoot_vit.infoot_helper.partial_batch import VERSION, solve_partial_batch
from infoot_vit.infoot_helper.sampling import sample_ids


def require_certificate(plan, report, config, mass):
    """Do not equate a status string, small damped step or feasibility with success."""
    last = report["history"][-1]
    inner = last["inner"]
    tol = inner["certificate_tolerance"]
    def bounded(value, limit):
        return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= limit
    if (report["status"] != "converged" or last["status"] != "converged"
            or (config["lam"] and not bounded(last["plan_delta_l1"], config["outer_tolerance"]))
            or not 0 < tol <= config["inner_tolerance"] or inner["failure_code"] != 0
            or inner["convergence_reason"] not in {"primal_dual_certificate", "legacy_residual_and_gap"}
            or any(not bounded(inner[key], tol) for key in
                   ("relative_duality_gap", "relative_plan_delta_l1", "kkt_error"))):
        raise ValueError("Pair lacks an undamped outer and certified inner convergence check.")
    return feasibility(plan, plan.new_full((len(plan),), 1 / len(plan)),
                       plan.new_full((plan.shape[1],), 1 / plan.shape[1]), mass, config)


def run_check(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new report file: {output}")
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    if analysis.get("schema") != "infoot_displacement_run_analysis_v1":
        raise ValueError("Expected the retained displacement analysis, not a fit or mapping manifest.")
    pairs = analysis["failed_pair_terminal_evidence"]
    keys = [(row["source_id"], row["target_id"]) for row in pairs]
    if not keys or len(set(keys)) != len(keys) or len(keys) != analysis["counts"]["failed"]:
        raise ValueError("Failed-pair evidence is empty, duplicated or inconsistent with its recorded count.")
    c = validate_config(deepcopy(analysis["configuration"]))
    if c["mode"] != "grouped_partial" or c["partial"]["keep_mass"] == 1.:
        raise ValueError("This check requires the original dense partial solver configuration.")
    device = resolve_device(args.device)
    banks = []
    for name, axis in (("source", 0), ("target", 1)):
        path = getattr(args, name + "_bank") or ROOT / c[name + "_bank"]
        bank = FeatureBank.load(path)
        if bank.artifact_id != analysis[name + "_bank_id"] or bank.manifest["split"] != "train":
            raise ValueError(f"Require the original {name} training bank identity.")
        selected = sample_ids(bank.ids, c["sampling"]["images_per_domain"], c["sampling"]["seed"])
        needed = sorted({key[axis] for key in keys})
        if not set(needed) <= set(selected):
            raise ValueError("Failed pairs must belong to the original seeded training population.")
        if output.is_relative_to(bank.path.resolve()):
            raise ValueError("Write the numerical report outside immutable feature banks.")
        banks.append(bank.subset(needed))
    compatible_banks(*banks)
    del bank
    pc, mass = c["partial"]["solver"], c["partial"]["keep_mass"]
    report = dict(schema="infoot_failed_pair_check_v1", status="running", solver_version=VERSION,
        analysis_sha256=file_hash(args.analysis), analysis=str(args.analysis), device=str(device),
        source_bank_id=banks[0].artifact_id, target_bank_id=banks[1].artifact_id,
        partial_solver=pc, keep_mass=mass, requested_pairs=len(pairs), batch_size=args.batch_size,
        implementation_sha256={name:file_hash(ROOT / "infoot_vit/infoot_helper" / name) for name in
            ("partial.py", "partial_batch.py", "partial_newton.py")},
        runner_sha256=file_hash(Path(__file__)), pairs=[],
        note="Numerical check of previously failed pairs only; not a complete fitted mapper or image-quality result.")
    write_json(output, report)
    started = time.perf_counter()
    try:
        x, y = [bank.features.to(device=device, dtype=torch.float64) for bank in banks]
        xi, yi = [{sid:i for i,sid in enumerate(bank.ids)} for bank in banks]
        kx, ky = [torch.stack([kernel_state(row, pc["h"])[0] for row in features]) for features in (x, y)]
        for offset in range(0, len(keys), args.batch_size):
            block = keys[offset:offset+args.batch_size]
            ix, iy = ([index[key[axis]] for key in block] for axis, index in enumerate((xi, yi)))
            seen = [None] * len(block)
            last_print = 0
            def progress(plans, states):
                nonlocal last_print
                for i, state in enumerate(states):
                    if state is not None:
                        seen[i] = state
                iteration = max((s["iterations"] for s in seen if s is not None), default=0)
                done = sum(s is not None and s["status"] != "running" for s in seen)
                if iteration != last_print and (iteration == 1 or iteration % 25 == 0 or done == len(block)):
                    print(f"Failed-pair check {offset+1}-{offset+len(block)}/{len(keys)}: "
                          f"outer={iteration}, finished={done}/{len(block)}", flush=True)
                    last_print = iteration
            plans, states = solve_partial_batch(distance(x[ix], y[iy]), kx[ix], ky[iy],
                config=pc, keep_mass=mass, on_step=progress)
            for i, ((sid, tid), state) in enumerate(zip(block, states)):
                row = dict(source_id=sid, target_id=tid, status=state["status"],
                    previous_status=pairs[offset+i]["status"], iterations=state["iterations"],
                    final_iteration=state["history"][-1] if state["history"] else None,
                    small_damped_steps=sum(r.get("small_damped_progress", False) for r in state["history"]))
                if state["status"] == "converged":
                    try:
                        row["feasibility"] = require_certificate(plans[i], state, pc, mass)
                    except ValueError as error:
                        row.update(status="certificate_failed", error=str(error))
                if row["status"] != "converged":
                    row["failure_history"] = state["history"]
                report["pairs"].append(row)
            write_json(output, report)
        counts = Counter(row["status"] for row in report["pairs"])
        report.update(status="passed" if counts["converged"] == len(keys) else "failed",
            successful_pairs=counts["converged"], failed_pairs=len(keys)-counts["converged"], status_counts=dict(counts),
            seconds=time.perf_counter()-started)
        write_json(output, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", seconds=time.perf_counter()-started)
        write_json(output, report)
        raise
    print(f"Numerical check {report['status']}: {report['successful_pairs']}/{len(keys)} pairs passed; {output}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "results/infoot_vit/displacement_next_analysis.json")
    parser.add_argument("--source-bank", type=Path)
    parser.add_argument("--target-bank", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="One new JSON report; no plans or log directories.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) < 1:
        parser.error("Batch size and threads must be positive.")
    torch.set_num_threads(args.threads)
    return 0 if run_check(args)["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
