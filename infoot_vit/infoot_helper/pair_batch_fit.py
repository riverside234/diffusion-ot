"""Batched dense pair orchestration; numerical kernels live in partial_batch."""
from __future__ import annotations

import json
import os
import time

import torch

from .feature_bank import digest, save_tensor, write_json
from .partial import distance
from .partial_batch import solve_partial_batch
from .storage import store_plan_state, validate_plan
from .plan_diagnostics import patch_plan_concentration


def fit_pairs_batched(*, directory, manifest, source, target, x, y, shared, selected_pairs, done, log):
    # Imported at call time to reuse the same atomic verification/registration
    # and cleanup contract as the serial fitter without a module import cycle.
    from .fit_mapping import _verified_save, _cleanup_latest, _sync_directory, _file_entry

    config = manifest["config"]
    pc, mass = config["partial"]["solver"], config["partial"]["keep_mass"]
    size, every = config["pair_batch_size"], config["pair_checkpoint_every"]
    fingerprint = manifest["fit_fingerprint"]
    pending = [(i, j, sid, tid, f"{i:06d}_{j:06d}_{digest([sid, tid])[:12]}")
               for i, sid in enumerate(source.ids) for j, tid in enumerate(target.ids)
               if (sid, tid) in selected_pairs and (sid, tid) not in done]
    failures = []
    geometry_rows = []
    index_path = directory / "pairs.jsonl"
    started = time.perf_counter()
    initial_done = len(done)
    attempted = 0

    def save_report(status):
        report = dict(status=status, device=str(x.device), pair_batch_size=size, attempted_pairs=attempted,
            successful_pairs=len(done)-initial_done, previously_completed_pairs=initial_done,
            failed_pairs=len(failures), elapsed_seconds=time.perf_counter()-started,
            failures=[{k: v for k, v in row.items() if k != "report"} for row in failures],
            resume_policy="Reuse registered successful pairs; restart unfinished/failed pairs from exact float64 feasible initialization.")
        if geometry_rows:
            report["completed_pair_geometry"] = dict(
                count=len(geometry_rows), population="new successes in this attempt; excludes failures and previously registered pairs",
                statistics={key: dict(min=min(r[key] for r in geometry_rows),
                    mean=sum(r[key] for r in geometry_rows)/len(geometry_rows), max=max(r[key] for r in geometry_rows))
                    for key in ("fitted_pair_effective_patches", "fitted_pair_top1_probability", "fitted_pair_normalized_entropy", "mean_row_cost_std")})
        write_json(directory / "pair_batch_report.json", report)
        log.write_json("pair_batch_report.json", report)
        manifest.update(pair_count=len(done), failed_pair_count=len(failures))
        if index_path.exists():
            manifest["pair_inventory"] = _file_entry(directory, index_path)

    def journal(path, value):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _sync_directory(directory)

    for offset in range(0, len(pending), size):
        batch = pending[offset:offset + size]
        attempted += len(batch)
        batch_start = time.perf_counter()
        batch_done = len(done)
        log.event("pair_batch_started", batch=offset // size, pairs=len(batch), configured_batch_size=size,
                  device=str(x.device), completed_pairs=len(done), total_pairs=len(selected_pairs))
        indices = torch.tensor([(i, j) for i, j, *_ in batch], device=x.device, dtype=torch.long)
        # Costs and cached per-image kernels are gathered on the selected device.
        costs = distance(x[indices[:, 0]], y[indices[:, 1]])
        scales = costs.mean((-1, -2)) if pc["cost_scale"] == "mean" else costs.new_full((len(batch),), pc["cost_scale"])
        # One small transfer outside solver loops. Compare entropy strength with
        # within-row geometric contrast, not just the mean-normalized cost (~1).
        cost_spread = (costs.std(-1, unbiased=False).mean(-1) / scales).detach().cpu().tolist()
        kx, ky = shared["kx"][indices[:, 0]], shared["ky"][indices[:, 1]]
        a = x.new_full((x.shape[1],), 1 / x.shape[1])
        b = y.new_full((y.shape[1],), 1 / y.shape[1])
        a_cpu, b_cpu = a.cpu(), b.cpu()  # Storage metadata, once per batch.
        failed_members = set()

        def fail(member, status, error, report):
            if member in failed_members:
                return
            failed_members.add(member)
            i, j, sid, tid, key = batch[member]
            entry = dict(source_id=sid, target_id=tid, source_index=i, target_index=j,
                status=status, error=error, run_id=log.run_id, batch=offset // size,
                checkpoint=f"plans/pairs/{key}/latest.pt", report=report)
            failures.append(entry)
            journal(directory / "pair_failures.jsonl", entry)
            log.event("pair_failed", **entry)
            if len(failures) == 1:
                # A bounded, exact reproducer when only logs can be copied from the lab.
                example = log.directory / "first_failed_pair.pt"
                try:
                    save_tensor(example, dict(cost=costs[member].detach().cpu(), kx=kx[member].detach().cpu(),
                        ky=ky[member].detach().cpu(), a=a_cpu, b=b_cpu, config=pc, keep_mass=mass,
                        source_id=sid, target_id=tid, report=report, fit_fingerprint=fingerprint))
                    reproduction = f"Exact reproduction inputs: {example}"
                except Exception as error:
                    # Optional evidence must not prevent successful neighbors from saving.
                    reproduction = f"Could not save reproduction inputs: {error}"
                    log.event("failure_reproducer_save_failed", error=str(error), source_id=sid, target_id=tid)
                last = report.get("history", [{}])[-1] if report.get("history") else {}
                print(f"First failed pair: {sid}/{tid}, {status}; inner={last.get('inner', {})}. "
                      f"{reproduction}", flush=True)

        def progress(plans, reports):
            # One bulk transfer only at checkpoint/terminal boundaries. Metrics
            # arrived in one packed transfer from the numerical solver.
            checkpoint_members = [i for i, r in enumerate(reports) if r is not None
                and i not in failed_members and (r["status"] != "running"
                    or r["iterations"] == 1 or r["iterations"] % every == 0)]
            if checkpoint_members:
                picked = torch.tensor(checkpoint_members, device=plans.device)
                snapshots = plans[picked].detach().cpu()
                snapshots = dict(zip(checkpoint_members, snapshots))
            else:
                snapshots = {}
            for member, report in enumerate(reports):
                if report is None or member in failed_members:
                    continue
                i, j, sid, tid, key = batch[member]
                pair_directory = directory / "plans/pairs" / key
                pair_directory.mkdir(parents=True, exist_ok=True)
                record = report["history"][-1] if report["history"] else dict(iteration=0, status=report["status"], error=report["error"])
                with (pair_directory / "iterations.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(record, run_id=log.run_id,
                        elapsed_seconds=time.perf_counter() - started), allow_nan=False) + "\n")
                log.event("solver_iteration", unit=f"pairs/{key}", **record)
                if member not in snapshots:
                    continue
                # Validate each accepted float64 plan before any quantization.
                # Failed states retain their last accepted feasible plan.
                snapshot = dict(plan=snapshots[member], a=a_cpu, b=b_cpu,
                    source_id=sid, target_id=tid, source_index=i, target_index=j,
                    fit_fingerprint=fingerprint, report=report)
                try:
                    validate_plan(snapshot, snapshot["a"], snapshot["b"], mass, pc)
                    snapshot = store_plan_state(snapshot)
                    latest = pair_directory / "latest.pt"
                    save_tensor(latest, snapshot)
                    with latest.open("r+b") as handle:
                        os.fsync(handle.fileno())
                    _sync_directory(pair_directory)
                    if report["status"] == "running":
                        continue
                    if report["status"] != "converged":
                        fail(member, report["status"], "Pair did not converge; last accepted plan retained.", report)
                        continue
                    final = pair_directory.with_suffix(".pt")
                    entry, saved = _verified_save(directory, final, snapshot,
                        lambda s: validate_plan(s, s["a"], s["b"], mass, pc))
                except Exception as error:
                    fail(member, "save_or_validation_failed", f"{type(error).__name__}: {error}", report)
                    continue
                entry.update(source_id=sid, target_id=tid, source_index=i, target_index=j)
                # Register durably before cleanup. If interrupted between these
                # operations, resume validates the journal and finishes cleanup.
                journal(index_path, entry)
                done[sid, tid] = entry
                _cleanup_latest(directory, entry, log)
                geometry = {k: float(v) for k, v in patch_plan_concentration(snapshots[member]).items()}
                geometry.update(mean_row_cost_std=cost_spread[member],
                    reg_over_mean_row_cost_std=pc["reg"]/cost_spread[member] if cost_spread[member] > 0 else None)
                geometry_rows.append(geometry)
                log.append_jsonl("pair_geometry.jsonl", dict(source_id=sid, target_id=tid, **geometry))
                log.event("pair_completed", source_id=sid, target_id=tid, completed_pairs=len(done),
                    total_pairs=len(selected_pairs), status=report["status"], last_iteration=record,
                    storage=saved["storage"], geometry=geometry)

        try:
            solve_partial_batch(costs, kx, ky, keep_mass=mass, config=pc, on_step=progress)
        except BaseException:
            save_report("interrupted_or_failed")
            raise
        elapsed = time.perf_counter() - batch_start
        log.event("pair_batch_finished", batch=offset // size, pairs=len(batch),
            successful_pairs=len(done)-batch_done, failed_pairs=len(failed_members),
            batch_seconds=elapsed, pairs_per_second=len(batch)/elapsed,
            timing_includes="cost construction, solver, validation, checkpoint and artifact I/O")
        print(f"Pairs {len(done)}/{len(selected_pairs)} accepted; {len(failures)} failed this attempt", flush=True)
        save_report("running")
    save_report("failed" if failures else "completed")
    if failures:
        raise RuntimeError(f"{len(failures)} partial pairs failed; {len(done)} successful pairs saved. "
                           f"See {directory / 'pair_failures.jsonl'} and pair_batch_report.json.")
