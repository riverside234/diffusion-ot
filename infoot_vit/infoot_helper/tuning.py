"""Console-only image-router tuning; no output directories or artifact writes."""
from pathlib import Path
import json
import time

import torch

from .conditional import BalancedModel
from .device import resolve_device
from .feature_bank import FeatureBank, compatible_banks, digest
from .fit_mapping import inspect_fit, resource_estimate


def tune_image_router(raw, *, root, lowrank=False, log_every=25):
    """Run the existing float64 router on the same selected training support.

    Patch solvers/kernel audits, RunLog and checkpoint utilities are deliberately
    outside this path. Nonconvergence is returned as a status, never accepted as
    a fitted mapper; numerical errors propagate to the caller's terminal.
    """
    if type(log_every) is not int or log_every < 1:
        raise ValueError("Tuning log interval must be a positive integer.")
    root = Path(root).resolve()
    if not lowrank and raw.get("mode", "patch_global") == "patch_global":
        raise ValueError("patch_global has no image router; --tune requires a grouped or whole_map config.")
    if raw.get("image_model") or raw.get("patch_model"):
        raise ValueError("--tune fits a fresh router; remove image_model/patch_model reuse options.")
    if lowrank:
        from ..lowrank.experiment import inspect_fit as inspect_lowrank
        inspection = inspect_lowrank(raw, root)
    else:
        inspection = inspect_fit(raw, root)
    c = inspection["config"]
    solver = c["image_solver" if lowrank else "solver"]
    device = resolve_device(c.get("device", "cpu"))
    if lowrank:
        torch.set_num_threads(c["threads"])
    sampling = inspection["sampling"]
    # Apply only the image-router memory limit, not full patch/storage budgets.
    representation = json.loads((root/c["source_bank"]/"manifest.json").read_text(encoding="utf-8"))["representation"]
    p, d = representation["grid"][0]*representation["grid"][1], representation["dim"]
    counts = [sampling[name]["selected_count"] for name in ("source", "target")]
    estimate = resource_estimate((counts[0], p, d), (counts[1], p, d), "whole_map")["estimated_working_gib"]
    if estimate >= c["resources"]["max_working_gib"]:
        raise MemoryError(f"Image-router memory estimate {estimate:.3g} GiB exceeds the configured limit.")
    expected = inspection["bank_ids"] if lowrank else [inspection["source_id"], inspection["target_id"]]
    banks = []
    for key, name, identity in zip(("source_bank", "target_bank"), ("source", "target"), expected):
        bank = FeatureBank.load(root/c[key])
        if bank.artifact_id != identity:
            raise ValueError("Training bank changed after tuning inspection.")
        banks.append(bank.subset(sampling[name]["ordered_ids"]))
    compatible_banks(*banks)
    print("Image-router tuning: console only; no logs/checkpoints/plans saved; patch fitting skipped.", flush=True)
    context = dict(mode=c["mode"], device=str(device), solver=solver,
        sampling={name: dict(count=counts[i], seed=sampling[name]["seed"],
                            ordered_ids_sha256=digest(sampling[name]["ordered_ids"]))
                  for i, name in enumerate(("source", "target"))},
        estimated_router_working_gib=estimate, source_bank_id=expected[0], target_bank_id=expected[1])
    print(json.dumps(context, indent=2, allow_nan=False), flush=True)

    def progress(plan, report):
        if report["iteration"] == 1 or report["iteration"] % log_every == 0 or report["status"] != "running":
            print(f"image FusedInfoOT {report['iteration']}/{solver['max_outer_steps']}: "
                  f"objective={report['objective']:.8g}, cost={report['cost']:.8g}, "
                  f"mi_term={report['mi_term']:.8g}, entropy_term={report['entropy_term']:.8g}, "
                  f"lam={solver['lam']:.6g}, reg={solver['reg']:.6g}, "
                  f"delta={report['plan_delta_l1']:.3g}, "
                  f"effective_targets={report['mean_row_effective_targets']:.1f}, status={report['status']}", flush=True)

    start = time.perf_counter()
    source, target = [b.features.to(device=device, dtype=torch.float64).flatten(1) for b in banks]
    model = BalancedModel.fit(source, target, solver, on_step=progress)
    report = dict(context, status=model.state["status"], seconds=time.perf_counter()-start,
        cost_scale=model.state["cost_scale"], last_iteration=model.state["history"][-1],
        kernel_diagnostics=model.state["kernel_diagnostics"])
    print("Tuning result (not a saved mapping):", flush=True)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report
