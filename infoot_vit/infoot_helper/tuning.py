"""Console-only image-router tuning; no output directories or artifact writes."""
from pathlib import Path
import json
import time

import torch

from .conditional import BalancedModel
from .device import resolve_device
from .feature_bank import FeatureBank, compatible_banks, digest
from .fit_mapping import inspect_fit, resource_estimate
from .pair_selection import select_pairs


def _pair_selection_preview(model, config, source_ids, target_ids):
    """Preview the existing selection rule at projection h, without saving edges."""
    start = time.perf_counter()
    multiplier = config["projection"]["bandwidth_multiplier"]
    router = BalancedModel(model.source, model.target, model.state, multiplier)
    selection = select_pairs(router, source_ids, target_ids, config["fit_pair_top_k"])
    # Only compact selected probabilities cross to CPU via the existing selector;
    # float64 kernel/scoring operations run on the router's configured device.
    probabilities = torch.tensor([row["probabilities"] for row in selection["rows"]], dtype=torch.float64)
    retained = probabilities.sum(1)
    normalized = probabilities / retained[:, None]
    return dict(projection_h=router.h, projection_bandwidth_multiplier=multiplier,
        effective_k=selection["effective_k"], pair_count=selection["pair_count"],
        mean_retained_probability=float(retained.mean()), median_retained_probability=float(retained.median()),
        min_retained_probability=float(retained.min()), max_retained_probability=float(retained.max()),
        mean_discarded_probability=float(1-retained.mean()),
        mean_renormalized_top1_probability=float(normalized.max(1).values.mean()),
        mean_renormalized_effective_targets=float((-torch.special.xlogy(normalized, normalized).sum(1)).exp().mean()),
        seconds=time.perf_counter()-start, plan_precision="float64_before_storage",
        interpretation="Training-source conditional routing at projection h; retained/discarded mass precedes edge renormalization. "
            "No partial OT rejection or held-out quality measurement; no pairs saved. Full-fit float32 storage can slightly change values/ties.")


def tune_image_router(raw, *, root, lowrank=False, log_every=25):
    """Run the existing float64 router on the same selected training support.

    Patch solvers/kernel audits, RunLog and checkpoint utilities are deliberately
    outside this path. Nonconvergence is returned as a status, never accepted as
    a fitted mapper; numerical errors propagate to the caller's terminal. A
    converged dense partial router previews its top-K selection at projection h.
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
    if not lowrank and c["mode"] == "grouped_partial" and model.state["status"] == "converged":
        preview = _pair_selection_preview(model, c, banks[0].ids, banks[1].ids)
        report["pair_selection_preview"] = preview
        print(f"Top-{preview['effective_k']} selection preview: retained={preview['mean_retained_probability']:.2%}, "
              f"discarded={preview['mean_discarded_probability']:.2%}, projection h={preview['projection_h']:g}, "
              f"renormalized top1={preview['mean_renormalized_top1_probability']:.2%}. No pairs saved.", flush=True)
    print("Tuning result (not a saved mapping):", flush=True)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report
