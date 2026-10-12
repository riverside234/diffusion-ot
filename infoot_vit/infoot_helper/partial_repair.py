"""Explicit, provenance-preserving repair of a stopped dense partial pair fit."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil

import torch

from .feature_bank import FeatureBank, checked_file, file_hash, write_json
from .fit_mapping import fit_identity, fit_mapping, inspect_fit, _file_entry, _verified_save, _revision
from .mapping_manifest import load_mapping_manifest
from .partial import OBJECTIVE
from .partial_batch import VERSION as BATCH_VERSION
from .storage import validate_plan, load_kernels

# Only the reviewed v5/v6 implementations, or the current implementation.
REPAIR_PREDECESSOR = {
    "partial_batch.py": "1ec14d02ba871e93275dc392c0ccc4b8f0bccfa5203c420e58f68bda39e9f3cb",
    "fit_mapping.py": "bb42c285b180288529ab74ae086f7e8933f8bb295f25b3d9aaadeca80ba50ee9",
}
REPAIR_PREDECESSOR_V6 = {
    "partial_batch.py": "56df523725f66358c1f4290b675d1992719ea5db4f8fcb07cfcc3960334531c2",
    "fit_mapping.py": "4100b67e08841d69a8ae5e34416bbdc099da7db3bfa389f9e335bf8cd7148c28",
}


def _validate_reused_pair(state, config, shape):
    pc, mass = config["partial"]["solver"], config["partial"]["keep_mass"]
    report = state["report"]
    if (report["status"] != "converged" or report["config"] != pc
            or report["objective_version"] != OBJECTIVE or report["keep_mass"] != mass
            or state["plan"].shape != shape or not report.get("history")):
        raise ValueError("Repair requires an unchanged, converged dense partial pair.")
    expected_a = torch.full((shape[0],), 1 / shape[0], dtype=torch.float64)
    expected_b = torch.full((shape[1],), 1 / shape[1], dtype=torch.float64)
    if not torch.equal(state["a"], expected_a) or not torch.equal(state["b"], expected_b):
        raise ValueError("Repair pair capacities must match the original uniform patch capacities.")
    last = report["history"][-1]
    inner = last["inner"]
    tolerance = inner.get("certificate_tolerance", pc["inner_tolerance"])
    def bounded(value, limit):
        return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= limit
    if (last["status"] != "converged" or inner["failure_code"] != 0
            or not bounded(tolerance, pc["inner_tolerance"]) or tolerance == 0
            or (pc["lam"] and not bounded(last.get("plan_delta_l1"), pc["outer_tolerance"]))):
        raise ValueError("Repair pair lacks a strict undamped convergence certificate.")
    if mass != 1.:
        if (inner.get("convergence_reason") not in {"primal_dual_certificate", "legacy_residual_and_gap"}
                or any(not bounded(inner.get(key), tolerance) for key in
                       ("kkt_error", "relative_duality_gap", "relative_plan_delta_l1"))):
            raise ValueError("Repair pair lacks strict inner primal-dual/KKT certificates.")
    elif not bounded(inner.get("error"), tolerance):
        raise ValueError("Repair balanced pair lacks its strict marginal certificate.")
    validate_plan(state, expected_a, expected_b, mass, pc)


def prepare_partial_repair(source, output, *, root):
    """Create a NEW fit; reuse certified plans without weakening normal resume.

    Only the batched solver and its fit orchestration may have changed. Saved
    configuration, banks, support order and runtime must match exactly. Pair
    tensors are preserved; wrappers acquire the new fingerprint and explicit
    source provenance. Original artifacts are never written or deleted.
    """
    source, output, root = (Path(p).resolve() for p in (source, output, root))
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Repair output must be a separate new directory outside the source fit.")
    if output.exists():
        raise FileExistsError(f"Repair output already exists: {output}; use --resume for a prepared repair.")
    snapshot = load_mapping_manifest(source, allow_failed_pairs=True)
    if snapshot.get("status") != "failed" or "incomplete_fit" not in snapshot:
        raise ValueError("Repair requires a stopped, failed batched grouped_partial fit.")
    if BATCH_VERSION != "dense_partial_infoot_batch_v7_certified_progress":
        raise ValueError("This repair migration requires the v7 certified-progress solver.")
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    config = original["config"]
    if not config.get("pair_batch_size"):
        raise ValueError("Repair requires the dense batched partial solver.")
    inspection = inspect_fit(config, root)
    if inspection["config"] != config or inspection["runtime"] != original["runtime"]:
        raise ValueError("Repair requires the exact saved configuration and runtime.")
    banks = [FeatureBank.load(root / config[key]) for key in ("source_bank", "target_bank")]
    for name, bank in zip(("source", "target"), banks):
        if (bank.artifact_id != original[name + "_bank"]["artifact_id"]
                or inspection["sampling"][name] != original["sampling"][name]
                or original[name + "_ids"] != inspection["sampling"][name]["ordered_ids"]):
            raise ValueError("Repair bank identity or sampled support order changed.")
    banks = [bank.subset(original[name + "_ids"]) for name, bank in zip(("source", "target"), banks)]
    fingerprint, implementation = fit_identity(config, *banks, inspection["runtime"])
    previous = original["solver_implementation_sha256"]
    changed = {name for name in previous.keys() | implementation.keys()
               if previous.get(name) != implementation.get(name)}
    if changed - REPAIR_PREDECESSOR.keys() or any(previous.get(name) not in
            {REPAIR_PREDECESSOR[name], REPAIR_PREDECESSOR_V6[name]} for name in changed):
        raise ValueError(f"Repair cannot migrate unrelated implementation changes: {sorted(changed)}")
    # Verify the immutable support artifacts before preparing any output.
    entries = list(original["models"].values()) + [original["pair_selection"], original["pair_kernels"]]
    for entry in entries:
        checked_file(source, entry)
    router = torch.load(checked_file(source, original["models"]["image"]), map_location="cpu", weights_only=True)
    if (router["status"] != "converged" or router["config"] != config["solver"]
            or router["source_ids"] != original["source_ids"] or router["target_ids"] != original["target_ids"]):
        raise ValueError("Repair router status, settings or support identity mismatch.")
    plan = router["plan"].double()
    validate_plan(router, plan.new_full((len(plan),), 1 / len(plan)),
                  plan.new_full((plan.shape[1],), 1 / plan.shape[1]), 1., config["solver"], balanced=True)
    load_kernels(torch.load(checked_file(source, original["pair_kernels"]), map_location="cpu", weights_only=True))
    inventory = [json.loads(line) for line in checked_file(source, original["pair_inventory"]).read_text(encoding="utf-8").splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    for entry in entries:
        destination = output / entry["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checked_file(source, entry), destination)
        checked_file(output, entry)
    archived = {}
    for name in ("manifest.json", "pairs.jsonl", "pair_failures.jsonl", "pair_batch_report.json"):
        destination = output / "repair_source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
        archived[name] = _file_entry(output, destination)
    if (source / "image_report.json").exists():
        shutil.copy2(source / "image_report.json", output / "image_report.json")
    copied = []
    pc, mass = config["partial"]["solver"], config["partial"]["keep_mass"]
    for entry in inventory:
        state = torch.load(checked_file(source, entry), map_location="cpu", weights_only=True)
        if (state["fit_fingerprint"] != original["fit_fingerprint"]
                or state["report"]["status"] != "converged"
                or state["report"]["objective_version"] != OBJECTIVE
                or state["report"]["config"] != pc
                or state["source_id"] != entry["source_id"] or state["target_id"] != entry["target_id"]
                or state["source_index"] != entry["source_index"] or state["target_index"] != entry["target_index"]
                or not 0 <= state["source_index"] < len(original["source_ids"])
                or not 0 <= state["target_index"] < len(original["target_ids"])
                or original["source_ids"][state["source_index"]] != state["source_id"]
                or original["target_ids"][state["target_index"]] != state["target_id"]):
            raise ValueError("Repair pair identity, settings or convergence mismatch.")
        _validate_reused_pair(state, config, (banks[0].features.shape[1], banks[1].features.shape[1]))
        state["fit_fingerprint"] = fingerprint
        state["reused_from"] = dict(source_directory=str(source), fit_fingerprint=original["fit_fingerprint"],
                                   file=entry["file"], sha256=entry["sha256"])
        updated, _ = _verified_save(output, output / entry["file"], state,
            lambda s: _validate_reused_pair(s, config, (banks[0].features.shape[1], banks[1].features.shape[1])))
        copied.append(dict(entry, **updated))
    index_path = output / "pairs.jsonl"
    index_path.write_text("".join(json.dumps(entry) + "\n" for entry in copied), encoding="utf-8")
    # Check source snapshots again; do not accept a concurrently resumed source.
    for name, entry in archived.items():
        if file_hash(source / name) != entry["sha256"]:
            raise ValueError("Source fit changed while preparing repair; stop fitting before repair.")
    if archived["manifest.json"]["sha256"] != snapshot["incomplete_fit"]["original_manifest_sha256"]:
        raise ValueError("Source manifest changed while preparing repair.")
    manifest = deepcopy(original)
    for key in ("error", "artifact_id", "failed_pair_count"):
        manifest.pop(key, None)
    manifest.update(status="interrupted", fit_fingerprint=fingerprint, solver_implementation_sha256=implementation,
                    local_revision=_revision(),
                    pair_inventory=_file_entry(output, index_path), pair_count=len(copied), fit_seconds=0.)
    for name, bank in zip(("source", "target"), banks):
        manifest[name + "_bank"]["path"] = os.path.relpath(bank.path, output)
    manifest["repair"] = dict(policy="strict_failed_pairs_new_artifact_v1", source_directory=str(source),
        source_fit_fingerprint=original["fit_fingerprint"], source_files=archived,
        source_solver_implementation_sha256=previous, changed_implementation_files=sorted(changed),
        repair_implementation_sha256=file_hash(Path(__file__)), source_local_revision=original.get("local_revision"),
        reused_successful_pairs=len(copied), pairs_to_retry=snapshot["incomplete_fit"]["failed_pair_count"],
        source_fit_seconds=original.get("fit_seconds"),
        note="Successful float32 plans and reports reused unchanged; only pair identity wrappers reserialized. "
             "Failed pairs restart in float64 under the new batched implementation. Original fit is unchanged.")
    write_json(output / "manifest.json", manifest)
    return output


def repair_partial_fit(source, output, *, root):
    directory = prepare_partial_repair(source, output, root=root)
    config = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["config"]
    return fit_mapping(config, root=root, resume=directory)
