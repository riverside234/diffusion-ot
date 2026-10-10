"""Read-only identities for complete fits and explicitly allowed failed-pair subsets."""
from copy import deepcopy
import json
from pathlib import Path

from .feature_bank import checked_file, digest, file_hash
from .pair_selection import selection_edges
from .storage import VERSION


def load_mapping_manifest(directory, *, allow_failed_pairs=False):
    directory = Path(directory)
    path = directory / "manifest.json"
    original_hash = file_hash(path)
    m = json.loads(path.read_text(encoding="utf-8"))
    schemas = {"siglip_infoot_mapping_v2", "siglip_infoot_mapping_v3",
               "siglip_lowrank_grouped_patch_v1", "siglip_lowrank_grouped_partial_v1"}
    if m.get("schema") not in schemas:
        raise ValueError("Mapping artifact has an unsupported or legacy schema.")
    if m.get("status") == "complete":
        if digest({k: v for k, v in m.items() if k != "artifact_id"}) != m.get("artifact_id"):
            raise ValueError("Mapping manifest fingerprint changed.")
        return m
    if not allow_failed_pairs:
        raise ValueError(f"Mapping artifact is incomplete (status={m.get('status')!r}). "
                         "For a failed batched grouped_partial fit, explicitly use --allow-failed-pairs to test its successful pairs.")
    if (m.get("status") != "failed" or m["schema"] != "siglip_infoot_mapping_v3"
            or m.get("config", {}).get("mode") != "grouped_partial"):
        raise ValueError("--allow-failed-pairs requires a stopped, failed batched grouped_partial fit; active/interrupted fits are not accepted.")
    # Failed fits have no completed artifact_id. Check the original fit identity
    # using SAVED implementation hashes, not the currently installed solver code.
    c = m["config"]
    expected_fingerprint = digest(dict(config=c, source=m["source_bank"]["artifact_id"],
        target=m["target_bank"]["artifact_id"], source_ids=m["source_ids"], target_ids=m["target_ids"],
        storage=VERSION, implementation=m["solver_implementation_sha256"], runtime=m["runtime"]))
    if m.get("artifact_id") is not None or m.get("fit_fingerprint") != expected_fingerprint:
        raise ValueError("Failed-fit fingerprint mismatch; cannot construct a successful-pair snapshot.")
    selection = json.loads(checked_file(directory, m["pair_selection"]).read_text(encoding="utf-8"))
    if selection["router_sha256"] != m["models"]["image"]["sha256"]:
        raise ValueError("Saved pair selection has a different image router.")
    selected = selection_edges(selection, m["source_ids"], m["target_ids"], c["fit_pair_top_k"])
    entries = [json.loads(line) for line in checked_file(directory, m["pair_inventory"]).read_text(encoding="utf-8").splitlines()]
    successes = {(e["source_id"], e["target_id"]) for e in entries}
    if len(successes) != len(entries) or len(successes) != m["pair_count"] or not successes <= selected:
        raise ValueError("Invalid successful-pair inventory; it cannot be treated as failed pairs.")
    if not successes:
        raise ValueError("No successful pairs are available for mapping.")
    # The final batch report identifies this attempt. Historical failure-journal
    # rows may refer to pairs subsequently recovered; never skip those successes.
    report_path, journal_path = directory / "pair_batch_report.json", directory / "pair_failures.jsonl"
    report_hash, journal_hash = file_hash(report_path), file_hash(journal_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = report["failures"]
    failed = {(e["source_id"], e["target_id"]) for e in failures}
    if (report["status"] != "failed" or not failed or len(failed) != len(failures)
            or report["failed_pairs"] != len(failed) or m.get("failed_pair_count") != len(failed)
            or report["successful_pairs"] + report["previously_completed_pairs"] != len(successes)
            or report["attempted_pairs"] != report["successful_pairs"] + len(failed)
            or selected - successes != failed or successes & failed):
        raise ValueError("Missing pairs must exactly match the final failed-pair report; unknown/missing successes cannot be skipped.")
    journal = {digest({k: v for k, v in json.loads(line).items() if k != "report"})
               for line in journal_path.read_text(encoding="utf-8").splitlines()}
    source_index = {sid: i for i, sid in enumerate(m["source_ids"])}
    target_index = {tid: j for j, tid in enumerate(m["target_ids"])}
    for row in failures:
        if (row["source_index"] != source_index.get(row["source_id"])
                or row["target_index"] != target_index.get(row["target_id"])
                or not row.get("status") or row["status"] in {"running", "converged"}
                or digest(row) not in journal):
            raise ValueError("Failed-pair identity/status does not match the failure journal.")
    snapshot = deepcopy(m)
    snapshot["incomplete_fit"] = dict(policy="skip_recorded_failed_pairs_v1", original_status=m["status"],
        original_manifest_sha256=original_hash, fit_fingerprint=m["fit_fingerprint"],
        selected_pair_count=len(selected), successful_pair_count=len(successes), failed_pair_count=len(failed),
        successful_pair_fraction=len(successes)/len(selected),
        failures=sorted(failures, key=lambda r: (r["source_id"], r["target_id"])),
        failure_report=dict(file=report_path.name, sha256=report_hash),
        failure_journal=dict(file=journal_path.name, sha256=journal_hash))
    # This is a test snapshot identity, NOT a claim that the fit completed.
    snapshot["artifact_id"] = digest(snapshot)
    if (file_hash(path) != original_hash or file_hash(report_path) != report_hash
            or file_hash(journal_path) != journal_hash):
        raise ValueError("Fit changed while creating the test snapshot; stop fitting before testing failed pairs.")
    return snapshot
