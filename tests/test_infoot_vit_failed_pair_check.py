"""The cheap failed-pair check must not promote incomplete or uncertified fits."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from test_infoot_vit_mapping import banks, config, cpu_threads
from infoot_vit import check_failed_pairs as check
from infoot_vit.infoot_helper.feature_bank import file_hash, write_json


@pytest.fixture
def request_data(tmp_path, banks):
    source, target, _ = banks
    data = dict(schema="infoot_displacement_run_analysis_v1", configuration=config("grouped_partial"),
        counts=dict(failed=2), source_bank_id=source.artifact_id, target_bank_id=target.artifact_id,
        failed_pair_terminal_evidence=[dict(source_id=source.ids[i], target_id=target.ids[i],
            status="line_search_stalled") for i in range(2)])
    data["configuration"]["partial"]["solver"].update(lam=.035, inner_acceleration="newton")
    path = tmp_path / "analysis.json"
    write_json(path, data)
    return SimpleNamespace(analysis=path, source_bank=source.path, target_bank=target.path,
        output=tmp_path / "results/check.json", device="cpu", batch_size=2), data


def test_exact_pair_replay_is_strict_read_only_and_writes_one_report(request_data, banks, monkeypatch):
    args, _ = request_data
    before = {str(p):file_hash(p) for bank in banks for p in bank.path.rglob("*") if p.is_file()}
    seen = []
    real = check.solve_partial_batch
    def observed(cost, kx, ky, **kwargs):
        assert cost.dtype == kx.dtype == ky.dtype == torch.float64
        seen.append(cost.shape[0])
        return real(cost, kx, ky, **kwargs)
    monkeypatch.setattr(check, "solve_partial_batch", observed)
    report = check.run_check(args)
    assert seen == [2] and report["status"] == "passed"
    assert report["successful_pairs"] == report["requested_pairs"] == 2
    assert report["failed_pairs"] == 0
    assert list(args.output.parent.iterdir()) == [args.output]
    assert report == json.loads(args.output.read_text())
    assert before == {str(p):file_hash(p) for bank in banks for p in bank.path.rglob("*") if p.is_file()}
    for row in report["pairs"]:
        assert row["final_iteration"]["plan_delta_l1"] <= report["partial_solver"]["outer_tolerance"]
    with pytest.raises(FileExistsError):
        check.run_check(args)


@pytest.mark.parametrize("damage", ["bank", "validation", "duplicate", "count", "unknown_id"])
def test_changed_input_rejected_before_creating_report(request_data, banks, damage):
    args, data = request_data
    if damage == "bank":
        data["target_bank_id"] = "different"
    elif damage == "validation":
        args.source_bank = banks[2].path
        data["source_bank_id"] = banks[2].artifact_id
    elif damage == "duplicate":
        data["failed_pair_terminal_evidence"][1] = deepcopy(data["failed_pair_terminal_evidence"][0])
    elif damage == "count":
        data["counts"]["failed"] += 1
    else:
        data["failed_pair_terminal_evidence"][0]["source_id"] = "missing"
    write_json(args.analysis, data)
    with pytest.raises(ValueError):
        check.run_check(args)
    assert not args.output.exists()


@pytest.mark.parametrize("damage", ["solver_failure", "false_convergence"])
def test_cli_stops_full_experiment_after_any_failure(request_data, monkeypatch, damage):
    args, _ = request_data
    real = check.solve_partial_batch
    def fail(cost, kx, ky, **kwargs):
        if damage == "solver_failure":
            cost = cost.clone()
            cost[0, 0, 0] = float("nan")
        plans, states = real(cost, kx, ky, **kwargs)
        if damage == "false_convergence":
            states[0]["history"][-1]["plan_delta_l1"] = 1.
        return plans, states
    monkeypatch.setattr(check, "solve_partial_batch", fail)
    exit_code = check.main(["--analysis", str(args.analysis), "--source-bank", str(args.source_bank),
        "--target-bank", str(args.target_bank), "--output", str(args.output), "--device", "cpu"])
    report = json.loads(args.output.read_text())
    assert exit_code == 2 and report["status"] == "failed" and report["failed_pairs"] == 1
    assert report["pairs"][0]["status"] == ("invalid_input" if damage == "solver_failure" else "certificate_failed")
    assert "failure_history" in report["pairs"][0]


def test_interrupted_check_preserves_one_failure_report(request_data, monkeypatch):
    args, _ = request_data
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic device failure")
    monkeypatch.setattr(check, "solve_partial_batch", fail)
    with pytest.raises(RuntimeError, match="device failure"):
        check.run_check(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed" and "synthetic device failure" in report["error"]
    assert list(args.output.parent.iterdir()) == [args.output]
