"""Persist enough context to diagnose successful, failed and interrupted runs."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infoot_vit.infoot_helper.run_logging import RunLog
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, validate_config, resource_estimate
from test_infoot_vit_mapping import banks, config, cpu_threads


def attempts(directory):
    return sorted((directory / "logs").iterdir())


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def events(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def test_run_logger_persists_error_and_keeps_console(tmp_path, capsys):
    with pytest.raises(ValueError, match="bad solve"):
        with RunLog(tmp_path, "test", {"path": tmp_path}) as log:
            print("iteration progress", flush=True)
            print("warning details", file=sys.stderr)
            log.event("iteration", iteration=3)
            raise ValueError("bad solve")
    output = capsys.readouterr()
    assert "iteration progress" in output.out and "bad solve" in output.err
    path = attempts(tmp_path)[0]
    assert "iteration progress" in (path / "stdout.log").read_text()
    assert "warning details" in (path / "stderr.log").read_text()
    assert "ValueError: bad solve" in (path / "traceback.txt").read_text()
    assert read(path / "run.json")["status"] == "failed"
    assert events(path)[-1]["status"] == "failed"


def test_fit_success_preflight_failure_and_resume_attempt_logs(tmp_path, banks):
    directory = fit_mapping(config("whole_map"), root=tmp_path)
    path = attempts(directory)[0]
    assert read(path / "run.json")["status"] == "completed"
    assert read(path / "inspection.json")["source_id"] == banks[0].artifact_id
    assert any(row["event"] == "solver_iteration" for row in events(path))
    assert read(path / "fit_report.json")["models"]["image"]["status"] == "converged"
    original_manifest = (directory / "manifest.json").read_bytes()
    fit_mapping(config("whole_map"), root=tmp_path, resume=directory)
    assert len(attempts(directory)) == 2
    assert (directory / "manifest.json").read_bytes() == original_manifest
    c = config("whole_map"); c["source_bank"] = "missing"
    with pytest.raises(FileNotFoundError):
        fit_mapping(c, root=tmp_path)
    failed = next(d for d in (tmp_path / "outputs/infoot_vit").iterdir() if d != directory)
    assert read(attempts(failed)[0] / "run.json")["status"] == "failed"
    assert (attempts(failed)[0] / "traceback.txt").is_file()


def test_keyboard_interrupt_and_truncated_pair_journal_resume(tmp_path, banks, monkeypatch):
    import importlib
    module = importlib.import_module("infoot_vit.infoot_helper.fit_mapping")
    original = module.solve_partial
    count = 0
    def interrupt(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "solve_partial", interrupt)
    with pytest.raises(KeyboardInterrupt):
        fit_mapping(config("grouped_partial"), root=tmp_path)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    assert read(directory / "manifest.json")["status"] == "interrupted"
    assert read(attempts(directory)[0] / "run.json")["status"] == "interrupted"
    with (directory / "pairs.jsonl").open("ab") as handle:
        handle.write(b'{"source_id": "partial')
    monkeypatch.setattr(module, "solve_partial", original)
    fit_mapping(config("grouped_partial"), root=tmp_path, resume=directory)
    assert read(directory / "manifest.json")["pair_count"] == 6
    assert any((d / "pairs_truncated_tail.bin").is_file() for d in attempts(directory))


@pytest.mark.parametrize("field,value", [("top_k_images", 1.5), ("top_k_images", True), ("seed", 1.2), ("query_chunk_size", True)])
def test_bad_projection_numbers_rejected_before_fit(field, value):
    c = config("whole_map"); c["projection"][field] = value
    with pytest.raises(ValueError):
        validate_config(c)


def test_bad_resource_limits_rejected():
    c = config("whole_map"); c["resources"] = {"max_working_gib": float("nan")}
    with pytest.raises(ValueError):
        validate_config(c)


def test_disk_limit_includes_mandatory_latest_plans(tmp_path, banks):
    r = resource_estimate((2, 4, 3), (3, 4, 3), "grouped_partial")
    assert r["required_plan_storage_bytes"] > 2 * r["plan_storage_bytes"]
    c = config("grouped_partial")
    c["resources"] = {"max_plan_gib": 1.5 * r["plan_storage_gib"]}
    with pytest.raises(MemoryError, match="required_plan_storage_gib"):
        fit_mapping(c, root=tmp_path)


def test_reporting_failure_does_not_double_count_fit_time(tmp_path, banks, monkeypatch):
    import importlib
    module = importlib.import_module("infoot_vit.infoot_helper.fit_mapping")
    now = [100.]
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    original = module.BalancedModel.fit.__func__
    def fitted(cls, *args, **kwargs):
        result = original(cls, *args, **kwargs)
        now[0] = 150.
        return result
    monkeypatch.setattr(module.BalancedModel, "fit", classmethod(fitted))
    def reporting_failure(*args):
        now[0] = 160.
        raise RuntimeError("report failed")
    monkeypatch.setattr(module, "summarize_fit", reporting_failure)
    with pytest.raises(RuntimeError, match="report failed"):
        fit_mapping(config("whole_map"), root=tmp_path)
    directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    manifest = read(directory / "manifest.json")
    assert manifest["status"] == "failed" and manifest["fit_seconds"] == 60.
    assert "artifact_id" not in manifest


def test_cli_loading_failure_logged_before_mapper_exists(tmp_path):
    from infoot_vit.infoot_test import main
    output = tmp_path / "failed-test"
    with pytest.raises(FileNotFoundError):
        main(["--mapping", str(tmp_path / "missing"), "--output-dir", str(output), "--threads", "1"])
    assert read(attempts(output)[0] / "run.json")["status"] == "failed"
    assert (attempts(output)[0] / "traceback.txt").is_file()
