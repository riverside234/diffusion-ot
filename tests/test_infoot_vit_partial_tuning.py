"""Partial tuning exercises real solvers without altering fitted artifacts."""
from copy import deepcopy
import json
import random

import pytest
import torch
import yaml

from test_infoot_vit_mapping import banks, config, cpu_threads
from test_infoot_vit_tuning import files
from infoot_vit import infoot_fit
from infoot_vit.infoot_helper import partial_tuning as tune
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, validate_config
from infoot_vit.infoot_helper.partial import distance, kernel_state, solve_partial


def recipe():
    c = config("grouped_partial")
    c.update(device="cpu", fit_pair_top_k=2, pair_batch_size=2)
    c["partial"]["solver"].update(lam=.01)
    c["projection"].update(bandwidth_multiplier=.5, patch_bandwidth=.2)
    return c


def test_exact_pairs_match_serial_and_chunk_invariance_without_writes(tmp_path, banks, monkeypatch, capsys):
    c = recipe()
    before = files(tmp_path)
    calls = []
    original = tune.solve_partial_batch
    def observed(cost, kx, ky, **kw):
        calls.append(len(cost))
        assert all(v.dtype == torch.float64 and v.device.type == "cpu" for v in (cost, kx, ky))
        return original(cost, kx, ky, **kw)
    monkeypatch.setattr(tune, "solve_partial_batch", observed)
    report = tune.tune_partial(c, root=tmp_path, pair_limit=3, seed=7, log_every=1)
    assert calls == [2, 1]
    assert report["status"] == "converged" and report["summary"]["successful_pairs"] == 3
    assert report["image_projection_h"] == pytest.approx(.35)
    assert report["patch_projection_h"] == .2
    assert report["pair_sampling"]["population_pairs"] == 4
    for row in report["pairs"]:
        x = banks[0].features[banks[0].ids.index(row["source_id"])]
        y = banks[1].features[banks[1].ids.index(row["target_id"])]
        sx, sy = kernel_state(x, .7), kernel_state(y, .7)
        plan, expected = solve_partial(distance(x, y), sx[0], sy[0], keep_mass=.8, config=c["partial"]["solver"])
        assert expected["status"] == "converged"
        actual = row["last_iteration"]
        for name in ("objective", "cost", "information", "entropy"):
            assert actual[name] == pytest.approx(expected["history"][-1][name], abs=2e-9)
        assert actual["objective"] == pytest.approx(actual["cost"]+actual["mi_term"]+actual["entropy_term"], abs=1e-12)
        assert row["feasibility"]["mass"] == pytest.approx(.8, abs=1e-9)
        probe = tune._projection_probe(x, y, plan, sx, sy, validate_config(c)["projection"])
        assert row["projection"]["projected_effective_patches"] == pytest.approx(probe["projected_effective_patches"], abs=1e-7)
        assert row["geometry"]["fitted_pair_effective_patches"] >= 1
    c["pair_batch_size"] = 1
    again = tune.tune_partial(c, root=tmp_path, pair_limit=3, seed=7)
    assert again["pair_sampling"] == report["pair_sampling"]
    for a,b in zip(again["pairs"], report["pairs"]):
        assert a["last_iteration"]["objective"] == pytest.approx(b["last_iteration"]["objective"], abs=1e-12)
    assert files(tmp_path) == before
    output = capsys.readouterr().out
    assert "mi_term=" in output and "kkt_error=" in output and "threshold:valid_fraction" in output
    assert '"last_iteration"' not in output and "Partial pair 1/" not in output
    summary_text = output.split("Partial tuning summary (not a saved mapping):")[-1]
    assert len(summary_text.splitlines()) <= 20
    assert report["pair_sampling"]["ordered_pairs_sha256"][:12] in summary_text


def test_saved_selection_reused_with_changed_partial_settings_and_json_report(tmp_path, banks, monkeypatch, capsys):
    c = recipe()
    directory = fit_mapping(c, root=tmp_path)
    before = files(directory)
    c["partial"]["solver"].update(h=.9, reg=.3)
    c["partial"]["keep_mass"] = .75
    monkeypatch.setattr(tune.BalancedModel, "fit", lambda *a, **kw: pytest.fail("Refitted frozen router"))
    monkeypatch.setattr(tune, "select_pairs", lambda *a, **kw: pytest.fail("Reselected saved pairs"))
    output = tmp_path / "trial.json"
    report = tune.tune_partial(c, root=tmp_path, from_fit=directory, pair_limit=64, report_path=output)
    assert report["status"] == "converged" and report["pair_sampling"]["selected_pairs"] == 4
    assert report["selection"]["kind"] == "saved_selection" and report["partial_solver"]["h"] == .9
    assert report["keep_mass"] == .75
    assert json.loads(output.read_text())["pairs"] == report["pairs"]
    terminal = capsys.readouterr().out
    assert "Saved partial tuning diagnostics:" in terminal and "Partial tuning summary" in terminal
    assert '"implementation_sha256"' not in terminal and '"threshold_sweep"' not in terminal
    assert "implementation_sha256" in json.loads(output.read_text())
    assert files(directory) == before
    with pytest.raises(FileExistsError):
        tune.tune_partial(c, root=tmp_path, from_fit=directory, report_path=output)
    with pytest.raises(ValueError, match="outside"):
        tune.tune_partial(c, root=tmp_path, from_fit=directory, report_path=directory/"new.json")
    for change in ("reg", "projection", "selection"):
        bad = deepcopy(c)
        if change == "reg": bad["solver"]["reg"] = .2
        elif change == "projection": bad["projection"]["bandwidth_multiplier"] = .9
        else: bad["fit_pair_top_k"] = 1
        with pytest.raises(ValueError, match="router/selection settings"):
            tune.tune_partial(bad, root=tmp_path, from_fit=directory)
    pair_file = directory / "pair_selection.json"
    pair_file.write_text(pair_file.read_text()+" ")
    with pytest.raises(ValueError, match="Missing/changed artifact"):
        tune.tune_partial(c, root=tmp_path, from_fit=directory)
    capsys.readouterr()


def test_mixed_failure_excluded_and_partial_cli_returns_failure_report(tmp_path, banks, monkeypatch, capsys):
    c = recipe()
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(c))
    original = tune.solve_partial_batch
    count = 0
    def one_bad_pair(costs, *args, **kw):
        nonlocal count
        costs = costs.clone()
        if count == 0: costs[0, 0, 0] = torch.nan
        count += 1
        return original(costs, *args, **kw)
    monkeypatch.setattr(tune, "solve_partial_batch", one_bad_pair)
    output = tmp_path/"failures.json"
    status = infoot_fit.main(["--config", str(path), "--project-root", str(tmp_path), "--tune-partial",
        "--tune-pairs", "3", "--tune-seed", "7", "--partial-h", ".8", "--partial-reg", ".3",
        "--partial-lam", "0", "--keep-mass", ".75", "--tune-report", str(output)])
    assert status == 2
    report = json.loads(output.read_text())
    assert report["summary"]["status_counts"] == {"invalid_input":1, "converged":2}
    assert report["summary"]["successful_objectives"]["cost"]["count"] == 2
    assert report["summary"]["feasibility_all_pairs"]["mass_error"]["count"] == 2
    assert report["partial_solver"]["h"] == .8 and report["image_solver"]["h"] == .7
    assert report["keep_mass"] == .75 and report["pairs"][0]["last_iteration"] == {}
    terminal = capsys.readouterr().out
    assert "invalid_input:1" in terminal and "failure " in terminal
    assert "objective terms(mean, converged)" in terminal


def test_serial_tuning_and_all_invalid_projection_are_explicit(tmp_path, banks, capsys):
    c = recipe()
    c["pair_batch_size"] = None
    c["partial"]["solver"]["lam"] = 0
    c["projection"]["confidence"]["threshold"] = 1.
    report = tune.tune_partial(c, root=tmp_path, pair_limit=1)
    assert report["summary"]["successful_pairs"] == 1
    assert report["pairs"][0]["runtime"]["partial_method"] == "sinkhorn_log"
    x, y = banks[0].features[0], banks[1].features[0]
    plan = x.new_full((4,4), .8/16)
    probe = tune._projection_probe(x, y, plan, kernel_state(x, .7), kernel_state(y, .7), validate_config(c)["projection"])
    assert probe["all_invalid"] and probe["valid_tokens"] == 0
    assert probe["mapped_to_target_variance_ratio"] is None
    capsys.readouterr()


def test_sampling_is_canonical_seeded_without_global_rng_effect():
    selection = dict(rows=[dict(source_id=f"s{i}", target_ids=[f"t{j}" for j in range(8)],
        probabilities=[.1]*8, retained_probability=.8) for i in range(12)])
    before = random.getstate()
    a = tune._sample_pairs(selection, 17, 42)
    b = tune._sample_pairs(dict(rows=list(reversed(selection["rows"]))), 17, 42)
    assert a == b and random.getstate() == before
    assert len({(r["source_id"],r["target_id"]) for r in a}) == 17
    assert a != tune._sample_pairs(selection, 17, 43)


def test_outer_budget_and_serial_exception_are_not_accepted(tmp_path, banks, monkeypatch, capsys):
    c = recipe()
    c["partial"]["solver"].update(lam=.1, max_outer_steps=1, outer_tolerance=1e-16)
    report = tune.tune_partial(c, root=tmp_path, pair_limit=2)
    assert report["status"] == "failed"
    assert report["summary"]["status_counts"] == {"max_outer_steps":2}
    assert not report["summary"]["successful_geometry"]
    c["pair_batch_size"] = None
    def fail(*args, **kwargs):
        raise RuntimeError("test inner failure")
    monkeypatch.setattr(tune, "solve_partial", fail)
    report = tune.tune_partial(c, root=tmp_path, pair_limit=2)
    assert report["summary"]["status_counts"] == {"solver_error":2}
    assert all(r["error"] == "test inner failure" for r in report["pairs"])
    capsys.readouterr()
    # A wholly failed large trial must stay pasteable and not claim zero errors
    # when no residual measurements exist. Full pair details remain in report.
    report["pairs"] *= 32
    report["summary"] = tune._summary(report["pairs"])
    tune._print_summary(report)
    terminal = capsys.readouterr().out
    assert "solver_error:64" in terminal and "mass_error=n/a" in terminal
    assert terminal.count("  failure ") == 3 and "+61 other failed pairs" in terminal
    assert len(terminal.splitlines()) <= 23


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable locally")
def test_cuda_costs_kernels_solves_and_projection_stay_on_device(tmp_path, banks, monkeypatch, capsys):
    c = dict(recipe(), device="cuda")
    original_solve, original_probe = tune.solve_partial_batch, tune._projection_probe
    observed = []
    def solve(cost, kx, ky, **kwargs):
        assert all(v.device.type == "cuda" and v.dtype == torch.float64 for v in (cost, kx, ky))
        plans, reports = original_solve(cost, kx, ky, **kwargs)
        assert plans.device.type == "cuda"
        observed.append("solve")
        return plans, reports
    def probe(x, y, plan, sx, sy, projection):
        assert all(v.device.type == "cuda" for v in (x, y, plan, sx[0], sy[0]))
        observed.append("projection")
        return original_probe(x, y, plan, sx, sy, projection)
    monkeypatch.setattr(tune, "solve_partial_batch", solve)
    monkeypatch.setattr(tune, "_projection_probe", probe)
    assert tune.tune_partial(c, root=tmp_path, pair_limit=2)["status"] == "converged"
    assert observed == ["solve", "projection", "projection"]
    capsys.readouterr()


@pytest.mark.parametrize("args", [
    ["--tune-partial", "--tune"], ["--tune-partial", "--resume", "old"],
    ["--tune-partial", "--output-root", "new"], ["--tune-partial", "--dry-run"],
    ["--tune-partial", "--tune-pairs", "0"], ["--tune-partial", "--tune-log-every", "0"],
    ["--tune", "--tune-from", "old"], ["--tune-report", "new.json"],
])
def test_bad_cli_combinations_fail_before_loading(args):
    with pytest.raises(SystemExit) as error:
        infoot_fit.main(["--config", "missing.yaml", *args])
    assert error.value.code == 2


def test_partial_tuning_rejects_other_modes_validation_data_and_cuda_fallback(tmp_path, banks, monkeypatch):
    with pytest.raises(ValueError, match="dense grouped_partial only"):
        tune.tune_partial(config("whole_map"), root=tmp_path)
    with pytest.raises(ValueError, match="train-split"):
        tune.tune_partial(dict(recipe(), source_bank="query"), root=tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        tune.tune_partial(dict(recipe(), device="cuda"), root=tmp_path)
