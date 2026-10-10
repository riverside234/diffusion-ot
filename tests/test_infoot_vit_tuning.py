"""Router tuning must execute the real solver without writing fit artifacts."""
import json

import pytest
import torch
import yaml

from test_infoot_vit_mapping import banks, cpu_threads, config as dense_config
from test_infoot_vit_lowrank import tiny_config
from infoot_vit import infoot_fit, infoot_fit_lowrank
from infoot_vit.infoot_helper import tuning
from infoot_vit.infoot_helper.feature_bank import digest, file_hash
from infoot_vit.infoot_helper.run_logging import RunLog
from infoot_vit.infoot_helper.sampling import sample_ids


def files(root):
    return {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("lowrank", [False, True])
def test_tuning_real_solver_weighted_terms_sampling_and_no_writes(tmp_path, banks, monkeypatch, capsys, lowrank):
    if lowrank:
        c, cli = tiny_config(), infoot_fit_lowrank
        # Full patch storage limits must not block an image-only tuning run.
        c["resources"] = dict(max_artifact_bytes=1)
    else:
        c, cli = dense_config("grouped_partial"), infoot_fit
        c.update(sampling=dict(images_per_domain=2, seed=42), device="cpu")
        c["resources"] = dict(max_plan_gib=1e-30)
    path = tmp_path/"tune.yaml"
    path.write_text(yaml.safe_dump(c))
    before = files(tmp_path)
    monkeypatch.setattr(RunLog, "__enter__", lambda *a: pytest.fail("Created disk logger"))
    monkeypatch.setattr(torch, "save", lambda *a, **kw: pytest.fail("Saved a plan/checkpoint"))
    monkeypatch.setattr(cli, "fit" if lowrank else "fit_mapping", lambda *a, **kw: pytest.fail("Started full fit"))
    original = tuning.BalancedModel.fit
    observed = {}
    def fit(source, target, options, **kwargs):
        assert source.device.type == target.device.type == "cpu"
        assert source.dtype == target.dtype == torch.float64
        observed.update(source=source.clone(), target=target.clone())
        return original(source, target, options, **kwargs)
    monkeypatch.setattr(tuning.BalancedModel, "fit", fit)
    args = ["--config", str(path), "--project-root", str(tmp_path), "--tune",
            "--h", ".7", "--lam", "0", "--reg", ".4", "--tune-log-every", "1"]
    assert cli.main(args) == 0
    output = capsys.readouterr().out
    assert "cost=" in output and "mi_term=" in output and "entropy_term=" in output
    result = json.loads(output.split("Tuning result (not a saved mapping):\n")[1])
    assert result["solver"]["h"] == .7 and result["solver"]["reg"] == .4
    last = result["last_iteration"]
    assert last["objective"] == pytest.approx(last["cost"] + last["mi_term"] + last["entropy_term"], abs=1e-12)
    assert last["mi_term"] == 0. and last["entropy_term"] < 0.
    for name, bank in zip(("source", "target"), banks):
        ids = sample_ids(bank.ids, 2, 42)
        assert result["sampling"][name]["ordered_ids_sha256"] == digest(ids)
        torch.testing.assert_close(observed[name], bank.subset(ids).features.flatten(1), rtol=0, atol=0)
    assert files(tmp_path) == before and not (tmp_path/"outputs").exists()


def test_tuning_nonconvergence_and_solver_exception_leave_no_files(tmp_path, banks, monkeypatch, capsys):
    c = dense_config("whole_map")
    c["solver"].update(lam=.1, max_outer_steps=1, outer_tolerance=1e-15)
    path = tmp_path/"tune.yaml"; path.write_text(yaml.safe_dump(c))
    before = files(tmp_path)
    args = ["--config", str(path), "--project-root", str(tmp_path), "--tune"]
    assert infoot_fit.main(args) == 2
    output = capsys.readouterr().out
    report = json.loads(output.split("Tuning result (not a saved mapping):\n")[1])
    assert report["status"] != "converged"
    assert files(tmp_path) == before
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")
    monkeypatch.setattr(tuning.BalancedModel, "fit", fail)
    with pytest.raises(RuntimeError, match="synthetic solver failure"):
        infoot_fit.main(args)
    assert files(tmp_path) == before and not (tmp_path/"outputs").exists()


@pytest.mark.parametrize("cli", [infoot_fit, infoot_fit_lowrank])
@pytest.mark.parametrize("option", [["--resume", "old"], ["--output-root", "new"], ["--dry-run"], ["--tune-log-every", "0"]])
def test_tuning_rejects_conflicting_storage_options_before_reading_files(cli, option):
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", "does-not-exist.yaml", "--tune", *option])
    assert error.value.code == 2


def test_tuning_rejects_validation_banks_and_cuda_fallback(tmp_path, banks, monkeypatch):
    before = files(tmp_path)
    c = dense_config("whole_map")
    with pytest.raises(ValueError, match="train-split"):
        tuning.tune_image_router(dict(c, source_bank="query"), root=tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        tuning.tune_image_router(dict(c, device="cuda"), root=tmp_path)
    with pytest.raises(ValueError, match="has no image router"):
        tuning.tune_image_router(dense_config("patch_global"), root=tmp_path)
    assert files(tmp_path) == before
