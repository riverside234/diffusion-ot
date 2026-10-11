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
from infoot_vit.infoot_helper.partial import distance


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
        c.update(sampling=dict(images_per_domain=2, seed=42), device="cpu", fit_pair_top_k=1)
        c["projection"]["bandwidth_multiplier"] = .5
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
    if lowrank:
        assert "pair_selection_preview" not in result
    else:
        preview = result["pair_selection_preview"]
        assert preview["projection_h"] == pytest.approx(.35)
        assert preview["effective_k"] == 1 and preview["pair_count"] == 2
        assert .5 <= preview["mean_retained_probability"] <= 1.
        assert preview["mean_renormalized_effective_targets"] == pytest.approx(1.)
        assert "No pairs saved" in output
    for name, bank in zip(("source", "target"), banks):
        ids = sample_ids(bank.ids, 2, 42)
        assert result["sampling"][name]["ordered_ids_sha256"] == digest(ids)
        torch.testing.assert_close(observed[name], bank.subset(ids).features.flatten(1), rtol=0, atol=0)
    assert files(tmp_path) == before and not (tmp_path/"outputs").exists()


def test_tuning_nonconvergence_and_solver_exception_leave_no_files(tmp_path, banks, monkeypatch, capsys):
    c = dense_config("grouped_partial")
    c["solver"].update(lam=.1, max_outer_steps=1, outer_tolerance=1e-15)
    path = tmp_path/"tune.yaml"; path.write_text(yaml.safe_dump(c))
    before = files(tmp_path)
    args = ["--config", str(path), "--project-root", str(tmp_path), "--tune"]
    assert infoot_fit.main(args) == 2
    output = capsys.readouterr().out
    report = json.loads(output.split("Tuning result (not a saved mapping):\n")[1])
    assert report["status"] != "converged"
    assert "pair_selection_preview" not in report
    assert files(tmp_path) == before
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")
    monkeypatch.setattr(tuning.BalancedModel, "fit", fail)
    with pytest.raises(RuntimeError, match="synthetic solver failure"):
        infoot_fit.main(args)
    assert files(tmp_path) == before and not (tmp_path/"outputs").exists()


@pytest.mark.parametrize("top_k", [1, 8, None])
def test_selection_preview_matches_dense_projection_formula_and_full_pair_coverage(top_k):
    # Asymmetric supports: selection must use projection h rather than fit h.
    x = torch.tensor([[0., 0.], [.2, .3], [1.8, .8]], dtype=torch.float64)
    y = torch.tensor([[.1, .2], [.3, .8], [1.5, .7], [2., 1.2]], dtype=torch.float64)
    model = tuning.BalancedModel.fit(x, y, dict(h=.8, lam=0., reg=.3))
    original_plan = model.plan.clone()
    report = tuning._pair_selection_preview(model,
        dict(projection=dict(bandwidth_multiplier=.5), fit_pair_top_k=top_k),
        ["c", "a", "b"], ["d", "c", "b", "a"])
    h = .4
    kx = torch.exp(-.5 * (distance(x, x)/(model.state["source_scale"]*h)).square())
    ky = torch.exp(-.5 * (distance(y, y)/(model.state["target_scale"]*h)).square())
    scores = (kx @ model.plan @ ky.T)/ky.mean(1)[None]
    full = scores/scores.sum(1, keepdim=True)
    k = len(y) if top_k is None else min(top_k, len(y))
    kept = full.topk(k, dim=1).values
    retained = kept.sum(1)
    norm = kept/retained[:, None]
    assert report["effective_k"] == k and report["pair_count"] == len(x)*k
    assert report["projection_h"] == pytest.approx(h)
    assert report["mean_retained_probability"] == pytest.approx(float(retained.mean()), abs=1e-12)
    assert report["min_retained_probability"] == pytest.approx(float(retained.min()), abs=1e-12)
    assert report["mean_discarded_probability"] == pytest.approx(float(1-retained.mean()), abs=1e-12)
    assert report["mean_renormalized_top1_probability"] == pytest.approx(float(norm.max(1).values.mean()), abs=1e-12)
    assert report["mean_renormalized_effective_targets"] == pytest.approx(
        float((-torch.special.xlogy(norm, norm).sum(1)).exp().mean()), abs=1e-12)
    if k == len(y):
        assert report["mean_retained_probability"] == pytest.approx(1.)
    else:
        # This check catches accidentally reusing the broader fit-time scorer.
        assert abs(report["mean_retained_probability"] - float(model.conditional_weights(x).max(1).values.mean())) > .01
    torch.testing.assert_close(model.plan, original_plan, atol=0, rtol=0)
    assert model.h == .8


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
