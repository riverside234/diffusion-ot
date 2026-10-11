"""Separate router/patch bandwidths and diagnose fitted-plan over-smoothing."""
from copy import deepcopy
import json

import pytest
import torch
import yaml

from test_infoot_vit_mapping import banks, config, cpu_threads, mapped
from test_infoot_vit_projection_overrides import hashes, forbid_fitting
from infoot_vit import infoot_test
from infoot_vit.infoot_helper.conditional import calibrate_support, normalize_rows
from infoot_vit.infoot_helper.plan_diagnostics import patch_plan_concentration
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, validate_config
from infoot_vit.infoot_helper.mapping import FeatureMapper, MappingResult, load_mapped, mapping_summary
from infoot_vit.infoot_helper.partial import distance


@pytest.mark.parametrize("multiplier,patch_h", [(1., .18), (.3, .9)])
def test_independent_patch_bandwidth_preserves_router_and_saved_pairs(tmp_path, banks, monkeypatch, multiplier, patch_h):
    c = config("grouped_partial")
    c.update(fit_pair_top_k=2, pair_batch_size=3)
    c["partial"]["solver"]["h"] = .9
    c["projection"].update(bandwidth_multiplier=multiplier, confidence=dict(
        support_calibration="fit_leave_one_out_log_density", all_invalid_policy="bypass"))
    directory = fit_mapping(c, root=tmp_path)
    before = hashes(directory)
    base = FeatureMapper.load(directory)
    forbid_fitting(monkeypatch)
    settings = infoot_test.projection_settings(base.config, None, patch_bandwidth=patch_h)
    mapper = FeatureMapper.load(directory, projection=settings)
    assert mapper.image.h == base.image.h
    torch.testing.assert_close(mapper.image.smoothing, base.image.smoothing, rtol=0, atol=0)
    assert mapper.patch_projection_h == patch_h
    assert mapper.pairs == base.pairs and torch.equal(mapper.pair_mask, base.pair_mask)
    for i, x in enumerate(mapper.x):
        assert mapper.shared["support"][i] == calibrate_support(x, mapper.shared["sx"][i], patch_h, settings["confidence"])
    for j, y in enumerate(mapper.y):
        expected = torch.exp(-.5*(distance(y, y)/(mapper.shared["sy"][j]*patch_h)).square())
        # At fit h the existing quantized kernel is deliberately reused.
        torch.testing.assert_close(mapper.projection_ky[j], expected, atol=5e-8, rtol=5e-8)
    sid, tid = next(iter(mapper.pairs))
    i, j = mapper.source_index[sid], mapper.target_index[tid]
    q = banks[2].features[0]
    candidate, _, diag, _ = mapper._partial_projection(q, i, j, sid, tid, mapper._partial_query(q, i))
    plan = mapper._pair(sid, tid)["plan"]
    kq = torch.exp(-.5*(distance(q, mapper.x[i])/(mapper.shared["sx"][i]*patch_h)).square())
    ky = mapper.projection_ky[j]
    expected = normalize_rows((kq @ plan @ ky.T)/ky.mean(0)) @ mapper.y[j]
    torch.testing.assert_close(candidate, expected, atol=1e-11, rtol=1e-11)
    assert 1 <= diag["plan_geometry"]["fitted_pair_effective_patches"] <= len(q)
    a = mapped(mapper, banks[2], return_metadata=True, chunk_size=1)
    b = mapped(mapper, banks[2], return_metadata=True, chunk_size=2)
    torch.testing.assert_close(a.mapped_features, b.mapped_features, atol=0, rtol=0)
    torch.testing.assert_close(a.match_confidence, b.match_confidence, atol=0, rtol=0)
    assert torch.equal(a.valid_mask, b.valid_mask)
    for row in a.diagnostics["queries"]:
        assert row["patch_projection_h"] == patch_h
        assert row["matched_rejected_group_mass_residual"] < 1e-10
        assert row["fitted_pair_effective_patches"] > 0
    assert hashes(directory) == before


def test_fit_calibration_cli_and_artifact_record_independent_bandwidth(tmp_path, banks, monkeypatch, capsys):
    c = config("grouped_partial")
    c["projection"].update(patch_bandwidth=.3, confidence=dict(
        support_calibration="fit_leave_one_out_log_density", all_invalid_policy="bypass"))
    directory = fit_mapping(c, root=tmp_path)
    shared = torch.load(directory / "plans/pair_kernels.pt", weights_only=True)
    assert shared["h_projection"] == .3
    before = hashes(directory)
    forbid_fitting(monkeypatch)
    args = ["--mapping", str(directory), "--query-bank", str(banks[2].path),
        "--output-dir", str(tmp_path / "test"), "--projection-bandwidth", ".3", "--patch-projection-bandwidth", ".2"]
    assert infoot_test.main(args) == 0
    result, _, manifest = load_mapped(tmp_path / "test")
    assert manifest["projection"]["patch_bandwidth"] == .2
    assert manifest["projection"]["top_k_images"] is None
    for row in result.diagnostics["queries"]:
        assert row["image_projection_h"] == pytest.approx(.3)
        assert row["patch_projection_h"] == .2
    report = json.loads(next((tmp_path / "test/logs").glob("*/mapping_report.json")).read_text())
    assert "fitted_pair_effective_patches" in report["partial_geometry"]
    assert "within_image_token_variance" in report["mapped_to_target_spread_ratio"]
    assert hashes(directory) == before
    capsys.readouterr()
    assert infoot_test.main(args + ["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["projection"] == manifest["projection"]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_bad_patch_bandwidth_fails_before_loading(tmp_path, value):
    with pytest.raises(SystemExit):
        infoot_test.main(["--mapping", str(tmp_path / "missing"), "--patch-projection-bandwidth", value])


@pytest.mark.parametrize("mode", ["whole_map", "patch_global", "grouped_patch", "grouped_patch_lowrank", "grouped_partial_lowrank"])
def test_independent_patch_override_is_explicitly_dense_partial_only(mode):
    c = dict(mode=mode, projection={})
    with pytest.raises(ValueError, match="only to dense grouped_partial"):
        infoot_test.projection_settings(c, None, patch_bandwidth=.2)


def test_plan_diagnostics_weight_retained_rows_and_ignore_zero_rows():
    plan = torch.tensor([[.2, 0.], [.1, .1], [0., 0.]], dtype=torch.float64)
    stats = patch_plan_concentration(plan)
    assert stats["fitted_pair_effective_patches"] == pytest.approx(1.5)
    assert stats["fitted_pair_top1_probability"] == pytest.approx(.75)
    assert stats["fitted_pair_normalized_entropy"] == pytest.approx(.5)
    batch = patch_plan_concentration(torch.stack([plan, plan*.5]))
    for key in stats:
        torch.testing.assert_close(batch[key], stats[key].expand(2))


def test_within_image_variance_excludes_padding_and_image_mean_changes():
    x = torch.tensor([[[0.], [2.]], [[10.], [5000.]]], dtype=torch.float64)
    mask = torch.tensor([[True, True], [True, False]])
    result = MappingResult(x, mask.double(), mask, dict(mode="grouped_partial", seconds=0., queries=[]))
    report = mapping_summary(result, x, x, x)
    # Valid image variances are 1 and 0; the between-image offset is separate.
    assert report["mapped"]["within_image_token_variance"] == pytest.approx(.5)
    assert report["mapped"]["image_mean_variance"] == pytest.approx(20.25)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "0.2", None])
def test_invalid_yaml_patch_bandwidth_rejected(value):
    c = config("grouped_partial")
    c["projection"]["patch_bandwidth"] = value
    with pytest.raises(ValueError, match="patch_bandwidth"):
        validate_config(c)


def test_batched_fit_persists_plan_geometry_before_complete_summary(tmp_path, banks):
    c = config("grouped_partial")
    c.update(fit_pair_top_k=2, pair_batch_size=3)
    directory = fit_mapping(c, root=tmp_path)
    batch_report = json.loads((directory / "pair_batch_report.json").read_text())
    assert batch_report["completed_pair_geometry"]["count"] == 4
    rows = [json.loads(line) for line in next((directory / "logs").glob("*/pair_geometry.jsonl")).read_text().splitlines()]
    assert len(rows) == 4 and all(r["mean_row_cost_std"] > 0 for r in rows)
    final = json.loads((directory / "fit_report.json").read_text())
    value = final["partial"]["fitted_pair_geometry"]["fitted_pair_effective_patches"]["mean"]
    assert value == pytest.approx(sum(r["fitted_pair_effective_patches"] for r in rows)/4, abs=1e-6)


def test_revised_config_tunes_fitting_and_keeps_broad_projection():
    c = validate_config(yaml.safe_load((infoot_test.ROOT / "infoot_vit/configs/grouped_partial.yaml").read_text()))
    assert (c["solver"]["h"], c["solver"]["reg"], c["solver"]["lam"]) == (.36, .0575, .070)
    assert (c["partial"]["solver"]["h"], c["partial"]["solver"]["reg"], c["partial"]["solver"]["lam"]) == (.35, .05, .025)
    assert c["partial"]["keep_mass"] == .8 and c["projection"]["confidence"]["threshold"] == .05
    assert c["solver"]["h"] * c["projection"]["bandwidth_multiplier"] == pytest.approx(.2)
    assert c["projection"]["patch_bandwidth"] == .2 and c["projection"]["top_k_images"] is None
    before = deepcopy(c)
    assert infoot_test.projection_settings(c, None, bandwidth=.3)["patch_bandwidth"] == .2
    assert c == before


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_diagnostics_stay_on_plan_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable locally")
    plan = torch.eye(4, device=device, dtype=torch.float64) * .2
    assert all(value.device == plan.device and torch.isfinite(value) for value in patch_plan_concentration(plan).values())
