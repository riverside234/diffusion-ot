"""Reuse a fitted whole-map plan while varying held-out KDE projection."""
from copy import deepcopy
import json

import pytest
import torch
import yaml

from test_infoot_vit_mapping import banks, config, cpu_threads, mapped
from infoot_vit import infoot_test
from infoot_vit.infoot_helper.conditional import BalancedModel
from infoot_vit.infoot_helper.feature_bank import file_hash
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, validate_config
from infoot_vit.infoot_helper.mapping import FeatureMapper, load_mapped


def test_bandwidth_override_matches_dense_formula_and_keeps_fit_immutable(tmp_path, banks, monkeypatch):
    directory = fit_mapping(config("whole_map"), root=tmp_path)
    before = {str(p): file_hash(p) for p in directory.rglob("*.pt")}
    manifest = (directory / "manifest.json").read_bytes()
    monkeypatch.setattr(BalancedModel, "fit", lambda *a, **k: pytest.fail("projection refit"))
    baseline = FeatureMapper.load(directory)
    baseline_output = mapped(baseline, banks[2])
    settings = infoot_test.projection_settings(baseline.config, None, bandwidth=.12)
    mapper = FeatureMapper.load(directory, projection=settings)
    assert mapper.image.h == pytest.approx(.12)
    assert mapper.manifest == json.loads(manifest)
    assert mapper.config["projection"] != mapper.manifest["config"]["projection"]
    model = mapper.image
    query = banks[2].features.flatten(1).double()
    # Independent direct KDE ratio, including target-density correction.
    kx = (-.5 * (torch.cdist(query, model.source) / (.12 * model.state["source_scale"])).square()).exp()
    ky = (-.5 * (torch.cdist(model.target, model.target) / (.12 * model.state["target_scale"])).square()).exp()
    scores = (kx @ model.plan @ ky.T) / ky.mean(1)[None]
    alpha = scores / scores.sum(1, keepdim=True)
    expected = (alpha @ model.target).reshape_as(banks[2].features)
    result = mapped(mapper, banks[2], return_metadata=True, chunk_size=1)
    torch.testing.assert_close(result.mapped_features, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(mapped(mapper, banks[2], chunk_size=2), expected, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(result.mapped_features, baseline_output)
    assert all(row["projection_h"] == pytest.approx(.12) for row in result.diagnostics["queries"])
    assert {str(p): file_hash(p) for p in directory.rglob("*.pt")} == before
    assert (directory / "manifest.json").read_bytes() == manifest


def test_cli_records_actual_bandwidth_topk_and_diagnostics(tmp_path, banks, monkeypatch, capsys):
    directory = fit_mapping(config("whole_map"), root=tmp_path)
    original = (directory / "manifest.json").read_bytes()
    monkeypatch.setattr(BalancedModel, "fit", lambda *a, **k: pytest.fail("projection refit"))
    outputs = []
    for k in (0, 1, 3):
        output = tmp_path / f"k{k}"
        args = ["--mapping", str(directory), "--query-bank", str(banks[2].path), "--output-dir", str(output),
                "--projection-bandwidth", ".15", "--top-k-images", str(k), "--threads", "1"]
        assert infoot_test.main(args) == 0
        result, _, manifest = load_mapped(output)
        assert manifest["projection"]["bandwidth_multiplier"] == pytest.approx(.15 / .7)
        assert manifest["projection"]["top_k_images"] == (k or None)
        assert manifest["mapper_id"] == json.loads(original)["artifact_id"]
        report = json.loads(next((output / "logs").glob("*/mapping_report.json")).read_text())
        routing = report["whole_map_routing"]
        assert routing["projection_h"]["mean"] == pytest.approx(.15)
        assert "query_source_effective_neighbors" in routing
        assert report["mapped_to_target_spread_ratio"]["token_variance"] >= 0
        if k == 1:
            assert routing["image_effective_targets"]["mean"] == pytest.approx(1.)
            assert routing["raw_image_effective_targets"]["mean"] > 1.
            for x, row in zip(result.mapped_features, result.diagnostics["queries"]):
                weights = torch.tensor(row["image_weights"])
                torch.testing.assert_close(x, banks[1].features[int(weights.argmax())])
                assert row["top_k_retained_mass"] == pytest.approx(row["raw_image_max_weight"])
        outputs.append(result.mapped_features)
        capsys.readouterr()
        assert infoot_test.main(args + ["--dry-run"]) == 0
        dry = json.loads(capsys.readouterr().out)
        assert dry["projection"] == manifest["projection"]
    torch.testing.assert_close(outputs[0], outputs[2])  # K covers every target.
    assert (directory / "manifest.json").read_bytes() == original


@pytest.mark.parametrize("flag,value", [("--projection-bandwidth", "0"), ("--projection-bandwidth", "nan"),
    ("--projection-bandwidth", "inf"), ("--projection-bandwidth", "-1"), ("--top-k-images", "-1")])
def test_invalid_overrides_rejected_before_loading(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        infoot_test.main(["--mapping", str(tmp_path / "missing"), flag, value])


@pytest.mark.parametrize("mode", ["patch_global", "grouped_patch", "grouped_partial", "grouped_patch_lowrank", "grouped_partial_lowrank"])
def test_new_override_cannot_change_saved_pairs_or_approximate_kernels(mode):
    c = config(mode)
    before = deepcopy(c)
    with pytest.raises(ValueError, match="only to whole_map"):
        infoot_test.projection_settings(c, None, bandwidth=.1)
    with pytest.raises(ValueError, match="only to whole_map"):
        infoot_test.projection_settings(c, None, top_k_images=4)
    assert c == before


def test_whole_map_config_restores_mi_weight_and_requests_projection_point_one():
    c = validate_config(yaml.safe_load((infoot_test.ROOT / "infoot_vit/configs/whole_map.yaml").read_text()))
    assert (c["solver"]["h"], c["solver"]["reg"], c["solver"]["lam"]) == (.35, .06, .075)
    assert c["solver"]["max_outer_steps"] >= 638
    assert c["solver"]["h"] * c["projection"]["bandwidth_multiplier"] == pytest.approx(.1)
    assert c["projection"]["top_k_images"] is None and c["projection"]["selection"] == "mean"
