"""Optional display reducer fits training points only and respects patch masks."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_infoot_vit_mapping import banks, config, cpu_threads
from infoot_vit import infoot_test
from infoot_vit.infoot_helper import umap_plot
from infoot_vit.infoot_helper.feature_bank import file_hash
from infoot_vit.infoot_helper.fit_mapping import fit_mapping
from infoot_vit.infoot_helper.mapping import FeatureMapper


@pytest.fixture
def fake_umap(monkeypatch):
    calls = []

    class Reducer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(self)

        def fit_transform(self, values):
            self.training = values.copy()
            return values[:, :2].copy()

        def transform(self, values):
            self.queries = values.copy()
            return values[:, :2].copy()

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=Reducer))
    monkeypatch.setattr(umap_plot.importlib.metadata, "version", lambda _: "test")
    return calls


def example(banks):
    source, target, query = banks
    mapper = SimpleNamespace(source=source, target=target, mode="grouped_partial",
        config={"projection": {"bandwidth_multiplier": .5}}, manifest={"artifact_id": "mapping"})
    mask = torch.tensor([[True, False, True, False], [False, False, False, False]])
    mapped = query.features.clone() + 10
    mapped[~mask] = 12345  # Masked values must never enter either visualization.
    result = SimpleNamespace(mapped_features=mapped, valid_mask=mask,
        diagnostics={"queries": [{"query_id": sid} for sid in query.ids]})
    return mapper, query, result


@pytest.mark.parametrize("level", ["image", "patch"])
def test_umap_training_only_masked_readout_and_nonmutation(tmp_path, banks, fake_umap, level):
    mapper, query, result = example(banks)
    before = [v.clone() for v in (mapper.source.features, mapper.target.features,
                                 query.features, result.mapped_features, result.valid_mask)]
    report = umap_plot.save_umap(mapper, query, result, tmp_path, level=level, max_points=1000)
    reducer = fake_umap[0]
    expected_training = torch.cat([mapper.source.features.mean(1), mapper.target.features.mean(1)]) if level == "image" else torch.cat([mapper.source.features.flatten(0, 1), mapper.target.features.flatten(0, 1)])
    np.testing.assert_allclose(reducer.training, expected_training.float().numpy(), rtol=1e-6, atol=1e-7)
    valid = result.valid_mask[0]
    source = query.features[0, valid].float()
    mapped = result.mapped_features[0, valid].float()
    expected_queries = torch.cat([source.mean(0, keepdim=True), mapped.mean(0, keepdim=True)]) if level == "image" else torch.cat([source, mapped])
    np.testing.assert_allclose(reducer.queries, expected_queries.numpy())
    assert report["all_invalid_query_ids"] == [query.ids[1]]
    assert report["fit_groups"] == ["source_training", "target_training"]
    assert reducer.kwargs["n_jobs"] == 1 and reducer.kwargs["random_state"] == 42
    assert reducer.kwargs["n_neighbors"] < len(expected_training)
    points = report["groups"]["mapped_query"]["points"]
    assert all(row["valid_tokens"] == 2 for row in points)
    assert [row["patch_index"] for row in points] == ([None] if level == "image" else [0, 2])
    assert (tmp_path / "umap.png").read_bytes().startswith(b"\x89PNG")
    assert json.loads((tmp_path / "umap.json").read_text())["level"] == level
    for original, after in zip(before, (mapper.source.features, mapper.target.features,
                                      query.features, result.mapped_features, result.valid_mask)):
        torch.testing.assert_close(original, after, rtol=0, atol=0)


@pytest.mark.parametrize("level", ["image", "patch"])
def test_sampling_is_reproducible_by_id_and_without_replacement(level):
    values = torch.arange(6 * 4 * 3).reshape(6, 4, 3).float()
    ids = [f"image{i}" for i in (4, 2, 0, 1, 5, 3)]
    mask = torch.ones(6, 4, dtype=torch.bool)
    mask[0, 2] = False
    permutation = torch.tensor([3, 5, 1, 2, 0, 4])
    a, records_a = umap_plot._sample_points(values, ids, mask=mask, level=level, limit=3, seed=42)
    b, records_b = umap_plot._sample_points(values[permutation], [ids[i] for i in permutation],
        mask=mask[permutation], level=level, limit=3, seed=42)
    assert records_a == records_b
    assert len({(r["sample_id"], r["patch_index"]) for r in records_a}) == 3
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_all_invalid_queries_skip_transform(tmp_path, banks, fake_umap):
    mapper, query, result = example(banks)
    result.valid_mask.zero_()
    report = umap_plot.save_umap(mapper, query, result, tmp_path)
    assert not hasattr(fake_umap[0], "queries")
    assert report["all_invalid_query_ids"] == query.ids
    assert report["groups"]["mapped_query"]["points"] == []


def test_missing_optional_dependency_reports_install_command(tmp_path, banks, monkeypatch):
    mapper, query, result = example(banks)
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(RuntimeError, match="pip install umap-learn matplotlib"):
        umap_plot.save_umap(mapper, query, result, tmp_path)


def test_bad_ids_or_non_training_fit_rejected(tmp_path, banks):
    mapper, query, result = example(banks)
    result.diagnostics["queries"].reverse()
    with pytest.raises(ValueError, match="ordered mapping query IDs"):
        umap_plot.save_umap(mapper, query, result, tmp_path)
    result.diagnostics["queries"].reverse()
    mapper.source = deepcopy(mapper.source)
    mapper.source.manifest["split"] = "val"
    with pytest.raises(ValueError, match="real training banks"):
        umap_plot.save_umap(mapper, query, result, tmp_path)


def test_cli_defaults_off_enable_and_dry_run(tmp_path, banks, fake_umap, monkeypatch, capsys):
    directory = fit_mapping(config("whole_map"), root=tmp_path)
    before = {str(p): file_hash(p) for p in directory.rglob("*") if p.is_file()}
    common = ["--mapping", str(directory), "--query-bank", str(banks[2].path), "--threads", "1"]
    monkeypatch.setattr("infoot_vit.infoot_helper.conditional.BalancedModel.fit",
                        lambda *a, **k: pytest.fail("Plot/testing must not refit InfoOT"))
    off, on = tmp_path / "umap-off", tmp_path / "umap-on"
    assert infoot_test.main(common + ["--output-dir", str(off)]) == 0
    assert not fake_umap and not (off / "umap.png").exists()
    assert infoot_test.main(common + ["--output-dir", str(on), "--umap"]) == 0
    assert len(fake_umap) == 1 and (on / "umap.png").is_file()
    a, b = (torch.load(path / "mapped.pt", weights_only=True) for path in (off, on))
    for key in ("mapped_features", "match_confidence", "valid_mask"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    capsys.readouterr()
    assert infoot_test.main(common + ["--output-dir", str(tmp_path / "dry"), "--umap", "--dry-run"]) == 0
    assert '"enabled": true' in capsys.readouterr().out
    assert len(fake_umap) == 1 and not (tmp_path / "dry").exists()
    events = next((on / "logs").glob("*/events.jsonl")).read_text()
    assert '"event": "umap_started"' in events and '"event": "umap_completed"' in events
    assert {str(p): file_hash(p) for p in directory.rglob("*") if p.is_file()} == before


def test_actual_umap_finite_forward_plot(tmp_path, banks):
    pytest.importorskip("umap")
    pytest.importorskip("matplotlib")
    mapper, query, result = example(banks)
    report = umap_plot.save_umap(mapper, query, result, tmp_path, level="patch", max_points=8, n_neighbors=3)
    for group in report["groups"].values():
        assert all(np.isfinite(row["xy"]).all() for row in group["points"])
    assert (tmp_path / "umap.png").stat().st_size > 1000
