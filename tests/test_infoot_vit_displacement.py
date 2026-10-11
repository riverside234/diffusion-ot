"""Residual-preserving projection with unchanged fitted plans and rejection."""
from copy import deepcopy
import json

import pytest
import torch

from test_infoot_vit_mapping import banks, config, cpu_threads, mapped
from test_infoot_vit_projection_overrides import hashes, forbid_fitting
from infoot_vit import infoot_test
from infoot_vit.infoot_helper.conditional_displacement import conditional_displacement
from infoot_vit.infoot_helper.dense_partial_projection import project_pairs
from infoot_vit.infoot_helper.fit_mapping import fit_mapping, validate_config
from infoot_vit.infoot_helper.mapping import FeatureMapper, load_mapped


def example():
    gen = torch.Generator().manual_seed(83)
    source = torch.randn(2, 4, 3, generator=gen, dtype=torch.float64)
    target = torch.randn(2, 5, 3, generator=gen, dtype=torch.float64)
    query = torch.randn(2, 6, 3, generator=gen, dtype=torch.float64)
    plans = torch.rand(2, 4, 5, generator=gen, dtype=torch.float64) * .01
    plans[:, 2] = 0  # Partial rejection is allowed and must remain zero.
    logs = -(query[:, :, None] - source[:, None]).square().sum(-1) * .7
    ky = torch.exp(-(target[:, :, None] - target[:, None]).square().sum(-1) * .2)
    b = torch.tensor([[.1, .2, .4, .1, .2], [.3, .1, .1, .4, .1]], dtype=torch.float64)
    return query, source, target, plans, logs, ky, b


def test_displacement_matches_independent_joint_expectation():
    q, x, y, plan, logs, ky, b = example()
    actual, _ = conditional_displacement(q, x, y, plan, logs, ky, b)
    # Independent five-axis oracle: query/source/transport target/KDE target.
    expected = torch.zeros_like(q)
    for batch in range(len(q)):
        density = ky[batch] @ b[batch]
        for r in range(q.shape[1]):
            total, moment = 0., q.new_zeros(q.shape[-1])
            for i in range(x.shape[1]):
                for j in range(y.shape[1]):
                    for k in range(y.shape[1]):
                        mass = logs[batch, r, i].exp() * plan[batch, i, j] * ky[batch, k, j] * b[batch, k] / density[k]
                        total += mass
                        moment += mass * (y[batch, j] - x[batch, i])
            expected[batch, r] = q[batch, r] + moment / total
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("offset", [0., 1.7])
def test_identity_and_constant_translation_preserve_unseen_query_detail(offset):
    # Support has no third-channel detail; the unseen query does. Absolute
    # target averaging MUST lose it; transporting a constant displacement must
    # preserve it, including with broad non-identity target KDE smoothing.
    x = torch.tensor([[[-1., 0., 0.], [0., 1., 0.], [1., 0., 0.]]], dtype=torch.float64)
    delta = torch.tensor([offset, -.2 * offset, 0.], dtype=torch.float64)
    y = x + delta
    q = torch.tensor([[[-.4, .3, 2.], [.6, .2, -2.]]], dtype=torch.float64)
    plan = torch.diag_embed(torch.tensor([[.25, .2, .3]], dtype=torch.float64))
    logs = -(q[:, :, None] - x[:, None]).square().sum(-1) / 4
    ky = torch.tensor([[[1., .8, .5], [.8, 1., .7], [.5, .7, 1.]]], dtype=torch.float64)
    a = b = torch.full((1, 3), 1/3, dtype=torch.float64)
    params = (logs, y, plan, a, b, ky, torch.tensor([-100.]), torch.tensor([1e-7]))
    baseline, raw0, valid0, _ = project_pairs(*params)
    candidate, raw1, valid1, _ = project_pairs(*params, value_mode="conditional_displacement", query=q, source=x)
    torch.testing.assert_close(candidate, q + delta, atol=1e-14, rtol=1e-14)
    assert torch.equal(raw0, raw1) and torch.equal(valid0, valid1)
    assert baseline[..., 2].abs().max() == 0
    assert candidate[..., 2].square().mean() == 4


def test_rejection_and_extreme_log_kernels_do_not_create_source_fallback():
    q, x, y, plans, logs, ky, b = example()
    plans[1] = 0
    # A rejected source is nearest; retained source scores underflow outside log space.
    logs[0] = -10000
    logs[0, :, 2] = 0
    values, stats = conditional_displacement(q, x, y, plans, logs, ky, b)
    assert torch.isfinite(values).all() and values[0].abs().sum() > 0
    assert not values[1].any()
    assert not stats["displacement_norm"][1].any()
    shifted, _ = conditional_displacement(q, x, y, plans, logs - 12000, ky, b)
    torch.testing.assert_close(values, shifted, atol=2e-11, rtol=2e-11)


def test_support_and_query_permutations_and_batch_partition():
    q, x, y, p, logs, ky, b = example()
    expected, _ = conditional_displacement(q, x, y, p, logs, ky, b)
    i, j, k = [3, 0, 2, 1], [2, 4, 0, 3, 1], [4, 1, 0, 3, 5, 2]
    actual, _ = conditional_displacement(q[:, k], x[:, i], y[:, j], p[:, i][:, :, j],
        logs[:, k][:, :, i], ky[:, j][:, :, j], b[:, j])
    torch.testing.assert_close(actual, expected[:, k], atol=1e-14, rtol=1e-14)
    for r in range(len(q)):
        value, _ = conditional_displacement(*[v[r:r+1] for v in (q, x, y, p, logs, ky, b)])
        torch.testing.assert_close(value, expected[r:r+1], atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("bad", ["dtype", "nan", "negative", "proposal", "shape"])
def test_malformed_inputs_rejected(bad):
    args = list(example())
    if bad == "dtype":
        args = [v.float() for v in args]
    elif bad == "nan":
        args[4][0, 0, 0] = float("nan")
    elif bad == "negative":
        args[3][0, 0, 0] = -.1
    elif bad == "proposal":
        args[-1][0, 0] = 0
    else:
        args[1] = args[1][:, :-1]
    with pytest.raises(ValueError):
        conditional_displacement(*args)


def test_mapper_preserves_confidence_inventory_routing_and_artifacts(tmp_path, banks, monkeypatch):
    c = config("grouped_partial")
    c.update(fit_pair_top_k=2, pair_batch_size=3)
    directory = fit_mapping(c, root=tmp_path)
    before = hashes(directory)
    base = FeatureMapper.load(directory, pair_projection_batch_size=2)
    baseline = mapped(base, banks[2], return_metadata=True)
    forbid_fitting(monkeypatch)
    settings = infoot_test.projection_settings(base.config, None, patch_value_mode="conditional_displacement")
    mapper = FeatureMapper.load(directory, projection=settings, pair_projection_batch_size=3)
    result = mapped(mapper, banks[2], return_metadata=True)
    torch.testing.assert_close(result.match_confidence, baseline.match_confidence, atol=1e-15, rtol=1e-15)
    assert torch.equal(result.valid_mask, baseline.valid_mask)
    assert not torch.allclose(result.mapped_features, baseline.mapped_features)
    for row, old in zip(result.diagnostics["queries"], baseline.diagnostics["queries"]):
        for key in ("image_weights", "image_projection_h", "patch_projection_h"):
            assert row[key] == old[key]
        assert row["matched_rejected_group_mass_residual"] < 1e-10
        assert row["patch_value_mode"] == "conditional_displacement"
    # Independent serial reference over the exact same pair, confidence and mask.
    sid, tid = next(iter(mapper.pairs))
    i, j = mapper.source_index[sid], mapper.target_index[tid]
    query = banks[2].features[0].double()
    logs = mapper._partial_query(query, i)
    expected, g, diag, tolerance = mapper._partial_projection(query, i, j, sid, tid, logs)
    pair = mapper._pair(sid, tid)
    value, raw, valid, _ = project_pairs(logs[None], mapper.y[j:j+1], pair["plan"][None],
        pair["a"][None], pair["b"][None], mapper.projection_ky[j:j+1], query.new_tensor([-float("inf")]),
        query.new_tensor([tolerance]), value_mode="conditional_displacement", query=query[None], source=mapper.x[i:i+1])
    torch.testing.assert_close(value[0], expected, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(raw[0] * valid[0], g)
    assert hashes(directory) == before
    out = tmp_path / "projected"
    mapper.project_bank(banks[2], out)
    restored, _, manifest = load_mapped(out)
    assert manifest["projection"]["patch_value_mode"] == "conditional_displacement"
    assert "conditional_displacement.py" in manifest["projection_dependency_sha256"]
    torch.testing.assert_close(restored.mapped_features, result.mapped_features)


@pytest.mark.parametrize("mode", ["whole_map", "patch_global", "grouped_patch", "grouped_partial_lowrank"])
def test_override_restricted_to_dense_partial(mode):
    with pytest.raises(ValueError, match="only to dense grouped_partial"):
        infoot_test.projection_settings(dict(mode=mode, projection={}), None, patch_value_mode="conditional_displacement")


@pytest.mark.parametrize("bad", [None, True, "argmax", {}, 1])
def test_config_rejects_invalid_mode(bad):
    c = config("grouped_partial")
    c["projection"]["patch_value_mode"] = bad
    with pytest.raises(ValueError, match="patch_value_mode"):
        validate_config(c)


def test_cli_saves_explicit_variant_and_preserves_legacy_default(tmp_path, banks, monkeypatch, capsys):
    c = config("grouped_partial")
    assert "patch_value_mode" not in validate_config(c)["projection"]
    before_config = deepcopy(c)
    directory = fit_mapping(c, root=tmp_path)
    forbid_fitting(monkeypatch)
    args = ["--mapping", str(directory), "--query-bank", str(banks[2].path),
        "--output-dir", str(tmp_path / "cli"), "--patch-value-mode", "conditional_displacement"]
    assert infoot_test.main(args) == 0
    _, _, manifest = load_mapped(tmp_path / "cli")
    capsys.readouterr()
    assert infoot_test.main(args + ["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["projection"] == manifest["projection"]
    assert c == before_config
