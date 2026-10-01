from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.evaluation import projection_audit as audit
from diffusion_ot.evaluation import stage1b_eval as evaluation


class OffsetReducer:
    """Known transform displacement with an explicit full-array cache shortcut."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_params(self, deep=False):
        return self.kwargs

    def fit_transform(self, values):
        self.fit = values.copy()
        return values[:, :2].copy()

    def transform(self, values):
        import numpy as np
        if np.array_equal(values, self.fit):
            return values[:, :2].copy()
        return values[:, :2] + np.array([3., 4.], dtype="float32")


def fixture_groups():
    rng = torch.Generator().manual_seed(23)
    target = torch.randn(12, 4, generator=rng)
    real = torch.randn(5, 4, generator=rng)
    weights = torch.softmax(torch.randn(5, 12, generator=rng), 1)
    groups = {"target_bank": target, "real_target_query": real, "conditional": weights @ target,
              "barycentric": target[[4, 3, 1, 3, 8]], "same_domain_conditional": real * .3}
    ids = {name: [f"{name}_{i}" for i in range(len(codes))] for name, codes in groups.items()}
    ids["same_domain_conditional"] = ids["real_target_query"]
    ids["barycentric"] = ids["conditional"]
    decoder = torch.nn.Sequential(torch.nn.LayerNorm(4, eps=.03), torch.nn.Linear(4, 6),
                                  torch.nn.SiLU(), torch.nn.Linear(6, 6)).eval()
    with torch.no_grad():
        decoder[0].weight.copy_(torch.tensor([.4, 2., 1.3, .7]))
        decoder[0].bias.copy_(torch.tensor([.2, -.3, .8, .1]))
    return groups, ids, weights, decoder


def test_geometry_uses_high_dimensional_distances_and_excludes_bank_self():
    bank = torch.tensor([[1., 0.], [4., 0.], [1., 4.]])
    stats, values = audit.code_geometry(bank, bank, exclude_self=True)
    torch.testing.assert_close(values["nn_euclidean"], torch.tensor([3., 3., 4.]))
    assert torch.all(values["nn_euclidean_indices"] != torch.arange(3))
    assert stats["nn_excludes_self"]
    heldout = torch.tensor([[2., 0.]])
    _, control = audit.code_geometry(heldout, bank)
    assert control["nn_euclidean"].item() == pytest.approx(1.)
    assert control["nn_cosine"].item() == pytest.approx(0.)


def test_decoder_audit_uses_checkpoint_affine_epsilon_and_preserves_model_rng():
    groups, _, _, decoder = fixture_groups()
    state = deepcopy(decoder.state_dict())
    rng = torch.get_rng_state().clone()
    representations, metadata = audit.decoder_representations(decoder, groups, batch_size=2)
    for key, codes in groups.items():
        torch.testing.assert_close(representations["post_layernorm"][key], decoder[0](codes))
        torch.testing.assert_close(representations["post_z_proj"][key], decoder(codes))
        assert not representations["post_z_proj"][key].requires_grad
    assert metadata["layernorm_eps"] == .03
    assert not torch.allclose(representations["post_layernorm"]["conditional"],
                              F.layer_norm(groups["conditional"], (4,)))
    for key, value in state.items():
        torch.testing.assert_close(decoder.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("normalize", [False, True])
def test_pca_fit_does_not_see_out_of_sample_outliers(normalize):
    groups, _, _, _ = fixture_groups()
    original = audit.target_fitted_pca(groups, normalize=normalize)
    altered = {**groups, "conditional": groups["conditional"] * 1000 + 10000}
    changed = audit.target_fitted_pca(altered, normalize=normalize)
    torch.testing.assert_close(original["components"], changed["components"], rtol=0, atol=0)
    torch.testing.assert_close(original["embeddings"]["target_bank"], changed["embeddings"]["target_bank"], rtol=0, atol=0)
    assert not torch.allclose(original["embeddings"]["conditional"], changed["embeddings"]["conditional"])


@pytest.mark.parametrize("visualize", [False, True])
def test_saved_audit_is_reproducible_and_displacement_uses_ids(tmp_path, monkeypatch, visualize):
    import sys
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=OffsetReducer))
    monkeypatch.setattr(audit, "_plot_controls", lambda *args, **kwargs: None)
    groups, ids, weights, decoder = fixture_groups()
    report, paths = audit.save_projection_audit(
        output_dir=tmp_path, direction="cat_to_dog", groups=groups, group_ids=ids,
        z_proj=decoder, tensors={"conditional_weights": weights}, protocol={"weights": "ema"},
        visualize=visualize, random_state=17)
    payload = torch.load(report["tensor_path"], weights_only=True)
    assert payload["group_ids"] == ids
    assert payload["protocol"]["weights"] == "ema"
    assert set(payload["representations"]) == {"raw", "post_layernorm", "post_z_proj"}
    assert len(payload["pca"]) == 6
    assert json.loads((tmp_path / "projection_audit.json").read_text())["decoder"]["layernorm_eps"] == .03
    torch.testing.assert_close(payload["projection_tensors"]["conditional_weights"] @ groups["target_bank"],
                               payload["representations"]["raw"]["conditional"])
    assert len(payload["bank_subset_indices"]) < len(groups["target_bank"])
    assert payload["bank_subset_ids"] == [ids["target_bank"][i] for i in payload["bank_subset_indices"]]
    if visualize:
        assert len(paths) == 14
        for key, info in report["umap"].items():
            assert Path(info["reducer_path"]).is_file()
            if key.endswith("joint"):
                assert info["transductive"] and not info["use_for_checkpoint_selection"]
            else:
                assert info["fit_transform_displacement"]["bank_subset"]["mean"] == pytest.approx(5.)
                assert info["fit_transform_displacement"]["selected_targets"]["mean"] == pytest.approx(5.)
    else:
        assert not paths and not report["umap"]


def test_audit_rejects_leaked_real_target_query(tmp_path):
    groups, ids, weights, decoder = fixture_groups()
    ids["real_target_query"][0] = ids["target_bank"][0]
    with pytest.raises(ValueError, match="disjoint"):
        audit.save_projection_audit(output_dir=tmp_path, direction="cat_to_dog", groups=groups,
                                    group_ids=ids, z_proj=decoder, tensors={"conditional_weights": weights},
                                    protocol={}, visualize=False, random_state=17)


@pytest.mark.parametrize("attributes", [[], ["pose"]])
def test_unlabeled_umap_has_distinct_role_colors_and_unclipped_footer(tmp_path, monkeypatch, attributes):
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=OffsetReducer))
    groups, ids, _, _ = fixture_groups()
    bank = evaluation.LatentBank("dog", "train", groups["target_bank"], groups["target_bank"],
                                 ids["target_bank"], [{} for _ in ids["target_bank"]], "checkpoint")
    figures = []
    real_subplots = plt.subplots

    def capture(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        figures.append(fig)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", capture)
    paths = {key: tmp_path / f"{key}.png" for key in ("conditional", "barycentric")}
    evaluation._save_umap_visualizations(
        bank, ids["conditional"], groups["conditional"], groups["barycentric"], labels={}, attributes=attributes,
        direction="cat_to_dog", random_state=13, n_jobs=1, target_alpha=.3, projection_alpha=.9,
        output_paths=paths, artifact_path=tmp_path / "umap.pt")
    for fig in figures:
        axis = fig.axes[0]
        assert not (axis.collections[0].get_facecolors()[0, :3] == axis.collections[1].get_facecolors()[0, :3]).all()
        assert axis.get_legend().get_title().get_text() == "Point type"
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        caption = fig.texts[-1]
        assert "\n" in caption.get_text()
        bounds = caption.get_window_extent(canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width
    for path in paths.values():
        with Image.open(path) as image:
            assert image.width > 500 and image.height > 500
    saved = torch.load(tmp_path / "umap.pt", weights_only=True)
    assert saved["target_ids"] == ids["target_bank"]
    assert saved["parameters"]["metric"] == "euclidean"


@pytest.mark.parametrize("field,value", [("identifier", "different"), ("ordered_sample_ids", {}),
                                         ("weights", "raw"), ("checkpoint_step", 8000)])
def test_initial_comparison_rejects_incomparable_protocols(field, value):
    protocol = dict(version=audit.AUDIT_VERSION, identifier="abc", ordered_sample_ids={"cat": ["a", "b"]},
                    weights="ema", checkpoint_step=0, seed=7)
    current = {**protocol, "checkpoint_step": 8000}
    evaluation._validate_baseline_protocol(current, protocol, require_step0=True)
    changed = {**protocol, field: value}
    with pytest.raises(ValueError):
        evaluation._validate_baseline_protocol(current, changed, require_step0=True)


def test_step0_recovery_forwards_all_overrides_before_current_model(tmp_path, monkeypatch):
    import yaml
    config = tmp_path / "alignment.yaml"
    config.write_text(yaml.safe_dump({"project_root": str(tmp_path)}))
    initial, current = tmp_path / "step_000000.pt", tmp_path / "step_008000.pt"
    torch.save({"stage": "stage1b_fused_infoot", "step": 0}, initial)
    current.write_bytes(b"current")
    calls = []
    baseline = SimpleNamespace(output_dir=str(tmp_path / "initial_report"))

    def run(*args, **kwargs):
        calls.append(kwargs)
        return baseline if kwargs.get("is_initial_baseline") else "final_report"

    monkeypatch.setattr(evaluation, "_run_stage1b_evaluation", run)
    result = evaluation.run_stage1b_evaluation(config, tmp_path / "eval.yaml", checkpoint_path=current,
                                              evaluate_step0=True, weights="raw", projection_bandwidth=.15,
                                              max_reference=12, max_projection=16, max_query=7,
                                              device_cat="cpu", device_dog="cpu")
    assert result == "final_report" and len(calls) == 2
    assert calls[0]["checkpoint_path"] == initial and calls[1]["checkpoint_path"] == current
    assert calls[1]["initial_baseline_report"] is baseline
    for field in ("weights", "projection_bandwidth", "max_reference", "max_projection", "max_query", "device_cat", "device_dog"):
        assert calls[0][field] == calls[1][field]
    initial.unlink()
    with pytest.raises(FileNotFoundError, match="Saved step-0"):
        evaluation.run_stage1b_evaluation(config, tmp_path / "eval.yaml", checkpoint_path=current, evaluate_step0=True)
    assert len(calls) == 2


def test_explicit_initial_checkpoint_must_really_be_step_zero(tmp_path):
    import yaml
    config = tmp_path / "alignment.yaml"
    config.write_text(yaml.safe_dump({"project_root": str(tmp_path)}))
    current = tmp_path / "latest.pt"
    torch.save({"stage": "stage1b_fused_infoot", "step": 4}, current)
    with pytest.raises(ValueError, match="must contain step 0"):
        evaluation.run_stage1b_evaluation(config, tmp_path / "eval.yaml", checkpoint_path=current,
                                          initial_checkpoint_path=current)
