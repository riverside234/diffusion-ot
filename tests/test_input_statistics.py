from copy import deepcopy
import json

import pytest
import torch

from diffusion_ot.evaluation.input_statistics import (
    appearance_targets, fit_appearance_probes, input_sensitivity, latent_statistics,
    probe_options, run_input_statistics_probe, validate_cohort_ids,
)
from diffusion_ot.models.pdae_sit import build_pdae_sit_branch
from test_pdae_sit_shapes import FakeSiT
from test_residual_encoder import tiny_config, small_cpu_thread_pool


def synthetic_cohorts():
    rng = torch.Generator().manual_seed(18)
    result = []
    matrix = torch.randn(8, 12, generator=rng)
    for name, count in (("train", 100), ("development", 40)):
        stats = torch.randn(count, 8, generator=rng)
        codes = torch.randn(count, 5, generator=rng)
        result.append(dict(ids=[f"{name}_{i}" for i in range(count)], latent_stats=stats,
                           targets=stats @ matrix, codes=codes, pre_encoder_layernorm=codes,
                           post_layernorm=codes, post_z_proj=codes))
    return result


def test_probe_tunes_and_scales_on_training_only_and_finds_added_information():
    train, dev = synthetic_cohorts()
    options = probe_options({"input_statistics": {"enabled": True, "ridge_alphas": [.0001, .1, 10.]}})
    rng = torch.get_rng_state().clone()
    report, fitted = fit_appearance_probes(train, dev, options)
    changed = deepcopy(dev)
    changed["targets"] = -100 * dev["targets"]
    other, other_fitted = fit_appearance_probes(train, changed, options)
    for name, model in fitted["models"].items():
        assert report["probes"][name]["alpha"] == other["probes"][name]["alpha"]
        for key in ("coef", "x_mean", "x_scale", "y_mean", "y_scale"):
            torch.testing.assert_close(model[key], other_fitted["models"][name][key], rtol=0, atol=0)
    assert set(report["fit_ids"]).isdisjoint(report["tuning_ids"])
    assert set(report["fit_ids"] + report["tuning_ids"]) == set(train["ids"])
    assert report["probes"]["codes_plus_latent_stats"]["development_rgb_mse"] < .01 * report["probes"]["codes"]["development_rgb_mse"]
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)


def test_cohort_overlap_and_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        validate_cohort_ids(["a", "b"], ["a", "c"])
    with pytest.raises(ValueError, match="unique"):
        validate_cohort_ids(["a", "a"], ["b"])


def test_latent_statistics_and_original_rgb_lab_units():
    x = torch.arange(4.).reshape(1, 4, 1, 1).expand(2, 4, 8, 8)
    stats = latent_statistics(x)
    torch.testing.assert_close(stats[0], torch.tensor([0., 1., 2., 3., 0., 0., 0., 0.]))
    target = appearance_targets(torch.ones(2, 3, 8, 8))
    assert target.shape == (2, 12)
    torch.testing.assert_close(target[:, :3], torch.ones(2, 3))
    torch.testing.assert_close(target[:, 6], torch.ones(2), atol=1e-5, rtol=0)


def test_raw_stem_sensitivity_keeps_internal_norms_and_module_modes():
    branch = build_pdae_sit_branch(FakeSiT(), stage_config=tiny_config()).train()
    norms = [m for m in branch.encoder.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert len(norms) > 2 and isinstance(branch.encoder.stem, torch.nn.Conv2d)
    before = {m: m.training for m in branch.modules()}
    report = input_sensitivity(branch, torch.randn(3, 4, 8, 8))
    assert report["raw_residual_stem"]
    assert set(report["first_conv_input_max_abs_error"].values()) == {0.}
    for change in report["changes"].values():
        assert change["stem_input_rms_change"] > 0 and change["pre_encoder_layernorm"] > 0
    assert before == {m: m.training for m in branch.modules()}
    assert norms == [m for m in branch.encoder.modules() if isinstance(m, torch.nn.GroupNorm)]


def test_saved_probe_uses_original_images_and_disjoint_cohorts(tmp_path, monkeypatch):
    import diffusion_ot.data.latent_dataset as data
    class Dataset:
        def __init__(self, path, domain, split, **kw):
            assert kw["random_horizontal_flip"] == 0 and kw["include_original_images"]
            self.records = [{"sample_id": f"{domain}_{split}_{i}"} for i in range(12)]
        def __len__(self): return len(self.records)
        def __getitem__(self, i):
            return dict(sample_id=self.records[i]["sample_id"], x0_latent=torch.full((4, 8, 8), i / 12),
                        encoder_image=torch.full((3, 8, 8), 2 * i / 12 - 1))
    monkeypatch.setattr(data, "CachedLatentDataset", Dataset)
    branch = build_pdae_sit_branch(FakeSiT(), stage_config=tiny_config())
    options = probe_options({"input_statistics": {"enabled": True, "train_samples": 10,
        "development_samples": 6, "batch_size": 3, "ridge_alphas": [.01]}})
    path = run_input_statistics_probe(branch, data_config_path="unused", domain="cat", project_root=tmp_path,
        device="cpu", dtype=torch.float32, options=options, output_dir=tmp_path / "probe", provenance={"weights": "ema"})
    report = json.loads(open(path).read())
    tensors = torch.load(report["tensors"], weights_only=True)
    assert report["counts"] == {"train": 10, "development": 6}
    assert tensors["provenance"] == {"weights": "ema"}
    for cohort in tensors["cohorts"].values():
        expected = torch.tensor([int(i.rsplit("_", 1)[1]) / 12 for i in cohort["ids"]])
        torch.testing.assert_close(cohort["targets"][:, 0], expected)
        assert cohort["latent_stats"].shape[1] == 8
