"""Regression coverage for mapping diagnostics, artifact identity and failure logs."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest
import torch

from test_infoot_vit_mapping import banks, config, mapped
from infoot_vit.infoot_helper.conditional import partial_projection
from infoot_vit.infoot_helper.fit_mapping import fit_mapping
from infoot_vit.infoot_helper.mapping import FeatureMapper, MappingResult, mapping_summary


@pytest.fixture(autouse=True)
def single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_partial_effective_patch_count_uses_matched_probability_weights(tmp_path, banks):
    mapper = FeatureMapper.load(fit_mapping(config("grouped_partial"), root=tmp_path))
    query = banks[2]
    result = mapped(mapper, query, return_metadata=True)
    for qi, patches in enumerate(query.features):
        theta = mapper.image.pair_weights(patches.reshape(1, -1))[0]
        weighted = patches.new_zeros(())
        mass = patches.new_zeros(())
        for i, sid in enumerate(mapper.source.ids):
            for j, tid in enumerate(mapper.target.ids):
                pair = mapper._pair(sid, tid)
                _, confidence, detail = partial_projection(patches, mapper.x[i], mapper.y[j],
                    pair["plan"], pair["a"], pair["b"], mapper.shared["sx"][i], mapper.shared["sy"][j],
                    mapper.shared["h_projection"], mapper.shared["support"][i])
                entropy = -(detail["weights"] * detail["weights"].clamp_min(1e-300).log()).sum(1)
                weighted += (theta[i, j] * confidence * entropy.exp()).sum()
                mass += (theta[i, j] * confidence).sum()
        reported = result.diagnostics["queries"][qi]["within_image_effective_patches"]
        assert reported == pytest.approx(float(weighted / mass), abs=1e-12)
        assert 1 <= reported <= patches.shape[0]


def test_mapping_retains_query_diagnostics_when_all_tokens_rejected(tmp_path, banks):
    cfg = config("grouped_partial")
    cfg["projection"]["confidence"].update(support_calibration="fixed", support_log_threshold=1.)
    mapper = FeatureMapper.load(fit_mapping(cfg, root=tmp_path))
    destination = tmp_path / "failed_map"
    with pytest.raises(ValueError, match="All condition tokens rejected"):
        mapper.project_bank(banks[2], destination)
    attempt = next((destination / "logs").iterdir())
    records = [json.loads(line) for line in (attempt / "queries.jsonl").read_text().splitlines()]
    assert [r["query_id"] for r in records] == banks[2].ids
    assert all(r["all_invalid"] for r in records)
    assert not (destination / "manifest.json").exists()
    assert (attempt / "run.json").exists()
    with pytest.raises(FileExistsError):
        mapper.project_bank(banks[2], destination)


def test_mapping_summary_and_success_logs(tmp_path, banks):
    mapper = FeatureMapper.load(fit_mapping(config("whole_map"), root=tmp_path))
    result, _ = mapper.project_bank(banks[2], tmp_path / "mapping")
    attempt = next((tmp_path / "mapping/logs").iterdir())
    report = json.loads((attempt / "mapping_report.json").read_text())
    assert report["mapped"]["valid_tokens"] == 8
    assert report["all_invalid_images"] == 0
    assert report["confidence_quantiles"]["min"] == 1
    invalid = MappingResult(result.mapped_features, torch.zeros_like(result.match_confidence),
                            torch.zeros_like(result.valid_mask), result.diagnostics)
    summary = mapping_summary(invalid, banks[2].features, banks[0].features, banks[1].features)
    assert summary["mapped"]["norm_mean"] is None
    assert summary["mapped"]["image_mean_variance"] is None
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("which", ["features", "confidence", "shape", "range"])
def test_conditioning_rejects_invalid_cache_values(which):
    result = MappingResult(torch.zeros(1, 4, 3), torch.ones(1, 4), torch.ones(1, 4, dtype=torch.bool), {})
    if which == "features":
        result.mapped_features[0, 0, 0] = torch.inf
    elif which == "confidence":
        result.match_confidence[0, 0] = torch.nan
    elif which == "shape":
        result.match_confidence = torch.ones(1, 3)
    else:
        result.match_confidence[0, 0] = 1.1
    with pytest.raises(ValueError, match="Invalid mapped-feature"):
        result.conditioning(["cat_val"])


def test_mapping_rejects_zero_chunk_size_instead_of_silently_defaulting(tmp_path, banks):
    mapper = FeatureMapper.load(fit_mapping(config("whole_map"), root=tmp_path))
    with pytest.raises(ValueError, match="chunk_size"):
        mapped(mapper, banks[2], return_metadata=True, chunk_size=0)


def test_generation_rejects_stale_query_order_before_model_load(tmp_path, banks, monkeypatch):
    from diffusion_ot.evaluation import stage1a_eval
    from infoot_vit.infoot_helper.evaluate_mapping import generate
    mapper = FeatureMapper.load(fit_mapping(config("whole_map"), root=tmp_path))
    result = mapped(mapper, banks[2], return_metadata=True)
    swapped = deepcopy(banks[2])
    swapped.ids.reverse()
    monkeypatch.setattr(stage1a_eval, "load_stage1a_evaluator", lambda *a, **k: pytest.fail("should validate before loading"))
    with pytest.raises(ValueError, match="ordered stable IDs"):
        generate(mapper, swapped, result, tmp_path, root=tmp_path, train_config="none", eval_config="none", device="cpu")
