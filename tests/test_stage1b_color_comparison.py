import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("compare_stage1b_color", Path(__file__).resolve().parents[1] / "scripts/compare_stage1b_color.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(values):
    return {"step": 500, "weights": "raw", "projection_probe": {"fit_bandwidth": .55, "projection_bandwidth": .1},
            "decoded_translation": {"seed": 20271905, "num_steps": 20, "guidance_scale": 1.,
                "query_ids": {"cat": ["c1", "c2"], "dog": ["d1", "d2"]},
                "reference_ids": {"cat": ["cr"], "dog": ["dr"]},
                "color_histogram_protocol": "histogan_rgbuv_hellinger_v1",
                "color_histogram_parameters": {"bins": 64, "input_size": 128, "sigma": .02},
                "color_histogram_target": "original_source_rgb_whole_image",
                **{direction: {"per_image_color_histogram_loss": values} for direction in ("cat_to_dog", "dog_to_cat")}}}


def test_comparison_uses_paired_examples_and_rejects_changed_protocol(tmp_path):
    a, b = tmp_path / "baseline.jsonl", tmp_path / "candidate.jsonl"
    a.write_text(json.dumps(record([.3, .5])) + "\n")
    b.write_text(json.dumps(record([.2, .4])) + "\n")
    result = module.compare_validation_logs(a, b)
    for row in result["directions"].values():
        assert row["paired_mean_delta"] == pytest.approx(-.1)
        assert row["fraction_improved"] == 1.
    for key, value in (("seed", 2), ("guidance_scale", 2.), ("query_ids", {"cat": ["c2", "c1"], "dog": ["d1", "d2"]}),
                       ("color_histogram_parameters", {"bins": 32}), ("color_histogram_target", "vae_source")):
        changed = record([.2, .4])
        changed["decoded_translation"][key] = value
        b.write_text(json.dumps(changed) + "\n")
        with pytest.raises(ValueError, match=key):
            module.compare_validation_logs(a, b)
    for key in ("fit_bandwidth", "projection_bandwidth"):
        for value in (None, .7):
            changed = record([.2, .4])
            changed["projection_probe"][key] = value
            b.write_text(json.dumps(changed) + "\n")
            with pytest.raises(ValueError, match=key):
                module.compare_validation_logs(a, b)
