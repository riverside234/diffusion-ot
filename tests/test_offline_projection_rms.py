"""Offline inference must retain the Stage 1B projection geometry and provenance."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from diffusion_ot.evaluation.full_infoot import project_full, reference_rms
from diffusion_ot.evaluation.offline_pipeline import checkpoint_rms_calibration, require_projection_mode
from diffusion_ot.losses.encoder_transport import encoder_conditional_readout
from diffusion_ot.losses.projection_rms import ReferenceRMSEMA


def test_offline_projection_matches_training_ema_readout_and_is_query_independent():
    rng = torch.Generator().manual_seed(46)
    refs = {d: torch.randn(n, 6, generator=rng, dtype=torch.float64)
            for d, n in (("cat", 7), ("dog", 9))}
    matching = {d: F.normalize(torch.randn(len(r), 4, generator=rng, dtype=torch.float64), dim=1)
                for d, r in refs.items()}
    query_matching = {d: F.normalize(torch.randn(5, 4, generator=rng, dtype=torch.float64), dim=1) for d in refs}
    queries = {d: torch.randn(5, 6, generator=rng, dtype=torch.float64) for d in refs}
    tracker = ReferenceRMSEMA(decay=.9)
    tracker.initialize({d: features + 1 for d, features in matching.items()})
    tracker.update(matching, step=1)
    config = {"projection_rms": {"mode": "reference_ema", "decay": .9}}
    checkpoint = {"config": config, "step": 1, "projection_rms_state": {"ema": tracker.state_dict()}}
    original = deepcopy(checkpoint)
    metadata, scales = checkpoint_rms_calibration(checkpoint, config, "ema")
    assert metadata["num_updates"] == 1
    assert scales != {d: reference_rms(features) for d, features in matching.items()}
    # A balanced, non-independent rectangular coupling; unequal domains matter.
    from diffusion_ot.evaluation.full_infoot import FullFitSettings, tiled_sinkhorn
    a, b = torch.full((7,), 1 / 7, dtype=torch.float64), torch.full((9,), 1 / 9, dtype=torch.float64)
    plan, checks, _ = tiled_sinkhorn(1 - matching["cat"] @ matching["dog"].T, a, b,
                                   FullFitSettings(entropy_epsilon=.5, block_size=3))
    assert checks["feasible"]
    training = encoder_conditional_readout(refs, queries, plan, bandwidth=.1,
        reference_matching=matching, query_matching=query_matching, projection_scales=scales)
    for source, target in (("cat", "dog"), ("dog", "cat")):
        expected = training.weights[f"{source}_to_{target}"] @ refs[target]
        for query_batch, target_block in ((1, 1), (2, 3), (20, 20)):
            def project(query):
                return project_full(query, matching[source], matching[target], plan if source == "cat" else plan.T,
                                    refs[target], source_scale=scales[source], target_scale=scales[target],
                                    bandwidth=.1, query_batch_size=query_batch, target_block_size=target_block)
            actual = project(query_matching[source])
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(project(query_matching[source][:1]), expected[:1], rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(project(query_matching[source].flip(0)), expected.flip(0), rtol=1e-10, atol=1e-10)
    assert checkpoint == original  # Evaluation never updates reference statistics.


def test_offline_rms_is_strict_about_history_and_recipe():
    config = {"projection_rms": {"mode": "reference_ema", "decay": .99}}
    state = {"version": 1, "decay": .99, "eps": 1e-8, "num_updates": 12,
             "variances": {"cat": .25, "dog": .36}, "batch_variances": {"cat": .2, "dog": .3}}
    checkpoint = {"config": config, "step": 12, "projection_rms_state": {"raw": state}}
    with pytest.raises(ValueError, match="no projection_rms_state.ema"):
        checkpoint_rms_calibration(checkpoint, config, "ema")
    with pytest.raises(ValueError, match="protocols disagree"):
        checkpoint_rms_calibration(checkpoint, {}, "raw")
    stale = deepcopy(checkpoint)
    stale["step"] = 13
    with pytest.raises(ValueError, match="update count"):
        checkpoint_rms_calibration(stale, config, "raw")
    broken = deepcopy(checkpoint)
    broken["projection_rms_state"]["raw"]["variances"]["dog"] = float("nan")
    with pytest.raises(ValueError, match="variances"):
        checkpoint_rms_calibration(broken, config, "raw")
    meta, scales = checkpoint_rms_calibration(checkpoint, config, "raw")
    assert scales == {"cat": .5, "dog": .6}
    require_projection_mode({"require_projection_rms_mode": "reference_ema"}, meta["mode"])
    with pytest.raises(ValueError, match="requires projection RMS"):
        require_projection_mode({"require_projection_rms_mode": "reference_ema"}, "full_bank")
    with pytest.raises(ValueError, match="must be"):
        require_projection_mode({"require_projection_rms_mode": "query_batch"}, "full_bank")


def test_main_offline_recipes_follow_current_self_supervised_ema_experiment():
    root = Path(__file__).resolve().parents[1]
    load = lambda path: yaml.safe_load((root / path).read_text(encoding="utf-8"))
    stage23 = load("configs/stage23_offline/full_sit_b2.yaml")
    stage4 = load("configs/stage4_eval/fid_ssim_sit_b2.yaml")
    stage1b = load(stage23["alignment_config"])
    assert stage23["require_projection_rms_mode"] == stage4["require_projection_rms_mode"] == "reference_ema"
    assert stage1b["projection_rms"]["mode"] == "reference_ema"
    assert stage1b["decoded_translation"]["supervision"] == "self_supervised"
    assert stage1b["decoded_translation"]["patchnce"]["sampler"] == "mlp_sample"
    assert stage1b["infoot"]["cross_cost_source"] == "encoder"
    assert stage23["fit"]["bandwidth"] == stage1b["matching"]["bandwidth_multiplier"] == .55
    assert stage23["projection_bandwidth"] == stage1b["conditional_projection"]["bandwidth_multiplier"] == .1
    for key in ("mi_weight", "entropy_epsilon", "cross_cost_weight"):
        assert stage23["fit"][key] == stage1b["infoot"][key]
