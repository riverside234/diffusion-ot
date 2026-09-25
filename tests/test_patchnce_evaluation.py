"""Checkpoint metadata must distinguish global and spatial retrieval experiments."""
from copy import deepcopy

import pytest

from diffusion_ot.evaluation.stage1b_eval import _validate_self_supervised_checkpoint


def _config():
    return {"decoded_translation": {
        "supervision": "self_supervised", "objective": "patchnce",
        "source_contrastive_readout": "target", "source_contrastive_weight": 0.,
        "patchnce": {"weight": .15, "temperature": .20, "num_patches": 64, "layers": [0, 1, 2]}},
        "infoot": {"variant": "fused", "cross_cost_source": "encoder"}}


def test_patchnce_evaluation_accepts_same_protocol_and_effective_defaults():
    config = _config()
    _validate_self_supervised_checkpoint(config, {"config": deepcopy(config)})
    saved = deepcopy(config)
    saved["decoded_translation"].pop("patchnce")
    _validate_self_supervised_checkpoint(config, {"config": saved})


@pytest.mark.parametrize("current_patch", [False, True])
def test_patchnce_evaluation_rejects_global_spatial_objective_mismatch(current_patch):
    config, saved = _config(), _config()
    (saved if current_patch else config)["decoded_translation"].pop("objective")
    with pytest.raises(ValueError, match="objective"):
        _validate_self_supervised_checkpoint(config, {"config": saved})


@pytest.mark.parametrize("key,value", [
    ("weight", .3), ("temperature", .07), ("num_patches", 32),
    ("layers", [1, 2]), ("include_all_negatives_from_minibatch", True),
])
def test_patchnce_evaluation_rejects_changed_protocol(key, value):
    config, saved = _config(), _config()
    saved["decoded_translation"]["patchnce"][key] = value
    with pytest.raises(ValueError, match="PatchNCE protocol"):
        _validate_self_supervised_checkpoint(config, {"config": saved})


def test_patchnce_evaluation_keeps_cross_encoder_readout_guard():
    config, saved = _config(), _config()
    saved["decoded_translation"]["source_contrastive_readout"] = "source"
    with pytest.raises(ValueError, match="readout"):
        _validate_self_supervised_checkpoint(config, {"config": saved})
