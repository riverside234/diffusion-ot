"""Rejected resume attempts must not invalidate existing mapping artifacts."""
from copy import deepcopy
import json

import pytest
import torch

from test_infoot_vit_mapping import banks, cpu_threads, mapped
from test_infoot_vit_lowrank import tiny_config
from test_infoot_vit_lowrank_partial import partial_config
from infoot_vit.infoot_helper.feature_bank import file_hash
from infoot_vit.infoot_helper.mapping import FeatureMapper
from infoot_vit.lowrank import experiment


def artifact_hashes(directory):
    return {str(path.relative_to(directory)): file_hash(path)
            for path in directory.rglob("*")
            if path.is_file() and "logs" not in path.relative_to(directory).parts}


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_rejected_resume_preserves_manifest_and_artifacts(tmp_path, banks, partial, complete):
    config = partial_config() if partial else tiny_config()
    if not complete:
        config["optimizer"].update(max_steps=1, stationarity_tolerance=1e-15)
        with pytest.raises(RuntimeError, match="max_steps"):
            experiment.fit(config, root=tmp_path)
        directory = next((tmp_path / "outputs/infoot_vit").iterdir())
    else:
        directory = experiment.fit(config, root=tmp_path)
        before_mapping = mapped(FeatureMapper.load(directory), banks[2], return_metadata=True)

    before = artifact_hashes(directory)
    changed = deepcopy(config)
    changed["optimizer"]["lam"] = .2
    with pytest.raises(ValueError, match="resume fingerprint changed"):
        experiment.fit(changed, root=tmp_path, resume=directory)

    assert artifact_hashes(directory) == before
    attempts = [json.loads(path.read_text()) for path in (directory / "logs").glob("*/run.json")]
    assert len(attempts) == 2
    assert any(run.get("error", "").startswith("Low-rank resume fingerprint changed") for run in attempts)
    if complete:
        after_mapping = mapped(FeatureMapper.load(directory), banks[2], return_metadata=True)
        torch.testing.assert_close(after_mapping.mapped_features, before_mapping.mapped_features, atol=0, rtol=0)
        torch.testing.assert_close(after_mapping.match_confidence, before_mapping.match_confidence, atol=0, rtol=0)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("failure_at", ["verification", "cleanup"])
def test_completed_resume_error_preserves_completed_manifest(tmp_path, banks, monkeypatch, partial, failure_at):
    config = partial_config() if partial else tiny_config()
    directory = experiment.fit(config, root=tmp_path)
    before = artifact_hashes(directory)

    def fail(*args, **kwargs):
        raise OSError("simulated completed-resume I/O failure")

    with monkeypatch.context() as patch:
        if failure_at == "verification":
            from infoot_vit.lowrank.mapping import LowRankMapper
            patch.setattr(LowRankMapper, "load", fail)
        else:
            patch.setattr(experiment, "_cleanup_latest", fail)
        with pytest.raises(OSError, match="completed-resume I/O failure"):
            experiment.fit(config, root=tmp_path, resume=directory)

    assert artifact_hashes(directory) == before
    assert FeatureMapper.load(directory).manifest["status"] == "complete"
