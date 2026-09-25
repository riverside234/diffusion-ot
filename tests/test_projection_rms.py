"""Reference EMA must remove query-batch calibration without hiding gradient changes."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from diffusion_ot.losses.encoder_transport import encoder_conditional_readout
from diffusion_ot.losses.infoot import infoot_distance_scale
from diffusion_ot.losses.projection_rms import (
    ReferenceRMSEMA, checkpoint_projection_rms, projection_rms_options, reference_variances,
)
from test_encoder_transport import banks
from test_stage1b_extensions import experiment, assert_tensor_tree_equal
from test_stage1b_self_supervised import own_encoder_run, self_supervised_recipe


OPTIONS = {"mode": "reference_ema", "decay": .99, "eps": 1e-8}


def rms_recipe(config):
    self_supervised_recipe(config)
    config["projection_rms"] = dict(OPTIONS)


def test_reference_variance_matches_infoot_pairwise_rms_and_averages_before_sqrt():
    _, _, refs, _, _ = banks()
    variances = reference_variances(refs)
    for domain, value in refs.items():
        normalized = F.normalize(value, dim=1)
        expected = torch.cdist(normalized, normalized).square().mean() / 2
        assert variances[domain] == pytest.approx(float(expected.detach()), abs=1e-12)
        assert variances[domain] == pytest.approx(float(infoot_distance_scale(normalized).square()), abs=1e-12)
    tracker = ReferenceRMSEMA(decay=.8)
    tracker.initialize(refs)
    changed = {d: value * .1 + torch.tensor([1., 0., 0.]) for d, value in refs.items()}
    new = reference_variances(changed)
    tracker.update(changed, step=1)
    for d in refs:
        assert tracker.scales()[d] ** 2 == pytest.approx(.8 * variances[d] + .2 * new[d])
    assert all(value.grad is None for value in refs.values())


def test_state_restore_continues_exactly_and_rejects_duplicate_updates():
    _, _, refs, _, _ = banks()
    tracker = ReferenceRMSEMA()
    tracker.initialize(refs)
    tracker.update(refs, step=1)
    restored = ReferenceRMSEMA()
    restored.load_state_dict(tracker.state_dict(), step=1)
    new = {d: value + 1 for d, value in refs.items()}
    for item in (tracker, restored):
        item.update(new, step=2)
    assert tracker.state_dict() == restored.state_dict()
    with pytest.raises(ValueError, match="exactly once"):
        restored.update(new, step=2)
    # Returned snapshots cannot mutate the tracker.
    snapshot = restored.state_dict()
    snapshot["variances"]["cat"] = 123
    assert restored.state_dict() == tracker.state_dict()
    for key, value in (("decay", .8), ("num_updates", 0), ("variances", {"cat": float("nan"), "dog": 1.})):
        bad = deepcopy(tracker.state_dict())
        bad[key] = value
        with pytest.raises(ValueError, match="Projection RMS|projection RMS"):
            restored.load_state_dict(bad, step=2)


@pytest.mark.parametrize("options", [
    {"mode": "unknown"}, {"mode": "query_batch", "decay": .9},
    {**OPTIONS, "decay": 1.}, {**OPTIONS, "decay": float("nan")},
    {**OPTIONS, "eps": 0.}, {**OPTIONS, "unexpected": True},
])
def test_invalid_protocol_rejected(options):
    with pytest.raises(ValueError, match="projection_rms"):
        projection_rms_options({"projection_rms": options})


def test_collapsed_references_remain_finite_and_are_visible_in_scale_lag():
    _, _, refs, _, _ = banks()
    tracker = ReferenceRMSEMA(eps=1e-5)
    tracker.initialize(refs)
    tracker.update({d: torch.ones_like(value) for d, value in refs.items()}, step=1)
    diagnostics = tracker.diagnostics(bandwidth=.1)
    assert diagnostics["running_to_batch_ratio"]["cat"] > 1000
    assert diagnostics["batch_scales"]["cat"] == pytest.approx(1e-5)
    assert all(torch.isfinite(torch.tensor(v)) for v in tracker.scales().values())


def test_frozen_scales_remove_query_batch_dependence_and_keep_true_gradients():
    refs, queries, match_ref, match_query, coupling = banks()
    tracker = ReferenceRMSEMA()
    tracker.initialize(match_ref)
    scales = tracker.scales()
    state = tracker.state_dict()

    def readout(cat_ref, dog_ref, cat_query, dog_query):
        result = encoder_conditional_readout(
            refs, {"cat": queries["cat"][:len(cat_query)], "dog": queries["dog"][:len(dog_query)]},
            coupling, bandwidth=.7, reference_matching={"cat": cat_ref, "dog": dog_ref},
            query_matching={"cat": cat_query, "dog": dog_query},
            differentiate_distance_scale=True, projection_scales=scales)
        return result.log_weights["cat_to_dog"], result.log_weights["dog_to_cat"]

    full = readout(*match_ref.values(), *match_query.values())
    single = readout(*match_ref.values(), *(v[:1] for v in match_query.values()))
    reordered = readout(*match_ref.values(), *(v.flip(0) for v in match_query.values()))
    for a, b, c in zip(full, single, reordered):
        torch.testing.assert_close(a[:1], b, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(a, c.flip(0), atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(readout, (*match_ref.values(), *match_query.values()), fast_mode=True)
    sum(x.square().mean() for x in full).backward()
    assert coupling.grad is None
    assert all(x.grad is not None and x.grad.norm() > 0 for x in (*match_ref.values(), *match_query.values()))
    assert tracker.state_dict() == state


def test_ema_projection_fixes_the_upstream_outlier_query_counterexample():
    cat = F.normalize(torch.tensor([[1., 0.], [1., .1], [1., -.1], [1., .3]], dtype=torch.float64), dim=1)
    dog = torch.eye(4, dtype=torch.float64)
    query_cat = F.normalize(torch.tensor([[1., .15], [-1., 0.]], dtype=torch.float64), dim=1)
    refs = {"cat": cat, "dog": dog}
    tracker = ReferenceRMSEMA()
    tracker.initialize(refs)
    for scales, invariant in ((None, False), (tracker.scales(), True)):
        results = []
        for query in (query_cat[:1], query_cat):
            output = encoder_conditional_readout(refs, {"cat": query, "dog": dog[:1]}, dog / 4,
                bandwidth=.1, projection_scales=scales)
            results.append(output.weights["cat_to_dog"][0])
        difference = float((results[0] - results[1]).abs().sum())
        assert difference < 1e-12 if invariant else difference > .9


def test_checkpoint_selects_raw_or_model_ema_geometry_and_fails_on_missing_state():
    _, _, refs, _, _ = banks()
    raw, ema = ReferenceRMSEMA(), ReferenceRMSEMA()
    raw.initialize(refs)
    ema.initialize({d: value + 1 for d, value in refs.items()})
    config = {"projection_rms": OPTIONS}
    cp = {"config": config, "step": 0, "projection_rms_state": {
        "raw": raw.state_dict(), "ema": ema.state_dict()}}
    for weights, tracker in (("raw", raw), ("ema", ema)):
        assert checkpoint_projection_rms(cp, config, weights=weights).scales() == tracker.scales()
    assert raw.scales() != ema.scales()
    del cp["projection_rms_state"]["ema"]
    with pytest.raises(ValueError, match="no projection_rms_state.ema"):
        checkpoint_projection_rms(cp, config, weights="ema")
    with pytest.raises(ValueError, match="protocols disagree"):
        checkpoint_projection_rms(cp, {}, weights="raw")


def test_training_and_standalone_readouts_agree_with_frozen_scales():
    from diffusion_ot.evaluation.stage1b_eval import LatentBank, _direction_projection_evaluation
    refs, queries, match_ref, match_query, coupling = banks()
    tracker = ReferenceRMSEMA()
    tracker.initialize(match_ref)
    def bank(domain, query=False):
        raw = (queries if query else refs)[domain].double()
        features = (match_query if query else match_ref)[domain]
        split = "val" if query else "train"
        return LatentBank(domain, split, raw, F.normalize(features.detach(), dim=1),
                          [f"{domain}_{split}_{i}" for i in range(len(raw))],
                          [{} for _ in raw], "test_checkpoint")
    train = encoder_conditional_readout(refs, queries, coupling, bandwidth=.7,
        reference_matching=match_ref, query_matching=match_query, projection_scales=tracker.scales())
    for source, target in (("cat", "dog"), ("dog", "cat")):
        report, tensors = _direction_projection_evaluation(
            bank(source), bank(target), bank(source, True), (coupling if source == "cat" else coupling.T).detach(),
            source_scale=.9, target_scale=.8, bandwidth=.55, eps=1e-8,
            distance_scale_mode="infoot_rms", projection_bandwidth=.7, projection_scales=tracker.scales())
        torch.testing.assert_close(tensors["conditional_weights"], train.weights[f"{source}_to_{target}"],
                                   atol=1e-10, rtol=1e-10)
        assert report["conditional_distance_scales"]["query_source"] == tracker.scales()[source]


def test_training_tracks_references_once_keeps_live_fit_gradients_and_resumes(own_encoder_run, monkeypatch):
    import diffusion_ot.losses.infoot as infoot
    run, _, _, _ = own_encoder_run
    original_loss, original_update = infoot.plain_infoot_feature_loss, ReferenceRMSEMA.update
    gradient_modes, updates = [], []

    def loss(*args, **kwargs):
        if torch.is_grad_enabled():
            gradient_modes.append((kwargs["distance_scale_x"].requires_grad,
                                   kwargs["distance_scale_y"].requires_grad))
        return original_loss(*args, **kwargs)

    def update(self, references, *, step):
        updates.append((step, {d: len(x) for d, x in references.items()}))
        return original_update(self, references, step=step)

    monkeypatch.setattr(infoot, "plain_infoot_feature_loss", loss)
    monkeypatch.setattr(ReferenceRMSEMA, "update", update)
    cp, logs = run("ema_projection", steps=1, modify=rms_recipe)
    assert gradient_modes == [(True, True)]
    assert updates == [(1, {"cat": 6, "dog": 6})] * 2  # Raw / EMA, excludes 2 queries.
    assert set(cp["projection_rms_state"]) == {"raw", "ema"}
    assert [v["projection_rms"]["num_updates"] for v in logs["validation"]] == [0, 1]
    for row in logs["validation"]:
        for metrics in row["conditional_projection"].values():
            assert metrics["query_batch_dependence_first_query_l1"] < 2e-5
    resumed, resumed_logs = run("ema_projection", resume=True, modify=rms_recipe)
    assert all(s["num_updates"] == 2 for s in resumed["projection_rms_state"].values())
    # Resume step-1 validation must reuse the exact saved statistics.
    resumed_initial = resumed_logs["validation"][-2]
    assert resumed_initial["step"] == 1
    assert resumed_initial["projection_rms"]["scales"] == logs["validation"][-1]["projection_rms"]["scales"]
    with pytest.raises(ValueError, match="Resume cannot change projection_rms"):
        run("ema_projection", resume=True, modify=self_supervised_recipe)
    def changed_model_ema(config):
        rms_recipe(config)
        config["ema"]["decay"] = .8
    with pytest.raises(ValueError, match="Resume cannot change ema"):
        run("ema_projection", resume=True, modify=changed_model_ema)


def test_model_ema_geometry_is_measured_using_its_own_weights(own_encoder_run):
    from diffusion_ot.training.train_joint_infoot import EncoderEMA, _projection_reference_features
    from diffusion_ot.models.matching_head import make_matching_head
    run, _, domains, _ = own_encoder_run
    cp, _ = run("rms_feature_models", steps=1, modify=rms_recipe)
    encoders = {d: v.branch.encoder for d, v in domains.items()}
    heads = {d: make_matching_head(cp["config"]["matching_head"], device="cpu") for d in domains}
    ema = EncoderEMA(encoders)
    head_ema = EncoderEMA(heads)
    latents = {d: torch.randn(5, 4, 8, 8) for d in domains}
    before = _projection_reference_features(domains, latents, heads, ema=ema, matching_ema=head_ema)
    with torch.no_grad():
        for encoder in encoders.values():
            for parameter in encoder.parameters():
                parameter.add_(torch.randn_like(parameter) * .1)
    raw = _projection_reference_features(domains, latents, heads)
    after = _projection_reference_features(domains, latents, heads, ema=ema, matching_ema=head_ema)
    assert_tensor_tree_equal(before, after)
    assert any(not torch.equal(raw[d], before[d]) for d in raw)


def test_validation_does_not_update_running_scales_or_training_rng(own_encoder_run):
    run, _, _, _ = own_encoder_run
    measured, _ = run("rms_measured", modify=rms_recipe)
    def quiet(config):
        rms_recipe(config)
        config["train"]["validation_every"] = 0
    silent, _ = run("rms_silent", modify=quiet)
    for key in ("projection_rms_state", "encoders", "matching_heads", "generators", "rng_state"):
        assert_tensor_tree_equal(measured[key], silent[key])


@pytest.mark.parametrize("stem", ["self_supervised", "self_supervised_patchnce"])
def test_new_configs_change_only_projection_calibration_and_output_paths(stem):
    root = Path(__file__).resolve().parents[1]
    for directory in ("stage1b_infoot", "stage1b_eval"):
        old = yaml.safe_load((root / f"configs/{directory}/{stem}_sit_b2.yaml").read_text())
        new = yaml.safe_load((root / f"configs/{directory}/{stem}_rms_ema_sit_b2.yaml").read_text())
        assert new.pop("projection_rms") == OPTIONS
        assert new.pop("output_dir") == old.pop("output_dir") + "_rms"
        if "quick_evaluation" in new:
            assert "rms_ema" in new["quick_evaluation"]["config"]
            new["quick_evaluation"]["config"] = old["quick_evaluation"]["config"]
        assert new == old
