from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "infoot"))
    from infoot_helper import infoot_cotraining_helper
    from infoot_helper.reference_rms import ReferenceRMSEMA, load_projection_scales
    return infoot_cotraining_helper, ReferenceRMSEMA, load_projection_scales


@pytest.fixture
def references():
    return {
        "cat": torch.tensor([[0., 1.], [1., 3.], [2., 2.], [4., 5.]], dtype=torch.float64),
        "dog": torch.tensor([[10., 100.], [30., 100.], [60., 100.], [90., 100.]], dtype=torch.float64),
    }


def test_raw_variance_ema_matches_pairwise_rms_and_preserves_inputs(helpers, references):
    _, Tracker, _ = helpers
    tracker = Tracker()
    originals = deepcopy(references)
    for v in references.values():
        v.requires_grad_()
    scales = tracker.update(references, step=1)
    initial = tracker.variance.clone()
    for name, v in references.items():
        expected = torch.cdist(v, v).square().mean() / 2
        assert scales[name] ** 2 == pytest.approx(float(expected.detach()))
        torch.testing.assert_close(v, originals[name])
        assert v.grad is None
    tracker.update({name: 3 * v + 1000 for name, v in references.items()}, step=2)
    torch.testing.assert_close(tracker.variance, .99 * initial + .01 * 9 * initial)
    assert not tracker.variance.requires_grad and not list(tracker.parameters())
    assert tracker.num_updates == 2


def test_tracker_requires_once_per_step_updates_and_handles_zero_spread(helpers, references):
    _, Tracker, _ = helpers
    tracker = Tracker(eps=1e-6)
    with pytest.raises(ValueError, match="initialized"):
        tracker.scales()
    with pytest.raises(ValueError, match="once"):
        tracker.update(references, step=2)
    scales = tracker.update({name: torch.ones_like(v) for name, v in references.items()}, step=1)
    assert scales == dict(cat=1e-6, dog=1e-6)
    with pytest.raises(ValueError, match="once"):
        tracker.update(references, step=1)
    with pytest.raises(ValueError, match="finite"):
        tracker.update({name: v * float("nan") for name, v in references.items()}, step=2)
    assert tracker.num_updates == 1


def test_frozen_scale_mapping_matches_existing_kernel_and_keeps_gradients(helpers, references):
    helper, Tracker, _ = helpers
    tracker = Tracker()
    scales = tracker.update(references, step=1)
    query = torch.tensor([[.5, 1.5], [2.5, 3.]], dtype=torch.float64, requires_grad=True)
    source, target = (v.requires_grad_() for v in references.values())
    plan = (torch.eye(4, dtype=torch.float64) / 4).requires_grad_()
    expected = helper.conditional_mapping(query, source, target, plan, h=.8)
    pair = (scales["cat"], scales["dog"])
    actual = helper.conditional_mapping(query, source, target, plan, h=.8, scales=pair)
    torch.testing.assert_close(actual, expected)
    state = deepcopy(tracker.state_dict())
    separate = torch.cat([
        helper.conditional_mapping(row[None], source, target, plan, h=.8, scales=pair)
        for row in query
    ])
    torch.testing.assert_close(actual, separate)
    torch.testing.assert_close(actual[:, 1], torch.full((2,), 100., dtype=torch.float64))
    actual.square().mean().backward()
    for v in (query, source, target):
        assert torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
    assert plan.grad is None
    for name, value in tracker.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_ema_scale_controls_both_directions_without_reestimating(helpers, references):
    helper, Tracker, _ = helpers
    from diffusion_ot.losses.infoot import conditional_reference_log_weights

    tracker = Tracker()
    tracker.update({name: v * .5 for name, v in references.items()}, step=1)
    scales = tracker.scales()
    plan = torch.eye(4, dtype=torch.float64) / 4
    for source, target in (("cat", "dog"), ("dog", "cat")):
        v_source, v_target = references[source], references[target]
        query = v_source[:2] + .2
        probabilities = conditional_reference_log_weights(
            query, v_source, v_target, plan, bandwidth=.8,
            distance_scale_x=scales[source], distance_scale_y=scales[target],
        ).exp()
        expected = probabilities @ v_target
        actual = helper.conditional_mapping(
            query, v_source, v_target, plan, h=.8, scales=(scales[source], scales[target]),
        )
        torch.testing.assert_close(actual, expected)
        old = helper.conditional_mapping(query, v_source, v_target, plan, h=.8)
        assert not torch.allclose(actual, old)


def test_clustered_float32_mapping_matches_float64_and_single_queries(helpers):
    helper, Tracker, _ = helpers
    generator = torch.Generator().manual_seed(34)
    base = torch.randn(1, 512, generator=generator)
    source = torch.nn.functional.layer_norm(
        base + .001 * torch.randn(24, 512, generator=generator), (512,),
    )
    target = torch.nn.functional.layer_norm(
        torch.randn(24, 512, generator=generator), (512,),
    )
    query = torch.nn.functional.layer_norm(
        source[:8] + .0001 * torch.randn(8, 512, generator=generator), (512,),
    )
    tracker = Tracker()
    scales = tracker.update({"cat": source, "dog": target}, step=1)
    scales = (scales["cat"], scales["dog"])
    plan = torch.eye(24) / 24
    together = helper.conditional_mapping(query, source, target, plan, scales=scales)
    separate = torch.cat([
        helper.conditional_mapping(row[None], source, target, plan, scales=scales)
        for row in query
    ])
    expected = helper.conditional_mapping(
        query.double(), source.double(), target.double(), plan.double(), scales=scales,
    )
    torch.testing.assert_close(together, expected.float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(together, separate, rtol=1e-5, atol=1e-6)


def test_rms_checkpoint_roundtrip_and_frozen_mapping(helpers, references, tmp_path):
    helper, Tracker, load_scales = helpers
    tracker = Tracker(decay=.8)
    tracker.update(references, step=1)
    tracker.update({name: v * 2 for name, v in references.items()}, step=2)
    branch = SimpleNamespace(pdae_state_dict=lambda: {"encoder": {}})
    domains = {name: SimpleNamespace(branch=branch, training_config={}) for name in references}
    helper.save_domain_checkpoints(domains, tmp_path, 2, tracker)
    paths = {name: tmp_path / f"{name}_step_000002.pt" for name in references}
    saved = torch.load(paths["cat"], weights_only=True)
    restored = Tracker()
    restored.load_state_dict(saved["projection_rms"])
    assert load_scales(paths, step=2) == tracker.scales() == restored.scales()
    source, target = references.values()
    plan = torch.eye(4, dtype=torch.float64) / 4
    before = helper.conditional_mapping(source[:2] + .5, source, target, plan,
                                        scales=tuple(tracker.scales().values()))
    after = helper.conditional_mapping(source[:2] + .5, source, target, plan,
                                       scales=tuple(load_scales(paths, step=2).values()))
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    for item in (restored, tracker):
        item.update(references, step=3)
    torch.testing.assert_close(restored.variance, tracker.variance, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["missing", "step", "history"])
def test_checkpoint_rejects_missing_or_mismatched_history(helpers, references, tmp_path, fault):
    _, Tracker, load_scales = helpers
    tracker = Tracker()
    tracker.update(references, step=1)
    paths = {name: tmp_path / f"{name}.pt" for name in references}
    for name, path in paths.items():
        state = {"step": 1, "domain": name, "projection_rms": deepcopy(tracker.state_dict())}
        if name == "dog":
            if fault == "missing":
                del state["projection_rms"]
            elif fault == "step":
                state["step"] = 2
            else:
                state["projection_rms"]["variance"].mul_(2)
        torch.save(state, path)
    with pytest.raises(ValueError, match="RMS"):
        load_scales(paths, step=1)
