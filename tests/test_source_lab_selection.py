"""Numerical and gradient contracts for the v6 appearance/selection experiment."""
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.losses.lab_swd import normalized_lab, source_lab_swd
from diffusion_ot.losses.source_selection import (
    SourceAwareSelection, source_selection_options, checkpoint_source_selection,
)
from diffusion_ot.losses.translation_image import texture_diagnostic


def rng(seed=42):
    return torch.Generator().manual_seed(seed)


def test_lab_known_d65_colors_and_finite_black_gradient():
    colors = torch.tensor([[0., 0., 0.], [1., 1., 1.], [1., 0., 0.]]).reshape(3, 3, 1, 1).requires_grad_()
    lab = normalized_lab(colors)
    expected = torch.tensor([[0., 0., 0.], [1., 0., 0.], [.532408, .625722, .525025]])
    torch.testing.assert_close(lab[:, :, 0, 0], expected, atol=1e-5, rtol=0)
    lab.sum().backward()
    assert torch.isfinite(colors.grad).all()
    dark = torch.full((1, 3, 1, 1), 1e-4, requires_grad=True)
    normalized_lab(dark).sum().backward()
    assert torch.isfinite(dark.grad).all() and dark.grad.norm() > 0


def test_lab_swd_matches_explicit_patch_projection_and_detaches_source():
    generated = torch.rand(2, 3, 8, 8, generator=rng()).requires_grad_()
    source = torch.rand(2, 3, 8, 8, generator=rng(3)).requires_grad_()
    state = torch.random.get_rng_state().clone()
    measured = source_lab_swd(generated, source, generator=rng(), sizes=[8], patch_size=3, directions=7)
    axes = F.normalize(torch.randn(7, 27, generator=rng()), dim=1)
    def reference(rgb):
        patches = F.unfold(F.pad(normalized_lab(rgb), (1, 1, 1, 1), mode="reflect"), 3)
        return (axes @ patches).sort(2).values
    expected = (reference(generated) - reference(source)).abs().mean((1, 2))
    torch.testing.assert_close(measured, expected)
    measured.mean().backward()
    assert torch.isfinite(generated.grad).all() and generated.grad.norm() > 0
    assert source.grad is None
    assert torch.equal(state, torch.random.get_rng_state())


def test_lab_swd_identity_exposure_hue_and_per_image_pairing():
    source = torch.rand(2, 3, 16, 16, generator=rng()) * .6 + .1
    def loss(actual, target=source):
        return source_lab_swd(actual, target, generator=rng(), sizes=[16, 8, 4], patch_size=3, directions=16)
    assert torch.equal(loss(source), torch.zeros(2))
    assert (loss(source + .2) > .01).all()
    assert (loss(source[:, [1, 2, 0]]) > 0).all()
    torch.testing.assert_close(loss(source + .1)[:1], loss((source + .1)[:1], source[:1]))


def selection(weights=(.25, .25)):
    config = {"source_aware_selection": dict(enabled=True, spatial_weight=weights[0], appearance_weight=weights[1])}
    runtime = SourceAwareSelection(source_selection_options(config))
    # Symmetric within-image relationship descriptors, not shared raw encoder axes.
    spatial = {d: F.normalize(torch.randn(4, 2, 4, 4, generator=rng(seed)), dim=-1) for d, seed in (("cat", 1), ("dog", 2))}
    appearance = {d: torch.rand(4, 12, generator=rng(seed)) for d, seed in (("cat", 3), ("dog", 4))}
    ids = {d: [f"{d}_train_{i}" for i in range(4)] for d in spatial}
    runtime.calibrate(spatial, appearance, ids)
    return runtime, config


def test_selection_formula_detached_costs_and_query_batch_independence():
    runtime, _ = selection()
    logits = torch.randn(3, 4, generator=rng(), requires_grad=True)
    costs = {k: torch.rand(3, 4, generator=rng(seed), requires_grad=True) for k, seed in (("spatial", 2), ("appearance", 3))}
    codes = torch.randn(4, 6, generator=rng(5))
    logp = logits.log_softmax(1)
    actual, metrics = runtime.apply(logp, costs, direction="cat_to_dog", target_codes=codes)
    scales = runtime.calibration["scales"]["cat_to_dog"]
    expected = ((logp.exp() + 1e-8).log() - sum(.25 * costs[k].detach() / scales[k] for k in costs)).log_softmax(1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.exp().sum(1), torch.ones(3))
    single, _ = runtime.apply(logp[:1], {k: v[:1] for k, v in costs.items()}, direction="cat_to_dog", target_codes=codes)
    torch.testing.assert_close(single, actual[:1], rtol=0, atol=0)
    (actual.exp() @ codes).square().sum().backward()
    assert logits.grad.norm() > 0 and all(v.grad is None for v in costs.values())
    assert metrics["mean_weight_l1_change"] > 0
    assert sum(metrics["cost_gain"][k] / scales[k] for k in costs) > 0


def test_selection_disabled_coefficients_preserve_exact_forward_and_gradients():
    runtime, _ = selection((0., 0.))
    logits = torch.randn(2, 4, generator=rng(), requires_grad=True)
    logp = logits.log_softmax(1)
    actual, _ = runtime.apply(logp, dict(spatial=torch.ones(2, 4), appearance=torch.ones(2, 4)),
                               direction="dog_to_cat", target_codes=torch.eye(4))
    assert actual is logp
    assert torch.equal(torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
                       torch.autograd.grad(logp.sum(), logits)[0])


def test_selection_checkpoint_strict_provenance_and_degenerate_fallback():
    runtime, config = selection()
    payload = dict(config=config, source_aware_selection_state=runtime.state_dict())
    assert checkpoint_source_selection(payload, config).state_dict() == runtime.state_dict()
    changed = deepcopy(config)
    changed["source_aware_selection"]["appearance_weight"] = .5
    with pytest.raises(ValueError, match="disagree"):
        checkpoint_source_selection(payload, changed)
    bad = deepcopy(payload)
    bad["source_aware_selection_state"]["calibration"]["split"] = "val"
    with pytest.raises(ValueError, match="train-only"):
        checkpoint_source_selection(bad, config)
    with pytest.raises(ValueError, match="Missing"):
        checkpoint_source_selection(dict(config=config), config)
    for scale in runtime.calibration["scales"].values():
        scale.update(spatial=0., appearance=0.)
    logp = torch.zeros(1, 4).log_softmax(1)
    actual, metrics = runtime.apply(logp, dict(spatial=torch.ones(1, 4), appearance=torch.ones(1, 4)),
                                    direction="cat_to_dog", target_codes=torch.eye(4))
    assert actual is logp and all(metrics["calibration_fallback"].values())


def test_texture_diagnostic_is_deterministic_has_real_baseline_and_no_graph():
    generated = torch.rand(2, 3, 8, 8, generator=rng(), requires_grad=True)
    records = [dict(sample_id=f"dog_{i}") for i in range(6)]
    images = torch.rand(6, 3, 8, 8, generator=rng(3), requires_grad=True)
    load = lambda rows: torch.stack([images[int(r["sample_id"].split("_")[1])] for r in rows])
    options = dict(sizes=[8, 4], patch_size=3, patches_per_image=4, directions=5, scale_floor=.01)
    state = torch.random.get_rng_state().clone()
    result = texture_diagnostic(generated, records, load, options, seed=53)
    assert result == texture_diagnostic(generated, records, load, options, seed=53)
    assert not set(result["reference_ids"]) & set(result["baseline_ids"])
    assert result["training_weight"] == 0 and result["real_to_real_distance"] > 0
    assert isinstance(result["distance"], float) and generated.grad is None and images.grad is None
    assert torch.equal(state, torch.random.get_rng_state())
