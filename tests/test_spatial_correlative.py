from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from diffusion_ot.losses.spatial_correlative import (
    SpatialCorrelativeCost, checkpoint_spatial_cost, spatial_correlative_options,
    spatial_descriptors, spatial_correlative_cost, centered_cost_rms,
    descriptor_diagnostics, conditional_spatial_diagnostics, encode_spatial_descriptors,
)


def options(**changes):
    return spatial_correlative_options({"spatial_correlative_cost": dict(enabled=True, layers=[0], grid_size=2, **changes)})


def sample():
    generator = torch.Generator().manual_seed(9)
    matching = {d: F.normalize(torch.randn(n, 7, generator=generator), dim=-1) for d, n in (("cat", 6), ("dog", 5))}
    desc = {d: spatial_descriptors([torch.randn(len(m), 8, 4, 4, generator=generator)], grid_size=2)
            for d, m in matching.items()}
    ids = {d: [f"{d}_{i}" for i in range(len(m))] for d, m in matching.items()}
    return matching, desc, ids


def test_dense_patchsim_construction_then_our_symmetric_reduction():
    # Official PatchSim(normalize=True, patch_size=-1): center, normalize
    # channel vectors, and correlate all ordered locations. No source copied.
    torch.manual_seed(7)
    maps = torch.randn(3, 5, 4, 4, requires_grad=True)
    pooled = F.adaptive_avg_pool2d(maps, 2)
    normalized = F.normalize(pooled - pooled.mean((2, 3), keepdim=True), dim=1)
    dense = torch.einsum("bcp,bcq->bpq", normalized.flatten(2), normalized.flatten(2))
    dense = dense * (1 - torch.eye(4))
    expected = F.normalize(dense, dim=-1).unsqueeze(1).detach()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = spatial_descriptors([maps], grid_size=2)
        cost = spatial_correlative_cost(actual, actual)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == cost.dtype == torch.float32
    assert not actual.requires_grad and not cost.requires_grad
    assert torch.count_nonzero(actual.diagonal(dim1=-2, dim2=-1)) == 0
    torch.testing.assert_close(cost.diag(), torch.zeros(3), atol=2e-7, rtol=0)


def test_orthogonal_channel_bases_and_spatial_order():
    torch.manual_seed(18)
    x = torch.randn(4, 5, 4, 4)
    q = torch.linalg.qr(torch.randn(5, 5)).Q
    rotated = torch.einsum("dc,bchw->bdhw", q, x)
    a, b = (spatial_descriptors([v], grid_size=2) for v in (x, rotated))
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
    # Equal relation geometry also works with different domain channel counts.
    padded = spatial_descriptors([torch.cat((x, torch.zeros(4, 2, 4, 4)), dim=1)], grid_size=2)
    torch.testing.assert_close(a, padded)
    permutation = torch.tensor([2, 0, 3, 1])
    changed = b[:, :, permutation][:, :, :, permutation]
    ab = spatial_correlative_cost(a, changed)
    torch.testing.assert_close(ab, spatial_correlative_cost(changed, a).T)
    assert ab.diag().mean() > .05  # Cannot discard the ordered spatial grid.
    explicit = 1 - torch.einsum("ilpq,jlpq->ij", a, changed) / 4
    torch.testing.assert_close(ab, explicit)


def test_degenerate_maps_are_finite_neutral_matches_and_fallback():
    zero = spatial_descriptors([torch.ones(6, 3, 4, 4)], grid_size=2)
    assert torch.count_nonzero(zero) == 0
    assert descriptor_diagnostics(zero)["near_zero_row_fraction"] == 1
    torch.testing.assert_close(spatial_correlative_cost(zero, zero), torch.ones(6, 6))
    matching, _, ids = sample()
    descriptors = dict(cat=zero, dog=zero[:5])
    runtime = SpatialCorrelativeCost(options())
    with pytest.warns(RuntimeWarning, match="falls back"):
        runtime.calibrate(matching, descriptors, ids)
    cost = runtime.costs(matching, descriptors)
    assert torch.equal(cost["encoder"], cost["mixed"])
    assert runtime.calibration["status"] == "fallback_encoder"
    with pytest.raises(FloatingPointError):
        spatial_descriptors([torch.full((1, 2, 2, 2), float("nan"))], grid_size=2)
    with pytest.raises(ValueError, match="at least"):
        spatial_descriptors([torch.ones(2, 3, 2, 2)], grid_size=8)


def test_fixed_calibration_centering_and_checkpoint_protocol():
    matching, descriptors, ids = sample()
    runtime = SpatialCorrelativeCost(options())
    runtime.calibrate(matching, descriptors, ids)
    cost = runtime.costs(matching, descriptors)
    assert centered_cost_rms(cost["encoder"]) == pytest.approx(centered_cost_rms(cost["mixed"]), rel=1e-6)
    offsets = torch.arange(6.)[:, None] + torch.arange(5.)[None, :]
    assert centered_cost_rms(cost["encoder"]) == pytest.approx(centered_cost_rms(cost["encoder"] + offsets), rel=1e-6)
    state = runtime.state_dict()
    runtime.costs({d: x[:3] for d, x in matching.items()}, {d: x[:3] for d, x in descriptors.items()})
    assert runtime.state_dict() == state  # No per-batch or query recalibration.
    with pytest.raises(ValueError, match="frozen"):
        runtime.calibrate(matching, descriptors, ids)
    config = {"spatial_correlative_cost": options()}
    payload = dict(config=config, spatial_correlative_cost_state=state)
    restored = checkpoint_spatial_cost(payload, config)
    torch.testing.assert_close(restored.costs(matching, descriptors)["mixed"], cost["mixed"], rtol=0, atol=0)
    for bad in ({}, {"spatial_correlative_cost": options(mixing_weight=.1)}):
        with pytest.raises(ValueError, match="disagree"):
            checkpoint_spatial_cost(payload, bad)
    with pytest.raises(ValueError, match="missing"):
        checkpoint_spatial_cost(dict(config=config), config)
    malformed = deepcopy(payload)
    malformed["spatial_correlative_cost_state"]["calibration"]["spatial_scale"] = float("inf")
    with pytest.raises(ValueError, match="Invalid"):
        checkpoint_spatial_cost(malformed, config)


def test_drift_and_conditional_diagnostics_do_not_change_plan_or_query_weights():
    matching, descriptors, ids = sample()
    runtime = SpatialCorrelativeCost(options())
    runtime.calibrate(matching, descriptors, ids)
    assert runtime.probe_drift(descriptors, ids) == dict(cat=0., dog=0.)
    saved = runtime.state_dict()
    restored = SpatialCorrelativeCost(options())
    restored.load_state_dict(saved)
    changed = {d: v * .5 for d, v in descriptors.items()}
    assert all(v > 0 for v in restored.probe_drift(changed, ids).values())
    weights = torch.softmax(torch.randn(2, 5), -1)
    before = weights.clone()
    metrics = conditional_spatial_diagnostics(descriptors["cat"][:2], descriptors["dog"], weights)
    assert 0 <= metrics["expected_cost"] <= 2
    assert torch.equal(weights, before)


def test_alpha_zero_is_disabled_and_encoder_modes_preserved():
    assert options(mixing_weight=0) is None
    from diffusion_ot.models.pdae_sit import PDAELatentEncoder
    model = PDAELatentEncoder(channels=(8,), z_dim=6, spatial_size=2, num_groups=2).train()
    latents = torch.randn(3, 4, 8, 8)
    rng = torch.get_rng_state().clone()
    result = encode_spatial_descriptors(model, latents, options(), device="cpu", dtype=torch.float32, batch_size=2)
    assert result.shape == (3, 1, 4, 4) and model.training
    assert not result.requires_grad
    assert torch.equal(rng, torch.get_rng_state())
