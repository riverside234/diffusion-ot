"""Sparse saved factors must project only groups retained by image routing."""
from copy import deepcopy

import pytest
import torch

from test_infoot_vit_mapping import bank, cpu_threads
from infoot_vit.infoot_helper.conditional import BalancedModel
from infoot_vit.infoot_helper.feature_bank import file_hash, save_tensor
from infoot_vit.infoot_helper.mapping import select_images
from infoot_vit.infoot_helper.storage import store_plan_state
from infoot_vit.lowrank import kernels
from infoot_vit.lowrank.config import canonical
from infoot_vit.lowrank.experiment import validate_kernels
from infoot_vit.lowrank.mapping import LowRankMapper
from infoot_vit.lowrank.storage import VERSION, pack_factors, validate_factors


@pytest.fixture
def sparse_mapper(tmp_path):
    """Load valid float32 states with underflowed off-block transport entries.

    The actual feature builder produces disjoint nonnegative kernel features
    with a narrow bandwidth. All target KDE densities remain strictly positive.
    """
    x = torch.zeros(2, 4, 3, dtype=torch.float64)
    x[0, :, 0], x[1, :, 0] = 1., -1.
    source = bank(tmp_path, "source", x, "cat")
    target = bank(tmp_path, "target", x, "dog")
    config = canonical(dict(device="cpu", transport_rank=2, kernel_rank=2,
        sampling=dict(images_per_domain=2), kernel=dict(h=.001),
        image_solver=dict(lam=0., reg=.4, h=.7)))
    q = torch.full((8, 2), 1e-60, dtype=torch.float64)
    q[:4, 0], q[4:, 1] = 1/8, 1/8
    g = torch.full((2,), .5, dtype=torch.float64)
    factors = pack_factors((q, q.clone(), g), config["optimizer"],
        fit_fingerprint="sparse-regression", solver_report=dict(status="converged_sampled_objective"))
    assert factors["quantization"]["q"]["underflow_entries"] == 8
    validate_factors(factors, config["optimizer"], (8, 8, 2))
    omega = torch.tensor([[1., -1.], [0., 0.], [0., 0.]], dtype=torch.float64)
    fx, sx = kernels.fit_features(x.reshape(-1, 3), 2, .001, 42, omega=omega)
    fy, sy = kernels.fit_features(x.reshape(-1, 3), 2, .001, 42, omega=omega)
    state = dict(fx=fx.float(), fy=fy.float(), source=sx, target=sy, storage_version=VERSION)
    validate_kernels(state, (8, 8, 2), 3)
    assert (state["fx"] == 0).any()
    image = BalancedModel.fit(x.reshape(2, -1), x.reshape(2, -1), config["image_solver"])
    manifest = dict(config=config, fit_fingerprint="sparse-regression", files={})
    for name, value in (("image", store_plan_state(image.state)), ("kernels", state), ("factors", factors)):
        path = tmp_path / f"{name}.pt"
        save_tensor(path, value)
        manifest["files"][name] = dict(file=path.name, sha256=file_hash(path))
    mapper = LowRankMapper(tmp_path, manifest, source, target, device="cpu")
    assert (mapper.density_y > 0).all()
    query = x[:1]
    return mapper, query, ["query-positive"], torch.ones((1, 4), dtype=torch.bool)


@pytest.mark.parametrize("selection,top_k", [("argmax", None), ("sample", None), ("mean", 1)])
def test_sparse_routing_skips_zero_score_excluded_targets_and_is_chunk_invariant(sparse_mapper, selection, top_k):
    mapper, query, ids, mask = sparse_mapper
    settings = mapper.config["projection"]
    settings.update(selection=selection, top_k_images=top_k)
    alpha, _ = select_images(mapper.image.conditional_weights(query.reshape(1, -1))[0], mapper.target.ids, ids[0], settings)
    torch.testing.assert_close(alpha, torch.tensor([1., 0.], dtype=torch.float64), atol=0, rtol=0)
    results = []
    for chunk_size in (1, 2, 8):
        settings["target_chunk_size"] = chunk_size
        results.append(mapper.map_features(query, ids, valid_mask=mask, return_metadata=True))
    for result in results:
        torch.testing.assert_close(result.mapped_features, mapper.y[:1], atol=0, rtol=0)
        torch.testing.assert_close(result.match_confidence, torch.ones_like(mask, dtype=query.dtype))
        assert result.valid_mask.all()
        assert result.diagnostics["queries"] == results[0].diagnostics["queries"]
        record = result.diagnostics["queries"][0]
        assert record["within_image_effective_patches"] == pytest.approx(4.)
        assert record["group_mass_residual"] == 0


@pytest.mark.parametrize("selection,top_k", [("argmax", None), ("sample", None), ("mean", 1)])
@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_selected_zero_score_target_still_raises(sparse_mapper, selection, top_k, chunk_size):
    mapper, query, ids, mask = sparse_mapper
    mapper.config["projection"].update(selection=selection, top_k_images=top_k, target_chunk_size=chunk_size)
    # Reverse a valid balanced saved coupling: router still selects dog_0, but
    # its conditional patch mass is now zero for this positive query.
    path = mapper.directory / "factors.pt"
    state = torch.load(path, weights_only=True)
    state["r"] = state["r"].flip(1)
    state["r_rows"], state["r_columns"] = state["r"].double().sum(1), state["r"].double().sum(0)
    validate_factors(state, mapper.config["optimizer"], (8, 8, 2))
    save_tensor(path, state)
    manifest = deepcopy(mapper.manifest)
    manifest["files"]["factors"]["sha256"] = file_hash(path)
    mapper = LowRankMapper(mapper.directory, manifest, mapper.source, mapper.target, device="cpu")
    with pytest.raises(ValueError, match="Zero-mass balanced conditional row"):
        mapper.map_features(query, ids, valid_mask=mask)


def test_mean_routing_checks_every_positive_weight_group(sparse_mapper):
    mapper, query, ids, mask = sparse_mapper
    mapper.config["projection"].update(selection="mean", top_k_images=None, target_chunk_size=2)
    assert (mapper.image.conditional_weights(query.reshape(1, -1)) > 0).all()
    with pytest.raises(ValueError, match="Zero-mass balanced conditional row"):
        mapper.map_features(query, ids, valid_mask=mask)
