"""Learned samplers: exact CUT forward/gradient parity and detached-key routing."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_ot.losses.patchnce import patchnce_loss, PATCHNCE_MLP_PROTOCOL
from diffusion_ot.models.patch_sampler import PatchSampleMLP, load_patch_projector
from diffusion_ot.third_party.cut.patchnce import PatchNCELoss
import diffusion_ot.third_party.cut.patchnce as upstream_module


def official_sampler(channels, ours):
    # Execute the pinned, verbatim PatchSampleF class with known channel shapes.
    # Initialization is bypassed because the exact same weights are then loaded.
    scope = {"torch": torch, "nn": torch.nn, "np": np,
             "init_net": lambda net, *args, **kwargs: net}
    path = Path(upstream_module.__file__).with_name("sampling_reference.txt")
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
    sampler = scope["PatchSampleF"](use_mlp=True, nc=ours.projection_dim)
    sampler.create_mlp([torch.empty(1, c, 2, 2) for c in channels])
    sampler.load_state_dict(deepcopy(ours.state_dict()), strict=True)
    return sampler


@pytest.mark.parametrize("batch", [1, 3])
def test_mlp_sampling_loss_and_live_gradients_match_official_cut(batch):
    rng = torch.Generator().manual_seed(17)
    channels = (5, 9)
    query = [torch.randn(batch, c, 3, 4, generator=rng, requires_grad=True) for c in channels]
    key = [torch.randn(batch, c, 3, 4, generator=rng, requires_grad=True) for c in channels]
    query_head, key_head = (PatchSampleMLP(channels, 16, seed=i) for i in (12, 23))
    loss, metrics = patchnce_loss(query, key, num_patches=7, generator=rng,
                                query_projector=query_head, key_projector=key_head)
    ids = [np.array(layer["patch_ids"]) for layer in metrics["layers"]]
    official_query = [q.detach().clone().requires_grad_() for q in query]
    official_key = [k.detach().clone().requires_grad_() for k in key]
    q_sampler = official_sampler(channels, query_head)
    k_sampler = official_sampler(channels, key_head)
    q_pool, _ = q_sampler(official_query, num_patches=7, patch_ids=ids)
    k_pool, _ = k_sampler(official_key, num_patches=7, patch_ids=ids)
    criterion = PatchNCELoss(SimpleNamespace(batch_size=batch, nce_T=.2,
                                            nce_includes_all_negatives_from_minibatch=False))
    expected = torch.stack([criterion(q, k).mean() for q, k in zip(q_pool, k_pool)]).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=2e-7)
    loss.backward()
    expected.backward()
    for actual, reference in zip(query + list(query_head.parameters()),
                                 official_query + list(q_sampler.parameters())):
        torch.testing.assert_close(actual.grad, reference.grad, atol=1e-6, rtol=2e-5)
        assert actual.grad.norm() > 0
    assert all(p.grad is None for p in key + list(key_head.parameters()))
    assert all(p.grad is None for p in official_key + list(k_sampler.parameters()))
    assert metrics["protocol"] == PATCHNCE_MLP_PROTOCOL
    assert metrics["projector"] == "domain_mlp"
    assert all(layer["embedding_dim"] == 16 for layer in metrics["layers"])
    assert [layer["feature_shape"][1] for layer in metrics["layers"]] == list(channels)


def test_mlp_initialization_is_eager_seeded_and_preserves_global_rng():
    before = torch.get_rng_state().clone()
    head = PatchSampleMLP((16, 32, 64), 256, seed=9)
    other = PatchSampleMLP((16, 32, 64), 256, seed=9)
    assert torch.equal(torch.get_rng_state(), before)
    assert len(list(head.parameters())) == 12
    for name, parameter in head.named_parameters():
        torch.testing.assert_close(parameter, other.state_dict()[name], atol=0, rtol=0)
        if name.endswith("bias"):
            assert parameter.count_nonzero() == 0
        else:
            assert parameter.std().item() == pytest.approx(.02, rel=.04)
    parameters = list(head.parameters())
    head(torch.ones(5, 16), 0).sum().backward()
    assert [id(p) for p in head.parameters()] == [id(p) for p in parameters]


def test_mlp_zero_maps_and_autocast_have_finite_gradients_and_honest_retrieval():
    head = PatchSampleMLP((8,), 12, seed=42)
    key_head = deepcopy(head)
    query = torch.zeros(2, 8, 2, 2, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, metrics = patchnce_loss([query], [query.detach()], generator=torch.Generator(),
                                    query_projector=head, key_projector=key_head)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(query.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())
    assert metrics["query_zero_norm_fraction"] == 1
    assert metrics["retrieval_top1"] == metrics["retrieval_chance"] == .25
    assert all(p.grad is None for p in key_head.parameters())


def test_mlp_evaluation_loads_selected_weights_and_rejects_missing_or_wrong_state():
    encoder = SimpleNamespace(spatial_feature_channels=(8,))
    options = {"sampler": "mlp_sample", "projection_dim": 12, "layers": [0]}
    raw, ema = (PatchSampleMLP((8,), 12, seed=i) for i in (1, 2))
    checkpoint = {"patch_projectors": {"cat": raw.state_dict()}, "patch_projector_ema": {"cat": ema.state_dict()}}
    for weights, expected in (("raw", raw), ("ema", ema)):
        actual = load_patch_projector(encoder, options, checkpoint, "cat", weights=weights, device="cpu")
        assert not actual.training
        for name, parameter in actual.state_dict().items():
            torch.testing.assert_close(parameter, expected.state_dict()[name], rtol=0, atol=0)
        with pytest.raises(ValueError, match="Checkpoint has no"):
            load_patch_projector(encoder, options, {}, "cat", weights=weights, device="cpu")
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_patch_projector(encoder, {**options, "projection_dim": 13}, checkpoint, "cat", weights="raw", device="cpu")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_mlp_source_and_query_projectors_can_live_on_separate_gpus():
    query = torch.randn(2, 8, 3, 3, device="cuda:1", requires_grad=True)
    key = torch.randn(2, 8, 3, 3, device="cuda:0", requires_grad=True)
    q_head = PatchSampleMLP((8,), 16).to("cuda:1")
    k_head = PatchSampleMLP((8,), 16).to("cuda:0")
    loss, _ = patchnce_loss([query], [key], generator=torch.Generator(),
                           query_projector=q_head, key_projector=k_head)
    loss.backward()
    assert query.grad.norm() > 0 and key.grad is None
    assert all(p.grad is not None for p in q_head.parameters())
    assert all(p.grad is None for p in k_head.parameters())
