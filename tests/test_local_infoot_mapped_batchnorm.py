from copy import deepcopy

import pytest
import torch

from test_local_infoot_pipeline import local_helpers
from test_local_infoot_distributed import training_state, latent_batch, weighted_loss
from test_local_infoot_resume import assert_same


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    losses, _ = model(latent_batch())
    weighted_loss(losses, model.settings).backward()
    optimizer.step()


@pytest.mark.parametrize("enabled", [False, True])
def test_mapped_batchnorm_checkpoint_resume_and_single_query_eval(local_helpers, tmp_path, enabled):
    from infoot_helper.cotraining_checkpoint import resume_latest
    from infoot_helper.distributed.checkpoint import save_checkpoint
    from infoot_helper.batchnorm_matching import load_batchnorm

    model, optimizer = training_state(mapped_batchnorm=enabled)
    update(model, optimizer)
    save_checkpoint(tmp_path, 1, model.domains, model.batch_norms, optimizer, model.settings,
                    torch.device("cpu"), mapped_batch_norms=model.mapped_batch_norms)
    random = torch.rand(5)
    restored, restored_optimizer = training_state(mapped_batchnorm=enabled)
    assert resume_latest(tmp_path, restored.domains, restored.batch_norms, restored_optimizer,
                         mapped_batch_norms=restored.mapped_batch_norms) == 1
    torch.testing.assert_close(torch.rand(5), random, rtol=0, atol=0)
    assert_same(restored.state_dict(), model.state_dict())
    assert_same(restored_optimizer.state_dict(), optimizer.state_dict())
    rng = torch.get_rng_state()
    update(model, optimizer)
    torch.set_rng_state(rng)
    update(restored, restored_optimizer)
    assert_same(restored.state_dict(), model.state_dict())
    assert_same(restored_optimizer.state_dict(), optimizer.state_dict())
    for name in model.domains:
        norm = load_batchnorm(tmp_path / f"{name}_step_000001.pt", domain=name, step=1,
                              device="cpu", key="mapped_batchnorm")
        assert not norm.training
        if enabled:
            assert isinstance(norm, torch.nn.BatchNorm1d) and not norm.affine
        else:
            assert isinstance(norm, torch.nn.Identity)
        before = deepcopy(norm.state_dict())
        query = torch.randn(5, 6)
        torch.testing.assert_close(norm(query[:1]), norm(query)[:1])
        torch.testing.assert_close(norm(query.flip(0)), norm(query).flip(0))
        assert_same(norm.state_dict(), before)


@pytest.mark.parametrize("saved_enabled,current_enabled", [(False, True), (True, False)])
def test_resume_can_enable_or_disable_mapped_batchnorm(local_helpers, tmp_path, saved_enabled, current_enabled):
    from infoot_helper.cotraining_checkpoint import resume_latest
    from infoot_helper.distributed.checkpoint import save_checkpoint

    model, optimizer = training_state(mapped_batchnorm=saved_enabled)
    update(model, optimizer)
    save_checkpoint(tmp_path, 1, model.domains, model.batch_norms, optimizer, model.settings,
                    torch.device("cpu"), mapped_batch_norms=model.mapped_batch_norms)
    restored, restored_optimizer = training_state(mapped_batchnorm=current_enabled)
    resume_latest(tmp_path, restored.domains, restored.batch_norms, restored_optimizer,
                  mapped_batch_norms=restored.mapped_batch_norms)
    assert bool(restored.mapped_batch_norms) == current_enabled
    for norm in restored.mapped_batch_norms.values():
        assert norm.num_batches_tracked == 0
    update(restored, restored_optimizer)
    for norm in restored.mapped_batch_norms.values():
        assert norm.num_batches_tracked == 1


def test_mapped_batchnorm_rejects_missing_or_untrained_state_and_singleton_training(local_helpers, tmp_path):
    from infoot_helper.batchnorm_matching import batchnorm_checkpoint, load_batchnorm
    from infoot_helper.batchnorm_matching.mapped import make_mapped_batchnorms, restore_mapped_batchnorms

    model, _ = training_state(mapped_batchnorm=True)
    with pytest.raises(ValueError, match="query_count"):
        make_mapped_batchnorms(model.batch_norms, dict(mapped_batchnorm=True, query_count=1))
    with pytest.raises(ValueError, match="missing"):
        restore_mapped_batchnorms(model.mapped_batch_norms, {"settings": {"mapped_batchnorm": True}})
    path = tmp_path / "dog.pt"
    torch.save(dict(domain="dog", step=1, mapped_batchnorm_enabled=True), path)
    with pytest.raises(ValueError, match="do not match"):
        load_batchnorm(path, domain="dog", step=1, device="cpu", key="mapped_batchnorm")
    torch.save(dict(domain="dog", step=1, mapped_batchnorm_enabled=True,
                    mapped_batchnorm=batchnorm_checkpoint(model.mapped_batch_norms["dog"])), path)
    with pytest.raises(ValueError, match="not observed"):
        load_batchnorm(path, domain="dog", step=1, device="cpu", key="mapped_batchnorm")
    torch.save(dict(domain="dog", step=1), path)
    assert isinstance(load_batchnorm(path, domain="dog", step=1, device="cpu", key="mapped_batchnorm"),
                      torch.nn.Identity)


def test_ddp_sync_averages_matching_and_mapped_statistics(local_helpers, monkeypatch):
    from infoot_helper.distributed import runtime

    model, _ = training_state(mapped_batchnorm=True)
    groups = (model.batch_norms, model.mapped_batch_norms)
    buffers = [buffer for group in groups for norm in group.values()
               for buffer in (norm.running_mean, norm.running_var)]
    peers = [torch.full_like(buffer, index + 2.) for index, buffer in enumerate(buffers)]
    expected = [(buffer.clone() + peer) / 2 for buffer, peer in zip(buffers, peers)]
    seen = []

    def all_reduce(buffer):
        index = len(seen)
        assert buffer is buffers[index]
        buffer.add_(peers[index])
        seen.append(buffer)

    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(runtime.dist, "all_reduce", all_reduce)
    runtime.sync_batchnorm_buffers(*groups)
    assert len(seen) == 8
    for actual, target in zip(buffers, expected):
        torch.testing.assert_close(actual, target)
