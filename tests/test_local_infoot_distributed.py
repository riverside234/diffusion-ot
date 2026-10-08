from copy import deepcopy
from datetime import timedelta
import logging
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_local_infoot_pipeline import local_helpers
from test_local_infoot_resume import assert_same
from test_self_supervised_translation import domains as tiny_domains


def training_state(affine=True):
    from diffusion_ot.models.generator_adaptation import configure_generator_adaptation
    from infoot_helper.distributed.model import CoTrainingStep

    torch.manual_seed(42)
    domains, norms, groups = tiny_domains(), torch.nn.ModuleDict(), []
    settings = dict(query_count=2, encode_batch_size=2, flow_batch_size=2,
                    fit_h=.8, projection_h=.8, mi_weight=.1, reg=.5,
                    fit_iterations=2, sampling_steps=2, flow_weight=1.,
                    infoOT_loss_weight=.1, contrastive_weight=.05, covariance_weight=.3)
    for name, context in domains.items():
        context.model_dtype = torch.float32
        context.training_config["class_conditioning"] = {"null_label": None}
        context.vae.eval().requires_grad_(False)
        view = configure_generator_adaptation(context.branch)
        norms[name] = torch.nn.BatchNorm1d(6, affine=affine)
        groups.extend([{"params": context.branch.encoder.parameters(), "lr": .001},
                       {"params": view.parameters(), "lr": .002}])
        if affine:
            groups.append({"params": norms[name].parameters(), "lr": .003})
    model = CoTrainingStep(domains, norms, settings)
    return model, torch.optim.Adam(groups)


def latent_batch(rank=0, step=0):
    generator = torch.Generator().manual_seed(100 + rank + 10 * step)
    return {name: torch.randn(6, 4, 8, 8, generator=generator) for name in ("cat", "dog")}


def test_single_process_loss_backward(local_helpers):
    model, optimizer = training_state()
    loss, metrics = model(latent_batch(), report=True)
    loss.backward()
    assert torch.isfinite(loss) and metrics["total"] == loss.item()
    for name, context in model.domains.items():
        assert any(p.grad is not None and p.grad.norm() > 0 for p in context.branch.encoder.parameters())
        assert model.batch_norms[name].weight.grad.norm() > 0
        assert all(p.grad is None for p in context.vae.parameters())
    optimizer.step()


def distributed_worker(rank, world_size, directory):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infoot"))
    from infoot_helper.batchnorm_matching import load_batchnorm
    from infoot_helper.cotraining_checkpoint import resume_latest
    from infoot_helper.distributed.checkpoint import save_checkpoint, restore_random_state
    from infoot_helper.distributed.runtime import (
        wrap_distributed, sync_batchnorm_buffers, mean_metrics, setup_logging, close_distributed,
    )

    torch.set_num_threads(1)
    directory, device = Path(directory), torch.device("cpu")
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(),
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=120))
    try:
        setup_logging(directory, rank)
        logging.info("writer_rank=%s", rank)
        model, optimizer = training_state(affine=world_size == 2)
        frozen = {name: value.clone() for name, value in model.named_parameters() if not value.requires_grad}
        ddp = wrap_distributed(model, device)
        torch.manual_seed(42 + rank)
        for step in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            expected = deepcopy(model)
            expected.zero_grad(set_to_none=True)
            rng = torch.get_rng_state()
            local_loss, local_metrics = expected(latent_batch(rank, step), report=True)
            local_loss.backward()
            torch.set_rng_state(rng)
            loss, metrics = ddp(latent_batch(rank, step), report=step == 1)
            torch.testing.assert_close(loss, local_loss)
            loss.backward()
            for actual, reference in zip(model.parameters(), expected.parameters()):
                if not reference.requires_grad:
                    assert actual.grad is None
                    continue
                gradient = reference.grad if reference.grad is not None else torch.zeros_like(reference)
                dist.all_reduce(gradient)
                gradient.div_(world_size)
                torch.testing.assert_close(actual.grad if actual.grad is not None else torch.zeros_like(actual),
                                           gradient, rtol=2e-4, atol=2e-6)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            sync_batchnorm_buffers(model.batch_norms)
            for norm, reference in zip(model.batch_norms.values(), expected.batch_norms.values()):
                for key in ("running_mean", "running_var"):
                    value = getattr(reference, key)
                    dist.all_reduce(value)
                    torch.testing.assert_close(getattr(norm, key), value / world_size)
            flat = torch.cat([p.detach().flatten() for p in model.parameters()])
            replicas = [torch.empty_like(flat) for _ in range(world_size)]
            dist.all_gather(replicas, flat)
            for replica in replicas:
                torch.testing.assert_close(replica, flat, rtol=0, atol=0)
            if step == 1:
                averaged = mean_metrics(metrics, device)
                expected_metrics = mean_metrics(local_metrics, device)
                assert averaged == pytest.approx(expected_metrics)
        for name, value in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)

        save_checkpoint(directory, 2, model.domains, model.batch_norms, optimizer, model.settings, device)
        expected_random = torch.rand(5)
        restored, restored_optimizer = training_state(affine=world_size == 2)
        assert resume_latest(directory, restored.domains, restored.batch_norms, restored_optimizer, device) == 2
        torch.testing.assert_close(torch.rand(5), expected_random, rtol=0, atol=0)
        assert_same(restored.state_dict(), model.state_dict())
        assert_same(restored_optimizer.state_dict(), optimizer.state_dict())
        for name in model.domains:
            norm = load_batchnorm(directory / f"{name}_step_000002.pt", domain=name, step=2, device=device)
            query = torch.randn(5, 6)
            torch.testing.assert_close(norm(query[:1]), norm(query)[:1])
            assert_same(norm.state_dict(), restored.batch_norms[name].state_dict())
        saved = torch.load(directory / "step_000002.pt", weights_only=True)
        assert saved["world_size"] == world_size and len(saved["rank_rng_states"]) == world_size
        assert not torch.equal(saved["rank_rng_states"][0]["cpu"], saved["rank_rng_states"][1]["cpu"])
        saved["rank_rng_states"] = saved["rank_rng_states"][:1]
        restore_random_state(saved, device)
        sample = torch.rand(5)
        generator = torch.Generator().manual_seed(42 + 2 * world_size + rank)
        torch.testing.assert_close(sample, torch.rand(5, generator=generator))
    finally:
        close_distributed()
        logging.shutdown()


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_losses_gradients_and_resume(local_helpers, tmp_path, world_size):
    mp.spawn(distributed_worker, args=(world_size, str(tmp_path)), nprocs=world_size, join=True)
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == [
        "cat_step_000002.pt", "dog_step_000002.pt", "step_000002.pt",
    ]
    text = (tmp_path / "train.log").read_text(encoding="utf-8")
    assert text.count("writer_rank=") == 1 and "writer_rank=0" in text
    assert text.count("Saved checkpoint") == 1


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_sharded_data_resume_and_rng(local_helpers, world_size):
    from infoot_helper.distributed.data import make_loader, cycle_batches

    records, batch_size = list(range(64)), 4
    epoch_samples = []
    for rank in range(world_size):
        loader = make_loader(records, batch_size, rank, world_size, workers=0)
        rng = torch.get_rng_state().clone()
        batches = cycle_batches(loader)
        first_epoch = [next(batches) for _ in range(len(loader))]
        second_epoch = [next(batches) for _ in range(len(loader))]
        assert first_epoch != second_epoch
        epoch_samples.append({item for batch in first_epoch for item in batch})
        resumed = cycle_batches(make_loader(records, batch_size, rank, world_size, workers=0), len(loader) + 1)
        assert next(resumed) == second_epoch[1]
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert set.union(*epoch_samples) == set(records)
    assert sum(map(len, epoch_samples)) == len(records)
