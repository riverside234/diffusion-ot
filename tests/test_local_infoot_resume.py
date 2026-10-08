from pathlib import Path
from types import SimpleNamespace
import runpy
import sys

import pytest
import torch
from torch.utils._pytree import tree_flatten

from test_local_infoot_pipeline import local_helpers
from test_pdae_sit_shapes import FakeSiT
from test_residual_encoder import tiny_config, small_cpu_thread_pool


def assert_same(actual, expected):
    actual_values, actual_structure = tree_flatten(actual)
    expected_values, expected_structure = tree_flatten(expected)
    assert actual_structure == expected_structure
    for actual_value, expected_value in zip(actual_values, expected_values):
        if isinstance(actual_value, torch.Tensor):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        else:
            assert actual_value == expected_value


def test_latest_checkpoint_uses_numeric_step_and_ignores_other_files(local_helpers, tmp_path):
    from infoot_helper.cotraining_checkpoint import latest_checkpoint, resume_latest

    assert latest_checkpoint(tmp_path) is None
    assert resume_latest(tmp_path, {}, {}, None) == 0
    for name in ("step_999999.pt", "step_1000000.pt", "cat_step_2000000.pt",
                 "dog_step_2000000.pt", "cat_to_dog_step_2000000_plan.pt",
                 "step_latest.pt", "step_3000000.tmp"):
        (tmp_path / name).touch()
    (tmp_path / "step_4000000.pt").mkdir()
    assert latest_checkpoint(tmp_path).name == "step_1000000.pt"


def training_state(affine):
    from diffusion_ot.models.pdae_sit import build_pdae_sit_branch
    from diffusion_ot.models.generator_adaptation import configure_generator_adaptation

    domains, norms, groups = {}, torch.nn.ModuleDict(), []
    for name in ("cat", "dog"):
        config = tiny_config()
        branch = build_pdae_sit_branch(FakeSiT(), stage_config=config)
        generator = configure_generator_adaptation(branch)
        domains[name] = SimpleNamespace(branch=branch, training_config=config)
        norms[name] = torch.nn.BatchNorm1d(12, affine=affine)
        if affine:
            groups.append({"params": norms[name].parameters(), "lr": .03})
        groups.extend([{"params": branch.encoder.parameters(), "lr": .01},
                       {"params": generator.parameters(), "lr": .02}])
    return domains, norms, torch.optim.Adam(groups)


def advance(norms, optimizer):
    optimizer.zero_grad(set_to_none=True)
    features = torch.arange(48, dtype=torch.float32).reshape(4, 12) / 10
    loss = sum((norm(features) - .3).square().mean() for norm in norms.values())
    loss = loss + sum((parameter - .2).square().mean()
                      for group in optimizer.param_groups for parameter in group["params"])
    loss.backward()
    optimizer.step()


@pytest.mark.parametrize("affine", [False, True])
def test_resume_restores_models_batchnorm_adam_rng_and_next_update(local_helpers, tmp_path, affine):
    from diffusion_ot.training.train_joint_infoot import _save_checkpoint
    from infoot_helper.batchnorm_matching import batchnorm_checkpoint
    from infoot_helper.cotraining_checkpoint import resume_latest

    torch.manual_seed(44)
    domains, norms, optimizer = training_state(affine)
    advance(norms, optimizer)
    _save_checkpoint(tmp_path / "step_000200.pt", {
        "step": 200,
        "models": {name: domain.branch.pdae_state_dict() for name, domain in domains.items()},
        "matching_batchnorm": {name: batchnorm_checkpoint(norm) for name, norm in norms.items()},
        "optimizer": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": [],
    })
    expected_random = torch.rand(5)
    restored, restored_norms, restored_optimizer = training_state(affine)
    restored_norms.eval()
    step = resume_latest(tmp_path, restored, restored_norms, restored_optimizer)
    assert step == 200 and list(range(step + 1, 203)) == [201, 202]
    torch.testing.assert_close(torch.rand(5), expected_random, rtol=0, atol=0)
    for name in domains:
        assert_same(restored[name].branch.pdae_state_dict(), domains[name].branch.pdae_state_dict())
        torch.testing.assert_close(restored_norms[name].state_dict(), norms[name].state_dict(), rtol=0, atol=0)
        assert restored_norms[name].training and restored[name].branch.encoder.training
    assert_same(restored_optimizer.state_dict(), optimizer.state_dict())

    advance(norms, optimizer)
    advance(restored_norms, restored_optimizer)
    for name in domains:
        assert_same(restored[name].branch.pdae_state_dict(), domains[name].branch.pdae_state_dict())
        torch.testing.assert_close(restored_norms[name].state_dict(), norms[name].state_dict(), rtol=0, atol=0)
    assert_same(restored_optimizer.state_dict(), optimizer.state_dict())


def test_resume_supports_existing_checkpoints_without_rng_and_rejects_bn_mismatch(local_helpers, tmp_path):
    from infoot_helper.batchnorm_matching import batchnorm_checkpoint
    from infoot_helper.cotraining_checkpoint import resume_latest

    domains, norms, optimizer = training_state(False)
    checkpoint = {
        "step": 400,
        "models": {name: domain.branch.pdae_state_dict() for name, domain in domains.items()},
        "matching_batchnorm": {name: batchnorm_checkpoint(norm) for name, norm in norms.items()},
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, tmp_path / "step_000400.pt")
    rng = torch.get_rng_state().clone()
    assert resume_latest(tmp_path, domains, norms, optimizer) == 400
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    norms["cat"].eps = .1
    with pytest.raises(ValueError, match="BatchNorm settings"):
        resume_latest(tmp_path, domains, norms, optimizer)


@pytest.mark.parametrize("arguments,expected_step", [([], 400), (["--step", "200"], 200)])
def test_evaluation_defaults_to_latest_but_allows_explicit_step(local_helpers, tmp_path, monkeypatch,
                                                               arguments, expected_step):
    from infoot_helper import infoot_test_helper as helper
    from diffusion_ot.data import manifests, ground_truth

    _, transport = local_helpers
    output = tmp_path / "outputs/infoot_cotraining"
    output.mkdir(parents=True)
    for name in ("step_000200.pt", "step_000400.pt", "cat_to_dog_step_999999_plan.pt"):
        (output / name).touch()
    bank_dir = tmp_path / "data/infoot_test"
    bank_dir.mkdir(parents=True)
    for name in ("cat", "dog"):
        torch.save({"latent_paths": []}, bank_dir / f"{name}_bank.pt")
    latent_dir = tmp_path / "data/latents/afhq_sit_b2_256/cat_val"
    latent_dir.mkdir(parents=True)
    for index in range(16):
        (latent_dir / f"{index:03d}.pt").touch()
    features = {name: torch.ones(3, 2) for name in ("cat", "dog")}
    domains = {name: SimpleNamespace(name=name) for name in features}

    def prepare(root, banks, step, device, **kwargs):
        assert root == tmp_path and step == expected_step
        return domains, features, features, {"cat": torch.nn.Identity()}, torch.eye(3) / 3

    def save_grid(dog, codes, images, path, **kwargs):
        assert dog is domains["dog"] and len(codes) == len(images) == 16
        assert path.name == f"cotraining_step_{expected_step:06d}_1.png"

    monkeypatch.setattr(helper, "prepare_cotraining_test", prepare)
    monkeypatch.setattr(helper, "encode_paths", lambda domain, paths: torch.ones(len(paths), 2))
    monkeypatch.setattr(helper, "generate_and_save_grid", save_grid)
    monkeypatch.setattr(transport, "conditional_mapping", lambda query, *args, **kwargs: query)
    monkeypatch.setattr(manifests, "read_jsonl", lambda path: [{"sample_id": f"{i:03d}"} for i in range(16)])
    monkeypatch.setattr(ground_truth, "load_ground_truth_images", lambda *args: torch.ones(16, 3, 2, 2))
    script = tmp_path / "infoot/infoot_test_cotraining.py"
    script.parent.mkdir()
    original = Path(__file__).resolve().parents[1] / "infoot/infoot_test_cotraining.py"
    script.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(script), *arguments])
    runpy.run_path(str(script), run_name="__main__")
