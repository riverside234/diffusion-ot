from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from test_pdae_sit_shapes import FakeSiT
from test_residual_encoder import tiny_config, small_cpu_thread_pool


@pytest.fixture
def local_helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "infoot"))
    from infoot_helper import infoot, infoot_cotraining_helper
    return infoot, infoot_cotraining_helper


@pytest.mark.parametrize("solver_name", ["InfoOT", "FusedInfoOT"])
@pytest.mark.parametrize("h", [None, 0.8])
def test_conditional_mapping_is_independent_of_query_batch(local_helpers, solver_name, h):
    infoot, _ = local_helpers
    source = torch.arange(4, dtype=torch.float64)[:, None]
    target = 10 * source
    solver = getattr(infoot, solver_name)(source, target, h=0.5)
    solver.P = torch.eye(4, dtype=source.dtype) / 4
    query = torch.tensor([[0.5], [2.3], [6.0]], dtype=source.dtype)
    together = infoot.projection(solver.conditional_score(query, h=h), target)
    separately = torch.cat([
        infoot.projection(solver.conditional_score(row[None], h=h), target)
        for row in query
    ])
    torch.testing.assert_close(together, separately)
    kernels = infoot.compute_kernel(solver.Cs, solver.Ct, solver.h if h is None else h)
    torch.testing.assert_close(solver.conditional_score(source, h=h),
                               infoot.ratio(solver.P, *kernels))


def test_conditional_mapping_retains_live_feature_gradients(local_helpers):
    _, helpers = local_helpers
    source = torch.tensor([[0.0], [1.0], [2.0], [3.0]], requires_grad=True)
    target = torch.tensor([[0.0], [10.0], [20.0], [30.0]], requires_grad=True)
    query = torch.tensor([[0.5], [2.3]], requires_grad=True)
    plan = (torch.eye(4) / 4).requires_grad_()
    helpers.conditional_mapping(query, source, target, plan).square().mean().backward()
    for features in (query, source, target):
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
        assert features.grad.abs().sum() > 0
    assert plan.grad is None


@pytest.mark.parametrize("solver_name", ["InfoOT", "FusedInfoOT"])
@pytest.mark.parametrize("verbose", [False, True])
def test_log_sinkhorn_preserves_marginals_on_raw_features(local_helpers, solver_name, verbose):
    infoot, _ = local_helpers
    generator = torch.Generator().manual_seed(42)
    source = torch.nn.functional.layer_norm(torch.randn(56, 512, generator=generator), (512,))
    target = torch.nn.functional.layer_norm(torch.randn(56, 512, generator=generator), (512,))
    kwargs = {"lam": 0.10} if solver_name == "FusedInfoOT" else {}
    solver = getattr(infoot, solver_name)(source, target, h=0.5, reg=0.02, **kwargs)
    plan = solver.solve(numIter=3, verbose=verbose)
    assert torch.isfinite(plan).all()
    assert (plan >= 0).all()
    for masses in (plan.sum(0), plan.sum(1)):
        torch.testing.assert_close(masses, torch.full_like(masses, 1 / 56),
                                   atol=1e-4, rtol=1e-3)


def test_domain_exports_load_with_bank_encoder_and_stage1a_evaluator(local_helpers, tmp_path, monkeypatch):
    from diffusion_ot.evaluation.stage1a_eval import load_stage1a_evaluator
    from diffusion_ot.models.pdae_sit import build_pdae_sit_branch
    from diffusion_ot.models.residual_encoder import PDAEResidualLatentEncoder
    import diffusion_ot.integrations.sit_diffusers as sit
    import diffusion_ot.data.latent_dataset as latent_dataset

    _, helpers = local_helpers
    transformer = FakeSiT()
    monkeypatch.setattr(sit, "load_sit_components", lambda *args, **kwargs:
                        SimpleNamespace(transformer=deepcopy(transformer), vae=torch.nn.Identity()))
    monkeypatch.setattr(latent_dataset, "CachedLatentDataset", lambda **kwargs: [])
    (tmp_path / "model.yaml").write_text("pretrained: pretrained.yaml\n")
    (tmp_path / "eval.yaml").write_text(yaml.safe_dump({
        "project_root": str(tmp_path), "sampling": {"variants": ["correct_z"]},
    }))
    domains = {}
    for index, name in enumerate(("cat", "dog")):
        config = dict(tiny_config(), domain=name, project_root=str(tmp_path),
                      data_config="data.yaml", model_config="model.yaml",
                      loss_weighting={"type": "cosmap"})
        branch = build_pdae_sit_branch(deepcopy(transformer), stage_config=config)
        with torch.no_grad():
            for parameter in branch.parameters():
                if parameter.requires_grad:
                    parameter.add_(0.01 * (index + 1))
        domains[name] = SimpleNamespace(branch=branch, training_config=config)
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(config))

    helpers.save_domain_checkpoints(domains, tmp_path, 1000)
    for name, domain in domains.items():
        path = tmp_path / f"{name}_step_001000.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["domain"] == name
        assert saved["native_flow_objective"]["loss_weighting"]["type"] == "cosmap"
        encoder_config = dict(saved["config"]["encoder"])
        encoder_config.pop("kind")
        encoder_config.pop("input_space")
        bank_encoder = PDAEResidualLatentEncoder(**encoder_config)
        bank_encoder.load_state_dict(saved["model"]["encoder"], strict=True)
        evaluator = load_stage1a_evaluator(tmp_path / f"{name}.yaml", tmp_path / "eval.yaml",
                                          checkpoint_path=path, device="cpu", weights="raw")
        assert evaluator.checkpoint_step == 1000
        torch.testing.assert_close(bank_encoder.state_dict(), evaluator.branch.encoder.state_dict())
        restored = dict(evaluator.branch.named_parameters())
        for key, expected in domain.branch.named_parameters():
            if expected.requires_grad:
                torch.testing.assert_close(restored[key], expected, rtol=0, atol=0)
