from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import runpy
import sys

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
    kernels = infoot.compute_kernel(
        solver.Cs, solver.Ct, solver.h if h is None else h
    )
    torch.testing.assert_close(solver.conditional_score(source, h=h),
                               infoot.ratio(solver.P, *kernels))


def test_conditional_mapping_retains_live_feature_gradients(local_helpers):
    _, helpers = local_helpers
    source = torch.tensor([[0.0], [1.0], [2.0], [3.0]], requires_grad=True)
    target = torch.tensor([[0.0], [10.0], [20.0], [30.0]], requires_grad=True)
    query = torch.tensor([[0.5], [2.3]], requires_grad=True)
    plan = (torch.eye(4) / 4).requires_grad_()
    helpers.conditional_mapping(
        query, source, target, plan,
    ).square().mean().backward()
    for features in (query, source, target):
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
        assert features.grad.abs().sum() > 0
    assert plan.grad is None


def test_encoding_preserves_raw_codes_and_reference_split(local_helpers):
    from infoot_helper.encoding import encode_batches

    raw = torch.tensor([
        [50.0, -10.0, 9.0], [100.0, 30.0, 9.0],
        [1.0, 6.0, 9.0], [2.0, 8.0, 9.0],
        [3.0, 10.0, 9.0], [4.0, 12.0, 9.0],
    ])
    latents = {"cat": raw, "dog": raw * 3 + 20}
    domain = SimpleNamespace(
        device="cpu", model_dtype=torch.float32,
        branch=SimpleNamespace(encode=lambda x: x),
    )
    domains = dict.fromkeys(latents, domain)
    encoded = encode_batches(domains, latents, query_count=2)
    for name, v in latents.items():
        torch.testing.assert_close(encoded[name]["v"], v)
        torch.testing.assert_close(encoded[name]["references"]["v"], v[2:])
        torch.testing.assert_close(encoded[name]["queries"]["v"], v[:2])
        assert set(encoded[name]["references"]) == {"v"}
        assert set(encoded[name]["queries"]) == {"x0", "v"}


def test_kernel_uses_reference_scale_and_preserves_global_scale_invariance(local_helpers):
    infoot, _ = local_helpers
    source = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    target = torch.tensor([[0.0, 3.0], [3.0, 0.0]])
    Kx, Ky = infoot.compute_kernel(source, target, h=0.5)
    off_diagonal = torch.exp(torch.tensor(-8.0))
    expected = torch.tensor([[1.0, off_diagonal], [off_diagonal, 1.0]])
    torch.testing.assert_close(Kx, expected)
    torch.testing.assert_close(Ky, expected)
    scaled = infoot.compute_kernel(source * 10, target * 0.2, h=0.5)
    torch.testing.assert_close((Kx, Ky), scaled)
    query = torch.tensor([[1.0, 3.0]])
    Kq, _ = infoot.compute_kernel(query, target, h=0.5, Cx_reference=source)
    torch.testing.assert_close(Kq, torch.exp(torch.tensor([[-2.0, -18.0]])))


def test_raw_alignment_matches_fit_and_trains_encoders(local_helpers):
    infoot, helpers = local_helpers
    generator = torch.Generator().manual_seed(17)
    scale = torch.logspace(-1, 1, 512)
    source = (torch.randn(24, 512, generator=generator) * scale + 30).requires_grad_()
    target = (torch.randn(24, 512, generator=generator) * scale.flip(0) - 20).requires_grad_()
    plan = helpers.fit_transport(source, target, iterations=3)
    assert not plan.requires_grad
    plan.requires_grad_()
    loss = helpers.alignment_loss(source, target, plan, h=0.4)
    with torch.no_grad():
        solver = infoot.FusedInfoOT(source, target, h=0.4, lam=0.1, reg=0.02)
        expected = infoot.fitting_loss(plan, solver.Ks, solver.Kt, 0.02, C=solver.C, mi_weight=0.1)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for features in (source, target):
        assert torch.isfinite(features.grad).all()
        assert features.grad.abs().sum() > 0
    assert plan.grad is None


def test_saved_raw_plan_preserves_query_mapping_and_target_scale(local_helpers, tmp_path):
    infoot, helpers = local_helpers
    source = torch.tensor([[0., 1.], [1., 2.], [2., 5.], [3., 9.]])
    target = torch.tensor([[20., 100.], [40., 100.], [70., 100.], [90., 100.]])
    plan = torch.eye(4) / 4
    path = tmp_path / "plan.pt"
    infoot.save_plan(path, plan, h=0.4, reg=0.02, lam=0.1)
    saved = torch.load(path, weights_only=True)
    query = torch.tensor([[0.5, 1.5], [2.3, 6.0]])

    def mapped(rows, state):
        return helpers.conditional_mapping(
            rows, source, target, state["P"], h=state["h"],
        )

    together = mapped(query, saved)
    separately = torch.cat([mapped(row[None], saved) for row in query])
    before = mapped(query, {"P": plan, "h": 0.4})
    torch.testing.assert_close(together, separately)
    torch.testing.assert_close(together, before)
    torch.testing.assert_close(together[:, 1], torch.full((2,), 100.))
    assert saved["h"] == 0.4 and saved["reg"] == 0.02
    assert saved["lam"] == 0.1 and saved["feature_space"] == "raw"
    assert "stats" not in saved


def test_cotraining_test_fits_reencoded_raw_training_banks(local_helpers, tmp_path, monkeypatch):
    from infoot_helper import infoot_test_helper as helper

    features = {
        "cat": torch.tensor([[1., 3.], [2., 7.], [4., 9.]]),
        "dog": torch.tensor([[20., 10.], [50., 70.], [90., 80.]]),
    }
    models = {name: SimpleNamespace(name=name) for name in features}
    output_dir = tmp_path / "outputs/infoot_cotraining"
    output_dir.mkdir(parents=True)

    def load_model(config, evaluation, **kwargs):
        name = config.name.split("_")[0]
        assert kwargs["checkpoint_path"] == output_dir / f"{name}_step_002000.pt"
        return models[name]

    def fit(source, target, **kwargs):
        torch.testing.assert_close(source, features["cat"])
        torch.testing.assert_close(target, features["dog"])
        return torch.eye(3) / 3

    monkeypatch.setattr(helper, "load_stage1a_evaluator", load_model)
    monkeypatch.setattr(helper, "encode_paths", lambda model, paths: features[model.name])
    monkeypatch.setattr(helper, "fit_transport", fit)
    result = helper.prepare_cotraining_test(
        tmp_path, {name: {"latent_paths": []} for name in features}, 2000, "cpu"
    )
    cat, dog, Xs, Xt, P = result
    assert cat is models["cat"] and dog is models["dog"]
    torch.testing.assert_close(Xs, features["cat"])
    torch.testing.assert_close(Xt, features["dog"])
    saved = torch.load(output_dir / "cat_to_dog_step_002000_plan.pt", weights_only=True)
    torch.testing.assert_close(saved["P"], P)
    assert saved["feature_space"] == "raw" and "stats" not in saved


def test_bank_fit_script_passes_raw_features_and_cli_lam(local_helpers, tmp_path, monkeypatch):
    infoot, _ = local_helpers
    bank_dir = tmp_path / "data/infoot_test"
    bank_dir.mkdir(parents=True)
    source = torch.tensor([[10., 30.], [20., 40.], [30., 70.]])
    target = source * 2 + 5
    for name, features in (("cat", source), ("dog", target)):
        torch.save({"v_bank": features}, bank_dir / f"{name}_bank.pt")
    script = tmp_path / "infoot/infoot_fit.py"
    script.parent.mkdir()
    original = Path(__file__).resolve().parents[1] / "infoot/infoot_fit.py"
    script.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")

    class Solver:
        def __init__(self, Xs, Xt, h, reg, lam):
            torch.testing.assert_close(Xs, source)
            torch.testing.assert_close(Xt, target)
            assert (h, reg, lam) == (0.4, 0.02, 0.37)

        def solve(self, numIter, verbose):
            assert numIter == 50 and verbose is True
            return torch.eye(3) / 3

    monkeypatch.setattr(infoot, "FusedInfoOT", Solver)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", [str(script), "--lam", "0.37"])
    runpy.run_path(str(script), run_name="__main__")
    saved = torch.load(bank_dir / "cat_to_dog_plan.pt", weights_only=True)
    assert saved["lam"] == 0.37 and saved["feature_space"] == "raw"
    assert "stats" not in saved


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
