import json
from types import SimpleNamespace

import pytest
import torch

from diffusion_ot.training.self_supervised_diagnostics import (
    image_correspondence, native_rgb_reconstruction, solver_health,
)
from diffusion_ot.training.self_supervised_translation import source_code_contrastive_loss
from diffusion_ot.training.train_joint_infoot import _solve_infoot_logged


def test_rgb_correspondence_distinguishes_identity_shuffling_and_collapse_without_encoder():
    originals = torch.rand(4, 3, 16, 16, generator=torch.Generator().manual_seed(1), requires_grad=True)
    good = image_correspondence(originals, originals)
    wrong = image_correspondence(originals.roll(1, 0), originals)
    flat = image_correspondence(torch.full_like(originals, .5), originals)
    assert good["paired_pooled_rgb_mse"] == 0
    assert good["pooled_rgb_source_retrieval_top1"] == 1
    assert good["paired_advantage_pooled_rgb_mse"] > 0
    assert good["edge_correlation"] == pytest.approx(1)
    assert wrong["pooled_rgb_source_retrieval_top1"] == 0
    assert wrong["paired_advantage_pooled_rgb_mse"] < 0
    assert flat["pooled_rgb_source_retrieval_top1"] == pytest.approx(.25)
    assert flat["paired_advantage_pooled_rgb_mse"] == pytest.approx(0, abs=1e-7)
    assert flat["output_to_source_pooled_rgb_variance_ratio"] == 0
    assert flat["edge_correlation"] is None and flat["edge_correlation_valid_samples"] == 0
    assert originals.grad is None
    json.dumps([good, wrong, flat], allow_nan=False)


def test_code_retrieval_success_does_not_automatically_pass_image_correspondence():
    keys = torch.eye(4)
    _, retrieval = source_code_contrastive_loss(keys, keys, keys)
    images = torch.rand(4, 3, 8, 8, generator=torch.Generator().manual_seed(9))
    correspondence = image_correspondence(images.roll(1, 0), images)
    assert retrieval["retrieval_top1"] == 1
    assert correspondence["pooled_rgb_source_retrieval_top1"] == 0


def test_tied_or_single_image_controls_cannot_claim_unique_retrieval():
    flat = torch.full((4, 3, 8, 8), .5)
    tied = image_correspondence(flat, flat)
    assert tied["pooled_rgb_source_retrieval_top1"] == pytest.approx(.25)
    assert tied["output_to_source_pooled_rgb_variance_ratio"] is None
    single = image_correspondence(flat[:1], flat[:1])
    assert single["pooled_rgb_source_retrieval_top1"] is None
    assert single["paired_advantage_pooled_rgb_mse"] is None
    assert single["edge_correlation"] is None
    json.dumps([tied, single], allow_nan=False)


def test_native_rgb_metrics_use_original_pixels_and_finite_psnr_without_rng():
    original = torch.full((2, 3, 8, 8), .25, requires_grad=True)
    state = torch.get_rng_state().clone()
    perfect = native_rgb_reconstruction(original, original)
    shifted = native_rgb_reconstruction(original + .1, original)
    assert perfect["pixel_mse"] == 0 and perfect["pixel_psnr_db"] == 120
    assert shifted["pixel_mse"] == pytest.approx(.01)
    assert shifted["pixel_psnr_db"] == pytest.approx(20)
    assert shifted["target"] == "original_rgb"
    assert original.grad is None
    assert torch.equal(state, torch.get_rng_state())
    json.dumps([perfect, shifted], allow_nan=False)


def test_rgb_metrics_preserve_original_float32_values_with_bfloat16_generation():
    generated = torch.full((2, 3, 4, 4), .125, dtype=torch.bfloat16)
    originals = torch.full((2, 3, 4, 4), .123456, dtype=torch.float32)
    expected = float((generated.float() - originals).square().mean())
    assert image_correspondence(generated, originals)["paired_pooled_rgb_mse"] == pytest.approx(expected)
    assert native_rgb_reconstruction(generated, originals)["pixel_mse"] == pytest.approx(expected)


def test_solver_health_distinguishes_recovery_and_outer_vs_marginal_tolerances():
    solution = SimpleNamespace(plan_delta_l1=8e-6, row_residual=5e-8, column_residual=1e-8,
        effective_projection_tolerance=1e-7, iterations=1250, recovery_iterations=50,
        unconverged_inner_steps=2)
    options = dict(outer_tolerance=1e-5, projection_tolerance=1e-5,
                   inner_iterations=1200, recovery_iterations=4800)
    report = solver_health(solution, options)
    assert report["outer_delta_to_tolerance"] == pytest.approx(.8)
    assert report["marginal_residual_to_tolerance"] == pytest.approx(.005)
    # Returned FP32 marginals use their advertised tolerance, not the tighter
    # internal FP64 recovery tolerance.
    assert report["outer_total_budget_fraction"] == pytest.approx(1250 / 6000)
    assert report["recovery_budget_fraction"] == pytest.approx(50 / 4800)
    assert report["recovery_used"] and report["unconverged_inner_steps"] == 2
    options.update(outer_tolerance=0, projection_tolerance=0, recovery_iterations=0)
    report = solver_health(solution, options)
    assert report["outer_delta_to_tolerance"] is None
    assert report["marginal_residual_to_tolerance"] is None
    assert report["recovery_budget_fraction"] is None


def test_failed_solver_is_logged_and_original_exception_is_reraised(monkeypatch, tmp_path):
    import diffusion_ot.losses.infoot as infoot

    failure = RuntimeError("InfoOT outer updates did not converge: plan_delta_l1=1.57e-05")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(infoot, "solve_infoot", fail)
    path = tmp_path / "logs/solver_failures.jsonl"
    with pytest.raises(RuntimeError) as caught:
        _solve_infoot_logged(torch.eye(3), torch.eye(4), failure_log=path, phase="validation_decoded",
            step=500, mi_weight=.1, outer_tolerance=1e-5, cross_cost=torch.ones(3, 4))
    assert caught.value is failure
    record = json.loads(path.read_text())
    assert record["step"] == 500 and record["phase"] == "validation_decoded"
    assert record["reference_counts"] == [3, 4]
    assert record["settings"] == {"mi_weight": .1, "outer_tolerance": 1e-5}
    assert record["error"] == str(failure)
