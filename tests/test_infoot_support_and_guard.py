from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from diffusion_ot.losses.infoot import sinkhorn_divergence
from diffusion_ot.losses.projection_support import projection_support_loss
from diffusion_ot.losses.semantic_prior import validate_prior_resume
from diffusion_ot.training.gradient_guard import guarded_backward


@pytest.mark.parametrize("nonuniform", [False, True])
def test_sinkhorn_divergence_values_and_gradients_match_independent_pot(nonuniform):
    ot = pytest.importorskip("ot")
    import numpy as np
    torch.manual_seed(211)
    source = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    target = torch.randn(6, 2, dtype=torch.float64)
    settings = dict(regularization=.5, max_iterations=3000, tolerance=1e-12, cost_scale=2.0)
    a = torch.arange(1, 5, dtype=torch.float64) / 10 if nonuniform else torch.ones(4, dtype=torch.float64) / 4
    b = torch.arange(1, 7, dtype=torch.float64) / 21 if nonuniform else torch.ones(6, dtype=torch.float64) / 6
    settings.update(source_masses=a, target_masses=b)
    result = sinkhorn_divergence(source, target, **settings)
    def ot_value(x, y, a, b):
        cost = ((x[:, None] - y[None, :])**2).sum(2) / 2
        plan = ot.sinkhorn(a, b, cost, .5, numItermax=5000, stopThr=1e-14)
        return (plan * cost).sum() + .5 * (plan * np.log(plan / np.outer(a, b))).sum()
    x, y = source.detach().numpy(), target.numpy()
    an, bn = a.numpy(), b.numpy()
    expected = ot_value(x, y, an, bn) - .5 * ot_value(x, x, an, an) - .5 * ot_value(y, y, bn, bn)
    assert float(result.loss.detach()) == pytest.approx(expected, abs=1e-9)
    assert result.converged and result.loss > 0
    assert torch.autograd.gradcheck(lambda z: sinkhorn_divergence(z, target, **settings).loss,
                                    (source,), atol=1e-4, rtol=1e-3)
    self_result = sinkhorn_divergence(source, source, **{**settings, "target_masses": a})
    assert abs(float(self_result.loss.detach())) < 1e-12
    self_result.loss.backward()
    torch.testing.assert_close(source.grad, torch.zeros_like(source), atol=1e-10, rtol=0)


def test_sinkhorn_loss_penalizes_contraction_without_an_output_norm_rescale():
    target = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=torch.float64)
    scale = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    loss = sinkhorn_divergence(scale * target, target, regularization=.15,
                               max_iterations=3000, tolerance=1e-12).loss
    loss.backward()
    assert scale.grad < 0  # Descent increases the spread from a contracted distribution.
    assert loss > sinkhorn_divergence(.8 * target, target, regularization=.15).loss


def test_support_teaches_weights_and_detaches_anchor_and_value_banks():
    torch.manual_seed(213)
    references = {d: torch.randn(7, 3, requires_grad=True) for d in ("cat", "dog")}
    anchors = {d: torch.randn(8, 3, requires_grad=True) for d in references}
    logits = {name: torch.randn(4, 7, requires_grad=True) for name in ("cat_to_dog", "dog_to_cat")}
    result = projection_support_loss({k: v.softmax(1) for k, v in logits.items()}, references,
                                     anchors, {"cat": 1., "dog": 1.}, regularization=.5,
                                     max_iterations=3000)
    result.loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.norm() > 0 for x in logits.values())
    assert all(x.grad is None for x in [*references.values(), *anchors.values()])
    assert all(r["converged"] for r in result.metrics.values())


def test_unconverged_support_plans_are_not_used_for_envelope_gradients():
    torch.manual_seed(214)
    x, y = torch.randn(4, 3), torch.randn(7, 3)
    with pytest.raises(RuntimeError, match="Projection-support Sinkhorn residual"):
        sinkhorn_divergence(x, y, regularization=.01, max_iterations=1, tolerance=1e-10)


def test_guard_caps_and_projects_per_encoder_instead_of_only_global_clipping():
    cat = torch.nn.Parameter(torch.zeros(2))
    dog = torch.nn.Parameter(torch.zeros(2))
    primary = cat[0] + 2 * dog[1]
    auxiliary = -10 * cat[0] + 20 * cat[1] + 10 * dog[0] + dog[1]
    metrics = guarded_backward(primary, auxiliary, {"cat": [cat], "dog": [dog]}, max_auxiliary_ratio=.25)
    for name, p, gp in (("cat", cat, torch.tensor([1., 0.])), ("dog", dog, torch.tensor([0., 2.]))):
        ga = p.grad - gp
        assert torch.dot(ga, gp) >= -1e-7
        assert ga.norm() <= .25 * gp.norm() + 1e-7
        assert metrics[name]["auxiliary_ratio_after"] <= .25 + 1e-7
    assert metrics["cat"]["conflict_projected"]
    assert not metrics["dog"]["conflict_projected"]


def test_guard_with_zero_primary_has_no_unbounded_auxiliary_update():
    parameter = torch.nn.Parameter(torch.ones(2))
    result = guarded_backward(parameter.sum() * 0, parameter.sum() * 10, {"cat": [parameter]})
    torch.testing.assert_close(parameter.grad, torch.zeros(2))
    assert result["cat"]["auxiliary_gradient_norm_after"] == 0


def test_resume_rejects_changes_to_guard_or_support_objective():
    config = {"gradient_guard": {"enabled": True, "max_auxiliary_ratio": .25},
              "projection_support": {"enabled": True, "regularization": .1}}
    validate_prior_resume(config, config)
    for key in config:
        changed = deepcopy(config)
        changed[key] = {"enabled": False}
        with pytest.raises(ValueError, match=f"Resume cannot change {key}"):
            validate_prior_resume(config, changed)


def test_legacy_snapshot_hashes_and_runtime_are_isolated_from_active_changes():
    root = Path(__file__).resolve().parents[1]
    snapshot = root / "legacy/infoot_official_v1"
    manifest = json.loads((snapshot / "snapshot.json").read_text())
    for record in manifest["files"]:
        assert hashlib.sha256((snapshot / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    checked = subprocess.run([sys.executable, str(root / "scripts/run_legacy_infoot.py"), "check"],
                             capture_output=True, text=True, check=True, timeout=60)
    data = json.loads(checked.stdout)
    assert data["verified_files"] == len(manifest["files"])
    assert all(Path(path).is_relative_to(snapshot) for path in data["modules"].values())
    assert not (snapshot / "src/diffusion_ot/training/gradient_guard.py").exists()
