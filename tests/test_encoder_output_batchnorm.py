"""Optional CNN output BatchNorm behavior and checkpoint compatibility."""
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from diffusion_ot.evaluation.stage1a_eval import stage1a_architecture_metadata
from diffusion_ot.integrations.hf_snapshot import load_yaml_config
from diffusion_ot.models.pdae_sit import build_pdae_sit_branch
from test_pdae_sit_shapes import FakeSiT
from test_residual_encoder import tiny_config, small_cpu_thread_pool


def make_branch(kind, enabled=None):
    config = tiny_config()
    config["encoder"]["kind"] = kind
    if enabled is not None:
        config["encoder"]["output_batchnorm"] = enabled
    return build_pdae_sit_branch(FakeSiT(), stage_config=config)


@pytest.mark.parametrize("kind", ["plain_cnn_v1", "residual_cnn_v1"])
def test_disabled_output_batchnorm_preserves_outputs_and_legacy_checkpoints(kind):
    torch.manual_seed(1051)
    original = make_branch(kind).eval()
    torch.manual_seed(1051)
    disabled = make_branch(kind, False).eval()
    assert isinstance(disabled.encoder.output_norm, torch.nn.Identity)
    assert "output_batchnorm" not in disabled.encoder_architecture
    torch.testing.assert_close(disabled.state_dict(), original.state_dict(), rtol=0, atol=0)

    saved = deepcopy(original.pdae_state_dict())
    assert saved["format_version"] == (4 if kind == "plain_cnn_v1" else 5)
    disabled.load_pdae_state_dict(saved)
    value = torch.randn(3, 4, 8, 8)
    torch.testing.assert_close(disabled.encode(value), original.encode(value), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["plain_cnn_v1", "residual_cnn_v1"])
def test_output_batchnorm_standardizes_features_and_preserves_gradients(kind):
    branch = make_branch(kind, True).double().train()
    norm = branch.encoder.output_norm
    assert isinstance(norm, torch.nn.BatchNorm1d)
    assert not norm.affine and norm.track_running_stats
    before = []
    handle = norm.register_forward_pre_hook(lambda _, inputs: before.append(inputs[0].detach()))
    value = torch.randn(24, 4, 8, 8, dtype=torch.float64, requires_grad=True)
    result = branch.encode(value)
    handle.remove()

    variance, mean = torch.var_mean(before[0], dim=0, unbiased=False)
    expected = (before[0] - mean) * torch.rsqrt(variance + norm.eps)
    torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(result.mean(0), torch.zeros_like(mean), rtol=0, atol=1e-10)
    torch.testing.assert_close(result.var(0, unbiased=False), variance / (variance + norm.eps))
    torch.testing.assert_close(norm.running_mean, norm.momentum * mean)
    torch.testing.assert_close(
        norm.running_var, (1 - norm.momentum) + norm.momentum * before[0].var(0, unbiased=True),
    )
    assert norm.num_batches_tracked.item() == 1
    (result * torch.randn_like(result)).sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["plain_cnn_v1", "residual_cnn_v1"])
def test_batchnorm_checkpoint_restores_running_statistics_and_single_image_evaluation(kind):
    branch = make_branch(kind, True).double().train()
    with torch.no_grad():
        branch.encode(torch.randn(12, 4, 8, 8, dtype=torch.float64))
    branch.eval()
    saved = deepcopy(branch.pdae_state_dict())
    assert saved["format_version"] == 5
    assert saved["encoder_architecture"]["output_batchnorm"] is True
    assert {"output_norm.running_mean", "output_norm.running_var",
            "output_norm.num_batches_tracked"}.issubset(saved["encoder"])
    assert stage1a_architecture_metadata(branch)["encoder_architecture"]["output_batchnorm"] is True

    restored = make_branch(kind, True).double().eval()
    restored.load_pdae_state_dict(saved)
    value = torch.randn(3, 4, 8, 8, dtype=torch.float64)
    with torch.no_grad():
        expected = branch.encode(value)
        torch.testing.assert_close(restored.encode(value), expected, rtol=0, atol=0)
        torch.testing.assert_close(restored.encode(value[:1]), expected[:1], rtol=1e-10, atol=1e-10)
    assert restored.encoder.output_norm.num_batches_tracked.item() == 1


@pytest.mark.parametrize("kind", ["plain_cnn_v1", "residual_cnn_v1"])
@pytest.mark.parametrize("enabled", [False, True])
def test_output_batchnorm_setting_must_match_checkpoint(kind, enabled):
    source = make_branch(kind, enabled)
    target = make_branch(kind, not enabled)
    with pytest.raises(ValueError, match="architecture mismatch"):
        target.load_pdae_state_dict(source.pdae_state_dict())


@pytest.mark.parametrize("kind", ["plain_cnn_v1", "residual_cnn_v1"])
@pytest.mark.parametrize("invalid", ["false", "true", 0, 1])
def test_output_batchnorm_requires_a_boolean(kind, invalid):
    with pytest.raises(ValueError, match="encoder.output_batchnorm must be a Boolean"):
        make_branch(kind, invalid)


@pytest.mark.parametrize("domain", ["cat", "dog"])
def test_default_cnn_recipes_disable_output_batchnorm(domain):
    root = Path(__file__).resolve().parents[1]
    recipe = root / "configs/stage1a_pdae" / f"{domain}_sit_b2_lora_residual_cosmap.yaml"
    assert load_yaml_config(recipe)["encoder"]["output_batchnorm"] is False
