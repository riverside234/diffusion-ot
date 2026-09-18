from __future__ import annotations



import pytest


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


def test_training_device_uses_process_visible_cuda_indices(monkeypatch):
    import pytest
    import torch
    from diffusion_ot.training.train_pdae_domain import _validate_training_device

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    _validate_training_device("cuda:0")
    _validate_training_device("cuda")
    with pytest.raises(ValueError, match="pass --device cuda:0"):
        _validate_training_device("cuda:1")
    # Multiple visible devices still permit an explicit logical index.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    _validate_training_device("cuda:1")
    with pytest.raises(ValueError, match="valid logical indices are 0 through 1"):
        _validate_training_device("cuda:2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    _validate_training_device("cpu")
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        _validate_training_device("cuda:0")


def test_trainable_ema_tracks_only_trainable_parameters_and_restores_model():
    from diffusion_ot.training.train_pdae_domain import TrainableEMA

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    for parameter in model[1].parameters():
        parameter.requires_grad_(False)
    ema = TrainableEMA(model, decay=0.9, warmup_steps=2)
    assert set(ema.shadow) == {"0.weight", "0.bias"}

    with torch.no_grad():
        model[0].weight.add_(2.0)
        model[0].bias.add_(2.0)
    current_weight = model[0].weight.detach().clone()
    ema.update(model)
    shadow_weight = ema.shadow["0.weight"].detach().clone()
    assert ema.effective_decay == pytest.approx(0.45)

    with ema.average_parameters(model):
        torch.testing.assert_close(model[0].weight, shadow_weight)
    torch.testing.assert_close(model[0].weight, current_weight)
