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


def test_learning_rate_warmup_boundaries_and_resumed_global_step():
    from diffusion_ot.training.learning_rate import apply_step_learning_rates, resolve_lr_schedule

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    optimizer = torch.optim.AdamW([
        {"params": model[0].parameters(), "lr": 1e-4},
        {"params": model[1].parameters(), "lr": 2.5e-5},
    ])
    peaks = [group["lr"] for group in optimizer.param_groups]
    schedule = resolve_lr_schedule({"type": "constant_with_warmup", "warmup_steps": 500})
    for step, factor in [(1, .002), (250, .5), (500, 1.), (501, 1.), (10000, 1.)]:
        assert apply_step_learning_rates(optimizer, peaks, schedule, step) == pytest.approx(factor)
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([lr * factor for lr in peaks])
    # Resuming a saved step 249 needs no scheduler counter reset or replay.
    apply_step_learning_rates(optimizer, peaks, schedule, 250)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    apply_step_learning_rates(optimizer, peaks, resolve_lr_schedule(None), 1)
    assert [group["lr"] for group in optimizer.param_groups] == peaks


@pytest.mark.parametrize("config", [
    "cosine", {"type": "cosine"}, {"type": "constant_with_warmup"},
    {"type": "constant", "warmup_steps": 500}, {"warmup_steps": -1},
    {"warmup_steps": True}, {"warmup_steps": 1.5}, {"unused": 3},
])
def test_learning_rate_schedule_rejects_unsupported_or_ignored_settings(config):
    from diffusion_ot.training.learning_rate import resolve_lr_schedule

    with pytest.raises(ValueError):
        resolve_lr_schedule(config)
