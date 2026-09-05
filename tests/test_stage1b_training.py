from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


class TinyBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 2)
        self.semantic_transformer = nn.Linear(2, 3)


class TinySemanticTransformer(nn.Linear):
    def trainable_state_dict(self):
        return self.state_dict()


def test_stage1b_freezes_generator_and_keeps_encoder_trainable():
    from diffusion_ot.training.train_joint_infoot import freeze_generator_train_encoder

    branch = TinyBranch()
    freeze_generator_train_encoder(branch)

    assert all(parameter.requires_grad for parameter in branch.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in branch.semantic_transformer.parameters())
    assert branch.encoder.training
    assert not branch.semantic_transformer.training


def test_reconstruction_gradient_passes_through_frozen_generator_to_encoder():
    from diffusion_ot.training.train_joint_infoot import freeze_generator_train_encoder

    branch = TinyBranch()
    freeze_generator_train_encoder(branch)
    value = torch.randn(4, 3)
    loss = branch.semantic_transformer(branch.encoder(value)).square().mean()
    loss.backward()

    assert all(parameter.grad is not None for parameter in branch.encoder.parameters())
    assert all(parameter.grad is None for parameter in branch.semantic_transformer.parameters())


def test_anchor_loss_is_variance_scaled_and_reference_is_detached():
    from diffusion_ot.training.train_joint_infoot import encoder_anchor_loss

    current = torch.tensor([[2.0, 4.0]], requires_grad=True)
    reference = torch.tensor([[0.0, 0.0]], requires_grad=True)
    loss = encoder_anchor_loss(current, reference, variance=2.0)
    loss.backward()

    assert float(loss.detach()) == pytest.approx(5.0)
    assert current.grad is not None
    assert reference.grad is None


def test_encoder_ema_exports_domain_states():
    from diffusion_ot.training.train_joint_infoot import EncoderEMA

    encoders = {"cat": nn.Linear(2, 2), "dog": nn.Linear(2, 2)}
    ema = EncoderEMA(encoders, decay=0.9, warmup_steps=0)
    with torch.no_grad():
        encoders["cat"].weight.add_(1.0)
    ema.update(encoders)
    exported = ema.export()

    assert set(exported) == {"cat", "dog"}
    assert set(exported["cat"]) == set(encoders["cat"].state_dict())
    assert ema.num_updates == 1


def test_joint_checkpoint_payload_contains_no_transport_plan():
    from diffusion_ot.training.train_joint_infoot import _build_checkpoint_payload

    encoders = {"cat": nn.Linear(2, 2), "dog": nn.Linear(2, 2)}
    optimizer = torch.optim.AdamW(
        [parameter for encoder in encoders.values() for parameter in encoder.parameters()]
    )
    domains = {
        domain: SimpleNamespace(
            checkpoint_path=f"{domain}.pt",
            checkpoint_step=25_000,
            branch=SimpleNamespace(semantic_transformer=TinySemanticTransformer(2, 3)),
        )
        for domain in encoders
    }
    generators = {domain: torch.Generator().manual_seed(1) for domain in encoders}
    payload = _build_checkpoint_payload(
        step=7,
        encoders=encoders,
        optimizer=optimizer,
        ema=None,
        config={"stage1a": {"weights": "ema"}},
        domains=domains,
        train_state={},
        loader_generators=generators,
    )

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key).lower()
                yield from keys(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from keys(child)

    all_keys = set(keys(payload))
    assert "gamma" not in all_keys
    assert "coupling" not in all_keys
    assert payload["stage"] == "stage1b_plain_infoot"
    assert set(payload["encoders"]) == {"cat", "dog"}
    assert set(payload["fixed_generators"]) == {"cat", "dog"}


def test_joint_checkpoint_roundtrip(tmp_path):
    from diffusion_ot.training.train_joint_infoot import _load_checkpoint, _save_checkpoint

    path = tmp_path / "latest.pt"
    payload = {
        "stage": "stage1b_plain_infoot",
        "step": 12,
        "encoders": {"cat": {}, "dog": {}},
    }
    _save_checkpoint(path, payload)
    loaded = _load_checkpoint(path)

    assert loaded["step"] == 12
    assert set(loaded["encoders"]) == {"cat", "dog"}
