from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


class TinyBranch(nn.Module):
    def __init__(self, velocity: float = 1.0) -> None:
        super().__init__()
        self.trainable = nn.Parameter(torch.tensor(0.0))
        self.frozen = nn.Parameter(torch.tensor(7.0), requires_grad=False)
        self.velocity = float(velocity)
        self.timesteps: list[float] = []

    def predict_with_z(self, x_t, timestep, z, class_labels):
        self.timesteps.append(float(timestep[0]))
        base = torch.full_like(x_t, self.velocity)
        return SimpleNamespace(base_sample=base, delta_sample=torch.zeros_like(base))


class TinyTransformer:
    config = SimpleNamespace(num_classes=3)


def test_ema_checkpoint_loading_updates_only_trainable_parameters(tmp_path):
    from diffusion_ot.evaluation.stage1a_eval import _apply_ema_weights, _load_checkpoint

    branch = TinyBranch()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": {"placeholder": {}},
            "ema": {"shadow": {"trainable": torch.tensor(4.5)}},
        },
        path,
    )

    checkpoint = _load_checkpoint(path)
    _apply_ema_weights(branch, checkpoint)

    assert float(branch.trainable.detach()) == pytest.approx(4.5)
    assert float(branch.frozen.detach()) == pytest.approx(7.0)


def test_reconstruction_is_deterministic_for_the_same_seed():
    from diffusion_ot.evaluation.stage1a_eval import _noise_like, integrate_pdae_flow

    template = torch.zeros(2, 1, 2, 2)
    first_noise = _noise_like(template, seed=19)
    second_noise = _noise_like(template, seed=19)
    branch = TinyBranch(velocity=0.25)
    z = torch.zeros(2, 4)

    first = integrate_pdae_flow(branch, TinyTransformer(), first_noise, z, num_steps=4)
    second = integrate_pdae_flow(branch, TinyTransformer(), second_noise, z, num_steps=4)

    torch.testing.assert_close(first_noise, second_noise)
    torch.testing.assert_close(first, second)


def test_all_z_variants_receive_identical_starting_noise(monkeypatch):
    import diffusion_ot.evaluation.stage1a_eval as evaluation

    observed = []

    def fake_integrate(branch, transformer, initial_state, z, **kwargs):
        observed.append(initial_state.clone())
        return initial_state + z[:, :1, None, None]

    monkeypatch.setattr(evaluation, "integrate_pdae_flow", fake_integrate)
    starting_noise = torch.randn(3, 1, 2, 2)
    original = starting_noise.clone()
    z = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    outputs = evaluation.reconstruct_z_variants(
        object(),
        object(),
        starting_noise,
        z,
        ["correct_z", "shuffled_z", "zero_z"],
        num_steps=2,
    )

    assert list(outputs) == ["correct_z", "shuffled_z", "zero_z"]
    assert len(observed) == 3
    for value in observed:
        torch.testing.assert_close(value, original)
    torch.testing.assert_close(starting_noise, original)


def test_solver_integrates_in_noise_to_data_direction():
    from diffusion_ot.evaluation.stage1a_eval import integrate_pdae_flow

    branch = TinyBranch(velocity=1.0)
    initial = torch.zeros(2, 1, 2, 2)
    result = integrate_pdae_flow(
        branch,
        TinyTransformer(),
        initial,
        torch.zeros(2, 4),
        num_steps=4,
        start_time=0.0,
        end_time=1.0,
    )

    torch.testing.assert_close(result, torch.ones_like(result))
    assert branch.timesteps == pytest.approx([0.125, 0.375, 0.625, 0.875])


def test_identical_image_metrics_are_exact():
    from diffusion_ot.evaluation.stage1a_eval import _mse_and_psnr

    image = torch.rand(3, 3, 16, 16)
    metrics = _mse_and_psnr(image, image)

    assert metrics["mse"] == 0.0
    assert metrics["psnr"] == float("inf")


def test_nearest_neighbors_exclude_the_query_itself():
    from diffusion_ot.evaluation.stage1a_eval import nearest_neighbor_indices

    z = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    neighbors = nearest_neighbor_indices(z, k=1, metric="cosine", exclude_self=True)

    assert neighbors.shape == (3, 1)
    assert torch.all(neighbors.squeeze(1) != torch.arange(3))
    assert int(neighbors[0, 0]) == 1


def test_interpolation_includes_exact_endpoints():
    from diffusion_ot.evaluation.stage1a_eval import interpolate_latents

    z_start = torch.tensor([[0.0, 1.0]])
    z_end = torch.tensor([[4.0, 3.0]])
    values = interpolate_latents(z_start, z_end, num_steps=5)

    assert values.shape == (5, 1, 2)
    torch.testing.assert_close(values[0], z_start)
    torch.testing.assert_close(values[-1], z_end)
    torch.testing.assert_close(values[2], (z_start + z_end) / 2.0)
