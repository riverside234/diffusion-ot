from copy import deepcopy

import pytest
import torch

from diffusion_ot.training.pcgrad import PCGradConfig, pcgrad_backward, project_task_gradients


def rng(seed=42):
    return torch.Generator().manual_seed(seed)


def test_conflicting_pair_matches_symmetric_projection_and_sum():
    gradients = {"flow": (torch.tensor([1., 0.]), None),
                 "image": (torch.tensor([-1., 1.]), None)}
    before = deepcopy(gradients)
    result, report = project_task_gradients(gradients, generator=rng())
    # g_flow -> [.5,.5]; g_image -> [0,1]. Neither task is privileged.
    torch.testing.assert_close(result[0], torch.tensor([.5, 1.5]))
    assert result[1] is None
    assert report["projection_count"] == 2 and report["conflicting_pairs_before"] == 1
    assert report["tasks"]["flow"]["sum_descent_fraction_after"] == pytest.approx(.5)
    assert report["tasks"]["image"]["sum_descent_fraction_after"] == pytest.approx(.5)
    for key in gradients:
        torch.testing.assert_close(gradients[key][0], before[key][0])


def test_nonconflicting_and_unused_tasks_preserve_sum_not_mean():
    result, report = project_task_gradients(
        {"one": (torch.tensor([1., 2.]), None),
         "two": (torch.tensor([2., 1.]), torch.tensor([4.])),
         "off": (None, None), "zero": (torch.zeros(2), None)}, generator=rng())
    torch.testing.assert_close(result[0], torch.tensor([3., 3.]))
    torch.testing.assert_close(result[1], torch.tensor([4.]))
    assert report["projection_count"] == 0 and report["correction_norm"] == 0
    assert report["active_tasks"] == ["one", "two"]


def test_gram_form_matches_direct_random_order_algorithm():
    matrix = torch.randn(5, 13, generator=rng(91))
    tasks = {str(i): (row[:8].reshape(2, 4), row[8:]) for i, row in enumerate(matrix)}
    merged, _ = project_task_gradients(tasks, generator=rng(7))
    projected = matrix.clone()
    gen = rng(7)
    for i in range(5):
        others = [j for j in range(5) if j != i]
        for index in torch.randperm(len(others), generator=gen).tolist():
            j = others[index]
            dot = projected[i] @ matrix[j]
            if dot < 0:
                projected[i] -= dot / matrix[j].square().sum() * matrix[j]
    torch.testing.assert_close(torch.cat([g.flatten() for g in merged]), projected.sum(0), rtol=2e-5, atol=2e-6)


def test_backward_preserves_group_routing_private_rng_and_step_order():
    e, h, g, unused = [torch.nn.Parameter(torch.tensor([1., 2.])) for _ in range(4)]
    state = torch.get_rng_state().clone()
    report = pcgrad_backward(
        {"flow": e.sum() + g.sum(), "transport": h.sum(), "image": -e.sum() + 2 * g.sum()},
        {"encoder": [e], "matching_head": [h], "generator": [g, unused]}, seed=9, step=2,
        routed_gradients={"code": {id(g): torch.tensor([1., 1.])}})
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert report["groups"]["encoder"]["active_tasks"] == ["flow", "image"]
    assert report["groups"]["matching_head"]["active_tasks"] == ["transport"]
    torch.testing.assert_close(e.grad, torch.zeros(2))
    torch.testing.assert_close(h.grad, torch.ones(2))
    torch.testing.assert_close(g.grad, torch.full((2,), 4.))
    assert unused.grad is None


def test_same_seed_step_reproduces_updates_without_external_rng_state():
    def run():
        p = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        vectors = torch.randn(4, 4, generator=rng(81))
        report = pcgrad_backward({str(i): p @ v for i, v in enumerate(vectors)},
                                 {"encoder": [p]}, seed=21, step=8)
        return p.grad, report
    first, report = run()
    torch.rand(20)
    second, other_report = run()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert report == other_report


def test_zero_losses_and_zero_gradients_are_safe():
    p = torch.nn.Parameter(torch.ones(3))
    report = pcgrad_backward({"off": torch.tensor(0.), "zero": (p * 0).sum()},
                             {"encoder": [p], "generator": []}, seed=1, step=1)
    torch.testing.assert_close(p.grad, torch.zeros_like(p))
    assert report["groups"]["encoder"]["active_tasks"] == []


@pytest.mark.parametrize("options", [{"enabled": 1}, {"eps": 0}, {"eps": float('nan')}, {"reduction": 'mean'}])
def test_invalid_config_rejected(options):
    with pytest.raises(ValueError, match="pcgrad"):
        PCGradConfig.from_mapping(options)


def test_nonfinite_gradients_and_overlapping_groups_rejected():
    with pytest.raises(FloatingPointError):
        project_task_gradients({"bad": (torch.tensor([float('nan')]),)}, generator=rng())
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="disjoint"):
        pcgrad_backward({"one": p.sum()}, {"encoder": [p], "generator": [p]}, seed=1, step=1)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_projection_accumulates_across_domain_devices():
    result, _ = project_task_gradients(
        {"one": (torch.tensor([1.], device='cuda:0'), torch.tensor([0.], device='cuda:1')),
         "two": (torch.tensor([-1.], device='cuda:0'), torch.tensor([1.], device='cuda:1'))}, generator=rng())
    assert float(result[0]) == pytest.approx(.5)
    assert float(result[1]) == pytest.approx(1.5)
