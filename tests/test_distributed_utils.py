"""CPU/gloo-backend tests for `src/distributed/utils.py` reduction and gradient-clipping helpers."""

import torch
from torch.distributed.device_mesh import init_device_mesh

from src.distributed import utils as dist_utils
from tests.conftest import run_distributed


def _worker_dist_reduce(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("world",))

    x = torch.tensor(float(rank + 1))
    expected_sum = float(sum(range(1, world_size + 1)))

    total = dist_utils.dist_sum(x.clone(), mesh)
    assert total == expected_sum, (total, expected_sum)

    maximum = dist_utils.dist_max(x.clone(), mesh)
    assert maximum == float(world_size), maximum

    average = dist_utils.dist_mean(x.clone(), mesh)
    assert abs(average - expected_sum / world_size) < 1e-6, average


def test_dist_sum_max_mean_over_gloo():
    """`dist_sum`/`dist_max`/`dist_mean` should reduce correctly across ranks over a CPU gloo mesh."""
    run_distributed(_worker_dist_reduce, world_size=2)


def _worker_clip_grad_norm(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("pp",))

    # Rank 0 "owns" a param whose grad has norm 5 (3-4-5 triangle); rank 1 owns one with
    # norm 12. Combined (as if these were two Pipeline Parallel stages of the same model),
    # the total L2 norm is sqrt(5**2 + 12**2) == 13.
    if rank == 0:
        grad = torch.tensor([3.0, 4.0])
    else:
        grad = torch.tensor([0.0, 12.0])
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.grad = grad.clone()

    total_norm = dist_utils.clip_grad_norm_(
        [param], max_norm=13.0, pp_mesh=mesh, ep_enabled=False
    )
    assert abs(total_norm.item() - 13.0) < 1e-4, total_norm.item()

    # max_norm == total_norm, so grads should be (approximately) unchanged.
    assert torch.allclose(param.grad, grad, atol=1e-4)


def test_clip_grad_norm_across_pp_mesh_over_gloo():
    """`clip_grad_norm_` should combine per-rank local gradient norms across a CPU gloo `pp_mesh`."""
    run_distributed(_worker_clip_grad_norm, world_size=2)
