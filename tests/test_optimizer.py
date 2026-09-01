"""Unit tests for `src/components/optimizer.py`: multi-part optimizer container (single process, no distributed group needed)."""

import torch

from src.components.optimizer import OptimizersContainer, build_optimizers
from src.config import Optimizer as OptimizerConfig
from src.distributed import ParallelDims


def test_optimizers_container_one_optimizer_per_model_part():
    """`OptimizersContainer` should create exactly one optimizer per model part."""
    model_parts = [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
    container = OptimizersContainer(model_parts, torch.optim.SGD, {"lr": 0.1})
    assert len(container) == 2
    assert len(list(iter(container))) == 2


def test_optimizers_container_step_updates_parameters():
    """Calling `step()` after a backward pass should change every model part's parameters."""
    torch.manual_seed(0)
    model_parts = [torch.nn.Linear(4, 4)]
    container = OptimizersContainer(model_parts, torch.optim.SGD, {"lr": 0.1})

    before = model_parts[0].weight.detach().clone()
    x = torch.randn(2, 4)
    loss = model_parts[0](x).sum()
    loss.backward()
    container.step()

    assert not torch.allclose(model_parts[0].weight.detach(), before)


def test_optimizers_container_zero_grad_clears_all_parts():
    """`zero_grad()` should clear gradients across every wrapped optimizer/model part."""
    model_parts = [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
    container = OptimizersContainer(model_parts, torch.optim.SGD, {"lr": 0.1})

    for model in model_parts:
        x = torch.randn(2, 4)
        model(x).sum().backward()
        assert model.weight.grad is not None

    container.zero_grad()
    for model in model_parts:
        assert model.weight.grad is None or torch.all(model.weight.grad == 0)


def test_build_optimizers_rejects_unknown_optimizer_name():
    """`build_optimizers` should raise for an optimizer name that isn't registered."""
    model_parts = [torch.nn.Linear(4, 4)]
    config = OptimizerConfig(name="NotARealOptimizer", implementation="for-loop")
    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, etp=1, world_size=1
    )
    try:
        build_optimizers(model_parts, config, parallel_dims)
    except NotImplementedError:
        pass
    else:
        raise AssertionError(
            "Expected NotImplementedError for an unregistered optimizer name"
        )
