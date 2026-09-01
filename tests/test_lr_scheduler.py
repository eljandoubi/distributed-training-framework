"""Unit tests for `src/components/lr_scheduler.py`: warmup-stable-decay LR schedule construction."""

import torch

from src.components.lr_scheduler import build_lr_schedulers
from src.components.optimizer import OptimizersContainer
from src.config import LRScheduler as LRSchedulerConfig


def _build_optimizers() -> OptimizersContainer:
    model = torch.nn.Linear(4, 4)
    return OptimizersContainer([model], torch.optim.SGD, {"lr": 1.0})


def test_lr_scheduler_warmup_then_decay_to_min_factor():
    """LR should ramp up linearly during warmup, then decay down to (at least) `min_lr_factor` by the last step."""
    optimizers = _build_optimizers()
    config = LRSchedulerConfig(
        warmup_steps=4, decay_ratio=None, decay_type="linear", min_lr_factor=0.1
    )
    training_steps = 10
    schedulers = build_lr_schedulers(optimizers, config, training_steps)

    lrs = []
    for _ in range(training_steps):
        lrs.append(schedulers.schedulers[0].get_last_lr()[0])
        schedulers.step()

    # Warmup: LR should strictly increase for the first `warmup_steps` steps.
    for i in range(1, config.warmup_steps):
        assert lrs[i] > lrs[i - 1]

    # After warmup, LR should never exceed the base LR (1.0) and should trend downward.
    assert max(lrs) <= 1.0 + 1e-6
    assert lrs[-1] < lrs[config.warmup_steps]


def test_lr_scheduler_warmup_capped_to_training_steps():
    """If `warmup_steps` exceeds `training_steps`, it should be silently capped rather than erroring."""
    optimizers = _build_optimizers()
    config = LRSchedulerConfig(warmup_steps=100, decay_ratio=None, min_lr_factor=0.0)
    # Should not raise despite warmup_steps > training_steps.
    build_lr_schedulers(optimizers, config, training_steps=5)


def test_lr_scheduler_state_dict_round_trip():
    """Saving and loading a scheduler's state dict should restore its `last_epoch` (step count)."""
    optimizers = _build_optimizers()
    config = LRSchedulerConfig(warmup_steps=2, min_lr_factor=0.0)
    schedulers = build_lr_schedulers(optimizers, config, training_steps=10)

    for _ in range(3):
        schedulers.step()
    state = schedulers.state_dict()

    fresh_optimizers = _build_optimizers()
    fresh_schedulers = build_lr_schedulers(fresh_optimizers, config, training_steps=10)
    fresh_schedulers.load_state_dict(state)

    assert (
        fresh_schedulers.schedulers[0].last_epoch
        == schedulers.schedulers[0].last_epoch
    )
