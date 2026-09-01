"""Cross-entropy loss with token-based (sum) reduction, so callers can normalize by a global valid-token count."""

from collections.abc import Callable

import torch
from loguru import logger

from src.config import JobConfig

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

type LossFunction = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with sum reduction for token-based normalization."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def build_cross_entropy_loss(job_config: JobConfig):
    """Return the cross-entropy loss function, optionally compiled per `job_config.compile`."""
    loss_fn = cross_entropy_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)
    return loss_fn