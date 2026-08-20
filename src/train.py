import contextlib
from collections.abc import Callable

import torch
from torch.distributed.checkpoint.stateful import Stateful

from src.components.checkpoint import CheckpointManager  # type: ignore
from src.components.dataloader import BaseDataLoader  # type: ignore
from src.components.loss import LossFunction  # type: ignore
from src.components.lr_scheduler import LRSchedulersContainer  # type: ignore
from src.components.metrics import MetricsProcessor  # type: ignore
from src.components.optimizer import OptimizersContainer  # type: ignore
from src.components.tokenizer import DeepSeekV3Tokenizer  # type: ignore
from src.config import JobConfig
from src.distributed import ParallelDims  # type: ignore
from src.tools import utils  # type: ignore


class Trainer(Stateful):
    job_config: JobConfig
    parallel_dims: ParallelDims

    tokenizer: DeepSeekV3Tokenizer
    dataloader: BaseDataLoader
    model_parts: list[torch.nn.Module]
    loss_fn: LossFunction
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: Callable[..., contextlib.AbstractContextManager]
    maybe_enable_amp: contextlib.AbstractContextManager
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    step: int
    ntokens_seen: int