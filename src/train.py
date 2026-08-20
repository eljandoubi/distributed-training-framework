import contextlib
import os
from collections.abc import Callable
from datetime import timedelta

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record

from src.components.checkpoint import CheckpointManager  # type: ignore
from src.components.dataloader import BaseDataLoader  # type: ignore
from src.components.loss import LossFunction  # type: ignore
from src.components.lr_scheduler import LRSchedulersContainer  # type: ignore
from src.components.metrics import MetricsProcessor  # type: ignore
from src.components.optimizer import OptimizersContainer  # type: ignore
from src.components.tokenizer import DeepSeekV3Tokenizer  # type: ignore
from src.config import JobConfig
from src.config.job_config import Parallelism
from src.distributed import ParallelDims  # type: ignore
from src.tools import device_utils, utils  # type: ignore


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


    @record
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config

        device_module, device_type = (
            device_utils.device_module,
            device_utils.device_type,
        )
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        # init distributed and build meshes
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=job_config.comm.init_timeout_seconds),
        )
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = self._create_parallel_dims(
            parallelism_config, world_size
        )



    def _create_parallel_dims(
        self, parallelism_config: Parallelism, world_size: int
    ) -> ParallelDims:
        pass