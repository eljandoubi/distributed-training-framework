"""Shared test utilities for spawning small multi-process `torch.distributed` groups on CPU (gloo backend)."""

import os
import socket
from collections.abc import Callable

import torch.multiprocessing as mp


def _find_free_port() -> int:
    """Return an ephemeral TCP port that is currently free on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker_entrypoint(
    rank: int,
    world_size: int,
    port: int,
    worker_fn: Callable,
    worker_kwargs: dict,
) -> None:
    """Initialize a CPU/gloo process group for this rank, run `worker_fn`, then tear the group down."""
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        worker_fn(rank=rank, world_size=world_size, **worker_kwargs)
    finally:
        dist.destroy_process_group()


def run_distributed(worker_fn: Callable, world_size: int = 2, **worker_kwargs) -> None:
    """Run `worker_fn(rank, world_size, **worker_kwargs)` in `world_size` CPU processes using the gloo backend.

    Any assertion failure raised inside `worker_fn` on any rank propagates back to the
    calling test process (via `mp.spawn`'s exception propagation), failing the test.
    """
    port = _find_free_port()
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _worker_entrypoint,
        args=(world_size, port, worker_fn, worker_kwargs),
        nprocs=world_size,
        join=True,
    )
