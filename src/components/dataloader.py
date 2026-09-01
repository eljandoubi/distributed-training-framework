"""Stateful, data-parallel-aware dataloader base classes for checkpoint-resumable training."""

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any

from loguru import logger
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader


class DataloaderExhaustedError(Exception):
    """Raised when the dataloader runs out of data mid-gradient-accumulation, signaling the current step should be aborted."""


class BaseDataLoader(Stateful, ABC):
    """Abstract base for dataloaders that are both iterable and checkpoint-stateful."""

    @abstractmethod
    def __iter__(self) -> Iterator: ...


class ParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """A `StatefulDataLoader` that saves/restores its state per data-parallel rank so training can resume exactly."""

    dp_rank: int
    dp_world_size: int
    batch_size: int

    def __init__(
        self,
        dataset: IterableDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        collate_fn: Callable | None = None,
    ):
        """Wrap `dataset` in a `StatefulDataLoader` scoped to this data-parallel rank."""
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        super().__init__(dataset, batch_size, collate_fn=collate_fn)
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> dict[str, Any]:
        """Return this rank's dataloader state, keyed by rank id, plus the world size for consistency checks on load."""
        # Store state only for dp rank to avoid replicating the same state across other dimensions.
        return {
            # We don't have to use pickle as DCP will serialize the state_dict. However, we have to keep this for backward compatibility.
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore this rank's dataloader state, validating the checkpoint's world size matches the current run."""
        # State being empty is valid.
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                f"expected key {self._rank_id}"
            )
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        # We don't have to use pickle as DCP will serialize the state_dict. However, we have to keep this for backward compatibility.
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))
