"""Package init: defines the `DatasetConfig` used to register loadable pretraining datasets."""

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["DatasetConfig"]


@dataclass
class DatasetConfig:
    """Registers a dataset by its Hugging Face path, a loader function, and a per-sample text-extraction function."""

    path: str
    loader: Callable
    sample_processor: Callable
