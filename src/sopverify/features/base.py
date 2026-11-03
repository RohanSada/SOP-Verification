"""Feature extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from sopverify.data.records import ExecutionRef


class FeatureExtractor(ABC):
    """`extract(ref, times_s) -> feats [T, D]` on a given time grid."""

    name: str = "base"

    @abstractmethod
    def cache_key(self) -> dict[str, Any]:
        """Everything the output depends on besides the execution and the grid."""

    @abstractmethod
    def extract(self, ref: ExecutionRef, times_s: np.ndarray) -> np.ndarray: ...

    @property
    def dim(self) -> int | None:
        return None
