"""StateScorer interface (spec 8.6)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from sopverify.data.records import ExecutionRef
from sopverify.procedure.schema import Step
from sopverify.state.types import DemoData


@runtime_checkable
class StateScorer(Protocol):
    name: str
    # True when scores depend on the video's duration (time_prior, oracle): truncated
    # variants must be re-scored instead of sliced.
    duration_dependent: bool

    def fit(self, demos: list[DemoData], steps: list[Step]) -> None: ...

    def score(self, times_s: np.ndarray, feats: np.ndarray | None, execution: ExecutionRef) -> np.ndarray:
        """[T, S] in [0, 1], columns in the order of the steps passed to fit()."""
        ...


class BaseScorer:
    name = "base"
    duration_dependent = False
    needs_features = True

    def __init__(self) -> None:
        self.step_ids: list[str] = []
        self.info: dict[str, dict] = {}

    def _check_fitted(self) -> None:
        if not self.step_ids:
            raise RuntimeError(f"{self.name} scorer is not fitted")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
