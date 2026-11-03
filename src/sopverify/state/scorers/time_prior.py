"""Progress-clock baseline: p_s(t) = 1 if t / duration >= mean normalized completion of s in demos."""

from __future__ import annotations

import numpy as np

from sopverify.data.records import ExecutionRef
from sopverify.procedure.schema import Step
from sopverify.state.scorers.base import BaseScorer
from sopverify.state.types import DemoData


class TimePriorScorer(BaseScorer):
    name = "time_prior"
    duration_dependent = True
    needs_features = False

    def __init__(self) -> None:
        super().__init__()
        self.threshold: dict[str, float] = {}

    def fit(self, demos: list[DemoData], steps: list[Step]) -> None:
        self.step_ids = [s.id for s in steps]
        for s in steps:
            vals = [d.completion_s[s.id] / d.duration_s for d in demos if d.completion_s.get(s.id) is not None]
            self.threshold[s.id] = float(np.mean(vals)) if vals else float("inf")
            self.info[s.id] = {"mean_norm_completion": self.threshold[s.id], "n": len(vals)}

    def score(self, times_s: np.ndarray, feats: np.ndarray | None, execution: ExecutionRef) -> np.ndarray:
        self._check_fitted()
        dur = execution.duration_s or (float(times_s[-1]) if len(times_s) else 1.0)
        prog = np.asarray(times_s) / max(dur, 1e-6)
        return np.stack([(prog >= self.threshold[s] - 1e-9).astype(np.float32) for s in self.step_ids], axis=1)
