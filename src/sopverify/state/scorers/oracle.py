"""Oracle scorer (testing only): traces built from ground truth to validate checker and metrics.

GT is injected as a function by `sopverify.eval` so this module never reads annotations.
`pipeline.run` refuses it unless `allow_oracle=True`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sopverify.data.records import ExecutionRef
from sopverify.procedure.schema import Step
from sopverify.state.scorers.base import BaseScorer
from sopverify.state.types import DemoData

# ref -> {step_id: (completion_s or None, done_at_end)}
GTFn = Callable[[ExecutionRef], dict[str, tuple[float | None, bool]]]


class OracleScorer(BaseScorer):
    name = "oracle"
    duration_dependent = True
    needs_features = False
    is_oracle = True

    def __init__(self, gt_fn: GTFn):
        super().__init__()
        self.gt_fn = gt_fn

    def fit(self, demos: list[DemoData], steps: list[Step]) -> None:
        self.step_ids = [s.id for s in steps]

    def score(self, times_s: np.ndarray, feats: np.ndarray | None, execution: ExecutionRef) -> np.ndarray:
        self._check_fitted()
        gt = self.gt_fn(execution)
        out = np.zeros((len(times_s), len(self.step_ids)), dtype=np.float32)
        for j, sid in enumerate(self.step_ids):
            c, done = gt.get(sid, (None, False))
            if done and c is not None:
                out[np.asarray(times_s) >= c - 1e-9, j] = 1.0
        return out
