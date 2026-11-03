"""Core containers for the state layer: demo data and score traces (spec 7.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sopverify.utils import atomic_save_npz


@dataclass
class DemoData:
    """One demonstration: features on the grid plus GT completion time per step (demo labels only)."""

    execution_id: str
    times_s: np.ndarray
    feats: np.ndarray | None
    completion_s: dict[str, float | None]
    duration_s: float
    participant: str | None = None
    video_path: str | None = None
    fps: float | None = None


@dataclass
class ScoreTrace:
    """Per-step probability that the step is complete at each grid time: scores [T, S] in [0, 1]."""

    execution_id: str
    times_s: np.ndarray
    scores: np.ndarray
    step_ids: list[str]
    scorer: str
    config_hash: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times_s = np.asarray(self.times_s, dtype=np.float64)
        self.scores = np.asarray(self.scores, dtype=np.float32)
        if self.scores.ndim != 2 or self.scores.shape != (len(self.times_s), len(self.step_ids)):
            raise ValueError(f"scores shape {self.scores.shape} != (T={len(self.times_s)}, S={len(self.step_ids)})")

    def column(self, step_id: str) -> np.ndarray:
        return self.scores[:, self.step_ids.index(step_id)]

    def truncate(self, t_end_s: float) -> ScoreTrace:
        n = int(np.searchsorted(self.times_s, t_end_s, side="left"))
        return ScoreTrace(self.execution_id, self.times_s[:n], self.scores[:n], list(self.step_ids),
                          self.scorer, self.config_hash, {**self.meta, "truncated_at_s": float(t_end_s)})

    def save(self, path: str | Path) -> Path:
        return atomic_save_npz(
            path, times_s=self.times_s, scores=self.scores, step_ids=np.array(self.step_ids),
            scorer=np.array(self.scorer), config_hash=np.array(self.config_hash),
            execution_id=np.array(self.execution_id),
        )

    @classmethod
    def load(cls, path: str | Path) -> ScoreTrace:
        with np.load(path, allow_pickle=False) as z:
            return cls(str(z["execution_id"]), z["times_s"], z["scores"], [str(s) for s in z["step_ids"]],
                       str(z["scorer"]), str(z["config_hash"]))
