"""kNN scorer: fraction of positive labels among the k nearest labeled demo frames (cosine)."""

from __future__ import annotations

import numpy as np

from sopverify.config import LabelsConfig, ScorerConfig
from sopverify.data.records import ExecutionRef
from sopverify.procedure.schema import Step
from sopverify.state.labels import EXCL, frame_labels
from sopverify.state.scorers.base import BaseScorer, l2n
from sopverify.state.types import DemoData


class KNNScorer(BaseScorer):
    name = "knn"

    def __init__(self, cfg: ScorerConfig | None = None, labels: LabelsConfig | None = None):
        super().__init__()
        self.cfg = cfg or ScorerConfig(name="knn")
        self.labels = labels or LabelsConfig()
        self.bank: np.ndarray | None = None  # [N, D] all demo frames
        self.lab: np.ndarray | None = None  # [N, S] int8 labels (-1 excluded)

    def fit(self, demos: list[DemoData], steps: list[Step]) -> None:
        self.step_ids = [s.id for s in steps]
        feats, labs = [], []
        for d in demos:
            assert d.feats is not None
            feats.append(l2n(np.asarray(d.feats, dtype=np.float32)))
            labs.append(np.stack([frame_labels(d.times_s, d.completion_s.get(s.id), d.duration_s, self.labels)
                                  for s in steps], axis=1))
        self.bank = np.concatenate(feats)
        self.lab = np.concatenate(labs)
        for j, sid in enumerate(self.step_ids):
            col = self.lab[:, j]
            self.info[sid] = {"k": self.cfg.knn_k, "n_pos": int((col == 1).sum()), "n_neg": int((col == 0).sum())}

    def score(self, times_s: np.ndarray, feats: np.ndarray | None, execution: ExecutionRef) -> np.ndarray:
        self._check_fitted()
        assert feats is not None and self.bank is not None and self.lab is not None
        sim = l2n(np.asarray(feats, dtype=np.float32)) @ self.bank.T  # [T, N]
        out = np.full((len(feats), len(self.step_ids)), 0.5, dtype=np.float32)
        for j in range(len(self.step_ids)):
            col = self.lab[:, j]
            valid = np.flatnonzero(col != EXCL)
            if len(valid) == 0 or (col[valid] == 1).sum() == 0:
                continue
            k = min(self.cfg.knn_k, len(valid))
            s = sim[:, valid]
            nn = np.argpartition(-s, k - 1, axis=1)[:, :k]
            out[:, j] = (col[valid][nn] == 1).mean(axis=1)
        return out
