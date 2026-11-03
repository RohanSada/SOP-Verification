"""Prototype scorer: sigmoid(alpha * (cos(x, after) - cos(x, before)) + beta), alpha/beta fit on demos."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from sopverify.config import LabelsConfig, ScorerConfig
from sopverify.data.records import ExecutionRef
from sopverify.procedure.schema import Step
from sopverify.state.labels import build_design
from sopverify.state.scorers.base import BaseScorer, l2n, sigmoid
from sopverify.state.types import DemoData


class PrototypeScorer(BaseScorer):
    name = "prototype"

    def __init__(self, cfg: ScorerConfig | None = None, labels: LabelsConfig | None = None):
        super().__init__()
        self.cfg = cfg or ScorerConfig(name="prototype")
        self.labels = labels or LabelsConfig()
        self.protos: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        self.calib: dict[str, tuple[float, float]] = {}

    @staticmethod
    def _protos(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if y.sum() == 0 or (y == 0).sum() == 0:
            return None
        Xn = l2n(X)
        return l2n(Xn[y == 1].mean(0)), l2n(Xn[y == 0].mean(0))

    @staticmethod
    def _margin(X: np.ndarray, protos: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        Xn = l2n(X)
        return Xn @ protos[0] - Xn @ protos[1]

    def fit(self, demos: list[DemoData], steps: list[Step]) -> None:
        self.step_ids = [s.id for s in steps]
        design = build_design(demos, steps, self.labels)
        for sid, d in design.items():
            protos = self._protos(d.X, d.y)
            self.protos[sid] = protos
            if protos is None:
                self.info[sid] = {"degenerate": True, "n_pos": d.n_pos, "n_neg": d.n_neg}
                continue
            # Calibrate on held-out-demo margins when K >= 2 (leave-one-demo-out), else in-sample.
            ms, ys = [], []
            groups = np.unique(d.groups)
            if len(groups) >= 2:
                for g in groups:
                    tr, te = d.groups != g, d.groups == g
                    p = self._protos(d.X[tr], d.y[tr])
                    if p is None or te.sum() == 0:
                        continue
                    ms.append(self._margin(d.X[te], p))
                    ys.append(d.y[te])
            if not ms:
                ms, ys = [self._margin(d.X, protos)], [d.y]
            m, yy = np.concatenate(ms), np.concatenate(ys)
            if len(np.unique(yy)) < 2:
                a, b = 20.0, 0.0
            else:
                lr = LogisticRegression(C=100.0, class_weight="balanced", fit_intercept=self.cfg.prototype_fit_bias)
                lr.fit(m[:, None], yy)
                a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0]) if self.cfg.prototype_fit_bias else 0.0
                if a <= 0:  # margins anti-correlated with labels: fall back to an uncalibrated slope
                    a, b = 20.0, 0.0
            self.calib[sid] = (a, b)
            self.info[sid] = {"alpha": a, "beta": b, "n_pos": d.n_pos, "n_neg": d.n_neg}

    def score(self, times_s: np.ndarray, feats: np.ndarray | None, execution: ExecutionRef) -> np.ndarray:
        self._check_fitted()
        assert feats is not None
        out = np.full((len(feats), len(self.step_ids)), 0.5, dtype=np.float32)
        for j, sid in enumerate(self.step_ids):
            p = self.protos.get(sid)
            if p is None:
                continue
            a, b = self.calib[sid]
            out[:, j] = sigmoid(a * self._margin(feats, p) + b)
        return out
