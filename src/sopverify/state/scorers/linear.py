"""Linear probe: class-balanced L2 logistic regression on standardized features, C by leave-one-demo-out."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sopverify.config import LabelsConfig, ScorerConfig
from sopverify.data.records import ExecutionRef
from sopverify.procedure.schema import Step
from sopverify.state.labels import build_design
from sopverify.state.scorers.base import BaseScorer
from sopverify.state.types import DemoData


class LinearProbeScorer(BaseScorer):
    name = "linear_probe"

    def __init__(self, cfg: ScorerConfig | None = None, labels: LabelsConfig | None = None):
        super().__init__()
        self.cfg = cfg or ScorerConfig(name="linear_probe")
        self.labels = labels or LabelsConfig()
        self.models: dict[str, object | None] = {}
        self.constant: dict[str, float] = {}

    def _make(self, C: float):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=self.cfg.linear_max_iter),
        )

    def _choose_C(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, dict]:
        if self.cfg.linear_C is not None:
            return self.cfg.linear_C, {"selection": "fixed"}
        uniq = np.unique(groups)
        if len(uniq) < 2:
            return 1.0, {"selection": "default (K<2)"}
        scores: dict[float, float] = {}
        for C in self.cfg.linear_C_grid:
            losses = []
            for g in uniq:
                tr, te = groups != g, groups == g
                if len(np.unique(y[tr])) < 2 or te.sum() == 0:
                    continue
                m = self._make(C).fit(X[tr], y[tr])
                p = np.clip(m.predict_proba(X[te])[:, 1], 1e-4, 1 - 1e-4)
                # Class-balanced log loss on the held-out demo.
                w = np.where(y[te] == 1, 0.5 / max(y[te].sum(), 1), 0.5 / max((y[te] == 0).sum(), 1))
                losses.append(log_loss(y[te], p, sample_weight=w, labels=[0, 1]))
            if losses:
                scores[C] = float(np.mean(losses))
        if not scores:
            return 1.0, {"selection": "default (no valid folds)"}
        best = min(scores, key=scores.get)  # type: ignore[arg-type]
        return best, {"selection": "leave-one-demo-out", "cv_logloss": scores}

    def fit(self, demos: list[DemoData], steps: list[Step]) -> None:
        self.step_ids = [s.id for s in steps]
        for sid, d in build_design(demos, steps, self.labels).items():
            if d.n_pos == 0 or d.n_neg == 0:
                self.models[sid] = None
                # No positives: cannot tell (0.5). No negatives: completed from the start.
                self.constant[sid] = 0.5 if d.n_pos == 0 else 1.0
                self.info[sid] = {"degenerate": True, "n_pos": d.n_pos, "n_neg": d.n_neg}
                continue
            C, sel = self._choose_C(d.X, d.y, d.groups)
            self.models[sid] = self._make(C).fit(d.X, d.y)
            self.info[sid] = {"C": C, **sel, "n_pos": d.n_pos, "n_neg": d.n_neg}

    def score(self, times_s: np.ndarray, feats: np.ndarray | None, execution: ExecutionRef) -> np.ndarray:
        self._check_fitted()
        assert feats is not None
        out = np.empty((len(feats), len(self.step_ids)), dtype=np.float32)
        for j, sid in enumerate(self.step_ids):
            m = self.models.get(sid)
            out[:, j] = self.constant.get(sid, 0.5) if m is None else m.predict_proba(feats)[:, 1]  # type: ignore[union-attr]
        return out
