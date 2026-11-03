"""Demo label builder (spec 8.5) and DemoData construction.

For each demo and step with GT completion c:
  positive: t >= c + pos_margin   (adaptive near the video end, decision D-006)
  negative: t <= c - neg_margin
  excluded: the transition band in between (often occluded by hands)
A step never completed in a demo: every frame is negative (logged).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sopverify.config import LabelsConfig
from sopverify.data.records import ExecutionRecord
from sopverify.logging_utils import get_logger
from sopverify.procedure.gt_mapping import completion_times
from sopverify.procedure.schema import Step
from sopverify.state.types import DemoData

log = get_logger(__name__)

POS, NEG, EXCL = 1, 0, -1


def frame_labels(
    times_s: np.ndarray, completion_s: float | None, duration_s: float, cfg: LabelsConfig
) -> np.ndarray:
    """int8 labels per grid time: 1 positive, 0 negative, -1 excluded."""
    times_s = np.asarray(times_s)
    if completion_s is None:
        return np.zeros(len(times_s), dtype=np.int8)
    lab = np.full(len(times_s), EXCL, dtype=np.int8)
    pos_from = completion_s + cfg.pos_margin_s
    if cfg.adaptive_tail:
        tail = duration_s - completion_s
        if tail < 2 * cfg.pos_margin_s:
            pos_from = completion_s + min(cfg.pos_margin_s, max(0.0, tail / 2))
    pos = times_s >= pos_from - 1e-9
    if cfg.adaptive_tail and not pos.any():
        after = np.flatnonzero(times_s >= completion_s - 1e-9)
        if len(after):
            pos[after[-1]] = True
    lab[pos] = POS
    lab[times_s <= completion_s - cfg.neg_margin_s + 1e-9] = NEG
    return lab


@dataclass
class StepDesign:
    step_id: str
    X: np.ndarray  # [N, D] labeled frames from all demos
    y: np.ndarray  # [N] 0/1
    groups: np.ndarray  # [N] demo index (for leave-one-demo-out)

    @property
    def n_pos(self) -> int:
        return int(self.y.sum())

    @property
    def n_neg(self) -> int:
        return int(len(self.y) - self.y.sum())


def build_design(demos: list[DemoData], steps: list[Step], cfg: LabelsConfig) -> dict[str, StepDesign]:
    out: dict[str, StepDesign] = {}
    for s in steps:
        Xs, ys, gs = [], [], []
        for di, d in enumerate(demos):
            if d.feats is None:
                raise ValueError(f"demo {d.execution_id} has no features")
            c = d.completion_s.get(s.id)
            if c is None:
                log.warning("step %s never completed in demo %s: all frames negative", s.id, d.execution_id)
            lab = frame_labels(d.times_s, c, d.duration_s, cfg)
            keep = lab != EXCL
            Xs.append(d.feats[keep])
            ys.append(lab[keep].astype(np.int64))
            gs.append(np.full(int(keep.sum()), di))
        X = np.concatenate(Xs) if Xs else np.zeros((0, 0), dtype=np.float32)
        out[s.id] = StepDesign(s.id, X, np.concatenate(ys), np.concatenate(gs))
        if out[s.id].n_pos == 0:
            log.warning("step %s has no positive demo frames", s.id)
    return out


def make_demo(record: ExecutionRecord, steps: list[Step], times_s: np.ndarray, feats: np.ndarray | None) -> DemoData:
    """DemoData from an annotated demo execution (the only place demo GT enters inference)."""
    return DemoData(
        execution_id=record.execution_id,
        times_s=np.asarray(times_s),
        feats=feats,
        completion_s=completion_times(record, steps),
        duration_s=record.duration_s,
        participant=record.participant,
        video_path=record.video_path,
        fps=record.fps,
    )
