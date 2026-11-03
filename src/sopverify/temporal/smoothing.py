"""Per-step temporal smoothing of score traces (spec 8.7).

Applied independently per step; the prerequisite graph is NOT used here so that order
violations remain detectable by the checker.

1. Median filter (window `median_window_s`, edges replicated).
2. Hysteresis: OFF -> ON at >= tau_on, ON -> OFF below tau_off.
3. ON runs shorter than `min_on_s` are dropped, except a run that reaches the end of the
   video, which needs only `min_on_end_s` (decision D-006: the last step often completes
   seconds before the recording stops).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

from sopverify.config import SmoothingConfig
from sopverify.state.types import ScoreTrace


@dataclass
class StepEstimate:
    step_id: str
    final_on: bool
    completion_time_s: float | None
    confidence: float
    max_score: float
    max_score_time_s: float | None
    on_runs: list[tuple[float, float]] = field(default_factory=list)  # (start_s, end_s)
    possible_rework: bool = False
    smoothed: np.ndarray | None = None

    @property
    def flags(self) -> list[str]:
        return ["possible_rework"] if self.possible_rework else []


def _dt(times_s: np.ndarray) -> float:
    if len(times_s) >= 2:
        return float(np.median(np.diff(times_s)))
    return 1.0


def median_smooth(x: np.ndarray, window_s: float, dt: float) -> np.ndarray:
    n = int(round(window_s / dt)) if dt > 0 else 1
    if n <= 1 or len(x) == 0:
        return np.asarray(x, dtype=np.float32).copy()
    if n % 2 == 0:
        n += 1
    return median_filter(np.asarray(x, dtype=np.float32), size=n, mode="nearest")


def hysteresis(x: np.ndarray, tau_on: float, tau_off: float) -> np.ndarray:
    if tau_off > tau_on:
        raise ValueError("tau_off must be <= tau_on")
    on = np.zeros(len(x), dtype=bool)
    state = False
    for i, v in enumerate(x):
        if not state and v >= tau_on:
            state = True
        elif state and v < tau_off:
            state = False
        on[i] = state
    return on


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of True runs."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def estimate_step(step_id: str, times_s: np.ndarray, scores: np.ndarray, cfg: SmoothingConfig) -> StepEstimate:
    times_s = np.asarray(times_s, dtype=np.float64)
    T = len(times_s)
    if T == 0:
        return StepEstimate(step_id, False, None, 0.0, 0.0, None, [], False, np.zeros(0, np.float32))
    dt = _dt(times_s)
    sm = median_smooth(scores, cfg.median_window_s, dt)
    raw_runs = runs(hysteresis(sm, cfg.tau_on, cfg.tau_off))
    kept: list[tuple[int, int]] = []
    for a, b in raw_runs:
        dur = (b - a + 1) * dt
        at_end = b == T - 1
        need = min(cfg.min_on_s, cfg.min_on_end_s) if at_end else cfg.min_on_s
        if dur + 1e-9 >= need:
            kept.append((a, b))
    final_on = bool(kept) and kept[-1][1] == T - 1
    imax = int(np.argmax(sm))
    max_score = float(sm[imax])
    if final_on:
        a, b = kept[-1]
        completion = float(times_s[a])
        confidence = float(sm[a : b + 1].mean())
    else:
        completion = None
        if kept:
            tail = sm[kept[-1][1] + 1 :]
            confidence = float(1.0 - (tail.max() if len(tail) else max_score))
        else:
            confidence = float(1.0 - max_score)
    rework = len(kept) > 1 or (bool(kept) and not final_on)
    return StepEstimate(
        step_id=step_id,
        final_on=final_on,
        completion_time_s=completion,
        confidence=confidence,
        max_score=max_score,
        max_score_time_s=float(times_s[imax]),
        on_runs=[(float(times_s[a]), float(times_s[b]) + dt) for a, b in kept],
        possible_rework=rework,
        smoothed=sm,
    )


def smooth_trace(trace: ScoreTrace, cfg: SmoothingConfig) -> dict[str, StepEstimate]:
    return {sid: estimate_step(sid, trace.times_s, trace.scores[:, j], cfg) for j, sid in enumerate(trace.step_ids)}


def evidence_time(est: StepEstimate, duration_s: float, cfg: SmoothingConfig) -> float:
    """Frame time to show as evidence: after completion if done, else the time of max score."""
    if est.completion_time_s is not None:
        t = est.completion_time_s + cfg.evidence_offset_s
    elif est.max_score_time_s is not None:
        t = est.max_score_time_s
    else:
        t = duration_s
    return float(min(max(t, 0.0), max(duration_s - 0.05, 0.0)))
