"""Synthetic "not done" negatives by truncation (spec 9.4, decision D-008).

A cut is placed between GT completions so that exactly m required steps that are complete in
the full video are not yet complete at the cut. Cuts keep `min_gap_s` away from every GT
completion so the truncated ground truth is unambiguous. Implemented by slicing cached
features/traces; nothing is re-encoded.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from sopverify.config import TruncationConfig
from sopverify.eval.ground_truth import ExecutionGT


@dataclass
class Truncation:
    execution_id: str
    index: int
    cut_s: float
    m: int

    @property
    def variant(self) -> str:
        return f"trunc{self.index}_m{self.m}"


def make_truncations(gt: ExecutionGT, cfg: TruncationConfig, min_duration_s: float = 10.0) -> list[Truncation]:
    if not cfg.enabled or cfg.n_per_video <= 0:
        return []
    comps = sorted(
        s.completion_s for s in gt.steps.values()
        if s.required and s.visually_verifiable and s.done and s.completion_s is not None
    )
    all_comps = sorted(s.completion_s for s in gt.steps.values() if s.completion_s is not None)
    n = len(comps)
    out: list[Truncation] = []
    rng = np.random.default_rng([cfg.seed, zlib.crc32(gt.execution_id.encode())])
    ms = [cfg.m_choices[j % len(cfg.m_choices)] for j in range(cfg.n_per_video)]
    for j, m in enumerate(ms):
        if m < 1 or m > n:
            continue
        lo = comps[n - m - 1] if n - m - 1 >= 0 else 0.0
        hi = comps[n - m]
        lo, hi = max(lo + cfg.min_gap_s, min_duration_s), hi - cfg.min_gap_s
        # Keep away from every completion (including non-required steps) as well.
        if hi <= lo:
            continue
        for _ in range(20):
            cut = float(rng.uniform(lo, hi))
            if all(abs(cut - c) >= cfg.min_gap_s for c in all_comps):
                out.append(Truncation(gt.execution_id, j, round(cut, 3), m))
                break
    return out
