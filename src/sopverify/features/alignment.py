"""Check that two feature streams on the same time grid are temporally aligned (M3 acceptance).

Method: compute each stream's change signal ||f_t - f_{t-1}|| (scene activity), z-score it,
and find the lag that maximises their cross-correlation. Aligned streams peak at lag ~0.
"""

from __future__ import annotations

import numpy as np


def change_signal(feats: np.ndarray) -> np.ndarray:
    f = np.asarray(feats, dtype=np.float64)
    d = np.linalg.norm(np.diff(f, axis=0), axis=1)
    return (d - d.mean()) / (d.std() + 1e-9)


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, np.ndarray]:
    """Lag (in samples) to shift `b` to best match `a`, and the correlation for lags -max..max."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            x, y = a[lag:], b[: n - lag]
        else:
            x, y = a[: n + lag], b[-lag:]
        corr[i] = float(np.mean(x * y)) if len(x) > 2 else np.nan
    return int(lags[int(np.nanargmax(corr))]), corr


def alignment_report(feats_a: np.ndarray, feats_b: np.ndarray, sample_fps: float, max_lag_s: float = 5.0) -> dict:
    lag, corr = best_lag(change_signal(feats_a), change_signal(feats_b), int(round(max_lag_s * sample_fps)))
    return {"best_lag_samples": lag, "best_lag_s": lag / sample_fps, "peak_corr": float(np.nanmax(corr)),
            "corr_at_zero": float(corr[len(corr) // 2])}
