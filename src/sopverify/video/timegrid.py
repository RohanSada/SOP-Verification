"""Uniform time grid shared by frames, features and score traces."""

from __future__ import annotations

import numpy as np


def make_grid(duration_s: float, sample_fps: float) -> np.ndarray:
    """Times k / sample_fps for every k with time < duration (float64, seconds)."""
    if duration_s <= 0:
        return np.zeros(0, dtype=np.float64)
    n = int(np.floor(duration_s * sample_fps - 1e-9)) + 1
    return np.arange(n, dtype=np.float64) / sample_fps


def frame_indices(times_s: np.ndarray, fps: float, n_frames: int | None = None) -> np.ndarray:
    """Index of the source frame on screen at each time (floor), clipped to the video."""
    idx = np.floor(np.asarray(times_s, dtype=np.float64) * fps + 1e-6).astype(np.int64)
    idx = np.maximum(idx, 0)
    if n_frames is not None:
        idx = np.minimum(idx, n_frames - 1)
    return idx


def resample_nearest(src_times: np.ndarray, values: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resampling of `values` [N, ...] sampled at `src_times` onto `dst_times`."""
    src_times = np.asarray(src_times, dtype=np.float64)
    if len(src_times) == 0:
        raise ValueError("empty source")
    pos = np.searchsorted(src_times, dst_times)
    pos = np.clip(pos, 1, len(src_times) - 1) if len(src_times) > 1 else np.zeros(len(dst_times), dtype=int)
    if len(src_times) > 1:
        left = src_times[pos - 1]
        right = src_times[pos]
        pos = np.where(np.abs(dst_times - left) <= np.abs(right - dst_times), pos - 1, pos)
    return values[pos]


def step_hold(src_times: np.ndarray, values: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    """Hold the last known value (values at src_times) at each dst time; before the first, use the first."""
    pos = np.searchsorted(np.asarray(src_times), dst_times, side="right") - 1
    pos = np.clip(pos, 0, len(src_times) - 1)
    return values[pos]


def slice_until(times_s: np.ndarray, t_end_s: float) -> slice:
    """Slice selecting grid points with time < t_end."""
    return slice(0, int(np.searchsorted(times_s, t_end_s, side="left")))
