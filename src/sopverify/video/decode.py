"""PyAV decoding: probe, decode onto a time grid, and read individual frames."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sopverify.logging_utils import get_logger
from sopverify.video.timegrid import frame_indices

log = get_logger(__name__)


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    duration_s: float
    width: int
    height: int


def probe(path: str | Path) -> VideoInfo:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 30.0)
        n = int(s.frames or 0)
        if s.duration is not None and s.time_base is not None:
            dur = float(s.duration * s.time_base)
        elif c.duration:
            dur = c.duration / 1e6
        else:
            dur = n / fps if n else 0.0
        if n == 0 and dur:
            n = int(round(dur * fps))
        if dur == 0 and n:
            dur = n / fps
        return VideoInfo(str(path), fps, n, dur, s.codec_context.width, s.codec_context.height)


def stage_to_scratch(path: str | Path, scratch: str | Path | None) -> Path:
    """Copy a video to local scratch before decoding (Drive I/O is slow); returns the local path."""
    path = Path(path)
    if scratch is None:
        return path
    dst = Path(scratch) / "videos" / path.name
    if dst.exists() and dst.stat().st_size == path.stat().st_size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(path, tmp)
    tmp.replace(dst)
    return dst


def decode_grid(
    path: str | Path,
    times_s: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    fps: float | None = None,
    threads: int = 4,
) -> np.ndarray:
    """Decode the frames on screen at `times_s` (sequential decode, no seeking).

    Frame k of the stream is taken to cover [k/fps, (k+1)/fps). `transform` maps an RGB
    uint8 frame [H, W, 3] to the stored array (crop + resize). Returns [T, ...].
    """
    import av

    times_s = np.asarray(times_s, dtype=np.float64)
    if len(times_s) == 0:
        return np.zeros((0,), dtype=np.uint8)
    out: list[np.ndarray | None] = [None] * len(times_s)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.codec_context.thread_count = threads
        vfps = fps or float(s.average_rate or 30.0)
        targets = frame_indices(times_s, vfps)
        order = np.argsort(targets, kind="stable")
        ti = 0
        last = None
        for i, frame in enumerate(c.decode(video=0)):
            idx = int(round(frame.time * vfps)) if frame.time is not None else i
            arr = None
            while ti < len(order) and targets[order[ti]] <= idx:
                if arr is None:
                    rgb = frame.to_ndarray(format="rgb24")
                    arr = transform(rgb) if transform else rgb
                out[order[ti]] = arr
                ti += 1
            if ti >= len(order):
                break
            last = frame
        if ti < len(order):
            # Grid points past the last decoded frame reuse it (container durations can over-report).
            if last is None:
                raise RuntimeError(f"no frames decoded from {path}")
            rgb = last.to_ndarray(format="rgb24")
            arr = transform(rgb) if transform else rgb
            missing = len(order) - ti
            if missing > 2:
                log.warning("%s: %d grid points beyond the last frame; padded with it", path, missing)
            while ti < len(order):
                out[order[ti]] = arr
                ti += 1
    return np.stack(out)  # type: ignore[arg-type]


def read_frames_at(path: str | Path, times_s: Sequence[float], fps: float | None = None) -> list[np.ndarray]:
    """Full-resolution RGB frames at arbitrary times (seeks; good for a handful of frames)."""
    import av

    results: dict[int, np.ndarray] = {}
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        vfps = fps or float(s.average_rate or 30.0)
        tb = s.time_base
        for j in np.argsort(times_s):
            t = max(0.0, float(times_s[j]))
            target = int(np.floor(t * vfps + 1e-6))
            c.seek(int(max(0.0, t - 1.0) / tb), stream=s, backward=True, any_frame=False)
            chosen = None
            for frame in c.decode(video=0):
                idx = int(round(frame.time * vfps)) if frame.time is not None else 0
                chosen = frame
                if idx >= target:
                    break
            if chosen is None:
                raise RuntimeError(f"could not decode a frame at {t:.2f}s from {path}")
            results[int(j)] = chosen.to_ndarray(format="rgb24")
    return [results[i] for i in range(len(times_s))]
