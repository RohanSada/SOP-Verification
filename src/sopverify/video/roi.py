"""Static workspace crop: cropping, resizing to encoder input and overlay previews."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def clip_box(box: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {list(box)} lies outside a {width}x{height} frame")
    return x0, y0, x1 - x0, y1 - y0


def scale_box(box: Sequence[int], from_size: Sequence[int], to_size: Sequence[int]) -> list[int]:
    """Rescale [x, y, w, h] from a (W, H) frame to another (W, H) frame."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    x, y, w, h = box
    return [int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy))]


def crop(frame: np.ndarray, box: Sequence[int] | None) -> np.ndarray:
    if box is None:
        return frame
    h, w = frame.shape[:2]
    x, y, bw, bh = clip_box(box, w, h)
    return frame[y : y + bh, x : x + bw]


def resize_square(img: np.ndarray, size: int, mode: str = "letterbox", pad_value: int = 0) -> np.ndarray:
    """Resize to size x size. 'letterbox' keeps aspect ratio and pads; 'squash' stretches."""
    h, w = img.shape[:2]
    if mode == "squash":
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, img.shape[2]), pad_value, dtype=img.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def resize_long_side(img: np.ndarray, max_long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_long_side:
        return img
    s = max_long_side / max(h, w)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def overlay_box(frame: np.ndarray, box: Sequence[int], color=(0, 255, 0), thickness: int = 4,
                label: str | None = None) -> np.ndarray:
    """Copy of an RGB frame with the box drawn, plus a coordinate ruler, for ROI confirmation."""
    out = np.ascontiguousarray(frame.copy())
    x, y, w, h = (int(v) for v in box)
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    H, W = out.shape[:2]
    for gx in range(0, W, 160):
        cv2.line(out, (gx, 0), (gx, 10), (255, 255, 0), 2)
        cv2.putText(out, str(gx), (gx + 3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    for gy in range(0, H, 120):
        cv2.line(out, (0, gy), (10, gy), (255, 255, 0), 2)
        cv2.putText(out, str(gy), (14, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    text = label or f"ROI [{x}, {y}, {w}, {h}]"
    cv2.putText(out, text, (x + 6, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out
