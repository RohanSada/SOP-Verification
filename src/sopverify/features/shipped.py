"""IMPACT's released VideoMAEv2 features, resampled onto our time grid.

Per the IMPACT repo (docs/FEATURES.md, tools/features/extract_videomaev2.py) the VideoMAEv2-G
features are frame-aligned: one 1408-D vector per video frame (16-frame clip centred on it),
stored as `1408 x T`. Feature index k therefore corresponds to frame k, i.e. time k / fps.
This is verified per file in M3 (T must equal the frame count; see DATA_NOTES 5.2 #6).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sopverify.data.records import ExecutionRef
from sopverify.features.base import FeatureExtractor
from sopverify.logging_utils import get_logger
from sopverify.video.timegrid import frame_indices

log = get_logger(__name__)

FEATURE_DIMS = {"VideoMAEv2": 1408, "I3D": 1024, "MViTv2": 768}


def load_feature_array(path: str, expected_dim: int | None = None) -> np.ndarray:
    """Load a shipped feature file and return it as [T, D] float32."""
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            arr = z[z.files[0]]
    else:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D feature array, got shape {arr.shape}")
    if expected_dim is not None:
        if arr.shape[0] == expected_dim and arr.shape[1] != expected_dim:
            arr = arr.T
        elif arr.shape[1] != expected_dim:
            raise ValueError(f"{path}: neither axis has the expected dim {expected_dim}: {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


class ShippedFeatureExtractor(FeatureExtractor):
    name = "shipped_videomaev2"

    def __init__(self, model: str = "VideoMAEv2", l2_normalize: bool = True, frame_tolerance: int = 32):
        self.model = model
        self.l2 = l2_normalize
        self.tol = frame_tolerance

    def cache_key(self) -> dict[str, Any]:
        return {"extractor": self.name, "model": self.model, "l2": self.l2}

    @property
    def dim(self) -> int | None:
        return FEATURE_DIMS.get(self.model)

    def extract(self, ref: ExecutionRef, times_s: np.ndarray) -> np.ndarray:
        path = ref.shipped_features.get(self.model)
        if not path:
            raise FileNotFoundError(f"no shipped {self.model} features for {ref.execution_id}")
        feats = load_feature_array(path, FEATURE_DIMS.get(self.model))
        if ref.fps is None:
            raise ValueError(f"{ref.execution_id}: fps unknown; cannot map feature index to time")
        T = feats.shape[0]
        if ref.n_frames is not None and abs(T - ref.n_frames) > self.tol:
            log.warning("%s: %d feature rows vs %d video frames; mapping assumes one row per frame",
                        ref.execution_id, T, ref.n_frames)
        idx = frame_indices(times_s, ref.fps, T)
        out = feats[idx]
        if self.l2:
            out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
        return out.astype(np.float32)
