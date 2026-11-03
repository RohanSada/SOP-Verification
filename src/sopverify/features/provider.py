"""Cached feature access: time grid -> frames (crop + resize) -> features, resumable by hash."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sopverify.config import Config
from sopverify.data.records import ExecutionRef
from sopverify.features.base import FeatureExtractor
from sopverify.features.cache import ArrayStore, FrameStore
from sopverify.logging_utils import get_logger
from sopverify.video import roi as R
from sopverify.video.timegrid import make_grid

log = get_logger(__name__)


class FeatureProvider:
    def __init__(
        self,
        cfg: Config,
        roi_box: Sequence[int] | None,
        roi_frame_size: Sequence[int] = (1280, 720),
        device: str | None = None,
        cache_root: str | Path | None = None,
        scratch: str | Path | None = None,
    ):
        self.cfg = cfg
        paths = cfg.paths_resolved()
        self.cache_root = Path(cache_root) if cache_root else paths["cache_root"]
        self.scratch = Path(scratch) if scratch else paths["scratch"]
        self.roi_box = list(roi_box) if roi_box is not None else None
        self.roi_frame_size = list(roi_frame_size)
        self.device = device
        self.features = ArrayStore(self.cache_root, "features")  # type: ignore[arg-type]
        frame_root = self.scratch if cfg.video.frame_cache_on_scratch else self.cache_root
        self.frames_store = FrameStore(frame_root)  # type: ignore[arg-type]
        self._extractor: FeatureExtractor | None = None

    # ------------------------------------------------------------------ keys
    @property
    def crop_box(self) -> list[int] | None:
        return self.roi_box if self.cfg.features.crop == "roi" else None

    def key_hash(self) -> str:
        return self.cfg.stage_hash("features", roi=self.crop_box)

    def frames_hash(self) -> str:
        return self.cfg.stage_hash("frames", roi=self.crop_box)

    def grid(self, ref: ExecutionRef) -> np.ndarray:
        if ref.duration_s is None:
            raise ValueError(f"{ref.execution_id}: duration unknown")
        return make_grid(ref.duration_s, self.cfg.video.sample_fps)

    # ------------------------------------------------------------------ frames
    def box_for(self, ref: ExecutionRef) -> list[int] | None:
        box = self.crop_box
        if box is None:
            return None
        if ref.width and ref.height and [ref.width, ref.height] != self.roi_frame_size:
            return R.scale_box(box, self.roi_frame_size, (ref.width, ref.height))
        return box

    def frames(self, ref: ExecutionRef, times_s: np.ndarray) -> np.ndarray:
        from sopverify.video.decode import decode_grid, stage_to_scratch

        h = self.frames_hash()
        if self.cfg.video.frame_cache and self.frames_store.has(h, ref.execution_id):
            arr = self.frames_store.load(h, ref.execution_id)
            if len(arr) == len(times_s):
                return arr
        if ref.video_path is None:
            raise FileNotFoundError(f"{ref.execution_id}: no video file")
        box = self.box_for(ref)
        size, mode = self.cfg.features.image_size, self.cfg.features.resize_mode

        def transform(rgb: np.ndarray) -> np.ndarray:
            return R.resize_square(R.crop(rgb, box), size, mode)

        local = stage_to_scratch(ref.video_path, self.scratch)
        arr = decode_grid(local, times_s, transform, fps=ref.fps, threads=self.cfg.video.decode_threads)
        if self.cfg.video.frame_cache:
            self.frames_store.save(h, ref.execution_id, arr)
        return arr

    # ------------------------------------------------------------------ features
    def extractor(self) -> FeatureExtractor:
        if self._extractor is None:
            f = self.cfg.features
            if f.extractor == "dinov2":
                from sopverify.features.dinov2 import DINOv2Extractor

                self._extractor = DINOv2Extractor(
                    model_id=f.model_id, pooling=f.pooling, batch_size=f.batch_size, dtype=f.dtype,
                    l2_normalize=f.l2_normalize, device=self.device, frames_fn=self.frames,
                    crop_key={"crop": f.crop, "roi": self.crop_box},
                )
            elif f.extractor == "shipped_videomaev2":
                from sopverify.features.shipped import ShippedFeatureExtractor

                self._extractor = ShippedFeatureExtractor(f.shipped_model, l2_normalize=f.l2_normalize)
            else:  # pragma: no cover - guarded by pydantic
                raise ValueError(f.extractor)
        return self._extractor

    def has(self, ref: ExecutionRef) -> bool:
        return self.features.has(self.key_hash(), ref.execution_id)

    def get(self, ref: ExecutionRef, compute: bool = True) -> tuple[np.ndarray, np.ndarray]:
        h = self.key_hash()
        if self.features.has(h, ref.execution_id):
            d = self.features.load(h, ref.execution_id)
            return d["times_s"], d["feats"]
        if not compute:
            raise FileNotFoundError(f"features for {ref.execution_id} not cached under {h}")
        times = self.grid(ref)
        feats = self.extractor().extract(ref, times)
        if len(feats) != len(times):
            raise RuntimeError(f"{ref.execution_id}: {len(feats)} features for {len(times)} grid points")
        self.features.save(h, ref.execution_id, times_s=times, feats=feats.astype(np.float32))
        return times, feats


class ArrayFeatureProvider:
    """Features held in memory (tests, synthetic data, uploads already processed)."""

    def __init__(self, data: dict[str, tuple[np.ndarray, np.ndarray]]):
        self.data = data

    def get(self, ref: ExecutionRef, compute: bool = True) -> tuple[np.ndarray, np.ndarray]:
        return self.data[ref.execution_id]

    def has(self, ref: ExecutionRef) -> bool:
        return ref.execution_id in self.data
