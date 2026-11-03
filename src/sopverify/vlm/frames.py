"""ROI crops from videos at given times, as PIL images for VLM prompts."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from sopverify.video import roi as R


class VideoFrameSource:
    def __init__(self, roi_box: Sequence[int] | None, roi_frame_size: Sequence[int] = (1280, 720),
                 max_long_side: int = 768, cache_size: int = 512):
        self.roi_box = list(roi_box) if roi_box is not None else None
        self.roi_frame_size = list(roi_frame_size)
        self.max_long_side = max_long_side
        self._cache: dict[tuple[str, float], Image.Image] = {}
        self._cache_size = cache_size

    def crops(self, video_path: str, times_s: Sequence[float], fps: float | None = None,
              size: tuple[int, int] | None = None) -> list[Image.Image]:
        from sopverify.video.decode import read_frames_at

        keys = [(video_path, round(float(t), 2)) for t in times_s]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            frames = read_frames_at(video_path, [times_s[i] for i in missing], fps=fps)
            for i, fr in zip(missing, frames):
                box = self.roi_box
                h, w = fr.shape[:2]
                if box is not None and [w, h] != self.roi_frame_size:
                    box = R.scale_box(box, self.roi_frame_size, (w, h))
                img = R.resize_long_side(R.crop(fr, box), self.max_long_side)
                if len(self._cache) >= self._cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[keys[i]] = Image.fromarray(img)
        return [self._cache[k] for k in keys]
