"""Factory for the VLM state scorer (implementation in sopverify.vlm.scorer_impl)."""

from __future__ import annotations

from sopverify.config import Config


def make_vlm_scorer(cfg: Config, backend=None, frames=None, log_path=None, roi_box=None, roi_frame_size=(1280, 720)):
    from sopverify.vlm.backend import make_backend
    from sopverify.vlm.frames import VideoFrameSource
    from sopverify.vlm.scorer_impl import VLMStateScorer

    backend = backend or make_backend(cfg.vlm)
    frames = frames or VideoFrameSource(roi_box, roi_frame_size, cfg.vlm.max_long_side)
    return VLMStateScorer(cfg.vlm, backend, frames, cfg.labels, log_path)
