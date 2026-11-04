"""DINOv2 frame features (CLS, optionally concatenated with mean patch tokens), L2-normalized."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from sopverify.data.records import ExecutionRef
from sopverify.features.base import FeatureExtractor
from sopverify.logging_utils import get_logger

log = get_logger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DINOv2Extractor(FeatureExtractor):
    name = "dinov2"

    def __init__(
        self,
        model_id: str = "facebook/dinov2-base",
        pooling: str = "cls",
        batch_size: int = 128,
        dtype: str = "float16",
        l2_normalize: bool = True,
        device: str | None = None,
        frames_fn: Callable[[ExecutionRef, np.ndarray], np.ndarray] | None = None,
        crop_key: dict[str, Any] | None = None,
    ):
        self.model_id = model_id
        self.pooling = pooling
        self.batch_size = batch_size
        self.dtype = dtype
        self.l2 = l2_normalize
        self.device = device
        self.frames_fn = frames_fn  # (ref, times) -> uint8 [T, S, S, 3], already cropped/resized
        self.crop_key = crop_key or {}
        self._model = None

    def cache_key(self) -> dict[str, Any]:
        return {"extractor": self.name, "model_id": self.model_id, "pooling": self.pooling,
                "l2": self.l2, **self.crop_key}

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel

            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = dev
            model = AutoModel.from_pretrained(self.model_id)
            model.eval().to(dev)
            self._model = model
            log.info("loaded %s on %s", self.model_id, dev)
        return self._model

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """uint8 [N, H, W, 3] (H, W multiples of 14 preferred) -> float32 [N, D]."""
        import torch

        model = self._load()
        use_amp = self.device == "cuda" and self.dtype != "float32"
        amp_dtype = torch.float16 if self.dtype == "float16" else torch.bfloat16
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        outs = []
        n_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
        with torch.inference_mode():
            for i in range(0, len(frames), self.batch_size):
                x = torch.from_numpy(np.array(frames[i : i + self.batch_size], copy=True)).to(self.device)
                x = x.permute(0, 3, 1, 2).float().div_(255.0)
                x = (x - mean) / std
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(pixel_values=x)
                hidden = out.last_hidden_state.float()
                cls = out.pooler_output.float() if getattr(out, "pooler_output", None) is not None else hidden[:, 0]
                if self.pooling == "cls_mean":
                    patches = hidden[:, 1 + n_reg :]
                    feat = torch.cat([cls, patches.mean(dim=1)], dim=1)
                else:
                    feat = cls
                if self.l2:
                    feat = torch.nn.functional.normalize(feat, dim=1)
                outs.append(feat.cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0), dtype=np.float32)

    def extract(self, ref: ExecutionRef, times_s: np.ndarray) -> np.ndarray:
        if self.frames_fn is None:
            raise RuntimeError("DINOv2Extractor needs a frames_fn (see features.provider)")
        frames = self.frames_fn(ref, times_s)
        return self.encode(np.asarray(frames))
