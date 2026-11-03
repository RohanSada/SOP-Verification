"""On-disk caches for frames, features and score traces, keyed by config/stage hashes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sopverify.utils import atomic_save_npz


class ArrayStore:
    """`<root>/<kind>/<hash>/<execution_id>.npz` holding named arrays."""

    def __init__(self, root: str | Path, kind: str):
        self.root = Path(root) / kind

    def path(self, key_hash: str, execution_id: str) -> Path:
        return self.root / key_hash / f"{execution_id}.npz"

    def has(self, key_hash: str, execution_id: str) -> bool:
        return self.path(key_hash, execution_id).exists()

    def load(self, key_hash: str, execution_id: str) -> dict[str, np.ndarray]:
        with np.load(self.path(key_hash, execution_id), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def save(self, key_hash: str, execution_id: str, **arrays: np.ndarray) -> Path:
        return atomic_save_npz(self.path(key_hash, execution_id), **arrays)


class FrameStore:
    """Per-video uint8 frame arrays `[T, H, W, 3]` stored as `.npy` (memory-mappable)."""

    def __init__(self, root: str | Path):
        self.root = Path(root) / "frames"

    def path(self, key_hash: str, execution_id: str) -> Path:
        return self.root / key_hash / f"{execution_id}.npy"

    def has(self, key_hash: str, execution_id: str) -> bool:
        return self.path(key_hash, execution_id).exists()

    def load(self, key_hash: str, execution_id: str) -> np.ndarray:
        return np.load(self.path(key_hash, execution_id), mmap_mode="r")

    def save(self, key_hash: str, execution_id: str, frames: np.ndarray) -> Path:
        p = self.path(key_hash, execution_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp.npy")
        np.save(tmp, frames)
        tmp.replace(p)
        return p
