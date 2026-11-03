"""Helpers for the Colab notebooks: environment variables and Drive folders."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = "/content/drive/MyDrive/sopverify"


def setup_env(root: str = DEFAULT_ROOT, scratch: str = "/content/scratch", annotations: str | None = None) -> dict[str, str]:
    """Set SOPVERIFY_* variables (spec 4.2) and create the Drive/scratch folders."""
    env = {
        "SOPVERIFY_DATA_ROOT": f"{root}/data/IMPACT-v1.1",
        "SOPVERIFY_CACHE": f"{root}/cache",
        "SOPVERIFY_RUNS": f"{root}/runs",
        "SOPVERIFY_EXPORTS": f"{root}/exports",
        "SOPVERIFY_SCRATCH": scratch,
    }
    if annotations:
        env["SOPVERIFY_ANNOTATIONS"] = annotations
    os.environ.update(env)
    for k, v in env.items():
        if k != "SOPVERIFY_ANNOTATIONS":
            Path(v).mkdir(parents=True, exist_ok=True)
    return env


def show(img, title: str | None = None, width: int = 12) -> None:
    """Display an RGB array inline."""
    import matplotlib.pyplot as plt

    h, w = img.shape[:2]
    plt.figure(figsize=(width, width * h / w))
    plt.imshow(img)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()
