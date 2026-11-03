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


GITHUB_ANNOTATION_DIRS = [
    ("PSR/labels", None), ("ASR/annotations", "_front_asr.json"), ("TAS/groundTruth_TAS-S", "_front.txt"),
    ("PPR/groundTruth_PPR_L", "_front.txt"), ("PPR/groundTruth_PPR_R", "_front.txt"),
    ("ATR/groundTruth_ATR_L", "_front.npy"), ("ATR/groundTruth_ATR_R", "_front.npy"),
]
GITHUB_ANNOTATION_FILES = ["TAS/mapping_TAS-S.txt", "PPR/mapping_PPR.txt", "ATR/mapping_ATR.txt"]


def copy_github_annotations(dataset_dir: str, dest: str) -> int:
    """Copy the front-view annotations from a clone of the IMPACT GitHub repo's `dataset/` folder.

    Fallback when the release bundle's layout is not recognised. Returns the number of files copied.
    """
    import shutil

    src, dst = Path(dataset_dir), Path(dest)
    n = 0
    for rel, suffix in GITHUB_ANNOTATION_DIRS:
        for f in (src / rel).rglob("*"):
            if f.is_file() and (suffix is None or f.name.endswith(suffix)):
                out = dst / f.relative_to(src)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out)
                n += 1
    for rel in GITHUB_ANNOTATION_FILES:
        if (src / rel).exists():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, dst / rel)
            n += 1
    return n
