"""Small shared helpers: stable hashing, git info, JSON/YAML IO."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump(mode="json"))
    return obj


def stable_hash(obj: Any, length: int = 12) -> str:
    """Hash of a JSON-able object that does not depend on dict ordering."""
    payload = json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def git_commit(repo_dir: str | Path | None = None) -> str:
    """Current git commit (with '-dirty' if the tree has changes), or 'unknown'."""
    cwd = str(repo_dir) if repo_dir else str(Path(__file__).resolve().parent)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def write_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_to_jsonable(obj), indent=indent))
    tmp.replace(path)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def read_yaml(path: str | Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_to_jsonable(obj), f, sort_keys=False)
    return path


def atomic_save_npz(path: str | Path, **arrays: Any) -> Path:
    """Write an .npz atomically so an interrupted Colab session never leaves a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return path
