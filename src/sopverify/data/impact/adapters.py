"""Readers for the IMPACT v1.1 annotation files, written against the real files.

Formats (verified on the GitHub-released annotations, see docs/DATA_NOTES.md):

* TAS-S, PPR-L/R: text, one label per video frame (30 fps), e.g. `INSTALL_ROTOR_ASSEMBLY`.
* PSR: `PSR_labels.csv` rows `<frame>.jpg,<category_id>,<description>`, e.g.
  `982.jpg,3,Install gearbox_housing`. Frames index the front video at its fps.
* ASR: JSON with `fps`, `frame_count`, `components[]` and `state_sequence[]` of
  `{frame, state[17]}` keyframes; states are -1/0/1.
* ATR-L/R: `.npy` uint8 arrays `[T, 6]` (one column per anomaly type).

Everything is converted to seconds here and nowhere else.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from sopverify.data.records import AnomalyTypeSpan, ASRStates, Phase, PSREvent, Segment

# ------------------------------------------------------------------ identifiers
EXECUTION_RE = re.compile(
    r"(?P<participant>[A-Z]{2}\d{2}[A-Z]{2}\d{2})_(?P<task>Disassembly|Reassembly)_(?P<model>[AB])_(?P<trial>\d{3})"
)
VIDEO_ID_RE = re.compile(EXECUTION_RE.pattern + r"_(?P<view>ego|front|left|right|top)")


def parse_execution_id(text: str) -> dict[str, Any] | None:
    """Parse `<participant>_<task>_<model>_<NNN>[_<view>]` found anywhere in `text`."""
    m = VIDEO_ID_RE.search(text) or EXECUTION_RE.search(text)
    if not m:
        return None
    d = m.groupdict()
    execution_id = f"{d['participant']}_{d['task']}_{d['model']}_{d['trial']}"
    return {
        "execution_id": execution_id,
        "participant": d["participant"],
        "task": d["task"],
        "model": d["model"],
        "trial": int(d["trial"]),
        "view": d.get("view"),
    }


# ------------------------------------------------------------------ time basis
def frames_to_seconds(frame: float, fps: float, offset_frames: int = 0) -> float:
    """Frame index (0-based, relative to `offset_frames`) -> seconds."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return (float(frame) - offset_frames) / fps


def seconds_to_frame(t_s: float, fps: float, n_frames: int | None = None) -> int:
    """Seconds -> index of the frame on screen at t (floor), clipped to the video."""
    idx = int(np.floor(t_s * fps + 1e-6))
    idx = max(idx, 0)
    if n_frames is not None:
        idx = min(idx, n_frames - 1)
    return idx


# ------------------------------------------------------------------ mappings
def read_mapping(path: str | Path) -> dict[int, str]:
    """`mapping_*.txt`: lines `<id> <NAME>`."""
    out: dict[int, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        idx, name = line.split(maxsplit=1)
        out[int(idx)] = name.strip()
    return out


_PSR_KIND_PREFIXES = (
    ("Incorrectly installed ", "incorrect"),
    ("Install ", "install"),
    ("Remove ", "remove"),
)


def psr_event_id(description: str) -> tuple[str, str, str]:
    """'Install gearbox_housing' -> ('install:gearbox_housing', 'install', 'gearbox_housing')."""
    desc = description.strip()
    for prefix, kind in _PSR_KIND_PREFIXES:
        if desc.startswith(prefix):
            comp = desc[len(prefix):].strip()
            return f"{kind}:{comp}", kind, comp
    raise ValueError(f"Unrecognized PSR event description: {description!r}")


def read_psr_vocabulary(procedure_info_path: str | Path) -> list[dict[str, Any]]:
    """`procedure_info_IMPACT.json` -> [{impact_id, id, kind, component, state_idx, description}]."""
    items = json.loads(Path(procedure_info_path).read_text())
    vocab = []
    for it in items:
        eid, kind, comp = psr_event_id(it["description"])
        vocab.append(
            {
                "impact_id": int(it["id"]),
                "id": eid,
                "kind": kind,
                "component": comp,
                "state_idx": int(it.get("state_idx", -1)),
                "description": it["description"],
            }
        )
    return vocab


# ------------------------------------------------------------------ frame-label files
def read_frame_labels(path: str | Path, mapping: dict[int, str] | None = None) -> list[str]:
    """One label per line. Numeric labels are mapped through `mapping` when given."""
    labels = [ln.strip() for ln in Path(path).read_text().splitlines()]
    while labels and labels[-1] == "":
        labels.pop()
    if mapping and labels and all(lab.lstrip("-").isdigit() for lab in labels[:50]):
        labels = [mapping[int(lab)] for lab in labels]
    return labels


def frame_labels_to_segments(
    labels: list[str], fps: float, ignore: tuple[str, ...] = ("NULL",)
) -> list[Segment]:
    """Run-length encode frame labels. A segment covers frames [s, e] -> [s/fps, (e+1)/fps)."""
    segs: list[Segment] = []
    if not labels:
        return segs
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            lab = labels[start]
            if lab not in ignore:
                segs.append(Segment(label=lab, start_s=start / fps, end_s=i / fps))
            start = i
    return segs


def read_tas_s(path: str | Path, fps: float, mapping: dict[int, str] | None = None) -> list[Segment]:
    return frame_labels_to_segments(read_frame_labels(path, mapping), fps)


def read_ppr(
    paths: dict[str, str | Path], fps: float, mapping: dict[int, str] | None = None
) -> tuple[list[Phase], dict[str, float]]:
    """PPR-L/R frame labels -> non-NORMAL phases per hand, plus anomaly fraction per hand."""
    phases: list[Phase] = []
    frac: dict[str, float] = {}
    for hand, p in sorted(paths.items()):
        labels = read_frame_labels(p, mapping)
        n = max(len(labels), 1)
        frac[hand] = sum(1 for lab in labels if lab == "ANOMALY") / n
        for seg in frame_labels_to_segments(labels, fps, ignore=("NORMAL", "NULL")):
            phases.append(Phase(hand=hand, phase=seg.label, start_s=seg.start_s, end_s=seg.end_s))
    phases.sort(key=lambda p: (p.start_s, p.hand))
    return phases, frac


def read_atr(paths: dict[str, str | Path], fps: float, type_names: list[str]) -> list[AnomalyTypeSpan]:
    """ATR `[T, n_types]` binary arrays -> spans where any type is active, with the active types."""
    spans: list[AnomalyTypeSpan] = []
    for hand, p in sorted(paths.items()):
        arr = np.load(p, allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"ATR array {p} has shape {arr.shape}, expected [T, n_types]")
        if arr.shape[1] != len(type_names) and arr.shape[0] == len(type_names):
            arr = arr.T
        active = arr.astype(bool).any(axis=1)
        i, n = 0, len(active)
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and active[j + 1]:
                j += 1
            cols = np.flatnonzero(arr[i : j + 1].any(axis=0))
            spans.append(
                AnomalyTypeSpan(
                    hand=hand,
                    start_s=i / fps,
                    end_s=(j + 1) / fps,
                    types=[type_names[c].lower().replace("error_", "") for c in cols],
                )
            )
            i = j + 1
    spans.sort(key=lambda s: (s.start_s, s.hand))
    return spans


# ------------------------------------------------------------------ PSR / ASR
def read_psr_events(path: str | Path, fps: float, vocab: list[dict[str, Any]] | None = None) -> list[PSREvent]:
    """`PSR_labels.csv` -> events in time order. Category ids are cross-checked with `vocab`."""
    by_impact_id = {v["impact_id"]: v for v in (vocab or [])}
    events: list[PSREvent] = []
    for raw in Path(path).read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        frame_tok, cat_tok, desc = raw.split(",", 2)
        frame = int(Path(frame_tok.strip()).stem)
        eid, _, _ = psr_event_id(desc)
        if by_impact_id:
            v = by_impact_id.get(int(cat_tok))
            if v is not None and v["id"] != eid:
                raise ValueError(f"{path}: category {cat_tok} is {v['id']} in vocabulary but row says {desc!r}")
        events.append(PSREvent(event_id=eid, time_s=frames_to_seconds(frame, fps), frame=frame))
    events.sort(key=lambda e: (e.time_s, e.event_id))
    return events


def read_asr(path: str | Path) -> tuple[ASRStates, dict[str, Any]]:
    """ASR JSON -> component state keyframes (seconds) and video metadata."""
    d = json.loads(Path(path).read_text())
    md = d.get("meta_data", {})
    fps = float(d.get("fps") or md.get("fps"))
    offset = int(d.get("view_start", md.get("view_start", 0)) or 0)
    comps = [c["name"] for c in sorted(d["components"], key=lambda c: c["id"])]
    times, states = [], []
    for kf in d["state_sequence"]:
        times.append(frames_to_seconds(kf["frame"], fps, offset))
        states.append([int(x) for x in kf["state"]])
    n_frames = int(d.get("frame_count") or md.get("num_frames"))
    res = md.get("resolution", {})
    meta = {
        "fps": fps,
        "n_frames": n_frames,
        "duration_s": n_frames / fps,
        "width": res.get("width"),
        "height": res.get("height"),
        "view_start": offset,
        "workflow": d.get("workflow", md.get("workflow")),
        "model_type": md.get("model_type"),
        "video_id": d.get("video_id"),
    }
    return ASRStates(times_s=times, components=comps, states=states), meta
