"""Catalog of IMPACT executions: builds canonical records and answers pool queries."""

from __future__ import annotations

import csv
import json
import re
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from sopverify.data.impact import adapters as A
from sopverify.data.impact.discover import AnnotationIndex
from sopverify.data.records import ExecutionRecord, ExecutionRef
from sopverify.logging_utils import get_logger

log = get_logger(__name__)

ANNOTATION_KINDS = ("psr", "tas_s", "asr", "ppr_l", "ppr_r", "atr_l", "atr_r")


class Catalog:
    def __init__(
        self,
        index: AnnotationIndex,
        expertise_overrides: dict[str, str] | None = None,
        video_view: str = "front",
    ) -> None:
        self.index = index
        self.view = video_view
        self._expertise_overrides = dict(expertise_overrides or {})
        self._records: dict[str, ExecutionRecord] = {}

    @classmethod
    def from_paths(
        cls,
        data_root: str | Path | None,
        annotations_root: str | Path | None = None,
        expertise_overrides: dict[str, str] | None = None,
        extra_roots: list[str | Path] | None = None,
    ) -> Catalog:
        # Patterns match on paths relative to each root (e.g. `videos/front/<id>.mp4`),
        # so data_root itself is scanned, not its sub-folders. Duplicates are ignored.
        roots: list[str | Path | None] = [annotations_root, data_root]
        roots += list(extra_roots or [])
        seen: set[Path] = set()
        uniq: list[str | Path] = []
        for r in roots:
            if r is None:
                continue
            rp = Path(r).resolve()
            if rp not in seen:
                seen.add(rp)
                uniq.append(r)
        return cls(AnnotationIndex.build(*uniq), expertise_overrides)

    # ------------------------------------------------------------------ vocab
    @cached_property
    def psr_vocab(self) -> list[dict[str, Any]]:
        p = self.index.singletons.get("psr_procedure_info")
        return A.read_psr_vocabulary(p) if p else []

    @cached_property
    def tas_s_mapping(self) -> dict[int, str]:
        p = self.index.singletons.get("mapping_tas_s")
        return A.read_mapping(p) if p else {}

    @cached_property
    def ppr_mapping(self) -> dict[int, str]:
        p = self.index.singletons.get("mapping_ppr")
        return A.read_mapping(p) if p else {}

    @cached_property
    def atr_types(self) -> list[str]:
        p = self.index.singletons.get("mapping_atr")
        if not p:
            return []
        m = A.read_mapping(p)
        return [m[i] for i in sorted(m)]

    @cached_property
    def expertise(self) -> dict[str, str]:
        found = find_expertise(self.index.metadata_files)
        found.update(self._expertise_overrides)
        return found

    # ------------------------------------------------------------------ queries
    def video_id(self, execution_id: str) -> str:
        return f"{execution_id}_{self.view}"

    def execution_ids(
        self,
        task: str | None = "Reassembly",
        model: str | None = "A",
        require: tuple[str, ...] | list[str] = ("psr", "tas_s", "asr"),
        require_video: bool = False,
        require_features: str | None = None,
    ) -> list[str]:
        ids: set[str] = set()
        for (_kind, vid) in self.index.files:
            if not vid.endswith(f"_{self.view}"):
                continue
            info = A.parse_execution_id(vid)
            if not info:
                continue
            if task and info["task"] != task:
                continue
            if model and info["model"] != model:
                continue
            ids.add(info["execution_id"])
        out = []
        for eid in sorted(ids):
            vid = self.video_id(eid)
            if any(self.index.get(k, vid) is None for k in require):
                continue
            if require_video and self.index.get("video", vid) is None:
                continue
            if require_features and self.index.feature(require_features, vid) is None:
                continue
            out.append(eid)
        return out

    def ref(self, execution_id: str) -> ExecutionRef:
        """Annotation-free reference. Reads only video/feature locations and ASR timing metadata."""
        vid = self.video_id(execution_id)
        info = A.parse_execution_id(vid) or {}
        meta: dict[str, Any] = {}
        asr = self.index.get("asr", vid)
        if asr is not None:
            # Only timing/resolution metadata is taken from the ASR file, never states.
            meta = _asr_meta_only(asr)
        video = self.index.get("video", vid)
        return ExecutionRef(
            execution_id=execution_id,
            view=self.view,
            participant=info.get("participant"),
            video_path=str(video) if video else None,
            fps=meta.get("fps"),
            n_frames=meta.get("n_frames"),
            duration_s=meta.get("duration_s"),
            width=meta.get("width"),
            height=meta.get("height"),
            source="impact",
            shipped_features=self._feature_paths(vid),
        )

    def _feature_paths(self, vid: str) -> dict[str, str]:
        return {m: str(p) for (m, v), p in self.index.feature_files.items() if v == vid}

    def record(self, execution_id: str) -> ExecutionRecord:
        if execution_id not in self._records:
            self._records[execution_id] = self._build_record(execution_id)
        return self._records[execution_id]

    def records(self, ids: list[str]) -> list[ExecutionRecord]:
        return [self.record(i) for i in ids]

    def _build_record(self, execution_id: str) -> ExecutionRecord:
        vid = self.video_id(execution_id)
        info = A.parse_execution_id(vid)
        if info is None:
            raise ValueError(f"Not an IMPACT execution id: {execution_id}")
        asr_path = self.index.get("asr", vid)
        if asr_path is None:
            raise FileNotFoundError(f"No ASR annotation for {vid}; fps/frame count come from it")
        asr, meta = A.read_asr(asr_path)
        fps = meta["fps"]
        psr_path = self.index.get("psr", vid)
        psr = A.read_psr_events(psr_path, fps, self.psr_vocab) if psr_path else []
        tas_path = self.index.get("tas_s", vid)
        tas = A.read_tas_s(tas_path, fps, self.tas_s_mapping) if tas_path else []
        ppr_paths = {h: self.index.get(f"ppr_{h.lower()}", vid) for h in ("L", "R")}
        ppr_paths = {h: p for h, p in ppr_paths.items() if p is not None}
        phases, frac = A.read_ppr(ppr_paths, fps, self.ppr_mapping) if ppr_paths else ([], {})
        atr_paths = {h: self.index.get(f"atr_{h.lower()}", vid) for h in ("L", "R")}
        atr_paths = {h: p for h, p in atr_paths.items() if p is not None}
        atr = A.read_atr(atr_paths, fps, self.atr_types) if (atr_paths and self.atr_types) else []
        video = self.index.get("video", vid)
        return ExecutionRecord(
            execution_id=info["execution_id"],
            participant=info["participant"],
            task=info["task"],
            model=info["model"],
            trial=info["trial"],
            expertise=self.expertise.get(info["participant"], "unknown"),
            view=self.view,
            video_path=str(video) if video else None,
            fps=fps,
            n_frames=meta["n_frames"],
            duration_s=meta["duration_s"],
            width=meta.get("width"),
            height=meta.get("height"),
            psr_events=psr,
            tas_s_segments=tas,
            asr_states=asr,
            phases=phases,
            anomaly_types=atr,
            has_anomaly=any(p.phase == "ANOMALY" for p in phases),
            anomaly_frac=frac,
            shipped_features=self._feature_paths(vid),
        )

    # ------------------------------------------------------------------ tables
    def summary_table(self, ids: list[str] | None = None) -> pd.DataFrame:
        """One row per execution (the `sopverify data summarize` table)."""
        ids = ids if ids is not None else self.execution_ids()
        rows = []
        for eid in ids:
            r = self.record(eid)
            final = r.final_component_states()
            rows.append(
                {
                    "execution_id": eid,
                    "participant": r.participant,
                    "expertise": r.expertise,
                    "duration_s": round(r.duration_s, 1),
                    "n_anomaly_segments": r.n_anomaly_segments(),
                    "anomaly_frac_L": round(r.anomaly_frac.get("L", float("nan")), 3),
                    "anomaly_frac_R": round(r.anomaly_frac.get("R", float("nan")), 3),
                    "n_psr_events": len(r.psr_events),
                    "n_remove_events": r.n_remove_events(),
                    "final_installed": sum(1 for v in final.values() if v == 1),
                    "final_misinstalled": sum(1 for v in final.values() if v == -1),
                    "final_uninstalled": sum(1 for v in final.values() if v == 0),
                    "has_front_video": r.video_path is not None,
                    "has_features": sorted(r.shipped_features),
                }
            )
        return pd.DataFrame(rows)


def _asr_meta_only(path: Path) -> dict[str, Any]:
    d = json.loads(Path(path).read_text())
    md = d.get("meta_data", {})
    fps = float(d.get("fps") or md.get("fps"))
    n = int(d.get("frame_count") or md.get("num_frames"))
    res = md.get("resolution", {})
    return {"fps": fps, "n_frames": n, "duration_s": n / fps, "width": res.get("width"), "height": res.get("height")}


# ---------------------------------------------------------------------- expertise
_PART_RE = re.compile(r"\b([A-Z]{2}\d{2}[A-Z]{2}\d{2})\b")
_LEVEL_RE = re.compile(r"\b(expert|novice|beginner|experienced|inexperienced)\b", re.IGNORECASE)
_LEVEL_MAP = {"expert": "expert", "experienced": "expert", "novice": "novice", "beginner": "novice",
              "inexperienced": "novice"}


def find_expertise(metadata_files: list[Path]) -> dict[str, str]:
    """Best-effort read of participant expertise from release metadata.

    The metadata format is not documented in the IMPACT GitHub repo (DATA_NOTES 5.2 #4),
    so this scans CSV/TSV/JSON/text rows for a participant id together with an
    expertise word. Anything it cannot resolve stays "unknown"; use
    `data.expertise_overrides` in the config to fill gaps once the format is known.
    """
    out: dict[str, str] = {}
    for p in metadata_files:
        try:
            if p.suffix.lower() == ".json":
                rows = _json_rows(json.loads(p.read_text()))
            elif p.suffix.lower() in (".csv", ".tsv", ".txt"):
                text = p.read_text(errors="ignore")
                dialect = "excel-tab" if p.suffix.lower() == ".tsv" or "\t" in text[:2000] else "excel"
                rows = [" ".join(r) for r in csv.reader(text.splitlines(), dialect=dialect)]
            else:
                continue
        except Exception as e:  # pragma: no cover - unexpected file content
            log.debug("could not read metadata %s: %s", p, e)
            continue
        for row in rows:
            parts = _PART_RE.findall(row)
            levels = _LEVEL_RE.findall(row)
            if len(set(parts)) == 1 and len({_LEVEL_MAP[x.lower()] for x in levels}) == 1:
                out.setdefault(parts[0], _LEVEL_MAP[levels[0].lower()])
    return out


def _json_rows(obj: Any) -> list[str]:
    rows: list[str] = []
    if isinstance(obj, dict):
        # {participant: {...}} or {participant: "expert"}
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                rows.append(f"{k} " + json.dumps(v))
                rows.extend(_json_rows(v))
            else:
                rows.append(f"{k} {v}")
    elif isinstance(obj, list):
        for v in obj:
            rows.append(json.dumps(v) if not isinstance(v, str) else v)
    return rows
