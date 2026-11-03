"""Walk an IMPACT release (or a clone of the IMPACT GitHub repo) and index its files.

Files are classified by filename patterns, not by a hard-coded directory layout,
because the annotation bundle and the GitHub `dataset/` folder arrange the same
files differently. `discover()` writes a machine-readable inventory and a short
human summary for docs/DATA_NOTES.md.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sopverify.data.impact.adapters import parse_execution_id
from sopverify.logging_utils import get_logger

log = get_logger(__name__)

_VID = r"(?P<vid>[A-Z]{2}\d{2}[A-Z]{2}\d{2}_(?:Disassembly|Reassembly)_[AB]_\d{3}_(?:ego|front|left|right|top))"

# kind -> regex on the posix path relative to the scanned root
PATTERNS: dict[str, re.Pattern[str]] = {
    "tas_s": re.compile(rf"(?:^|/)(?:groundTruth_TAS-S|TAS-S)/(?:.*/)?{_VID}\.txt$"),
    "tas_bl": re.compile(rf"(?:^|/)groundTruth_TAS-BL/(?:.*/)?{_VID}\.txt$"),
    "tas_br": re.compile(rf"(?:^|/)groundTruth_TAS-BR/(?:.*/)?{_VID}\.txt$"),
    "psr": re.compile(rf"(?:^|/){_VID}/PSR_labels\.csv$"),
    "psr_with_errors": re.compile(rf"(?:^|/){_VID}/PSR_labels_with_errors\.csv$"),
    "psr_raw": re.compile(rf"(?:^|/){_VID}/PSR_labels_raw\.csv$"),
    "asr": re.compile(rf"(?:^|/){_VID}_asr\.json$"),
    "ppr_l": re.compile(rf"(?:^|/)groundTruth_PPR_L/(?:.*/)?{_VID}\.txt$"),
    "ppr_r": re.compile(rf"(?:^|/)groundTruth_PPR_R/(?:.*/)?{_VID}\.txt$"),
    "atr_l": re.compile(rf"(?:^|/)groundTruth_ATR_L/(?:.*/)?{_VID}\.npy$"),
    "atr_r": re.compile(rf"(?:^|/)groundTruth_ATR_R/(?:.*/)?{_VID}\.npy$"),
    "video": re.compile(rf"(?:^|/)videos/(?:.*/)?{_VID}\.(?:mp4|mkv|avi|mov)$", re.IGNORECASE),
    "feature": re.compile(
        rf"(?:^|/)features/(?P<model>I3D|MViTv2|VideoMAEv2)/(?:.*/)?{_VID}\.(?:npy|npz|pt)$"
    ),
}
# Single (not per-video) files: kind -> filename regex
SINGLETONS: dict[str, re.Pattern[str]] = {
    "mapping_tas_s": re.compile(r"(?:^|/)mapping_TAS-S\.txt$"),
    "mapping_tas_b": re.compile(r"(?:^|/)mapping_TAS-B\.txt$"),
    "mapping_ppr": re.compile(r"(?:^|/)mapping_PPR\.txt$"),
    "mapping_atr": re.compile(r"(?:^|/)mapping_ATR\.txt$"),
    "psr_procedure_info": re.compile(r"(?:^|/)procedure_info_IMPACT\.json$"),
    "psr_component_names": re.compile(r"(?:^|/)component_names\.json$"),
}
_SKIP_DIRS = {".git", "__pycache__", "third_party", "website", "node_modules"}


@dataclass
class AnnotationIndex:
    """(kind, video_id) -> path, plus singleton files and everything under metadata/."""

    files: dict[tuple[str, str], Path] = field(default_factory=dict)
    feature_files: dict[tuple[str, str], Path] = field(default_factory=dict)  # (model, video_id)
    singletons: dict[str, Path] = field(default_factory=dict)
    metadata_files: list[Path] = field(default_factory=list)
    split_files: list[Path] = field(default_factory=list)
    roots: list[Path] = field(default_factory=list)

    @classmethod
    def build(cls, *roots: str | Path | None) -> AnnotationIndex:
        idx = cls()
        for root in roots:
            if root is None:
                continue
            root = Path(root)
            if not root.exists():
                log.debug("index root %s does not exist; skipped", root)
                continue
            idx.roots.append(root)
            idx._scan(root)
        return idx

    def _scan(self, root: Path) -> None:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                full = Path(dirpath) / fn
                rel = full.relative_to(root).as_posix()
                self._classify(rel, full)

    def _classify(self, rel: str, full: Path) -> None:
        for kind, rx in SINGLETONS.items():
            if rx.search(rel):
                self.singletons.setdefault(kind, full)
                return
        if "/metadata/" in f"/{rel}" or rel.startswith("metadata/"):
            self.metadata_files.append(full)
        if re.search(r"\.split\d+\.bundle$|(?:^|/)splits?/", rel):
            self.split_files.append(full)
        for kind, rx in PATTERNS.items():
            m = rx.search(rel)
            if m:
                vid = m.group("vid")
                if kind == "feature":
                    self.feature_files.setdefault((m.group("model"), vid), full)
                else:
                    self.files.setdefault((kind, vid), full)
                return

    # ------------------------------------------------------------------ queries
    def get(self, kind: str, video_id: str) -> Path | None:
        return self.files.get((kind, video_id))

    def feature(self, model: str, video_id: str) -> Path | None:
        return self.feature_files.get((model, video_id))

    def video_ids(self, kind: str) -> set[str]:
        return {vid for (k, vid) in self.files if k == kind}

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for (kind, vid), p in self.files.items():
            rows.append({"kind": kind, "video_id": vid, "model": None, "path": str(p)})
        for (model, vid), p in self.feature_files.items():
            rows.append({"kind": "feature", "video_id": vid, "model": model, "path": str(p)})
        df = pd.DataFrame(rows, columns=["kind", "video_id", "model", "path"])
        if df.empty:
            return df
        parsed = df["video_id"].map(parse_execution_id)
        for key in ("execution_id", "participant", "task", "model_type", "trial", "view"):
            src = "model" if key == "model_type" else key
            df[key] = parsed.map(lambda d, s=src: d.get(s) if d else None)
        df["size_bytes"] = df["path"].map(lambda p: os.path.getsize(p) if os.path.exists(p) else None)
        return df.sort_values(["kind", "video_id"]).reset_index(drop=True)


def discover(
    roots: list[str | Path], out_path: str | Path | None = None
) -> tuple[AnnotationIndex, pd.DataFrame, str]:
    """Index `roots`, optionally save the inventory (parquet, csv fallback), return a markdown summary."""
    idx = AnnotationIndex.build(*roots)
    df = idx.to_frame()
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(out)
        except Exception as e:  # pragma: no cover - pyarrow missing
            log.warning("parquet unavailable (%s); writing CSV", e)
            df.to_csv(out.with_suffix(".csv"), index=False)
    return idx, df, summarize(idx, df)


def summarize(idx: AnnotationIndex, df: pd.DataFrame) -> str:
    lines = ["## Inventory summary", "", f"Scanned roots: {', '.join(str(r) for r in idx.roots) or '(none)'}", ""]
    if df.empty:
        lines.append("No IMPACT files found.")
        return "\n".join(lines)
    lines.append("| kind | files | executions | views |")
    lines.append("|---|---:|---:|---|")
    for kind, g in df.groupby("kind"):
        views = Counter(g["view"].dropna())
        lines.append(
            f"| {kind} | {len(g)} | {g['execution_id'].nunique()} | "
            + ", ".join(f"{v}:{n}" for v, n in sorted(views.items()))
            + " |"
        )
    lines += ["", "Singleton files:", ""]
    for k, p in sorted(idx.singletons.items()):
        lines.append(f"- `{k}`: `{p}`")
    lines += ["", f"Metadata files: {len(idx.metadata_files)}"]
    for p in idx.metadata_files[:30]:
        lines.append(f"- `{p}`")
    lines += ["", f"Split files: {len(idx.split_files)}"]
    return "\n".join(lines)
