"""Pipeline entry point (spec 8.10).

    run(video, procedure, demos, cfg) -> VerdictReport

`video` is an `ExecutionRef` (IMPACT execution, no annotations) or a path to an uploaded video.
Demos arrive as annotated records or prepared `DemoData`; the judged video's annotations are
never read here (test-label firewall, spec 9.1). This module must not import `sopverify.eval`.
"""

from __future__ import annotations

import datetime as _dt
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sopverify.compliance.checker import CheckResult, check
from sopverify.compliance.verdicts import StepResult, StepVerdict, Summary, VerdictReport, VerifierInfo
from sopverify.config import Config
from sopverify.data.records import ExecutionRecord, ExecutionRef
from sopverify.logging_utils import get_logger
from sopverify.procedure.loader import Procedure
from sopverify.procedure.schema import Step
from sopverify.state.labels import make_demo
from sopverify.state.scorers import make_scorer
from sopverify.state.scorers.base import BaseScorer
from sopverify.state.types import DemoData, ScoreTrace
from sopverify.temporal.smoothing import StepEstimate, evidence_time, smooth_trace
from sopverify.utils import git_commit

log = get_logger(__name__)

ProgressFn = Callable[[float, str], None]


class FirewallError(RuntimeError):
    """Raised when inference would see annotations of the video it is judging."""


# ---------------------------------------------------------------------- inputs
def ref_from_upload(path: str | Path, cfg: Config) -> tuple[ExecutionRef, list[str]]:
    from sopverify.video.decode import probe

    info = probe(path)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem) or "upload"
    warnings = ["Uploaded video: it must come from the same fixed camera setup as the demos."]
    exp = tuple(cfg.video.expected_resolution)
    if (info.width, info.height) != exp:
        warnings.append(f"Resolution {info.width}x{info.height} differs from the expected {exp[0]}x{exp[1]}; "
                        "the ROI is rescaled proportionally.")
    ref = ExecutionRef(execution_id=f"upload_{stem}", video_path=str(path), fps=info.fps, n_frames=info.n_frames,
                       duration_s=info.duration_s, width=info.width, height=info.height, source="upload")
    return ref, warnings


def build_demo_data(
    demos: Sequence[ExecutionRecord | DemoData], steps: list[Step], features: Any | None, need_features: bool = True
) -> list[DemoData]:
    out: list[DemoData] = []
    for d in demos:
        if isinstance(d, DemoData):
            out.append(d)
            continue
        if need_features:
            times, feats = features.get(d.to_ref())
        else:
            from sopverify.video.timegrid import make_grid

            times, feats = make_grid(d.duration_s, 2.0), None
        out.append(make_demo(d, steps, times, feats))
    return out


def resolve_requires(procedure: Procedure, step_set: str, cfg: Config, demos: Sequence[DemoData] | None = None,
                     requires: Mapping[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Prerequisites to check against. inferred_all must be passed in by eval (it uses pool GT)."""
    if requires is not None:
        return {k: list(v) for k, v in requires.items()}
    source = cfg.graph.source or procedure.spec.graph_source
    if source == "inferred_all":
        raise FirewallError("graph source inferred_all uses all executions' GT; eval must pass `requires`")
    if source == "inferred_demos":
        if not demos:
            raise ValueError("inferred_demos needs demos")
        return procedure.requires(step_set, source, [d.completion_s for d in demos], cfg.graph.min_support)
    return procedure.requires(step_set, source)


# ---------------------------------------------------------------------- core
def judge(
    trace: ScoreTrace,
    steps: Sequence[Step],
    requires: Mapping[str, Sequence[str]],
    cfg: Config,
    *,
    procedure_id: str,
    step_set: str,
    duration_s: float | None = None,
    overrides: Mapping[str, tuple[StepVerdict, float | None]] | None = None,
    estimates: Mapping[str, StepEstimate] | None = None,
) -> tuple[VerdictReport, dict[str, StepEstimate], CheckResult]:
    """Smoothing + checker on a score trace. Cheap; used for every eval variant and the app."""
    est = dict(estimates) if estimates is not None else smooth_trace(trace, cfg.smoothing)
    res = check(steps, est, requires, cfg.checker, overrides)
    dur = duration_s if duration_s is not None else (float(trace.times_s[-1]) if len(trace.times_s) else 0.0)
    results = []
    for s in steps:
        c = res.steps[s.id]
        e = est.get(s.id)
        results.append(StepResult(
            step_id=s.id, name=s.name, verdict=c.verdict, completion_time_s=c.completion_time_s,
            confidence=round(float(c.confidence), 4), max_score=None if c.max_score is None else round(c.max_score, 4),
            violated_prerequisites=c.violated_prerequisites, flags=c.flags,
            evidence_time_s=evidence_time(e, dur, cfg.smoothing) if e is not None and s.visually_verifiable else None,
            required=s.required, visually_verifiable=s.visually_verifiable,
        ))
    report = VerdictReport(
        execution_id=trace.execution_id, procedure_id=procedure_id, step_set=step_set,
        video_verdict=res.video_verdict, summary=Summary.of(results), steps=results, scorer=trace.scorer,
        config_hash=trace.config_hash, duration_s=dur, graph={k: list(v) for k, v in requires.items()},
        manual_check=[s.id for s in steps if not s.visually_verifiable],
    )
    return report, est, res


def scorer_label(cfg: Config) -> str:
    if cfg.scorer.name in ("time_prior", "oracle"):
        return cfg.scorer.name
    if cfg.scorer.name == "vlm":
        return f"vlm:{cfg.vlm.model_id.split('/')[-1]}"
    feat = cfg.features.model_id.split("/")[-1] if cfg.features.extractor == "dinov2" else cfg.features.shipped_model
    crop = cfg.features.crop if cfg.features.extractor == "dinov2" else "full"
    return f"{cfg.scorer.name}:{feat}:{crop}"


def run(
    video: ExecutionRef | str | Path,
    procedure: Procedure,
    demos: Sequence[ExecutionRecord | DemoData],
    cfg: Config,
    *,
    step_set: str | None = None,
    features: Any | None = None,
    scorer: BaseScorer | None = None,
    requires: Mapping[str, list[str]] | None = None,
    verifier: Any | None = None,
    allow_oracle: bool = False,
    out_dir: str | Path | None = None,
    write: bool = True,
    html: bool | None = None,
    gt_overlay: Any | None = None,
    progress: ProgressFn | None = None,
) -> VerdictReport:
    t0 = time.time()
    prog = progress or (lambda f, m: None)
    if isinstance(video, ExecutionRecord):
        raise FirewallError("pass ExecutionRef (record.to_ref()), not an annotated ExecutionRecord, for the judged video")
    warnings: list[str] = []
    if isinstance(video, (str, Path)):
        ref, warnings = ref_from_upload(video, cfg)
    else:
        ref = video
    demo_ids = {d.execution_id for d in demos}
    if ref.execution_id in demo_ids:
        raise FirewallError(f"{ref.execution_id} is one of the demos; it cannot be judged against itself")
    if cfg.scorer.name == "oracle" and not allow_oracle:
        raise FirewallError("the oracle scorer reads ground truth and is refused outside eval (allow_oracle=False)")

    step_set = step_set or cfg.step_set or procedure.spec.default_step_set
    steps = procedure.steps(step_set)
    if features is None and cfg.scorer.name not in ("time_prior", "oracle", "vlm"):
        from sopverify.features.provider import FeatureProvider

        features = FeatureProvider(cfg, procedure.roi.box, procedure.roi.frame_size)

    # 1. scorer (fit on demos unless a fitted one is given)
    prog(0.05, "preparing demos")
    need_feats = cfg.scorer.name not in ("time_prior", "oracle", "vlm")
    demo_data = build_demo_data(demos, steps, features, need_features=need_feats)
    if scorer is None:
        scorer = make_scorer(cfg)
        if getattr(scorer, "is_oracle", False) and not allow_oracle:
            raise FirewallError("oracle scorer refused")
        prog(0.15, "fitting scorer")
        scorer.fit(demo_data, steps)
    elif getattr(scorer, "is_oracle", False) and not allow_oracle:
        raise FirewallError("oracle scorer refused")
    req = resolve_requires(procedure, step_set, cfg, demo_data, requires)

    # 2. features + score trace
    prog(0.3, "features")
    if getattr(scorer, "needs_features", True):
        times, feats = features.get(ref)
    else:
        from sopverify.video.timegrid import make_grid

        times, feats = make_grid(ref.duration_s or 0.0, cfg.video.sample_fps), None
    prog(0.6, "scoring")
    scores = scorer.score(times, feats, ref)
    config_hash = cfg.stage_hash("scores", demos=sorted(demo_ids), roi=procedure.roi.box)
    trace = ScoreTrace(ref.execution_id, times, np.clip(scores, 0, 1), [s.id for s in steps],
                       scorer_label(cfg), config_hash)

    # 3. smoothing + checker (+ optional VLM verifier, then re-check)
    prog(0.75, "checking")
    report, est, res = judge(trace, steps, req, cfg, procedure_id=procedure.procedure_id, step_set=step_set,
                             duration_s=ref.duration_s)
    if verifier is not None:
        prog(0.8, "VLM verifier")
        ver = verifier.verify(ref, steps, est, res, demo_data, procedure)
        if ver:
            overrides = {k: (v[0], v[1]) for k, v in ver.items()}
            report, est, res = judge(trace, steps, req, cfg, procedure_id=procedure.procedure_id,
                                     step_set=step_set, duration_s=ref.duration_s, overrides=overrides, estimates=est)
            for k, v in ver.items():
                report.step(k).verifier = v[2]
            report.meta["verifier"] = True

    report.warnings = warnings
    report.source = ref.source
    report.git_commit = git_commit()
    report.created_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    report.meta.update({"demos": sorted(demo_ids), "scorer_info": getattr(scorer, "info", {}),
                        "graph_source": cfg.graph.source or procedure.spec.graph_source,
                        "roi": procedure.roi.box, "roi_confirmed": procedure.roi.confirmed})

    # 4. outputs
    if write:
        prog(0.85, "writing outputs")
        out = Path(out_dir) if out_dir else _default_out_dir(cfg, ref.execution_id)
        out.mkdir(parents=True, exist_ok=True)
        trace.save(out / "trace.npz")
        if cfg.report.evidence and ref.video_path:
            try:
                save_evidence(ref, report, procedure, out)
            except Exception as e:  # pragma: no cover - depends on video files
                log.warning("evidence frames failed: %s", e)
                report.warnings.append(f"evidence frames not saved: {e}")
        report.runtime_s = round(time.time() - t0, 3)
        report.save(out / "verdict.json")
        if html if html is not None else cfg.report.html:
            from sopverify.report.html_report import write_html_report

            write_html_report(report, procedure, trace, est, cfg, out / "report.html", out_dir=out, gt=gt_overlay)
    report.runtime_s = round(time.time() - t0, 3)
    prog(1.0, "done")
    return report


def _default_out_dir(cfg: Config, execution_id: str) -> Path:
    runs = cfg.paths_resolved()["runs_root"]
    run_id = cfg.run_name or _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return Path(runs) / run_id / execution_id  # type: ignore[arg-type]


def save_evidence(ref: ExecutionRef, report: VerdictReport, procedure: Procedure, out_dir: Path,
                  max_long_side: int = 768) -> None:
    """Evidence keyframe per verifiable step: ROI crop at completion + offset (or at max score)."""
    from PIL import Image

    from sopverify.video import roi as R
    from sopverify.video.decode import read_frames_at

    items = [s for s in report.steps if s.evidence_time_s is not None]
    if not items or not ref.video_path:
        return
    frames = read_frames_at(ref.video_path, [s.evidence_time_s for s in items], fps=ref.fps)  # type: ignore[misc]
    ev = out_dir / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    for s, fr in zip(items, frames):
        box = procedure.roi_for(s.step_id)
        if ref.width and ref.height and [ref.width, ref.height] != list(procedure.roi.frame_size):
            box = R.scale_box(box, procedure.roi.frame_size, (ref.width, ref.height))
        img = R.resize_long_side(R.crop(fr, box), max_long_side)
        Image.fromarray(img).save(ev / f"{s.step_id}.jpg", quality=88)
        s.evidence_frame = f"evidence/{s.step_id}.jpg"


__all__ = ["run", "judge", "FirewallError", "resolve_requires", "build_demo_data", "VerifierInfo"]
