"""Self-contained HTML report (Jinja2 + inline Plotly; spec 8.11)."""

from __future__ import annotations

import base64
import html as _html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from sopverify.compliance.verdicts import VerdictReport
from sopverify.procedure.graph import VERDICT_COLORS, render_svg
from sopverify.state.types import ScoreTrace

TEMPLATES = Path(__file__).parent / "templates"
STEP_COLORS = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#0099c6", "#dd4477", "#66aa00",
               "#b82e2e", "#316395", "#994499", "#22aa99", "#aaaa11", "#6633cc", "#e67300", "#8b0707", "#329262"]


@dataclass
class GTOverlay:
    """Ground truth shown in reports for IMPACT executions (evaluation only)."""

    completion_s: dict[str, float | None]
    done: dict[str, bool]
    compliant: bool
    order_violation: dict[str, bool] = field(default_factory=dict)
    anomaly_spans: list[tuple[float, float, str]] = field(default_factory=list)  # (start, end, label)


def timeline_figure(report: VerdictReport, trace: ScoreTrace, estimates: dict[str, Any] | None, cfg,
                    gt: GTOverlay | None = None):
    """Plotly figure: per-step raw/smoothed scores, thresholds, predicted (and optional GT) completions."""
    import plotly.graph_objects as go

    fig = go.Figure()
    t = trace.times_s
    for j, sid in enumerate(trace.step_ids):
        color = STEP_COLORS[j % len(STEP_COLORS)]
        raw = trace.scores[:, j]
        sm = estimates[sid].smoothed if estimates and sid in estimates and estimates[sid].smoothed is not None else None
        fig.add_trace(go.Scatter(x=t, y=raw, mode="lines", name=f"{sid} raw", line={"color": color, "width": 1},
                                 opacity=0.35, legendgroup=sid, showlegend=False, hoverinfo="skip"))
        if sm is not None:
            fig.add_trace(go.Scatter(x=t, y=sm, mode="lines", name=sid, line={"color": color, "width": 2.2},
                                     legendgroup=sid))
        r = report.step(sid)
        if r.completion_time_s is not None:
            fig.add_trace(go.Scatter(x=[r.completion_time_s], y=[1.04], mode="markers", legendgroup=sid,
                                     marker={"symbol": "triangle-down", "size": 12, "color": color},
                                     name=f"{sid} predicted completion", showlegend=False,
                                     hovertext=f"{sid} predicted {r.completion_time_s:.1f}s ({r.verdict.value})"))
        if gt is not None and gt.completion_s.get(sid) is not None:
            fig.add_trace(go.Scatter(x=[gt.completion_s[sid]], y=[-0.06], mode="markers", legendgroup=sid,
                                     marker={"symbol": "triangle-up", "size": 11, "color": color,
                                             "line": {"width": 1, "color": "#000"}},
                                     name=f"{sid} GT completion (eval only)", showlegend=False,
                                     hovertext=f"{sid} GT {gt.completion_s[sid]:.1f}s (evaluation only)"))
    s = cfg.smoothing
    for y, lab, dash in ((s.tau_on, "tau_on", "dash"), (s.tau_off, "tau_off", "dot"), (cfg.checker.tau_low, "tau_low", "dashdot")):
        fig.add_hline(y=y, line={"dash": dash, "width": 1, "color": "#888"}, annotation_text=lab,
                      annotation_position="right")
    if gt is not None:
        for a, b, lab in gt.anomaly_spans:
            fig.add_vrect(x0=a, x1=b, fillcolor="#e53935", opacity=0.10, line_width=0,
                          annotation_text="" if b - a < 3 else lab, annotation_position="top left",
                          annotation_font_size=9)
    fig.update_layout(height=420, margin={"l": 50, "r": 60, "t": 30, "b": 40}, yaxis={"range": [-0.12, 1.1],
                      "title": "P(step complete)"}, xaxis={"title": "time (s)"}, legend={"orientation": "h", "y": -0.2},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template="plotly_white")
    return fig


def _timeline(report: VerdictReport, trace: ScoreTrace, estimates: dict[str, Any] | None, cfg, gt: GTOverlay | None,
              include_js: bool | str = True) -> str:
    fig = timeline_figure(report, trace, estimates, cfg, gt)
    return fig.to_html(full_html=False, include_plotlyjs=include_js, config={"displaylogo": False})


def _img_b64(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def render_html_report(
    report: VerdictReport,
    procedure,
    trace: ScoreTrace | None,
    estimates: dict[str, Any] | None,
    cfg,
    out_dir: str | Path | None = None,
    gt: GTOverlay | None = None,
    include_plotlyjs: bool | str = True,
) -> str:
    out_dir = Path(out_dir) if out_dir else None
    verdict_of = {s.step_id: s.verdict.value for s in report.steps}
    colors = {k: VERDICT_COLORS.get(v, VERDICT_COLORS[None]) for k, v in verdict_of.items()}
    names = {s.step_id: s.name for s in report.steps}
    svg = render_svg(report.graph or {s.step_id: [] for s in report.steps}, labels=names, colors=colors,
                     order=[s.step_id for s in report.steps])
    gallery = []
    for s in report.steps:
        src = _img_b64(out_dir / s.evidence_frame) if (out_dir and s.evidence_frame) else None
        if src:
            gallery.append({"step": s, "src": src})
    timeline = _timeline(report, trace, estimates, cfg, gt, include_plotlyjs) if trace is not None and len(trace.times_s) else ""
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    tpl = env.get_template("report.html.j2")
    return tpl.render(
        r=report, svg=svg, timeline=timeline, gallery=gallery, gt=gt, colors=VERDICT_COLORS,
        procedure_name=getattr(getattr(procedure, "spec", None), "name", report.procedure_id),
        roi_confirmed=report.meta.get("roi_confirmed"),
        config_yaml=yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False) if cfg is not None else "",
        esc=_html.escape, fmt=lambda x, n=1: "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}",
    )


def write_html_report(report, procedure, trace, estimates, cfg, path, out_dir=None, gt=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(report, procedure, trace, estimates, cfg, out_dir or path.parent, gt))
    return path
