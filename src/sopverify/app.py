"""Gradio app (spec 10): Run, Report, Procedure and Results tabs. Launch with `sopverify app --share`."""

from __future__ import annotations

import datetime as _dt
import html
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from sopverify.config import Config, load_config, repo_root
from sopverify.logging_utils import get_logger

log = get_logger(__name__)

PRESETS = {
    "Best feature scorer": {},
    "Feature scorer + VLM verifier": {"vlm.enabled": True, "vlm.verifier": True},
    "VLM scorer": {"scorer.name": "vlm", "vlm.enabled": True},
}
UPLOAD_FALLBACK = {"features.extractor": "dinov2", "features.crop": "roi"}
VERDICT_BG = {"COMPLIANT": "#2e7d32", "NON_COMPLIANT": "#c62828", "NEEDS_REVIEW": "#b26a00"}


def _default_config_path() -> Path:
    best = repo_root() / "configs" / "best.yaml"
    return best if best.exists() else repo_root() / "configs" / "default.yaml"


@dataclass
class AppState:
    cfg: Config
    procedures: dict[str, Any] = field(default_factory=dict)
    catalog: Any = None
    plan: Any = None
    scorers: dict[tuple, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    vlm_backend: Any = None
    last: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- lazy loaders
    def procedure(self, name: str):
        from sopverify.procedure.loader import load_procedure

        if name not in self.procedures:
            self.procedures[name] = load_procedure(repo_root() / "procedures" / name)
        return self.procedures[name]

    def ensure_catalog(self) -> None:
        if self.catalog is not None:
            return
        from sopverify.data.impact.catalog import Catalog
        from sopverify.data.impact.splits import make_split_plan

        p = self.cfg.paths_resolved()
        try:
            self.catalog = Catalog.from_paths(p["data_root"], p["annotations_root"], self.cfg.data.expertise_overrides)
            ids = self.catalog.execution_ids(self.cfg.data.task, self.cfg.data.model, tuple(self.cfg.data.require))
            recs = self.catalog.records(ids)
            self.plan = make_split_plan(recs, self.cfg.demos, self.cfg.splits) if recs else None
        except Exception as e:  # pragma: no cover - depends on data
            log.warning("IMPACT catalog unavailable: %s", e)
            self.catalog, self.plan = None, None

    def demo_ids(self) -> list[str]:
        if self.cfg.demos.fixed_ids:
            return list(self.cfg.demos.fixed_ids)
        return list(self.plan.demo_draws[0]) if self.plan else []

    def execution_choices(self) -> list[tuple[str, str]]:
        self.ensure_catalog()
        if not self.plan:
            return []
        demos = set(self.demo_ids())
        out = []
        for split, ids in (("dev", self.plan.dev), ("test", self.plan.test)):
            for e in ids:
                tag = "demo" if e in demos else split
                out.append((f"{e}  [{tag}]", e))
        return sorted(out)

    def feature_provider(self, cfg: Config, procedure):
        from sopverify.features.provider import FeatureProvider

        key = cfg.stage_hash("features", roi=procedure.roi.box)
        if key not in self.features:
            self.features[key] = FeatureProvider(cfg, procedure.roi.box, procedure.roi.frame_size)
        return self.features[key]

    def backend(self, cfg: Config):
        if self.vlm_backend is None:
            from sopverify.vlm.backend import make_backend

            self.vlm_backend = make_backend(cfg.vlm)
        return self.vlm_backend


# ---------------------------------------------------------------------- rendering helpers
def verdict_badge(report) -> str:
    v = report.video_verdict.value
    s = report.summary
    warn = "".join(f"<div style='margin-top:6px;color:#b26a00'>{html.escape(w)}</div>" for w in report.warnings)
    return (f"<div style='display:flex;gap:14px;align-items:center;flex-wrap:wrap'>"
            f"<span style='background:{VERDICT_BG[v]};color:#fff;padding:8px 16px;border-radius:999px;font-weight:700'>"
            f"{v.replace('_', ' ')}</span><span><b>{html.escape(report.execution_id)}</b> · {report.step_set} · "
            f"{html.escape(report.scorer)} · {report.runtime_s:.1f} s</span>"
            f"<span>done {s.done} · not done {s.not_done} · out of order {s.out_of_order} · uncertain {s.uncertain}"
            f" · manual {s.not_verifiable}</span></div>{warn}")


def steps_frame(report) -> pd.DataFrame:
    from sopverify.report.json_report import steps_table

    return pd.DataFrame(steps_table(report))


def gallery_items(report, out_dir: Path) -> list[tuple[str, str]]:
    items = []
    for s in report.steps:
        if s.evidence_frame and (out_dir / s.evidence_frame).exists():
            t = f" @ {s.evidence_time_s:.1f}s" if s.evidence_time_s is not None else ""
            items.append((str(out_dir / s.evidence_frame), f"{s.step_id} · {s.verdict.value}{t}"))
    return items


def graph_html(report) -> str:
    from sopverify.procedure.graph import VERDICT_COLORS, render_svg

    colors = {s.step_id: VERDICT_COLORS.get(s.verdict.value) for s in report.steps}
    return render_svg(report.graph, {s.step_id: s.name for s in report.steps}, colors, [s.step_id for s in report.steps])


# ---------------------------------------------------------------------- actions
def run_video(state: AppState, procedure_name: str, step_set: str, source: str, execution: str | None,
              upload: str | None, preset: str, progress=None) -> dict[str, Any]:
    from sopverify.pipeline import build_demo_data, run
    from sopverify.state.scorers import make_scorer

    t0 = time.time()
    prog = progress or (lambda *a, **k: None)
    proc = state.procedure(procedure_name)
    cfg = state.cfg.with_overrides({"step_set": step_set, **PRESETS[preset]})
    state.ensure_catalog()
    if state.catalog is None:
        raise RuntimeError("IMPACT annotations are needed for the demos; check paths in the config")
    warnings: list[str] = []
    if source == "Upload video":
        if not upload:
            raise ValueError("upload a video first")
        if cfg.features.extractor == "shipped_videomaev2" and cfg.scorer.name != "vlm":
            cfg = cfg.with_overrides(UPLOAD_FALLBACK)
            warnings.append("Shipped VideoMAEv2 features do not exist for uploads; using DINOv2 ROI features.")
        target: Any = Path(upload)
    else:
        if not execution:
            raise ValueError("choose an execution")
        target = state.catalog.ref(execution)
    demo_ids = [d for d in state.demo_ids() if d != getattr(target, "execution_id", None)]
    steps = proc.steps(step_set)
    feats = None if cfg.scorer.name in ("vlm", "time_prior") else state.feature_provider(cfg, proc)
    key = (procedure_name, step_set, preset, tuple(demo_ids), cfg.stage_hash("scores"))
    prog(0.1, desc="preparing demos and scorer")
    demos = build_demo_data(state.catalog.records(demo_ids), steps, feats, need_features=feats is not None)
    scorer = state.scorers.get(key)
    if scorer is None:
        kwargs = {}
        if cfg.scorer.name == "vlm":
            from sopverify.vlm.frames import VideoFrameSource

            kwargs = {"backend": state.backend(cfg),
                      "frames": VideoFrameSource(proc.roi.box, proc.roi.frame_size, cfg.vlm.max_long_side)}
        scorer = make_scorer(cfg, **kwargs)
        scorer.fit(demos, steps)
        state.scorers[key] = scorer
    verifier = None
    if cfg.vlm.enabled and cfg.vlm.verifier:
        from sopverify.vlm.frames import VideoFrameSource
        from sopverify.vlm.verifier import VLMVerifier

        verifier = VLMVerifier(cfg, state.backend(cfg), VideoFrameSource(proc.roi.box, proc.roi.frame_size))
    gt = None
    if source != "Upload video":
        from sopverify.eval.ground_truth import gt_overlay  # display only; never fed to the pipeline

        gt = gt_overlay(state.catalog.record(execution), steps, proc.requires(step_set))
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.paths_resolved()["runs_root"]) / "app" / stamp  # type: ignore[arg-type]
    eid = getattr(target, "execution_id", Path(str(target)).stem)
    prog(0.3, desc="scoring video")
    report = run(target, proc, demos, cfg, step_set=step_set, features=feats, scorer=scorer, verifier=verifier,
                 out_dir=out_dir / eid, gt_overlay=gt if cfg.report.show_gt else None,
                 progress=lambda f, m: prog(0.3 + 0.7 * f, desc=m))
    report.warnings = warnings + report.warnings
    from sopverify.state.types import ScoreTrace
    from sopverify.temporal.smoothing import smooth_trace

    trace = ScoreTrace.load(out_dir / eid / "trace.npz")
    est = smooth_trace(trace, cfg.smoothing)
    state.last = {"report": report, "trace": trace, "est": est, "cfg": cfg, "gt": gt, "dir": out_dir / eid,
                  "elapsed": time.time() - t0}
    return state.last


def build_app(config_path: str | Path | None = None, state: AppState | None = None):
    import gradio as gr

    from sopverify.report.html_report import timeline_figure

    cfg = load_config(config_path or _default_config_path())
    state = state or AppState(cfg)
    proc_names = sorted(p.name for p in (repo_root() / "procedures").iterdir() if (p / "procedure.yaml").exists())
    default_proc = state.procedure(proc_names[0]) if proc_names else None

    with gr.Blocks(title="sopverify") as demo:
        gr.Markdown("# SOP compliance checker\nJudge a video against a procedure: every step is **done**, "
                    "**not done**, **out of order** or **can't tell**, with timestamps and evidence frames.")
        with gr.Tabs() as tabs:
            # ------------------------------------------------------------ Run
            with gr.Tab("Run", id="run"):
                with gr.Row():
                    proc_dd = gr.Dropdown(proc_names, value=proc_names[0] if proc_names else None, label="Procedure")
                    ss_dd = gr.Dropdown(list(default_proc.step_sets) if default_proc else [],
                                        value=(cfg.step_set or (default_proc.spec.default_step_set if default_proc else None)),
                                        label="Step set")
                    preset_dd = gr.Dropdown(list(PRESETS), value="Best feature scorer", label="Scorer preset")
                source = gr.Radio(["IMPACT execution", "Upload video"], value="IMPACT execution", label="Input")
                exec_dd = gr.Dropdown(state.execution_choices(), label="IMPACT execution (dev/test pool; demos marked)")
                upload = gr.File(label="Upload MP4 (same fixed camera setup as the demos)", file_types=[".mp4", ".mov", ".mkv"],
                                 visible=False)
                gr.Markdown("Uploads must come from the same camera setup; the procedure's crop box is rescaled if "
                            "the resolution differs.", visible=True)
                run_btn = gr.Button("Run", variant="primary")
                status = gr.Markdown()
            # ------------------------------------------------------------ Report
            with gr.Tab("Report", id="report"):
                badge = gr.HTML()
                steps_df = gr.Dataframe(interactive=False, wrap=True)
                gt_toggle = gr.Checkbox(value=False, label="Show ground truth (evaluation only, IMPACT videos)")
                plot = gr.Plot()
                gallery = gr.Gallery(label="Evidence keyframes", columns=4, height="auto")
                graph = gr.HTML()
                with gr.Row():
                    dl_json = gr.File(label="verdict.json")
                    dl_html = gr.File(label="report.html")
            # ------------------------------------------------------------ Procedure
            with gr.Tab("Procedure", id="procedure"):
                sop_md = gr.Markdown(default_proc.sop_text if default_proc else "")
                proc_steps = gr.Dataframe(interactive=False, wrap=True)
                proc_graph = gr.HTML()
                gr.Markdown("### ROI preview")
                with gr.Row():
                    roi_exec = gr.Dropdown(state.execution_choices(), label="Execution")
                    roi_t = gr.Slider(0, 600, value=60, step=1, label="Time (s)")
                    roi_box = gr.Textbox(value=",".join(map(str, default_proc.roi.box)) if default_proc else "",
                                         label="Box x,y,w,h")
                roi_btn = gr.Button("Preview crop box")
                roi_img = gr.Image(label="Frame with ROI", type="numpy")
            # ------------------------------------------------------------ Results
            with gr.Tab("Results", id="results"):
                res_path = repo_root() / "docs" / "RESULTS.md"
                gr.Markdown(res_path.read_text() if res_path.exists() else "_docs/RESULTS.md not found._")

        # ---------------------------------------------------------------- callbacks
        def on_source(src):
            return gr.update(visible=src == "IMPACT execution"), gr.update(visible=src == "Upload video")

        source.change(on_source, source, [exec_dd, upload])

        def on_proc(name):
            p = state.procedure(name)
            return (gr.update(choices=list(p.step_sets), value=p.spec.default_step_set), p.sop_text,
                    _proc_steps(p, p.spec.default_step_set), p.render_graph(p.spec.default_step_set),
                    ",".join(map(str, p.roi.box)))

        def _proc_steps(p, ss):
            return pd.DataFrame([{"id": s.id, "name": s.name, "requires": ", ".join(s.requires), "required": s.required,
                                  "visually verifiable": s.visually_verifiable, "end state": s.end_state_text}
                                 for s in p.steps(ss)])

        proc_dd.change(on_proc, proc_dd, [ss_dd, sop_md, proc_steps, proc_graph, roi_box])
        ss_dd.change(lambda n, ss: (_proc_steps(state.procedure(n), ss), state.procedure(n).render_graph(ss)),
                     [proc_dd, ss_dd], [proc_steps, proc_graph])

        def render_last(show_gt):
            L = state.last
            if not L:
                return [gr.update()] * 7
            rep = L["report"]
            fig = timeline_figure(rep, L["trace"], L["est"], L["cfg"], L["gt"] if show_gt else None)
            d = L["dir"]
            return (verdict_badge(rep), steps_frame(rep), fig, gallery_items(rep, d), graph_html(rep),
                    str(d / "verdict.json"), str(d / "report.html") if (d / "report.html").exists() else None)

        def on_run(pname, ss, src, ex, up, preset, show_gt, progress=gr.Progress()):
            try:
                run_video(state, pname, ss, src, ex, getattr(up, "name", up), preset, progress)
            except Exception as e:
                log.exception("run failed")
                return [f"**Failed:** {html.escape(str(e))}"] + [gr.update()] * 7 + [gr.update()]
            msg = f"Done in {state.last['elapsed']:.1f} s. See the Report tab."
            return [msg, *render_last(show_gt), gr.Tabs(selected="report")]

        run_btn.click(on_run, [proc_dd, ss_dd, source, exec_dd, upload, preset_dd, gt_toggle],
                      [status, badge, steps_df, plot, gallery, graph, dl_json, dl_html, tabs])
        gt_toggle.change(render_last, gt_toggle, [badge, steps_df, plot, gallery, graph, dl_json, dl_html])

        def on_roi(pname, ex, t, box):
            from sopverify.video.decode import read_frames_at
            from sopverify.video.roi import overlay_box

            state.ensure_catalog()
            ref = state.catalog.ref(ex) if (state.catalog and ex) else None
            if ref is None or not ref.video_path:
                raise gr.Error("choose an execution whose video is available")
            b = [int(x) for x in box.split(",")]
            return overlay_box(read_frames_at(ref.video_path, [float(t)])[0], b)

        roi_btn.click(on_roi, [proc_dd, roi_exec, roi_t, roi_box], roi_img)
        if default_proc is not None:
            demo.load(lambda: (_proc_steps(default_proc, default_proc.spec.default_step_set),
                               default_proc.render_graph(default_proc.spec.default_step_set)),
                      None, [proc_steps, proc_graph])
    return demo


def launch(config_path: str | Path | None = None, share: bool = False, port: int = 7860) -> None:
    demo = build_app(config_path)
    cfg = load_config(config_path or _default_config_path())
    allowed = [str(cfg.paths_resolved()["runs_root"]), str(cfg.paths_resolved()["data_root"])]
    demo.queue().launch(share=share, server_port=port, allowed_paths=allowed, show_error=True)


__all__ = ["build_app", "launch", "run_video", "AppState", "PRESETS"]
