"""sopverify command line (spec 8.12)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sopverify.config import Config, load_config, repo_root
from sopverify.logging_utils import get_logger, setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help="SOP compliance checker (sopverify).")
data_app = typer.Typer(no_args_is_help=True, help="IMPACT data: download, discover, summarize.")
proc_app = typer.Typer(no_args_is_help=True, help="Procedure packages: validate, render, stats, ROI preview.")
feat_app = typer.Typer(no_args_is_help=True, help="Feature extraction (GPU for DINOv2).")
app.add_typer(data_app, name="data")
app.add_typer(proc_app, name="procedure")
app.add_typer(feat_app, name="features")
log = get_logger("cli")

CONFIG_OPT = typer.Option(None, "--config", "-c", help="Config YAML (default: configs/default.yaml).")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


def _cfg(path: Path | None) -> Config:
    return load_config(path)


def _catalog(cfg: Config):
    from sopverify.data.impact.catalog import Catalog

    p = cfg.paths_resolved()
    return Catalog.from_paths(p["data_root"], p["annotations_root"], cfg.data.expertise_overrides)


def _procedure(cfg: Config, path: Path | None = None):
    from sopverify.procedure.loader import load_procedure

    return load_procedure(path or cfg.procedure_path())


# ---------------------------------------------------------------------- data
@data_app.command("summarize")
def data_summarize(config: Path = CONFIG_OPT, out: Path = typer.Option(None, help="Also write the table as CSV."),
                   all_executions: bool = typer.Option(False, "--all", help="Every execution, not only the pool.")):
    """One row per pool execution: participant, expertise, duration, anomalies, PSR events, video/features."""
    cfg = _cfg(config)
    cat = _catalog(cfg)
    ids = cat.execution_ids(None if all_executions else cfg.data.task, None if all_executions else cfg.data.model,
                            tuple(cfg.data.require))
    df = cat.summary_table(ids)
    import pandas as pd

    with pd.option_context("display.max_rows", 500, "display.width", 200):
        typer.echo(df.to_string(index=False))
    typer.echo(f"\n{len(df)} executions, {df['participant'].nunique() if len(df) else 0} participants; "
               f"with front video: {int(df['has_front_video'].sum()) if len(df) else 0}")
    if out:
        df.to_csv(out, index=False)


@data_app.command("discover")
def data_discover(config: Path = CONFIG_OPT, extra_root: list[Path] = typer.Option(None, help="More roots to scan."),
                  out: Path = typer.Option(None, help="Markdown summary output.")):
    """Index the release, write cache/inventory.parquet and print a summary for DATA_NOTES.md."""
    from sopverify.data.impact.discover import discover

    cfg = _cfg(config)
    p = cfg.paths_resolved()
    roots = [r for r in [p["annotations_root"], p["data_root"], *(extra_root or [])] if r]
    _, df, md = discover(roots, Path(p["cache_root"]) / "inventory.parquet")  # type: ignore[arg-type]
    typer.echo(md)
    if out:
        Path(out).write_text(md)


@data_app.command("download")
def data_download(
    what: str = typer.Argument(..., help="annotations | sample | videos | features"),
    config: Path = CONFIG_OPT,
    dest: Path = typer.Option(None, help="Destination root (default: data_root's parent for bundles, data_root for members)."),
    model: str = typer.Option("VideoMAEv2", help="Feature model for 'features'."),
    ids: str = typer.Option("pool", help="'pool' or comma-separated execution ids (videos/features)."),
):
    """Download small bundles whole (checksum-verified) or extract pool files from big remote zips."""
    from sopverify.data.impact import download as D

    cfg = _cfg(config)
    p = cfg.paths_resolved()
    data_root = Path(p["data_root"])  # type: ignore[arg-type]
    if what in ("annotations", "sample"):
        # The 8 MiB annotations zip may live on Drive; the 2.8 GB sample stays on local scratch.
        target = dest or (data_root.parent / "downloads" if what == "annotations" else Path(p["scratch"]) / "downloads")  # type: ignore[arg-type]
        z = D.download_bundle(what, target)
        out_root = data_root if what == "annotations" else Path(p["scratch"]) / "impact_sample"  # type: ignore[arg-type]
        D.extract_zip(z, out_root)
        typer.echo(f"extracted {z.name} into {out_root}")
        return
    if what not in ("videos", "features"):
        raise typer.BadParameter("what must be annotations, sample, videos or features")
    if ids == "pool":
        cat = _catalog(cfg)
        exec_ids = cat.execution_ids(cfg.data.task, cfg.data.model, tuple(cfg.data.require))
    else:
        exec_ids = [x.strip() for x in ids.split(",") if x.strip()]
    vids = [f"{e}_{cfg.data.view}" for e in exec_ids]
    key = f"videos:{cfg.data.view}" if what == "videos" else f"features:{model}"
    paths = D.fetch_members_remote(key, vids, dest or data_root)
    typer.echo(f"{len(paths)} of {len(vids)} files present under {dest or data_root}")
    missing = set(vids) - {Path(x).stem for x in paths}
    if missing:
        typer.echo(f"not found in the bundle: {sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}")


# ---------------------------------------------------------------------- procedure
@proc_app.command("validate")
def procedure_validate(path: Path = typer.Argument(..., help="Procedure package folder."),
                       step_set: str = typer.Option(None, "--step-set", help="Default: all step sets.")):
    """Schema + DAG checks (unique ids, known prerequisites, no cycles, reachability, GT vocabulary)."""
    proc = _procedure(Config(), path)
    names = [step_set] if step_set else list(proc.step_sets)
    bad = False
    for n in names:
        errs = proc.validate(n)
        status = "OK" if not errs else "FAILED"
        typer.echo(f"{proc.procedure_id}/{n}: {len(proc.steps(n))} steps, {status}")
        for e in errs:
            typer.echo(f"  - {e}")
        bad |= bool(errs)
        typer.echo(f"  requires: {json.dumps({k: v for k, v in proc.requires(n).items() if v})}")
    typer.echo(f"ROI {proc.roi.box} confirmed={proc.roi.confirmed}; SOP status: {proc.spec.sop_status or 'final'}")
    raise typer.Exit(1 if bad else 0)


@proc_app.command("render")
def procedure_render(path: Path = typer.Argument(...), step_set: str = typer.Option(None, "--step-set"),
                     source: str = typer.Option(None, help="manual | inferred_all | inferred_demos | released"),
                     out: Path = typer.Option(None, help="SVG output (default: <package>/graph_<set>.svg)."),
                     config: Path = CONFIG_OPT):
    """Render the prerequisite graph to SVG (inferred sources read pool annotations)."""
    cfg = _cfg(config)
    proc = _procedure(cfg, path)
    name = step_set or proc.spec.default_step_set
    comps = None
    if source in ("inferred_all", "inferred_demos"):
        from sopverify.eval.ground_truth import pool_completions

        cat = _catalog(cfg)
        recs = cat.records(cat.execution_ids(cfg.data.task, cfg.data.model, tuple(cfg.data.require)))
        comps = pool_completions(recs, proc.steps(name))
    svg = proc.render_graph(name, source, completions=comps)  # type: ignore[arg-type]
    out = out or proc.root / f"graph_{name}.svg"
    Path(out).write_text(svg)
    typer.echo(f"wrote {out}")


@proc_app.command("stats")
def procedure_stats(path: Path = typer.Argument(...), step_set: str = typer.Option(None, "--step-set"),
                    config: Path = CONFIG_OPT):
    """Completion-order statistics of the pool under a step set (for reviewing the graph)."""
    from collections import Counter

    from sopverify.eval.ground_truth import compute_gt, diagnostics

    cfg = _cfg(config)
    proc = _procedure(cfg, path)
    name = step_set or proc.spec.default_step_set
    cat = _catalog(cfg)
    recs = cat.records(cat.execution_ids(cfg.data.task, cfg.data.model, tuple(cfg.data.require)))
    req = proc.requires(name)
    gts = [compute_gt(r, proc.steps(name), req) for r in recs]
    orders = Counter(" < ".join(sorted((k for k, v in g.steps.items() if v.completion_s is not None),
                                       key=lambda k: g.steps[k].completion_s)) for g in gts)
    typer.echo(f"{len(recs)} executions; graph: {json.dumps({k: v for k, v in req.items() if v})}")
    for o, n in orders.most_common(15):
        typer.echo(f"{n:4d}  {o}")
    d = diagnostics(gts)
    typer.echo(json.dumps({k: v for k, v in d.items() if k != "per_step"}, indent=1))
    for g in gts:
        if g.n_order_violations:
            typer.echo(f"order violation: {g.execution_id}: " + ", ".join(
                f"{k}<-{v.violated}" for k, v in g.steps.items() if v.order_violation))


@proc_app.command("roi-preview")
def procedure_roi_preview(path: Path = typer.Argument(...), execution: str = typer.Option(None),
                          video: Path = typer.Option(None), t: float = typer.Option(60.0, help="Time (s)."),
                          box: str = typer.Option(None, help="Override box 'x,y,w,h' to try."),
                          out: Path = typer.Option(Path("roi_preview.jpg")), config: Path = CONFIG_OPT):
    """Save one frame with the ROI box drawn, to set or confirm the crop."""
    from PIL import Image

    from sopverify.video.decode import read_frames_at
    from sopverify.video.roi import overlay_box

    cfg = _cfg(config)
    proc = _procedure(cfg, path)
    if video is None:
        ref = _catalog(cfg).ref(execution)
        if not ref.video_path:
            raise typer.BadParameter(f"no video for {execution}")
        video = Path(ref.video_path)
    b = [int(x) for x in box.split(",")] if box else proc.roi.box
    frame = read_frames_at(str(video), [t])[0]
    Image.fromarray(overlay_box(frame, b)).save(out, quality=90)
    typer.echo(f"wrote {out} (frame {frame.shape[1]}x{frame.shape[0]}, box {b})")


# ---------------------------------------------------------------------- features
@feat_app.command("extract")
def features_extract(config: Path = CONFIG_OPT, executions: str = typer.Option("pool", help="'pool' or ids."),
                     extractor: str = typer.Option(None, help="Override features.extractor."),
                     crop: str = typer.Option(None, help="Override features.crop (full|roi).")):
    """Extract and cache features for the pool (resumable: cached executions are skipped)."""
    import time

    from tqdm import tqdm

    from sopverify.features.provider import FeatureProvider

    cfg = _cfg(config)
    ov = {}
    if extractor:
        ov["features.extractor"] = extractor
    if crop:
        ov["features.crop"] = crop
    cfg = cfg.with_overrides(ov) if ov else cfg
    proc = _procedure(cfg)
    cat = _catalog(cfg)
    ids = (cat.execution_ids(cfg.data.task, cfg.data.model, tuple(cfg.data.require)) if executions == "pool"
           else [x.strip() for x in executions.split(",")])
    fp = FeatureProvider(cfg, proc.roi.box, proc.roi.frame_size)
    typer.echo(f"feature cache key {fp.key_hash()} ({cfg.features.extractor}, crop={cfg.features.crop})")
    done = failed = 0
    for eid in tqdm(ids):
        ref = cat.ref(eid)
        if fp.has(ref):
            done += 1
            continue
        try:
            t0 = time.time()
            times, feats = fp.get(ref)
            log.info("%s: %d x %d in %.1fs", eid, *feats.shape, time.time() - t0)
            done += 1
        except Exception as e:
            failed += 1
            log.error("%s failed: %s", eid, e)
    typer.echo(f"cached {done}/{len(ids)}; failed {failed}")


# ---------------------------------------------------------------------- run / eval / report
@app.command("run")
def run_cmd(execution: str = typer.Option(None, help="IMPACT execution id."),
            video: Path = typer.Option(None, help="Uploaded video file (same camera setup)."),
            config: Path = CONFIG_OPT, demos: str = typer.Option("auto", help="'auto' or comma-separated demo ids."),
            out: Path = typer.Option(None, help="Output folder."),
            show_gt: bool = typer.Option(None, help="Overlay GT in the HTML report (IMPACT only).")):
    """Judge one video and write verdict.json, evidence, trace and report.html."""
    from sopverify.data.impact.splits import make_split_plan
    from sopverify.pipeline import run

    cfg = _cfg(config)
    proc = _procedure(cfg)
    cat = _catalog(cfg)
    if demos == "auto":
        pool = cat.records(cat.execution_ids(cfg.data.task, cfg.data.model, tuple(cfg.data.require)))
        plan = make_split_plan(pool, cfg.demos, cfg.splits)
        demo_ids = [d for d in plan.demo_draws[0] if d != execution]
        if len(demo_ids) < cfg.demos.k:
            demo_ids += [c for c in plan.candidates if c not in demo_ids and c != execution][: cfg.demos.k - len(demo_ids)]
    else:
        demo_ids = [x.strip() for x in demos.split(",")]
    demo_recs = cat.records(demo_ids)
    gt = None
    if execution:
        target = cat.ref(execution)
        if (show_gt if show_gt is not None else cfg.report.show_gt):
            from sopverify.eval.ground_truth import gt_overlay

            step_set = cfg.step_set or proc.spec.default_step_set
            gt = gt_overlay(cat.record(execution), proc.steps(step_set), proc.requires(step_set))
    elif video:
        target = video  # type: ignore[assignment]
    else:
        raise typer.BadParameter("give --execution or --video")
    rep = run(target, proc, demo_recs, cfg, out_dir=out, gt_overlay=gt)
    from sopverify.report.json_report import summary_line

    typer.echo(summary_line(rep))
    for s in rep.steps:
        typer.echo(f"  {s.step_id:4s} {s.verdict.value:15s} t={s.completion_time_s} conf={s.confidence:.2f} "
                   f"{s.violated_prerequisites or ''} {s.flags or ''}")


@app.command("eval")
def eval_cmd(config: Path = typer.Option(..., "--config", "-c", help="Experiment file (configs/experiments/*.yaml)."),
             only: str = typer.Option(None, help="Comma-separated variant names."),
             out_root: Path = typer.Option(None, help="Runs folder (default: paths.runs_root)."),
             set_: list[str] = typer.Option(None, "--set", help="Base override key=value, e.g. features.crop=full.")):
    """Run an experiment: all variants x demo seeds x dev/test (+ truncations); writes runs/<run>/."""
    from sopverify.config import parse_set_args
    from sopverify.eval.runner import run_experiment

    res = run_experiment(config, only.split(",") if only else None, out_root, overrides=parse_set_args(set_))
    for m in res.values():
        agg = m["results"].get("test", {}).get("all", {}).get("all", {})
        show = {k: round(agg[k]["mean"], 3) for k in ("nd_recall", "nd_precision", "fa_per_compliant", "review_rate",
                                                      "timing_mae_s", "coverage") if k in agg}
        typer.echo(f"{m['run_name']}: {show}")


@app.command("tables")
def tables_cmd(runs: Path = typer.Option(None, help="Runs folder (default: paths.runs_root)."),
               experiments: str = typer.Option(None, help="Comma-separated experiment names."),
               out: Path = typer.Option(None, help="Write tables to this Markdown file."),
               inject: Path = typer.Option(None, help="Replace the TABLES block in this file (e.g. docs/RESULTS.md)."),
               config: Path = CONFIG_OPT):
    """Aggregate run metrics into Markdown tables."""
    from sopverify.eval.tables import inject as do_inject
    from sopverify.eval.tables import render_tables

    root = runs or _cfg(config).paths_resolved()["runs_root"]
    md = render_tables(root, experiments.split(",") if experiments else None)  # type: ignore[arg-type]
    if out:
        Path(out).write_text(md)
    if inject:
        do_inject(inject, md)
    if not out and not inject:
        typer.echo(md)


@app.command("report")
def report_cmd(run: Path = typer.Option(..., "--run", help="runs/<run_id> or one execution folder."),
               config: Path = CONFIG_OPT):
    """(Re)build report.html for every execution folder holding verdict.json + trace.npz."""
    from sopverify.compliance.verdicts import VerdictReport
    from sopverify.report.html_report import write_html_report
    from sopverify.state.types import ScoreTrace
    from sopverify.temporal.smoothing import smooth_trace

    if config is not None:
        cfg = load_config(config)
    elif (run / "config.yaml").exists():  # full config dump written by the runner
        from sopverify.utils import read_yaml

        cfg = Config.model_validate(read_yaml(run / "config.yaml"))
    else:
        cfg = load_config()
    proc = _procedure(cfg)
    n = 0
    for v in sorted(run.rglob("verdict.json")):
        d = v.parent
        rep = VerdictReport.load(v)
        trace = ScoreTrace.load(d / "trace.npz") if (d / "trace.npz").exists() else None
        est = smooth_trace(trace, cfg.smoothing) if trace is not None else None
        write_html_report(rep, proc, trace, est, cfg, d / "report.html", d)
        n += 1
    typer.echo(f"wrote {n} report(s)")


@app.command("app")
def app_cmd(share: bool = typer.Option(False, "--share", help="Public Gradio share link (Colab)."),
            config: Path = CONFIG_OPT, port: int = typer.Option(7860)):
    """Launch the Gradio app."""
    from sopverify.app import launch

    launch(config, share=share, port=port)


@app.command("info")
def info_cmd(config: Path = CONFIG_OPT):
    """Print resolved paths and environment facts."""
    cfg = _cfg(config)
    typer.echo(json.dumps({k: str(v) for k, v in cfg.paths_resolved().items()}, indent=1))
    typer.echo(f"repo root: {repo_root()}")
    try:
        import torch

        typer.echo(f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}"
                   + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    except Exception:
        typer.echo("torch not installed")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
