"""Experiment runner (spec 9): demo seeds x splits x (original + truncations) -> metrics and run folder.

Flow per experiment variant:
1. Build the pool, the split plan (dev/test fixed, demos per seed) and the active graph.
2. Per seed: fit the scorer on that seed's demos; score every dev/test video once; derive
   truncated variants by slicing the trace (or re-scoring duration-dependent scorers).
3. Optional tuning: grid-search smoothing/checker thresholds on DEV traces only.
4. Judge every variant with the final thresholds, compute GT, metrics (mean +- std over
   seeds; originals / truncations / combined; seen / unseen participants), write the run.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sopverify.config import Config, load_experiment
from sopverify.data.impact.catalog import Catalog
from sopverify.data.impact.splits import SplitPlan, make_split_plan
from sopverify.data.records import ExecutionRecord, ExecutionRef
from sopverify.eval import metrics as M
from sopverify.eval.ground_truth import ExecutionGT, compute_gt, diagnostics, oracle_gt_fn, pool_completions
from sopverify.eval.truncation import make_truncations
from sopverify.logging_utils import get_logger
from sopverify.pipeline import build_demo_data, judge, scorer_label
from sopverify.procedure.loader import Procedure, load_procedure
from sopverify.state.scorers import make_scorer
from sopverify.state.types import ScoreTrace
from sopverify.utils import git_commit, write_json, write_yaml
from sopverify.video.timegrid import make_grid

log = get_logger(__name__)


@dataclass
class Item:
    seed: int
    split: str
    execution_id: str
    variant: str
    kind: str  # original | truncation
    seen: bool
    t_end: float | None
    trace: ScoreTrace
    duration_s: float
    gt: ExecutionGT | None = None
    ref: ExecutionRef | None = None


@dataclass
class Context:
    cfg: Config
    procedure: Procedure
    catalog: Catalog
    step_set: str
    records: dict[str, ExecutionRecord]
    plan: SplitPlan
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def steps(self):
        return self.procedure.steps(self.step_set)


def build_context(cfg: Config, catalog: Catalog | None = None, procedure: Procedure | None = None) -> Context:
    paths = cfg.paths_resolved()
    if catalog is None:
        catalog = Catalog.from_paths(paths["data_root"], paths["annotations_root"], cfg.data.expertise_overrides)
    procedure = procedure or load_procedure(cfg.procedure_path())
    step_set = cfg.step_set or procedure.spec.default_step_set
    ids = catalog.execution_ids(cfg.data.task, cfg.data.model, tuple(cfg.data.require))
    if not ids:
        raise RuntimeError(f"empty pool: no {cfg.data.task}_{cfg.data.model} executions with {cfg.data.require} "
                           f"under {catalog.index.roots}")
    records = {r.execution_id: r for r in catalog.records(ids)}
    plan = make_split_plan(list(records.values()), cfg.demos, cfg.splits)
    return Context(cfg, procedure, catalog, step_set, records, plan)


def active_requires(ctx: Context, cfg: Config, demo_ids: list[str]) -> dict[str, list[str]]:
    source = cfg.graph.source or ctx.procedure.spec.graph_source
    steps = ctx.steps
    if source == "inferred_all":
        return ctx.procedure.requires(ctx.step_set, source, pool_completions(list(ctx.records.values()), steps),
                                      cfg.graph.min_support)
    if source == "inferred_demos":
        comps = pool_completions([ctx.records[d] for d in demo_ids], steps)
        return ctx.procedure.requires(ctx.step_set, source, comps, cfg.graph.min_support)
    return ctx.procedure.requires(ctx.step_set, source)


# ---------------------------------------------------------------------- traces
def collect_items(
    ctx: Context,
    cfg: Config,
    features: Any | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
    trace_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[Item], dict[int, dict[str, list[str]]], dict[int, dict]]:
    steps = ctx.steps
    step_ids = [s.id for s in steps]
    needs_feats = cfg.scorer.name not in ("time_prior", "oracle", "vlm")
    if needs_feats and features is None:
        from sopverify.features.provider import FeatureProvider

        features = FeatureProvider(cfg, ctx.procedure.roi.box, ctx.procedure.roi.frame_size)
    items: list[Item] = []
    requires_by_seed: dict[int, dict[str, list[str]]] = {}
    scorer_info: dict[int, dict] = {}
    ctx.extras["demos_by_seed"] = {}
    for seed, demo_ids in enumerate(ctx.plan.demo_draws):
        req = active_requires(ctx, cfg, demo_ids)
        requires_by_seed[seed] = req
        demos = build_demo_data([ctx.records[d] for d in demo_ids], steps, features, need_features=needs_feats)
        ctx.extras["demos_by_seed"][seed] = demos
        kwargs = dict(scorer_kwargs or {})
        if cfg.scorer.name == "oracle":
            kwargs["gt_fn"] = oracle_gt_fn(ctx.records, steps, req)
        scorer = make_scorer(cfg, **kwargs)
        scorer.fit(demos, steps)
        scorer_info[seed] = {"demos": demo_ids, "info": getattr(scorer, "info", {})}
        label = scorer_label(cfg)
        h = cfg.stage_hash("scores", demos=demo_ids, roi=ctx.procedure.roi.box)
        for split in cfg.eval.splits:
            for eid in ctx.plan.eval_ids(split, seed, cfg.demos.held_out_participant):
                rec = ctx.records[eid]
                ref = rec.to_ref()
                if progress:
                    progress(f"seed {seed} {split} {eid}")
                if needs_feats:
                    times, feats = features.get(ref)
                else:
                    times, feats = make_grid(rec.duration_s, cfg.video.sample_fps), None
                trace = ScoreTrace(eid, times, np.clip(scorer.score(times, feats, ref), 0, 1), step_ids, label, h)
                if trace_dir is not None:
                    trace.save(trace_dir / f"seed{seed}" / f"{eid}.npz")
                seen = ctx.plan.is_seen(eid, seed)
                gt_full = compute_gt(rec, steps, req, cfg.checker.final_state_check)
                items.append(Item(seed, split, eid, "original", "original", seen, None, trace, rec.duration_s,
                                  gt_full, ref))
                for tr in make_truncations(gt_full, cfg.eval.truncation):
                    ref_t: ExecutionRef = ref.truncated(tr.cut_s)
                    if getattr(scorer, "duration_dependent", False):
                        n = int(np.searchsorted(times, tr.cut_s, side="left"))
                        sc = scorer.score(times[:n], None if feats is None else feats[:n], ref_t)
                        ttrace = ScoreTrace(eid, times[:n], np.clip(sc, 0, 1), step_ids, label, h,
                                            {"truncated_at_s": tr.cut_s})
                    else:
                        ttrace = trace.truncate(tr.cut_s)
                    gt_t = compute_gt(rec, steps, req, cfg.checker.final_state_check, t_end=tr.cut_s)
                    items.append(Item(seed, split, eid, tr.variant, "truncation", seen, tr.cut_s, ttrace,
                                      tr.cut_s, gt_t, ref_t))
    return items, requires_by_seed, scorer_info


# ---------------------------------------------------------------------- judging
def evaluate_items(
    items: list[Item], ctx: Context, cfg: Config, requires_by_seed: dict[int, dict[str, list[str]]],
    keep_reports: bool = False, verifier: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple, Any]]:
    steps = ctx.steps
    srows, vrows = [], []
    reports: dict[tuple, Any] = {}
    for it in items:
        req = requires_by_seed[it.seed]
        rep, est, res = judge(it.trace, steps, req, cfg, procedure_id=ctx.procedure.procedure_id,
                              step_set=ctx.step_set, duration_s=it.duration_s)
        if verifier is not None and it.ref is not None:
            ver = verifier.verify(it.ref, steps, est, res, ctx.extras.get("demos_by_seed", {}).get(it.seed, []),
                                  ctx.procedure)
            if ver:
                rep, est, res = judge(it.trace, steps, req, cfg, procedure_id=ctx.procedure.procedure_id,
                                      step_set=ctx.step_set, duration_s=it.duration_s,
                                      overrides={k: (v[0], v[1]) for k, v in ver.items()}, estimates=est)
                for k, v in ver.items():
                    rep.step(k).verifier = v[2]
        gt = it.gt
        assert gt is not None
        base = {"seed": it.seed, "split": it.split, "execution_id": it.execution_id, "variant": it.variant,
                "kind": it.kind, "seen": it.seen}
        for s in steps:
            r = rep.step(s.id)
            g = gt.steps[s.id]
            srows.append({**base, "step_id": s.id, "required": s.required, "verifiable": s.visually_verifiable,
                          "pred": r.verdict.value, "pred_completion_s": r.completion_time_s,
                          "gt_done": g.done, "gt_completion_s": g.completion_s,
                          "gt_order_violation": g.order_violation, "gt_rework": g.rework,
                          "confidence": r.confidence, "max_score": r.max_score})
        vrows.append({**base, "pred_video": rep.video_verdict.value, "gt_compliant": gt.compliant,
                      "duration_s": it.duration_s, "n_gt_not_done": len(gt.not_done()),
                      "n_gt_order_violations": gt.n_order_violations})
        if keep_reports:
            reports[(it.seed, it.execution_id, it.variant)] = rep
    return pd.DataFrame(srows), pd.DataFrame(vrows), reports


def _subsets(steps: pd.DataFrame, videos: pd.DataFrame, split: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    sv = videos[videos["split"] == split]
    ss = steps[steps["split"] == split]
    for kind in ("all", "original", "truncation"):
        kv = sv if kind == "all" else sv[sv["kind"] == kind]
        ks = ss if kind == "all" else ss[ss["kind"] == kind]
        out[kind] = {}
        for who in ("all", "seen", "unseen"):
            wv = kv if who == "all" else kv[kv["seen"] == (who == "seen")]
            ws = ks if who == "all" else ks[ks["seen"] == (who == "seen")]
            out[kind][who] = M.aggregate_over_seeds(ws, wv) if len(wv) else {}
    return out


# ---------------------------------------------------------------------- tuning
def tune_on_dev(items: list[Item], ctx: Context, cfg: Config, req: dict[int, dict[str, list[str]]]
                ) -> tuple[Config, list[dict]]:
    grid = cfg.eval.tune.grid
    if not cfg.eval.tune.enabled or not grid:
        return cfg, []
    for k in grid:
        if not (k.startswith("smoothing.") or k.startswith("checker.")):
            raise ValueError(f"tuning only supports smoothing.* and checker.* keys (cheap, trace-level): {k}")
    dev = [it for it in items if it.split == "dev"]
    if not dev:
        raise ValueError("tuning needs dev items (eval.splits must include dev)")
    keys = list(grid)
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        ov = dict(zip(keys, combo))
        try:
            c = cfg.with_overrides(ov)
            if c.smoothing.tau_off > c.smoothing.tau_on:
                continue
        except Exception:
            continue
        s, v, _ = evaluate_items(dev, ctx, c, req)
        agg = M.aggregate_over_seeds(s, v)
        h = {k: agg[k]["mean"] for k in agg}
        results.append({"overrides": ov, **h})
    t = cfg.targets

    def key(r: dict) -> tuple:
        fa, rv, rec = r["fa_per_compliant"], r["review_rate"], r["nd_recall"]
        fa = 99.0 if np.isnan(fa) else fa
        rec = -1.0 if np.isnan(rec) else rec
        viol = max(0.0, fa - t.max_false_alarms_per_compliant) + max(0.0, rv - t.max_review_rate)
        f1 = -1.0 if np.isnan(r["nd_f1"]) else r["nd_f1"]
        return (viol > 0, viol, -rec, -f1, fa)

    results.sort(key=key)
    best = results[0]["overrides"] if results else {}
    log.info("tuning: best on dev %s (of %d combos)", best, len(results))
    return cfg.with_overrides(best), results


# ---------------------------------------------------------------------- run
def run_variant(
    cfg: Config,
    ctx: Context | None = None,
    out_root: str | Path | None = None,
    features: Any | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
    experiment: str = "",
    variant: str = "default",
    progress: Callable[[str], None] | None = None,
    vlm_backend: Any | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    ctx = ctx or build_context(cfg)
    run_name = cfg.run_name or f"{experiment or 'run'}__{variant}"
    out = Path(out_root or cfg.paths_resolved()["runs_root"]) / run_name  # type: ignore[arg-type]
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = out / "traces" if cfg.eval.save_traces else None
    scorer_kwargs = dict(scorer_kwargs or {})
    verifier = None
    if cfg.scorer.name == "vlm" or (cfg.vlm.enabled and cfg.vlm.verifier):
        from sopverify.vlm.backend import make_backend
        from sopverify.vlm.frames import VideoFrameSource

        backend = vlm_backend or make_backend(cfg.vlm)
        frames = VideoFrameSource(ctx.procedure.roi.box, ctx.procedure.roi.frame_size, cfg.vlm.max_long_side)
        if cfg.scorer.name == "vlm":
            scorer_kwargs.setdefault("backend", backend)
            scorer_kwargs.setdefault("frames", frames)
            scorer_kwargs.setdefault("log_path", out / "vlm_scorer_log.jsonl")
        if cfg.vlm.enabled and cfg.vlm.verifier:
            from sopverify.vlm.verifier import VLMVerifier

            verifier = VLMVerifier(cfg, backend, frames, out / "vlm_verifier_log.jsonl")
    items, req, sinfo = collect_items(ctx, cfg, features, scorer_kwargs, trace_dir, progress)
    t_scored = time.time()
    final_cfg, tuning = tune_on_dev(items, ctx, cfg, req)
    t_v = time.time()
    steps_df, videos_df, reports = evaluate_items(items, ctx, final_cfg, req, keep_reports=True, verifier=verifier)
    verifier_s = time.time() - t_v if verifier is not None else 0.0

    results = {split: _subsets(steps_df, videos_df, split) for split in final_cfg.eval.splits}
    detailed = {}
    for split in final_cfg.eval.splits:
        detailed[split] = M.compute_all(steps_df[steps_df["split"] == split], videos_df[videos_df["split"] == split])
    gts = [it.gt for it in items if it.kind == "original" and it.seed == 0 and it.gt is not None]
    gts_t = [it.gt for it in items if it.kind == "truncation" and it.seed == 0 and it.gt is not None]
    dev_head = {k: v["mean"] for k, v in results.get("dev", {}).get("all", {}).get("all", {}).items()}
    metrics = {
        "experiment": experiment, "variant": variant, "run_name": run_name,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "scorer": scorer_label(final_cfg), "step_set": ctx.step_set,
        "graph_source": final_cfg.graph.source or ctx.procedure.spec.graph_source,
        "oracle_procedure_knowledge": (final_cfg.graph.source == "inferred_all"),
        "config_hash": final_cfg.stage_hash("scores"),
        "n_seeds": len(ctx.plan.demo_draws), "k": final_cfg.demos.k,
        "results": results, "detailed_pooled_seeds": detailed,
        "targets_dev": M.meets_targets(dev_head, final_cfg.targets) if dev_head else {},
        "tuning": {"enabled": bool(tuning), "best": final_cfg.model_dump(include={"smoothing", "checker"}),
                   "n_combos": len(tuning), "top": tuning[:10]},
        "gt_diagnostics": {"originals": diagnostics(gts), "truncations": diagnostics(gts_t)},
        "graph": req.get(0, {}),
        "runtime_s": {"scoring": round(t_scored - t0, 2), "total": round(time.time() - t0, 2),
                      "per_video_scoring": round((t_scored - t0) / max(1, sum(i.kind == 'original' for i in items)), 3),
                      "verifier": round(verifier_s, 2)},
        "vlm": {"scorer_stats": {s: v.get("info", {}).get("stats") for s, v in sinfo.items()},
                "verifier_stats": verifier.stats if verifier is not None else None},
        "scorer_info": sinfo,
    }
    write_yaml(out / "config.yaml", final_cfg.model_dump(mode="json"))
    write_json(out / "metrics.json", metrics)
    write_json(out / "split_plan.json", ctx.plan.to_dict())
    (out / "git_commit.txt").write_text(git_commit() + "\n")
    test_split = "test" if "test" in final_cfg.eval.splits else final_cfg.eval.splits[0]
    M.per_step_breakdown(steps_df[steps_df["split"] == test_split]).to_csv(out / "per_step.csv", index=False)
    if "dev" in final_cfg.eval.splits:
        M.per_step_breakdown(steps_df[steps_df["split"] == "dev"]).to_csv(out / "per_step_dev.csv", index=False)
    steps_df.to_csv(out / "predictions_steps.csv", index=False)
    videos_df.to_csv(out / "predictions_videos.csv", index=False)
    if tuning:
        pd.DataFrame([{**r["overrides"], **{k: v for k, v in r.items() if k != "overrides"}} for r in tuning]
                     ).to_csv(out / "tuning_dev.csv", index=False)
    for (seed, eid, var), rep in reports.items():
        if var == "original":
            rep.git_commit = git_commit()
            rep.save(out / "verdicts" / f"seed{seed}" / f"{eid}.json")
    log.info("run %s written to %s (%.1fs)", run_name, out, time.time() - t0)
    return metrics


def run_experiment(
    path: str | Path,
    only: list[str] | None = None,
    out_root: str | Path | None = None,
    catalog: Catalog | None = None,
    features_factory: Callable[[Config, Procedure], Any] | None = None,
    scorer_kwargs_factory: Callable[[Config, Context], dict[str, Any]] | None = None,
    progress: Callable[[str], None] | None = None,
    vlm_backend: Any | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    exp = load_experiment(path, only, overrides)
    out: dict[str, dict[str, Any]] = {}
    ctx_cache: dict[tuple, Context] = {}
    for vname, cfg in exp.variants.items():
        log.info("=== %s / %s", exp.name, vname)
        key = (cfg.procedure, cfg.step_set, cfg.model_dump_json(include={"demos", "splits", "data", "paths"}))
        if key not in ctx_cache:
            ctx_cache[key] = build_context(cfg, catalog)
        ctx = ctx_cache[key]
        ctx = Context(cfg, ctx.procedure, ctx.catalog, ctx.step_set, ctx.records, ctx.plan)
        feats = features_factory(cfg, ctx.procedure) if features_factory else None
        skw = scorer_kwargs_factory(cfg, ctx) if scorer_kwargs_factory else None
        needs_vlm = cfg.scorer.name == "vlm" or (cfg.vlm.enabled and cfg.vlm.verifier)
        if needs_vlm and vlm_backend is None:
            from sopverify.vlm.backend import make_backend

            vlm_backend = make_backend(cfg.vlm)  # loaded once, shared by all variants
        out[vname] = run_variant(cfg, ctx, out_root, feats, skw, exp.name, vname, progress, vlm_backend)
    return out
