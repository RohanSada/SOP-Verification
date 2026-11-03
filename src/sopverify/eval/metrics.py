"""Evaluation metrics (spec 9.5) over flat prediction tables.

Step table columns: seed, split, execution_id, variant, kind ('original'|'truncation'), seen,
  step_id, required, verifiable, pred, pred_completion_s, gt_done, gt_completion_s,
  gt_order_violation, confidence
Video table columns: seed, split, execution_id, variant, kind, seen, pred_video, gt_compliant

Only visually verifiable steps enter the step metrics. "Not done" is the positive class;
OUT_OF_ORDER counts as done. Two variants: on decided steps (UNCERTAIN excluded) and strict
(UNCERTAIN counted as the wrong answer).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DONE_LIKE = ("DONE", "OUT_OF_ORDER")
MIN_ORDER_SUPPORT = 10


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (tp + fp and tp + fn and (p + r) > 0) else float("nan")
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def step_metrics(steps: pd.DataFrame) -> dict[str, Any]:
    df = steps[steps["verifiable"]]
    out: dict[str, Any] = {"n_step_instances": int(len(df))}
    if df.empty:
        return out
    pred = df["pred"].to_numpy()
    gt_done = df["gt_done"].to_numpy(dtype=bool)
    gt_nd = ~gt_done
    unc = pred == "UNCERTAIN"
    dec = ~unc
    nd_pred = pred == "NOT_DONE"
    done_pred = np.isin(pred, DONE_LIKE)

    # Not-done detection, decided steps only.
    out["not_done"] = _prf(int((nd_pred & gt_nd & dec).sum()), int((nd_pred & gt_done & dec).sum()),
                           int((done_pred & gt_nd).sum()))
    # Strict: UNCERTAIN counts as wrong (a miss for GT not-done, a false alarm for GT done).
    out["not_done_strict"] = _prf(int((nd_pred & gt_nd).sum()), int(((nd_pred | unc) & gt_done).sum()),
                                  int(((done_pred | unc) & gt_nd).sum()))
    out["done"] = _prf(int((done_pred & gt_done).sum()), int((done_pred & gt_nd).sum()),
                       int((nd_pred & gt_done).sum()))
    out["done_strict"] = _prf(int((done_pred & gt_done).sum()), int((done_pred & gt_nd).sum()),
                              int(((nd_pred | unc) & gt_done).sum()))
    out["coverage"] = float(dec.mean())
    out["n_gt_not_done"] = int(gt_nd.sum())

    # Timing on true-positive DONE steps.
    tp = done_pred & gt_done & df["pred_completion_s"].notna().to_numpy() & df["gt_completion_s"].notna().to_numpy()
    err = np.abs(df["pred_completion_s"].to_numpy(dtype=float)[tp] - df["gt_completion_s"].to_numpy(dtype=float)[tp])
    out["timing"] = {
        "n": int(tp.sum()),
        "mae_s": float(err.mean()) if len(err) else float("nan"),
        "median_s": float(np.median(err)) if len(err) else float("nan"),
        **{f"within_{w}s": float((err <= w).mean()) if len(err) else float("nan") for w in (5, 15, 30)},
    }

    # Order.
    ooo = pred == "OUT_OF_ORDER"
    gov = df["gt_order_violation"].to_numpy(dtype=bool)
    order = _prf(int((ooo & gov).sum()), int((ooo & ~gov).sum()), int((~ooo & gov).sum()))
    order["support"] = int(gov.sum())
    order["low_support"] = bool(gov.sum() < MIN_ORDER_SUPPORT)
    out["order"] = order
    return out


def video_metrics(videos: pd.DataFrame, steps: pd.DataFrame | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"n_videos": int(len(videos))}
    if videos.empty:
        return out
    preds = ["COMPLIANT", "NON_COMPLIANT", "NEEDS_REVIEW"]
    conf = {p: {"gt_compliant": 0, "gt_non_compliant": 0} for p in preds}
    for p, g in zip(videos["pred_video"], videos["gt_compliant"]):
        conf[p]["gt_compliant" if g else "gt_non_compliant"] += 1
    out["confusion"] = conf
    decided = videos[videos["pred_video"] != "NEEDS_REVIEW"]
    correct = (decided["pred_video"] == "COMPLIANT") == decided["gt_compliant"].astype(bool)
    out["accuracy_decided"] = float(correct.mean()) if len(decided) else float("nan")
    out["review_rate"] = float((videos["pred_video"] == "NEEDS_REVIEW").mean())
    out["n_gt_compliant"] = int(videos["gt_compliant"].sum())
    out["n_gt_non_compliant"] = int((~videos["gt_compliant"].astype(bool)).sum())
    # Non-compliant videos caught (predicted NON_COMPLIANT).
    nc = videos[~videos["gt_compliant"].astype(bool)]
    out["non_compliant_recall"] = float((nc["pred_video"] == "NON_COMPLIANT").mean()) if len(nc) else float("nan")
    if steps is not None and not steps.empty:
        out["false_alarms_per_compliant"] = false_alarms_per_compliant(videos, steps)
    return out


def false_alarms_per_compliant(videos: pd.DataFrame, steps: pd.DataFrame) -> float:
    """Mean number of non-DONE verdicts (verifiable steps) per GT-compliant video."""
    keys = ["seed", "execution_id", "variant"]
    comp = videos[videos["gt_compliant"].astype(bool)][keys]
    if comp.empty:
        return float("nan")
    s = steps[steps["verifiable"]].merge(comp, on=keys)
    per = s.assign(fa=s["pred"] != "DONE").groupby(keys)["fa"].sum()
    per = per.reindex(pd.MultiIndex.from_frame(comp), fill_value=0)
    return float(per.mean())


def per_step_breakdown(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = steps[steps["verifiable"]]
    for sid, g in df.groupby("step_id", sort=False):
        gt_nd = ~g["gt_done"].astype(bool)
        gt_d = g["gt_done"].astype(bool)
        pred = g["pred"]
        tp = pred.isin(DONE_LIKE) & gt_d & g["pred_completion_s"].notna() & g["gt_completion_s"].notna()
        err = (g.loc[tp, "pred_completion_s"] - g.loc[tp, "gt_completion_s"]).abs()
        rows.append({
            "step_id": sid,
            "n": int(len(g)),
            "n_gt_not_done": int(gt_nd.sum()),
            "not_done_recall": float(((pred == "NOT_DONE") & gt_nd).sum() / gt_nd.sum()) if gt_nd.sum() else float("nan"),
            "false_alarm_rate": float(((pred != "DONE") & gt_d).sum() / gt_d.sum()) if gt_d.sum() else float("nan"),
            "uncertain_rate": float((pred == "UNCERTAIN").mean()),
            "out_of_order_rate": float((pred == "OUT_OF_ORDER").mean()),
            "timing_mae_s": float(err.mean()) if len(err) else float("nan"),
        })
    return pd.DataFrame(rows)


def compute_all(steps: pd.DataFrame, videos: pd.DataFrame) -> dict[str, Any]:
    return {"step": step_metrics(steps), "video": video_metrics(videos, steps)}


def headline(m: dict[str, Any]) -> dict[str, float]:
    """Flat dict of the numbers used in tables and target checks."""
    s, v = m.get("step", {}), m.get("video", {})
    g = lambda d, *ks: _dig(d, ks)  # noqa: E731
    return {
        "nd_precision": g(s, "not_done", "precision"),
        "nd_recall": g(s, "not_done", "recall"),
        "nd_f1": g(s, "not_done", "f1"),
        "nd_recall_strict": g(s, "not_done_strict", "recall"),
        "done_precision": g(s, "done", "precision"),
        "done_recall": g(s, "done", "recall"),
        "coverage": s.get("coverage", float("nan")),
        "timing_mae_s": g(s, "timing", "mae_s"),
        "within_15s": g(s, "timing", "within_15s"),
        "order_precision": g(s, "order", "precision"),
        "order_recall": g(s, "order", "recall"),
        "order_support": g(s, "order", "support"),
        "fa_per_compliant": v.get("false_alarms_per_compliant", float("nan")),
        "video_acc_decided": v.get("accuracy_decided", float("nan")),
        "review_rate": v.get("review_rate", float("nan")),
        "nc_video_recall": v.get("non_compliant_recall", float("nan")),
    }


def _dig(d: dict, ks: tuple) -> float:
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return float("nan")
        d = d[k]
    return float(d) if d is not None else float("nan")


def aggregate_over_seeds(steps: pd.DataFrame, videos: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Headline metrics per seed, then mean and std across seeds."""
    per_seed = []
    for seed in sorted(videos["seed"].unique()):
        per_seed.append(headline(compute_all(steps[steps["seed"] == seed], videos[videos["seed"] == seed])))
    if not per_seed:
        return {}
    keys = per_seed[0].keys()
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([p[k] for p in per_seed], dtype=float)
        ok = ~np.isnan(vals)
        out[k] = {
            "mean": float(vals[ok].mean()) if ok.any() else float("nan"),
            "std": float(vals[ok].std(ddof=0)) if ok.any() else float("nan"),
            "n_seeds": int(ok.sum()),
        }
    return out


def meets_targets(h: dict[str, float], targets) -> dict[str, bool]:
    return {
        "not_done_recall": bool(h["nd_recall"] >= targets.not_done_recall),
        "false_alarms": bool(h["fa_per_compliant"] <= targets.max_false_alarms_per_compliant)
        if not np.isnan(h["fa_per_compliant"]) else False,
        "review_rate": bool(h["review_rate"] <= targets.max_review_rate),
    }
