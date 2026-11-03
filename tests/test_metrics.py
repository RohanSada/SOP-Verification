import math

import pandas as pd
import pytest

from sopverify.config import TargetsConfig
from sopverify.eval import metrics as M


def rows(spec):
    """spec: list of (exec, step, pred, gt_done, pred_c, gt_c, gt_ov)."""
    out = []
    for e, s, p, gd, pc, gc, ov in spec:
        out.append({"seed": 0, "split": "test", "execution_id": e, "variant": "original", "kind": "original",
                    "seen": False, "step_id": s, "required": True, "verifiable": True, "pred": p,
                    "pred_completion_s": pc, "gt_done": gd, "gt_completion_s": gc, "gt_order_violation": ov,
                    "confidence": 0.9})
    return pd.DataFrame(out)


STEPS = rows([
    ("e1", "A", "DONE", True, 10.0, 12.0, False),
    ("e1", "B", "DONE", True, 50.0, 30.0, False),
    ("e1", "C", "NOT_DONE", False, None, None, False),
    ("e2", "A", "NOT_DONE", True, None, 10.0, False),        # false alarm
    ("e2", "B", "UNCERTAIN", False, None, None, False),      # undecided miss
    ("e2", "C", "OUT_OF_ORDER", True, 40.0, 41.0, True),
    ("e3", "A", "DONE", False, 5.0, None, False),            # missed not-done
    ("e3", "B", "OUT_OF_ORDER", True, 20.0, 22.0, False),    # false order alarm
    ("e3", "C", "DONE", True, 70.0, 74.0, True),             # missed order violation
])
VIDEOS = pd.DataFrame([
    {"seed": 0, "split": "test", "execution_id": "e1", "variant": "original", "kind": "original", "seen": False,
     "pred_video": "NON_COMPLIANT", "gt_compliant": False},
    {"seed": 0, "split": "test", "execution_id": "e2", "variant": "original", "kind": "original", "seen": False,
     "pred_video": "NEEDS_REVIEW", "gt_compliant": False},
    {"seed": 0, "split": "test", "execution_id": "e4", "variant": "original", "kind": "original", "seen": False,
     "pred_video": "COMPLIANT", "gt_compliant": True},
])


def test_step_metrics_hand_computed():
    m = M.step_metrics(STEPS)
    nd = m["not_done"]
    assert (nd["tp"], nd["fp"], nd["fn"]) == (1, 1, 1)
    assert nd["precision"] == 0.5 and nd["recall"] == 0.5
    strict = m["not_done_strict"]
    assert (strict["tp"], strict["fp"], strict["fn"]) == (1, 1, 2)
    d = m["done"]
    assert (d["tp"], d["fp"], d["fn"]) == (5, 1, 1)
    assert m["coverage"] == pytest.approx(8 / 9)
    assert m["timing"]["n"] == 5
    assert m["timing"]["mae_s"] == pytest.approx((2 + 20 + 1 + 2 + 4) / 5)
    assert m["timing"]["within_5s"] == pytest.approx(4 / 5)
    o = m["order"]
    assert (o["tp"], o["fp"], o["fn"], o["support"], o["low_support"]) == (1, 1, 1, 2, True)


def test_video_metrics_and_false_alarms():
    v = M.video_metrics(VIDEOS)
    assert v["confusion"]["NEEDS_REVIEW"]["gt_non_compliant"] == 1
    assert v["accuracy_decided"] == 1.0 and v["review_rate"] == pytest.approx(1 / 3)
    steps = pd.concat([STEPS, rows([("e4", "A", "DONE", True, 1.0, 1.0, False),
                                    ("e4", "B", "UNCERTAIN", True, None, 2.0, False)])])
    assert M.false_alarms_per_compliant(VIDEOS, steps) == 1.0


def test_per_step_breakdown():
    b = M.per_step_breakdown(STEPS).set_index("step_id")
    assert b.loc["A", "false_alarm_rate"] == 0.5 and b.loc["C", "not_done_recall"] == 1.0
    assert math.isnan(b.loc["B", "not_done_recall"]) is False


def test_aggregate_and_targets():
    s2 = STEPS.assign(seed=1)
    v2 = VIDEOS.assign(seed=1)
    agg = M.aggregate_over_seeds(pd.concat([STEPS, s2]), pd.concat([VIDEOS, v2]))
    assert agg["nd_recall"]["mean"] == 0.5 and agg["nd_recall"]["std"] == 0.0 and agg["nd_recall"]["n_seeds"] == 2
    h = {k: v["mean"] for k, v in agg.items()}
    t = M.meets_targets(h, TargetsConfig())
    assert t["not_done_recall"] is False
