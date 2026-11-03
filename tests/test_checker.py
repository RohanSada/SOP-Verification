"""Table-driven tests for the compliance checker (spec 8.8 requires >= 20 cases)."""

from __future__ import annotations

import pytest

from helpers import step
from sopverify.compliance.checker import check
from sopverify.compliance.verdicts import StepVerdict as V
from sopverify.compliance.verdicts import VideoVerdict as VV
from sopverify.config import CheckerConfig
from sopverify.temporal.smoothing import StepEstimate

CFG = CheckerConfig(tau_low=0.3, order_tolerance_s=5.0)

CHAIN = {"A": [], "B": ["A"], "C": ["B"], "D": ["A"]}
DIAMOND = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}


def done(t):
    return ("done", t)


NOT = ("max", 0.1)
UNC = ("max", 0.5)


def est(sid, spec):
    if spec is None:
        return None
    kind, val = spec[0], spec[1]
    rework = len(spec) > 2 and spec[2]
    if kind == "done":
        return StepEstimate(sid, True, float(val), 0.9, 0.95, float(val), [(float(val), 999.0)], rework)
    return StepEstimate(sid, False, None, 1 - val, float(val), 10.0, [], rework)


# (name, graph, estimates, expected verdicts, expected video verdict, options)
CASES = [
    ("all_done_in_order", CHAIN, {"A": done(10), "B": done(30), "C": done(50), "D": done(40)},
     {"A": V.DONE, "B": V.DONE, "C": V.DONE, "D": V.DONE}, VV.COMPLIANT, {}),
    ("all_not_done", CHAIN, {k: NOT for k in "ABCD"}, {k: V.NOT_DONE for k in "ABCD"}, VV.NON_COMPLIANT, {}),
    ("all_uncertain", CHAIN, {k: UNC for k in "ABCD"}, {k: V.UNCERTAIN for k in "ABCD"}, VV.NEEDS_REVIEW, {}),
    ("child_before_parent", CHAIN, {"A": done(100), "B": done(50), "C": done(120), "D": done(110)},
     {"A": V.DONE, "B": V.OUT_OF_ORDER, "C": V.DONE, "D": V.DONE}, VV.NON_COMPLIANT, {"violated": {"B": ["A"]}}),
    ("within_tolerance", CHAIN, {"A": done(100), "B": done(98), "C": done(120), "D": done(110)},
     {"A": V.DONE, "B": V.DONE, "C": V.DONE, "D": V.DONE}, VV.COMPLIANT, {}),
    ("tie_at_tolerance_boundary", CHAIN, {"A": done(100), "B": done(95), "C": done(120), "D": done(110)},
     {"A": V.DONE, "B": V.DONE, "C": V.DONE, "D": V.DONE}, VV.COMPLIANT, {}),
    ("just_past_tolerance", CHAIN, {"A": done(100), "B": done(94.9), "C": done(120), "D": done(110)},
     {"A": V.DONE, "B": V.OUT_OF_ORDER, "C": V.DONE, "D": V.DONE}, VV.NON_COMPLIANT, {}),
    ("missing_prerequisite", CHAIN, {"A": NOT, "B": done(30), "C": done(50), "D": NOT},
     {"A": V.NOT_DONE, "B": V.OUT_OF_ORDER, "C": V.DONE, "D": V.NOT_DONE}, VV.NON_COMPLIANT,
     {"violated": {"B": ["A"], "C": []}}),
    ("uncertain_prerequisite", CHAIN, {"A": UNC, "B": done(30), "C": done(50), "D": done(40)},
     {"A": V.UNCERTAIN, "B": V.DONE, "C": V.DONE, "D": V.DONE}, VV.NEEDS_REVIEW,
     {"flags": {"B": ["prereq_uncertain"], "D": ["prereq_uncertain"]}}),
    ("diamond_one_branch_late", DIAMOND, {"A": done(10), "B": done(20), "C": done(80), "D": done(50)},
     {"A": V.DONE, "B": V.DONE, "C": V.DONE, "D": V.OUT_OF_ORDER}, VV.NON_COMPLIANT, {"violated": {"D": ["C"]}}),
    ("diamond_both_branches_missing", DIAMOND, {"A": done(10), "B": NOT, "C": NOT, "D": done(50)},
     {"A": V.DONE, "B": V.NOT_DONE, "C": V.NOT_DONE, "D": V.OUT_OF_ORDER}, VV.NON_COMPLIANT,
     {"violated": {"D": ["B", "C"]}}),
    ("diamond_all_fine", DIAMOND, {"A": done(10), "B": done(30), "C": done(20), "D": done(50)},
     {"A": V.DONE, "B": V.DONE, "C": V.DONE, "D": V.DONE}, VV.COMPLIANT, {}),
    ("not_verifiable_ignored", CHAIN, {"A": done(10), "B": NOT, "C": done(50), "D": done(40)},
     {"A": V.DONE, "B": V.NOT_VERIFIABLE, "C": V.DONE, "D": V.DONE}, VV.COMPLIANT,
     {"nv": {"B"}, "flags": {"C": ["prereq_not_verifiable"]}}),
    ("look_through_not_verifiable", CHAIN, {"A": NOT, "B": done(30), "C": done(50), "D": NOT},
     {"A": V.NOT_DONE, "B": V.NOT_VERIFIABLE, "C": V.OUT_OF_ORDER, "D": V.NOT_DONE}, VV.NON_COMPLIANT,
     {"nv": {"B"}, "violated": {"C": ["A"]}}),
    ("look_through_order", CHAIN, {"A": done(100), "B": done(30), "C": done(50), "D": done(120)},
     {"A": V.DONE, "B": V.NOT_VERIFIABLE, "C": V.OUT_OF_ORDER, "D": V.DONE}, VV.NON_COMPLIANT,
     {"nv": {"B"}, "violated": {"C": ["A"]}}),
    ("optional_not_done", CHAIN, {"A": done(10), "B": done(30), "C": done(50), "D": NOT},
     {"A": V.DONE, "B": V.DONE, "C": V.DONE, "D": V.NOT_DONE}, VV.COMPLIANT, {"optional": {"D"}}),
    ("optional_uncertain", CHAIN, {"A": done(10), "B": done(30), "C": done(50), "D": UNC},
     {"A": V.DONE, "B": V.DONE, "C": V.DONE, "D": V.UNCERTAIN}, VV.COMPLIANT, {"optional": {"D"}}),
    ("not_done_beats_uncertain", CHAIN, {"A": done(10), "B": UNC, "C": NOT, "D": done(40)},
     {"A": V.DONE, "B": V.UNCERTAIN, "C": V.NOT_DONE, "D": V.DONE}, VV.NON_COMPLIANT, {}),
    ("max_score_equal_tau_low_is_uncertain", CHAIN, {"A": done(10), "B": done(30), "C": ("max", 0.3), "D": done(40)},
     {"A": V.DONE, "B": V.DONE, "C": V.UNCERTAIN, "D": V.DONE}, VV.NEEDS_REVIEW, {}),
    ("max_score_below_tau_low_is_not_done", CHAIN, {"A": done(10), "B": done(30), "C": ("max", 0.2999), "D": done(40)},
     {"A": V.DONE, "B": V.DONE, "C": V.NOT_DONE, "D": V.DONE}, VV.NON_COMPLIANT, {}),
    ("missing_estimate_is_uncertain", CHAIN, {"A": done(10), "B": done(30), "C": None, "D": done(40)},
     {"A": V.DONE, "B": V.DONE, "C": V.UNCERTAIN, "D": V.DONE}, VV.NEEDS_REVIEW, {}),
    ("verifier_override_done_then_order", CHAIN, {"A": UNC, "B": done(40), "C": done(60), "D": done(70)},
     {"A": V.DONE, "B": V.OUT_OF_ORDER, "C": V.DONE, "D": V.DONE}, VV.NON_COMPLIANT,
     {"overrides": {"A": (V.DONE, 50.0)}, "violated": {"B": ["A"]}}),
    ("verifier_override_not_done", CHAIN, {"A": UNC, "B": done(40), "C": done(60), "D": done(70)},
     {"A": V.NOT_DONE, "B": V.OUT_OF_ORDER, "C": V.DONE, "D": V.OUT_OF_ORDER}, VV.NON_COMPLIANT,
     {"overrides": {"A": (V.NOT_DONE, None)}}),
    ("rework_flag_propagates", CHAIN, {"A": ("done", 10, True), "B": done(30), "C": done(50), "D": done(40)},
     {"A": V.DONE, "B": V.DONE, "C": V.DONE, "D": V.DONE}, VV.COMPLIANT, {"flags": {"A": ["possible_rework"]}}),
    ("provisional_statuses_used", CHAIN, {"A": done(100), "B": done(50), "C": done(60), "D": done(110)},
     {"A": V.DONE, "B": V.OUT_OF_ORDER, "C": V.DONE, "D": V.DONE}, VV.NON_COMPLIANT, {}),
    ("no_graph_no_order_violations", {k: [] for k in "ABCD"}, {"A": done(100), "B": done(5), "C": done(1), "D": done(3)},
     {k: V.DONE for k in "ABCD"}, VV.COMPLIANT, {}),
]


@pytest.mark.parametrize("name,graph,ests,expected,video,opts", CASES, ids=[c[0] for c in CASES])
def test_checker_cases(name, graph, ests, expected, video, opts):
    nv = opts.get("nv", set())
    optional = opts.get("optional", set())
    steps = [step(k, graph[k], required=k not in optional, verifiable=k not in nv) for k in graph]
    estimates = {k: e for k, spec in ests.items() if (e := est(k, spec)) is not None}
    res = check(steps, estimates, graph, CFG, opts.get("overrides"))
    assert {k: v.verdict for k, v in res.steps.items()} == expected
    assert res.video_verdict == video
    for k, vio in opts.get("violated", {}).items():
        assert sorted(res.steps[k].violated_prerequisites) == sorted(vio)
    for k, flags in opts.get("flags", {}).items():
        for f in flags:
            assert f in res.steps[k].flags


def test_case_count():
    assert len(CASES) >= 20


def test_result_independent_of_step_order():
    graph = DIAMOND
    ests = {"A": done(10), "B": done(80), "C": done(20), "D": done(50)}
    s1 = [step(k, graph[k]) for k in "ABCD"]
    s2 = [step(k, graph[k]) for k in "DCBA"]
    e = {k: est(k, v) for k, v in ests.items()}
    r1, r2 = check(s1, e, graph, CFG), check(s2, e, graph, CFG)
    assert r1.verdicts() == r2.verdicts()
