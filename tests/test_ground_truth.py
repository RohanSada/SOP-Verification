import pytest

from sopverify.config import TruncationConfig
from sopverify.eval.ground_truth import compute_gt, diagnostics, pool_completions
from sopverify.eval.truncation import make_truncations
from sopverify.procedure.gt_mapping import step_completion
from sopverify.procedure.schema import GTMapping, Step

from helpers import chain_steps, make_record

REQ = {"A": [], "B": ["A"], "C": ["B"], "D": ["A"]}


def test_basic_gt():
    gt = compute_gt(make_record(), chain_steps(), REQ)
    assert gt.compliant and all(s.done for s in gt.steps.values())
    assert gt.steps["C"].completion_s == 110.0 and gt.n_order_violations == 0


def test_never_completed_and_order_violation():
    rec = make_record(installs={"a": 50.0, "b": 30.0, "d": 70.0})  # C never, B before A
    gt = compute_gt(rec, chain_steps(), REQ)
    assert gt.steps["C"].completion_s is None and not gt.steps["C"].done
    assert gt.steps["B"].order_violation and gt.steps["B"].violated == ["A"]
    assert not gt.compliant


def test_done_without_prerequisite_is_violation():
    rec = make_record(installs={"b": 30.0, "c": 50.0, "d": 70.0})
    gt = compute_gt(rec, chain_steps(), REQ)
    assert gt.steps["B"].order_violation and gt.steps["D"].order_violation and not gt.steps["C"].order_violation


def test_rework_final_completion():
    rec = make_record(installs={"a": [20.0, 90.0], "b": 100.0, "c": 110.0, "d": 150.0}, removes={"a": [40.0]})
    gt = compute_gt(rec, chain_steps(), REQ)
    assert gt.steps["A"].rework and gt.steps["A"].completion_s == 90.0 and gt.steps["A"].done


def test_undone_at_end_not_done():
    rec = make_record(installs={"a": 20.0, "b": 60.0, "c": 110.0, "d": 150.0}, removes={"d": [180.0]})
    gt = compute_gt(rec, chain_steps(), REQ)
    assert gt.steps["D"].completion_s == 150.0 and not gt.steps["D"].done and not gt.compliant


def test_asr_final_state_check():
    rec = make_record(final_states={"c": -1})
    gt = compute_gt(rec, chain_steps(), REQ)
    assert not gt.steps["C"].done and gt.final_misinstalled
    gt2 = compute_gt(make_record(final_states={"a": 1}), chain_steps(), REQ, final_state_check=True)
    assert gt2.compliant


def test_truncated_gt():
    gt = compute_gt(make_record(), chain_steps(), REQ, t_end=100.0)
    assert [k for k, v in gt.steps.items() if not v.done] == ["C", "D"] and gt.t_end == 100.0


def test_tas_completion_is_end_frame():
    rec = make_record(tas=[("X", 0.0, 10.0), ("X", 20.0, 30.0), ("Y", 30.0, 200.0)])
    s = Step(id="T", name="t", gt_mapping=GTMapping(tas_s_labels=["X"]))
    assert step_completion(rec, s).completion_s == pytest.approx(30.0 - 1 / 30)
    assert step_completion(rec, s, t_end=25.0).completion_s == pytest.approx(10.0 - 1 / 30)
    y = Step(id="Y", name="y", gt_mapping=GTMapping(tas_s_labels=["Y"]))
    assert step_completion(rec, y).completion_s == pytest.approx(200.0 - 1 / 30)
    assert step_completion(rec, y, t_end=150.0).completion_s is None  # still running at the cut


@pytest.mark.parametrize("m", [1, 2, 3])
def test_truncation_leaves_m_incomplete(m):
    rec = make_record()
    full = compute_gt(rec, chain_steps(), REQ)
    trs = make_truncations(full, TruncationConfig(n_per_video=1, m_choices=[m], min_gap_s=5.0))
    assert len(trs) == 1
    tr = trs[0]
    gt = compute_gt(rec, chain_steps(), REQ, t_end=tr.cut_s)
    assert len(gt.not_done()) == m and not gt.compliant
    assert all(abs(tr.cut_s - c) >= 5.0 for c in (20, 60, 110, 150))


def test_truncation_deterministic_and_skips_impossible():
    rec = make_record(installs={"a": 20.0, "b": 22.0, "c": 110.0, "d": 150.0})
    full = compute_gt(rec, chain_steps(), REQ)
    cfg = TruncationConfig(n_per_video=3, m_choices=[1, 2, 3], min_gap_s=5.0)
    a, b = make_truncations(full, cfg), make_truncations(full, cfg)
    assert [t.cut_s for t in a] == [t.cut_s for t in b]
    assert 3 not in [t.m for t in a]  # A and B 2 s apart: no valid cut between them


def test_diagnostics_and_pool_completions():
    recs = [make_record(), make_record(installs={"a": 50.0, "b": 30.0})]
    gts = [compute_gt(r, chain_steps(), REQ) for r in recs]
    d = diagnostics(gts)
    assert d["executions"] == 2 and d["compliant"] == 1 and d["order_violation_instances"] >= 1
    assert pool_completions(recs, chain_steps())[1]["C"] is None
