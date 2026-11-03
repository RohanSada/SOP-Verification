import numpy as np

from sopverify.compliance.verdicts import StepResult, StepVerdict, Summary, VerdictReport, VideoVerdict
from sopverify.state.types import ScoreTrace


def test_verdict_report_roundtrip(tmp_path):
    steps = [StepResult(step_id="G1", name="x", verdict=StepVerdict.DONE, completion_time_s=4.5, confidence=0.9),
             StepResult(step_id="G2", name="y", verdict=StepVerdict.OUT_OF_ORDER, violated_prerequisites=["G1"],
                        flags=["possible_rework"])]
    r = VerdictReport(execution_id="e", procedure_id="p", step_set="grouped", video_verdict=VideoVerdict.NON_COMPLIANT,
                      summary=Summary.of(steps), steps=steps, scorer="s", graph={"G2": ["G1"]})
    p = r.save(tmp_path / "v.json")
    r2 = VerdictReport.load(p)
    assert r2 == r and r2.summary.out_of_order == 1
    import json
    d = json.loads(p.read_text())
    for key in ("execution_id", "procedure_id", "step_set", "video_verdict", "summary", "steps", "scorer",
                "config_hash", "git_commit", "runtime_s"):
        assert key in d
    assert set(d["steps"][0]) >= {"step_id", "name", "verdict", "completion_time_s", "confidence",
                                  "violated_prerequisites", "flags", "evidence_frame", "verifier"}


def test_score_trace_roundtrip(tmp_path):
    t = ScoreTrace("e", np.arange(5) / 2, np.random.rand(5, 2), ["A", "B"], "knn", "abc")
    p = t.save(tmp_path / "t.npz")
    t2 = ScoreTrace.load(p)
    assert t2.step_ids == ["A", "B"] and t2.scorer == "knn" and np.allclose(t2.scores, t.scores)
    tt = t.truncate(1.2)
    assert len(tt.times_s) == 3 and tt.scores.shape == (3, 2)
