from sopverify.compliance.verdicts import StepResult, StepVerdict, Summary, VerdictReport, VideoVerdict
from sopverify.eval.official import gt_entries, pred_entries

from helpers import make_record


def test_official_format_conversion():
    comps = ["a", "b", "c", "d"]
    rec = make_record(installs={"a": 1.0, "b": 2.0}, removes={"a": [3.0]})
    gt = gt_entries(rec, comps)
    assert gt == [{"frame": 30, "id": 0, "conf": 1}, {"frame": 60, "id": 2, "conf": 1}, {"frame": 90, "id": 1, "conf": 1}]
    steps = [StepResult(step_id="P1", name="a", verdict=StepVerdict.DONE, completion_time_s=1.5),
             StepResult(step_id="P2", name="b", verdict=StepVerdict.NOT_DONE)]
    rep = VerdictReport(execution_id="e", procedure_id="p", step_set="psr", video_verdict=VideoVerdict.NON_COMPLIANT,
                        summary=Summary.of(steps), steps=steps, scorer="s")
    assert pred_entries(rep, {"P1": "a", "P2": "b"}, comps, 30.0) == [{"frame": 45, "id": 0, "conf": 1}]
