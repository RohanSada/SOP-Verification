import json

import numpy as np
import pytest
from PIL import Image

from sopverify.compliance.checker import check
from sopverify.compliance.verdicts import StepVerdict as V
from sopverify.config import Config, VLMConfig
from sopverify.data.records import ExecutionRef
from sopverify.state.types import DemoData
from sopverify.temporal.smoothing import StepEstimate
from sopverify.vlm.backend import FakeBackend, PromptLogger, VLMRequest
from sopverify.vlm.parsing import extract_json
from sopverify.vlm.scorer_impl import VLMStateScorer, answer_to_prob, run_with_retries
from sopverify.vlm.verifier import VLMVerifier

from helpers import chain_steps


class FakeFrames:
    """Image whose pixel value encodes the requested time."""

    def __init__(self):
        self.calls = []

    def crops(self, video_path, times, fps=None, size=None):
        self.calls.append((video_path, list(times)))
        return [Image.new("RGB", (32, 24), (int(t) % 256, 0, 0)) for t in times]


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('Sure! ```json\n{"a": true}\n``` done', {"a": True}),
    ("prefix {'a': True, 'b': None,} suffix", {"a": True, "b": None}),
    ('{"x": {"y": "}"}} trailing', {"x": {"y": "}"}}),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_extract_json_fails():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_answer_to_prob():
    assert answer_to_prob({"complete": True, "confidence": 0.8}) == 0.8
    assert answer_to_prob({"complete": False, "confidence": 0.8}) == pytest.approx(0.2)
    assert answer_to_prob({"complete": None, "confidence": 0.9}) == 0.5
    assert answer_to_prob({"complete": "yes", "confidence": 2}) == 1.0
    assert answer_to_prob("garbage") == 0.5


def test_retries_and_logging(tmp_path):
    n = {"calls": 0}

    def respond(req):
        n["calls"] += 1
        return "not json" if n["calls"] == 1 else '{"ok": 1}'

    log = PromptLogger(tmp_path / "log.jsonl")
    res, fails = run_with_retries(FakeBackend(respond), [VLMRequest(["q"])], lambda o: o, 2, log, "t")
    assert res == [{"ok": 1}] and fails == 1
    lines = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [x["parse_ok"] for x in lines] == [False, True]
    res2, fails2 = run_with_retries(FakeBackend(lambda r: "nope"), [VLMRequest(["q"])], lambda o: o, 2, log, "t")
    assert res2 == [None] and fails2 == 3


def _scorer(tmp_path, refs=True):
    steps = chain_steps()
    completion = {"A": 20.0, "B": 60.0, "C": 110.0, "D": 150.0}

    def respond(req):
        t = req.meta["t"]
        ids = req.json_schema["required"]
        return json.dumps({s: {"complete": t >= completion[s], "confidence": 0.9} for s in ids})

    backend = FakeBackend(respond)
    frames = FakeFrames()
    cfg = VLMConfig(keyframe_s=10.0, in_context_refs=refs, max_steps_per_prompt=3)
    sc = VLMStateScorer(cfg, backend, frames, log_path=tmp_path / "vlm.jsonl")
    demo = DemoData("d", np.arange(0, 200, 0.5), None, completion, 200.0, video_path="demo.mp4", fps=30.0)
    sc.fit([demo], steps)
    return sc, backend, frames


def test_vlm_scorer_trace(tmp_path):
    sc, backend, frames = _scorer(tmp_path)
    assert sorted(sc.refs) == ["A", "B", "C", "D"]
    times = np.arange(0, 180, 0.5)
    ref = ExecutionRef(execution_id="t", video_path="test.mp4", duration_s=180.0, fps=30.0)
    P = sc.score(times, None, ref)
    assert P.shape == (len(times), 4)
    assert P[times < 20, 0].max() == pytest.approx(0.1) and P[times >= 20, 0].min() == pytest.approx(0.9)
    assert P[times >= 150, 3].min() == pytest.approx(0.9)
    kf = sc.keyframes(180.0)
    assert kf[-1] == pytest.approx(179.8) and kf[0] == 0.0
    # 4 steps with max 3 per prompt -> 2 chunks per keyframe
    assert sc.stats["requests"] == 2 * len(kf) and sc.stats["parse_failures"] == 0
    assert any(r.n_images > 1 for r in backend.calls)  # reference crops included
    sc2, backend2, _ = _scorer(tmp_path, refs=False)
    sc2.score(times, None, ref)
    assert all(r.n_images == 1 for r in backend2.calls)


def test_vlm_scorer_via_pipeline_factory(tmp_path):
    from sopverify.state.scorers import make_scorer

    cfg = Config().with_overrides({"scorer.name": "vlm"})
    sc = make_scorer(cfg, backend=FakeBackend(lambda r: "{}"), frames=FakeFrames())
    assert sc.name == "vlm" and not sc.needs_features


def test_verifier_maps_answers(tmp_path):
    steps = chain_steps()
    est = {
        "A": StepEstimate("A", True, 10.0, 0.9, 0.95, 10.0, [(10.0, 100.0)]),
        "B": StepEstimate("B", False, None, 0.5, 0.5, 42.0),
        "C": StepEstimate("C", False, None, 0.5, 0.55, 80.0),
        "D": StepEstimate("D", False, None, 0.85, 0.15, 30.0),
    }
    req = {"A": [], "B": ["A"], "C": ["B"], "D": ["A"]}
    res = check(steps, est, req, Config().checker)
    assert res.steps["B"].verdict == V.UNCERTAIN and res.steps["D"].verdict == V.NOT_DONE
    answers = {"B": "yes", "C": "cannot_tell", "D": "no"}
    backend = FakeBackend(lambda r: json.dumps({"answer": answers[r.meta["step_id"]], "rationale": "seen"}))
    cfg = Config().with_overrides({"vlm.verifier": True, "vlm.enabled": True, "vlm.verify_not_done_below": 0.9})
    ver = VLMVerifier(cfg, backend, FakeFrames(), tmp_path / "v.jsonl")
    demo = DemoData("d", np.arange(0, 100, 0.5), None, {"A": 5.0, "B": 30.0, "C": 60.0, "D": 70.0}, 100.0,
                    video_path="demo.mp4", fps=30.0)
    out = ver.verify(ExecutionRef(execution_id="t", video_path="t.mp4", duration_s=100.0, fps=30.0),
                     steps, est, res, [demo])
    assert set(out) == {"B", "C", "D"}
    assert out["B"][0] == V.DONE and out["B"][1] == 42.0 and out["B"][2].answer == "yes"
    assert out["C"][0] == V.UNCERTAIN and out["D"][0] == V.NOT_DONE
    assert all(r.n_images == 2 + 2 + 3 for r in backend.calls)
    res2 = check(steps, est, req, Config().checker, {k: (v[0], v[1]) for k, v in out.items()})
    assert res2.steps["B"].verdict == V.DONE
