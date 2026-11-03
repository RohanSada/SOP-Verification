"""Synthetic end-to-end test: planted per-step signal -> full pipeline -> expected verdicts."""

import json

import pytest

from sopverify.compliance.verdicts import StepVerdict as V
from sopverify.compliance.verdicts import VideoVerdict as VV
from sopverify.config import Config
from sopverify.features.provider import ArrayFeatureProvider
from sopverify.pipeline import run
from sopverify.procedure.loader import load_procedure
from sopverify.video.timegrid import make_grid

from helpers import make_record, planted_features, write_toy_procedure

STEP_IDS = ["A", "B", "C", "D"]


def _setup(tmp_path, scorer: str):
    proc = load_procedure(write_toy_procedure(tmp_path / "proc"))
    cfg = Config().with_overrides({
        "scorer.name": scorer, "video.sample_fps": 2.0, "report.html": False, "report.evidence": False,
        "paths.runs_root": str(tmp_path / "runs"),
    })
    demo_specs = {
        "DM01AA01_Reassembly_A_001": {"a": 20.0, "b": 60.0, "c": 110.0, "d": 150.0},
        "DM02BB02_Reassembly_A_001": {"a": 25.0, "b": 70.0, "c": 120.0, "d": 140.0},
    }
    feats = {}
    demos = []
    for i, (eid, inst) in enumerate(demo_specs.items()):
        rec = make_record(eid, installs=inst, duration=200.0)
        t = make_grid(200.0, 2.0)
        feats[eid] = (t, planted_features(t, {k.upper(): v for k, v in inst.items()}, STEP_IDS, seed=i))
        demos.append(rec)
    # Test video: B before A (order violation), C after both, D never done.
    test_c = {"A": 80.0, "B": 30.0, "C": 120.0, "D": None}
    t = make_grid(180.0, 2.0)
    feats["TS09TT09_Reassembly_A_003"] = (t, planted_features(t, test_c, STEP_IDS, seed=7))
    ref = make_record("TS09TT09_Reassembly_A_003", installs={}, duration=180.0).to_ref()
    return proc, cfg, demos, ArrayFeatureProvider(feats), ref


def test_planted_order_violation_end_to_end(tmp_path):
    # Only the linear probe can isolate per-step evidence here: whole-frame similarity scorers
    # (prototype, knn) learn "after B" as "A and B done" because A always precedes B in the
    # demos, so they cannot see B completing before A. See test_all_scorers_missing_step.
    proc, cfg, demos, feats, ref = _setup(tmp_path, "linear_probe")
    rep = run(ref, proc, demos, cfg, features=feats, out_dir=tmp_path / "out")
    v = {s.step_id: s.verdict for s in rep.steps}
    assert v == {"A": V.DONE, "B": V.OUT_OF_ORDER, "C": V.DONE, "D": V.NOT_DONE}, v
    assert rep.step("B").violated_prerequisites == ["A"]
    assert rep.video_verdict == VV.NON_COMPLIANT
    assert abs(rep.step("A").completion_time_s - 80.0) <= 3.0
    assert abs(rep.step("C").completion_time_s - 120.0) <= 3.0
    out = tmp_path / "out"
    assert (out / "verdict.json").exists() and (out / "trace.npz").exists()
    d = json.loads((out / "verdict.json").read_text())
    assert d["video_verdict"] == "NON_COMPLIANT" and d["summary"]["out_of_order"] == 1


@pytest.mark.parametrize("scorer", ["linear_probe", "prototype", "knn"])
def test_all_scorers_missing_step(tmp_path, scorer):
    proc, cfg, demos, feats, _ = _setup(tmp_path, scorer)
    t = make_grid(190.0, 2.0)
    feats.data["MS01MS01_Reassembly_A_002"] = (t, planted_features(t, {"A": 22, "B": 65, "C": 115, "D": None}, STEP_IDS, seed=5))
    ref = make_record("MS01MS01_Reassembly_A_002", installs={}, duration=190.0).to_ref()
    rep = run(ref, proc, demos, cfg, features=feats, write=False)
    v = {s.step_id: s.verdict for s in rep.steps}
    assert v == {"A": V.DONE, "B": V.DONE, "C": V.DONE, "D": V.NOT_DONE}, v
    assert rep.video_verdict == VV.NON_COMPLIANT


def test_compliant_video(tmp_path):
    proc, cfg, demos, feats, _ = _setup(tmp_path, "linear_probe")
    t = make_grid(200.0, 2.0)
    feats.data["OK01OK01_Reassembly_A_001"] = (t, planted_features(t, {"A": 15, "B": 50, "C": 100, "D": 130}, STEP_IDS, seed=3))
    ref = make_record("OK01OK01_Reassembly_A_001", installs={}, duration=200.0).to_ref()
    rep = run(ref, proc, demos, cfg, features=feats, write=False)
    assert rep.video_verdict == VV.COMPLIANT
    assert all(s.verdict == V.DONE for s in rep.steps)


def test_time_prior_needs_no_features(tmp_path):
    proc, cfg, demos, _, ref = _setup(tmp_path, "time_prior")
    rep = run(ref, proc, demos, cfg, write=False)
    assert rep.scorer == "time_prior" and len(rep.steps) == 4
