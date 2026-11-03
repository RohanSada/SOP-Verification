"""App smoke tests: the UI builds, and an IMPACT execution runs end to end without video (time prior)."""

import time

import pytest

from sopverify.config import load_config

gr = pytest.importorskip("gradio")


def test_build_app(repo):
    from sopverify.app import build_app

    demo = build_app(repo / "configs" / "default.yaml")
    assert demo is not None


@pytest.mark.data
def test_run_video_time_prior(repo, impact_annotations, tmp_path, monkeypatch):
    from sopverify.app import AppState, run_video

    monkeypatch.setenv("SOPVERIFY_ANNOTATIONS", str(impact_annotations))
    cfg = load_config(repo / "configs" / "default.yaml", {"scorer.name": "time_prior",
                                                          "paths.runs_root": str(tmp_path / "runs")})
    state = AppState(cfg)
    choices = state.execution_choices()
    assert choices and any("[demo]" in c[0] for c in choices)
    eid = next(v for label, v in choices if "[test]" in label)
    t0 = time.time()
    out = run_video(state, "impact_reassembly_A", "grouped", "IMPACT execution", eid, None, "Best feature scorer")
    assert time.time() - t0 < 15
    rep = out["report"]
    assert rep.execution_id == eid and len(rep.steps) == 5 and out["gt"] is not None
    assert (out["dir"] / "verdict.json").exists() and (out["dir"] / "report.html").exists()
