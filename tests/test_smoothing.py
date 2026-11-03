import numpy as np
import pytest

from sopverify.config import SmoothingConfig
from sopverify.temporal.smoothing import estimate_step, evidence_time, hysteresis, median_smooth, runs

CFG = SmoothingConfig(median_window_s=2.5, tau_on=0.6, tau_off=0.4, min_on_s=5.0, min_on_end_s=1.0)
T = np.arange(0, 100, 0.5)


def trace(on_from=None, spans=()):
    x = np.zeros_like(T, dtype=np.float32)
    if on_from is not None:
        x[T >= on_from] = 0.9
    for a, b in spans:
        x[(T >= a) & (T < b)] = 0.9
    return x


def test_hysteresis_band():
    x = np.array([0.1, 0.65, 0.5, 0.45, 0.39, 0.55, 0.61])
    assert hysteresis(x, 0.6, 0.4).tolist() == [False, True, True, True, False, False, True]
    with pytest.raises(ValueError):
        hysteresis(x, 0.4, 0.6)


def test_runs_and_median():
    assert runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 2), (4, 4)]
    x = np.zeros(20, dtype=np.float32)
    x[10] = 1.0  # single-sample blip removed
    assert median_smooth(x, 2.5, 0.5).max() == 0.0


def test_done_final_on():
    e = estimate_step("A", T, trace(on_from=40), CFG)
    assert e.final_on and e.completion_time_s == pytest.approx(40.0) and not e.possible_rework
    assert e.confidence == pytest.approx(0.9, abs=1e-6)


def test_short_blip_dropped_min_duration():
    e = estimate_step("A", T, trace(spans=[(30, 33)]), CFG)  # 3 s < 5 s
    assert not e.final_on and e.on_runs == [] and e.max_score == pytest.approx(0.9)


def test_rework_detection():
    e = estimate_step("A", T, trace(spans=[(20, 40), (60, 100)]), CFG)
    assert e.final_on and e.possible_rework and e.completion_time_s == pytest.approx(60.0)
    e2 = estimate_step("A", T, trace(spans=[(20, 40)]), CFG)
    assert not e2.final_on and e2.possible_rework


def test_end_of_video_run_needs_min_on_end_only():
    e = estimate_step("A", T, trace(on_from=98.0), CFG)  # 2 s at the very end
    assert e.final_on and e.completion_time_s == pytest.approx(98.0)
    e2 = estimate_step("A", T, trace(on_from=98.0), CFG.model_copy(update={"min_on_end_s": 5.0}))
    assert not e2.final_on


def test_no_on_confidence_and_evidence():
    x = np.full_like(T, 0.2, dtype=np.float32)
    x[50] = 0.35
    e = estimate_step("A", T, x, CFG.model_copy(update={"median_window_s": 0.0}))
    assert not e.final_on and e.confidence == pytest.approx(0.65) and e.max_score_time_s == pytest.approx(25.0)
    assert evidence_time(e, 100.0, CFG) == pytest.approx(25.0)
    done = estimate_step("A", T, trace(on_from=40), CFG)
    assert evidence_time(done, 100.0, CFG) == pytest.approx(43.0)
    assert evidence_time(estimate_step("A", T, trace(on_from=98.5), CFG), 100.0, CFG) <= 99.95


def test_empty_trace():
    e = estimate_step("A", np.zeros(0), np.zeros(0), CFG)
    assert not e.final_on and e.max_score == 0.0
