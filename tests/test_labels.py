import numpy as np

from helpers import planted_features, step
from sopverify.config import LabelsConfig
from sopverify.state.labels import EXCL, NEG, POS, build_design, frame_labels
from sopverify.state.types import DemoData


def test_margins_and_excluded_band():
    t = np.arange(0, 100, 0.5)
    lab = frame_labels(t, 50.0, 100.0, LabelsConfig(pos_margin_s=3, neg_margin_s=10))
    assert (lab[t >= 53] == POS).all()
    assert (lab[t <= 40] == NEG).all()
    assert (lab[(t > 40) & (t < 53)] == EXCL).all()


def test_never_completed_all_negative():
    t = np.arange(0, 20, 0.5)
    assert (frame_labels(t, None, 20.0, LabelsConfig()) == NEG).all()


def test_adaptive_tail():
    t = np.arange(0, 100, 0.5)  # last grid time 99.5
    lab = frame_labels(t, 98.0, 100.0, LabelsConfig(pos_margin_s=3, adaptive_tail=True))
    assert (lab == POS).sum() >= 1 and lab[-1] == POS  # positives from 98 + 1.0
    assert (frame_labels(t, 98.0, 100.0, LabelsConfig(pos_margin_s=3, adaptive_tail=False)) == POS).sum() == 0
    lab2 = frame_labels(t, 99.8, 100.0, LabelsConfig(pos_margin_s=3, adaptive_tail=True))
    assert (lab2 == POS).sum() == 0  # completion after the last grid point: nothing to label


def test_build_design_groups():
    t = np.arange(0, 60, 0.5)
    steps = [step("A"), step("B")]
    demos = [DemoData(f"d{i}", t, planted_features(t, {"A": 20, "B": 40}, ["A", "B"]), {"A": 20.0, "B": 40.0}, 60.0)
             for i in range(2)]
    d = build_design(demos, steps, LabelsConfig())
    assert set(np.unique(d["A"].groups)) == {0, 1}
    assert d["A"].n_pos > 0 and d["A"].n_neg > 0 and len(d["A"].X) == len(d["A"].y)
