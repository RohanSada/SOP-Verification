import numpy as np
import pytest

from sopverify.config import Config
from sopverify.data.records import ExecutionRef
from sopverify.state.scorers import make_scorer
from sopverify.state.types import DemoData

from helpers import planted_features, step

T = np.arange(0, 120, 0.5)
STEPS = [step("A"), step("B")]


def demos(k=2):
    out = []
    for i in range(k):
        c = {"A": 30.0 + 5 * i, "B": 70.0 + 5 * i}
        out.append(DemoData(f"d{i}", T, planted_features(T, c, ["A", "B"], seed=i), c, 120.0))
    return out


@pytest.mark.parametrize("name", ["prototype", "linear_probe", "knn"])
@pytest.mark.parametrize("k", [1, 2])
def test_feature_scorers_separate(name, k):
    s = make_scorer(Config().with_overrides({"scorer.name": name}))
    s.fit(demos(k), STEPS)
    test_c = {"A": 50.0, "B": 90.0}
    X = planted_features(T, test_c, ["A", "B"], seed=9)
    p = s.score(T, X, ExecutionRef(execution_id="t", duration_s=120.0))
    assert p.shape == (len(T), 2) and p.min() >= 0 and p.max() <= 1
    assert p[T < 40, 0].mean() < 0.2 and p[T > 60, 0].mean() > 0.8
    assert p[T < 80, 1].mean() < 0.2 and p[T > 100, 1].mean() > 0.8


def test_never_completed_step_is_uncertain_constant():
    d = demos(2)
    for x in d:
        x.completion_s["B"] = None
    for name in ("prototype", "linear_probe", "knn"):
        s = make_scorer(Config().with_overrides({"scorer.name": name}))
        s.fit(d, STEPS)
        p = s.score(T, d[0].feats, ExecutionRef(execution_id="t", duration_s=120.0))
        assert np.allclose(p[:, 1], 0.5), name


def test_time_prior():
    s = make_scorer(Config().with_overrides({"scorer.name": "time_prior"}))
    s.fit(demos(2), STEPS)
    assert s.threshold["A"] == pytest.approx(32.5 / 120)
    p = s.score(T, None, ExecutionRef(execution_id="t", duration_s=120.0))
    assert p[T < 32, 0].max() == 0 and p[T >= 33, 0].min() == 1
    p2 = s.score(T[:120], None, ExecutionRef(execution_id="t", duration_s=60.0))  # truncated duration
    assert p2[T[:120] >= 17, 0].min() == 1


def test_oracle_requires_gt_fn():
    with pytest.raises(ValueError):
        make_scorer(Config().with_overrides({"scorer.name": "oracle"}))
    s = make_scorer(Config().with_overrides({"scorer.name": "oracle"}), gt_fn=lambda ref: {"A": (10.0, True), "B": (20.0, False)})
    s.fit([], STEPS)
    p = s.score(T, None, ExecutionRef(execution_id="t", duration_s=120.0))
    assert p[T >= 10, 0].min() == 1 and p[T < 10, 0].max() == 0 and p[:, 1].max() == 0
