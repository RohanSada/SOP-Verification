import numpy as np

from sopverify.features.alignment import alignment_report


def test_alignment_detects_shift():
    rng = np.random.default_rng(0)
    events = (rng.random(400) < 0.05).astype(float)
    base = np.cumsum(events[:, None] * rng.normal(size=(400, 16)), axis=0)
    a = base + 0.01 * rng.normal(size=base.shape)
    b = np.roll(base, 3, axis=0) @ rng.normal(size=(16, 24)) + 0.01 * rng.normal(size=(400, 24))
    assert alignment_report(a, base, 2.0)["best_lag_samples"] == 0
    r = alignment_report(a, b, 2.0)
    assert r["best_lag_samples"] == -3 and r["best_lag_s"] == -1.5
