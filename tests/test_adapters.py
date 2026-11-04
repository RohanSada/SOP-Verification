import json

import numpy as np
import pytest

from sopverify.data.impact import adapters as A


def test_frames_seconds_roundtrip():
    assert A.frames_to_seconds(982, 30.0) == pytest.approx(32.7333, abs=1e-3)
    assert A.frames_to_seconds(50, 25.0) == 2.0
    assert A.frames_to_seconds(130, 30.0, offset_frames=10) == 4.0
    for fps in (25.0, 30.0, 29.97):
        for f in (0, 1, 299, 5213):
            assert A.seconds_to_frame(A.frames_to_seconds(f, fps), fps) == f
    assert A.seconds_to_frame(1000.0, 30.0, n_frames=100) == 99
    with pytest.raises(ValueError):
        A.frames_to_seconds(1, 0)


def test_segments_from_frame_labels():
    labels = ["NULL"] * 3 + ["X"] * 6 + ["Y"] * 3 + ["X"] * 3
    segs = A.frame_labels_to_segments(labels, fps=3.0)
    assert [(s.label, s.start_s, s.end_s) for s in segs] == [("X", 1.0, 3.0), ("Y", 3.0, 4.0), ("X", 4.0, 5.0)]
    assert A.frame_labels_to_segments(["X"] * 50, fps=25.0)[0].end_s == 2.0


def test_parse_execution_id():
    d = A.parse_execution_id("AL07EJ17_Reassembly_A_002_front_asr.json")
    assert d == {"execution_id": "AL07EJ17_Reassembly_A_002", "participant": "AL07EJ17", "task": "Reassembly",
                 "model": "A", "trial": 2, "view": "front"}
    assert A.parse_execution_id("nothing here") is None


def test_psr_event_ids():
    assert A.psr_event_id("Install gearbox_housing") == ("install:gearbox_housing", "install", "gearbox_housing")
    assert A.psr_event_id("Incorrectly installed M6_nut")[0] == "incorrect:M6_nut"
    assert A.psr_event_id("Remove lever")[0] == "remove:lever"
    with pytest.raises(ValueError):
        A.psr_event_id("Paint housing")


def test_read_psr_csv(tmp_path):
    p = tmp_path / "PSR_labels.csv"
    p.write_text("982.jpg,3,Install gearbox_housing\n90.jpg,0,Install anti_vibration_handle\n")
    vocab = [{"impact_id": 3, "id": "install:gearbox_housing"}, {"impact_id": 0, "id": "install:anti_vibration_handle"}]
    ev = A.read_psr_events(p, 30.0, vocab)
    assert [e.event_id for e in ev] == ["install:anti_vibration_handle", "install:gearbox_housing"]
    assert ev[1].time_s == pytest.approx(982 / 30) and ev[1].frame == 982
    with pytest.raises(ValueError):
        A.read_psr_events(p, 30.0, [{"impact_id": 3, "id": "install:lever"}])


def test_read_asr(tmp_path):
    d = {"fps": 25.0, "frame_count": 500, "view_start": 0, "meta_data": {"resolution": {"width": 1280, "height": 720}},
         "components": [{"id": 1, "name": "b"}, {"id": 0, "name": "a"}],
         "state_sequence": [{"frame": 0, "state": [0, 0]}, {"frame": 100, "state": [1, -1]}]}
    p = tmp_path / "X_asr.json"
    p.write_text(json.dumps(d))
    asr, meta = A.read_asr(p)
    assert asr.components == ["a", "b"] and asr.times_s == [0.0, 4.0]
    assert meta["duration_s"] == 20.0 and meta["width"] == 1280
    assert asr.state_at(3.9) == [0, 0] and asr.state_at(4.0) == [1, -1]


def test_read_ppr_and_atr(tmp_path):
    for h, labs in (("L", ["NORMAL"] * 4 + ["ANOMALY"] * 2 + ["RECOVERY"] * 2 + ["NORMAL"] * 2), ("R", ["NORMAL"] * 10)):
        (tmp_path / f"{h}.txt").write_text("\n".join(labs) + "\n")
    phases, frac = A.read_ppr({"L": tmp_path / "L.txt", "R": tmp_path / "R.txt"}, fps=2.0)
    assert [(p.hand, p.phase, p.start_s, p.end_s) for p in phases] == [("L", "ANOMALY", 2.0, 3.0), ("L", "RECOVERY", 3.0, 4.0)]
    assert frac == {"L": 0.2, "R": 0.0}
    arr = np.zeros((10, 6), dtype=np.uint8)
    arr[2:5, 0] = 1
    arr[4:6, 5] = 1
    np.save(tmp_path / "atr.npy", arr)
    spans = A.read_atr({"L": tmp_path / "atr.npy"}, fps=2.0, type_names=["ERROR_TEMPORAL", "B", "C", "D", "E", "ERROR_PROCEDURAL"])
    assert len(spans) == 1 and spans[0].start_s == 1.0 and spans[0].end_s == 3.0
    assert spans[0].types == ["temporal", "procedural"]


def test_expertise_from_excel(tmp_path):
    import pandas as pd

    from sopverify.data.impact.catalog import find_expertise

    df = pd.DataFrame({"Participant": ["AL07EJ17", "ER07AD15", "KI05KO01"], "Group": ["Expert", "novice", ""],
                       "Mental demand": [40, 55, 30]})
    p = tmp_path / "NASA_TLX_anonymized.xlsx"
    df.to_excel(p, index=False)
    assert find_expertise([p]) == {"AL07EJ17": "expert", "ER07AD15": "novice"}
