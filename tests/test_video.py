"""Decoding tests on a tiny synthetic video whose frame index is encoded in its colour."""

import numpy as np
import pytest

from sopverify.video import roi as R
from sopverify.video.decode import decode_grid, probe, read_frames_at
from sopverify.video.timegrid import frame_indices, make_grid, resample_nearest, step_hold

FPS, N, W, H = 10, 40, 64, 48


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    av = pytest.importorskip("av")
    path = tmp_path_factory.mktemp("vid") / "v.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"qscale": "1"}
        for i in range(N):
            img = np.full((H, W, 3), i * 6, dtype=np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def _index(frame) -> int:
    return int(round(float(frame[..., 0].mean()) / 6))


def test_grid_helpers():
    assert make_grid(2.0, 2.0).tolist() == [0.0, 0.5, 1.0, 1.5]
    assert make_grid(2.1, 2.0).tolist()[-1] == 2.0
    assert frame_indices(np.array([0, 0.49, 0.5, 100]), 2.0, 10).tolist() == [0, 0, 1, 9]
    src = np.array([0.0, 1.0, 2.0])
    assert resample_nearest(src, np.array([10, 20, 30]), np.array([0.4, 0.6, 5.0])).tolist() == [10, 20, 30]
    assert step_hold(src, np.array([1, 2, 3]), np.array([-1.0, 0.5, 1.0, 9.0])).tolist() == [1, 1, 2, 3]


def test_probe_and_decode_grid(video):
    info = probe(video)
    assert (info.width, info.height) == (W, H) and info.fps == FPS
    assert abs(info.duration_s - N / FPS) < 0.2
    times = make_grid(N / FPS, 2.0)  # every 5th frame
    frames = decode_grid(video, times, fps=FPS)
    assert frames.shape == (len(times), H, W, 3)
    assert [_index(f) for f in frames] == [int(t * FPS) for t in times]
    crops = decode_grid(video, times, lambda f: R.resize_square(R.crop(f, [8, 8, 32, 16]), 28), fps=FPS)
    assert crops.shape == (len(times), 28, 28, 3)


def test_read_frames_at(video):
    frames = read_frames_at(video, [2.0, 0.0, 3.5], fps=FPS)
    assert [_index(f) for f in frames] == [20, 0, 35]


def test_roi_helpers():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert R.crop(img, [440, 300, 480, 360]).shape == (360, 480, 3)
    assert R.clip_box([-10, -10, 50, 50], 100, 100) == (0, 0, 40, 40)
    with pytest.raises(ValueError):
        R.clip_box([200, 200, 10, 10], 100, 100)
    assert R.scale_box([440, 300, 480, 360], (1280, 720), (640, 360)) == [220, 150, 240, 180]
    lb = R.resize_square(np.full((100, 200, 3), 255, np.uint8), 50)
    assert lb.shape == (50, 50, 3) and lb[0, 0, 0] == 0 and lb[25, 25, 0] == 255
    assert R.resize_square(np.ones((100, 200, 3), np.uint8), 50, "squash").shape == (50, 50, 3)
    assert R.resize_long_side(img, 768).shape[:2] == (432, 768)
    assert R.overlay_box(img, [10, 10, 100, 100]).shape == img.shape
