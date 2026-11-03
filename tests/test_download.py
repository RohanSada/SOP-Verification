import io
import zipfile

from sopverify.data.impact.download import RangeFile, extract_members, parse_checksums, select_by_video_ids


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for vid in ("AA01BB02_Reassembly_A_001_front", "AA01BB02_Reassembly_A_001_left", "CC03DD04_Reassembly_A_002_front"):
            zf.writestr(f"IMPACT-v1.1/videos/x/{vid}.mp4", (vid * 5000).encode())
        zf.writestr("IMPACT-v1.1/README.txt", b"hello")
    return buf.getvalue()


def test_range_file_extracts_only_selected(tmp_path):
    data = _zip_bytes()
    rf = RangeFile(len(data), lambda a, b: data[a : b + 1], block_size=4096, max_blocks=4)
    with zipfile.ZipFile(io.BufferedReader(rf)) as zf:
        sel = select_by_video_ids(["AA01BB02_Reassembly_A_001_front", "CC03DD04_Reassembly_A_002_front"], (".mp4",))
        paths = extract_members(zf, sel, tmp_path)
    assert sorted(p.name for p in paths) == ["AA01BB02_Reassembly_A_001_front.mp4", "CC03DD04_Reassembly_A_002_front.mp4"]
    assert (tmp_path / "videos/x/AA01BB02_Reassembly_A_001_front.mp4").read_bytes() == b"AA01BB02_Reassembly_A_001_front" * 5000
    assert not (tmp_path / "videos/x/AA01BB02_Reassembly_A_001_left.mp4").exists()
    assert rf.bytes_fetched < 3 * len(data)  # did not need to read everything many times
    # Second run skips existing files.
    with zipfile.ZipFile(io.BufferedReader(RangeFile(len(data), lambda a, b: data[a : b + 1]))) as zf:
        assert len(extract_members(zf, sel, tmp_path)) == 2


def test_parse_checksums(tmp_path):
    p = tmp_path / "SHA256SUMS"
    p.write_text("abc  annotations/IMPACT-v1.1-annotations.zip\ndef *./sample/IMPACT-v1.1-sample.zip\n")
    assert parse_checksums(p) == {"annotations/IMPACT-v1.1-annotations.zip": "abc", "sample/IMPACT-v1.1-sample.zip": "def"}
