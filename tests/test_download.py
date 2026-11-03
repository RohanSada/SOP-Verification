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


def test_http_range_file_with_redirect_and_expiring_urls(tmp_path):
    """Local server that behaves like HF: resolve URL -> 302 to a signed storage URL that
    refuses HEAD and expires every few requests."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from sopverify.data.impact.download import http_range_file

    payload = bytes(range(256)) * 4 + __import__("os").urandom(40_000)  # incompressible
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("IMPACT-v1.1/videos/front/AA01BB02_Reassembly_A_001_front.mp4", payload[::-1])
        z.writestr("IMPACT-v1.1/videos/front/CC03DD04_Reassembly_A_002_front.mp4", payload)
    data = buf.getvalue()
    state = {"gen": 0, "hits": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_HEAD(self):
            self.send_response(403)
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/resolve/"):
                self.send_response(302)
                self.send_header("Location", f"/storage/f.zip?sig={state['gen']}")
                self.end_headers()
                return
            sig = int(self.path.split("sig=")[1])
            if sig != state["gen"]:
                self.send_response(403)
                self.end_headers()
                return
            a, b = self.headers["Range"].split("=")[1].split("-")
            a, b = int(a), min(int(b), len(data) - 1)
            body = data[a : b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            state["hits"] += 1
            if state["hits"] % 5 == 0:
                state["gen"] += 1  # expire the current signature after serving

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/resolve/f.zip"
        rf = http_range_file(url, block_size=512)
        with zipfile.ZipFile(io.BufferedReader(rf)) as zf:
            paths = extract_members(zf, select_by_video_ids(["CC03DD04_Reassembly_A_002_front"]), tmp_path)
        assert len(paths) == 1 and paths[0].read_bytes() == payload
        assert state["gen"] > 0  # signatures expired and were refreshed along the way
    finally:
        srv.shutdown()
