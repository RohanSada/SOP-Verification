"""Download IMPACT bundles, or extract single members from remote zips with HTTP range requests.

The front-view video bundle (13.5-29 GiB) and the VideoMAEv2 feature bundle (21.6 GiB,
all five views) are far larger than what V1 needs (45 front-view executions). Python's
`zipfile` works on any seekable file object, so `HTTPRangeFile` exposes a remote zip
as one: only the central directory and the requested members are transferred.

Integrity: whole-archive SHA256 checksums cannot be verified for partial downloads;
each extracted member is CRC-32 checked by `zipfile` instead (decision D-011).
Small bundles (annotations, sample) are downloaded whole and checked against SHA256SUMS.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path

from sopverify.logging_utils import get_logger

log = get_logger(__name__)

REPO_ID = "KratosWen/IMPACT"
VERSION = "v1.1"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{version}/{path}"

BUNDLES: dict[str, str] = {
    "annotations": "annotations/IMPACT-v1.1-annotations.zip",
    "sample": "sample/IMPACT-v1.1-sample.zip",
    **{f"videos:{v}": f"videos/IMPACT-v1.1-videos-{v}.zip" for v in ("ego", "front", "left", "right", "top")},
    **{f"depth:{v}": f"depth/IMPACT-v1.1-depth-{v}.zip" for v in ("front", "left", "right", "top")},
    **{f"features:{m}": f"features/IMPACT-v1.1-features-{m}.zip" for m in ("I3D", "MViTv2", "VideoMAEv2")},
}


def bundle_url(key: str, revision: str = "main") -> str:
    return HF_RESOLVE.format(repo=REPO_ID, rev=revision, version=VERSION, path=BUNDLES[key])


# ---------------------------------------------------------------------- range file
RangeFetcher = Callable[[int, int], bytes]  # (start, end_inclusive) -> bytes


class RangeFile(io.RawIOBase):
    """Seekable read-only file over a `fetch(start, end)` function, with an LRU block cache."""

    def __init__(self, size: int, fetch: RangeFetcher, block_size: int = 4 << 20, max_blocks: int = 16):
        super().__init__()
        self._size = size
        self._fetch = fetch
        self._block = block_size
        self._max_blocks = max_blocks
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._pos = 0
        self.bytes_fetched = 0
        self.requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(whence)
        self._pos = max(0, self._pos)
        return self._pos

    def _get_block(self, b: int) -> bytes:
        if b in self._cache:
            self._cache.move_to_end(b)
            return self._cache[b]
        start = b * self._block
        end = min(start + self._block, self._size) - 1
        data = self._fetch(start, end)
        if len(data) != end - start + 1:
            raise OSError(f"short range read: wanted {end - start + 1} bytes at {start}, got {len(data)}")
        self.bytes_fetched += len(data)
        self.requests += 1
        self._cache[b] = data
        if len(self._cache) > self._max_blocks:
            self._cache.popitem(last=False)
        return data

    def readinto(self, buf) -> int:  # type: ignore[override]
        if self._pos >= self._size:
            return 0
        n = min(len(buf), self._size - self._pos)
        out = memoryview(buf)
        written = 0
        while written < n:
            b, off = divmod(self._pos, self._block)
            block = self._get_block(b)
            take = min(n - written, len(block) - off)
            out[written : written + take] = block[off : off + take]
            written += take
            self._pos += take
        return written


def http_range_file(url: str, token: str | None = None, block_size: int = 8 << 20) -> RangeFile:
    """RangeFile over an HTTP(S) URL that supports `Range` (Hugging Face resolve URLs do)."""
    import time

    import requests

    session = requests.Session()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    state: dict[str, str] = {}

    def resolve() -> int:
        """Follow HF -> storage redirects with a 1-byte ranged GET (signed storage URLs may refuse HEAD).

        requests drops the Authorization header when a redirect changes host, so the token
        never reaches the storage CDN.
        """
        r = session.get(url, headers={**headers, "Range": "bytes=0-0"}, allow_redirects=True, timeout=60, stream=True)
        if r.status_code in (401, 403):
            raise PermissionError(f"HTTP {r.status_code} for {url}: accept the dataset terms on Hugging Face and set HF_TOKEN")
        if r.status_code != 206:
            raise OSError(f"server ignored Range request (HTTP {r.status_code}) for {url}")
        state["url"] = r.url
        r.close()
        return int(r.headers["Content-Range"].split("/")[-1])

    size = resolve()

    def fetch(start: int, end: int) -> bytes:
        refreshes = 0
        for attempt in range(6):
            try:
                cur = state["url"]
                h = headers if cur.startswith("https://huggingface.co") else {}
                r = session.get(cur, headers={**h, "Range": f"bytes={start}-{end}"}, timeout=180)
                if r.status_code in (400, 401, 403, 404, 410) and refreshes < 3:
                    # Signed storage URLs expire (often after ~1 h): get a fresh one and retry at once.
                    refreshes += 1
                    resolve()
                    cur = state["url"]
                    h = headers if cur.startswith("https://huggingface.co") else {}
                    r = session.get(cur, headers={**h, "Range": f"bytes={start}-{end}"}, timeout=180)
                if r.status_code != 206:
                    raise OSError(f"range request failed (HTTP {r.status_code})")
                return r.content
            except Exception as e:  # pragma: no cover - network
                if attempt == 5:
                    raise
                log.warning("range fetch %d-%d failed (%s); retrying", start, end, e)
                time.sleep(min(2 ** attempt, 30))
        raise AssertionError("unreachable")

    return RangeFile(size, fetch, block_size=block_size)


# ---------------------------------------------------------------------- extraction
def list_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [i for i in zf.infolist() if not i.is_dir()]


def extract_members(
    zf: zipfile.ZipFile,
    select: Callable[[str], bool],
    dest_root: str | Path,
    strip_prefix: str | None = "IMPACT-v1.1/",
    chunk: int = 8 << 20,
) -> list[Path]:
    """Extract members whose name passes `select`. Existing files of the right size are skipped."""
    dest_root = Path(dest_root)
    out: list[Path] = []
    for info in list_members(zf):
        if not select(info.filename):
            continue
        rel = info.filename
        if strip_prefix and rel.startswith(strip_prefix):
            rel = rel[len(strip_prefix):]
        target = dest_root / rel
        if target.exists() and target.stat().st_size == info.file_size:
            out.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        with zf.open(info) as src, open(tmp, "wb") as dst:  # zipfile checks CRC-32 at EOF
            shutil.copyfileobj(src, dst, chunk)
        tmp.replace(target)
        log.info("extracted %s (%.1f MB)", rel, info.file_size / 1e6)
        out.append(target)
    return out


def select_by_video_ids(video_ids: Iterable[str], suffixes: tuple[str, ...] | None = None) -> Callable[[str], bool]:
    """Member filter: the file's basename stem equals one of `video_ids`."""
    wanted = set(video_ids)

    def _sel(name: str) -> bool:
        base = name.rsplit("/", 1)[-1]
        stem, _, ext = base.rpartition(".")
        if suffixes and f".{ext.lower()}" not in suffixes:
            return False
        return stem in wanted

    return _sel


def fetch_members_remote(
    bundle_key: str,
    video_ids: Iterable[str],
    dest_root: str | Path,
    token: str | None = None,
    revision: str = "main",
    url: str | None = None,
) -> list[Path]:
    """Extract only the given videos/features from a remote bundle (HTTP range requests)."""
    url = url or bundle_url(bundle_key, revision)
    token = token or os.environ.get("HF_TOKEN")
    rf = http_range_file(url, token)
    with zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20)) as zf:
        paths = extract_members(zf, select_by_video_ids(video_ids), dest_root)
    log.info("%s: %d files, %.1f MB transferred in %d requests", bundle_key, len(paths), rf.bytes_fetched / 1e6, rf.requests)
    return paths


def remote_listing(bundle_key: str, token: str | None = None, revision: str = "main") -> list[tuple[str, int]]:
    """(name, size) of every member of a remote bundle, transferring only the zip directory."""
    rf = http_range_file(bundle_url(bundle_key, revision), token or os.environ.get("HF_TOKEN"))
    with zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20)) as zf:
        return [(i.filename, i.file_size) for i in list_members(zf)]


# ---------------------------------------------------------------------- whole bundles
def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_checksums(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[-1].lstrip("*").removeprefix("./")] = parts[0]
    return out


def download_bundle(key: str, out_dir: str | Path, verify: bool = True, revision: str = "main") -> Path:
    """Download one whole bundle with huggingface_hub and verify its SHA256 (small bundles only)."""
    from huggingface_hub import hf_hub_download

    out_dir = Path(out_dir)
    local = Path(
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f"{VERSION}/{BUNDLES[key]}",
                        revision=revision, local_dir=out_dir)
    )
    if verify:
        sums_path = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f"{VERSION}/SHA256SUMS",
                                         revision=revision, local_dir=out_dir))
        sums = parse_checksums(sums_path)
        expected = sums.get(BUNDLES[key])
        if expected is None:
            raise RuntimeError(f"No published checksum for {BUNDLES[key]}")
        actual = sha256(local)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {local}: {actual} != {expected}")
        log.info("verified %s", local.name)
    return local


def extract_zip(path: str | Path, dest_root: str | Path, strip_prefix: str | None = "IMPACT-v1.1/") -> list[Path]:
    with zipfile.ZipFile(path) as zf:
        return extract_members(zf, lambda _n: True, dest_root, strip_prefix=strip_prefix)
