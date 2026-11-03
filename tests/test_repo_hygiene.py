"""Guards against shipping an incomplete package (an unanchored .gitignore rule once hid src/sopverify/data/)."""

import importlib
import pkgutil
import shutil
import subprocess
from pathlib import Path

import pytest

import sopverify

REPO = Path(__file__).resolve().parents[1]
OPTIONAL = {"sopverify.app"}  # needs gradio (the [app] extra)


def test_every_module_imports():
    failures = []
    for m in pkgutil.walk_packages(sopverify.__path__, "sopverify."):
        if m.name in OPTIONAL and importlib.util.find_spec("gradio") is None:
            continue
        try:
            importlib.import_module(m.name)
        except Exception as e:  # pragma: no cover - reported below
            failures.append(f"{m.name}: {e!r}")
    assert not failures, failures


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(), reason="not a git checkout")
def test_no_source_files_ignored_by_git():
    out = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--directory",
         "src", "tests", "configs", "procedures", "notebooks", "docs", "scripts"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split()
    bad = [p for p in out if "__pycache__" not in p and ".egg-info" not in p and ".pytest_cache" not in p]
    assert not bad, f"source paths ignored by .gitignore: {bad}"
