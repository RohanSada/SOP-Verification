"""Static test-label firewall (spec 9.1): inference code must not import sopverify.eval."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "sopverify"
GUARDED = ["pipeline.py", "state", "temporal", "compliance", "vlm", "features", "video"]


def _modules(target: str):
    p = SRC / target
    return [p] if p.is_file() else sorted(p.rglob("*.py"))


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


@pytest.mark.parametrize("target", GUARDED)
def test_no_eval_imports(target):
    for mod in _modules(target):
        bad = sorted(i for i in _imports(mod) if i == "sopverify.eval" or i.startswith("sopverify.eval."))
        assert not bad, f"{mod.relative_to(SRC)} imports {bad}"


def test_pipeline_refuses_oracle_and_records(tmp_path):
    from helpers import make_record, write_toy_procedure
    from sopverify.config import Config
    from sopverify.pipeline import FirewallError, run
    from sopverify.procedure.loader import load_procedure

    proc = load_procedure(write_toy_procedure(tmp_path))
    rec = make_record()
    with pytest.raises(FirewallError):
        run(rec, proc, [], Config(), write=False)  # annotated record as the judged video
    with pytest.raises(FirewallError):
        run(rec.to_ref(), proc, [rec], Config(), write=False)  # judged video is a demo
    with pytest.raises(FirewallError):
        run(make_record("XX01YY02_Reassembly_A_009").to_ref(), proc, [rec],
            Config().with_overrides({"scorer.name": "oracle"}), write=False)
