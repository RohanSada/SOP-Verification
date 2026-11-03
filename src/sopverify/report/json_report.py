"""JSON verdict reports (schema 7.5 lives in sopverify.compliance.verdicts)."""

from __future__ import annotations

from pathlib import Path

from sopverify.compliance.verdicts import VerdictReport


def load_report(path: str | Path) -> VerdictReport:
    p = Path(path)
    if p.is_dir():
        p = p / "verdict.json"
    return VerdictReport.load(p)


def summary_line(r: VerdictReport) -> str:
    s = r.summary
    return (f"{r.execution_id}: {r.video_verdict.value} | done {s.done}, not done {s.not_done}, "
            f"out of order {s.out_of_order}, uncertain {s.uncertain}, not verifiable {s.not_verifiable}")


def steps_table(r: VerdictReport) -> list[dict]:
    rows = []
    for s in r.steps:
        rows.append({
            "step": s.step_id, "name": s.name, "verdict": s.verdict.value,
            "completion_s": None if s.completion_time_s is None else round(s.completion_time_s, 1),
            "confidence": round(s.confidence, 2),
            "violated": ", ".join(s.violated_prerequisites), "flags": ", ".join(s.flags),
            "verifier": s.verifier.answer if s.verifier.used else "",
        })
    return rows
