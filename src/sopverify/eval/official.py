"""Optional: IMPACT's official PSR step metrics on our `psr` step-set predictions (spec 9.5).

The released evaluator (`tasks/PSR/gemini_3_1_pro/evaluate_psr.py` in the IMPACT repo) exposes
`determine_step_metrics(gt, pred, proc_info)` over entries `{"frame", "id", "conf"}` where the id of
component i is 2*i for install and 2*i+1 for remove (`build_proc_info`). Its POS score is computed
inside `main()` and is not reusable, so only step F1 and delay are reported here. Our setting is
few-shot and front-view only, so numbers are not directly comparable to the paper.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from sopverify.compliance.verdicts import VerdictReport
from sopverify.data.records import ExecutionRecord


def component_ids(components: list[str]) -> dict[str, int]:
    return {c: 2 * i for i, c in enumerate(components)}


def gt_entries(rec: ExecutionRecord, components: list[str]) -> list[dict[str, int]]:
    ids = component_ids(components)
    out = []
    for e in rec.psr_events:
        kind, comp = e.event_id.split(":", 1)
        if comp in ids and kind in ("install", "remove"):
            out.append({"frame": int(round(e.time_s * rec.fps)), "id": ids[comp] + (kind == "remove"), "conf": 1})
    return out


def pred_entries(report: VerdictReport, step_to_component: dict[str, str], components: list[str],
                 fps: float) -> list[dict[str, int]]:
    """DONE / OUT_OF_ORDER steps of the psr step set -> install events at their completion frame."""
    ids = component_ids(components)
    out = []
    for s in report.steps:
        if s.verdict.value in ("DONE", "OUT_OF_ORDER") and s.completion_time_s is not None:
            comp = step_to_component.get(s.step_id)
            if comp in ids:
                out.append({"frame": int(round(s.completion_time_s * fps)), "id": ids[comp], "conf": 1})
    return sorted(out, key=lambda d: d["frame"])


def load_official(impact_repo: str | Path):
    path = Path(impact_repo) / "tasks" / "PSR" / "gemini_3_1_pro" / "evaluate_psr.py"
    if not path.exists():
        raise FileNotFoundError(f"official evaluator not found at {path}")
    spec = importlib.util.spec_from_file_location("impact_evaluate_psr", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def official_step_metrics(mod, gt: list[dict], pred: list[dict], components: list[str]) -> dict[str, Any]:
    proc_info = mod.build_proc_info(components, {})
    return mod.determine_step_metrics(gt, pred, proc_info)
