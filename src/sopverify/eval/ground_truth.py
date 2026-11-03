"""Ground truth per execution and step set (spec 9.3). The ONLY module that reads judged-video GT.

* gt_completion_s[s]: final completion via gt_mapping (None if never completed); gt_rework if undone/redone.
* gt_done[s]: completion exists and the end-state holds at t_end (ASR states where mapped).
* gt_order_violation[s]: s is done at t_end and, for some prerequisite p, completion(s) <
  completion(p) or p is not done at t_end (never completed or undone). This mirrors what the
  checker can observe ("done without its prerequisite"; D-014). Prerequisites that are not
  visually verifiable are looked through, as in the checker.
* compliant: every required step done and no order violation on a required step; with
  final_state_check also no component in ASR state -1 at t_end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sopverify.data.records import ExecutionRecord, ExecutionRef
from sopverify.procedure.gt_mapping import completions
from sopverify.procedure.schema import Step


@dataclass
class StepGT:
    step_id: str
    completion_s: float | None
    done: bool
    rework: bool
    order_violation: bool = False
    violated: list[str] = field(default_factory=list)
    required: bool = True
    visually_verifiable: bool = True


@dataclass
class ExecutionGT:
    execution_id: str
    t_end: float
    steps: dict[str, StepGT]
    compliant: bool
    final_misinstalled: bool = False

    @property
    def n_order_violations(self) -> int:
        return sum(1 for s in self.steps.values() if s.order_violation)

    def not_done(self) -> list[str]:
        return [k for k, v in self.steps.items() if not v.done]


def _prereqs(s: str, requires: Mapping[str, Sequence[str]], verifiable: Mapping[str, bool]) -> list[str]:
    out, stack, seen = [], list(requires.get(s, [])), set()
    while stack:
        p = stack.pop(0)
        if p in seen:
            continue
        seen.add(p)
        if not verifiable.get(p, True):
            stack.extend(requires.get(p, []))
        else:
            out.append(p)
    return out


def compute_gt(
    rec: ExecutionRecord,
    steps: Sequence[Step],
    requires: Mapping[str, Sequence[str]],
    final_state_check: bool = False,
    t_end: float | None = None,
) -> ExecutionGT:
    t_end = rec.duration_s if t_end is None else min(t_end, rec.duration_s)
    comps = completions(rec, list(steps), t_end)
    verifiable = {s.id: s.visually_verifiable for s in steps}
    out: dict[str, StepGT] = {}
    for s in steps:
        c = comps[s.id]
        out[s.id] = StepGT(s.id, c.completion_s, c.done_at_end, c.rework, required=s.required,
                           visually_verifiable=s.visually_verifiable)
    for s in steps:
        cs = out[s.id].completion_s
        if cs is None or not out[s.id].done:
            continue
        for p in _prereqs(s.id, requires, verifiable):
            cp = out[p].completion_s
            if cp is None or not out[p].done or cs < cp:
                out[s.id].order_violation = True
                out[s.id].violated.append(p)
    misinstalled = False
    if rec.asr_states is not None:
        st = rec.asr_states.state_at(t_end)
        misinstalled = bool(st is not None and any(v == -1 for v in st))
    req = [s.id for s in steps if s.required]
    compliant = all(out[k].done for k in req) and not any(out[k].order_violation for k in req)
    if final_state_check and misinstalled:
        compliant = False
    return ExecutionGT(rec.execution_id, t_end, out, compliant, misinstalled)


def oracle_gt_fn(records: Mapping[str, ExecutionRecord], steps: Sequence[Step], requires: Mapping[str, Sequence[str]]):
    """GT callback for the oracle scorer; honours truncation through ref.duration_s."""

    def fn(ref: ExecutionRef) -> dict[str, tuple[float | None, bool]]:
        rec = records[ref.execution_id]
        gt = compute_gt(rec, steps, requires, t_end=ref.duration_s)
        return {k: (v.completion_s, v.done) for k, v in gt.steps.items()}

    return fn


def pool_completions(records: Sequence[ExecutionRecord], steps: Sequence[Step]) -> list[dict[str, float | None]]:
    """Completion times for graph inference (inferred_all uses every pool execution: oracle knowledge)."""
    out = []
    for r in records:
        comps = completions(r, list(steps))
        out.append({k: v.completion_s for k, v in comps.items()})
    return out


def diagnostics(gts: Sequence[ExecutionGT]) -> dict:
    """How many executions/steps fall in each GT category (spec 9.3 diagnostics)."""
    n_steps = sum(len(g.steps) for g in gts)
    per_step: dict[str, dict[str, int]] = {}
    for g in gts:
        for k, s in g.steps.items():
            d = per_step.setdefault(k, {"not_done": 0, "order_violation": 0, "rework": 0, "never_completed": 0})
            d["not_done"] += int(not s.done)
            d["order_violation"] += int(s.order_violation)
            d["rework"] += int(s.rework)
            d["never_completed"] += int(s.completion_s is None)
    return {
        "executions": len(gts),
        "compliant": sum(g.compliant for g in gts),
        "non_compliant": sum(not g.compliant for g in gts),
        "executions_with_order_violation": sum(g.n_order_violations > 0 for g in gts),
        "step_instances": n_steps,
        "not_done_instances": sum(len(g.not_done()) for g in gts),
        "order_violation_instances": sum(g.n_order_violations for g in gts),
        "final_misinstalled_executions": sum(g.final_misinstalled for g in gts),
        "per_step": per_step,
    }


def gt_overlay(rec: ExecutionRecord, steps: Sequence[Step], requires: Mapping[str, Sequence[str]]):
    """Ground truth packaged for HTML reports / the app (clearly labelled evaluation-only there)."""
    from sopverify.report.html_report import GTOverlay

    gt = compute_gt(rec, steps, requires)
    spans = [(p.start_s, p.end_s, f"anomaly {p.hand}") for p in rec.phases if p.phase == "ANOMALY"]
    return GTOverlay(completion_s={k: v.completion_s for k, v in gt.steps.items()},
                     done={k: v.done for k, v in gt.steps.items()}, compliant=gt.compliant,
                     order_violation={k: v.order_violation for k, v in gt.steps.items()}, anomaly_spans=spans)
