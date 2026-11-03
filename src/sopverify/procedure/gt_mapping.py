"""Step completion times from annotations, via each step's `gt_mapping`.

Used by two callers only:
* the demo label builder (demo executions), and
* `sopverify.eval.ground_truth` (any execution, evaluation only).

The inference pipeline never calls this for the video it judges.

Rules (decision D-005):
* psr_events: each event's completion is its LAST occurrence (at or before t_end);
  the step completes when ALL member events have completed (max over members).
  A member that occurred, was undone (opposite event) and occurred again marks rework.
* tas_s_labels: completion = time of the end frame of the LAST segment with any listed label.
* done_at_end: completion exists and the end-state holds at t_end. For psr mappings the
  member components' ASR state at t_end must match the event (install -> 1, remove -> 0);
  without ASR, no undo event may follow the last completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sopverify.data.records import ExecutionRecord
from sopverify.procedure.schema import Step

_EXPECTED_STATE = {"install": 1, "remove": 0, "incorrect": -1}
_UNDO = {"install": "remove", "remove": "install"}


@dataclass
class StepCompletion:
    step_id: str
    completion_s: float | None
    done_at_end: bool
    rework: bool = False
    member_times: dict[str, float | None] = field(default_factory=dict)


def _split(event_id: str) -> tuple[str, str]:
    kind, _, comp = event_id.partition(":")
    return kind, comp


def step_completion(rec: ExecutionRecord, step: Step, t_end: float | None = None) -> StepCompletion:
    t_end = rec.duration_s if t_end is None else t_end
    m = step.gt_mapping
    if m.psr_events:
        return _psr_completion(rec, step, list(m.psr_events), t_end)
    if m.tas_s_labels:
        labels = set(m.tas_s_labels)
        ends = [s.end_s for s in rec.tas_s_segments if s.label in labels and s.end_s <= t_end + 1e-9]
        # A segment still running at t_end is not complete. Segments store an exclusive end
        # ((last frame + 1) / fps); completion is the time of the segment's END FRAME (spec 7.2).
        c = (max(ends) - 1.0 / rec.fps) if ends else None
        n_segments = sum(1 for s in rec.tas_s_segments if s.label in labels and s.end_s <= t_end + 1e-9)
        return StepCompletion(step.id, c, c is not None, rework=False, member_times={"segments": float(n_segments)})
    raise ValueError(f"step {step.id} has no gt_mapping")


def _psr_completion(rec: ExecutionRecord, step: Step, events: list[str], t_end: float) -> StepCompletion:
    evs = [e for e in rec.psr_events if e.time_s <= t_end + 1e-9]
    member_times: dict[str, float | None] = {}
    rework = False
    still_holds = True
    for eid in events:
        kind, comp = _split(eid)
        times = [e.time_s for e in evs if e.event_id == eid]
        last = max(times) if times else None
        member_times[eid] = last
        undo = _UNDO.get(kind)
        undo_times = [e.time_s for e in evs if undo and e.event_id == f"{undo}:{comp}"]
        if times and undo_times and min(undo_times) > min(times):
            rework = True  # completed, undone, (maybe) completed again
        if last is not None and any(t > last for t in undo_times):
            still_holds = False
    if any(v is None for v in member_times.values()):
        completion = None
    else:
        completion = max(v for v in member_times.values() if v is not None)
    done = completion is not None and still_holds
    if done and rec.asr_states is not None:
        state = rec.asr_states.state_at(t_end)
        if state is not None:
            comp_state = dict(zip(rec.asr_states.components, state))
            for eid in events:
                kind, comp = _split(eid)
                exp = _EXPECTED_STATE.get(kind)
                if exp is not None and comp in comp_state and comp_state[comp] != exp:
                    done = False
    return StepCompletion(step.id, completion, done, rework, member_times)


def completions(rec: ExecutionRecord, steps: list[Step], t_end: float | None = None) -> dict[str, StepCompletion]:
    return {s.id: step_completion(rec, s, t_end) for s in steps}


def completion_times(rec: ExecutionRecord, steps: list[Step], t_end: float | None = None) -> dict[str, float | None]:
    """step id -> completion time (None if never completed)."""
    return {k: v.completion_s for k, v in completions(rec, steps, t_end).items()}
