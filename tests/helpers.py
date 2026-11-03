"""Synthetic fixtures shared by the fast tests (no IMPACT data needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from sopverify.data.records import ASRStates, ExecutionRecord, PSREvent, Segment
from sopverify.procedure.schema import GTMapping, Step

COMPONENTS = ["a", "b", "c", "d"]


def step(sid: str, requires=(), events=None, required=True, verifiable=True) -> Step:
    return Step(id=sid, name=f"step {sid}", description=f"do {sid}", requires=list(requires), required=required,
                visually_verifiable=verifiable,
                gt_mapping=GTMapping(psr_events=events or [f"install:{sid.lower()}"]))


def chain_steps() -> list[Step]:
    """A -> B -> C, A -> D (a small tree)."""
    return [step("A"), step("B", ["A"]), step("C", ["B"]), step("D", ["A"])]


def make_record(
    eid: str = "AB01CD02_Reassembly_A_001",
    installs: dict[str, float | list[float]] | None = None,
    removes: dict[str, list[float]] | None = None,
    duration: float = 200.0,
    fps: float = 30.0,
    final_states: dict[str, int] | None = None,
    anomaly_frac: dict[str, float] | None = None,
    tas: list[tuple[str, float, float]] | None = None,
) -> ExecutionRecord:
    installs = installs if installs is not None else {"a": 20.0, "b": 60.0, "c": 110.0, "d": 150.0}
    removes = removes or {}
    events: list[PSREvent] = []
    for comp, ts in installs.items():
        for t in ([ts] if isinstance(ts, (int, float)) else ts):
            events.append(PSREvent(event_id=f"install:{comp}", time_s=float(t), frame=int(round(t * fps))))
    for comp, ts in removes.items():
        for t in ts:
            events.append(PSREvent(event_id=f"remove:{comp}", time_s=float(t), frame=int(round(t * fps))))
    events.sort(key=lambda e: e.time_s)
    # ASR keyframes at every event; state follows the events.
    state = {c: 0 for c in COMPONENTS}
    times, states = [0.0], [[0] * len(COMPONENTS)]
    for e in events:
        kind, comp = e.event_id.split(":")
        if comp in state:
            state[comp] = 1 if kind == "install" else 0
        times.append(e.time_s)
        states.append([state[c] for c in COMPONENTS])
    if final_states:
        for c, v in final_states.items():
            state[c] = v
        times.append(duration - 0.5)
        states.append([state[c] for c in COMPONENTS])
    part, task, model, trial = eid.split("_")
    return ExecutionRecord(
        execution_id=eid, participant=part, task=task, model=model, trial=int(trial), fps=fps,
        n_frames=int(round(duration * fps)), duration_s=duration, psr_events=events,
        tas_s_segments=[Segment(label=lab, start_s=s, end_s=e) for lab, s, e in (tas or [])],
        asr_states=ASRStates(times_s=times, components=COMPONENTS, states=states),
        anomaly_frac=anomaly_frac if anomaly_frac is not None else {"L": 0.01, "R": 0.01},
    )


def write_toy_procedure(root: Path, steps: list[Step] | None = None, graph_source: str = "manual") -> Path:
    steps = steps or chain_steps()
    (root / "step_sets").mkdir(parents=True, exist_ok=True)
    (root / "procedure.yaml").write_text(yaml.safe_dump({
        "procedure_id": "toy", "name": "Toy procedure", "default_step_set": "grouped",
        "step_sets": {"grouped": "step_sets/grouped.yaml"}, "graph_source": graph_source,
        "roi": "roi.yaml", "vocabulary": "vocab.yaml"}))
    (root / "step_sets" / "grouped.yaml").write_text(yaml.safe_dump(
        {"step_set": "grouped", "steps": [s.model_dump(exclude_none=True) for s in steps]}))
    (root / "roi.yaml").write_text(yaml.safe_dump({"box": [10, 10, 100, 80], "frame_size": [160, 120]}))
    (root / "vocab.yaml").write_text(yaml.safe_dump({
        "components": COMPONENTS,
        "psr_events": [{"id": f"{k}:{c}", "kind": k, "component": c} for c in COMPONENTS for k in ("install", "remove")],
        "tas_s_labels": []}))
    (root / "sop.md").write_text("# Toy SOP\n")
    return root


def planted_features(times: np.ndarray, completions: dict[str, float | None], step_ids: list[str],
                     dim: int = 32, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """Features where dimension j switches on when step j completes (plus noise and a bias dim)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, noise, size=(len(times), dim)).astype(np.float32)
    X[:, -1] += 1.0
    for j, sid in enumerate(step_ids):
        c = completions.get(sid)
        if c is not None:
            X[times >= c, j] += 1.0
    return X
