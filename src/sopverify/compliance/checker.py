"""Deterministic compliance rules (spec 8.8).

Per step s:
  1. not visually_verifiable          -> NOT_VERIFIABLE
  2. final_on                         -> provisionally DONE
  3. max_score < tau_low              -> NOT_DONE
  4. otherwise                        -> UNCERTAIN
Order check for each provisionally DONE s and each prerequisite p (all judged on the
provisional statuses, so the result does not depend on iteration order):
  p DONE and completion(s) < completion(p) - tol -> s OUT_OF_ORDER (violated += p)
  p NOT_DONE                                     -> s OUT_OF_ORDER (done without prerequisite)
  p UNCERTAIN                                    -> s stays DONE, flag prereq_uncertain
  p NOT_VERIFIABLE -> look through to p's own prerequisites, flag prereq_not_verifiable (D-013)
Video verdict: NON_COMPLIANT if any required step is NOT_DONE or OUT_OF_ORDER; else
NEEDS_REVIEW if any required step is UNCERTAIN; else COMPLIANT.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sopverify.compliance.verdicts import StepVerdict, VideoVerdict
from sopverify.config import CheckerConfig
from sopverify.procedure.schema import Step
from sopverify.temporal.smoothing import StepEstimate

V = StepVerdict


@dataclass
class StepCheck:
    step_id: str
    verdict: StepVerdict
    completion_time_s: float | None
    confidence: float
    max_score: float | None
    violated_prerequisites: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
    steps: dict[str, StepCheck]
    video_verdict: VideoVerdict

    def verdicts(self) -> dict[str, StepVerdict]:
        return {k: v.verdict for k, v in self.steps.items()}


def provisional(step: Step, est: StepEstimate | None, cfg: CheckerConfig) -> tuple[StepVerdict, float | None]:
    if not step.visually_verifiable:
        return V.NOT_VERIFIABLE, None
    if est is None:
        return V.UNCERTAIN, None
    if est.final_on:
        return V.DONE, est.completion_time_s
    if est.max_score < cfg.tau_low:
        return V.NOT_DONE, None
    return V.UNCERTAIN, None


def _effective_prereqs(
    s: str, requires: Mapping[str, Sequence[str]], status: Mapping[str, StepVerdict]
) -> tuple[list[str], bool]:
    out: list[str] = []
    through_nv = False
    stack = list(requires.get(s, []))
    seen: set[str] = set()
    while stack:
        p = stack.pop(0)
        if p in seen or p not in status:
            continue
        seen.add(p)
        if status[p] == V.NOT_VERIFIABLE:
            through_nv = True
            stack.extend(requires.get(p, []))
        else:
            out.append(p)
    return out, through_nv


def video_verdict(steps: Sequence[Step], verdicts: Mapping[str, StepVerdict]) -> VideoVerdict:
    req = [s for s in steps if s.required]
    if any(verdicts[s.id] in (V.NOT_DONE, V.OUT_OF_ORDER) for s in req):
        return VideoVerdict.NON_COMPLIANT
    if any(verdicts[s.id] == V.UNCERTAIN for s in req):
        return VideoVerdict.NEEDS_REVIEW
    return VideoVerdict.COMPLIANT


def check(
    steps: Sequence[Step],
    estimates: Mapping[str, StepEstimate],
    requires: Mapping[str, Sequence[str]],
    cfg: CheckerConfig,
    overrides: Mapping[str, tuple[StepVerdict, float | None]] | None = None,
) -> CheckResult:
    """Apply the rules. `overrides` (from the VLM verifier) replace a step's provisional status."""
    status: dict[str, StepVerdict] = {}
    comp: dict[str, float | None] = {}
    for s in steps:
        v, c = provisional(s, estimates.get(s.id), cfg)
        if overrides and s.id in overrides and v != V.NOT_VERIFIABLE:
            v, c = overrides[s.id]
        status[s.id], comp[s.id] = v, c

    out: dict[str, StepCheck] = {}
    for s in steps:
        est = estimates.get(s.id)
        flags = list(est.flags) if est is not None else []
        violated: list[str] = []
        final = status[s.id]
        if status[s.id] == V.DONE:
            prereqs, through_nv = _effective_prereqs(s.id, requires, status)
            if through_nv:
                flags.append("prereq_not_verifiable")
            for p in prereqs:
                if status[p] == V.DONE:
                    cs, cp = comp[s.id], comp[p]
                    if cs is not None and cp is not None and cs < cp - cfg.order_tolerance_s:
                        violated.append(p)
                elif status[p] == V.NOT_DONE:
                    violated.append(p)
                elif status[p] == V.UNCERTAIN and "prereq_uncertain" not in flags:
                    flags.append("prereq_uncertain")
            if violated:
                final = V.OUT_OF_ORDER
        conf = est.confidence if est is not None else 0.0
        out[s.id] = StepCheck(s.id, final, comp[s.id], float(conf),
                              float(est.max_score) if est is not None else None, violated, flags)
    return CheckResult(out, video_verdict(steps, {k: v.verdict for k, v in out.items()}))
