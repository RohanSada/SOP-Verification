"""Verdict enums and the verdict report schema (spec 7.5)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StepVerdict(str, Enum):
    DONE = "DONE"
    NOT_DONE = "NOT_DONE"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    UNCERTAIN = "UNCERTAIN"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"


class VideoVerdict(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class VerifierInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    used: bool = False
    answer: str | None = None  # yes | no | cannot_tell
    rationale: str | None = None
    prior_verdict: str | None = None
    parse_failures: int = 0


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str
    name: str
    verdict: StepVerdict
    completion_time_s: float | None = None
    confidence: float = 0.0
    max_score: float | None = None
    violated_prerequisites: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    evidence_frame: str | None = None
    evidence_time_s: float | None = None
    required: bool = True
    visually_verifiable: bool = True
    verifier: VerifierInfo = Field(default_factory=VerifierInfo)


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    done: int = 0
    not_done: int = 0
    out_of_order: int = 0
    uncertain: int = 0
    not_verifiable: int = 0

    @classmethod
    def of(cls, steps: list[StepResult]) -> Summary:
        c = {v: 0 for v in StepVerdict}
        for s in steps:
            c[s.verdict] += 1
        return cls(done=c[StepVerdict.DONE], not_done=c[StepVerdict.NOT_DONE],
                   out_of_order=c[StepVerdict.OUT_OF_ORDER], uncertain=c[StepVerdict.UNCERTAIN],
                   not_verifiable=c[StepVerdict.NOT_VERIFIABLE])


class VerdictReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    execution_id: str
    procedure_id: str
    step_set: str
    video_verdict: VideoVerdict
    summary: Summary
    steps: list[StepResult]
    scorer: str
    config_hash: str = ""
    git_commit: str = "unknown"
    runtime_s: float = 0.0
    duration_s: float | None = None
    graph: dict[str, list[str]] = Field(default_factory=dict)
    manual_check: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source: str = "impact"
    created_at: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def step(self, step_id: str) -> StepResult:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        raise KeyError(step_id)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> VerdictReport:
        return cls.model_validate_json(Path(path).read_text())
