"""Pydantic schemas for procedure packages (spec 7.1, 7.2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

GraphSource = Literal["released", "manual", "inferred_all", "inferred_demos"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GTMapping(_M):
    """How eval derives a step's ground truth from dataset annotations."""

    psr_events: list[str] | None = None  # complete when ALL listed events complete
    tas_s_labels: list[str] | None = None  # complete at end of the LAST segment with any listed label

    def kinds(self) -> list[str]:
        return [k for k in ("psr_events", "tas_s_labels") if getattr(self, k)]


class Step(_M):
    id: str
    name: str
    description: str = ""
    # What the workspace looks like once the step is complete; used by the VLM prompts.
    end_state: str | None = None
    requires: list[str] = Field(default_factory=list)
    required: bool = True
    visually_verifiable: bool = True
    roi: list[int] | None = None  # optional per-step crop override [x, y, w, h]
    gt_mapping: GTMapping = Field(default_factory=GTMapping)
    notes: str | None = None

    @field_validator("roi")
    @classmethod
    def _roi_len(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and len(v) != 4:
            raise ValueError("roi must be [x, y, w, h]")
        return v

    @property
    def end_state_text(self) -> str:
        return self.end_state or self.description or self.name


class StepSet(_M):
    step_set: str
    description: str = ""
    status: str | None = None  # e.g. "draft - pending user review"
    steps: list[Step]

    def ids(self) -> list[str]:
        return [s.id for s in self.steps]

    def by_id(self) -> dict[str, Step]:
        return {s.id: s for s in self.steps}


class ROIConfig(_M):
    """Static workspace crop in front-view pixel coordinates."""

    box: list[int]  # [x, y, w, h]
    frame_size: list[int] = Field(default_factory=lambda: [1280, 720])  # [width, height]
    confirmed: bool = False
    notes: str | None = None
    step_overrides: dict[str, list[int]] = Field(default_factory=dict)

    @field_validator("box")
    @classmethod
    def _box(cls, v: list[int]) -> list[int]:
        if len(v) != 4 or v[2] <= 0 or v[3] <= 0:
            raise ValueError("box must be [x, y, w, h] with positive w, h")
        return v


class VocabEvent(_M):
    id: str
    kind: str
    component: str
    impact_id: int | None = None


class Vocabulary(_M):
    """Annotation vocabulary the GT mappings may refer to."""

    source: str | None = None
    components: list[str] = Field(default_factory=list)
    psr_events: list[VocabEvent] = Field(default_factory=list)
    tas_s_labels: list[str] = Field(default_factory=list)

    def psr_ids(self) -> set[str]:
        return {e.id for e in self.psr_events}

    def event(self, event_id: str) -> VocabEvent | None:
        for e in self.psr_events:
            if e.id == event_id:
                return e
        return None


class ProcedureSpec(_M):
    procedure_id: str
    name: str
    sop_text: str = "sop.md"
    sop_status: str | None = None
    default_step_set: str
    step_sets: dict[str, str]
    graph_source: GraphSource = "manual"
    # Event-level prerequisite graph shipped with the dataset (for graph_source: released).
    released_graph: str | None = None
    roi: str = "roi.yaml"
    camera: str = "front"
    vocabulary: str | None = "vocab.yaml"
    dataset: str | None = None
