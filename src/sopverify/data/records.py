"""Canonical, dataset-independent execution records (spec 7.3).

Two types matter for the test-label firewall:

* `ExecutionRecord` carries annotations. Only demo builders and `sopverify.eval`
  may hold records of executions that are being judged.
* `ExecutionRef` carries no annotations. `pipeline.run` only accepts this for the
  video it judges.

All times are seconds.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class _Rec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PSREvent(_Rec):
    event_id: str  # e.g. "install:gearbox_housing"
    time_s: float
    frame: int | None = None


class Segment(_Rec):
    label: str
    start_s: float
    end_s: float


class ASRStates(_Rec):
    times_s: list[float]
    components: list[str]
    states: list[list[int]]  # [n_keyframes][n_components], values in {-1, 0, 1}

    def state_at(self, t_s: float) -> list[int] | None:
        """State vector in force at time t (last keyframe with time <= t)."""
        cur: list[int] | None = None
        for t, s in zip(self.times_s, self.states):
            if t <= t_s + 1e-9:
                cur = s
            else:
                break
        return cur

    def final(self) -> list[int] | None:
        return self.states[-1] if self.states else None


class Phase(_Rec):
    hand: str  # "L" | "R"
    phase: str  # "ANOMALY" | "RECOVERY" | "NORMAL"
    start_s: float
    end_s: float


class AnomalyTypeSpan(_Rec):
    hand: str
    start_s: float
    end_s: float
    types: list[str]


class ExecutionRef(_Rec):
    """Everything the inference pipeline may know about a video: no annotations."""

    execution_id: str
    view: str = "front"
    participant: str | None = None
    video_path: str | None = None
    fps: float | None = None
    n_frames: int | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    source: str = "impact"  # "impact" | "upload"
    shipped_features: dict[str, str] = Field(default_factory=dict)  # model -> path

    def truncated(self, t_end_s: float) -> ExecutionRef:
        dur = min(self.duration_s or t_end_s, t_end_s)
        n = int(round(dur * self.fps)) if self.fps else None
        return self.model_copy(update={"duration_s": dur, "n_frames": n})

    @property
    def video(self) -> Path | None:
        return Path(self.video_path) if self.video_path else None


class ExecutionRecord(_Rec):
    """Annotated execution (spec 7.3)."""

    execution_id: str
    participant: str
    task: str
    model: str
    trial: int
    expertise: str = "unknown"  # "expert" | "novice" | "unknown"
    view: str = "front"
    video_path: str | None = None
    fps: float
    n_frames: int
    duration_s: float
    width: int | None = None
    height: int | None = None
    psr_events: list[PSREvent] = Field(default_factory=list)
    tas_s_segments: list[Segment] = Field(default_factory=list)
    asr_states: ASRStates | None = None
    phases: list[Phase] = Field(default_factory=list)
    anomaly_types: list[AnomalyTypeSpan] = Field(default_factory=list)
    has_anomaly: bool = False
    anomaly_frac: dict[str, float] = Field(default_factory=dict)  # hand -> fraction of frames
    shipped_features: dict[str, str] = Field(default_factory=dict)

    def to_ref(self) -> ExecutionRef:
        return ExecutionRef(
            execution_id=self.execution_id,
            view=self.view,
            participant=self.participant,
            video_path=self.video_path,
            fps=self.fps,
            n_frames=self.n_frames,
            duration_s=self.duration_s,
            width=self.width,
            height=self.height,
            source="impact",
            shipped_features=dict(self.shipped_features),
        )

    # Convenience views ---------------------------------------------------
    def n_remove_events(self) -> int:
        return sum(1 for e in self.psr_events if e.event_id.startswith("remove:"))

    def n_anomaly_segments(self) -> int:
        return sum(1 for p in self.phases if p.phase == "ANOMALY")

    def final_component_states(self) -> dict[str, int]:
        if self.asr_states is None or not self.asr_states.states:
            return {}
        return dict(zip(self.asr_states.components, self.asr_states.states[-1]))
