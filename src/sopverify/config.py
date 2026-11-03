"""Typed configuration: pydantic models, YAML loading with env expansion, and stage hashes.

Configs are layered: `configs/default.yaml` is the base, experiment files say
`extends: ../default.yaml` and add `variants`, each a dict of overrides. Every
stage (frames, features, scores, ...) has a hash built only from the fields it
depends on, and caches are keyed by that hash.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sopverify.utils import read_yaml, stable_hash


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PathsConfig(_Base):
    data_root: str = "${SOPVERIFY_DATA_ROOT:-/content/drive/MyDrive/sopverify/data/IMPACT-v1.1}"
    # Annotations may live inside data_root (release bundle) or in a clone of the
    # IMPACT GitHub repo (`<clone>/dataset`). None means `<data_root>/annotations`.
    annotations_root: str | None = "${SOPVERIFY_ANNOTATIONS:-}"
    cache_root: str = "${SOPVERIFY_CACHE:-/content/drive/MyDrive/sopverify/cache}"
    runs_root: str = "${SOPVERIFY_RUNS:-/content/drive/MyDrive/sopverify/runs}"
    exports_root: str = "${SOPVERIFY_EXPORTS:-/content/drive/MyDrive/sopverify/exports}"
    scratch: str = "${SOPVERIFY_SCRATCH:-/content/scratch}"

    def resolved(self) -> dict[str, Path | None]:
        out: dict[str, Path | None] = {}
        for k, v in self.model_dump().items():
            val = expand_env(v) if isinstance(v, str) else v
            out[k] = Path(val) if val else None
        if out["annotations_root"] is None and out["data_root"] is not None:
            out["annotations_root"] = out["data_root"] / "annotations"
        return out


class DataConfig(_Base):
    dataset: Literal["impact"] = "impact"
    task: str = "Reassembly"
    model: str = "A"
    view: str = "front"
    # An execution is in the pool only if it has all of these annotation kinds.
    require: list[str] = Field(default_factory=lambda: ["psr", "tas_s", "asr"])
    # Fill in after M1 if the release metadata does not state expertise.
    expertise_overrides: dict[str, Literal["expert", "novice"]] = Field(default_factory=dict)


class DemosConfig(_Base):
    k: int = 2
    # complete_low_anomaly: all components installed at the end, no Remove events,
    #   anomaly time <= max_anomaly_frac per hand (decision D-003).
    # any_complete: completeness only. experts_only: completeness + expert participant.
    mode: Literal["complete_low_anomaly", "any_complete", "experts_only"] = "complete_low_anomaly"
    max_anomaly_frac: float = 0.05
    expertise: Literal["any", "prefer_expert", "expert_only"] = "prefer_expert"
    n_seeds: int = 5
    seed: int = 0
    distinct_participants: bool = True
    # Remove all executions of demo participants from dev/test (spec 9.2 optional mode).
    held_out_participant: bool = False
    # Fixed demo ids (used by the app and `sopverify run`); overrides the random draw.
    fixed_ids: list[str] | None = None


class SplitsConfig(_Base):
    dev_fraction: float = 0.3
    participant_disjoint: bool = True
    seed: int = 0


class VideoConfig(_Base):
    sample_fps: float = 2.0
    # The IMPACT ASR metadata reports 1280x720 for every front video (spec said 1280x800, D-002).
    expected_resolution: tuple[int, int] = (1280, 720)
    frame_cache: bool = True
    # Frame caches are an intermediate; keep them on local scratch, not Drive (D-010).
    frame_cache_on_scratch: bool = True
    decode_threads: int = 4


class FeaturesConfig(_Base):
    extractor: Literal["dinov2", "shipped_videomaev2"] = "dinov2"
    model_id: str = "facebook/dinov2-base"
    crop: Literal["full", "roi"] = "roi"
    pooling: Literal["cls", "cls_mean"] = "cls"
    image_size: int = 224
    resize_mode: Literal["letterbox", "squash"] = "letterbox"
    batch_size: int = 128
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    l2_normalize: bool = True
    shipped_model: str = "VideoMAEv2"


class LabelsConfig(_Base):
    pos_margin_s: float = 3.0
    neg_margin_s: float = 10.0
    # When a step completes less than pos_margin_s before the video ends (the last
    # step usually does, D-006), use frames from completion + tail/2 as positives.
    adaptive_tail: bool = True


class ScorerConfig(_Base):
    name: Literal["prototype", "linear_probe", "knn", "time_prior", "oracle", "vlm"] = "linear_probe"
    knn_k: int = 15
    linear_C: float | None = None
    linear_C_grid: list[float] = Field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    linear_max_iter: int = 2000
    prototype_fit_bias: bool = True


class SmoothingConfig(_Base):
    median_window_s: float = 2.5
    tau_on: float = 0.6
    tau_off: float = 0.4
    min_on_s: float = 5.0
    # A run that reaches the end of the video only needs this long (D-006).
    min_on_end_s: float = 1.0
    evidence_offset_s: float = 3.0


class CheckerConfig(_Base):
    tau_low: float = 0.3
    order_tolerance_s: float = 5.0
    final_state_check: bool = False


class GraphConfig(_Base):
    # None = use the procedure package's graph_source.
    source: Literal["released", "manual", "inferred_all", "inferred_demos"] | None = None
    min_support: int = 5


class VLMConfig(_Base):
    enabled: bool = False
    backend: Literal["vllm", "transformers", "auto"] = "vllm"
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    keyframe_s: float = 10.0
    in_context_refs: bool = True
    verifier: bool = False
    # Also send NOT_DONE steps whose confidence is below this to the verifier (None = off).
    verify_not_done_below: float | None = None
    max_long_side: int = 768
    max_retries: int = 2
    temperature: float = 0.0
    max_new_tokens: int = 768
    batch_size: int = 16
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.85
    max_steps_per_prompt: int = 20
    guided_json: bool = True
    log_prompts: bool = True


class TruncationConfig(_Base):
    enabled: bool = True
    n_per_video: int = 3
    m_choices: list[int] = Field(default_factory=lambda: [1, 2, 3])
    seed: int = 0
    # Cut points stay at least this far from any GT completion (D-008).
    min_gap_s: float = 5.0


class TuneConfig(_Base):
    enabled: bool = False
    # Dotted config paths -> candidate values, searched on dev traces only.
    grid: dict[str, list[Any]] = Field(default_factory=dict)


class EvalConfig(_Base):
    truncation: TruncationConfig = Field(default_factory=TruncationConfig)
    tune: TuneConfig = Field(default_factory=TuneConfig)
    splits: list[Literal["dev", "test"]] = Field(default_factory=lambda: ["dev", "test"])
    save_traces: bool = True


class TargetsConfig(_Base):
    not_done_recall: float = 0.90
    max_false_alarms_per_compliant: float = 1.0
    max_review_rate: float = 0.20


class ReportConfig(_Base):
    html: bool = True
    show_gt: bool = True
    evidence: bool = True


class Config(_Base):
    procedure: str = "procedures/impact_reassembly_A"
    step_set: str | None = None
    run_name: str | None = None
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    demos: DemosConfig = Field(default_factory=DemosConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    scorer: ScorerConfig = Field(default_factory=ScorerConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    checker: CheckerConfig = Field(default_factory=CheckerConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    # ------------------------------------------------------------------ helpers
    def paths_resolved(self) -> dict[str, Path | None]:
        return self.paths.resolved()

    def procedure_path(self, base: Path | None = None) -> Path:
        p = Path(expand_env(self.procedure))
        if p.is_absolute() or p.exists():
            return p
        root = base or repo_root()
        return root / p

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with nested and/or dotted-key overrides applied."""
        data = self.model_dump()
        data = deep_merge(data, expand_dotted(overrides))
        return Config.model_validate(data)

    def stage_hash(self, stage: str, **extra: Any) -> str:
        return stage_hash(self, stage, **extra)


# --------------------------------------------------------------------- env / IO
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    """Expand `${VAR}` and `${VAR:-default}` (bash-style; empty counts as unset)."""

    def repl(m: re.Match[str]) -> str:
        var, default = m.group(1), m.group(2)
        val = os.environ.get(var)
        if val:
            return val
        return default if default is not None else ""

    return _ENV_RE.sub(repl, value)


def repo_root() -> Path:
    """Repository root (the folder holding `configs/` and `procedures/`)."""
    env = os.environ.get("SOPVERIFY_REPO")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "configs").is_dir() and (parent / "procedures").is_dir():
            return parent
    return Path.cwd()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_LITERAL_DOTTED_KEYS = {"grid"}  # dicts whose keys are dotted config paths, kept as-is


def expand_dotted(d: dict[str, Any]) -> dict[str, Any]:
    """{'a.b': 1, 'c': {'d': 2}} -> {'a': {'b': 1}, 'c': {'d': 2}} (except inside `grid`)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict) and k.split(".")[-1] not in _LITERAL_DOTTED_KEYS:
            v = expand_dotted(v)
        parts = k.split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        if isinstance(v, dict) and isinstance(cur.get(parts[-1]), dict):
            cur[parts[-1]] = deep_merge(cur[parts[-1]], v)
        else:
            cur[parts[-1]] = v
    return out


def _load_layered(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Config 'extends' cycle at {path}")
    seen.add(path)
    raw = read_yaml(path) or {}
    parent = raw.pop("extends", None)
    if parent:
        base = _load_layered((path.parent / parent), seen)
        raw = deep_merge(base, expand_dotted(raw))
    return expand_dotted(raw)


_EXPERIMENT_KEYS = ("name", "question", "variants", "description", "notes")


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a config file (following `extends`), dropping experiment-only keys."""
    if path is None:
        path = repo_root() / "configs" / "default.yaml"
    data = _load_layered(Path(path))
    for key in _EXPERIMENT_KEYS:
        data.pop(key, None)
    if overrides:
        data = deep_merge(data, expand_dotted(overrides))
    return Config.model_validate(data)


class Experiment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    question: str = ""
    path: str
    variants: dict[str, Config]


def load_experiment(path: str | Path, only: list[str] | None = None) -> Experiment:
    """An experiment file = base config + named variants (each a dict of overrides)."""
    path = Path(path)
    data = _load_layered(path)
    name = data.pop("name", path.stem)
    question = data.pop("question", "")
    variants_raw = data.pop("variants", None) or {"default": {}}
    for key in _EXPERIMENT_KEYS:
        data.pop(key, None)
    variants: dict[str, Config] = {}
    for vname, ov in variants_raw.items():
        if only and vname not in only:
            continue
        cfg = Config.model_validate(deep_merge(data, expand_dotted(ov or {})))
        if cfg.run_name is None:
            cfg = cfg.model_copy(update={"run_name": f"{name}__{vname}"})
        variants[vname] = cfg
    return Experiment(name=name, question=question, path=str(path), variants=variants)


# ------------------------------------------------------------------ stage hashes
# Each stage lists the config sections (or dotted fields) its output depends on.
STAGE_FIELDS: dict[str, list[str]] = {
    "frames": ["video.sample_fps", "features.crop", "features.image_size", "features.resize_mode"],
    "features": [
        "video.sample_fps", "features.extractor", "features.model_id", "features.crop",
        "features.pooling", "features.image_size", "features.resize_mode", "features.l2_normalize",
        "features.shipped_model",
    ],
    "scores": [
        "procedure", "step_set", "video.sample_fps", "features", "labels", "scorer", "demos.k",
        "demos.mode", "demos.max_anomaly_frac", "demos.expertise",
    ],
    "vlm_scores": ["procedure", "step_set", "video.sample_fps", "vlm", "demos.k", "demos.mode"],
}


def _get_dotted(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def stage_hash(cfg: Config, stage: str, **extra: Any) -> str:
    """Hash of only the fields `stage` depends on, plus any extra inputs (ROI box, demo ids...)."""
    if stage not in STAGE_FIELDS:
        raise KeyError(f"Unknown stage {stage!r}; known: {sorted(STAGE_FIELDS)}")
    dumped = cfg.model_dump(mode="json")
    payload = {f: _get_dotted(dumped, f) for f in STAGE_FIELDS[stage]}
    if stage in ("features", "frames") and _get_dotted(dumped, "features.extractor") == "shipped_videomaev2":
        # Shipped features ignore crop/size settings.
        payload = {
            "video.sample_fps": payload.get("video.sample_fps"),
            "features.extractor": "shipped_videomaev2",
            "features.shipped_model": _get_dotted(dumped, "features.shipped_model"),
        }
    payload["_stage"] = stage
    payload["_extra"] = extra
    return stable_hash(payload)
