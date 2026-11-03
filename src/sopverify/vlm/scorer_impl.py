"""VLM state scorer (spec 8.9): per-keyframe step completion judgements -> score trace."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from sopverify.config import LabelsConfig, VLMConfig
from sopverify.data.records import ExecutionRef
from sopverify.logging_utils import get_logger
from sopverify.procedure.schema import Step
from sopverify.state.scorers.base import BaseScorer
from sopverify.state.types import DemoData
from sopverify.video.timegrid import step_hold
from sopverify.vlm.backend import PromptLogger, VLMBackend, VLMRequest
from sopverify.vlm.parsing import extract_json
from sopverify.vlm.prompts import scorer_request

log = get_logger(__name__)


def answer_to_prob(ans: object) -> float:
    """{"complete": true|false|null, "confidence": c} -> P(complete). null -> 0.5."""
    if not isinstance(ans, dict):
        return 0.5
    c = ans.get("complete")
    try:
        conf = float(ans.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = min(max(conf, 0.0), 1.0)
    if c is None:
        return 0.5
    if isinstance(c, str):
        c = c.strip().lower() in ("true", "yes", "1")
    return conf if c else 1.0 - conf


def run_with_retries(backend: VLMBackend, requests: list[VLMRequest], validate, max_retries: int,
                     logger: PromptLogger, kind: str, batch_size: int = 16) -> tuple[list[object | None], int]:
    """Generate, parse and validate; re-ask failures up to `max_retries` times. Returns (parsed, n_failures)."""
    results: list[object | None] = [None] * len(requests)
    pending = list(range(len(requests)))
    failures = 0
    for attempt in range(max_retries + 1):
        if not pending:
            break
        reqs = []
        for i in pending:
            r = requests[i]
            if attempt > 0:
                r = VLMRequest(content=[*r.content, "Your previous answer was not valid JSON for the requested "
                                        "format. Reply with the JSON object only."], system=r.system,
                               json_schema=r.json_schema, meta=r.meta)
            reqs.append(r)
        texts: list[str] = []
        for b in range(0, len(reqs), batch_size):
            texts += backend.generate(reqs[b : b + batch_size])
        still = []
        for i, r, text in zip(pending, reqs, texts):
            try:
                parsed = validate(extract_json(text))
                results[i] = parsed
                logger.log(kind, r, text, True, attempt)
            except Exception:
                failures += 1
                logger.log(kind, r, text, False, attempt)
                still.append(i)
        pending = still
    return results, failures


class VLMStateScorer(BaseScorer):
    name = "vlm"
    needs_features = False

    def __init__(self, cfg: VLMConfig, backend: VLMBackend, frames, labels: LabelsConfig | None = None,
                 log_path=None):
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        self.frames = frames  # VideoFrameSource-like: crops(video_path, times, fps) -> [PIL]
        self.labels = labels or LabelsConfig()
        self.logger = PromptLogger(log_path)
        self.steps: list[Step] = []
        self.refs: dict[str, object] = {}
        self.stats = {"requests": 0, "parse_failures": 0, "videos": 0, "seconds": 0.0}

    def fit(self, demos: list[DemoData], steps: list[Step]) -> None:
        self.steps = list(steps)
        self.step_ids = [s.id for s in steps]
        self.refs = {}
        if not self.cfg.in_context_refs:
            return
        for s in steps:
            for d in demos:
                c = d.completion_s.get(s.id)
                if c is None or not d.video_path:
                    continue
                t = min(c + self.labels.pos_margin_s, d.duration_s - self.cfg.final_keyframe_margin_s)
                self.refs[s.id] = self.frames.crops(d.video_path, [max(t, c)], d.fps)[0]
                break
        self.info = {"refs": sorted(self.refs)}

    def keyframes(self, duration_s: float) -> np.ndarray:
        kf = np.arange(0.0, max(duration_s - self.cfg.final_keyframe_margin_s, 0.0), self.cfg.keyframe_s)
        last = max(duration_s - self.cfg.final_keyframe_margin_s, 0.0)
        if len(kf) == 0 or last - kf[-1] > 0.5:
            kf = np.append(kf, last)
        return kf

    def _chunks(self) -> list[list[Step]]:
        n = max(1, self.cfg.max_steps_per_prompt)
        return [self.steps[i : i + n] for i in range(0, len(self.steps), n)]

    def score(self, times_s: np.ndarray, feats: np.ndarray | None, execution: ExecutionRef) -> np.ndarray:
        self._check_fitted()
        if not execution.video_path:
            raise FileNotFoundError(f"{execution.execution_id}: the VLM scorer needs the video file")
        t0 = time.time()
        dur = execution.duration_s or float(times_s[-1])
        kf = self.keyframes(dur)
        imgs = self.frames.crops(execution.video_path, list(kf), execution.fps)
        reqs, where = [], []
        for ki, (t, img) in enumerate(zip(kf, imgs)):
            for chunk in self._chunks():
                refs = {s.id: self.refs[s.id] for s in chunk if s.id in self.refs} if self.cfg.in_context_refs else None
                reqs.append(scorer_request(chunk, img, float(t), refs,
                                           meta={"execution_id": execution.execution_id, "t": float(t)}))
                where.append((ki, [s.id for s in chunk]))

        def validate(obj):
            if not isinstance(obj, dict):
                raise ValueError("not an object")
            return obj

        parsed, fails = run_with_retries(self.backend, reqs, validate, self.cfg.max_retries, self.logger,
                                         "scorer", self.cfg.batch_size)
        P = np.full((len(kf), len(self.step_ids)), 0.5, dtype=np.float32)
        col = {s: j for j, s in enumerate(self.step_ids)}
        for (ki, ids), obj in zip(where, parsed):
            if not isinstance(obj, dict):
                continue
            for sid in ids:
                P[ki, col[sid]] = answer_to_prob(obj.get(sid))
        self.stats["requests"] += len(reqs)
        self.stats["parse_failures"] += sum(1 for o in parsed if o is None)
        self.stats["parse_retries"] = self.stats.get("parse_retries", 0) + fails
        self.stats["videos"] += 1
        self.stats["seconds"] += time.time() - t0
        self.info = {**self.info, "stats": dict(self.stats)}
        return step_hold(kf, P, np.asarray(times_s))


def refs_at(frames, demos: Sequence[DemoData], step_id: str, offset_s: float, n: int, after: bool,
            margin_s: float = 0.2) -> list:
    """Up to n demo crops after (completion + offset) or before (completion - offset) a step.

    One crop per demo first; with fewer demos than n, further crops come from the same demos
    at larger offsets (2x, 3x ...), so K=1 still yields n distinct references.
    """
    usable = [d for d in demos if d.completion_s.get(step_id) is not None and d.video_path]
    out: list = []
    mult = 1
    while usable and len(out) < n and mult <= n:
        for d in usable:
            c = d.completion_s[step_id]
            assert c is not None
            t = min(c + mult * offset_s, d.duration_s - margin_s) if after else max(c - mult * offset_s, 0.0)
            out += frames.crops(d.video_path, [t], d.fps)
            if len(out) >= n:
                break
        mult += 1
    return out[:n]
