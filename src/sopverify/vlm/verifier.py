"""VLM verifier for UNCERTAIN (and optionally low-confidence NOT_DONE) steps (spec 8.9)."""

from __future__ import annotations

from collections.abc import Sequence

from sopverify.compliance.checker import CheckResult
from sopverify.compliance.verdicts import StepVerdict, VerifierInfo
from sopverify.config import Config
from sopverify.data.records import ExecutionRef
from sopverify.logging_utils import get_logger
from sopverify.procedure.schema import Step
from sopverify.state.types import DemoData
from sopverify.temporal.smoothing import StepEstimate
from sopverify.vlm.backend import PromptLogger, VLMBackend
from sopverify.vlm.prompts import verifier_request
from sopverify.vlm.scorer_impl import refs_at, run_with_retries

log = get_logger(__name__)


class VLMVerifier:
    def __init__(self, cfg: Config, backend: VLMBackend, frames, log_path=None):
        self.cfg = cfg
        self.backend = backend
        self.frames = frames
        self.logger = PromptLogger(log_path)
        self.stats = {"requests": 0, "parse_failures": 0}

    def candidates(self, steps: Sequence[Step], res: CheckResult) -> list[Step]:
        out = []
        thr = self.cfg.vlm.verify_not_done_below
        for s in steps:
            c = res.steps[s.id]
            if c.verdict == StepVerdict.UNCERTAIN:
                out.append(s)
            elif thr is not None and c.verdict == StepVerdict.NOT_DONE and c.confidence < thr:
                out.append(s)
        return out

    def verify(self, ref: ExecutionRef, steps: Sequence[Step], estimates: dict[str, StepEstimate],
               res: CheckResult, demos: Sequence[DemoData], procedure=None
               ) -> dict[str, tuple[StepVerdict, float | None, VerifierInfo]]:
        todo = self.candidates(steps, res)
        if not todo or not ref.video_path:
            return {}
        dur = ref.duration_s or 0.0
        m = self.cfg.vlm.final_keyframe_margin_s
        reqs = []
        for s in todo:
            est = estimates.get(s.id)
            times = sorted({round(max(0.9 * dur, 0.0), 2), round(max(dur - m, 0.0), 2)})
            if est is not None and est.max_score_time_s is not None:
                times.append(round(est.max_score_time_s, 2))
            imgs = self.frames.crops(ref.video_path, times, ref.fps)
            after = refs_at(self.frames, demos, s.id, self.cfg.labels.pos_margin_s, 2, after=True, margin_s=m)
            before = refs_at(self.frames, demos, s.id, self.cfg.labels.neg_margin_s, 2, after=False)
            reqs.append(verifier_request(s, after, before, list(zip(times, imgs)), dur,
                                         meta={"execution_id": ref.execution_id, "step_id": s.id}))

        def validate(obj):
            if not isinstance(obj, dict) or obj.get("answer") not in ("yes", "no", "cannot_tell"):
                raise ValueError("bad verifier answer")
            return obj

        parsed, fails = run_with_retries(self.backend, reqs, validate, self.cfg.vlm.max_retries, self.logger,
                                         "verifier", self.cfg.vlm.batch_size)
        self.stats["requests"] += len(reqs)
        self.stats["parse_failures"] += sum(1 for p in parsed if p is None)
        out: dict[str, tuple[StepVerdict, float | None, VerifierInfo]] = {}
        for s, obj in zip(todo, parsed):
            prior = res.steps[s.id].verdict.value
            if not isinstance(obj, dict):
                out[s.id] = (res.steps[s.id].verdict, None,
                             VerifierInfo(used=True, answer=None, rationale="unparseable", prior_verdict=prior,
                                          parse_failures=self.cfg.vlm.max_retries + 1))
                continue
            ans, why = obj["answer"], str(obj.get("rationale", ""))[:400]
            est = estimates.get(s.id)
            if ans == "yes":
                v, t = StepVerdict.DONE, (est.max_score_time_s if est else None)
            elif ans == "no":
                v, t = StepVerdict.NOT_DONE, None
            else:
                v, t = StepVerdict.UNCERTAIN, None
            out[s.id] = (v, t, VerifierInfo(used=True, answer=ans, rationale=why, prior_verdict=prior))
        return out
