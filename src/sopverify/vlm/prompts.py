"""Prompt and JSON-schema builders for the VLM scorer and verifier."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from sopverify.procedure.schema import Step
from sopverify.vlm.backend import VLMRequest

SYSTEM = (
    "You are a careful quality inspector checking an assembly workstation against a standard operating "
    "procedure (SOP). You are shown crops from a fixed camera above the work table. Judge only what is "
    "visible; if a part is hidden by hands or tools, say you cannot tell."
)


def scorer_schema(step_ids: Sequence[str]) -> dict:
    item = {
        "type": "object",
        "properties": {
            "complete": {"type": ["boolean", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["complete", "confidence"],
    }
    return {"type": "object", "properties": {s: item for s in step_ids}, "required": list(step_ids),
            "additionalProperties": False}


VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no", "cannot_tell"]},
        "rationale": {"type": "string", "maxLength": 400},
    },
    "required": ["answer", "rationale"],
}


def _step_lines(steps: Sequence[Step]) -> str:
    return "\n".join(f"- {s.id}: {s.name}. Completed end-state: {s.end_state_text}" for s in steps)


def scorer_request(
    steps: Sequence[Step], frame: Image.Image, t_s: float, refs: dict[str, Image.Image] | None = None,
    meta: dict | None = None,
) -> VLMRequest:
    content: list = []
    if refs:
        content.append("Reference images from an expert demonstration, each showing a step's COMPLETED end-state:")
        for s in steps:
            if s.id in refs:
                content += [f"Reference for {s.id} ({s.name}), completed:", refs[s.id]]
    content += [f"CURRENT frame of the video being checked (t = {t_s:.1f} s):", frame]
    ids = [s.id for s in steps]
    content.append(
        "SOP steps:\n" + _step_lines(steps) + "\n\n"
        "For EACH step, decide whether its completed end-state is visible in the CURRENT frame. "
        "Steps can be completed in any order, so judge each one on its own. Answer with a JSON object only, "
        "mapping each step id to {\"complete\": true | false | null, \"confidence\": 0.0-1.0}. Use null when "
        "the relevant area is occluded or you cannot tell. Step ids: " + ", ".join(ids) + "."
    )
    return VLMRequest(content=content, system=SYSTEM, json_schema=scorer_schema(ids), meta=meta or {})


def verifier_request(
    step: Step, after_refs: Sequence[Image.Image], before_refs: Sequence[Image.Image],
    test_frames: Sequence[tuple[float, Image.Image]], duration_s: float, meta: dict | None = None,
) -> VLMRequest:
    content: list = [f"SOP step {step.id}: {step.name}. {step.description}\nCompleted end-state: {step.end_state_text}"]
    for i, im in enumerate(after_refs):
        content += [f"Expert demonstration AFTER this step (completed) #{i + 1}:", im]
    for i, im in enumerate(before_refs):
        content += [f"Expert demonstration BEFORE this step (not yet done) #{i + 1}:", im]
    for t, im in test_frames:
        content += [f"Video under inspection at t = {t:.1f} s (video length {duration_s:.1f} s):", im]
    content.append(
        "Question: is this step's completed end-state present at the END of the video under inspection? "
        "Compare with the reference images. Answer with JSON only: {\"answer\": \"yes\" | \"no\" | \"cannot_tell\", "
        "\"rationale\": \"one or two short sentences\"}."
    )
    return VLMRequest(content=content, system=SYSTEM, json_schema=VERIFIER_SCHEMA, meta=meta or {})
