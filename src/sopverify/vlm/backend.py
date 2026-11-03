"""VLM backends: vLLM (preferred on A100), transformers (fallback) and a fake for tests.

Decoding is deterministic (temperature 0). When a JSON schema is given and the backend
supports it, output is constrained to the schema; otherwise callers parse and retry.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from sopverify.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class VLMRequest:
    """A single-turn request. `content` items are str (text) or PIL images, in order."""

    content: list[str | Image.Image]
    system: str | None = None
    json_schema: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def text_only(self) -> str:
        return "\n".join(c if isinstance(c, str) else f"<image {c.size[0]}x{c.size[1]}>" for c in self.content)

    @property
    def n_images(self) -> int:
        return sum(1 for c in self.content if isinstance(c, Image.Image))


def _data_uri(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class VLMBackend:
    name = "base"
    supports_guided_json = False

    def generate(self, requests: list[VLMRequest]) -> list[str]:
        raise NotImplementedError


class VLLMBackend(VLMBackend):
    name = "vllm"
    supports_guided_json = True

    def __init__(self, model_id: str, max_model_len: int = 32768, gpu_memory_utilization: float = 0.85,
                 max_new_tokens: int = 768, temperature: float = 0.0, max_images: int = 20, guided_json: bool = True):
        from vllm import LLM

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.guided = guided_json
        self.llm = LLM(model=model_id, dtype="bfloat16", max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, limit_mm_per_prompt={"image": max_images},
                       seed=0)

    def _params(self, schema: dict[str, Any] | None):
        from vllm import SamplingParams

        kw: dict[str, Any] = {"temperature": self.temperature, "max_tokens": self.max_new_tokens, "seed": 0}
        if schema is not None and self.guided:
            try:  # vLLM >= 0.10.2
                from vllm.sampling_params import StructuredOutputsParams

                return SamplingParams(structured_outputs=StructuredOutputsParams(json=schema), **kw)
            except (ImportError, TypeError):
                try:
                    from vllm.sampling_params import GuidedDecodingParams

                    return SamplingParams(guided_decoding=GuidedDecodingParams(json=schema), **kw)
                except (ImportError, TypeError):
                    log.warning("this vLLM has no structured output support; parsing free text")
        return SamplingParams(**kw)

    @staticmethod
    def _messages(req: VLMRequest) -> list[dict[str, Any]]:
        content = [{"type": "text", "text": c} if isinstance(c, str) else
                   {"type": "image_url", "image_url": {"url": _data_uri(c)}} for c in req.content]
        msgs: list[dict[str, Any]] = []
        if req.system:
            msgs.append({"role": "system", "content": req.system})
        msgs.append({"role": "user", "content": content})
        return msgs

    def generate(self, requests: list[VLMRequest]) -> list[str]:
        if not requests:
            return []
        # One SamplingParams per schema; group requests by schema to batch them.
        out: list[str | None] = [None] * len(requests)
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(requests):
            groups.setdefault(json.dumps(r.json_schema, sort_keys=True), []).append(i)
        for idxs in groups.values():
            params = self._params(requests[idxs[0]].json_schema)
            res = self.llm.chat([self._messages(requests[i]) for i in idxs], params, use_tqdm=False)
            for i, r in zip(idxs, res):
                out[i] = r.outputs[0].text
        return [o or "" for o in out]


class TransformersBackend(VLMBackend):
    name = "transformers"

    def __init__(self, model_id: str, max_new_tokens: int = 768, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device_map)
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                                     device_map=device_map)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()

    def generate(self, requests: list[VLMRequest]) -> list[str]:
        import torch

        outs = []
        for req in requests:
            content = [{"type": "text", "text": c} if isinstance(c, str) else {"type": "image", "image": c}
                       for c in req.content]
            if req.json_schema is not None:
                content.append({"type": "text", "text": "Respond with a single JSON object only."})
            msgs: list[dict[str, Any]] = []
            if req.system:
                msgs.append({"role": "system", "content": [{"type": "text", "text": req.system}]})
            msgs.append({"role": "user", "content": content})
            inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                        return_dict=True, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            outs.append(self.processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
        return outs


class FakeBackend(VLMBackend):
    """Deterministic backend for tests: `respond(request) -> str`."""

    name = "fake"

    def __init__(self, respond):
        self.respond = respond
        self.calls: list[VLMRequest] = []

    def generate(self, requests: list[VLMRequest]) -> list[str]:
        self.calls.extend(requests)
        return [self.respond(r) for r in requests]


def make_backend(cfg) -> VLMBackend:
    """Build the backend named in a VLMConfig ('auto' tries vLLM, then transformers)."""
    kinds = ["vllm", "transformers"] if cfg.backend == "auto" else [cfg.backend]
    last: Exception | None = None
    for kind in kinds:
        try:
            if kind == "vllm":
                return VLLMBackend(cfg.model_id, cfg.max_model_len, cfg.gpu_memory_utilization, cfg.max_new_tokens,
                                   cfg.temperature, cfg.max_images, cfg.guided_json)
            return TransformersBackend(cfg.model_id, cfg.max_new_tokens)
        except Exception as e:  # pragma: no cover - needs GPU stack
            log.warning("VLM backend %s unavailable: %s", kind, e)
            last = e
    raise RuntimeError(f"no VLM backend could be created: {last}")


class PromptLogger:
    """Append every prompt/response to a JSONL file in the run folder (images summarized)."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, req: VLMRequest, response: str, ok: bool, attempt: int) -> None:
        if not self.path:
            return
        rec = {"kind": kind, "meta": req.meta, "attempt": attempt, "parse_ok": ok, "n_images": req.n_images,
               "prompt": req.text_only(), "response": response}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
