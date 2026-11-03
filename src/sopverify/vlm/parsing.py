"""Robust JSON extraction from VLM output."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _balanced_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _repair(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    return s


def extract_json(text: str) -> Any:
    """Parse the first JSON object in `text` (handles code fences, prose, light repairs)."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        for attempt in (cand, _balanced_object(cand)):
            if not attempt:
                continue
            for s in (attempt, _repair(attempt)):
                try:
                    return json.loads(s)
                except (json.JSONDecodeError, ValueError):
                    continue
    raise ValueError(f"no JSON object found in model output: {text[:200]!r}")
