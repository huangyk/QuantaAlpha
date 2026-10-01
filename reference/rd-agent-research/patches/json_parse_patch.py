"""Harden RD-Agent JSON parsing for Agnes (empty / fenced / Python-bool replies)."""
from __future__ import annotations

import sys

import json
import re

from rdagent.oai.backend.base import JSONParser

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract(content: str) -> str:
    text = (content or "").strip()
    if not text:
        raise json.JSONDecodeError("empty model response", "", 0)
    match = _FENCE.search(text)
    if match:
        text = match.group(1).strip()
    if not text.startswith("{") and not text.startswith("["):
        obj = _OBJECT.search(text)
        if obj:
            text = obj.group(0)
    return text


def _parse(self: JSONParser, content: str) -> str:
    original = content
    candidates = [content, _extract(content)]
    for candidate in candidates:
        for strategy in self.strategies:
            try:
                return strategy(candidate)
            except json.JSONDecodeError:
                continue
            except Exception:
                continue
    if not self.add_json_in_prompt:
        error = json.JSONDecodeError(
            "Failed to parse JSON after all attempts, maybe because 'messages' must contain the word 'json' in some form",
            original or "",
            0,
        )
        error.message = error.args[0]  # type: ignore[attr-defined]
        raise error
    raise json.JSONDecodeError("Failed to parse JSON after all attempts", original or "", 0)


def apply() -> None:
    if getattr(JSONParser, "_agnes_json_patched", False):
        return
    JSONParser.parse = _parse  # type: ignore[method-assign]
    JSONParser._agnes_json_patched = True
    print("[json-parse-patch] applied", file=sys.stderr, flush=True)
