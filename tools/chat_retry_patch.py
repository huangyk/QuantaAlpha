"""When Agnes keeps returning non-JSON under json_mode, soft-fail instead of aborting.

RD-Agent retries chat completion 10 times on JSONDecodeError then raises RuntimeError
and kills the whole fin_factor run. This patch returns a minimal reject JSON so
CoSTEER can continue evolving.
"""
from __future__ import annotations

import json
import sys
from typing import Any


_FALLBACK = json.dumps(
    {
        "final_decision": False,
        "final_feedback": "Model returned non-JSON after retries; defaulting to reject.",
        "output_format_decision": False,
        "output_format_feedback": "Model returned non-JSON after retries; defaulting to reject.",
        "ok": False,
        "reason": "json_parse_exhausted",
    }
)


def apply() -> None:
    try:
        from rdagent.oai.backend.base import APIBackend
    except Exception:
        return
    if getattr(APIBackend, "_agnes_chat_retry_patched", False):
        return

    orig = APIBackend._try_create_chat_completion_or_embedding

    def _try_create_chat_completion_or_embedding(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        try:
            return orig(self, *args, **kwargs)
        except RuntimeError as e:
            msg = str(e)
            if "Failed to create chat completion after" not in msg:
                raise
            # Only soft-fail for chat JSON flows (factor coding / final decision).
            if kwargs.get("chat_completion", True) is False:
                raise
            json_mode = bool(kwargs.get("json_mode", False))
            # Messages often require json even when json_mode flag path already failed.
            messages = kwargs.get("messages") or []
            wants_json = json_mode or any(
                isinstance(m, dict) and "json" in str(m.get("content", "")).lower() for m in messages
            )
            if not wants_json:
                raise
            print(
                f"[chat-retry-patch] soft-fail after chat retries: {msg}",
                file=sys.stderr,
                flush=True,
            )
            return _FALLBACK

    APIBackend._try_create_chat_completion_or_embedding = _try_create_chat_completion_or_embedding  # type: ignore[method-assign]
    APIBackend._agnes_chat_retry_patched = True
    print("[chat-retry-patch] applied", file=sys.stderr, flush=True)
