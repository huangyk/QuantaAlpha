"""Inject local Research directions into RD-Agent hypothesis generation.

QlibFactorHypothesisGen has no official extra-docs hook: RAG is a hardcoded
string, and hypothesis_specification comes from site-packages prompts.yaml.
This patch appends config/rd_research_prompt.md (or RD_RESEARCH_PROMPT) into
both slots so fin_factor Research follows the user's factor ideas.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PROMPT = _REPO_ROOT / "config" / "rd_research_prompt.md"
_HEADER = "User research directions (from config/rd_research_prompt.md):"


def _prompt_path() -> Path:
    override = os.environ.get("RD_RESEARCH_PROMPT", "").strip()
    return Path(override) if override else _DEFAULT_PROMPT


def _is_blank_prompt(text: str) -> bool:
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<!--") or line.startswith(">"):
            continue
        if line.startswith("#"):
            continue
        if "在此粘贴" in line:
            continue
        return False
    return True


def load_user_research_prompt() -> str:
    path = _prompt_path()
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"[research-prompt-patch] cannot read {path}: {exc}", file=sys.stderr, flush=True)
        return ""
    if not text or _is_blank_prompt(text):
        return ""
    return text


def _inject(context: dict) -> None:  # type: ignore[type-arg]
    text = load_user_research_prompt()
    if not text:
        return
    block = f"{_HEADER}\n{text}"
    spec = str(context.get("hypothesis_specification") or "").rstrip()
    context["hypothesis_specification"] = f"{spec}\n\n{block}" if spec else block
    rag = str(context.get("RAG") or "").rstrip()
    context["RAG"] = f"{rag}\n\n{block}" if rag else block


def apply() -> None:
    try:
        from rdagent.scenarios.qlib.proposal.factor_proposal import QlibFactorHypothesisGen
    except Exception:
        return

    if getattr(QlibFactorHypothesisGen, "_local_research_prompt_patched", False):
        return

    orig = QlibFactorHypothesisGen.prepare_context

    def prepare_context(self, trace):  # type: ignore[no-untyped-def]
        context_dict, json_flag = orig(self, trace)
        if isinstance(context_dict, dict):
            _inject(context_dict)
        return context_dict, json_flag

    QlibFactorHypothesisGen.prepare_context = prepare_context  # type: ignore[method-assign]
    QlibFactorHypothesisGen._local_research_prompt_patched = True
    print(f"[research-prompt-patch] watching {_prompt_path()}", file=sys.stderr, flush=True)
