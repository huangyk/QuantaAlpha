"""Hardening for Agnes incomplete JSON in CoSTEER factor evaluators.

If the model returns JSON without required decision/feedback keys after retries,
default to reject (False) so the evolve loop continues instead of crashing.
"""
from __future__ import annotations

import sys


def apply() -> None:
    try:
        from rdagent.components.coder.factor_coder.eva_utils import (
            FactorFinalDecisionEvaluator,
            FactorOutputFormatEvaluator,
        )
    except Exception:
        return

    if not getattr(FactorFinalDecisionEvaluator, "_agnes_final_decision_patched", False):
        orig_final = FactorFinalDecisionEvaluator.evaluate

        def evaluate_final(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return orig_final(self, *args, **kwargs)
            except KeyError as e:
                feedback = (
                    "Model response missing final_decision/final_feedback after retries; "
                    f"defaulting to reject. error={e!r}"
                )
                print(f"[final-decision-patch] {feedback}", file=sys.stderr, flush=True)
                return False, feedback

        FactorFinalDecisionEvaluator.evaluate = evaluate_final  # type: ignore[method-assign]
        FactorFinalDecisionEvaluator._agnes_final_decision_patched = True

    if not getattr(FactorOutputFormatEvaluator, "_agnes_output_format_patched", False):
        orig_fmt = FactorOutputFormatEvaluator.evaluate

        def evaluate_format(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return orig_fmt(self, *args, **kwargs)
            except KeyError as e:
                feedback = (
                    "Model response missing output_format_decision/output_format_feedback "
                    f"after retries; defaulting to reject. error={e!r}"
                )
                print(f"[final-decision-patch] {feedback}", file=sys.stderr, flush=True)
                return feedback, False

        FactorOutputFormatEvaluator.evaluate = evaluate_format  # type: ignore[method-assign]
        FactorOutputFormatEvaluator._agnes_output_format_patched = True

    print("[final-decision-patch] applied", file=sys.stderr, flush=True)
