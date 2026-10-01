"""Convert factor combine/runtime pandas index errors into FactorEmptyError.

LLM-generated factors sometimes produce broken MultiIndex frames; pd.concat then
raises NotImplementedError and kills the whole fin_factor run. FactorEmptyError is
already in FactorRDLoop.skip_loop_error, so remapping keeps the evolve loop alive.
"""
from __future__ import annotations

import sys


def apply() -> None:
    try:
        from rdagent.core.exception import FactorEmptyError
        from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
    except Exception:
        return

    if getattr(QlibFactorRunner, "_agnes_factor_empty_patched", False):
        return

    orig = QlibFactorRunner.develop

    def develop(self, exp):  # type: ignore[no-untyped-def]
        try:
            return orig(self, exp)
        except FactorEmptyError:
            raise
        except (NotImplementedError, ValueError, KeyError) as e:
            msg = str(e)
            soft = (
                "Can only union MultiIndex" in msg
                or "occurs multiple times" in msg
                or "cannot insert" in msg
                or "Column not found" in msg
            )
            if not soft:
                raise
            wrapped = FactorEmptyError(f"Factor combine/runtime failed: {msg}")
            print(f"[factor-empty-patch] {wrapped}", file=sys.stderr, flush=True)
            raise wrapped from e

    QlibFactorRunner.develop = develop  # type: ignore[method-assign]
    QlibFactorRunner._agnes_factor_empty_patched = True
    print("[factor-empty-patch] applied", file=sys.stderr, flush=True)
