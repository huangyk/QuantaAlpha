"""Make qlib factor feedback metric formatting robust to NaN IC.

Also ensure process_results never crashes the loop when IC/ICIR are blank
(common on small-sample / constant-pred runs).
"""
from __future__ import annotations

import math
import sys
from typing import Any


def _fmt(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(x):
        return "nan"
    return f"{x:.6f}"


def apply() -> None:
    try:
        from rdagent.scenarios.qlib.developer import feedback as fb
    except Exception:
        return
    if getattr(fb, "_local_feedback_metrics_patched", False):
        return

    important = list(getattr(fb, "IMPORTANT_METRICS", []))

    def process_results(current_result: Any, sota_result: Any) -> str:
        import pandas as pd

        current_df = pd.DataFrame(current_result)
        sota_df = pd.DataFrame(sota_result)
        current_df.index.name = "metric"
        sota_df.index.name = "metric"
        # Series name / column may be str "0" or int 0 depending on pandas path
        current_df = current_df.rename(columns={"0": "Current Result", 0: "Current Result"})
        sota_df = sota_df.rename(columns={"0": "SOTA Result", 0: "SOTA Result"})
        if "Current Result" not in current_df.columns and current_df.shape[1] >= 1:
            current_df = current_df.rename(columns={current_df.columns[0]: "Current Result"})
        if "SOTA Result" not in sota_df.columns and sota_df.shape[1] >= 1:
            sota_df = sota_df.rename(columns={sota_df.columns[0]: "SOTA Result"})

        combined_df = pd.concat([current_df, sota_df], axis=1)
        labels = [m for m in important if m in combined_df.index]
        if not labels:
            return "No overlapping metrics between current and SOTA results."
        filtered = combined_df.loc[labels]
        parts = []
        for metric, row in filtered.iterrows():
            parts.append(
                f"{metric} of Current Result is {_fmt(row.get('Current Result'))}, "
                f"of SOTA Result is {_fmt(row.get('SOTA Result'))}"
            )
        return "; ".join(parts)

    fb.process_results = process_results  # type: ignore[assignment]
    fb._local_feedback_metrics_patched = True
    print("[feedback-metrics-patch] process_results NaN-safe", file=sys.stderr, flush=True)
