"""Make RD-Agent Streamlit UI metric selection defensive.

app.py does `sms.loc[QLIB_SELECTED_METRICS]` assuming a Series indexed by
IC / 1day.excess_return_with_cost.*. Our qlib_res.csv often:

- stores metric names as the index (and IC as a blank / NaN cell), or
- is a one-row frame with metric names as columns, or
- is a baseline that only has l2.train / l2.valid

Missing keys used to KeyError and kill the whole log view.
"""
from __future__ import annotations

import sys
from collections.abc import Hashable

import pandas as pd

QLIB_SELECTED_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
    "1day.excess_return_with_cost.max_drawdown",
]

# feedback.py uses a shorter list
IMPORTANT_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
]

# hypothesis_and_feedback jinja (prompts.yaml) uses without_cost
PROMPT_RESULT_METRICS = [
    "IC",
    "1day.excess_return_without_cost.annualized_return",
    "1day.excess_return_without_cost.max_drawdown",
]

_KNOWN_METRIC_LISTS = (QLIB_SELECTED_METRICS, IMPORTANT_METRICS, PROMPT_RESULT_METRICS)


def _is_selected_metrics_key(key: object) -> bool:
    if isinstance(key, str):
        return False
    try:
        labels = [str(x) for x in list(key)]
    except TypeError:
        return False
    if labels in _KNOWN_METRIC_LISTS:
        return True
    # Any IC + excess_return* bundle used by templates / UI
    if labels and labels[0] == "IC" and any("excess_return" in x for x in labels[1:]):
        return True
    return False


def _to_metric_series(sms: object) -> pd.Series:
    if sms is None:
        return pd.Series(dtype="float64")
    if isinstance(sms, pd.Series):
        return pd.to_numeric(sms, errors="coerce")
    if isinstance(sms, pd.DataFrame):
        wanted = QLIB_SELECTED_METRICS
        selected_in_cols = any(k in sms.columns for k in wanted)
        selected_in_idx = any(k in sms.index for k in wanted)
        if selected_in_cols and not selected_in_idx:
            return pd.to_numeric(sms.iloc[0], errors="coerce")
        if sms.shape[1] >= 1:
            return pd.to_numeric(sms.iloc[:, 0], errors="coerce")
        return pd.Series(dtype="float64")
    if isinstance(sms, dict):
        return pd.to_numeric(pd.Series(sms), errors="coerce")
    try:
        return pd.to_numeric(pd.Series(sms), errors="coerce")
    except Exception:
        return pd.Series(dtype="float64")


def select_qlib_metrics(sms: object, labels: list[str] | None = None) -> pd.Series:
    series = _to_metric_series(sms)
    wanted = labels or QLIB_SELECTED_METRICS
    selected = series.reindex(wanted)
    if getattr(sms, "name", None) is not None:
        selected.name = sms.name
    return selected


def _should_intercept_loc(obj: object, labels: list[str]) -> bool:
    """Rewrite metric-list lookups; keep Current/SOTA frames as DataFrames."""
    if isinstance(obj, pd.Series):
        return True
    if isinstance(obj, pd.DataFrame):
        cols = {str(c) for c in obj.columns}
        if cols & {"Current Result", "SOTA Result"}:
            return True  # reindex rows → still a DataFrame (see select below)
        if obj.shape[1] == 1:
            return True
        if any(k in obj.columns for k in labels) or any(k in obj.index for k in labels):
            return True
        if "IC" in labels and ("IC" in obj.index or "IC" in obj.columns):
            return True
        return False
    return False


def _select_metric_frame(obj: pd.DataFrame, labels: list[str]) -> pd.DataFrame | pd.Series:
    cols = {str(c) for c in obj.columns}
    if cols & {"Current Result", "SOTA Result"}:
        # Keep 2-col comparison shape for .iterrows(); missing metrics → NaN rows
        return obj.reindex(labels)
    return select_qlib_metrics(obj, labels)


def apply() -> None:
    try:
        from pandas.core.indexing import _LocIndexer
    except Exception:
        return
    if getattr(_LocIndexer, "_qlib_ui_metrics_patched", False):
        return

    orig = _LocIndexer.__getitem__

    def __getitem__(self: object, key: Hashable) -> object:
        if _is_selected_metrics_key(key):
            labels = [str(x) for x in list(key)]
            obj = getattr(self, "obj", None)
            if _should_intercept_loc(obj, labels):
                if isinstance(obj, pd.DataFrame):
                    return _select_metric_frame(obj, labels)
                return select_qlib_metrics(obj, labels)
        return orig(self, key)

    _LocIndexer.__getitem__ = __getitem__  # type: ignore[method-assign]
    _LocIndexer._qlib_ui_metrics_patched = True
    print("[ui-metrics-patch] loc[QLIB_SELECTED_METRICS] is defensive", file=sys.stderr, flush=True)
