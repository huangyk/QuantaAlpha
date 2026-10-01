"""Prevent Windows fatal crash in qlib SigAna when predictions are constant.

NumPy 2 + pandas Series.corr on (near) zero-variance columns can raise
Windows fatal exception 0xc06d007f during calc_ic. Never call pandas.corr.
"""
from __future__ import annotations

import sys

import math

import numpy as np
import pandas as pd


def _safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    x = pd.to_numeric(a, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    y = pd.to_numeric(b, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    if method == "spearman":
        # rank data; ties get average rank via pandas then convert
        x = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    sx = float(np.sqrt(np.dot(x, x)))
    sy = float(np.sqrt(np.dot(y, y)))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    val = float(np.dot(x, y) / (sx * sy))
    if not math.isfinite(val):
        return float("nan")
    return val


def apply() -> None:
    try:
        from qlib.contrib.eva import alpha as alpha_mod
    except Exception:
        return
    if getattr(alpha_mod, "_safe_ic_patched", False):
        return

    def calc_ic(
        pred: pd.Series,
        label: pd.Series,
        date_col: str = "datetime",
        dropna: bool = False,
    ) -> tuple[pd.Series, pd.Series]:
        df = pd.DataFrame({"pred": pred, "label": label})
        ic = df.groupby(date_col, group_keys=False).apply(
            lambda g: _safe_corr(g["pred"], g["label"], method="pearson")
        )
        ric = df.groupby(date_col, group_keys=False).apply(
            lambda g: _safe_corr(g["pred"], g["label"], method="spearman")
        )
        if dropna:
            return ic.dropna(), ric.dropna()
        return ic, ric

    alpha_mod.calc_ic = calc_ic  # type: ignore[assignment]
    alpha_mod._safe_ic_patched = True
    print("[safe-ic-patch] calc_ic zero-variance guard", file=sys.stderr, flush=True)
