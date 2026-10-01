"""Make pandas parquet reads tolerate bad MultiIndex metadata.

RD-Agent writes ``combined_factors_df.parquet`` with MultiIndex columns.
If a level contains ``np.str_``, pandas metadata stores
``('feature', np.str_('name'))`` and ``ast.literal_eval`` fails on read.
"""
from __future__ import annotations

import ast
import re
import sys
from typing import Any

import pandas as pd


def _fix_col_name(raw: str) -> Any:
    text = str(raw)
    # ('feature', np.str_('foo')) or ("feature", "foo")
    m = re.match(
        r"^\(\s*['\"]([^'\"]+)['\"]\s*,\s*(?:np\.str_\()?['\"]([^'\"]+)['\"]\)?\s*\)$",
        text,
    )
    if m:
        return (m.group(1), m.group(2))
    try:
        val = ast.literal_eval(text)
        if isinstance(val, tuple):
            return tuple(str(x) for x in val)
        return val
    except Exception:
        return text


def _read_parquet_resilient(*args: Any, **kwargs: Any) -> pd.DataFrame:
    try:
        return _orig_read_parquet(*args, **kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "malformed node" not in msg and "literal_eval" not in msg:
            raise
        import pyarrow.parquet as pq

        path = args[0] if args else kwargs.get("path") or kwargs.get("path_or_buf")
        table = pq.read_table(path)
        df = table.to_pandas(ignore_metadata=True)
        # Restore index if present as columns
        index_cols = []
        for name in ("datetime", "instrument"):
            if name in df.columns:
                index_cols.append(name)
        # Fix feature column names
        rename = {}
        new_tuples = []
        feature_cols = []
        for col in df.columns:
            if col in index_cols:
                continue
            fixed = _fix_col_name(col)
            rename[col] = fixed
            if isinstance(fixed, tuple):
                new_tuples.append(fixed)
                feature_cols.append(col)
            else:
                feature_cols.append(col)
        df = df.rename(columns=rename)
        if index_cols:
            df = df.set_index(index_cols)
        if new_tuples and all(isinstance(c, tuple) for c in df.columns):
            df.columns = pd.MultiIndex.from_tuples(list(df.columns))
        print(f"[parquet-read-patch] recovered {path}", file=sys.stderr, flush=True)
        return df


_orig_read_parquet = pd.read_parquet


def apply() -> None:
    if getattr(pd, "_parquet_read_patched", False):
        return
    pd.read_parquet = _read_parquet_resilient  # type: ignore[assignment]
    pd._parquet_read_patched = True  # type: ignore[attr-defined]
    print("[parquet-read-patch] applied", file=sys.stderr, flush=True)
