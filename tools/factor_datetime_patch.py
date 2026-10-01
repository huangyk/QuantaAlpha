"""Coerce factor datetime index to timestamps before RD-Agent's time_diff check.

Also normalize column labels to plain Python ``str`` so parquet round-trips work.
numpy ``np.str_`` / ``np.bytes_`` in MultiIndex names become
``('feature', np.str_('x'))`` in pandas metadata and break ``ast.literal_eval`` on read.
"""
from __future__ import annotations

import sys

import pandas as pd

from rdagent.scenarios.qlib.developer import utils as factor_utils


def _plain_label(value: object) -> object:
    """Convert numpy scalar strings to builtin str; leave ints/floats alone."""
    if isinstance(value, tuple):
        return tuple(_plain_label(v) for v in value)
    # np.str_ / np.bytes_ / pandas StringDtype scalars
    if hasattr(value, "item") and type(value).__name__ in {"str_", "bytes_", "str0"}:
        try:
            return value.item()
        except Exception:
            return str(value)
    if type(value).__module__.startswith("numpy") and isinstance(value, (str, bytes)):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        levels = [[_plain_label(v) for v in level] for level in out.columns.levels]
        # rebuild from tuples to drop numpy scalar types in codes
        out.columns = pd.MultiIndex.from_tuples(
            [tuple(_plain_label(x) for x in col) for col in out.columns],
            names=[_plain_label(n) if n is not None else None for n in out.columns.names],
        )
        _ = levels  # kept for clarity; tuples path is authoritative
    else:
        out.columns = [_plain_label(c) for c in out.columns]
    return out


def _as_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or "datetime" not in getattr(df.index, "names", []):
        return df
    names = list(df.index.names)
    reset = df.reset_index()
    reset["datetime"] = pd.to_datetime(reset["datetime"], errors="coerce")
    out = reset.set_index(names)
    return out.sort_index()


_orig = factor_utils.process_factor_data


def process_factor_data(exp_or_list):  # type: ignore[no-untyped-def]
    if isinstance(exp_or_list, list):
        pass
    # Wrap original by coercing execute() dataframes via monkeypatch of the inner loop
    from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
    from rdagent.core.conf import RD_AGENT_SETTINGS
    from rdagent.core.exception import FactorEmptyError
    from rdagent.core.utils import multiprocessing_wrapper
    from rdagent.log import rdagent_logger as logger
    from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

    if isinstance(exp_or_list, QlibFactorExperiment):
        exp_or_list = [exp_or_list]
    factor_dfs = []
    error_message = ""

    for exp in exp_or_list:
        if not isinstance(exp, QlibFactorExperiment):
            continue
        if len(exp.sub_tasks) == 0:
            continue
        assert isinstance(exp.prop_dev_feedback, CoSTEERMultiFeedback)
        message_and_df_list = multiprocessing_wrapper(
            [
                (implementation.execute, ("All",))
                for implementation, fb in zip(exp.sub_workspace_list, exp.prop_dev_feedback)
                if implementation and fb
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )
        for message, df in message_and_df_list:
            df = _normalize_columns(_as_datetime_index(df))
            if df is not None and "datetime" in df.index.names:
                dt = pd.to_datetime(df.index.get_level_values("datetime"), errors="coerce")
                time_diff = dt.to_series().diff().dropna().unique()
                if pd.Timedelta(minutes=1) not in time_diff:
                    factor_dfs.append(df)
                    logger.info(
                        f"Factor data from {exp.hypothesis.concise_justification} is successfully generated."
                    )
                else:
                    logger.warning(f"Factor data from {exp.hypothesis.concise_justification} is not generated.")
            else:
                error_message += (
                    f"Factor data from {exp.hypothesis.concise_justification} is not generated because of {message}"
                )
                logger.warning(
                    f"Factor data from {exp.hypothesis.concise_justification} is not generated because of {message}"
                )

    if factor_dfs:
        return _normalize_columns(pd.concat(factor_dfs, axis=1))
    raise FactorEmptyError(
        f"No valid factor data found to merge (in process_factor_data) because of {error_message}."
    )


def apply() -> None:
    if getattr(factor_utils, "_datetime_patch", False):
        return
    factor_utils.process_factor_data = process_factor_data
    try:
        from rdagent.scenarios.qlib.developer import factor_runner

        factor_runner.process_factor_data = process_factor_data
    except Exception:
        pass
    factor_utils._datetime_patch = True
    print("[factor-datetime-patch] applied", file=sys.stderr, flush=True)
