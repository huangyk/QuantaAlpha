"""Drop-in replacement for RD-Agent workspace read_exp_res.py.

Picks the best qlib recorder:
1) Prefer recorders with portfolio / IC metrics
2) Prefer end_time finished
3) Fall back to latest start_time
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import qlib

qlib.init()

from qlib.workflow import R


def _score_recorder(recorder) -> tuple:
    info = recorder.info
    metrics = {}
    try:
        metrics = recorder.list_metrics() or {}
    except Exception:
        metrics = {}
    try:
        arts = [str(a) for a in (recorder.list_artifacts() or [])]
    except Exception:
        arts = []

    has_port = any("portfolio_analysis" in a or "report_normal" in a for a in arts)
    has_ic = any(k in metrics for k in ("IC", "Rank IC"))
    has_excess = any("excess_return" in str(k) for k in metrics)
    metric_n = len(metrics)
    end_time = info.get("end_time")
    start_time = info.get("start_time") or ""

    # Higher tuple sorts later as "better"
    return (
        1 if has_port else 0,
        1 if has_excess else 0,
        1 if has_ic else 0,
        metric_n,
        1 if end_time is not None else 0,
        str(end_time or ""),
        str(start_time),
    )


def _pick_latest_recorder():
    best = None
    best_score = None
    for experiment in R.list_experiments():
        for recorder_id in R.list_recorders(experiment_name=experiment):
            if recorder_id is None:
                continue
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
            score = _score_recorder(recorder)
            if best is None or score > best_score:
                best = recorder
                best_score = score
    return best


def main() -> int:
    latest_recorder = _pick_latest_recorder()
    if latest_recorder is None:
        print("No recorders found")
        return 1

    print(f"Latest recorder: {latest_recorder}")
    metrics = pd.Series(latest_recorder.list_metrics())
    output_path = Path.cwd() / "qlib_res.csv"
    metrics.to_csv(output_path)
    print(f"Output has been saved to {output_path}")

    try:
        ret_data_frame = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        ret_data_frame.to_pickle("ret.pkl")
    except Exception as exc:
        print(f"Warning: portfolio report not ready yet ({exc!r}); writing stub ret.pkl")
        pd.DataFrame({"return": []}).to_pickle("ret.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
