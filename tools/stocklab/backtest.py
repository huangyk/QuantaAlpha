"""评估层：绩效指标 + 汇总。指标复用 tools.index_lab.metrics（freq='w' 周频年化=×52）。"""
import numpy as np
import pandas as pd

from tools.index_lab.metrics import metrics   # 复用既有实现（已支持 freq='d'/'w'/'m'）


def weekly_metrics(ret_series, freq="w"):
    """周频收益序列 -> 年化/波动/Sharpe/回撤/终值。"""
    s = pd.Series(ret_series).replace([np.inf, -np.inf], np.nan).dropna()
    return metrics(s.values, freq=freq)


def summarize(name, ret_series):
    m = weekly_metrics(ret_series)
    return dict(name=name, ann=m["ann"], sharpe=m["sharpe"], mdd=m["mdd"],
                vol=m["vol"], nav=m["nav"])


def yearly_table(ret_series):
    """逐年收益（自然年），用于看稳定性。"""
    s = pd.Series(ret_series).dropna()
    yr = s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1)
    return yr
