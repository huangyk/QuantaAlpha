"""信号层：特征/状态 -> 目标仓位 -> 组合周收益。持仓口径与 ETF 底盘一致。

信号：个股站上 N 周均线且时点可交易 -> 持有，组内按逆波动率分配；否则现金。
关键防前视：第 t 周定权重，收益用 **t+1** 周（weights.shift(1) * rets）。
"""
import numpy as np
import pandas as pd

from . import features as F


def equal_weight_weekly(close_wk, eligible):
    """同池等权基线：每周在时点可交易池中等权持有（毛收益，未计成本，作为上界参照）。"""
    rets = F.weekly_returns(close_wk)
    m = eligible & rets.notna()
    n = m.sum(axis=1)
    n = n.where(n > 0)
    port = rets.where(m).sum(axis=1) / n
    return port.fillna(0.0)


def trend_portfolio_weekly(close_wk, eligible, ma_weeks=40, vol_weeks=12,
                           cost_bps=1.0, floor=0.05):
    """股票周频趋势跟随组合周收益序列（含换手成本）。

    返回 dict(nav/ret/turnover/time_in_market/n_held_avg)。
    """
    rets = F.weekly_returns(close_wk)
    sig = F.trend_signal(close_wk, ma_weeks) & eligible & rets.notna()
    inv = F.inverse_vol(rets, vol_weeks, floor)
    inv = inv.where(sig, 0.0)
    denom = inv.sum(axis=1)
    w = inv.div(denom.replace(0, np.nan), axis=0).fillna(0.0)   # 行和为1（全现金时为0）
    wlag = w.shift(1).fillna(0.0)
    gross = (wlag * rets.fillna(0.0)).sum(axis=1)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost
    nav = (1 + net).cumprod()
    n_held = sig.sum(axis=1)
    return dict(ret=net, gross=gross, nav=nav, turnover=turnover,
                time_in_market=(denom > 0).astype(float), n_held=n_held,
                weights=w)
