"""方向4：指数动量 / 趋势跟踪（与抄跌反方向，学术上最该有 edge）。

方法（已修前视）：
  * MA 择时(仅多)：用"前一日收盘 > 前一日 MA(W)"判定 -> 次日均(open)进场。
  * TSMOM 多空：仓位 = sign(收_{t-1}/收_{t-1-L}-1) -> 次日均(open)进场。
切换成本在日收益中扣除。指标与买入持有对比。
"""
import numpy as np
import pandas as pd

from .config import INDEX_COST
from .metrics import metrics


def ma_timing_rets(df, W=120, cost=INDEX_COST):
    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    ma = pd.Series(c).rolling(W).mean().values
    pos = np.zeros(len(c))
    for t in range(1, len(c)):
        # 前一日收盘 vs 前一日 MA 判定，次日均(open)进场 —— 无前视
        if np.isfinite(ma[t - 1]) and c[t - 1] > ma[t - 1]:
            pos[t] = 1.0
    rets = []
    for t in range(1, len(c)):
        r = pos[t] * (c[t] / o[t] - 1)
        if t >= 2 and pos[t] != pos[t - 1]:
            r -= cost
        rets.append(r)
    return np.array(rets)


def tsmom_rets(df, L=250, cost=INDEX_COST):
    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    pos = np.zeros(len(c))
    for t in range(L + 1, len(c)):
        sig = c[t - 1] / c[t - 1 - L] - 1
        pos[t] = 1.0 if sig > 0 else (-1.0 if sig < 0 else 0.0)
    rets = []
    for t in range(1, len(c)):
        r = pos[t] * (c[t] / o[t] - 1)
        if t >= 2 and pos[t] != pos[t - 1]:
            r -= cost
        rets.append(r)
    return np.array(rets)


def buyhold_rets(df):
    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    return np.array([c[t] / o[t] - 1 for t in range(1, len(c))])


def run(df, name="", W=120, L=250):
    """单指数便捷入口（CLI 用），返回 dict(买入持有 / MA择时 / TSMOM 指标)。"""
    bh = metrics(buyhold_rets(df))
    ma = metrics(ma_timing_rets(df, W))
    ts = metrics(tsmom_rets(df, L))
    return dict(module="momentum", index=name,
                buyhold=dict(ann=bh["ann"], sharpe=bh["sharpe"], mdd=bh["mdd"]),
                ma=dict(W=W, ann=ma["ann"], sharpe=ma["sharpe"], mdd=ma["mdd"]),
                tsmom=dict(L=L, ann=ts["ann"], sharpe=ts["sharpe"], mdd=ts["mdd"]))


def run_pool(dfs: dict, Ws=(60, 120, 200), Ls=(120, 250)):
    """跨多指数平均，返回 {key: 平均指标}。每个 W/L 独立计算后取均值。"""
    out = {}
    out["buyhold"] = _avg_dicts([metrics(buyhold_rets(df)) for df in dfs.values()])
    for W in Ws:
        out[f"ma{W}"] = _avg_dicts([metrics(ma_timing_rets(df, W)) for df in dfs.values()])
    for L in Ls:
        out[f"tsmom{L}"] = _avg_dicts([metrics(tsmom_rets(df, L)) for df in dfs.values()])
    out["_per_index"] = {nm: run(df, nm) for nm, df in dfs.items()}
    return out


def _avg_dicts(dicts):
    return dict(ann=float(np.mean([d["ann"] for d in dicts])),
                sharpe=float(np.mean([d["sharpe"] for d in dicts])),
                mdd=float(np.mean([d["mdd"] for d in dicts])))
