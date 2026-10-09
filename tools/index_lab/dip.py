"""方向3：顺趋势抄跌(干净前向收益)。

方法（杜绝前视）：价格自 20 日高回撤 >= depth% 且为"首次触达"(rising-edge + 冷却)
-> 次日均(open)进场 -> 持 H 日后收盘退出。分三类条件：
  A 任意抄跌(无趋势过滤) / B 上升趋势中抄跌(收>MA60) / C 下降趋势中抄跌(收<MA60)。
"""
import numpy as np
import pandas as pd

from .config import INDEX_COST
from .metrics import t_stat, win_rate


def dip_signals(df, depth=0.05, ma_w=60):
    """返回 (sig_any, sig_up, sig_dn) 三个布尔信号数组。"""
    c = df["close"].values.astype(float); n = len(c)
    rollmax = pd.Series(c).rolling(20).max().values
    ma = pd.Series(c).rolling(ma_w).mean().values
    cond_any = c <= (1 - depth) * rollmax
    cond_up = cond_any & (c > ma)
    cond_dn = cond_any & (c < ma)

    def triggers(mask):
        sig = np.zeros(n, bool); cool = False
        for i in range(n):
            if cool:
                if not mask[i]:
                    cool = False
                continue
            if mask[i]:
                sig[i] = True; cool = True
        return sig

    return triggers(cond_any), triggers(cond_up), triggers(cond_dn)


def fwd_returns(df, sig, H=20, cost=INDEX_COST):
    """次日均进场、持 H 日真实收盘退出。返回 [(净收益, 年份), ...]。"""
    o = df["open"].values.astype(float)
    c = df["close"].values.astype(float)
    dates = df.index
    n = len(df); out = []
    for i in range(n):
        if not sig[i] or i + 1 + H >= n:
            continue
        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        ex = c[i + 1 + H]
        if not np.isfinite(ex):
            continue
        out.append((ex / entry - 1 - cost, dates[i + 1].year))
    return out


_COND = {"any": 0, "up": 1, "dn": 2}


def run(df, name="", depth=0.05, H=20, cond="any"):
    """单指数单条件便捷入口（CLI 用）。cond ∈ {any,up,dn}。"""
    sig = dip_signals(df, depth)[_COND[cond]]
    g = fwd_returns(df, sig, H)
    if not g:
        return dict(module="dip", index=name, depth=depth, H=H, cond=cond,
                    n=0, mean_bp=0.0, t=0.0, wr=0.0, by_year={})
    rets = np.array([x[0] for x in g]) * 1e4
    yrs = pd.Series([x[0] for x in g], index=[x[1] for x in g])
    by_year = {int(y): float(v) for y, v in
               yrs.groupby(level=0).mean().mul(1e4).items()}
    return dict(module="dip", index=name, depth=depth, H=H, cond=cond,
                n=len(rets), mean_bp=float(rets.mean()), t=float(t_stat(rets)),
                wr=float(win_rate(rets)), by_year=by_year)


def run_pool(dfs: dict, depths, horizons, conds=("any", "up", "dn")):
    """跨多指数 pooling，返回全部 (cond, depth, H, N, mean_bp, t, wr) 行。"""
    rows = []
    for depth in depths:
        for H in horizons:
            for cond in conds:
                allg = []; ally = []
                for nm, df in dfs.items():
                    sig = dip_signals(df, depth)[_COND[cond]]
                    for g, y in fwd_returns(df, sig, H):
                        allg.append(g); ally.append(y)
                g = np.array(allg) * 1e4
                N = len(g)
                if N < 30:
                    continue
                rows.append((cond, depth * 100, H, N, float(g.mean()),
                             float(t_stat(g)), float(win_rate(g))))
    return rows
