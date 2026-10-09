"""方向2：指数支撑带抢反弹"赚几点就走"回测。

方法：价格触达 关卡±band -> 次日均(open)买入 -> 路径退出(+tp 止盈 / -sl 止损 /
持满 maxhold 日)。成本默认万3。支持单关回测与多指数×多关卡网格 pooling。
"""
import numpy as np
import pandas as pd

from .config import INDEX_COST
from .metrics import metrics, t_stat, win_rate


def bounce_trades(df, level, band=0.02, tp=0.05, sl=0.03, maxhold=15, cost=INDEX_COST):
    """返回 [(净收益, 退出类型, 进场日), ...]。含边界：入场价非有限/<=0 跳过。"""
    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    dates = df.index
    n = len(df); out = []
    i = 0
    while i < n - 1:
        if level * (1 - band) <= c[i] <= level * (1 + band):
            j = i + 1
            if j >= n:
                break
            entry = o[j]
            if not np.isfinite(entry) or entry <= 0:
                i += 1; continue
            tgt = entry * (1 + tp); stp = entry * (1 - sl)
            done = False
            for k in range(j, min(j + maxhold, n)):
                if l[k] <= stp:
                    out.append((stp / entry - 1 - cost, "stop", dates[j])); done = True; i = k + 1; break
                if h[k] >= tgt:
                    out.append((tgt / entry - 1 - cost, "tp", dates[j])); done = True; i = k + 1; break
            if not done:
                ex = c[min(j + maxhold - 1, n - 1)]
                out.append((ex / entry - 1 - cost, "hold", dates[j])); i = j + maxhold
        else:
            i += 1
    return out


def run_single(df, level, band=0.02, tp=0.05, sl=0.03, maxhold=15, cost=INDEX_COST):
    tr = bounce_trades(df, level, band, tp, sl, maxhold, cost)
    if not tr:
        return dict(n=0, net_bp=0.0, t=0.0, sharpe=0.0, wr=0.0, ann=None)
    g = np.array([x[0] for x in tr])
    m = metrics(g)
    return dict(n=len(g), net_bp=float(g.mean() * 1e4), t=float(t_stat(g)),
                sharpe=m["sharpe"], wr=float(win_rate(g)), mdd=m["mdd"])


def run_grid(dfs: dict, levels_map: dict, tps, sls, bands, maxhold=15, cost=INDEX_COST):
    """多指数×多关卡 pooling 网格。

    dfs: {name: df}; levels_map: {name: [level,...]}。
    返回按 (tp,sl,band) 聚合的列表，每项含 N/均净bp/t/强关占比。
    """
    rows = []
    for tp in tps:
        for sl in sls:
            for band in bands:
                allg = []; strong = 0; total_lv = 0
                for nm, df in dfs.items():
                    for lv in levels_map.get(nm, []):
                        tr = bounce_trades(df, lv, band, tp, sl, maxhold, cost)
                        if len(tr) >= 3:
                            total_lv += 1
                            g = np.array([x[0] for x in tr])
                            if g.mean() > 0:
                                strong += 1
                            allg.extend(g)
                allg = np.array(allg)
                N = len(allg)
                if N < 30:
                    continue
                net = (allg - cost) * 1e4
                mean_net = float(net.mean()); std = float(net.std())
                t = mean_net / (std / np.sqrt(N)) if std > 0 else 0.0
                sharpe = mean_net / std * np.sqrt(252 / maxhold) if std > 0 else 0.0
                sel = strong / total_lv if total_lv else 0.0
                rows.append(dict(tp=tp, sl=sl, band=band, N=N, net_bp=mean_net,
                                 t=t, sharpe=sharpe, strong_ratio=sel))
    if rows:
        rows.sort(key=lambda r: -r["t"])
    return rows


def run(df, name="", level=None, band=0.02, tp=0.05, sl=0.03, maxhold=15):
    """单指数单关便捷入口（CLI 用）。"""
    r = run_single(df, level, band, tp, sl, maxhold)
    r["module"] = "bounce"; r["index"] = name; r["level"] = level
    r["tp"] = tp; r["sl"] = sl
    return r
