"""统一绩效指标：年化 / 波动 / 夏普 / 最大回撤 / 终值。"""
import numpy as np

from .config import TRADING_DAYS


def metrics(rets, switch_cost: float = 0.0):
    """rets: 日收益序列(已含切换成本)。返回 dict。

    对空/非有限序列做防御，避免上游因个别 NaN 崩。
    """
    rets = np.asarray(rets, dtype=float)
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return dict(ann=0.0, vol=0.0, sharpe=0.0, mdd=0.0, nav=1.0, n=0)
    nav = np.cumprod(1 + rets)
    n = len(rets)
    ann = nav[-1] ** (TRADING_DAYS / n) - 1 if nav[-1] > 0 else -1.0
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann / vol if vol > 0 else 0.0
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    return dict(ann=float(ann), vol=float(vol), sharpe=float(sharpe),
                mdd=mdd, nav=float(nav[-1]), n=n)


def t_stat(x):
    """单样本 t 统计量（均值 / 标准误），空序列返回 0。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std() == 0:
        return 0.0
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def win_rate(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float((x > 0).mean()) if len(x) else 0.0
