"""方向1：圆整数关口支撑检验。

方法：历史价格触达 关卡±band 后，未来 fwd 日收益分布(均值/中位/胜率)，
据此判"真支撑 / 非支撑(易破) / 中性"。另提供通用整数关口扫描。
"""
import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS  # noqa  (占位，保持导入一致)


def support_test(close: pd.Series, level: float, band: float = 0.015, fwd: int = 20):
    """对单个关卡检验支撑强度。返回 dict(n, mean, med, wr, verdict)。"""
    s = close.dropna()
    if len(s) < fwd + 5:
        return dict(n=0, mean=np.nan, med=np.nan, wr=np.nan, verdict="数据不足")
    touch = s[(s >= level * (1 - band)) & (s <= level * (1 + band))]
    n = len(touch)
    if n < 3:
        return dict(n=n, mean=np.nan, med=np.nan, wr=np.nan, verdict="触达不足")
    rets = []
    for d in touch.index:
        i = s.index.get_loc(d)
        if i + fwd < len(s):
            rets.append(s.iloc[i + fwd] / s.iloc[i] - 1)
    rets = np.array(rets)
    mean = float(rets.mean()); med = float(np.median(rets)); wr = float((rets > 0).mean())
    if wr >= 0.55 and mean > 0:
        verdict = "真支撑"
    elif wr < 0.5:
        verdict = "非支撑(易破)"
    else:
        verdict = "中性"
    return dict(n=n, mean=mean, med=med, wr=wr, verdict=verdict)


def scan_levels(close: pd.Series, band: float = 0.015, fwd: int = 20, step: int = None):
    """通用整数关口扫描，返回 [(level, n, mean, wr, verdict), ...]。"""
    s = close.dropna()
    latest = s.iloc[-1]
    step = step or (100 if latest > 2000 else 50)
    lo, hi = s.iloc[-250:].min(), s.iloc[-250:].max()
    res = []
    for lv in range(int(lo // step) * step, int(hi // step) * step + 1, step):
        r = support_test(s, lv, band, fwd)
        if r["n"] >= 3 and np.isfinite(r["mean"]):
            res.append((lv, r["n"], r["mean"], r["wr"], r["verdict"]))
    return res


def run(df: pd.DataFrame, name: str = "", levels=None, band: float = 0.015, fwd: int = 20):
    """单指数支撑检验。levels: 指定关卡列表(可空=仅扫描)。返回 dict。"""
    s = df["close"]
    latest = float(s.iloc[-1]); d0 = s.index.max().date()
    out = dict(module="support", index=name, date=str(d0), latest=latest,
               levels=[], scan=[], summary="")
    for lv in (levels or []):
        r = support_test(s, lv, band, fwd)
        pos = "上方" if latest > lv else ("下方" if latest < lv else "持平")
        out["levels"].append(dict(level=lv, position=pos,
                                  dist=(latest / lv - 1) * 100, **r))
    out["scan"] = scan_levels(s, band, fwd)
    good = [x for x in out["scan"] if x[4] == "真支撑"]
    bad = [x for x in out["scan"] if x[4] == "非支撑(易破)"]
    out["summary"] = (f"真支撑关口 {len(good)} 个 / 易破 {len(bad)} 个 / "
                      f"全部{len(out['scan'])}关均值收益平均 "
                      f"{np.mean([x[2] for x in out['scan']])*100:+.1f}%")
    return out
