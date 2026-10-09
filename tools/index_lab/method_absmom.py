"""好方法：跨资产绝对动量 / 趋势跟随（文献稳健的实战策略）。

为什么是它（而非功能B的单指数MA门控）：
  功能B 用"中证500 MA60"给整个跨资产ETF池当总开关——单一外部指数与
  债券/黄金/商品走势脱钩，反而把池子频繁甩进现金、吃掉收益。
  文献稳健做法（Antonacci GEM / Faber 200日SMA战术配置 / Moskowitz TSMOM）
  是**对每个资产独立判趋势**：该ETF自身 12个月动量>0(或价格在200日线上)
  才持有，否则该份额转现金。这套在多元资产池(含商品/黄金/REITs/债券)上
  被反复验证能降回撤、提夏普（Faber 2007 测 83 个指数一致有效）。

信号-收益对齐：月末收盘算动量 -> 次月生效(月频调仓)，无前视。
默认 12-1 动量(剔除最近1月短期反转)，等权持有通过筛选的ETF。
"""
import numpy as np
import pandas as pd
import psycopg2

from .config import DB, ETF_COMM, GATE_MA, CLEAN_SOURCE
from .io_db import DataError, load_etf_universe
from .metrics import metrics
from .etf_guard import etf_equal_weight


def _load_etf(cur, codes, start, end):
    rows = []
    for code in codes:
        cur.execute("""SELECT trade_date, open, high, low, close, volume
                       FROM bars_1d WHERE asset_code=%s AND asset_type='etf'
                       AND trade_date>=%s AND trade_date<=%s ORDER BY trade_date""",
                    (code, start, end))
        for r in cur.fetchall():
            rows.append((code, r[0], r[1], r[2], r[3], r[4], r[5]))
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for code, g in df.groupby("code"):
        out[code] = g.sort_values("date").set_index("date")
    return out


def abs_momentum_etf(codes, start, end, signal="absmom", lookback=252, skip=21,
                     weight="equal", cost=ETF_COMM, sma_w=200):
    """跨资产绝对动量 / 趋势跟随回测。

    signal: 'absmom' 12-1动量>0 才持有；'sma200' 收盘价>200日SMA 才持有。
    weight: 'equal' 等权 / 'inv_vol' 逆波动率。
    返回 dict(nav, ann, vol, sharpe, mdd, time_in_market, n_rebal)。
    """
    con = psycopg2.connect(**DB); cur = con.cursor()
    try:
        data = _load_etf(cur, codes, start, end)
    finally:
        con.close()

    all_days = sorted(set().union(*[set(g.index) for g in data.values()]))
    warmup = (lookback + skip) if signal == "absmom" else sma_w
    if len(all_days) < warmup + 30:
        raise DataError(f"数据不足({len(all_days)} < {warmup + 30})")

    # 预计算每ETF每日信号(用收盘价)
    sig = {}
    for code, g in data.items():
        c = g["close"].astype(float)
        if signal == "absmom":
            s = (c.shift(skip) / c.shift(skip + lookback) - 1)   # 12-1 动量
        else:  # sma200
            s = (c > c.rolling(sma_w).mean()).astype(float)
        sig[code] = s.values

    cash = 1.0; nav = []; nav_dates = []; inmkt = []
    prev_w = {}; last_month = None; n_rebal = 0
    L = len(all_days)
    for i, d in enumerate(all_days):
        if i > 0:
            prev_d = all_days[i - 1]
            do_rb = (i > warmup) and (last_month is None
                                       or d.month != last_month.month
                                       or d.year != last_month.year)
            if do_rb:
                si = i - 1
                held = [code for code in codes
                        if si < len(sig[code]) and np.isfinite(sig[code][si])
                        and sig[code][si] > 0]
                if held:
                    if weight == "inv_vol":
                        w = {}
                        for code in held:
                            ret = data[code]["close"].astype(float).pct_change()
                            vol = ret.iloc[max(0, si - 59):si + 1].std() * np.sqrt(252)
                            w[code] = (1.0 / max(vol, 0.05)) if np.isfinite(vol) and vol > 0 else 0.0
                        tot = sum(w.values())
                        target = {c: w[c] / tot for c in held} if tot > 0 else {}
                    else:
                        target = {c: 1.0 / len(held) for c in held}
                else:
                    target = {}
                turnover = sum(abs(target.get(c, 0.0) - prev_w.get(c, 0.0))
                                for c in set(list(target) + list(prev_w)))
                cash *= (1 - turnover * cost)
                prev_w = target
                last_month = d
                n_rebal += 1
            # 当日收益(用昨日权重, T+1 对齐)
            day_ret = 0.0
            if prev_w:
                tot_w = 0.0
                for code, w in prev_w.items():
                    if w <= 0:
                        continue
                    g = data[code]
                    ct = g["close"].get(d); cp = g["close"].get(prev_d)
                    if ct is not None and cp is not None and pd.notna(ct) and pd.notna(cp) and cp > 0:
                        day_ret += w * (ct / cp - 1); tot_w += w
                if tot_w > 0:
                    day_ret /= tot_w
            cash *= (1 + day_ret)
        nav.append(cash); nav_dates.append(d)
        inmkt.append(1.0 if prev_w else 0.0)

    nav_s = pd.Series(nav, index=nav_dates)
    m = metrics(nav_s.pct_change().dropna())
    m["nav"] = nav_s
    m["time_in_market"] = float(np.mean(inmkt))
    m["n_rebal"] = n_rebal
    return m


def run(start="2019-01-01", end="2026-09-30", clean_only=True, signals=("absmom", "sma200")):
    """对比：纯等权 vs 绝对动量(12-1) vs 200日SMA趋势跟随。返回 dict。"""
    codes = load_etf_universe(start, end, clean_only)
    if not codes:
        raise DataError("ETF 池为空")
    out = dict(module="absmom", n_etf=len(codes), start=start, end=end, configs={})
    base = etf_equal_weight(codes, start, end, gate_code=None)
    out["equal_weight"] = dict(ann=base["ann"], sharpe=base["sharpe"],
                               mdd=base["mdd"], time_in_market=1.0, n_rebal=0)
    for sig in signals:
        for w in ("equal", "inv_vol"):
            key = f"{sig}/{w}"
            r = abs_momentum_etf(codes, start, end, signal=sig, weight=w)
            out["configs"][key] = dict(ann=r["ann"], sharpe=r["sharpe"], mdd=r["mdd"],
                                        time_in_market=r["time_in_market"], n_rebal=r["n_rebal"])
    return out
