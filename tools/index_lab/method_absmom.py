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
    if not data:
        raise DataError("该区间内无任何ETF数据")
    codes = list(data.keys())   # 只用实际有数据的标的，避免子窗口子集 KeyError

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


def current_signal(codes, as_of=None, signal="sma200", sma_w=200,
                   weight="inv_vol", lookback=252, skip=21):
    """截至 as_of(默认最新交易日) 各ETF趋势状态与建议持仓权重。

    返回 dict(as_of, signal, weight, n_hold, n_total, holdings)。
    holdings 每项: code, close, sma(仅sma200), mom(仅absmom),
                   gap_pct(偏离%), vol_60(60日年化波动), hold, weight。
    hold=True 表示趋势向上、应持有；权重按 equal 或 inv_vol 在"持有子集"内归一。
    """
    con = psycopg2.connect(**DB); cur = con.cursor()
    try:
        if as_of is None:
            cur.execute("SELECT MAX(trade_date) FROM bars_1d WHERE asset_type='etf'")
            as_of = cur.fetchone()[0]
        win = (lookback + skip + 10) if signal == "absmom" else (sma_w + 10)
        start = (pd.Timestamp(as_of) - pd.Timedelta(days=int(win * 1.5))).strftime("%Y-%m-%d")
        data = _load_etf(cur, codes, start, as_of)
    finally:
        con.close()

    rows = []
    for code, g in data.items():
        c = g["close"].astype(float).dropna()
        rec = dict(code=code, close=float(c.iloc[-1]) if len(c) else float("nan"),
                   hold=False, weight=0.0, gap_pct=float("nan"),
                   sma=float("nan"), mom=float("nan"),
                   vol_60=float("nan"), n=len(c))
        if signal == "absmom":
            need = lookback + skip + 1
            if len(c) >= need:
                mom = c.iloc[-1 - skip] / c.iloc[-1 - skip - lookback] - 1
                rec["mom"] = float(mom); rec["gap_pct"] = float(mom * 100)
                rec["hold"] = bool(mom > 0)
        else:
            if len(c) >= sma_w:
                sma = float(c.rolling(sma_w).mean().iloc[-1])
                rec["sma"] = sma
                rec["gap_pct"] = float((c.iloc[-1] / sma - 1) * 100)
                rec["hold"] = bool(c.iloc[-1] > sma)
        if len(c) > 61:
            v = c.pct_change().iloc[-60:].std() * np.sqrt(252)
            rec["vol_60"] = float(v) if np.isfinite(v) else float("nan")
        rows.append(rec)

    held = [r for r in rows if r["hold"]]
    if weight == "inv_vol" and held:
        for r in held:
            r["_w"] = (1.0 / max(r["vol_60"], 0.05)) if np.isfinite(r["vol_60"]) else 0.0
        tot = sum(r["_w"] for r in held)
        for r in held:
            r["weight"] = round(r["_w"] / tot, 4) if tot > 0 else 0.0
    elif held:
        for r in held:
            r["weight"] = round(1.0 / len(held), 4)
    for r in rows:
        r.pop("_w", None)
    return dict(as_of=str(as_of), signal=signal, weight=weight,
                n_hold=len(held), n_total=len(rows), holdings=rows)


def run_signal(as_of=None, signal="sma200", weight="inv_vol", clean_only=True):
    """算出当前(最新交易日)持仓信号；返回 current_signal 的 dict（含 start）。"""
    from .io_db import load_etf_universe
    if as_of is None:
        con = psycopg2.connect(**DB); cur = con.cursor()
        cur.execute("SELECT MAX(trade_date) FROM bars_1d WHERE asset_type='etf'")
        as_of = cur.fetchone()[0]; con.close()
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=800)).strftime("%Y-%m-%d")
    codes = load_etf_universe(start, as_of, clean_only)
    if not codes:
        raise DataError("ETF 池为空")
    sig = current_signal(codes, as_of, signal=signal, weight=weight)
    sig["start"] = start
    return sig


def regime_split(codes, start="2019-01-01", end="2026-09-30", signal="sma200",
                 weight="inv_vol", sma_w=200, lookback=252, skip=21):
    """按基准(等权ETF)滚动12月动量划分 牛/震荡/熊，对比策略与基准在各 regime 的月度表现。

    目的：证明好方法在熊市段真减仓(常转现金)、不是单纯牛市漏赚。
    regime: 基准12月动量 > +10% 牛市 / < -10% 熊市 / 其余震荡。
    """
    strat = abs_momentum_etf(codes, start, end, signal=signal, weight=weight,
                             sma_w=sma_w, lookback=lookback, skip=skip)
    base = etf_equal_weight(codes, start, end, gate_code=None)
    sr = strat["nav"].resample("ME").last().pct_change().dropna()
    br = base["nav"].resample("ME").last().pct_change().dropna()
    bm = (1 + br).cumprod()
    mom12 = bm / bm.shift(12) - 1
    months = sorted(set(sr.index) & set(br.index) & set(mom12.index))
    recs = []
    for m in months:
        if pd.isna(mom12[m]):
            continue
        regime = "bull" if mom12[m] > 0.10 else ("bear" if mom12[m] < -0.10 else "side")
        recs.append(dict(month=m, regime=regime, s=sr[m], b=br[m], excess=sr[m] - br[m]))
    from collections import defaultdict
    agg = defaultdict(lambda: dict(n=0, s=0.0, b=0.0, ex=0.0, win=0))
    for r in recs:
        a = agg[r["regime"]]
        a["n"] += 1; a["s"] += r["s"]; a["b"] += r["b"]; a["ex"] += r["excess"]
        a["win"] += 1 if r["excess"] > 0 else 0
    out = dict(n_months=len(recs), config=f"{signal}/{weight}", regimes={})
    for k, a in agg.items():
        n = a["n"]
        out["regimes"][k] = dict(n=n,
                                 avg_strat=a["s"] / n, avg_base=a["b"] / n,
                                 avg_excess=a["ex"] / n, win_rate=a["win"] / n,
                                 ann_strat=(1 + a["s"] / n) ** 12 - 1,
                                 ann_base=(1 + a["b"] / n) ** 12 - 1)
    return out


def run_regime(signal="sma200", weight="inv_vol", clean_only=True,
               start="2019-01-01", end="2026-09-30"):
    """跑 regime 切分；返回 regime_split 的 dict（含 n_etf/start/end）。"""
    from .io_db import load_etf_universe
    codes = load_etf_universe(start, end, clean_only)
    if not codes:
        raise DataError("ETF 池为空")
    out = regime_split(codes, start=start, end=end, signal=signal, weight=weight)
    out["n_etf"] = len(codes); out["start"] = start; out["end"] = end
    return out
