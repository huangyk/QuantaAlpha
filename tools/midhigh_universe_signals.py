"""找不依赖小盘风格的新战法：只在**中高成交额分档**里测技术信号
==============================================================

背景（第一轮结论）：F4/F6/F7 的正超额几乎全部来自最低 amount 十分位，
是"小盘风格 + 止损"，不是选股 alpha；且 OOS 已转负。

本轮思路：把宇宙限制在**中高成交额**（amount decile >= 5，即后 50%），
在这里测一批经典技术信号。若某个信号在中高成交额档仍有正超额，
它才可能是真 alpha（不依赖小盘/流动性风格）。

测试信号（全部 T 日收盘判定 → T+1 开盘买入 → 最早 T+2 收盘出，
出场 = 首个 {止损≤-8% / 止盈≥+10% / 最长持有20交易日}，成本 0.25%）：
  S1 均线多头回踩（F4 同款，作对照）
  S2 20日新高突破 + 放量
  S3 缩量回踩 MA10（量能退潮型低吸）
  S4 MACD 金叉（DIF>DEA 且 昨 DIF<=昨 DEA）
  S5 RSI(14) 从下方上穿 50（动量转强）
  S6 布林带下轨支撑（close < 下轨 且 收>下轨*0.99）
  S7 量价齐升（close>MA20 且 vol>MA5*1.5 且 close 连涨3日）
  S8 低波突破（近20日波动率处历史低分位 且 close 创20日新高）

铁律：train 内不选参；本脚本只在 2020-2024 跑，2025-2026 作 OOS 复验。
"""
import os
import tempfile
import json
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
COST = 0.0025
STOP_LOSS = -0.08
TAKE_PROFIT = 0.10
MAX_HOLD = 20
LOAD_START = "2018-01-01"
SIM_START = "2020-01-02"
END = "2026-09-30"
MIN_AMT_DEC = 5   # 只用 amount 十分位 >= 5（中高成交额）


def load_panel(start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close,
               b.volume, b.amount, coalesce(m.is_st, false) as is_st
        from bars_1d b
        left join asset_master m on m.asset_code = b.asset_code
        where b.trade_date between %s and %s
          and b.asset_type is distinct from 'etf' and b.close is not null
        order by b.asset_code, b.trade_date
        """,
        (start, end),
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "amount", "is_st"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def build_benchmarks(panel):
    g = panel.groupby("code")
    panel["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    valid_b = panel["listed_n"] >= 250
    d = panel.loc[valid_b & panel["dret"].notna() & (panel["dret"].abs() < 0.5),
                  ["date", "dret", "size_bucket"]].copy()
    nav_broad = (1 + d.groupby("date")["dret"].mean()).cumprod()
    nav_median = (1 + d.groupby("date")["dret"].median()).cumprod()
    nav_dec = (1 + d.groupby(["date", "size_bucket"])["dret"].mean().unstack("size_bucket")).cumprod()
    return {"broad": nav_broad, "median": nav_median, "decile": nav_dec}


def make_signals(panel):
    g = panel.groupby("code")
    c = panel["close"]
    v = panel["volume"]
    panel["sma5"] = g["close"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    panel["sma10"] = g["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    panel["sma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    panel["sma60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    panel["svol5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    panel["svol20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    panel["tsmax20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).max())
    panel["tsmin20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).min())
    for col in ["sma5", "sma10", "sma20", "sma60", "svol5", "svol20", "tsmax20", "tsmin20"]:
        panel[col] = g[col].transform(lambda s: s.ffill(limit=5))
    # MACD(12,26,9)
    ema12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    panel["dif"] = ema12 - ema26
    panel["dea"] = panel["dif"].groupby(panel["code"]).transform(lambda s: s.ewm(span=9, adjust=False).mean())
    # RSI(14)
    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.groupby(panel["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    al = loss.groupby(panel["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    panel["rsi14"] = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    # 布林带(20,2)
    std20 = g["close"].transform(lambda s: s.rolling(20, min_periods=20).std())
    panel["bb_up"] = panel["sma20"] + 2 * std20
    panel["bb_dn"] = panel["sma20"] - 2 * std20
    # 20日波动率 + 其历史分位
    lr = g["close"].transform(lambda s: np.log(s / s.shift(1)))
    vol20 = lr.groupby(panel["code"]).transform(lambda s: s.rolling(20, min_periods=20).std())
    panel["vol20pct"] = vol20.groupby(panel["code"]).transform(
        lambda s: s.rolling(250, min_periods=60).rank(pct=True))
    # 连涨3日
    panel["up3"] = (c > panel["sma20"]) & (c > g["close"].shift(1)) & \
                   (g["close"].shift(1) > g["close"].shift(2)) & \
                   (g["close"].shift(2) > g["close"].shift(3))

    p1 = panel["prev_close"]
    prev_sma20 = g["sma20"].shift(1)
    prev_dif = g["dif"].shift(1)
    prev_dea = g["dea"].shift(1)
    prev_rsi = g["rsi14"].shift(1)

    panel["S1_f4"] = ((c - panel["sma20"]).abs() / panel["sma20"] < 0.03) & (c > panel["sma20"]) & (p1 <= prev_sma20)
    panel["S2_breakout"] = (c > panel["tsmax20"].shift(1)) & (v > panel["svol5"] * 1.5) & (c > panel["sma60"])
    panel["S3_shrink_pull"] = (c > panel["sma10"]) & (v < panel["svol20"] * 0.7) & \
                              (g["close"].shift(1) < g["sma10"].shift(1))
    panel["S4_macd"] = (panel["dif"] > panel["dea"]) & (prev_dif <= prev_dea) & (c > panel["sma20"])
    panel["S5_rsi50"] = (panel["rsi14"] > 50) & (prev_rsi <= 50) & (c > panel["sma20"])
    panel["S6_bbdn"] = (g["low"].shift(1) < panel["bb_dn"].shift(1)) & (c > panel["sma20"]) & \
                       (c > g["low"].shift(1))
    panel["S7_volup3"] = panel["up3"] & (v > panel["svol5"] * 1.5)
    panel["S8_lowvol_break"] = (panel["vol20pct"] < 0.3) & (c > panel["tsmax20"].shift(1))
    return panel


def simulate(panel, benches, signal_col, only_after=None, only_before=None):
    nav_b = {ts.date(): v for ts, v in benches["broad"].items()}
    nav_m = {ts.date(): v for ts, v in benches["median"].items()}
    nav_d = {b: {ts.date(): v for ts, v in benches["decile"][b].items()} for b in benches["decile"].columns}
    trades = []
    for code, g in panel.groupby("code"):
        g = g.sort_values("date")
        ao = g["open"].values
        ah = g["high"].values
        al = g["low"].values
        ac = g["close"].values
        sig = g[signal_col].values
        ad = g["date"].values
        ab = g["size_bucket"].values
        n = len(g)
        for i in range(n):
            if i + 2 > n - 1:
                break
            if not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                continue
            buy = ao[i + 1]
            b = ab[i + 1]
            last_j = min(i + 1 + MAX_HOLD, n - 1)
            exited = False
            ej = last_j
            exit_price = ac[last_j]
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(ac[j]) or ac[j] <= 0:
                    continue
                hi = ah[j] / buy - 1
                lo = al[j] / buy - 1
                if lo <= STOP_LOSS:
                    exit_price = buy * (1 + STOP_LOSS)
                    ej = j
                    exited = True
                    break
                if hi >= TAKE_PROFIT:
                    exit_price = buy * (1 + TAKE_PROFIT)
                    ej = j
                    exited = True
                    break
            if exit_price is None or not np.isfinite(exit_price):
                continue
            ret = exit_price / buy - 1 - COST
            d0 = pd.Timestamp(ad[i + 1]).date()
            d1 = pd.Timestamp(ad[ej]).date()
            entry = pd.Timestamp(ad[i + 1])
            if only_after is not None and entry < only_after:
                continue
            if only_before is not None and entry >= only_before:
                continue
            exb = exm = exd = np.nan
            if d0 in nav_b and d1 in nav_b and nav_b[d0] > 0:
                exb = ret - (nav_b[d1] / nav_b[d0] - 1)
            if d0 in nav_m and d1 in nav_m and nav_m[d0] > 0:
                exm = ret - (nav_m[d1] / nav_m[d0] - 1)
            dm = nav_d.get(b)
            if dm is not None and d0 in dm and d1 in dm and dm[d0] > 0:
                exd = ret - (dm[d1] / dm[d0] - 1)
            trades.append(dict(entry=entry, ex_broad=exb, ex_median=exm, ex_decile=exd, ret=ret))
    return pd.DataFrame(trades)


def st(df, col):
    v = df[col].dropna().values
    if len(v) == 0:
        return "n=0"
    m = v.mean()
    t = m / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0.0
    return f"n={len(v):>7,} {m*1e4:+8.2f}bp t={t:+7.2f} 胜率{(v>0).mean():.1%}"


def main():
    print(f"[1] 加载 2018~2026 全A（只用 amount decile>={MIN_AMT_DEC}）...")
    df = load_panel(LOAD_START, END)
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    df = df[df["date"] >= pd.Timestamp(SIM_START)].copy()
    df["size_bucket"] = df.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    print(f"     全A有效 {len(df):,} 行")
    panel = df[df["size_bucket"] >= MIN_AMT_DEC].copy()
    print(f"     中高成交额(decile>={MIN_AMT_DEC}) {len(panel):,} 行 / "
          f"{panel['code'].nunique():,} 只  {panel['date'].min().date()}~{panel['date'].max().date()}")

    print("\n[2] 建基准 ...")
    benches = build_benchmarks(panel)
    for k in ["broad", "median"]:
        nv = benches[k]
        print(f"     {k}: {nv.index.min().date()}~{nv.index.max().date()} 终点 {nv.iloc[-1]:.3f}")

    print("\n[3] 算信号 ...")
    panel = make_signals(panel)
    sig_cols = [c2 for c2 in panel.columns if c2.startswith("S") and "_" in c2]
    for c2 in sig_cols:
        print(f"     {c2}: {int(panel[c2].sum()):,}")

    out = {}
    print("\n[4] 2020-2024 IS 三口径 ...")
    for c2 in sig_cols:
        tr = simulate(panel, benches, c2, only_before=pd.Timestamp("2025-01-01"))
        print(f"  {c2}")
        for name, col in [("等权", "ex_broad"), ("中位数", "ex_median"), ("同档", "ex_decile")]:
            print(f"     [{name}] {st(tr, col)}")
        out[f"{c2}_is"] = {col: (float(tr[col].dropna().mean()) * 1e4 if not tr.empty else None)
                           for col in ["ex_broad", "ex_median", "ex_decile"]}
        out[f"{c2}_is_n"] = int(len(tr))

    print("\n[5] 2025-2026 OOS 三口径 ...")
    for c2 in sig_cols:
        tr = simulate(panel, benches, c2, only_after=pd.Timestamp("2025-01-01"))
        print(f"  {c2}")
        for name, col in [("等权", "ex_broad"), ("中位数", "ex_median"), ("同档", "ex_decile")]:
            print(f"     [{name}] {st(tr, col)}")
        out[f"{c2}_oos"] = {col: (float(tr[col].dropna().mean()) * 1e4 if not tr.empty else None)
                            for col in ["ex_broad", "ex_median", "ex_decile"]}
        out[f"{c2}_oos_n"] = int(len(tr))

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "midhigh_universe_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
