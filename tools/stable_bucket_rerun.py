"""用修正后的分档口径重跑 F4/F6/F5/F7 + S3
=========================================

第一轮发现"每日 amount 排名分档"做基准是**因果倒置**：
decile9 年化 +231%（每天成交额最大的 10% 当天平均涨 48.55bp）——
因为**当天大涨本身导致成交额暴增**，所以"当天成交额排名"包含了当天收益信息。
这不是资产属性，不可作为基准或宇宙划分。

本轮改用**滞后 60 日平均 amount 排名**分档（稳定资产属性），重跑：
  1. F4/F6/F5/F7 在稳定分档口径下的 IS(2020-2024)/OOS(2025-2026)
  2. 中高成交额宇宙（lagged decile>=5）里的 S1-S8 信号
  3. 三口径：等权 / 中位数 / 同稳定档

规则不变：T+1 开盘买、止损-8%/止盈+10%/持有20日、成本 0.25%。
"""
import os
import tempfile
import json
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
COST = 0.0025
STOP = -0.08
TAKE = 0.10
MAX_HOLD = 20
LOAD_START = "2018-01-01"
SIM_START = "2020-01-02"
END = "2026-09-30"


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
    d = panel.loc[panel["dret"].notna() & (panel["dret"].abs() < 0.5),
                  ["date", "dret", "bkt"]].copy()
    return {
        "broad": (1 + d.groupby("date")["dret"].mean()).cumprod(),
        "median": (1 + d.groupby("date")["dret"].median()).cumprod(),
        "decile": (1 + d.groupby(["date", "bkt"])["dret"].mean().unstack("bkt")).cumprod(),
    }


def make_signals(panel):
    g = panel.groupby("code")
    c = panel["close"]
    v = panel["volume"]
    for w, name in [(5, "sma5"), (10, "sma10"), (20, "sma20"), (60, "sma60")]:
        panel[name] = g["close"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    panel["svol5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    panel["svol20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    panel["tsmax20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).max())
    panel["tsmin20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).min())
    panel["max60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).max())
    for col in ["sma5", "sma10", "sma20", "sma60", "svol5", "svol20", "tsmax20", "tsmin20", "max60"]:
        panel[col] = g[col].transform(lambda s: s.ffill(limit=5))
    p1 = panel["prev_close"]
    p2 = panel["prev2_close"]
    prev_sma20 = g["sma20"].shift(1)
    prev_sma10 = g["sma10"].shift(1)

    panel["F4"] = ((c - panel["sma20"]).abs() / panel["sma20"] < 0.03) & (c > panel["sma20"]) & (p1 <= prev_sma20)
    panel["F6"] = (panel["tsmin20"] < panel["sma60"] * 0.93) & (c > panel["sma20"]) & (v > panel["svol5"] * 1.5)
    panel["F5"] = (p1 / p2 > 1.095) & (c < p1) & (v > panel["svol5"] * 1.5)
    panel["F7"] = (panel["max60"].shift(20) / panel["max60"] > 1.25) & (c > panel["sma20"]) & \
                  (c > panel["sma10"]) & (v > panel["svol5"] * 1.3)
    # S 族
    panel["S1_f4"] = panel["F4"]
    panel["S2_breakout"] = (c > panel["tsmax20"].shift(1)) & (v > panel["svol5"] * 1.5) & (c > panel["sma60"])
    panel["S3_shrink_pull"] = (c > panel["sma10"]) & (v < panel["svol20"] * 0.7) & \
                              (g["close"].shift(1) < prev_sma10)
    up3 = (c > g["close"].shift(1)) & (g["close"].shift(1) > g["close"].shift(2)) & \
          (g["close"].shift(2) > g["close"].shift(3))
    panel["S7_volup3"] = up3 & (v > panel["svol5"] * 1.5)
    # RSI(14)
    delta = g["close"].diff()
    ag = delta.clip(lower=0).groupby(panel["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    al = (-delta).clip(lower=0).groupby(panel["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    rsi = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    panel["S5_rsi50"] = (rsi > 50) & (rsi.groupby(panel["code"]).shift(1) <= 50) & (c > panel["sma20"])
    # MACD
    dif = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean()) - \
        g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    dea = dif.groupby(panel["code"]).transform(lambda s: s.ewm(span=9, adjust=False).mean())
    panel["S4_macd"] = (dif > dea) & (dif.groupby(panel["code"]).shift(1) <= dea.groupby(panel["code"]).shift(1)) & \
                       (c > panel["sma20"])
    # 低波突破
    lr = g["close"].transform(lambda s: np.log(s / s.shift(1)))
    vol20 = lr.groupby(panel["code"]).transform(lambda s: s.rolling(20, min_periods=20).std())
    vol20pct = vol20.groupby(panel["code"]).transform(lambda s: s.rolling(250, min_periods=60).rank(pct=True))
    panel["S8_lowvol_break"] = (vol20pct < 0.3) & (c > panel["tsmax20"].shift(1))
    return panel


def simulate(panel, benches, sig_col, only_after=None, only_before=None):
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
        sig = g[sig_col].values
        ad = g["date"].values
        ab = g["bkt"].values
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
                if lo <= STOP:
                    exit_price = buy * (1 + STOP)
                    ej = j
                    exited = True
                    break
                if hi >= TAKE:
                    exit_price = buy * (1 + TAKE)
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
            trades.append(dict(entry=entry, bkt=int(b) if pd.notna(b) else -1,
                               ex_broad=exb, ex_median=exm, ex_decile=exd))
    return pd.DataFrame(trades)


def st(df, col):
    v = df[col].dropna().values
    if len(v) == 0:
        return "n=0"
    m = v.mean()
    t = m / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0.0
    return f"n={len(v):>7,} {m*1e4:+8.2f}bp t={t:+7.2f} 胜率{(v>0).mean():.1%}"


def main():
    print("[1] 加载 + 稳定分档（滞后60日平均 amount 排名）...")
    df = load_panel(LOAD_START, END)
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df["prev2_close"] = grp["close"].shift(2)
    df = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    df = df[df["date"] >= pd.Timestamp(SIM_START)].copy()
    df["amt60_prev"] = df.groupby("code")["amount"].transform(
        lambda s: s.rolling(60, min_periods=60).mean()).groupby(df["code"]).shift(1)
    df["bkt"] = df.groupby("date")["amt60_prev"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    print(f"     {len(df):,} 行 / {df['code'].nunique():,} 只")

    print("\n[2] 全A 三口径基准（稳定分档）...")
    benches = build_benchmarks(df)
    for k in ["broad", "median"]:
        nv = benches[k]
        yrs = len(nv) / 252
        print(f"     {k}: 终点 {nv.iloc[-1]:.3f} 年化 {(nv.iloc[-1]**(1/yrs)-1)*100:+.2f}%")

    print("\n[3] 信号 ...")
    df = make_signals(df)
    sigs = ["F4", "F6", "F5", "F7", "S2_breakout", "S3_shrink_pull",
            "S4_macd", "S5_rsi50", "S7_volup3", "S8_lowvol_break"]
    for s in sigs:
        print(f"     {s}: {int(df[s].sum()):,}")

    out = {}
    print("\n[4] 全A IS(2020-2024) / OOS(2025-2026) 三口径 ...")
    for s in sigs:
        a = simulate(df, benches, s, only_before=pd.Timestamp("2025-01-01"))
        b = simulate(df, benches, s, only_after=pd.Timestamp("2025-01-01"))
        print(f"  {s}")
        print(f"     IS  [等权] {st(a,'ex_broad')} | [中位] {st(a,'ex_median')} | [同档] {st(a,'ex_decile')}")
        print(f"     OOS [等权] {st(b,'ex_broad')} | [中位] {st(b,'ex_median')} | [同档] {st(b,'ex_decile')}")
        out[s] = {"is": {c2: float(a[c2].dropna().mean())*1e4 if not a.empty else None
                         for c2 in ["ex_broad", "ex_median", "ex_decile"]},
                  "oos": {c2: float(b[c2].dropna().mean())*1e4 if not b.empty else None
                          for c2 in ["ex_broad", "ex_median", "ex_decile"]},
                  "is_n": int(len(a)), "oos_n": int(len(b))}

    print("\n[5] 中高成交额宇宙（lagged bkt>=5）...")
    sub = df[df["bkt"] >= 5].copy()
    bsub = build_benchmarks(sub)
    for s in sigs:
        a = simulate(sub, bsub, s, only_before=pd.Timestamp("2025-01-01"))
        b = simulate(sub, bsub, s, only_after=pd.Timestamp("2025-01-01"))
        print(f"  {s}")
        print(f"     IS  [等权] {st(a,'ex_broad')} | [中位] {st(a,'ex_median')} | [同档] {st(a,'ex_decile')}")
        print(f"     OOS [等权] {st(b,'ex_broad')} | [中位] {st(b,'ex_median')} | [同档] {st(b,'ex_decile')}")
        out[s + "_midhigh"] = {"is": {c2: float(a[c2].dropna().mean())*1e4 if not a.empty else None
                                      for c2 in ["ex_broad", "ex_median", "ex_decile"]},
                               "oos": {c2: float(b[c2].dropna().mean())*1e4 if not b.empty else None
                                       for c2 in ["ex_broad", "ex_median", "ex_decile"]},
                               "is_n": int(len(a)), "oos_n": int(len(b))}

    print("\n[6] S3 在稳定分档下的规模分档拆解（全A, 同档口径）...")
    tr = simulate(df, benches, "S3_shrink_pull")
    for d in range(10):
        s = tr[tr["bkt"] == d]
        if not s.empty:
            print(f"     bkt{d}: IS {st(s[s['entry']<pd.Timestamp('2025-01-01')],'ex_decile') if not s[s['entry']<pd.Timestamp('2025-01-01')].empty else 'n=0'} | "
                  f"OOS {st(s[s['entry']>=pd.Timestamp('2025-01-01')],'ex_decile') if not s[s['entry']>=pd.Timestamp('2025-01-01')].empty else 'n=0'}")

    print("\n[7] S3 逐年（全A, 同档口径）...")
    for y in range(2020, 2027):
        s = tr[tr["entry"].dt.year == y]
        if not s.empty:
            print(f"     {y}: {st(s,'ex_decile')}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "stable_bucket_rerun_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
