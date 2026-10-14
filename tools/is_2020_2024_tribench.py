"""F4/F6/F5/F7 长周期 IS 三口径复核（2020-2024）
============================================

目的：上一轮（tools/oos_2025_tribench.py）在 2025-2026 OOS 发现
「同 amount 十分位口径」下 F4 超额归零（-12.23bp, 去极值 -1.89bp），
而中位数口径 +253bp。怀疑旧 dixi 结论"F4 全A +22.8bp(t+14.6)"
也是等权基准被妖股绑架 + 小盘风格效应造成的。

本脚本把同一套三口径搬到 2020-2024 长 IS，逐信号逐年输出，
用于判定 docs/warfares_catalog.md §A 战法2 的 ✅ 是否要下调。

规则与 dixi_fullpanel_backtest.py 完全一致：
  T 收盘出信号 -> T+1 开盘买 -> 最早 T+2 收盘出
  出场 = 首个 {止损<=-8% / 止盈>=+10% / 最长持有20交易日}
  成本 0.25%，剔次新(<250日)/ST/open<=0/|日收益|>50%
  指标暖机：多载 2018 起数据，2020-01-02 起才允许出信号
"""
import json
import os
import tempfile
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
END = "2024-12-31"


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
    daily_broad = d.groupby("date")["dret"].mean()
    daily_median = d.groupby("date")["dret"].median()
    daily_dec = d.groupby(["date", "size_bucket"])["dret"].mean().unstack("size_bucket")
    return {
        "broad": (1 + daily_broad).cumprod(),
        "median": (1 + daily_median).cumprod(),
        "decile": (1 + daily_dec).cumprod(),
    }


def make_signals(panel):
    g = panel.groupby("code")
    panel["sma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    panel["sma60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    panel["sma10"] = g["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    panel["tsmin20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).min())
    panel["svol5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    panel["max60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).max())
    for col in ["sma20", "sma60", "sma10", "tsmin20", "max60"]:
        panel[col] = g[col].transform(lambda s: s.ffill(limit=5))
    c = panel["close"]
    p1 = panel["prev_close"]
    p2 = panel["prev2_close"]
    sma20 = panel["sma20"]
    sma60 = panel["sma60"]
    sma10 = panel["sma10"]
    tsmin20 = panel["tsmin20"]
    svol5 = panel["svol5"]
    max60 = panel["max60"]
    prev_sma20 = g["sma20"].shift(1)
    panel["F4"] = ((c - sma20).abs() / sma20 < 0.03) & (c > sma20) & (p1 <= prev_sma20)
    panel["F6"] = (tsmin20 < sma60 * 0.93) & (c > sma20) & (panel["volume"] > svol5 * 1.5)
    panel["F5"] = (p1 / p2 > 1.095) & (c < p1) & (panel["volume"] > svol5 * 1.5)
    panel["F7"] = (max60.shift(20) / max60 > 1.25) & (c > sma20) & (c > sma10) & (panel["volume"] > svol5 * 1.3)
    return panel


def simulate(panel, benches, signal_col, year=None):
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
            reason = "timeout"
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(ac[j]) or ac[j] <= 0:
                    continue
                hi = ah[j] / buy - 1
                lo = al[j] / buy - 1
                if lo <= STOP_LOSS:
                    exit_price = buy * (1 + STOP_LOSS)
                    ej = j
                    exited = True
                    reason = "stop"
                    break
                if hi >= TAKE_PROFIT:
                    exit_price = buy * (1 + TAKE_PROFIT)
                    ej = j
                    exited = True
                    reason = "tp"
                    break
            if exit_price is None or not np.isfinite(exit_price):
                continue
            ret = exit_price / buy - 1 - COST
            d0 = pd.Timestamp(ad[i + 1]).date()
            d1 = pd.Timestamp(ad[ej]).date()
            entry = pd.Timestamp(ad[i + 1])
            if year is not None and entry.year != year:
                continue
            exb = exm = exd = np.nan
            if d0 in nav_b and d1 in nav_b and nav_b[d0] > 0:
                exb = ret - (nav_b[d1] / nav_b[d0] - 1)
            if d0 in nav_m and d1 in nav_m and nav_m[d0] > 0:
                exm = ret - (nav_m[d1] / nav_m[d0] - 1)
            dm = nav_d.get(b)
            if dm is not None and d0 in dm and d1 in dm and dm[d0] > 0:
                exd = ret - (dm[d1] / dm[d0] - 1)
            trades.append(dict(entry=entry, ex_broad=exb, ex_median=exm, ex_decile=exd, ret=ret, reason=reason))
    return pd.DataFrame(trades)


def stats(df, col):
    v = df[col].dropna().values
    if len(v) == 0:
        return dict(n=0)
    m = v.mean()
    t = m / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0.0
    lo_c, hi_c = np.percentile(v, [1, 99])
    trimmed = v[(v >= lo_c) & (v <= hi_c)]
    return dict(n=len(v), mean_bp=float(m * 1e4), t=float(t),
                median_bp=float(np.median(v) * 1e4), trimmed_bp=float(trimmed.mean() * 1e4),
                win=float((v > 0).mean()))


def main():
    print(f"[1] 加载 {LOAD_START}~{END} ...")
    df = load_panel(LOAD_START, END)
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df["prev2_close"] = grp["close"].shift(2)
    panel = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    panel = panel[panel["date"] >= pd.Timestamp(SIM_START)].copy()
    panel["size_bucket"] = panel.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    print(f"     有效样本 {len(panel):,}  {panel['date'].min().date()} ~ {panel['date'].max().date()}")
    benches = build_benchmarks(panel)
    for k in ["broad", "median"]:
        nav = benches[k]
        print(f"     {k} NAV 终点 {nav.iloc[-1]:.3f} ({nav.index.min().date()}~{nav.index.max().date()})")
    panel = make_signals(panel)
    for col in ["F4", "F6", "F5", "F7"]:
        print(f"     {col} 信号天数: {int(panel[col].sum()):,}")

    out = {}
    print("\n[2] 2020-2024 全期 + 逐年 三口径 ...")
    for col in ["F4", "F6", "F5", "F7"]:
        tr = simulate(panel, benches, col)
        print(f"  === {col} 2020-2024 全期 ===")
        for name, c in [("等权", "ex_broad"), ("中位数", "ex_median"), ("同amount档", "ex_decile")]:
            s = stats(tr, c)
            if s["n"]:
                print(f"    [{name}] n={s['n']:,} {s['mean_bp']:+.2f}bp t={s['t']:+.2f} "
                      f"中位 {s['median_bp']:+.1f} 去极值 {s['trimmed_bp']:+.2f} 胜率 {s['win']:.1%}")
                out[f"{col}_all_{c}"] = s
        for y in range(2020, 2025):
            sub = simulate(panel, benches, col, year=y)
            if sub.empty:
                continue
            line = f"    {y}: "
            for name, c in [("等权", "ex_broad"), ("中位数", "ex_median"), ("同档", "ex_decile")]:
                s = stats(sub, c)
                line += f"{name} {s['mean_bp']:+7.1f}bp(t={s['t']:+6.1f}) | " if s["n"] else f"{name} -- | "
            print(line)
            out[f"{col}_{y}"] = {c: stats(sub, c) for c in ["ex_broad", "ex_median", "ex_decile"]}

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "is_2020_2024_tribench_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
