"""F4/F6 在 2025-2026 OOS 段的深度拆解。

目的：
1. 2025 年样本极少（F4 41笔 / F6 18笔）→ 那 -868bp/-818bp 极可能是少数极端离群值，
   必须看分布而非均值，否则会误判"OOS 崩了"。
2. 2026 年是主样本（F4 2456笔 / F6 5476笔），要看是否稳定、是否被少数日期主导。
3. 加随机入场对照，隔离"止损机制收益" vs "信号择时"。

规则与 tools/oos_2025_backtest.py 完全一致（T+1开盘买、≤8%止损/+10%止盈/最长20日、
同池等权+同市值档基准、成本0.25%）。
"""
import os
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
COST = 0.0025
STOP_LOSS = -0.08
TAKE_PROFIT = 0.10
MAX_HOLD = 20
LOAD_START = "2023-01-01"
SIM_START = "2024-06-03"


def load_panel(start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close,
               b.volume, b.float_shares, coalesce(m.is_st, false) as is_st
        from bars_1d b
        left join asset_master m on m.asset_code = b.asset_code
        where b.trade_date between %s and %s
          and b.asset_type is distinct from 'etf'
          and b.close is not null
        order by b.asset_code, b.trade_date
        """,
        (start, end),
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "float_shares", "is_st"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def build_benchmark(panel):
    g = panel.groupby("code")
    panel["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    valid_b = panel["listed_n"] >= 250
    d = panel.loc[valid_b & panel["dret"].notna() & (panel["dret"].abs() < 0.5), ["date", "dret", "size_bucket"]].copy()
    nav_broad = (1 + d.groupby("date")["dret"].mean()).cumprod()
    nav_size = (1 + d.groupby(["date", "size_bucket"])["dret"].mean().unstack("size_bucket")).cumprod()
    return {"broad": nav_broad, "size": nav_size}


def make_signals(panel):
    g = panel.groupby("code")
    panel["sma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    panel["sma60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    panel["sma10"] = g["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    panel["tsmin20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).min())
    panel["svol5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    for col in ["sma20", "sma60", "sma10", "tsmin20"]:
        panel[col] = g[col].transform(lambda s: s.ffill(limit=5))
    c = panel["close"]
    p1 = panel["prev_close"]
    p2 = panel["prev2_close"]
    sma20 = panel["sma20"]
    sma60 = panel["sma60"]
    sma10 = panel["sma10"]
    tsmin20 = panel["tsmin20"]
    svol5 = panel["svol5"]
    prev_sma20 = g["sma20"].shift(1)
    panel["F4"] = ((c - sma20).abs() / sma20 < 0.03) & (c > sma20) & (p1 <= prev_sma20)
    panel["F6"] = (tsmin20 < sma60 * 0.93) & (c > sma20) & (panel["volume"] > svol5 * 1.5)
    panel["F5"] = (p1 / p2 > 1.095) & (c < p1) & (panel["volume"] > svol5 * 1.5)
    return panel


def simulate(panel, bench, signal_col, only_after=None):
    nav_map = {ts.date(): v for ts, v in bench["broad"].items()}
    size_maps = {b: {ts.date(): v for ts, v in bench["size"][b].items()} for b in bench["size"].columns}
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
            nv0 = nav_map.get(d0)
            nv1 = nav_map.get(d1)
            if nv0 is None or nv1 is None or nv0 <= 0:
                continue
            ex = ret - (nv1 / nv0 - 1)
            sm = size_maps.get(b)
            if sm is not None:
                sv0 = sm.get(d0)
                sv1 = sm.get(d1)
                exs = ret - (sv1 / sv0 - 1) if (sv0 is not None and sv1 is not None and sv0 > 0) else np.nan
            else:
                exs = np.nan
            entry = pd.Timestamp(ad[i + 1])
            if only_after is not None and entry < only_after:
                continue
            trades.append(dict(entry=entry, exit=pd.Timestamp(ad[ej]), code=code, ex=ex, exs=exs, ret=ret,
                               reason=("stop" if exited and lo <= STOP_LOSS else ("tp" if exited else "timeout"))))
    return pd.DataFrame(trades)


def describe(df, label):
    if df.empty:
        print(f"    {label}: 空")
        return
    ex = df["ex"].values
    exs = df["exs"].dropna().values
    m = ex.mean()
    t = m / (ex.std() / np.sqrt(len(ex))) if ex.std() > 0 else 0.0
    med = np.median(ex)
    p10, p90 = np.percentile(ex, [10, 90])
    print(f"    {label}: n={len(ex):,} mean={m*1e4:+.2f}bp t={t:+.2f} | median={med*1e4:+.2f}bp "
          f"p10={p10*1e4:+.1f} p90={p90*1e4:+.1f} | min={ex.min()*1e4:+.0f} max={ex.max()*1e4:+.0f} "
          f"| win={(ex>0).mean():.1%} | size_mean={exs.mean()*1e4:+.2f}bp")
    # 去掉最好/最差各 1% 后的均值（抗离群）
    lo_c, hi_c = np.percentile(ex, [1, 99])
    trimmed = ex[(ex >= lo_c) & (ex <= hi_c)]
    print(f"        去掉两端各1%后: mean={trimmed.mean()*1e4:+.2f}bp (n={len(trimmed):,})")
    # 按出场原因拆
    for r, gg in df.groupby("reason"):
        print(f"        出场[{r}]: n={len(gg):,} mean={gg['ex'].mean()*1e4:+.2f}bp")


def main():
    print("[1] 加载 + 建基准 ...")
    df = load_panel(LOAD_START, "2026-09-30")
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df["prev2_close"] = grp["close"].shift(2)
    df["float_mcap"] = df["close"] * df["float_shares"]
    panel = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0) & df["float_shares"].notna()].copy()
    panel = panel[panel["date"] >= pd.Timestamp(SIM_START)].copy()
    panel["size_bucket"] = panel.groupby("date")["float_mcap"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    bench = build_benchmark(panel)
    panel = make_signals(panel)

    for col in ["F4", "F6"]:
        print(f"\n[2] {col} 分布拆解 ...")
        tr = simulate(panel, bench, col, only_after=pd.Timestamp("2025-01-01"))
        tr.to_pickle(os.path.join(os.environ.get("QA_OUT_DIR", "."), f"trades_{col}.pkl"))
        describe(tr, "OOS 2025-2026 全段")
        for y in [2025, 2026]:
            sub = tr[tr["entry"].dt.year == y]
            describe(sub, f"{y} 年")
        # 2026 按半年拆
        sub26 = tr[tr["entry"].dt.year == 2026]
        for h, lo, hi in [("H1", 1, 6), ("H2", 7, 12)]:
            s = sub26[(sub26["entry"].dt.month >= lo) & (sub26["entry"].dt.month <= hi)]
            describe(s, f"2026-{h}")
        # 按入场日聚合后看是否被少数日期主导
        daily = tr.groupby("entry")["ex"].mean()
        top = daily.sort_values(ascending=False)
        print(f"    2025-26 每日均值: 最好5日 {[f'{v*1e4:+.0f}bp' for v in top.head(5).values]}")
        print(f"                     最差5日 {[f'{v*1e4:+.0f}bp' for v in top.tail(5).values]}")
        print(f"                     正收益日占比 {float((daily>0).mean()):.1%}  共 {len(daily)} 个入场日")

    print("\n[3] F6 在更长 IS(2020-2024) 复核 —— 旧结论说 F6≈0，本次看是否反转 ...")
    df2 = load_panel("2018-01-01", "2024-12-31")
    g2 = df2.groupby("code")
    df2["listed_n"] = g2.cumcount() + 1
    df2["prev_close"] = g2["close"].shift(1)
    df2["prev2_close"] = g2["close"].shift(2)
    df2["float_mcap"] = df2["close"] * df2["float_shares"]
    p2 = df2[(df2["listed_n"] >= 250) & (~df2["is_st"]) & (df2["open"] > 0) & df2["float_shares"].notna()].copy()
    p2 = p2[p2["date"] >= pd.Timestamp("2020-01-02")].copy()
    p2["size_bucket"] = p2.groupby("date")["float_mcap"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    b2 = build_benchmark(p2)
    p2 = make_signals(p2)
    for col in ["F4", "F6"]:
        tr = simulate(p2, b2, col)
        describe(tr, f"{col} 2020-2024 (IS 复核)")


if __name__ == "__main__":
    main()
