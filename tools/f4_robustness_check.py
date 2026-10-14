"""F4 同 amount 档口径的年份稳定性 + 成本敏感性 + 无止损对照
================================================================

上一轮 is_2020_2024_tribench.py 得到：
  F4 同 amount 档 2020-2024 全期 +36.53bp (t=+20.76)
  但逐年：2020 -13.1 / 2021 +70.5 / 2022 +112.5 / 2023 +27.7 / 2024 -11.6
  → 不稳定，2020 和 2024 为负，年际摆动 >120bp。这本身是弱信号/regime 信号的特征。

本脚本做三件事（都在"同 amount 档"口径下，即最严基准）：
  1. 成本敏感性：0 / 0.25% / 0.5% / 1% 四档，看 +36.53bp 是否被成本吃掉。
  2. 无止损对照（仅持有20日）：验证"必须带止损"这条铁律在同档口径下是否仍成立。
  3. F4 的规模分档拆解：看 +36.53bp 来自哪些 amount 十分位
     （如果集中在最高/最低档，说明还是风格而非选股）。

规则其余部分与 dixi_fullpanel_backtest.py 一致。
"""
import os
import tempfile
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
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
    return {"broad": (1 + daily_broad).cumprod(),
            "median": (1 + daily_median).cumprod(),
            "decile": (1 + daily_dec).cumprod()}


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
    return panel


def simulate(panel, benches, signal_col, cost, use_stop=True):
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
                if use_stop:
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
            ret = exit_price / buy - 1 - cost
            d0 = pd.Timestamp(ad[i + 1]).date()
            d1 = pd.Timestamp(ad[ej]).date()
            dm = nav_d.get(b)
            if dm is None or d0 not in dm or d1 not in dm or dm[d0] <= 0:
                continue
            exd = ret - (dm[d1] / dm[d0] - 1)
            trades.append(dict(entry=pd.Timestamp(ad[i + 1]), decile=int(b), ex=exd, ret=ret, reason=reason))
    return pd.DataFrame(trades)


def st(v):
    v = np.asarray(v)
    if len(v) == 0:
        return "n=0"
    m = v.mean()
    t = m / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0.0
    return f"n={len(v):,} {m*1e4:+.2f}bp t={t:+.2f} 胜率{(v>0).mean():.1%}"


def main():
    print("[1] 加载 + 基准 ...")
    df = load_panel(LOAD_START, END)
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df["prev2_close"] = grp["close"].shift(2)
    panel = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    panel = panel[panel["date"] >= pd.Timestamp(SIM_START)].copy()
    panel["size_bucket"] = panel.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    benches = build_benchmarks(panel)
    panel = make_signals(panel)

    print("\n[2] F4 成本敏感性（同 amount 档口径, 带止损）...")
    for cost in [0.0, 0.0025, 0.005, 0.01]:
        tr = simulate(panel, benches, "F4", cost, use_stop=True)
        print(f"    成本 {cost*100:.2f}%: {st(tr['ex'].values)}")

    print("\n[3] F4 无止损对照（同 amount 档口径, 成本0.25%）...")
    for use_stop in [True, False]:
        tr = simulate(panel, benches, "F4", 0.0025, use_stop=use_stop)
        label = "带≤8%止损/+10%止盈" if use_stop else "无止损(仅持有20日)"
        print(f"    {label}: {st(tr['ex'].values)}")
        if not tr.empty:
            for r, gg in tr.groupby("reason"):
                print(f"        出场[{r}] n={len(gg):,} {gg['ex'].mean()*1e4:+.2f}bp")

    print("\n[4] F4 规模分档拆解（成本0.25%, 带止损, 同档基准）...")
    tr = simulate(panel, benches, "F4", 0.0025, use_stop=True)
    print("    decile  0=最低amount ... 9=最高amount")
    for d in range(10):
        sub = tr[tr["decile"] == d]
        if sub.empty:
            continue
        print(f"    {d}: {st(sub['ex'].values)}")

    print("\n[5] F4 逐年 + 分档（同 amount 档口径）...")
    for y in range(2020, 2025):
        sub = tr[tr["entry"].dt.year == y]
        print(f"    {y}: {st(sub['ex'].values)}")

    print("\n[6] F6 同档口径分档 + 成本敏感性 ...")
    for cost in [0.0, 0.0025, 0.005]:
        tr6 = simulate(panel, benches, "F6", cost, use_stop=True)
        print(f"    F6 成本 {cost*100:.2f}%: {st(tr6['ex'].values)}")
    tr6 = simulate(panel, benches, "F6", 0.0025, use_stop=True)
    for d in range(10):
        sub = tr6[tr6["decile"] == d]
        if not sub.empty:
            print(f"    F6 decile {d}: {st(sub['ex'].values)}")


if __name__ == "__main__":
    main()
