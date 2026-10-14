"""股票侧最后一搏：不依赖分档、不依赖小盘的信号族
===========================================

五轮下来股票侧结论：
  - 横截面技术信号（F4/F6/F5/F7 + 10 个 S 族）在"同稳定档"口径下全负
  - 每日分档做基准是未来函数（已禁用）
  - 小盘风格已被证实不可依赖（2025-26 崩塌）

本轮换**最后一条思路**：测**不依赖横截面排名、不依赖小盘**的信号族，
只用单股自身历史 + 大盘环境：

  G1 **大盘环境门控**：沪深300/中证全A 在其 SMA200 之上才交易个股（择时不选股）
  G2 **个股时间序列动量**：ret_60D > 0 且 close > SMA200
  G3 **低波 + 趋势**：vol20 处历史低分位 且 close > SMA200
  G4 **均线多头排列**：SMA20 > SMA60 > SMA200
  G5 **突破 + 大盘门控**：创 60 日新高 且 大盘在 SMA200 上
  G6 **量价齐升 + 大盘门控**
  G7 **回踩 + 大盘门控**（F4 加大盘门控，看门控能否救 F4）

全部：
  - 宇宙：**只取 amount 稳定档 >= 7**（中高成交额，规避小盘）
  - 基准：**同稳定档等权** + 中位数 + 等权（三口径）
  - T+1 开盘买、止损 -8%/止盈 +10%/持有 20 日、成本 0.25%
  - IS 2020-2024 / OOS 2025-2026
  - 成本敏感 0.25%/0.5%/1%

若这些仍全负，则股票侧可以正式定案"无可用 alpha"，全部精力转 ETF。
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
MIN_BKT = 7   # 只用中高成交额稳定档（>=7）


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
    d = panel.loc[panel["dret"].notna() & (panel["dret"].abs() < 0.5), ["date", "dret", "bkt"]].copy()
    return {
        "broad": (1 + d.groupby("date")["dret"].mean()).cumprod(),
        "median": (1 + d.groupby("date")["dret"].median()).cumprod(),
        "decile": (1 + d.groupby(["date", "bkt"])["dret"].mean().unstack("bkt")).cumprod(),
    }


def make_signals(panel, index_gate):
    """index_gate: Series[date] -> bool，大盘是否在 SMA200 上。"""
    g = panel.groupby("code")
    c = panel["close"]
    v = panel["volume"]
    for w, name in [(5, "sma5"), (10, "sma10"), (20, "sma20"), (60, "sma60"), (200, "sma200")]:
        panel[name] = g["close"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    panel["svol5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    panel["tsmax60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).max())
    panel["ret60"] = g["close"].transform(lambda s: s / s.shift(60) - 1)
    for col in ["sma5", "sma10", "sma20", "sma60", "sma200", "svol5", "tsmax60", "ret60"]:
        panel[col] = g[col].transform(lambda s: s.ffill(limit=5))
    lr = g["close"].transform(lambda s: np.log(s / s.shift(1)))
    vol20 = lr.groupby(panel["code"]).transform(lambda s: s.rolling(20, min_periods=20).std())
    panel["vol20pct"] = vol20.groupby(panel["code"]).transform(
        lambda s: s.rolling(250, min_periods=60).rank(pct=True))
    gate = panel["date"].map(index_gate).fillna(False)

    p1 = panel["prev_close"]
    prev_sma20 = g["sma20"].shift(1)
    up3 = (c > g["close"].shift(1)) & (g["close"].shift(1) > g["close"].shift(2)) & \
          (g["close"].shift(2) > g["close"].shift(3))

    panel["G1_gate_only"] = gate   # 只做大盘门控，不做个股筛选（全池等权入场）
    panel["G2_tsmom"] = (panel["ret60"] > 0) & (c > panel["sma200"])
    panel["G3_lowvol_trend"] = (panel["vol20pct"] < 0.3) & (c > panel["sma200"])
    panel["G4_ma_bull"] = (panel["sma20"] > panel["sma60"]) & (panel["sma60"] > panel["sma200"]) & (c > panel["sma20"])
    panel["G5_break_gate"] = (c > panel["tsmax60"].shift(1)) & gate
    panel["G6_volup_gate"] = up3 & (v > panel["svol5"] * 1.5) & gate
    panel["G7_pull_gate"] = ((c - panel["sma20"]).abs() / panel["sma20"] < 0.03) & (c > panel["sma20"]) & \
                            (p1 <= prev_sma20) & gate
    return panel


def simulate(panel, benches, sig_col, cost=COST, only_after=None, only_before=None):
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
            ret = exit_price / buy - 1 - cost
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
            trades.append(dict(entry=entry, ex_broad=exb, ex_median=exm, ex_decile=exd))
    return pd.DataFrame(trades)


def st(df, col):
    v = df[col].dropna().values
    if len(v) == 0:
        return "n=0"
    m = v.mean()
    t = m / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0.0
    return f"n={len(v):>7,} {m*1e4:+8.2f}bp t={t:+6.2f} 胜率{(v>0).mean():.1%}"


def build_index_gate(start, end):
    """大盘门控：用全 A 等权指数（bars_1d 全A 每日等权）在 SMA200 上。"""
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select trade_date, avg(close) as avg_close, count(*) as n
        from bars_1d
        where trade_date between %s and %s
          and asset_type is distinct from 'etf' and close is not null
        group by trade_date order by trade_date
        """,
        (start, end),
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    s = pd.Series({pd.Timestamp(r[0]): r[1] for r in rows})
    sma = s.rolling(200, min_periods=200).mean()
    gate = s > sma
    print(f"     大盘 SMA200 门控: 允许交易 {int(gate.sum())}/{len(gate)} 日 ({gate.mean():.1%})")
    return gate


def main():
    print("[1] 加载全A + 中高成交额稳定档宇宙 ...")
    df = load_panel(LOAD_START, END)
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    df = df[df["date"] >= pd.Timestamp(SIM_START)].copy()
    df["amt60_prev"] = df.groupby("code")["amount"].transform(
        lambda s: s.rolling(60, min_periods=60).mean()).groupby(df["code"]).shift(1)
    df["bkt"] = df.groupby("date")["amt60_prev"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    panel = df[df["bkt"] >= MIN_BKT].copy()
    print(f"     {len(panel):,} 行 / {panel['code'].nunique():,} 只  bkt>={MIN_BKT}")

    print("\n[2] 建三口径基准 ...")
    benches = build_benchmarks(panel)
    for k in ["broad", "median"]:
        nv = benches[k]
        yrs = len(nv) / 252
        print(f"     {k}: 终点 {nv.iloc[-1]:.3f} 年化 {(nv.iloc[-1]**(1/yrs)-1)*100:+.2f}%")

    print("\n[3] 大盘门控 ...")
    gate = build_index_gate(LOAD_START, END)
    panel = make_signals(panel, gate)
    sigs = ["G1_gate_only", "G2_tsmom", "G3_lowvol_trend", "G4_ma_bull",
            "G5_break_gate", "G6_volup_gate", "G7_pull_gate"]
    for s in sigs:
        print(f"     {s}: {int(panel[s].sum()):,}")

    out = {}
    print("\n[4] IS(2020-2024) / OOS(2025-2026) 三口径 ...")
    for s in sigs:
        a = simulate(panel, benches, s, only_before=pd.Timestamp("2025-01-01"))
        b = simulate(panel, benches, s, only_after=pd.Timestamp("2025-01-01"))
        print(f"  {s}")
        print(f"     IS  [等权] {st(a,'ex_broad')} | [中位] {st(a,'ex_median')} | [同档] {st(a,'ex_decile')}")
        print(f"     OOS [等权] {st(b,'ex_broad')} | [中位] {st(b,'ex_median')} | [同档] {st(b,'ex_decile')}")
        out[s] = {"is": {c2: float(a[c2].dropna().mean()) * 1e4 if not a.empty else None
                         for c2 in ["ex_broad", "ex_median", "ex_decile"]},
                  "oos": {c2: float(b[c2].dropna().mean()) * 1e4 if not b.empty else None
                          for c2 in ["ex_broad", "ex_median", "ex_decile"]},
                  "is_n": int(len(a)), "oos_n": int(len(b))}

    print("\n[5] 成本敏感性（同档口径, IS/OOS）...")
    for s in sigs:
        line = f"     {s:18s}: "
        for cost in [0.0025, 0.005, 0.01]:
            a = simulate(panel, benches, s, cost=cost, only_before=pd.Timestamp("2025-01-01"))
            b = simulate(panel, benches, s, cost=cost, only_after=pd.Timestamp("2025-01-01"))
            line += f"{cost*100:.2f}%: IS {a['ex_decile'].mean()*1e4:+7.1f}/OOS {b['ex_decile'].mean()*1e4:+7.1f} | "
        print(line)

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "stock_gate_signals_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
