"""为什么股票（通达信）战法在量化中系统性失效 —— 定量归因
==========================================================

用 F4「回踩MA20」——**历史上表现最好的战法**——做解剖，
把"失效"拆成 4 个可量化的机制，全部用真实数据说话：

  M1 信号密度：技术信号是否"人人都发"（扎堆 → 等于买市场，无选股力）
  M2 毛边 vs 成本：毛超额 vs 年化换手成本，看边是否被成本吃光
  M3 风格寄生：正超额是否只在极端规模档（滞后60日amount分档，非当日）
  M4 行业寄生：正超额是否只在少数行业（SW1）

规则与 f4_bench_compare.py 一致（T+1开盘买、≤8%止损/+10%止盈/最长20日）。
基准用 **买入持有同池等权**（2026-10-13 确认的正确基准），非"每日横截面复利"。
"""
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
STOP_LOSS, TAKE_PROFIT, MAX_HOLD = -0.08, 0.10, 20
LOAD_START, SIM_START, END = "2018-01-01", "2020-01-02", "2024-12-31"
COSTS = [0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02]


def main():
    c = psycopg2.connect(**DB)
    c.autocommit = True
    cur = c.cursor()
    cur.execute("set max_parallel_workers_per_gather=0")
    cur.execute(
        """select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close, b.volume, b.amount
           from bars_1d b where b.trade_date between %s and %s and b.asset_type='stock' and b.close>0
           order by b.asset_code, b.trade_date""", (LOAD_START, END))
    rows = cur.fetchall()
    # 行业映射
    cur.execute("""select distinct on (asset_code) asset_code, sector_code from sector_membership
                   where sector_code like 'SW1%' and sector_code not like '%加权%'
                   order by asset_code, start_date desc""")
    imap = {r[0]: r[1] for r in cur.fetchall()}
    c.close()

    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code")
    df["n"] = g.cumcount() + 1
    df["ret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    df["sma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["sma20"] = df.groupby("code")["sma20"].transform(lambda s: s.ffill(limit=5))
    df["prev_close"] = g["close"].shift(1)
    df["F4"] = ((df["close"] - df["sma20"]).abs() / df["sma20"] < 0.03) & (df["close"] > df["sma20"]) \
               & (df["prev_close"] <= g["sma20"].shift(1))
    # 滞后 60 日平均 amount（避免当日分档的未来函数）
    df["amt60"] = g["amount"].transform(lambda s: s.rolling(60, min_periods=20).mean().shift(1))
    df["industry"] = df["code"].map(imap)

    p = df[(df["n"] >= 250) & (df["open"] > 0) & (df["date"] >= SIM_START)].copy()

    # ---- M1 信号密度 ----
    daily = p.groupby("date").agg(n_stock=("code", "size"), n_sig=("F4", "sum"))
    daily["density"] = daily["n_sig"] / daily["n_stock"]
    print("=" * 68)
    print("M1 信号密度（F4）")
    print(f"  交易日 {len(daily)}  日均可用股票 {daily['n_stock'].mean():.0f}")
    print(f"  日均信号数 {daily['n_sig'].mean():.0f}  日均密度 {daily['density'].mean():.2%}"
          f"  密度中位 {daily['density'].median():.2%}")
    print(f"  单日密度>30% 的天数占比 {(daily['density']>0.30).mean():.1%}"
          f"   >50% 占比 {(daily['density']>0.50).mean():.1%}")

    # ---- 交易模拟（多成本）----
    # 买入持有等权基准 NAV（固定池 = 首日已有标的）
    d0 = p["date"].min()
    base = p[p["date"] == d0]["code"].unique()
    piv = p[p["code"].isin(base)].pivot_table(index="date", columns="code", values="close").ffill()
    nav_bh = (piv / piv.apply(lambda s: s.dropna().iloc[0])).mean(axis=1)
    nav_bh_d = {ts.date(): v for ts, v in nav_bh.items()}

    trades = []
    for code, gg in p.groupby("code"):
        gg = gg.sort_values("date")
        ao, ah, al, ac = gg["open"].values, gg["high"].values, gg["low"].values, gg["close"].values
        sig, ad, ab, ai = gg["F4"].values, gg["date"].values, gg["amt60"].values, gg["industry"].values
        n = len(gg)
        for i in range(n):
            if i + 2 > n - 1 or not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                continue
            buy = ao[i + 1]
            last_j = min(i + 1 + MAX_HOLD, n - 1)
            ej, exit_price = last_j, ac[last_j]
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(ac[j]) or ac[j] <= 0:
                    continue
                if al[j] / buy - 1 <= STOP_LOSS:
                    exit_price, ej = buy * (1 + STOP_LOSS), j
                    break
                if ah[j] / buy - 1 >= TAKE_PROFIT:
                    exit_price, ej = buy * (1 + TAKE_PROFIT), j
                    break
            gross = exit_price / buy - 1
            dd0, dd1 = pd.Timestamp(ad[i + 1]).date(), pd.Timestamp(ad[ej]).date()
            bench = (nav_bh_d[dd1] / nav_bh_d[dd0] - 1) if (dd0 in nav_bh_d and dd1 in nav_bh_d and nav_bh_d[dd0] > 0) else np.nan
            trades.append(dict(entry=pd.Timestamp(ad[i + 1]), gross=gross, bench=bench,
                               hold=(pd.Timestamp(ad[ej]) - pd.Timestamp(ad[i + 1])).days,
                               amt60=ab[i + 1], industry=ai[i + 1]))
    tr = pd.DataFrame(trades)
    tr = tr[tr["bench"].notna()].copy()
    avg_hold = tr["hold"].mean()
    turns_per_year = 242.0 / avg_hold if avg_hold > 0 else np.nan
    print("\n" + "=" * 68)
    print(f"M2 毛边 vs 成本（{len(tr):,} 笔）")
    print(f"  平均持有 {avg_hold:.1f} 天 → 年换手 ≈ {turns_per_year:.1f} 次（单边）")
    print(f"  毛收益均值 {tr['gross'].mean()*1e4:+.2f}bp  基准 {tr['bench'].mean()*1e4:+.2f}bp")
    gross_ex = tr["gross"] - tr["bench"]
    print(f"  **毛超额 {gross_ex.mean()*1e4:+.2f}bp/笔 → 年化毛超额 {gross_ex.mean()*turns_per_year*100:+.2f}%**")
    print(f"  年化换手成本 = {turns_per_year:.1f} × 单边费率（含冲击）")
    print(f"  {'费率(往返)':>10s} {'净超额bp/笔':>12s} {'年化净超额':>11s} {'t':>7s} {'胜率':>7s}")
    for cst in COSTS:
        ex = gross_ex - cst
        t = ex.mean() / (ex.std() / np.sqrt(len(ex))) if ex.std() > 0 else 0
        print(f"  {cst*100:>9.2f}% {ex.mean()*1e4:>12.2f} {ex.mean()*turns_per_year*100:>10.2f}% {t:>7.2f} {(ex>0).mean():>6.1%}")

    # ---- M3 规模档寄生（滞后60日 amount 十分位）----
    v = tr[tr["amt60"].notna()].copy()
    v["bk"] = pd.qcut(v["amt60"].rank(method="first"), 10, labels=False)
    ex = (v["gross"] - v["bench"])
    print("\n" + "=" * 68)
    print("M3 风格寄生：超额按【滞后60日 amount 十分位】（0=最小额，9=最大额）")
    print(f"  {'档':>3s} {'笔数':>8s} {'毛超额bp':>10s} {'t':>7s}")
    for bk, sub in v.groupby("bk"):
        e = sub["gross"] - sub["bench"]
        t = e.mean() / (e.std() / np.sqrt(len(e))) if e.std() > 0 else 0
        print(f"  {int(bk):>3d} {len(sub):>8,} {e.mean()*1e4:>10.1f} {t:>7.2f}")
    lo3 = v[v["bk"] <= 2]
    e_lo = lo3["gross"] - lo3["bench"]
    e_hi = v[v["bk"] >= 5]
    e_hi = e_hi["gross"] - e_hi["bench"]
    print(f"  最低3档超额 {e_lo.mean()*1e4:+.1f}bp | 中高7档超额 {e_hi.mean()*1e4:+.1f}bp"
          f"  → 极端档贡献占比 {(e_lo.sum()/ (v['gross']-v['bench']).sum())*100:.0f}%")

    # ---- M4 行业寄生 ----
    print("\n" + "=" * 68)
    print("M4 行业寄生：超额按 SW1 行业（按超额排序，前后各 5）")
    v2 = tr[tr["industry"].notna()].copy()
    v2["ex"] = v2["gross"] - v2["bench"]
    agg = v2.groupby("industry")["ex"].agg(["mean", "size"])
    agg = agg[agg["size"] >= 300].sort_values("mean", ascending=False)
    for nm, r in pd.concat([agg.head(5), agg.tail(5)]).iterrows():
        print(f"  {nm:>16s} n={int(r['size']):>6,} 超额 {r['mean']*1e4:>+8.1f}bp")
    print(f"  行业数 {len(agg)}  超额>0 的行业 {int((agg['mean']>0).sum())} 个"
          f"（占 {(agg['mean']>0).mean():.0%}）")


if __name__ == "__main__":
    main()
