"""W3c 涨停爆发 —— 最优配置的最终核实（集中度 + 组合 NAV + walk-forward）
=======================================================================

`w3c_limitup_recheck.py` 筛出的最优配置：
  涨停（板块自适应）+ 成交额 ≤ 自身 60 日中位数 × 0.5 + 近 20 日涨停次数 ≥ 2
  → 单笔超额（同 amount 档口径）IS +495bp / OOS +951bp

稀有信号最容易骗人的地方是**被少数几只妖股撑起来**。本脚本做最终核实：
  1. **集中度**：股票数/最大单只占比/前10只占比；剔除头部后是否还成立
  2. **随机对照**：同股票池随机入场（隔离"涨停后惯性" vs 选股能力）
  3. **逐笔削峰**：剔除最大 N 笔后均值是否还在
  4. **组合 NAV**：每日≤N 只，等权，真实可执行口径的年化超额
  5. **walk-forward**：2019-2021 / 2022 / 2023 / 2024 / 2025-2026 分段
  6. **成本三档**
  7. **规模分档 + 逐年 + 板块**
"""
import json
import os
import tempfile
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
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
    d = panel.loc[panel["dret"].notna() & (panel["dret"].abs() < 0.5), ["date", "dret", "bkt"]].copy()
    return {
        "broad": (1 + d.groupby("date")["dret"].mean()).cumprod(),
        "median": (1 + d.groupby("date")["dret"].median()).cumprod(),
        "decile": (1 + d.groupby(["date", "bkt"])["dret"].mean().unstack("bkt")).cumprod(),
    }


def limit_pct(code):
    c = code.split(".")[0]
    if c.startswith(("688", "30")):
        return 0.20
    if c.startswith(("8", "4", "9")):
        return 0.30
    return 0.10


def sim(panel, benches, sig_col, cost=0.0025, stop=-0.05, take=None, max_hold=10,
        mode="signal", seed=0, rng_pick=12):
    nav_d = {b: {ts.date(): v for ts, v in benches["decile"][b].items()} for b in benches["decile"].columns}
    rng = np.random.default_rng(seed) if mode == "random" else None
    recs = []
    for code, g in panel.groupby("code"):
        g = g.sort_values("date")
        ao = g["open"].values; ah = g["high"].values; al = g["low"].values
        ac = g["close"].values; ad = g["date"].values; ab = g["bkt"].values
        n = len(g)
        if mode == "signal":
            sig = g[sig_col].values
        else:
            sig = np.zeros(n, dtype=bool)
            feas = np.array([k for k in range(n - 2) if np.isfinite(ao[k + 1]) and ao[k + 1] > 0])
            if len(feas):
                cnt = int(g[sig_col].values.sum())
                if cnt:
                    picks = rng.choice(feas, size=min(cnt, len(feas)), replace=False)
                    sig[picks] = True
        for i in range(n):
            if i + 2 > n - 1 or not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                continue
            buy = ao[i + 1]; b = ab[i + 1]
            last_j = min(i + 1 + max_hold, n - 1); ej = last_j; ep = ac[last_j]
            reason = "timeout"
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(ac[j]) or ac[j] <= 0:
                    continue
                if stop is not None and al[j] / buy - 1 <= stop:
                    ep = buy * (1 + stop); ej = j; reason = "stop"; break
                if take is not None and ah[j] / buy - 1 >= take:
                    ep = buy * (1 + take); ej = j; reason = "tp"; break
            if not np.isfinite(ep):
                continue
            ret = ep / buy - 1 - cost
            d0 = pd.Timestamp(ad[i + 1]).date(); d1 = pd.Timestamp(ad[ej]).date()
            dm = nav_d.get(b)
            exd = np.nan
            if dm is not None and d0 in dm and d1 in dm and dm[d0] > 0:
                exd = ret - (dm[d1] / dm[d0] - 1)
            recs.append(dict(entry=pd.Timestamp(ad[i + 1]), code=code,
                             bkt=int(b) if pd.notna(b) else -1, ex=exd, ret=ret, reason=reason))
    return pd.DataFrame(recs)


def m(v, label=""):
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return f"{label}: n=0"
    mm = v.mean()
    t = mm / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0
    return (f"{label}: n={len(v):>6,} {mm*1e4:+9.2f}bp t={t:+6.2f} "
            f"中位{np.median(v)*1e4:+8.1f}bp 胜率{(v>0).mean():.1%}")


def main():
    print("[1] 加载 + 最优信号 ...")
    df = load_panel(LOAD_START, END)
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    df = df[df["date"] >= pd.Timestamp(SIM_START)].copy()
    df["amt60_prev"] = df.groupby("code")["amount"].transform(
        lambda s: s.rolling(60, min_periods=60).mean()).groupby(df["code"]).shift(1)
    df["bkt"] = df.groupby("date")["amt60_prev"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    df["lim"] = df["code"].map(limit_pct)
    df["prev_c"] = df.groupby("code")["close"].shift(1)
    df["up_ratio"] = df["close"] / df["prev_c"] - 1
    df["amt_med60"] = df.groupby("code")["amount"].transform(
        lambda s: s.rolling(60, min_periods=20).median())
    df["amt_rel"] = df["amount"] / df["amt_med60"]
    up = (df["up_ratio"] >= df["lim"] * 0.98).astype(float)
    df["up_20"] = up.groupby(df["code"]).transform(lambda s: s.rolling(20, min_periods=1).sum())
    df["sig"] = (up == 1) & (df["amt_rel"] <= 0.5) & (df["up_20"] >= 2)
    print(f"     {len(df):,} 行 / {df['code'].nunique():,} 只；信号 {int(df['sig'].sum()):,} 个信号日")
    benches = build_benchmarks(df)

    print("\n[2] 基准 ...")
    for k in ["broad", "median"]:
        nv = benches[k]
        yrs = len(nv) / 252
        print(f"     {k}: 年化 {(nv.iloc[-1]**(1/yrs)-1)*100:+.2f}%")

    print("\n[3] 主结果（止损-5%, 持有10日, 成本0.25%）...")
    tr = sim(df, benches, "sig", stop=-0.05, max_hold=10)
    out = {}
    print("     " + m(tr["ex"].values, "全期"))
    for y in range(2020, 2027):
        s = tr[tr["entry"].dt.year == y]
        if not s.empty:
            print("     " + m(s["ex"].values, f"{y}"))
    out["all"] = m(tr["ex"].values, "")

    print("\n[4] ★ 集中度检验（稀有信号最常见的坑）...")
    cc = tr["code"].value_counts()
    print(f"     涉及 {len(cc)} 只股票，最大单只 {cc.iloc[0]} 笔，"
          f"前10只占 {cc.head(10).sum()/len(tr)*100:.1f}%，前50只占 {cc.head(50).sum()/len(tr)*100:.1f}%")
    v = tr.sort_values("ex", ascending=False)["ex"].values
    print(f"     最好1笔 {v[0]*1e4:+.0f}bp，最差1笔 {v[-1]*1e4:+.0f}bp")
    print(f"     剔除最好 1/5/10/20/50 笔后的均值:")
    for k in [1, 5, 10, 20, 50]:
        vv = tr.sort_values("ex", ascending=False)["ex"].values[k:]
        print(f"       去 top{k:3d}: {vv.mean()*1e4:+8.2f}bp (n={len(vv):,})")
    print(f"     剔除最差 1/5/10/20 笔后的均值:")
    for k in [1, 5, 10, 20]:
        vv = tr.sort_values("ex")["ex"].values[:-k]
        print(f"       去 bottom{k:3d}: {vv.mean()*1e4:+8.2f}bp (n={len(vv):,})")
    # 剔除头部股票
    top50 = set(cc.head(50).index)
    v2 = tr[~tr["code"].isin(top50)]["ex"].values
    print(f"     ★剔除成交最频繁的 50 只股票后: {v2.mean()*1e4:+8.2f}bp (n={len(v2):,})")
    top200 = set(cc.head(200).index)
    v3 = tr[~tr["code"].isin(top200)]["ex"].values
    print(f"     ★剔除最频繁的 200 只后:       {v3.mean()*1e4:+8.2f}bp (n={len(v3):,})")

    print("\n[5] 随机入场对照（同股票池、同笔数、随机日期）...")
    for seed in [11, 22, 33]:
        rr = sim(df, benches, "sig", stop=-0.05, max_hold=10, mode="random", seed=seed)
        print("     " + m(rr["ex"].values, f"随机 seed{seed}"))

    print("\n[6] 组合 NAV（每日等权，最多 N 只 → 真实可执行）...")
    tr2 = tr.copy()
    tr2 = tr2.sort_values(["entry", "ex"], ascending=[True, False])
    tr2["rank"] = tr2.groupby("entry").cumcount()
    for cap in [1, 3, 5]:
        s = tr2[tr2["rank"] < cap]
        daily = s.groupby("entry")["ex"].mean().dropna()
        nav = (1 + daily / 252).cumprod()
        yrs = max(len(nav) / 252, 1e-9)
        ann = (nav.iloc[-1] ** (1 / yrs) - 1) * 100
        print(f"     每日≤{cap}只: 交易日 {len(nav):,} 累计超额 {(nav.iloc[-1]-1)*100:+.2f}% "
              f"(≈年化 {ann:+.2f}%) 正超额日 {(daily>0).mean():.1%} "
              f"最差月 {s.groupby(s['entry'].dt.to_period('M'))['ex'].mean().min()*100:+.1f}%")

    print("\n[7] walk-forward 分段 ...")
    for lo, hi, label in [(2020, 2021, "IS-1 2020-21"), (2022, 2022, "IS-2 2022"),
                          (2023, 2023, "IS-3 2023"), (2024, 2024, "IS-4 2024"),
                          (2025, 2026, "OOS 2025-26")]:
        s = tr[(tr["entry"].dt.year >= lo) & (tr["entry"].dt.year <= hi)]
        if not s.empty:
            print("     " + m(s["ex"].values, label))

    print("\n[8] 成本三档 ...")
    for cost in [0.0025, 0.005, 0.01]:
        aa = sim(df, benches, "sig", cost=cost, stop=-0.05, max_hold=10)
        print("     " + m(aa["ex"].values, f"成本{cost*100:.2f}%"))

    print("\n[9] 止损 / 持有期 / 成交额阈值 联合成簇 ...")
    print("     stop × hold (amt<=0.5, 连板>=2):")
    for stop in [-0.03, -0.05, -0.08]:
        row = []
        for mh in [5, 10, 20]:
            aa = sim(df, benches, "sig", stop=stop, max_hold=mh)
            row.append(f"hold{mh}:{aa['ex'].mean()*1e4:+8.1f}")
        print(f"     stop{stop*100:>3.0f}%: " + " | ".join(row))
    print("     amt_rel 阈值 (stop=-5%, hold=10):")
    for thr in [0.3, 0.5, 0.7, 1.0]:
        df["sig_t"] = (up == 1) & (df["amt_rel"] <= thr) & (df["up_20"] >= 2)
        aa = sim(df, benches, "sig_t", stop=-0.05, max_hold=10)
        print(f"     amt<={thr:.1f}: n={len(aa):>6,} {aa['ex'].mean()*1e4:+9.2f}bp")

    print("\n[10] 规模分档 ...")
    for k in range(10):
        s = tr[tr["bkt"] == k]
        if not s.empty:
            print(f"     bkt{k}: {m(s['ex'].values, '')}")

    print("\n[11] 收益结构 ...")
    v = tr["ex"].dropna().values
    win = v[v > 0]; loss = v[v <= 0]
    print(f"     盈利单均 {win.mean()*1e4:+.1f}bp(n={len(win):,}) 亏损单均 {loss.mean()*1e4:+.1f}bp(n={len(loss):,}) "
          f"盈亏比 {abs(win.mean()/loss.mean()):.2f}:1  胜率 {(v>0).mean():.1%}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "w3c_final_out.json")
    with open(path, "w") as f:
        json.dump({"mean_bp": float(np.nanmean(v)) * 1e4,
                   "median_bp": float(np.nanmedian(v)) * 1e4,
                   "win": float((v > 0).mean()), "n": int(len(v))}, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()