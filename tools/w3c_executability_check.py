"""W3c 涨停爆发 —— 可执行性 + 集中度 终审
========================================

上一轮 `w3c_final_verify.py` 有两个必须修的问题：

**问题1（致命）：可执行性未校验**
信号是"缩量涨停"（低成交额+连板），这类票 T+1 经常**一字板开盘**，
实际**买不进去**。项目铁律明确要求"剔除涨停买不进（open/昨收≥1.098）"。
原 simulate() 没加这个过滤 → 若收益主要来自买不进的一字板，信号是假的。

**问题2：集中度检验被 NaN 污染**
`tr.sort_values(...)["ex"].values[k:].mean()` 遇 NaN 返回 NaN，
导致"剔除 top N 笔"全部显示 +nanbp。

本脚本重做：
  1. **T+1 开盘可买性过滤**：剔除 T+1 开盘价 ≥ 前收 ×(1+涨停幅×0.98) 的样本
     （即开盘就封涨停，买不进）
  2. 再算剔除后的超额，看信号是否还成立
  3. 修复集中度检验（先 dropna 再 trim）
  4. 剔除最频繁股票后的均值
  5. 按入场日聚合看是否被少数日期主导
  6. 组合 NAV（可执行口径）
  7. 随机对照
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
        mode="signal", seed=0, filter_unbuyable=True):
    """filter_unbuyable=True: 剔除 T+1 开盘就封涨停（买不进）的样本。"""
    nav_d = {b: {ts.date(): v for ts, v in benches["decile"][b].items()} for b in benches["decile"].columns}
    rng = np.random.default_rng(seed) if mode == "random" else None
    recs = []
    n_unbuyable = 0
    for code, g in panel.groupby("code"):
        g = g.sort_values("date")
        ao = g["open"].values; ah = g["high"].values; al = g["low"].values
        ac = g["close"].values; ad = g["date"].values; ab = g["bkt"].values
        lim = float(g["lim"].iloc[0])
        n = len(g)
        if mode == "signal":
            sig = g[sig_col].values
        else:
            sig = np.zeros(n, dtype=bool)
            feas = np.array([k for k in range(n - 2) if np.isfinite(ao[k + 1]) and ao[k + 1] > 0])
            cnt = int(g[sig_col].values.sum())
            if len(feas) and cnt:
                picks = rng.choice(feas, size=min(cnt, len(feas)), replace=False)
                sig[picks] = True
        for i in range(n):
            if i + 2 > n - 1 or not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                continue
            # ★ 可执行性：T+1 开盘是否已封涨停（买不进）
            if filter_unbuyable and np.isfinite(ac[i]) and ac[i] > 0:
                if ao[i + 1] / ac[i] - 1 >= lim * 0.98:
                    n_unbuyable += 1
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
    return pd.DataFrame(recs), n_unbuyable


def stat(v, label=""):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return f"{label}: n=0"
    mm = v.mean(); t = mm / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0
    return (f"{label}: n={len(v):>6,} {mm*1e4:+9.2f}bp t={t:+6.2f} "
            f"中位{np.median(v)*1e4:+8.1f}bp 胜率{(v>0).mean():.1%}")


def main():
    print("[1] 加载 + 最优信号（缩量涨停 + 连板>=2）...")
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
    print(f"     {len(df):,} 行 / {df['code'].nunique():,} 只；信号 {int(df['sig'].sum()):,}")
    benches = build_benchmarks(df)

    print("\n[2] ★ 可执行性对比（这是决定性的）...")
    tr_all, n_un = sim(df, benches, "sig", stop=-0.05, max_hold=10, filter_unbuyable=False)
    tr_buy, n_un2 = sim(df, benches, "sig", stop=-0.05, max_hold=10, filter_unbuyable=True)
    print(f"     不过滤可买性: {stat(tr_all['ex'].values)}")
    print(f"     过滤买不进后: {stat(tr_buy['ex'].values)}")
    print(f"     → 剔除的'开盘一字板买不进'样本数: {n_un2}")
    print(f"     → 剔除占比: {n_un2/(n_un2+len(tr_buy))*100:.1f}%")

    print("\n[3] 逐年（过滤买不进后）...")
    for y in range(2020, 2027):
        s = tr_buy[tr_buy["entry"].dt.year == y]
        if not s.empty:
            print("     " + stat(s["ex"].values, f"{y}"))

    print("\n[4] ★ 集中度检验（修复 NaN 版）...")
    v = tr_buy["ex"].values
    v = v[np.isfinite(v)]
    vs = np.sort(v)[::-1]
    print(f"     n={len(v):,} 最好1笔 {vs[0]*1e4:+.0f}bp 最差1笔 {vs[-1]*1e4:+.0f}bp")
    print("     剔除最好 N 笔后:")
    for k in [1, 5, 10, 20, 50, 100]:
        if k < len(vs):
            print(f"       去 top{k:4d}: {vs[k:].mean()*1e4:+9.2f}bp (n={len(vs)-k:,})")
    print("     剔除最差 N 笔后:")
    for k in [1, 5, 10, 20, 50, 100]:
        if k < len(vs):
            print(f"       去 bottom{k:4d}: {vs[:-k].mean()*1e4:+9.2f}bp (n={len(vs)-k:,})")
    cc = tr_buy["code"].value_counts()
    print(f"     涉及 {len(cc)} 只，最大单只 {cc.iloc[0]} 笔，前10只占 {cc.head(10).sum()/len(tr_buy)*100:.1f}%")
    for k in [20, 50, 100, 200]:
        top = set(cc.head(k).index)
        sub = tr_buy[~tr_buy["code"].isin(top)]["ex"].values
        sub = sub[np.isfinite(sub)]
        print(f"     ★剔除最频繁 {k:3d} 只股票后: {sub.mean()*1e4:+9.2f}bp (n={len(sub):,})")
    # 按入场日
    dd = tr_buy.groupby("entry")["ex"].mean().dropna()
    print(f"     入场日 {len(dd)} 个，正超额日 {(dd>0).mean():.1%}")
    ds = np.sort(dd.values)[::-1]
    for k in [1, 5, 10, 20]:
        if k < len(ds):
            print(f"       去最好{k:3d}个入场日: {ds[k:].mean()*1e4:+9.2f}bp")
    # 按年剔除最好年份
    for y in range(2020, 2027):
        sub = tr_buy[tr_buy["entry"].dt.year != y]["ex"].values
        sub = sub[np.isfinite(sub)]
        print(f"     剔除 {y} 整年: {sub.mean()*1e4:+9.2f}bp (n={len(sub):,})")

    print("\n[5] 随机对照（同股票池、同笔数、同可买性过滤）...")
    for seed in [11, 22, 33, 44]:
        rr, _ = sim(df, benches, "sig", stop=-0.05, max_hold=10, mode="random", seed=seed)
        print("     " + stat(rr["ex"].values, f"随机 seed{seed}"))

    print("\n[6] 组合 NAV（可执行口径：过滤买不进 + 每日≤N 只）...")
    tr2 = tr_buy.sort_values(["entry", "ex"], ascending=[True, False]).copy()
    tr2["rank"] = tr2.groupby("entry").cumcount()
    for cap in [1, 3, 5, 10]:
        s = tr2[tr2["rank"] < cap]
        daily = s.groupby("entry")["ex"].mean().dropna()
        nav = (1 + daily / 252).cumprod()
        yrs = max(len(nav) / 252, 1e-9)
        ann = (nav.iloc[-1] ** (1 / yrs) - 1) * 100
        monthly = s.groupby(s["entry"].dt.to_period("M"))["ex"].mean()
        print(f"     每日≤{cap:2d}只: 交易日 {len(nav):,} 累计 {(nav.iloc[-1]-1)*100:+7.2f}% "
              f"(≈年化 {ann:+6.2f}%) 正超额日 {(daily>0).mean():.1%} "
              f"最差月 {monthly.min()*100:+6.1f}% 最差日 {daily.min()*100:+6.1f}%")

    print("\n[7] 成本三档（过滤买不进后）...")
    for cost in [0.0025, 0.005, 0.01, 0.02]:
        aa, _ = sim(df, benches, "sig", cost=cost, stop=-0.05, max_hold=10)
        print("     " + stat(aa["ex"].values, f"成本{cost*100:5.2f}%"))

    print("\n[8] 参数成簇（可执行口径）...")
    for stop in [-0.03, -0.05, -0.08]:
        row = []
        for mh in [5, 10, 20]:
            aa, _ = sim(df, benches, "sig", stop=stop, max_hold=mh)
            row.append(f"hold{mh}:{aa['ex'].mean()*1e4:+8.1f}(n={len(aa)})")
        print(f"     stop{stop*100:>3.0f}%: " + " | ".join(row))
    for thr in [0.3, 0.5, 0.7, 1.0]:
        df["sig_t"] = (up == 1) & (df["amt_rel"] <= thr) & (df["up_20"] >= 2)
        aa, nu = sim(df, benches, "sig_t", stop=-0.05, max_hold=10)
        print(f"     amt<={thr:.1f}: n={len(aa):>6,} 买不进剔除 {nu:>5,} {aa['ex'].mean()*1e4:+9.2f}bp")
    df["sig"] = (up == 1) & (df["amt_rel"] <= 0.5) & (df["up_20"] >= 2)

    print("\n[9] 收益结构（可执行口径）...")
    v = tr_buy["ex"].values; v = v[np.isfinite(v)]
    win = v[v > 0]; loss = v[v <= 0]
    print(f"     均值 {v.mean()*1e4:+.1f}bp 中位 {np.median(v)*1e4:+.1f}bp 胜率 {(v>0).mean():.1%}")
    print(f"     盈利单均 {win.mean()*1e4:+.1f}bp(n={len(win)}) 亏损单均 {loss.mean()*1e4:+.1f}bp(n={len(loss)}) "
          f"盈亏比 {abs(win.mean()/loss.mean()):.2f}:1")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "w3c_exec_out.json")
    with open(path, "w") as f:
        json.dump({"unfiltered": float(np.nanmean(tr_all["ex"].values)) * 1e4,
                   "filtered": float(v.mean()) * 1e4,
                   "n_unbuyable": int(n_un2), "n_trades": int(len(v)),
                   "median_bp": float(np.median(v)) * 1e4,
                   "win": float((v > 0).mean())}, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()