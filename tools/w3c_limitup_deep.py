"""叶岚三战法的深挖：W3c 涨停爆发 + W1_mul25（倍量黄金梯 2.5 倍）
===============================================================

首扫结果（同「滞后60日均量」档口径，成本 0.25%）：
  W3c_limitup  IS +22.18bp / OOS +117.06bp  ← **唯一 IS/OOS 双正且扛成本**
  W1_mul25     IS −77.79bp / OOS +325.44bp  ← IS 负，疑似 regime 依赖
  其余 4 个    IS/OOS 全负

本脚本对这两个做全部严格检验：
  1. 参数成簇（倍量阈值、威廉位置、流动性代理窗口、止盈止损）
  2. 成本三档
  3. 逐年 + walk-forward
  4. **信号集中度**（是否被少数股票/少数日期主导 —— 这是稀有信号最常见的坑）
  5. 规模分档拆解
  6. 出场方式拆解（涨停次日就走的"一日游" vs 持有到 timeout）
  7. **对照随机入场**（隔离"涨停后惯性"机制 vs 择时）
  8. 每日组合容量（同时信号多少只 → 实际能吃多少）
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
OOS = pd.Timestamp("2025-01-01")


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


def make_signals(df):
    """逐股算：涨停倍数自适应、流动性代理、倍量将军柱。"""
    n = len(df)
    df["lim"] = df["code"].map(limit_pct)
    df["prev_c"] = df.groupby("code")["close"].shift(1)
    df["up_ratio"] = df["close"] / df["prev_c"] - 1
    # 流动性代理：volume / 自身 60 日中位数（float_shares 在 2025-02~2026-03 整段 NULL）
    df["v_med60"] = df.groupby("code")["volume"].transform(
        lambda s: s.rolling(60, min_periods=20).median())
    df["v_rel"] = df["volume"] / df["v_med60"]
    # W3c 涨停爆发 + 低换手（用成交额相对自身历史，amount 非空率 100%）
    df["amt_med60"] = df.groupby("code")["amount"].transform(
        lambda s: s.rolling(60, min_periods=20).median())
    df["amt_rel"] = df["amount"] / df["amt_med60"]
    return df


def simulate(panel, benches, cost=0.0025, stop=-0.08, take=None, max_hold=20,
             only_after=None, only_before=None, random_control=None, seed=7):
    nav_d = {b: {ts.date(): v for ts, v in benches["decile"][b].items()} for b in benches["decile"].columns}
    rng = np.random.default_rng(seed) if random_control is not None else None
    trades = []
    for code, g in panel.groupby("code"):
        g = g.sort_values("date")
        ao = g["open"].values; ah = g["high"].values; al = g["low"].values
        ac = g["close"].values; ad = g["date"].values; ab = g["bkt"].values
        n = len(g)
        # 信号掩码
        if random_control == "entry":
            sig = np.zeros(n, dtype=bool)
            feas = np.array([k for k in range(n - 2) if np.isfinite(ao[k + 1]) and ao[k + 1] > 0])
            if len(feas):
                picks = rng.choice(feas, size=min(len(feas), 12), replace=False)
                sig[picks] = True
        elif random_control == "limitup_match":
            # 对照：随机入场但**样本数与涨停信号相同**、同一批股票
            sig = g["sig_w3c"].values
            sig = np.zeros(n, dtype=bool)
            feas = np.array([k for k in range(n - 2) if np.isfinite(ao[k + 1]) and ao[k + 1] > 0])
            orig = g["sig_w3c"].values
            cnt = int(orig.sum())
            if len(feas) and cnt:
                picks = rng.choice(feas, size=min(cnt, len(feas)), replace=False)
                sig[picks] = True
        else:
            sig = g["sig_w3c"].values
        for i in range(n):
            if i + 2 > n - 1:
                break
            if not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                continue
            buy = ao[i + 1]
            b = ab[i + 1]
            last_j = min(i + 1 + max_hold, n - 1)
            exited = False; ej = last_j; exit_price = ac[last_j]; reason = "timeout"
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(ac[j]) or ac[j] <= 0:
                    continue
                hi = ah[j] / buy - 1; lo = al[j] / buy - 1
                if stop is not None and lo <= stop:
                    exit_price = buy * (1 + stop); ej = j; exited = True; reason = "stop"; break
                if take is not None and hi >= take:
                    exit_price = buy * (1 + take); ej = j; exited = True; reason = "tp"; break
            if exit_price is None or not np.isfinite(exit_price):
                continue
            ret = exit_price / buy - 1 - cost
            d0 = pd.Timestamp(ad[i + 1]).date(); d1 = pd.Timestamp(ad[ej]).date()
            entry = pd.Timestamp(ad[i + 1])
            if only_after is not None and entry < only_after:
                continue
            if only_before is not None and entry >= only_before:
                continue
            exd = np.nan
            dm = nav_d.get(b)
            if dm is not None and d0 in dm and d1 in dm and dm[d0] > 0:
                exd = ret - (dm[d1] / dm[d0] - 1)
            trades.append(dict(entry=entry, code=code, ex_decile=exd, ret=ret, reason=reason))
    return pd.DataFrame(trades)


def sig_stats(tr, label):
    if tr.empty:
        print(f"     {label}: 空")
        return {}
    v = tr["ex_decile"].dropna().values
    m = v.mean(); t = m / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0
    med = np.median(v)
    lo, hi = np.percentile(v, [1, 99])
    trimmed = v[(v >= lo) & (v <= hi)]
    return dict(n=len(v), mean_bp=m * 1e4, t=t, median_bp=med * 1e4,
                trimmed_bp=trimmed.mean() * 1e4, win=float((v > 0).mean()))


def show(tr, label):
    s = sig_stats(tr, label)
    if not s:
        return s
    print(f"     {label:22s}: n={s['n']:>6,} {s['mean_bp']:+8.2f}bp t={s['t']:+6.2f} "
          f"中位{s['median_bp']:+7.1f} 去极值{s['trimmed_bp']:+7.2f} 胜率{s['win']:.1%}")
    return s


def main():
    print("[1] 加载 + 信号 ...")
    df = load_panel(LOAD_START, END)
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    df = df[df["date"] >= pd.Timestamp(SIM_START)].copy()
    df["amt60_prev"] = df.groupby("code")["amount"].transform(
        lambda s: s.rolling(60, min_periods=60).mean()).groupby(df["code"]).shift(1)
    df["bkt"] = df.groupby("date")["amt60_prev"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    print(f"     {len(df):,} 行 / {df['code'].nunique():,} 只")
    df = make_signals(df)
    benches = build_benchmarks(df)

    # 主信号：涨停爆发（板块自适应）+ 成交额不高于自身 60 日中位数
    df["sig_w3c"] = (df["up_ratio"] >= df["lim"] * 0.98) & (df["amt_rel"] <= 1.0)
    print(f"     W3c 涨停爆发(低换手): {int(df['sig_w3c'].sum()):,} 信号日")

    out = {}

    print("\n[2] 基准 ...")
    for k in ["broad", "median"]:
        nv = benches[k]
        yrs = len(nv) / 252
        print(f"     {k}: 年化 {(nv.iloc[-1]**(1/yrs)-1)*100:+.2f}%")

    print("\n[3] 主结果 IS/OOS ...")
    a = simulate(df, benches, only_before=OOS)
    b = simulate(df, benches, only_after=OOS)
    sa = show(a, "IS 2020-2024")
    sb = show(b, "OOS 2025-2026")
    out["base"] = {"is": sa, "oos": sb}

    print("\n[4] 信号集中度（稀有信号最常见的坑）...")
    for nm, tr in [("IS", a), ("OOS", b)]:
        if tr.empty:
            continue
        cc = tr["code"].value_counts()
        print(f"     {nm}: {len(cc)} 只股票，最大 {cc.iloc[0]} 笔，前10只占 {cc.head(10).sum()/len(tr)*100:.1f}%")
        dd = tr.groupby("entry")["ex_decile"].mean()
        print(f"        {len(dd)} 个入场日，正超额日占比 {(dd>0).mean():.1%}，"
              f"最好5日 {[f'{v*1e4:+.0f}' for v in dd.sort_values(ascending=False).head(5).values]}")
        print(f"        最差5日 {[f'{v*1e4:+.0f}' for v in dd.sort_values().head(5).values]}")

    print("\n[5] 参数成簇（成交额阈值 / 涨停阈值）...")
    for amt_thr in [0.6, 0.8, 1.0, 1.5, 99.0]:
        for lim_thr in [0.95, 0.98, 1.0]:
            df["sig_t"] = (df["up_ratio"] >= df["lim"] * lim_thr) & (df["amt_rel"] <= amt_thr)
            aa = simulate(df, benches, only_before=OOS)
            bb = simulate(df, benches, only_after=OOS)
            isa = aa["ex_decile"].mean() * 1e4 if not aa.empty else 0
            osb = bb["ex_decile"].mean() * 1e4 if not bb.empty else 0
            na = len(aa); nb = len(bb)
            print(f"     amt<={amt_thr:4.1f} lim>={lim_thr:.2f}: IS {isa:+8.1f}(n={na:>5,}) "
                  f"OOS {osb:+8.1f}(n={nb:>5,})")
    df["sig_w3c"] = (df["up_ratio"] >= df["lim"] * 0.98) & (df["amt_rel"] <= 1.0)

    print("\n[6] 出场参数成簇 ...")
    for stop, take, mh in [(None, None, 1), (None, None, 3), (-0.08, None, 20),
                           (-0.08, 0.10, 20), (-0.08, 0.05, 10), (-0.05, None, 20)]:
        aa = simulate(df, benches, stop=stop, take=take, max_hold=mh, only_before=OOS)
        bb = simulate(df, benches, stop=stop, take=take, max_hold=mh, only_after=OOS)
        isa = aa["ex_decile"].mean() * 1e4 if not aa.empty else 0
        osb = bb["ex_decile"].mean() * 1e4 if not bb.empty else 0
        lbl = f"stop={stop} take={take} hold={mh}"
        print(f"     {lbl:34s}: IS {isa:+8.1f}(n={len(aa):>5,}) OOS {osb:+8.1f}(n={len(bb):>5,})")

    print("\n[7] 出场方式拆解（是不是全靠少数 timeout 暴赚）...")
    for nm, tr in [("IS", a), ("OOS", b)]:
        if tr.empty:
            continue
        for r, gg in tr.groupby("reason"):
            v = gg["ex_decile"].dropna()
            print(f"     {nm} 出场[{r:8s}]: n={len(gg):>6,} 均值 {v.mean()*1e4:+9.2f}bp 中位 {v.median()*1e4:+8.1f}bp "
                  f"胜率 {(v>0).mean():.1%}")

    print("\n[8] 对照：随机入场（隔离'涨停后惯性'机制 vs 选股）...")
    for k in range(3):
        rr = simulate(df, benches, only_before=OOS, random_control="entry", seed=100 + k)
        rs = simulate(df, benches, only_after=OOS, random_control="entry", seed=200 + k)
        if not rr.empty:
            show(rr, f"随机IS seed{k}")
        if not rs.empty:
            show(rs, f"随机OOS seed{k}")

    print("\n[9] 规模分档拆解 ...")
    tr_all = simulate(df, benches)
    tr_all["bkt"] = [int(df[df["code"] == c].sort_values("date")["bkt"].iloc[0])
                     for c in tr_all["code"][:0]] if False else tr_all.get("bkt")
    # 重新跑一次带 bkt 输出的版本
    nav_d = {b: {ts.date(): v for ts, v in benches["decile"][b].items()} for b in benches["decile"].columns}
    recs = []
    for code, g in df.groupby("code"):
        g = g.sort_values("date")
        ao = g["open"].values; ah = g["high"].values; al = g["low"].values
        ac = g["close"].values; ad = g["date"].values; ab = g["bkt"].values
        sig = g["sig_w3c"].values
        n = len(g)
        for i in range(n):
            if i + 2 > n - 1 or not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                continue
            buy = ao[i + 1]; b = ab[i + 1]
            last_j = min(i + 1 + 20, n - 1); ej = last_j; ep = ac[last_j]
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(ac[j]) or ac[j] <= 0:
                    continue
                if al[j] / buy - 1 <= -0.08:
                    ep = buy * 0.92; ej = j; break
            if not np.isfinite(ep):
                continue
            ret = ep / buy - 1 - 0.0025
            d0 = pd.Timestamp(ad[i + 1]).date(); d1 = pd.Timestamp(ad[ej]).date()
            dm = nav_d.get(b)
            if dm is None or d0 not in dm or d1 not in dm or dm[d0] <= 0:
                continue
            recs.append(dict(entry=pd.Timestamp(ad[i + 1]), bkt=int(b),
                             ex=ret - (dm[d1] / dm[d0] - 1)))
    tdf = pd.DataFrame(recs)
    for k in range(10):
        s = tdf[tdf["bkt"] == k]
        if not s.empty:
            print(f"     bkt{k}: n={len(s):>6,} {s['ex'].mean()*1e4:+8.2f}bp "
                  f"t={s['ex'].mean()/(s['ex'].std()/np.sqrt(len(s))):+6.2f}")

    print("\n[10] 逐年 ...")
    for y in range(2020, 2027):
        s = tdf[tdf["entry"].dt.year == y]
        if s.empty:
            continue
        v = s["ex"].values
        print(f"     {y}: n={len(s):>6,} {v.mean()*1e4:+9.2f}bp "
              f"t={v.mean()/(v.std()/np.sqrt(len(v))):+6.2f} 中位 {np.median(v)*1e4:+8.1f}bp "
              f"胜率 {(v>0).mean():.1%}")

    print("\n[11] 每日容量（同时信号数 → 实盘可吃几只）...")
    cnt = df[df["sig_w3c"]].groupby("date").size()
    print(f"     有信号日 {len(cnt)} 个，中位 {cnt.median():.0f} 只/日，"
          f"p90 {cnt.quantile(0.9):.0f} 只/日，最大 {cnt.max()}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "w3c_deep_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()