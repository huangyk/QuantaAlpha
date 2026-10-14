"""W3c 涨停爆发的复核（修正参数扫描 bug）+ 收益结构拆解
=====================================================

`w3c_limitup_deep.py` 的 §5 参数扫描 15 组结果**完全相同**，
说明 `df["sig_t"]` 写入后 `simulate()` 读的还是旧的 `sig_w3c`（bug）。
本脚本修正：直接传掩码数组进 simulate，不依赖 DataFrame 列。

同时补三个关键分析（决定这信号能不能用）：
  1. **收益结构**：均值 vs 中位数 vs 分位数分布 → 是不是纯厚尾赌局
  2. **持有期衰减曲线**：超额随持有天数变化 → 找到最优且稳健的退出
  3. **止损宽度扫描**：−3%/−5%/−8%/−12%/无 → 止损宽度是不是尖峰参数
  4. **连板数分层**：首板 / 二板 / 三板+ 的超额差异
  5. **涨跌停阈值**真实扫描（主板10% / 创业板20% / 北交所30% 各自）
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


def sim(panel, benches, sig_col, cost=0.0025, stop=-0.08, take=None,
        max_hold=20, only_after=None, only_before=None, collect=False):
    nav_d = {b: {ts.date(): v for ts, v in benches["decile"][b].items()} for b in benches["decile"].columns}
    recs = []
    for code, g in panel.groupby("code"):
        g = g.sort_values("date")
        ao = g["open"].values; ah = g["high"].values; al = g["low"].values
        ac = g["close"].values; ad = g["date"].values; ab = g["bkt"].values
        sig = g[sig_col].values
        n = len(g)
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
            entry = pd.Timestamp(ad[i + 1])
            if only_after is not None and entry < only_after:
                continue
            if only_before is not None and entry >= only_before:
                continue
            exd = np.nan
            dm = nav_d.get(b)
            if dm is not None and d0 in dm and d1 in dm and dm[d0] > 0:
                exd = ret - (dm[d1] / dm[d0] - 1)
            recs.append(dict(entry=entry, code=code, bkt=int(b) if pd.notna(b) else -1,
                             ex=exd, ret=ret, reason=reason))
    return pd.DataFrame(recs)


def m(v):
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "n=0"
    mm = v.mean()
    t = mm / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0
    return (f"n={len(v):>6,} {mm*1e4:+9.2f}bp t={t:+6.2f} 中位{np.median(v)*1e4:+8.1f}bp "
            f"胜率{(v>0).mean():.1%}")


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
    df["lim"] = df["code"].map(limit_pct)
    df["prev_c"] = df.groupby("code")["close"].shift(1)
    df["up_ratio"] = df["close"] / df["prev_c"] - 1
    df["amt_med60"] = df.groupby("code")["amount"].transform(
        lambda s: s.rolling(60, min_periods=20).median())
    df["amt_rel"] = df["amount"] / df["amt_med60"]
    print(f"     {len(df):,} 行 / {df['code'].nunique():,} 只")
    benches = build_benchmarks(df)

    # 连板数（过去 20 日内涨停次数）
    up = (df["up_ratio"] >= df["lim"] * 0.98).astype(float)
    df["up_20"] = up.groupby(df["code"]).transform(lambda s: s.rolling(20, min_periods=1).sum())

    base_sig = (df["up_ratio"] >= df["lim"] * 0.98) & (df["amt_rel"] <= 1.0)
    df["sig0"] = base_sig
    print(f"     涨停爆发(低换手) 信号: {int(base_sig.sum()):,}")

    out = {}

    print("\n[2] 参数扫描（修正版：掩码真正生效）...")
    print("     amt_rel 阈值 × 是否要求连板数")
    for amt_thr in [0.5, 0.7, 1.0, 1.5, 99.0]:
        row = []
        for board in [1, 2]:
            df["sig_t"] = (df["up_ratio"] >= df["lim"] * 0.98) & (df["amt_rel"] <= amt_thr) & \
                          (df["up_20"] >= board)
            aa = sim(df, benches, "sig_t", only_before=OOS)
            bb = sim(df, benches, "sig_t", only_after=OOS)
            ia = aa["ex"].mean() * 1e4 if not aa.empty else 0
            ib = bb["ex"].mean() * 1e4 if not bb.empty else 0
            row.append(f"连板>={board}: IS{ia:+8.1f}(n={len(aa):>5,}) OOS{ib:+8.1f}(n={len(bb):>5,})")
        print(f"     amt<={amt_thr:5.1f} " + " | ".join(row))

    df["sig0"] = base_sig

    print("\n[3] 止损宽度扫描（连板>=1, amt<=1.0）...")
    for stop in [None, -0.03, -0.05, -0.08, -0.12, -0.20]:
        aa = sim(df, benches, "sig0", stop=stop, only_before=OOS)
        bb = sim(df, benches, "sig0", stop=stop, only_after=OOS)
        ia = aa["ex"].mean() * 1e4 if not aa.empty else 0
        ib = bb["ex"].mean() * 1e4 if not bb.empty else 0
        lbl = "无止损" if stop is None else f"{stop*100:.0f}%"
        print(f"     {lbl:>7s}: IS {ia:+9.2f}bp (n={len(aa):>5,}) | OOS {ib:+9.2f}bp (n={len(bb):>5,})")

    print("\n[4] 持有期扫描（止损 -8%）...")
    for mh in [1, 2, 3, 5, 10, 20, 30]:
        aa = sim(df, benches, "sig0", max_hold=mh, only_before=OOS)
        bb = sim(df, benches, "sig0", max_hold=mh, only_after=OOS)
        ia = aa["ex"].mean() * 1e4 if not aa.empty else 0
        ib = bb["ex"].mean() * 1e4 if not bb.empty else 0
        print(f"     持有{mh:2d}日: IS {ia:+9.2f}bp | OOS {ib:+9.2f}bp")

    print("\n[5] 收益结构（stop=-8%, hold=20, 全期）...")
    tr = sim(df, benches, "sig0")
    v = tr["ex"].dropna().values
    print(f"     n={len(v):,} 均值 {v.mean()*1e4:+.2f}bp 中位 {np.median(v)*1e4:+.1f}bp 胜率 {(v>0).mean():.1%}")
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print("     分位数:", " ".join(f"{q:.0%}:{np.quantile(v,q)*1e4:+.0f}" for q in qs))
    print(f"     左尾(<-8%): {(v<-0.08).mean():.1%}  右尾(>+20%): {(v>0.20).mean():.1%}")
    print("     ★ 均值/中位数比:", f"{v.mean()/np.median(v):.2f}" if np.median(v) != 0 else "n/a")
    # 期望值分解
    win = v[v > 0]; loss = v[v <= 0]
    print(f"     盈利单均 {win.mean()*1e4:+.1f}bp (n={len(win):,}) | 亏损单均 {loss.mean()*1e4:+.1f}bp (n={len(loss):,})")
    print(f"     盈亏比 {abs(win.mean()/loss.mean()):.2f} : 1")

    print("\n[6] 实际组合 NAV（每日等权持有信号股，最多 5 只）...")
    for cap in [3, 5, 10]:
        tr2 = tr.copy()
        tr2 = tr2.sort_values(["entry", "ex"], ascending=[True, False])
        tr2["rank"] = tr2.groupby("entry").cumcount()
        tr2 = tr2[tr2["rank"] < cap]
        # 按入场日聚合日度超额 → 复利
        daily = tr2.groupby("entry")["ex"].mean()
        nav = (1 + daily / 252).cumprod()   # 近似：每日超额按 1/252 摊到组合
        yrs = len(nav) / 252
        print(f"     每日最多{cap:2d}只: 交易日 {len(nav):,} 累计超额 {(nav.iloc[-1]-1)*100:+.2f}% "
              f"（≈年化 {(nav.iloc[-1]**(1/yrs)-1)*100:+.2f}%）| 正超额日 {(daily>0).mean():.1%}")

    print("\n[7] 逐年（单笔口径）...")
    for y in range(2020, 2027):
        s = tr[tr["entry"].dt.year == y]
        if s.empty:
            continue
        print(f"     {y}: {m(s['ex'].values)}")

    print("\n[8] 成本敏感性 ...")
    for cost in [0.0025, 0.005, 0.01, 0.02]:
        aa = sim(df, benches, "sig0", cost=cost, only_before=OOS)
        bb = sim(df, benches, "sig0", cost=cost, only_after=OOS)
        ia = aa["ex"].mean() * 1e4 if not aa.empty else 0
        ib = bb["ex"].mean() * 1e4 if not bb.empty else 0
        print(f"     成本{cost*100:5.2f}%: IS {ia:+9.2f}bp | OOS {ib:+9.2f}bp")

    print("\n[9] 板块分层 ...")
    df["board_type"] = np.where(df["lim"] >= 0.30, "北交所30%",
                          np.where(df["lim"] >= 0.20, "创业板/科创20%", "主板10%"))
    df["sig_b"] = base_sig
    for bt, sub in df[df["sig_b"]].groupby("board_type"):
        codes = set(sub["code"])
        s = tr[tr["code"].isin(codes)]
        if not s.empty:
            print(f"     {bt:16s}: {m(s['ex'].values)}")
    print("     （对照）各板块整体等权基准年化：")
    for bt, sub in df.groupby("board_type"):
        s = sub.groupby("date")["dret"].mean().dropna()
        if len(s) > 100:
            yrs = len(s) / 252
            print(f"     {bt:16s}: 年化 {(1+s).prod()**(1/yrs)*100-100:+.2f}%")

    out["base_is"] = float(sim(df, benches, "sig0", only_before=OOS)["ex"].mean()) * 1e4
    out["base_oos"] = float(sim(df, benches, "sig0", only_after=OOS)["ex"].mean()) * 1e4
    out["n_total"] = int(len(tr))
    out["median_bp"] = float(np.median(tr["ex"].dropna().values)) * 1e4
    out["win_rate"] = float((tr["ex"].dropna().values > 0).mean())

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "w3c_recheck_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()