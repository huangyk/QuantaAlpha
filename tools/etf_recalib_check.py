"""ETF 底盘的再校准：为什么我得到 7.6%/Sharpe0.82，而报告是 10.0%/Sharpe1.03？
================================================================================

第一版实现有两个 bug/偏差：
  bug1：信号日用 month_last[pm]，而 pm 是"上一个交易日的月份"——
        在跨月时会反复取同一个信号日，且**漏掉了真正的月末日**
        （月末日当天 month_first==d 不成立，因为月末日本身就是当月最后一天，
         它的信号要下个月才用；但当月第一个交易日 prev 在上月，
         sig_date = month_last[上月] 正确。真正的问题在下面的 bug2）
  bug2（关键）：`by_date` 用 groupby("date") 后 set_index("code")，
        但 `held` 来自 `sub`（已 set_index），我用 `w.items()` 修正了索引，
        可是 `sub["close"] > sub["sma200"]` 里的 close/sma200 是**当日值**，
        而信号日应为"上月末"——这部分逻辑对。
        真正偏差：**我用的池是 36 只（剔 5 只坏），报告是 34 只**；
        且报告的"等权持有基线 10.9%"与我算的 9.80% 也不同 → 池不同。

本脚本重新实现，逐项对齐报告口径，并明确列出每一处口径选择：
  1. 池：仅 qmt_local_86400 来源（干净 QMT 回填），剔 5 只坏 → 34 只
  2. 信号：上月末收盘 vs SMA200，次月首个交易日生效
  3. 权重：逆波动率（60日年化，下限0.05），持有子集内归一
  4. 成本：单边万1，按调仓换手扣
  5. 收益：close-to-close
  6. 输出：等权/中位数/同 amount 档三口径基准 + 逐年 + walk-forward + 参数成簇
"""
import os
import tempfile
import json
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
START = "2019-01-02"
END = "2026-09-30"
BAD = {"159915.SZ", "159919.SZ", "159922.SZ", "159949.SZ", "510050.SH"}


def load_etf(start, end, only_clean_source=True):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    if only_clean_source:
        cur.execute(
            """
            select asset_code, trade_date, open, high, low, close, volume, amount
            from bars_1d
            where trade_date between %s and %s and asset_type = 'etf'
              and source = 'qmt_local_86400'
            order by asset_code, trade_date
            """,
            (start, end),
        )
    else:
        cur.execute(
            """
            select asset_code, trade_date, open, high, low, close, volume, amount
            from bars_1d
            where trade_date between %s and %s and asset_type = 'etf'
            order by asset_code, trade_date
            """,
            (start, end),
        )
    rows = cur.fetchall()
    cur.close()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def perf(nav):
    r = nav.pct_change().dropna()
    if len(r) < 2:
        return {}
    total = nav.iloc[-1]
    years = len(nav) / 252
    ann = total ** (1 / years) - 1 if total > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    return dict(total=float(total), ann=float(ann), vol=float(vol),
                sharpe=float(sharpe), mdd=float(dd), n_days=int(len(nav)))


def run_absmom(df, sma_win=200, vol_win=60, comm=0.0001, vol_floor=0.05):
    """月末信号 → 次月首个交易日调仓（扣单边佣金 comm × 换手）。"""
    d = df.copy()
    g = d.groupby("code")
    d["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    d["sma"] = g["close"].transform(lambda s: s.rolling(sma_win, min_periods=sma_win).mean())
    d["vol"] = g["dret"].transform(lambda s: s.rolling(vol_win, min_periods=vol_win).std() * np.sqrt(252))
    dates = [pd.Timestamp(x) for x in sorted(d["date"].unique())]
    by_date = {dt: grp.set_index("code") for dt, grp in d.groupby("date")}
    # 每月最后一个交易日
    month_last = {}
    for dt in dates:
        month_last[dt.to_period("M")] = dt
    nav = 1.0
    navs = []
    cur_w = {}
    n_rebal = 0
    turnover_sum = 0.0
    for i, dt in enumerate(dates):
        m = dt.to_period("M")
        # 当月首个交易日：用上月末信号调仓
        is_month_start = (i == 0) or (dates[i - 1].to_period("M") != m)
        if is_month_start:
            prev_m = dates[i - 1].to_period("M") if i > 0 else None
            if prev_m is not None:
                sig = by_date.get(month_last[prev_m])
                if sig is not None:
                    sig = sig[sig["sma"].notna() & sig["vol"].notna()]
                    held = sig[sig["close"] > sig["sma"]]
                    if held.empty:
                        new_w = {}
                    else:
                        inv = 1.0 / held["vol"].clip(lower=vol_floor)
                        w = inv / inv.sum()
                        new_w = {c2: float(v) for c2, v in w.items()}
                    codes = set(new_w) | set(cur_w)
                    turnover = sum(abs(new_w.get(c2, 0.0) - cur_w.get(c2, 0.0)) for c2 in codes)
                    turnover_sum += turnover
                    nav *= (1 - comm * turnover)
                    cur_w = new_w
                    n_rebal += 1
        sub = by_date.get(dt)
        day_ret = 0.0
        if cur_w and sub is not None:
            for code, wt in cur_w.items():
                if code in sub.index:
                    r = sub.loc[code, "dret"]
                    if pd.notna(r):
                        day_ret += wt * r
        nav *= (1 + day_ret)
        navs.append((dt, nav))
    s = pd.Series(dict(navs))
    return s, n_rebal, turnover_sum


def main():
    print("[1] 加载 ETF（仅 qmt_local_86400 干净源）...")
    df = load_etf(START, END, only_clean_source=True)
    df = df[~df["code"].isin(BAD)].copy()
    g = df.groupby("code")
    df["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    n_bad_ret = int((df["dret"].abs() > 0.30).sum())
    print(f"     {len(df):,} 行 / {df['code'].nunique()} 只  "
          f"{df['date'].min().date()}~{df['date'].max().date()}  |单日|涨跌|>30% 的行: {n_bad_ret}")
    print(f"     池: {sorted(df['code'].unique())}")

    print("\n[2] 底盘（SMA200 + 逆波动率, 万1, 月末信号次月调仓）...")
    nav, nr, to = run_absmom(df, comm=0.0001)
    p = perf(nav)
    print(f"     年化 {p['ann']*100:+.2f}%  波动 {p['vol']*100:.2f}%  夏普 {p['sharpe']:.2f}  "
          f"回撤 {p['mdd']*100:+.1f}%  调仓{nr}次  累计换手{to:.1f}")

    d = df[df["dret"].notna() & (df["dret"].abs() < 0.5)]
    nav_eq = (1 + d.groupby("date")["dret"].mean()).cumprod()
    nav_md = (1 + d.groupby("date")["dret"].median()).cumprod()
    d2 = d.copy()
    d2["amt_dec"] = d2.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    nav_lo = (1 + d2[d2["amt_dec"] == 0].groupby("date")["dret"].mean()).cumprod()
    nav_hi = (1 + d2[d2["amt_dec"] == 9].groupby("date")["dret"].mean()).cumprod()
    print("\n[3] 四口径基准 ...")
    for k, v in [("等权", nav_eq), ("中位数", nav_md), ("最低amount档", nav_lo), ("最高amount档", nav_hi)]:
        pp = perf(v)
        print(f"     {k:12s}: 年化 {pp['ann']*100:+6.2f}%  波动 {pp['vol']*100:5.2f}%  "
              f"夏普 {pp['sharpe']:.2f}  回撤 {pp['mdd']*100:+.1f}%")

    print("\n[4] 逐年（底盘 vs 等权 vs 中位数）...")
    for y in range(2019, 2027):
        a = nav[nav.index.year == y]
        e = nav_eq[nav_eq.index.year == y]
        mm = nav_md[nav_md.index.year == y]
        if len(a) < 2 or len(e) < 2:
            continue
        rm = (mm.iloc[-1] / mm.iloc[0] - 1) if len(mm) >= 2 else float("nan")
        print(f"     {y}: 底盘 {a.iloc[-1]/a.iloc[0]-1:+7.2%} | 等权 {e.iloc[-1]/e.iloc[0]-1:+7.2%} | "
              f"中位数 {rm:+7.2%}")

    print("\n[5] walk-forward（2019-2022 训练 / 2023-2026 测试）...")
    wf = pd.Timestamp("2023-01-01")
    for label, v in [("底盘", nav), ("等权", nav_eq), ("中位数", nav_md)]:
        tr = perf(v[v.index < wf])
        te = perf(v[v.index >= wf])
        print(f"     {label}: 训练 {tr['ann']*100:+6.2f}%/夏普{tr['sharpe']:.2f}/回撤{tr['mdd']*100:+6.1f}% "
              f"|| 测试 {te['ann']*100:+6.2f}%/夏普{te['sharpe']:.2f}/回撤{te['mdd']*100:+6.1f}%")

    print("\n[6] 参数成簇（SMA 窗口 × 波动窗口）...")
    for sw in [150, 180, 200, 220, 250]:
        row = []
        for vw in [40, 60, 90]:
            nv, _, _ = run_absmom(df, sma_win=sw, vol_win=vw, comm=0.0001)
            pp = perf(nv)
            row.append(f"vw{vw}: {pp['ann']*100:+6.2f}%/S{pp['sharpe']:.2f}")
        print(f"     SMA{sw:3d}: " + " | ".join(row))

    print("\n[7] 成本敏感性 ...")
    for comm in [0.0, 0.0001, 0.0005, 0.001]:
        nv, _, _ = run_absmom(df, comm=comm)
        pp = perf(nv)
        print(f"     单边佣金 {comm*10000:.0f}bp: 年化 {pp['ann']*100:+6.2f}%  夏普 {pp['sharpe']:.2f}  "
              f"回撤 {pp['mdd']*100:+.1f}%")

    print("\n[8] 暴露档（60%/50%）...")
    for ex in [1.0, 0.6, 0.5]:
        r = nav.pct_change().fillna(0.0) * ex
        nv = (1 + r).cumprod()
        nv.iloc[0] = 1.0
        pp = perf(nv)
        print(f"     暴露 {ex*100:.0f}%: 年化 {pp['ann']*100:+5.2f}%  波动 {pp['vol']*100:.2f}%  "
              f"夏普 {pp['sharpe']:.2f}  回撤 {pp['mdd']*100:+.1f}%")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_recalib_out.json")
    res = {"absmom": p, "bench_equal": perf(nav_eq), "bench_median": perf(nav_md),
           "bench_amt_lo": perf(nav_lo), "bench_amt_hi": perf(nav_hi),
           "n_rebal": nr, "turnover_sum": float(to), "pool_size": int(df["code"].nunique())}
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
