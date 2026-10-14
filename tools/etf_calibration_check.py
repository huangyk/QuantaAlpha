"""ETF 底盘策略的口径核对：为什么我的实现得到 18.77%/Sharpe2.19，
而 docs/strategy_absmom.md 报 10.0%/Sharpe1.03？

差异必须查清，否则结论不可信。逐项对比：
  1. 池子（我只剔了 5 只坏数据，报告用 34/38 只）
  2. 调仓成本（我目前**完全没扣成本**！）
  3. 权重再平衡（报告月度调仓=换仓，我只在信号变化时换）
  4. 数据源与复权
本脚本逐项排查并给出修正后的结果。
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
SMA_WIN = 200
VOL_WIN = 60
BAD = {"159915.SZ", "159919.SZ", "159922.SZ", "159949.SZ", "510050.SH"}


def load_etf(start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
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
                sharpe=float(sharpe), mdd=float(dd))


def run_absmom(panel, comm=0.0001, rebalance_monthly=True, min_hist=200):
    """200日SMA + 逆波动率。

    - 月末用当日收盘算信号，次月首个交易日按目标权重调仓（扣佣金 comm 单边）
    - rebalance_monthly=True：每月强制再平衡（回到目标权重）
      rebalance_monthly=False：仅在持仓集合变化时调仓
    - 收益用 close-to-close 近似（ETF 流动性好，且原报告未说明成交价假设）
    """
    df = panel.copy()
    g = df.groupby("code")
    df["sma200"] = g["close"].transform(lambda s: s.rolling(SMA_WIN, min_periods=SMA_WIN).mean())
    df["vol60"] = g["dret"].transform(lambda s: s.rolling(VOL_WIN, min_periods=VOL_WIN).std() * np.sqrt(252))
    dates = [pd.Timestamp(d) for d in sorted(df["date"].unique())]
    month_last = {}
    for d in dates:
        month_last[d.to_period("M")] = d
    month_first = {}
    for d in dates:
        m = d.to_period("M")
        if m not in month_first:
            month_first[m] = d

    by_date = {d: grp.set_index("code") for d, grp in df.groupby("date")}
    nav = 1.0
    navs = []
    cur_w = {}
    n_rebal = 0
    for i, d in enumerate(dates):
        # 当日是否为新月份首个交易日 → 按上月末信号调仓
        m = d.to_period("M")
        if month_first.get(m) == d and i > 0:
            prev = dates[i - 1]
            pm = prev.to_period("M")
            sig_date = month_last.get(pm, prev)
            sub = by_date.get(sig_date)
            if sub is not None:
                sub = sub[sub["sma200"].notna() & sub["vol60"].notna()]
                held = sub[sub["close"] > sub["sma200"]]
                if held.empty:
                    new_w = {}
                else:
                    inv = 1.0 / held["vol60"].clip(lower=0.05)
                    w = inv / inv.sum()
                    new_w = {c2: float(v) for c2, v in w.items()}
                # 换手成本（单边佣金 × 双边换手）
                codes = set(new_w) | set(cur_w)
                turnover = sum(abs(new_w.get(c2, 0.0) - cur_w.get(c2, 0.0)) for c2 in codes)
                nav *= (1 - comm * turnover)
                cur_w = new_w
                n_rebal += 1
        sub = by_date.get(d)
        day_ret = 0.0
        if cur_w and sub is not None:
            for code, wt in cur_w.items():
                if code in sub.index:
                    r = sub.loc[code, "dret"]
                    if pd.notna(r):
                        day_ret += wt * r
        nav *= (1 + day_ret)
        navs.append((d, nav))
    return pd.Series(dict(navs)), n_rebal


def main():
    print("[1] 加载 ETF ...")
    df = load_etf(START, END)
    df = df[~df["code"].isin(BAD)].copy()
    g = df.groupby("code")
    df["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    df = df[df["dret"].abs() < 0.5].copy()
    print(f"     {len(df):,} 行 / {df['code'].nunique()} 只  "
          f"{df['date'].min().date()}~{df['date'].max().date()}")

    print("\n[2] 口径差异逐项排查 ...")
    print("     A. 无成本、仅换仓时调仓（我第一版，=报告口径？）")
    nav, nr = run_absmom(df, comm=0.0, rebalance_monthly=False)
    p = perf(nav)
    print(f"        年化 {p['ann']*100:+.2f}% 夏普 {p['sharpe']:.2f} 回撤 {p['mdd']*100:+.1f}% 调仓{nr}次")

    print("     B. 无成本、每月强制再平衡")
    nav, nr = run_absmom(df, comm=0.0, rebalance_monthly=True)
    p = perf(nav)
    print(f"        年化 {p['ann']*100:+.2f}% 夏普 {p['sharpe']:.2f} 回撤 {p['mdd']*100:+.1f}% 调仓{nr}次")

    print("     C. 万1佣金、每月再平衡（报告口径）")
    nav_c, nr = run_absmom(df, comm=0.0001, rebalance_monthly=True)
    p = perf(nav_c)
    print(f"        年化 {p['ann']*100:+.2f}% 夏普 {p['sharpe']:.2f} 回撤 {p['mdd']*100:+.1f}% 调仓{nr}次")

    print("     D. 千1佣金、每月再平衡（成本压力）")
    nav_d, nr = run_absmom(df, comm=0.001, rebalance_monthly=True)
    p = perf(nav_d)
    print(f"        年化 {p['ann']*100:+.2f}% 夏普 {p['sharpe']:.2f} 回撤 {p['mdd']*100:+.1f}% 调仓{nr}次")

    # 等权基准（同池，同口径）
    d = df[df["dret"].notna()]
    nav_eq = (1 + d.groupby("date")["dret"].mean()).cumprod()
    nav_md = (1 + d.groupby("date")["dret"].median()).cumprod()
    pe = perf(nav_eq)
    pm_ = perf(nav_md)
    print(f"\n     同池等权基准: 年化 {pe['ann']*100:+.2f}% 夏普 {pe['sharpe']:.2f} 回撤 {pe['mdd']*100:+.1f}%")
    print(f"     同池中位数基准: 年化 {pm_['ann']*100:+.2f}% 夏普 {pm_['sharpe']:.2f} 回撤 {pm_['mdd']*100:+.1f}%")

    print("\n[3] 逐年（万1 + 每月再平衡）...")
    for y in range(2019, 2027):
        a = nav_c[nav_c.index.year == y]
        e = nav_eq[nav_eq.index.year == y]
        if len(a) < 2 or len(e) < 2:
            continue
        print(f"     {y}: 底盘 {a.iloc[-1]/a.iloc[0]-1:+.2%} | 等权 {e.iloc[-1]/e.iloc[0]-1:+.2%}")

    print("\n[4] walk-forward 万1 每月再平衡 ...")
    wf = pd.Timestamp("2023-01-01")
    for label, nav in [("底盘", nav_c), ("等权", nav_eq), ("中位数", nav_md)]:
        tr = perf(nav[nav.index < wf])
        te = perf(nav[nav.index >= wf])
        print(f"     {label}: 训练(19-22) {tr['ann']*100:+.2f}%/夏普{tr['sharpe']:.2f}/回撤{tr['mdd']*100:+.1f}% "
              f"|| 测试(23-26) {te['ann']*100:+.2f}%/夏普{te['sharpe']:.2f}/回撤{te['mdd']*100:+.1f}%")

    print("\n[5] SMA 窗口敏感性（万1 + 每月再平衡, 200±）...")
    global SMA_WIN
    for w in [120, 150, 180, 200, 220, 250]:
        SMA_WIN = w
        nav_w, _ = run_absmom(df, comm=0.0001, rebalance_monthly=True)
        p = perf(nav_w)
        print(f"     SMA{w}: 年化 {p['ann']*100:+6.2f}% 夏普 {p['sharpe']:.2f} 回撤 {p['mdd']*100:+.1f}%")
    SMA_WIN = 200

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_calibration_out.json")
    res = {
        "A_nocost_onswitch": perf(run_absmom(df, comm=0.0, rebalance_monthly=False)[0]),
        "B_nocost_monthly": perf(run_absmom(df, comm=0.0, rebalance_monthly=True)[0]),
        "C_comm1bp_monthly": perf(nav_c),
        "D_comm10bp_monthly": perf(nav_d),
        "bench_equal": pe,
        "bench_median": pm_,
    }
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
