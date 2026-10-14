"""ETF 底盘的"报告 vs 复现"差异定位
====================================

`tools/etf_pool38_check.py` 结果：
  - 报告 POOL 38 只（含 2 只"坏数据"ETF：159915、159949，另 161226 库里无数据）
  - 38只池：年化 +8.54% / 夏普 0.86 / 回撤 −11.1%
  - 36只池（剔坏）：年化 +7.59% / 夏普 0.82 / 回撤 −9.9%
  - 报告宣称：+10.0% / 1.03 / −9.4%

仍有差距（8.54% vs 10.0%，0.86 vs 1.03）。差异必须在实现细节里。
本脚本逐项排查最可能的几个来源：
  1. **成交价假设**：报告未说明；我用 close-to-close。
     改用 **T+1 开盘买入**（与股票侧铁律一致）看差异。
  2. **SMA/波动窗口的起算**：报告"近200日收盘均值"是否含当日？
  3. **信号生效时点**：我用"当月首个交易日调仓，信号取上月末"。
     报告伪代码是"每月末判信号，次月调仓"——但**次月哪一天**？
     若次月首个交易日就用**上月末收盘价**成交，会有 1 个月的信息差。
  4. **权重再平衡频率**：是否每月都再平衡（我每月调仓=再平衡）。
  5. **vol 下限**：报告 0.05，我用 0.05。

逐项对比后给出最可能解释，并输出修正后的复现值。
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

POOL_REPORT = [
    "159915.SZ", "159949.SZ", "159967.SZ", "159981.SZ", "159985.SZ", "161226.SH",
    "501018.SH", "510030.SH", "510300.SH", "510500.SH", "510900.SH", "511010.SH",
    "511090.SH", "511220.SH", "511380.SH", "511880.SH", "512100.SH", "512400.SH",
    "512500.SH", "512890.SH", "513030.SH", "513050.SH", "513080.SH", "513090.SH",
    "513100.SH", "513130.SH", "513180.SH", "513330.SH", "513500.SH", "513520.SH",
    "513650.SH", "513660.SH", "513730.SH", "513980.SH", "518880.SH", "563300.SH",
    "588000.SH", "588080.SH",
]


def load_etf(codes, start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select asset_code, trade_date, open, high, low, close, volume, amount
        from bars_1d
        where trade_date between %s and %s and asset_type = 'etf' and asset_code = any(%s)
        order by asset_code, trade_date
        """,
        (start, end, list(codes)),
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


def run_absmom(df, sma_win=200, vol_win=60, comm=0.0001, vol_floor=0.05,
               exec_price="close", sig_lag="month_end"):
    """exec_price: 'close'=按当日收盘成交；'next_open'=次月首个交易日开盘成交
    sig_lag: 'month_end'=信号用上月末；'prev_day'=信号用调仓前一交易日
    """
    d = df.copy()
    g = d.groupby("code")
    d["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    d["sma"] = g["close"].transform(lambda s: s.rolling(sma_win, min_periods=sma_win).mean())
    d["vol"] = g["dret"].transform(lambda s: s.rolling(vol_win, min_periods=vol_win).std() * np.sqrt(252))
    dates = [pd.Timestamp(x) for x in sorted(d["date"].unique())]
    by_date = {dt: grp.set_index("code") for dt, grp in d.groupby("date")}
    month_last = {}
    for dt in dates:
        month_last[dt.to_period("M")] = dt
    nav = 1.0
    navs = []
    cur_w = {}
    n_rebal = 0
    turnover_sum = 0.0
    for i, dt in enumerate(dates):
        is_month_start = (i == 0) or (dates[i - 1].to_period("M") != dt.to_period("M"))
        if is_month_start and i > 0:
            prev_m = dates[i - 1].to_period("M")
            sig_dt = month_last[prev_m] if sig_lag == "month_end" else dates[i - 1]
            sig = by_date.get(sig_dt)
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
        if sub is None:
            navs.append((dt, nav))
            continue
        # 当日组合收益：按持仓权重，用当日价格变动
        day_ret = 0.0
        if cur_w:
            px = "open" if exec_price == "next_open" else "close"
            # 调仓当日：新权重按当日价买入；之前持仓按当日价变动
            for code, wt in cur_w.items():
                if code in sub.index:
                    if exec_price == "next_open" and is_month_start:
                        # 调仓日：从昨日收盘到今日开盘的变动不计（开盘即买入）
                        # 之后按当日收盘相对开盘计
                        o = sub.loc[code, "open"]
                        cl = sub.loc[code, "close"]
                        if pd.notna(o) and pd.notna(cl) and o > 0:
                            day_ret += wt * (cl / o - 1)
                    else:
                        r = sub.loc[code, "dret"]
                        if pd.notna(r):
                            day_ret += wt * r
        nav *= (1 + day_ret)
        navs.append((dt, nav))
    return pd.Series(dict(navs)), n_rebal, turnover_sum


def main():
    print("[1] 加载报告 38 只 POOL ...")
    df = load_etf(set(POOL_REPORT), START, END)
    print(f"     {len(df):,} 行 / {df['code'].nunique()} 只")

    print("\n[2] 口径逐项对比（38只池，万1）...")
    variants = [
        ("A 收盘成交 + 上月末信号（我原版）", dict(exec_price="close", sig_lag="month_end")),
        ("B 开盘成交 + 上月末信号", dict(exec_price="next_open", sig_lag="month_end")),
        ("C 收盘成交 + 调仓前一日信号", dict(exec_price="close", sig_lag="prev_day")),
        ("D 开盘成交 + 调仓前一日信号", dict(exec_price="next_open", sig_lag="prev_day")),
    ]
    for label, kw in variants:
        nav, nr, to = run_absmom(df, **kw)
        p = perf(nav)
        print(f"     {label}: 年化 {p['ann']*100:+6.2f}%  夏普 {p['sharpe']:.2f}  "
              f"回撤 {p['mdd']*100:+6.1f}%  调仓{nr}")

    print("\n[3] 成本敏感性（A 口径）...")
    for comm in [0.0, 0.0001, 0.0005, 0.001]:
        nav, _, _ = run_absmom(df, comm=comm)
        p = perf(nav)
        print(f"     单边{comm*10000:4.0f}bp: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+.1f}%")

    print("\n[4] 参数成簇（A 口径）...")
    for sw in [150, 180, 200, 220, 250]:
        row = []
        for vw in [40, 60, 90]:
            nav, _, _ = run_absmom(df, sma_win=sw, vol_win=vw)
            p = perf(nav)
            row.append(f"vw{vw}:{p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}")
        print(f"     SMA{sw:3d}: " + " | ".join(row))

    print("\n[5] vol_floor 敏感性 ...")
    for vf in [0.01, 0.05, 0.10, 0.15]:
        nav, _, _ = run_absmom(df, vol_floor=vf)
        p = perf(nav)
        print(f"     floor {vf:.2f}: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+.1f}%")

    print("\n[6] 逐年（A 口径）...")
    nav_a, _, _ = run_absmom(df)
    for y in range(2019, 2027):
        s = nav_a[nav_a.index.year == y]
        if len(s) >= 2:
            print(f"     {y}: {s.iloc[-1]/s.iloc[0]-1:+7.2%}")

    print("\n[7] walk-forward（A 口径）...")
    wf = pd.Timestamp("2023-01-01")
    tr = perf(nav_a[nav_a.index < wf])
    te = perf(nav_a[nav_a.index >= wf])
    print(f"     训练(19-22): {tr['ann']*100:+.2f}%/S{tr['sharpe']:.2f}/回撤{tr['mdd']*100:+.1f}%")
    print(f"     测试(23-26): {te['ann']*100:+.2f}%/S{te['sharpe']:.2f}/回撤{te['mdd']*100:+.1f}%")

    print("\n[8] 持仓统计（A 口径）...")
    # 重新跑一遍记录持仓数
    d = df.copy()
    g = d.groupby("code")
    d["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    d["sma"] = g["close"].transform(lambda s: s.rolling(200, min_periods=200).mean())
    d["vol"] = g["dret"].transform(lambda s: s.rolling(60, min_periods=60).std() * np.sqrt(252))
    dates = [pd.Timestamp(x) for x in sorted(d["date"].unique())]
    by_date = {dt: grp.set_index("code") for dt, grp in d.groupby("date")}
    month_last = {}
    for dt in dates:
        month_last[dt.to_period("M")] = dt
    holds = []
    for i, dt in enumerate(dates):
        if i == 0 or dates[i - 1].to_period("M") == dt.to_period("M"):
            continue
        prev_m = dates[i - 1].to_period("M")
        sig = by_date.get(month_last[prev_m])
        if sig is None:
            continue
        sig = sig[sig["sma"].notna() & sig["vol"].notna()]
        held = sig[sig["close"] > sig["sma"]]
        holds.append(len(held))
    print(f"     月均持仓 {np.mean(holds):.1f} 只，范围 {min(holds)}~{max(holds)}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_variant_out.json")
    res = {}
    for label, kw in variants:
        nav, nr, to = run_absmom(df, **kw)
        res[label] = perf(nav)
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
