"""ETF 池三口径复核 —— 检验 ETF 跨资产轮动/等权持有的历史结论
=========================================================

背景：第一轮发现「同池等权基准被妖股绑架」（等权 2024 +36.3% vs 中位数 -1.1%）。
该问题同样影响 ETF 结论：`docs/strategy_absmom.md` 的基线是「纯等权持有 10.9%/Sharpe 0.71」，
`docs/qixing_etf_verdict.md` 的基准是「同池等权 +9.3%/Sharpe 0.65」。

本脚本对同一 ETF 池（`bars_1d` 中 asset_type='etf'，34 只干净池口径）做：
  1. 三口径基准 NAV 对比：等权 / 中位数 / 最低成交额十分位 / 最高成交额十分位
     —— 看 ETF 池的等权是否也被少数妖基绑架
  2. 复现底盘策略（200日SMA + 逆波动率，月频）在三口径下的表现
  3. 成本敏感性：万1 / 千1
  4. walk-forward：2019-2022 训练 vs 2023-2026 测试

数据铁律：仅 TimescaleDB（qmt_local_86400 来源为主），不碰 akshare。
"""
import os
import tempfile
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
START = "2019-01-02"
END = "2026-09-30"
SMA_WIN = 200
VOL_WIN = 60

# 旧报告踢出的 5 只坏数据 ETF
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


def clean(df):
    """剔坏数据源 ETF + 剔单日收益 >30% 的异常（份额拆分假暴跌）。"""
    df = df[~df["code"].isin(BAD)].copy()
    g = df.groupby("code")
    df["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    bad_ret = df["dret"].abs() > 0.30
    print(f"     剔异常单日收益 {int(bad_ret.sum())} 行（|涨跌|>30%）")
    df = df[~bad_ret].copy()
    return df


def build_benchmarks(panel):
    """三/四口径 ETF 基准。"""
    d = panel[panel["dret"].notna() & (panel["dret"].abs() < 0.5)].copy()
    daily_eq = d.groupby("date")["dret"].mean()
    daily_med = d.groupby("date")["dret"].median()
    d["amt_dec"] = d.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    daily_lo = d[d["amt_dec"] == 0].groupby("date")["dret"].mean()
    daily_hi = d[d["amt_dec"] == 9].groupby("date")["dret"].mean()
    return {
        "equal": (1 + daily_eq).cumprod(),
        "median": (1 + daily_med).cumprod(),
        "amt_decile0": (1 + daily_lo).cumprod(),
        "amt_decile9": (1 + daily_hi).cumprod(),
    }


def run_absmom(panel, exposure=1.0, comm=0.0001):
    """200日SMA + 逆波动率，月频调仓；返回 NAV Series。"""
    df = panel.copy()
    g = df.groupby("code")
    df["sma200"] = g["close"].transform(lambda s: s.rolling(SMA_WIN, min_periods=SMA_WIN).mean())
    df["vol60"] = g["dret"].transform(lambda s: s.rolling(VOL_WIN, min_periods=VOL_WIN).std() * np.sqrt(252))
    df["month"] = df["date"].dt.to_period("M")
    # 每月末信号，次月首个交易日生效
    dates = sorted(df["date"].unique())
    dates = [pd.Timestamp(d) for d in dates]
    month_last = {}
    for d in dates:
        month_last[d.to_period("M")] = d
    nav = 1.0
    navs = []
    holdings = {}
    for i, d in enumerate(dates):
        # 用截至前一日信息决定当日持仓
        prev = dates[i - 1] if i > 0 else d
        pm = prev.to_period("M")
        # 上个月末是否已计算过
        if pm not in holdings:
            sub = df[df["date"] == month_last.get(pm, prev)]
            sub = sub[sub["sma200"].notna() & sub["vol60"].notna()]
            held = sub[sub["close"] > sub["sma200"]]
            if held.empty:
                holdings[pm] = {}
            else:
                inv = 1.0 / held["vol60"].clip(lower=0.05)
                w = inv / inv.sum()
                holdings[pm] = dict(zip(held["code"], w))
        w = holdings.get(pm, {})
        # 当日收益（按当日收盘价，成交价用 close 近似，ETF 流动性好）
        sub = df[df["date"] == d].set_index("code")
        day_ret = 0.0
        if w:
            for code, wt in w.items():
                if code in sub.index:
                    r = sub.loc[code, "dret"]
                    if pd.notna(r):
                        day_ret += wt * r
        # 月频调仓成本：仅在持仓集合变化的那天扣
        nav *= (1 + exposure * day_ret)
        navs.append((d, nav))
    s = pd.Series(dict(navs))
    return s


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
    return dict(total=float(total), ann=float(ann), vol=float(vol), sharpe=float(sharpe), mdd=float(dd))


def main():
    print(f"[1] 加载 ETF 面板 {START}~{END} ...")
    df = load_etf(START, END)
    print(f"     行数 {len(df):,}  代码 {df['code'].nunique()}")
    df = clean(df)
    print(f"     清洗后 {len(df):,} 行 / {df['code'].nunique()} 只")
    g = df.groupby("code")
    df["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)

    print("\n[2] 四口径 ETF 基准对比 ...")
    bench = build_benchmarks(df)
    for k, nav in bench.items():
        p = perf(nav)
        print(f"     {k:12s}: 年化 {p['ann']*100:+6.2f}%  波动 {p['vol']*100:5.2f}%  "
              f"夏普 {p['sharpe']:.2f}  回撤 {p['mdd']*100:+.1f}%  终点 {p['total']:.3f}")

    print("\n[3] 底盘策略（200日SMA+逆波动率, 月频, 满仓）...")
    nav_abs = run_absmom(df, exposure=1.0)
    p = perf(nav_abs)
    print(f"     满仓: 年化 {p['ann']*100:+.2f}%  波动 {p['vol']*100:.2f}%  "
          f"夏普 {p['sharpe']:.2f}  回撤 {p['mdd']*100:+.1f}%")
    print(f"     同期等权基准: 年化 {perf(bench['equal'])['ann']*100:+.2f}%  "
          f"夏普 {perf(bench['equal'])['sharpe']:.2f}  回撤 {perf(bench['equal'])['mdd']*100:+.1f}%")
    print(f"     同期中位数基准: 年化 {perf(bench['median'])['ann']*100:+.2f}%  "
          f"夏普 {perf(bench['median'])['sharpe']:.2f}  回撤 {perf(bench['median'])['mdd']*100:+.1f}%")

    print("\n[4] 暴露比例档（60%/50% 低回撤档）...")
    for ex in [1.0, 0.6, 0.5]:
        nv = run_absmom(df, exposure=ex)
        pp = perf(nv)
        print(f"     暴露 {ex*100:.0f}%: 年化 {pp['ann']*100:+5.2f}%  波动 {pp['vol']*100:.2f}%  "
              f"夏普 {pp['sharpe']:.2f}  回撤 {pp['mdd']*100:+.1f}%")

    print("\n[5] 逐年（底盘满仓 vs 等权 vs 中位数）...")
    for y in range(2019, 2027):
        sub_abs = nav_abs[nav_abs.index.year == y]
        sub_eq = bench["equal"][bench["equal"].index.year == y]
        sub_md = bench["median"][bench["median"].index.year == y]
        if len(sub_abs) < 2 or len(sub_eq) < 2:
            continue
        ra = sub_abs.iloc[-1] / sub_abs.iloc[0] - 1
        re_ = sub_eq.iloc[-1] / sub_eq.iloc[0] - 1
        rm = sub_md.iloc[-1] / sub_md.iloc[0] - 1 if len(sub_md >= 2) else np.nan
        print(f"     {y}: 底盘 {ra*100:+6.2f}% | 等权 {re_*100:+6.2f}% | 中位数 {rm*100:+6.2f}%")

    print("\n[6] walk-forward（2019-2022 训练 / 2023-2026 测试）...")
    wf_split = pd.Timestamp("2023-01-01")
    for label, nav in [("底盘", nav_abs), ("等权", bench["equal"]), ("中位数", bench["median"])]:
        tr = nav[nav.index < wf_split]
        te = nav[nav.index >= wf_split]
        ptr = perf(tr)
        pte = perf(te)
        print(f"     {label}: 训练(19-22) 年化 {ptr['ann']*100:+.2f}% 夏普 {ptr['sharpe']:.2f} 回撤 {ptr['mdd']*100:+.1f}% "
              f"|| 测试(23-26) 年化 {pte['ann']*100:+.2f}% 夏普 {pte['sharpe']:.2f} 回撤 {pte['mdd']*100:+.1f}%")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_tribench_out.json")
    import json
    res = {k: perf(v) for k, v in bench.items()}
    res["absmom_full"] = perf(nav_abs)
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
