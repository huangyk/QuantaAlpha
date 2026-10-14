"""ETF 底盘口径差异核实：报告 POOL 38 只 vs 我 36 只
==============================================

报告 `docs/strategy_absmom.md` 说 POOL 38 只，我上一轮只用 qmt_local_86400 源、
剔 5 只坏数据得 36 只。差异来源要查清：
  1. 报告的 38 只具体是哪些（quantaalpha/strategy/absmom_ptrade.py 的 POOL 有 38 个）
  2. 这 38 只里，库里各自的数据情况（来源、起始日、是否有坏段）
  3. 用**报告原版 38 只 POOL** 重跑，看能否复现 +10.0%/1.03/−9.4%
  4. 对比 36 只（剔坏）与 38 只（原版）的差异

关键：报告的 5 只"坏数据"ETF（159915/159919/159922/159949/510050）里，
159915 和 159949 就在报告 POOL 里 → 报告可能用了含坏数据的池。
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

# 报告 POOL（quantaalpha/strategy/absmom_ptrade.py）
POOL_REPORT = [
    "159915.SZ", "159949.SZ", "159967.SZ", "159981.SZ", "159985.SZ", "161226.SH",
    "501018.SH", "510030.SH", "510300.SH", "510500.SH", "510900.SH", "511010.SH",
    "511090.SH", "511220.SH", "511380.SH", "511880.SH", "512100.SH", "512400.SH",
    "512500.SH", "512890.SH", "513030.SH", "513050.SH", "513080.SH", "513090.SH",
    "513100.SH", "513130.SH", "513180.SH", "513330.SH", "513500.SH", "513520.SH",
    "513650.SH", "513660.SH", "513730.SH", "513980.SH", "518880.SH", "563300.SH",
    "588000.SH", "588080.SH",
]
BAD = {"159915.SZ", "159919.SZ", "159922.SZ", "159949.SZ", "510050.SH"}


def load_etf(codes, start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select asset_code, trade_date, open, high, low, close, volume, amount, source
        from bars_1d
        where trade_date between %s and %s and asset_type = 'etf'
          and asset_code = any(%s)
        order by asset_code, trade_date
        """,
        (start, end, list(codes)),
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "amount", "source"])
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


def run_absmom(df, sma_win=200, vol_win=60, comm=0.0001, vol_floor=0.05, dedup_588=False):
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
            sig = by_date.get(month_last[prev_m])
            if sig is not None:
                sig = sig[sig["sma"].notna() & sig["vol"].notna()]
                held = sig[sig["close"] > sig["sma"]]
                if dedup_588:
                    # 588000 与 588080 高度相关，只留波动低的
                    pair = held[held.index.isin(["588000.SH", "588080.SH"])]
                    if len(pair) == 2:
                        drop = pair["vol"].idxmax()
                        held = held.drop(index=drop)
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
    return pd.Series(dict(navs)), n_rebal, turnover_sum


def main():
    print("[1] 报告 POOL 38 只的数据情况 ...")
    df = load_etf(set(POOL_REPORT), START, END)
    summary = []
    for code in POOL_REPORT:
        sub = df[df["code"] == code]
        if sub.empty:
            summary.append((code, 0, None, None, None))
            continue
        g = sub.sort_values("date").groupby("code")
        dr = sub["close"] / sub["close"].shift(1) - 1
        big = int((dr.abs() > 0.30).sum())
        summary.append((code, len(sub), str(sub["date"].min().date()), str(sub["date"].max().date()),
                        sub["source"].iloc[0]))
    s = pd.DataFrame(summary, columns=["code", "rows", "first", "last", "source"])
    pd.set_option("display.width", 220)
    print(s.to_string(index=False))

    print("\n[2] 用报告原版 38 只 POOL 重跑（万1）...")
    nav38, nr38, to38 = run_absmom(df)
    p38 = perf(nav38)
    print(f"     38只: 年化 {p38['ann']*100:+.2f}%  波动 {p38['vol']*100:.2f}%  "
          f"夏普 {p38['sharpe']:.2f}  回撤 {p38['mdd']*100:+.1f}%  调仓{nr38}次 换手{to38:.1f}")

    print("\n[3] 剔 5 只坏数据后（36 只）重跑 ...")
    clean = df[~df["code"].isin(BAD)].copy()
    nav36, nr36, to36 = run_absmom(clean)
    p36 = perf(nav36)
    print(f"     36只: 年化 {p36['ann']*100:+.2f}%  波动 {p36['vol']*100:.2f}%  "
          f"夏普 {p36['sharpe']:.2f}  回撤 {p36['mdd']*100:+.1f}%  调仓{nr36}次 换手{to36:.1f}")

    print("\n[4] 基准对比（38只池 / 36只池）...")
    for label, d in [("38只", df), ("36只", clean)]:
        dd = d.copy()
        g = dd.groupby("code")
        dd["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
        dd = dd[dd["dret"].notna() & (dd["dret"].abs() < 0.5)]
        nav_eq = (1 + dd.groupby("date")["dret"].mean()).cumprod()
        nav_md = (1 + dd.groupby("date")["dret"].median()).cumprod()
        pe = perf(nav_eq)
        pm_ = perf(nav_md)
        print(f"     {label} 等权: {pe['ann']*100:+.2f}%/S{pe['sharpe']:.2f}/回撤{pe['mdd']*100:+.1f}%  "
              f"|| 中位数: {pm_['ann']*100:+.2f}%/S{pm_['sharpe']:.2f}/回撤{pm_['mdd']*100:+.1f}%")

    print("\n[5] 逐年（38只底盘 vs 38只等权）...")
    dd = df.copy()
    g = dd.groupby("code")
    dd["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    dd = dd[dd["dret"].notna() & (dd["dret"].abs() < 0.5)]
    nav_eq = (1 + dd.groupby("date")["dret"].mean()).cumprod()
    for y in range(2019, 2027):
        a = nav38[nav38.index.year == y]
        e = nav_eq[nav_eq.index.year == y]
        if len(a) < 2 or len(e) < 2:
            continue
        print(f"     {y}: 底盘 {a.iloc[-1]/a.iloc[0]-1:+7.2%} | 等权 {e.iloc[-1]/e.iloc[0]-1:+7.2%}")

    print("\n[6] walk-forward（38只）...")
    wf = pd.Timestamp("2023-01-01")
    tr = perf(nav38[nav38.index < wf])
    te = perf(nav38[nav38.index >= wf])
    print(f"     训练(19-22): {tr['ann']*100:+.2f}%/S{tr['sharpe']:.2f}/回撤{tr['mdd']*100:+.1f}%")
    print(f"     测试(23-26): {te['ann']*100:+.2f}%/S{te['sharpe']:.2f}/回撤{te['mdd']*100:+.1f}%")

    print("\n[7] 暴露档（38只）...")
    for ex in [1.0, 0.6, 0.5]:
        r = nav38.pct_change().fillna(0.0) * ex
        nv = (1 + r).cumprod()
        nv.iloc[0] = 1.0
        pp = perf(nv)
        print(f"     暴露 {ex*100:.0f}%: {pp['ann']*100:+5.2f}%/S{pp['sharpe']:.2f}/回撤{pp['mdd']*100:+.1f}%")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_pool38_out.json")
    res = {"pool38": p38, "pool36": p36, "n_rebal38": nr38, "turnover38": float(to38)}
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
