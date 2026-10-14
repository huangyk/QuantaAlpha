"""P6 双动量 top5 深度检验（本轮唯一年化 >12% 的候选）
==================================================

`tools/etf_timeseries_signals.py` 结果里，P6 双动量 top5 全期 +12.36%/S0.60/−24.3%，
是年化最高的。但回撤 −24.3% 比等权基准（−22.6%）还大，且 Sharpe 低于
TSMOM+SMA200（0.60 vs 0.71）—— 集中度换收益的典型特征。

必须查清：
  1. topk 敏感性（top3/5/8/10/全部）——看是否尖峰
  2. 权重方案（等权 vs 动量加权 vs 逆波动）
  3. 调仓频率（月频 vs 季频）
  4. 与同池基准的三口径对比（等权/中位数/最低稳定档/最高稳定档）
  5. 成本敏感（万1/千1）
  6. 逐年 + walk-forward
  7. 与现役底盘（TSMOM+SMA200+逆波动）的对比

铁律：train 内不选参。本脚本 2020-2024 作 IS、2025-2026 作 OOS。
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


def load_etf(start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select asset_code, trade_date, close, amount
        from bars_1d
        where trade_date between %s and %s and asset_type = 'etf'
          and source = 'qmt_local_86400'
        order by asset_code, trade_date
        """,
        (start, end),
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "close", "amount"])
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


def simulate(px, ret, rule, comm=0.0001, mom_win=252, vol_win=60, topk=5,
             rebal_every=1, exposure=1.0):
    dates = px.index
    months = dates.to_period("M")
    nav = 1.0
    navs = []
    cur_w = {}
    turnover_sum = 0.0
    month_count = 0
    for i, d in enumerate(dates):
        is_month_start = (i == 0) or (months[i] != months[i - 1])
        if is_month_start and i > 0:
            month_count += 1
            if month_count % rebal_every == 0:
                new_w = rule(px.iloc[:i], ret.iloc[:i], mom_win=mom_win,
                             vol_win=vol_win, topk=topk)
                codes = set(new_w) | set(cur_w)
                turnover = sum(abs(new_w.get(c2, 0.0) - cur_w.get(c2, 0.0)) for c2 in codes)
                turnover_sum += turnover
                nav *= (1 - comm * turnover)
                cur_w = new_w
        day_ret = 0.0
        if cur_w:
            r = ret.iloc[i]
            for code, wt in cur_w.items():
                if code in r.index and pd.notna(r[code]):
                    day_ret += wt * r[code]
        nav *= (1 + exposure * day_ret)
        navs.append(nav)
    return pd.Series(navs, index=dates), turnover_sum


def r_topk_eq(px, ret, mom_win=252, topk=5, **kw):
    mom = px.iloc[-1] / px.iloc[-min(mom_win, len(px))] - 1
    mom = mom[mom > 0].sort_values(ascending=False)
    held = mom.head(topk).index
    if len(held) == 0:
        return {}
    return {c: 1.0 / len(held) for c in held}


def r_topk_momw(px, ret, mom_win=252, topk=5, **kw):
    mom = px.iloc[-1] / px.iloc[-min(mom_win, len(px))] - 1
    mom = mom[mom > 0].sort_values(ascending=False)
    held = mom.head(topk)
    if held.empty:
        return {}
    w = held / held.sum()
    return {c: float(v) for c, v in w.items()}


def r_topk_invvol(px, ret, mom_win=252, vol_win=60, topk=5, **kw):
    mom = px.iloc[-1] / px.iloc[-min(mom_win, len(px))] - 1
    mom = mom[mom > 0].sort_values(ascending=False)
    held = mom.head(topk).index
    if len(held) == 0:
        return {}
    v = ret.iloc[-vol_win:][list(held)].std() * np.sqrt(252)
    v = v[v > 0]
    if v.empty:
        return {c: 1.0 / len(held) for c in held}
    inv = 1.0 / v
    w = inv / inv.sum()
    return {c: float(x) for c, x in w.items()}


def r_tsmom_invvol(px, ret, mom_win=252, vol_win=60, **kw):
    """现役底盘：TSMOM 过滤 + 逆波动率权重（全池通过者）。"""
    mom = px.iloc[-1] / px.iloc[-min(mom_win, len(px))] - 1
    held = mom[mom > 0].index
    if len(held) == 0:
        return {}
    v = ret.iloc[-vol_win:][list(held)].std() * np.sqrt(252)
    v = v[v > 0]
    if v.empty:
        return {c: 1.0 / len(held) for c in held}
    inv = 1.0 / v.clip(lower=0.05)
    w = inv / inv.sum()
    return {c: float(x) for c, x in w.items()}


def main():
    print("[1] 加载 ...")
    df = load_etf(START, END)
    df = df[~df["code"].isin(BAD)].copy()
    px = df.pivot(index="date", columns="code", values="close").sort_index()
    ret = px.pct_change(fill_method=None)
    print(f"     {px.shape[0]} 日 x {px.shape[1]} 只")

    print("\n[2] topk 敏感性（等权, 月频, 万1）...")
    for k in [3, 5, 8, 10, 15, 36]:
        nav, to = simulate(px, ret, r_topk_eq, topk=k)
        p = perf(nav)
        wf = pd.Timestamp("2025-01-01")
        oos = perf(nav[nav.index >= wf])
        print(f"     top{k:2d}: 全期 {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}% "
              f"|| OOS(25-26) {oos['ann']*100:+6.2f}%/S{oos['sharpe']:.2f}/回撤{oos['mdd']*100:+6.1f}% "
              f"|| 换手{to:.0f}")

    print("\n[3] 权重方案（top5, 月频, 万1）...")
    for label, rule in [("等权", r_topk_eq), ("动量加权", r_topk_momw), ("逆波动", r_topk_invvol)]:
        nav, to = simulate(px, ret, rule, topk=5)
        p = perf(nav)
        wf = pd.Timestamp("2025-01-01")
        oos = perf(nav[nav.index >= wf])
        print(f"     {label:8s}: 全期 {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}% "
              f"|| OOS {oos['ann']*100:+6.2f}%/S{oos['sharpe']:.2f}/回撤{oos['mdd']*100:+6.1f}%")

    print("\n[4] 调仓频率（top5 等权）...")
    for every in [1, 2, 3, 6]:
        nav, to = simulate(px, ret, r_topk_eq, topk=5, rebal_every=every)
        p = perf(nav)
        print(f"     每{every}个月: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}% 换手{to:.0f}")

    print("\n[5] 成本敏感（top5 等权 vs 现役底盘）...")
    for label, rule, kw in [("top5等权", r_topk_eq, dict(topk=5)),
                            ("现役底盘", r_tsmom_invvol, {})]:
        for comm in [0.0001, 0.001]:
            nav, _ = simulate(px, ret, rule, comm=comm, **kw)
            p = perf(nav)
            print(f"     {label} 单边{comm*10000:.0f}bp: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+.1f}%")

    print("\n[6] 基准口径对比（36只池, 全期）...")
    r = ret.iloc[1:]
    nav_eq = (1 + r.mean(axis=1)).cumprod()
    nav_md = (1 + r.median(axis=1)).cumprod()
    d2 = df.copy()
    d2["amt60_prev"] = d2.groupby("code")["amount"].transform(
        lambda s: s.rolling(60, min_periods=60).mean()).groupby(d2["code"]).shift(1)
    d2["bkt_stab"] = d2.groupby("date")["amt60_prev"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    lo = d2[d2["bkt_stab"] == 0].pivot(index="date", columns="code", values="close").sort_index().pct_change(fill_method=None)
    hi = d2[d2["bkt_stab"] == 9].pivot(index="date", columns="code", values="close").sort_index().pct_change(fill_method=None)
    benches = {"等权": nav_eq, "中位数": nav_md,
               "最低稳定档": (1 + lo.mean(axis=1)).cumprod(),
               "最高稳定档": (1 + hi.mean(axis=1)).cumprod()}
    for k, v in benches.items():
        p = perf(v)
        print(f"     {k:10s}: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}%")

    print("\n[7] 逐年（top5等权 vs 现役底盘 vs 等权基准）...")
    nav_t, _ = simulate(px, ret, r_topk_eq, topk=5)
    nav_b, _ = simulate(px, ret, r_tsmom_invvol)
    for y in range(2019, 2027):
        a = nav_t[nav_t.index.year == y]
        b = nav_b[nav_b.index.year == y]
        e = nav_eq[nav_eq.index.year == y]
        if len(a) < 2:
            continue
        print(f"     {y}: top5 {a.iloc[-1]/a.iloc[0]-1:+7.2%} | 底盘 {b.iloc[-1]/b.iloc[0]-1:+7.2%} | "
              f"等权 {e.iloc[-1]/e.iloc[0]-1:+7.2%}")

    print("\n[8] 持仓集中度（top5等权）...")
    mom = px / px.shift(252) - 1
    held_counts = []
    for i in range(252, len(px)):
        m = mom.iloc[i]
        m = m[m > 0].sort_values(ascending=False)
        held_counts.append(len(m.head(5)))
    hs = pd.Series(held_counts)
    print(f"     top5 实际持仓数: 平均 {hs.mean():.2f} 最少 {hs.min()} 最多 {hs.max()}")
    print(f"     通过 TSMOM 的资产数: 平均 {(mom.iloc[252:] > 0).sum(axis=1).mean():.1f}")

    print("\n[9] 参数成簇（topk × 动量窗口）...")
    for k in [3, 5, 8]:
        row = []
        for mw in [189, 252, 315]:
            nav, _ = simulate(px, ret, r_topk_eq, topk=k, mom_win=mw)
            p = perf(nav)
            row.append(f"mom{mw}:{p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}")
        print(f"     top{k}: " + " | ".join(row))

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_dualmom_out.json")
    res = {}
    for k in [3, 5, 8, 10, 15, 36]:
        nav, to = simulate(px, ret, r_topk_eq, topk=k)
        res[f"top{k}"] = perf(nav)
    res["baseline_equal"] = perf(nav_eq)
    res["baseline_median"] = perf(nav_md)
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
