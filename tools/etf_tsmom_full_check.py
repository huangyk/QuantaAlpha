"""TSMOM 全池（top36）的深度检验 —— 本轮最稳候选
============================================

`tools/etf_dualmom_check.py` 结果：
  - top36（= 全部通过 TSMOM 的资产，等权）：全期 **+10.38%/S0.72/−16.0%**，
    OOS(25-26) +54.88%/S3.28/−12.1%
  - top5 集中：+12.50%/S0.60/−24.3%（年化更高但 Sharpe 更低、回撤更大）
  - 现役底盘（TSMOM+逆波动）：+8.18%/S0.78/−13.9%

**关键观察**：topk 从 3 到 36，年化 14.30%→10.38% 单调下降，
而 Sharpe 0.58→0.72 单调上升、回撤 −42.5%→−16.0% 单调改善。
→ **topk 不是尖峰参数，是"集中度-风险"连续谱**，这是健康的参数结构。

本脚本把 top36（TSMOM 全池等权）作为主候选，做完整检验：
  1. 三/四口径基准对比（等权/中位数/最低稳定档/最高稳定档）
  2. walk-forward（2019-2022 / 2023-2026）
  3. 成本敏感（万1/千1）
  4. 参数成簇（动量窗口 × 是否加 SMA 过滤）
  5. 暴露档（100/80/60/50%）
  6. 逐年
  7. 与现役底盘、top5 三方对比
  8. 组合层叠加：逆波动率权重 vs 等权 vs 风险平价
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
    r = nav.pct_change(fill_method=None).dropna()
    if len(r) < 2:
        return {}
    total = nav.iloc[-1]
    years = len(nav) / 252
    ann = total ** (1 / years) - 1 if total > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    calmar = ann / abs(dd) if dd < 0 else np.nan
    return dict(total=float(total), ann=float(ann), vol=float(vol),
                sharpe=float(sharpe), mdd=float(dd), calmar=float(calmar))


def erc_weights(cov):
    n = cov.shape[0]
    if n <= 1:
        return np.ones(max(n, 1)) / max(n, 1)
    w = np.ones(n) / n
    for _ in range(300):
        mrc = cov @ w
        rc = w * mrc
        total = rc.sum()
        if total <= 0:
            break
        target = total / n
        new_w = w * (target / np.maximum(rc, 1e-12)) ** 0.5
        s = new_w.sum()
        if s <= 0:
            break
        new_w /= s
        if np.max(np.abs(new_w - w)) < 1e-9:
            w = new_w
            break
        w = new_w
    return w


def simulate(px, ret, mom_win=252, sma_win=None, weight="equal", comm=0.0001,
             vol_win=60, exposure=1.0):
    dates = px.index
    months = dates.to_period("M")
    nav = 1.0
    navs = []
    cur_w = {}
    turnover_sum = 0.0
    for i, d in enumerate(dates):
        is_month_start = (i == 0) or (months[i] != months[i - 1])
        if is_month_start and i > 0:
            hist = px.iloc[:i]
            hret = ret.iloc[:i]
            mom = hist.iloc[-1] / hist.iloc[-min(mom_win, len(hist))] - 1
            held = mom[mom > 0].index.tolist()
            if sma_win is not None:
                sma = hist.rolling(sma_win, min_periods=sma_win).mean().iloc[-1]
                last = hist.iloc[-1]
                held = [c for c in held if pd.notna(sma[c]) and last[c] > sma[c]]
            if not held:
                new_w = {}
            elif weight == "equal":
                new_w = {c: 1.0 / len(held) for c in held}
            elif weight == "invvol":
                v = hret.iloc[-vol_win:][held].std() * np.sqrt(252)
                v = v[v > 0]
                if v.empty:
                    new_w = {c: 1.0 / len(held) for c in held}
                else:
                    inv = 1.0 / v.clip(lower=0.05)
                    w = inv / inv.sum()
                    new_w = {c: float(x) for c, x in w.items()}
            elif weight == "erc":
                r = hret.iloc[-vol_win:][held].dropna(axis=1, how="any")
                if r.shape[1] < 2 or len(r) < 20:
                    new_w = {c: 1.0 / len(held) for c in held}
                else:
                    cov = r.cov().values * 252
                    w = erc_weights(cov)
                    new_w = {c: float(v) for c, v in zip(r.columns, w) if v > 1e-6}
            else:
                new_w = {c: 1.0 / len(held) for c in held}
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


def main():
    print("[1] 加载 36 只干净跨资产 ETF ...")
    df = load_etf(START, END)
    df = df[~df["code"].isin(BAD)].copy()
    px = df.pivot(index="date", columns="code", values="close").sort_index()
    ret = px.pct_change(fill_method=None)
    print(f"     {px.shape[0]} 日 x {px.shape[1]} 只")

    print("\n[2] 四口径基准 ...")
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
        print(f"     {k:10s}: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}%/Calmar{p['calmar']:.2f}")

    print("\n[3] 主候选：TSMOM 全池等权（万1）...")
    nav, to = simulate(px, ret, weight="equal")
    p = perf(nav)
    print(f"     全期 {p['ann']*100:+.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+.1f}%/Calmar{p['calmar']:.2f} 换手{to:.0f}")
    wf = pd.Timestamp("2023-01-01")
    tr = perf(nav[nav.index < wf])
    te = perf(nav[nav.index >= wf])
    print(f"     训练(19-22) {tr['ann']*100:+.2f}%/S{tr['sharpe']:.2f}/回撤{tr['mdd']*100:+.1f}%")
    print(f"     测试(23-26) {te['ann']*100:+.2f}%/S{te['sharpe']:.2f}/回撤{te['mdd']*100:+.1f}%")
    wf2 = pd.Timestamp("2025-01-01")
    oos = perf(nav[nav.index >= wf2])
    print(f"     OOS(25-26) {oos['ann']*100:+.2f}%/S{oos['sharpe']:.2f}/回撤{oos['mdd']*100:+.1f}%")

    print("\n[4] 权重方案对比（TSMOM 全池）...")
    for w in ["equal", "invvol", "erc"]:
        nav_w, to_w = simulate(px, ret, weight=w)
        pw = perf(nav_w)
        tew = perf(nav_w[nav_w.index >= wf])
        print(f"     {w:7s}: 全期 {pw['ann']*100:+6.2f}%/S{pw['sharpe']:.2f}/回撤{pw['mdd']*100:+6.1f}% "
              f"|| 测试 {tew['ann']*100:+6.2f}%/S{tew['sharpe']:.2f} || 千1 ", end="")
        nav_c, _ = simulate(px, ret, weight=w, comm=0.001)
        pc = perf(nav_c)
        print(f"{pc['ann']*100:+6.2f}%/S{pc['sharpe']:.2f}")

    print("\n[5] 参数成簇（动量窗口 × 权重）...")
    for mw in [126, 189, 252, 315, 378]:
        row = []
        for w in ["equal", "invvol"]:
            nav_m, _ = simulate(px, ret, mom_win=mw, weight=w)
            pm = perf(nav_m)
            row.append(f"{w}:{pm['ann']*100:+6.2f}%/S{pm['sharpe']:.2f}")
        print(f"     动量{mw:3d}日({mw/21:.0f}M): " + " | ".join(row))

    print("\n[6] 是否加 SMA200 过滤 ...")
    for sw in [None, 150, 200, 250]:
        nav_s, _ = simulate(px, ret, sma_win=sw, weight="equal")
        ps = perf(nav_s)
        label = "无SMA" if sw is None else f"SMA{sw}"
        print(f"     {label:8s}: 全期 {ps['ann']*100:+6.2f}%/S{ps['sharpe']:.2f}/回撤{ps['mdd']*100:+6.1f}%")

    print("\n[7] 暴露档（TSMOM 全池等权）...")
    for ex in [1.0, 0.8, 0.6, 0.5]:
        nav_e, _ = simulate(px, ret, weight="equal", exposure=ex)
        pe = perf(nav_e)
        print(f"     暴露 {ex*100:3.0f}%: {pe['ann']*100:+6.2f}%/S{pe['sharpe']:.2f}/回撤{pe['mdd']*100:+6.1f}%")

    print("\n[8] 逐年（TSMOM全池等权 vs 等权基准 vs 现役底盘）...")
    nav_b, _ = simulate(px, ret, weight="invvol")
    for y in range(2019, 2027):
        a = nav[nav.index.year == y]
        b = nav_b[nav_b.index.year == y]
        e = nav_eq[nav_eq.index.year == y]
        if len(a) < 2:
            continue
        print(f"     {y}: TSMOM等权 {a.iloc[-1]/a.iloc[0]-1:+7.2%} | 现役底盘 {b.iloc[-1]/b.iloc[0]-1:+7.2%} | "
              f"等权基准 {e.iloc[-1]/e.iloc[0]-1:+7.2%}")

    print("\n[9] 月度胜率（TSMOM 全池等权 vs 等权基准）...")
    m_a = nav.resample("ME").last().pct_change(fill_method=None).dropna()
    m_e = nav_eq.resample("ME").last().pct_change(fill_method=None).dropna()
    common = m_a.index.intersection(m_e.index)
    print(f"     TSMOM 月胜率 {(m_a[common]>0).mean():.1%}  等权基准月胜率 {(m_e[common]>0).mean():.1%}  n={len(common)}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_tsmom_full_out.json")
    res = {"tsmom_equal": p, "train": tr, "test": te, "oos_25_26": oos,
           "bench_equal": perf(nav_eq), "bench_median": perf(nav_md),
           "bench_lo": perf(benches["最低稳定档"]), "bench_hi": perf(benches["最高稳定档"]),
           "turnover": float(to)}
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
