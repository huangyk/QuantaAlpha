"""TSMOM 家族的完整变体扫描 + 混合权重 + 波动率目标
==============================================

第三轮确认 TSMOM 全池等权（+10.38%/S0.72/−16.0%）是健康候选。
本轮在其基础上系统扫变体，找是否有更优的风险调整组合：

  1. **混合权重**：等权 × 逆波动的凸组合（λ 从 0 到 1）
  2. **波动率目标**：组合年化波动超阈值就降暴露（5%/8%/10%/12%/15%）
  3. **回撤降级**：NAV 自高点回落超阈值就全清（6%/8%/10%）
  4. **动量加权 + 等权混合**
  5. **TSMOM + 单资产止损**（持仓 ETF 从买入价回撤 >X% 就剔出）

全部在 36 只干净池、2019-01~2026-09、万1、月频下测，
输出四口径基准对照 + walk-forward + 成本敏感。

铁律：train 内不选参；变体扫描只用来说明"是否成簇"，不作为选参依据。
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
    s = pd.Series(nav)
    r = s.pct_change(fill_method=None).dropna()
    if len(r) < 2:
        return {}
    total = float(s.iloc[-1])
    years = len(s) / 252
    ann = total ** (1 / years) - 1 if total > 0 else np.nan
    vol = float(r.std() * np.sqrt(252))
    sharpe = ann / vol if vol > 0 else np.nan
    dd = float((s / s.cummax() - 1).min())
    calmar = ann / abs(dd) if dd < 0 else np.nan
    return dict(total=total, ann=float(ann), vol=vol, sharpe=float(sharpe),
                mdd=dd, calmar=float(calmar))


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


def simulate(px, ret, mom_win=252, weight="equal", lam=0.0, comm=0.0001,
             vol_win=60, vol_floor=0.05, vol_target=None, dd_guard=None,
             asset_stop=None, exposure=1.0):
    """weight: 'equal' | 'invvol' | 'mix'(lam*invvol + (1-lam)*equal) | 'momw'
    vol_target: 组合年化波动上限（超限按比例降暴露）
    dd_guard: NAV 自高点回落超此值 → 全清（恢复：回到 -dd_guard/2 内）
    asset_stop: 单资产从建仓价回撤超此值 → 剔出该资产
    """
    dates = px.index
    months = dates.to_period("M")
    nav = 1.0
    navs = []
    cur_w = {}
    entry_px = {}
    turnover_sum = 0.0
    peak = 1.0
    in_cash = False
    for i, d in enumerate(dates):
        is_month_start = (i == 0) or (months[i] != months[i - 1])
        if is_month_start and i > 0:
            hist = px.iloc[:i]
            hret = ret.iloc[:i]
            mom = hist.iloc[-1] / hist.iloc[-min(mom_win, len(hist))] - 1
            held = mom[mom > 0].index.tolist()
            # 单资产止损
            if asset_stop is not None and cur_w:
                drop = []
                for code in cur_w:
                    ep = entry_px.get(code)
                    if ep is not None and ep > 0:
                        cur_px = hist.iloc[-1][code]
                        if pd.notna(cur_px) and cur_px / ep - 1 <= asset_stop:
                            drop.append(code)
                held = [c for c in held if c not in drop]
            if not held:
                new_w = {}
            else:
                v = hret.iloc[-vol_win:][held].std() * np.sqrt(252)
                v = v[v > 0]
                if v.empty:
                    w_eq = {c: 1.0 / len(held) for c in held}
                    new_w = w_eq
                else:
                    inv = 1.0 / v.clip(lower=vol_floor)
                    w_iv = inv / inv.sum()
                    if weight == "equal":
                        new_w = {c: 1.0 / len(held) for c in held}
                    elif weight == "invvol":
                        new_w = {c: float(x) for c, x in w_iv.items()}
                    elif weight == "mix":
                        w_mix = {}
                        for c in held:
                            e = 1.0 / len(held)
                            iv = float(w_iv.get(c, 0.0))
                            w_mix[c] = lam * iv + (1 - lam) * e
                        s = sum(w_mix.values())
                        new_w = {c: x / s for c, x in w_mix.items()} if s > 0 else {c: 1.0 / len(held) for c in held}
                    elif weight == "momw":
                        m = mom[held].sort_values(ascending=False)
                        m = m / m.sum()
                        new_w = {c: float(x) for c, x in m.items()}
                    else:
                        new_w = {c: 1.0 / len(held) for c in held}
            codes = set(new_w) | set(cur_w)
            t = sum(abs(new_w.get(x, 0.0) - cur_w.get(x, 0.0)) for x in codes)
            turnover_sum += t
            nav *= (1 - comm * t)
            cur_w = new_w
            entry_px = {c: float(hist.iloc[-1][c]) for c in new_w}
        # 回撤降级
        peak = max(peak, nav)
        if dd_guard is not None:
            dd = nav / peak - 1
            if not in_cash and dd <= -dd_guard:
                in_cash = True
                codes = set(cur_w)
                t = sum(cur_w.values())
                turnover_sum += t
                nav *= (1 - comm * t)
                cur_w = {}
            elif in_cash and dd >= -dd_guard / 2:
                in_cash = False
        day_ret = 0.0
        if cur_w:
            r = ret.iloc[i]
            for code, wt in cur_w.items():
                if code in r.index and pd.notna(r[code]):
                    day_ret += wt * r[code]
        port_ret = exposure * day_ret
        if vol_target is not None and len(navs) >= 60:
            hs = pd.Series(navs[-60:]).pct_change(fill_method=None).dropna()
            if len(hs) > 20:
                v = hs.std() * np.sqrt(252)
                if v > vol_target:
                    port_ret *= vol_target / v
        nav *= (1 + port_ret)
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
    for k, v in [("等权", nav_eq), ("中位数", nav_md),
                 ("最低稳定档", (1 + lo.mean(axis=1)).cumprod()),
                 ("最高稳定档", (1 + hi.mean(axis=1)).cumprod())]:
        p = perf(v.values)
        print(f"     {k:10s}: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}%/Calmar{p['calmar']:.2f}")

    wf = pd.Timestamp("2023-01-01")
    oos_s = pd.Timestamp("2025-01-01")

    def report(label, **kw):
        nav, to = simulate(px, ret, **kw)
        p = perf(nav.values)
        tr = perf(nav[nav.index < wf].values)
        te = perf(nav[nav.index >= wf].values)
        oos = perf(nav[nav.index >= oos_s].values)
        nav_c, _ = simulate(px, ret, comm=0.001, **kw)
        pc = perf(nav_c.values)
        print(f"     {label:26s}: 全期 {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}% "
              f"|| 测试 {te['ann']*100:+6.2f}%/S{te['sharpe']:.2f}/回撤{te['mdd']*100:+6.1f}% "
              f"|| 千1 {pc['ann']*100:+6.2f}%/S{pc['sharpe']:.2f} || 换手{to:.0f}")
        return p, tr, te, oos, pc

    print("\n[3] 混合权重（λ×逆波动 + (1−λ)×等权）...")
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        report(f"λ={lam}", weight="mix", lam=lam)

    print("\n[4] 动量加权 ...")
    report("动量加权", weight="momw")

    print("\n[5] 波动率目标（TSMOM 等权 + vol_target）...")
    for vt in [0.05, 0.08, 0.10, 0.12, 0.15]:
        report(f"vol_target={vt:.0%}", vol_target=vt)

    print("\n[6] 回撤降级（TSMOM 等权 + dd_guard）...")
    for g in [0.06, 0.08, 0.10, 0.15]:
        report(f"dd_guard={g:.0%}", dd_guard=g)

    print("\n[7] 单资产止损（TSMOM 等权 + asset_stop）...")
    for a in [-0.08, -0.10, -0.15, -0.20]:
        report(f"asset_stop={a:.0%}", asset_stop=a)

    print("\n[8] 组合：波动率目标 + 回撤降级 ...")
    for vt in [0.08, 0.10, 0.12]:
        for g in [0.08, 0.10]:
            report(f"vt={vt:.0%}+guard={g:.0%}", vol_target=vt, dd_guard=g)

    print("\n[9] 动量窗口 × 权重（稳健性）...")
    for mw in [189, 252, 315]:
        row = []
        for w, kw in [("equal", {}), ("invvol", dict(weight="invvol")),
                      ("mix0.5", dict(weight="mix", lam=0.5))]:
            nav, _ = simulate(px, ret, mom_win=mw, **kw)
            p = perf(nav.values)
            row.append(f"{w}:{p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}")
        print(f"     动量{mw}: " + " | ".join(row))

    print("\n[10] 逐年（TSMOM 等权 + vt=10% + guard=10%）...")
    nav_x, _ = simulate(px, ret, vol_target=0.10, dd_guard=0.10)
    for y in range(2019, 2027):
        a = nav_x[nav_x.index.year == y]
        e = nav_eq[nav_eq.index.year == y]
        if len(a) < 2:
            continue
        print(f"     {y}: 组合 {a.iloc[-1]/a.iloc[0]-1:+7.2%} | 等权基准 {e.iloc[-1]/e.iloc[0]-1:+7.2%}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_tsmom_variants_out.json")
    res = {}
    for label, kw in [("equal", {}), ("invvol", dict(weight="invvol")),
                      ("mix0.5", dict(weight="mix", lam=0.5)),
                      ("momw", dict(weight="momw")),
                      ("vt10", dict(vol_target=0.10)),
                      ("guard10", dict(dd_guard=0.10)),
                      ("vt10_guard10", dict(vol_target=0.10, dd_guard=0.10))]:
        nav, to = simulate(px, ret, **kw)
        res[label] = perf(nav.values)
        res[label]["turnover"] = float(to)
    res["bench_equal"] = perf(nav_eq.values)
    res["bench_median"] = perf(nav_md.values)
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
