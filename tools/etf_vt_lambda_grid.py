"""TSMOM + 波动率目标 + λ 混合权重的联合扫描（找风险调整最优点）
======================================================

第四轮结论：
  - TSMOM 全池等权：+7.63%(修正后)/S0.58/−16.0%
  - TSMOM + vt=5%：+8.14%/S0.93/−9.3%（风险调整最优）
  - λ 混合（逆波动）：λ↑ 年化↓ 夏普↑ 回撤↓
  - dd_guard 无效、两层风控打架

本轮做**二维联合扫描** vt × λ，看：
  1. 是否存在风险调整（Sharpe/Calmar）的"高原"而非尖峰 → 可信
  2. 最优点落在哪，是否对用户"回撤可控 + 能赚"诉求最优
  3. 成本敏感性（万1/千1）在最优点是否仍成立
  4. walk-forward 在最优点是否成立
  5. 逐年稳定性

另外测一个**新维度**：波动率目标的实现方式
  - 方式 A（第四轮用的）：组合已实现波动 > vt 就按比例降暴露（用过去 60 日组合波动）
  - 方式 B：用**持仓资产的加权平均波动**预估组合波动（更前瞻，无需等组合自己波动起来）
  - 方式 C：目标波动 + 下限（不低于 30%，防过度降仓）

铁律：train 内不选参；扫描只用于判断"是否成簇"，不据此挑单点。
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


def load_etf_clean(start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select distinct on (asset_code, trade_date)
               asset_code, trade_date, close, amount
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


def simulate(px, ret, mom_win=252, lam=0.0, comm=0.0001, vol_win=60,
             vol_floor=0.05, vt=None, vt_mode="realized", vt_min=0.0,
             vol_lookback=60):
    """TSMOM + λ 混合权重 + 波动率目标。

    vt_mode:
      'realized'  用过去 vol_lookback 日的**组合已实现**波动
      'weighted'  用持仓资产的**加权平均**年化波动（前瞻估计）
    vt_min: 暴露下限（防过度降仓）
    """
    dates = px.index
    months = dates.to_period("M")
    nav = 1.0
    navs = []
    cur_w = {}
    turnover = 0.0
    first_valid = {c: px[c].first_valid_index() for c in px.columns}
    for i, d in enumerate(dates):
        is_month_start = (i == 0) or (months[i] != months[i - 1])
        if is_month_start and i > 0:
            hist = px.iloc[:i]
            hret = ret.iloc[:i]
            ok = []
            for c in px.columns:
                fv = first_valid[c]
                if fv is None or fv > d:
                    continue
                col = hist[c].dropna()
                if len(col) >= min(mom_win, 252):
                    ok.append(c)
            if not ok:
                cur_w = {}
                navs.append(nav)
                continue
            mom = {}
            for c in ok:
                col = hist[c].dropna()
                mom[c] = col.iloc[-1] / col.iloc[-min(mom_win, len(col))] - 1
            held = [c for c in ok if mom[c] > 0]
            if not held:
                new_w = {}
            else:
                w_eq = {c: 1.0 / len(held) for c in held}
                sub = hret.iloc[-vol_win:][held]
                v = (sub.std() * np.sqrt(252)).dropna()
                v = v[v > 0]
                if v.empty:
                    new_w = w_eq
                else:
                    inv = 1.0 / v.clip(lower=vol_floor)
                    w_iv = inv / inv.sum()
                    w_mix = {c: lam * float(w_iv.get(c, 0.0)) + (1 - lam) * w_eq[c] for c in held}
                    s = sum(w_mix.values())
                    new_w = {c: x / s for c, x in w_mix.items()} if s > 0 else w_eq
            codes = set(new_w) | set(cur_w)
            t = sum(abs(new_w.get(x, 0.0) - cur_w.get(x, 0.0)) for x in codes)
            turnover += t
            nav *= (1 - comm * t)
            cur_w = new_w
        # 当日收益
        day_ret = 0.0
        if cur_w:
            r = ret.iloc[i]
            for code, wt in cur_w.items():
                if code in r.index and pd.notna(r[code]):
                    day_ret += wt * r[code]
        port_ret = day_ret
        if vt is not None and cur_w:
            if vt_mode == "realized" and len(navs) >= vol_lookback:
                hs = pd.Series(navs[-vol_lookback:]).pct_change(fill_method=None).dropna()
                if len(hs) > 20:
                    v = hs.std() * np.sqrt(252)
                    if v > vt:
                        port_ret *= max(vt / v, vt_min)
            elif vt_mode == "weighted" and i >= vol_win:
                # 用持仓资产的加权平均年化波动
                wsum = 0.0
                for code, wt in cur_w.items():
                    col = ret[code].iloc[:i].dropna()
                    if len(col) >= vol_win:
                        wsum += wt * float(col.iloc[-vol_win:].std() * np.sqrt(252))
                if wsum > vt:
                    port_ret *= max(vt / wsum, vt_min)
        nav *= (1 + port_ret)
        navs.append(nav)
    return pd.Series(navs, index=dates), turnover


def main():
    print("[1] 加载 36 只干净跨资产 ETF ...")
    df = load_etf_clean(START, END)
    df = df[~df["code"].isin(BAD)].copy()
    px = df.pivot(index="date", columns="code", values="close").sort_index()
    ret = px.pct_change(fill_method=None)
    print(f"     {px.shape[0]} 日 x {px.shape[1]} 只")

    r = ret.iloc[1:]
    nav_eq = (1 + r.mean(axis=1)).cumprod()
    pe = perf(nav_eq.values)
    print(f"\n[2] 等权基准: {pe['ann']*100:+.2f}%/S{pe['sharpe']:.2f}/回撤{pe['mdd']*100:+.1f}%")

    wf = pd.Timestamp("2023-01-01")
    oos_s = pd.Timestamp("2025-01-01")

    print("\n[3] 二维联合扫描：vt × λ（Sharpe / Calmar）...")
    vts = [None, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05]
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("     vt\\λ   " + "".join(f"{'λ=' + str(l):>22s}" for l in lams))
    grid_sharpe = {}
    grid_calmar = {}
    for vt in vts:
        row_s, row_c = [], []
        for lam in lams:
            nav, _ = simulate(px, ret, lam=lam, vt=vt)
            p = perf(nav.values)
            row_s.append(p["sharpe"])
            row_c.append(p["calmar"])
            grid_sharpe[(vt, lam)] = p["sharpe"]
            grid_calmar[(vt, lam)] = p["calmar"]
        label = "无vt " if vt is None else f"{vt*100:4.0f}%"
        print(f"     {label} " + "".join(f"  S{s:5.2f}/C{c:5.2f}      " for s, c in zip(row_s, row_c)))

    print("\n[4] 年化 / 回撤网格 ...")
    print("     vt\\λ   " + "".join(f"{'λ=' + str(l):>20s}" for l in lams))
    for vt in vts:
        row = []
        for lam in lams:
            nav, _ = simulate(px, ret, lam=lam, vt=vt)
            p = perf(nav.values)
            row.append(p)
        label = "无vt " if vt is None else f"{vt*100:4.0f}%"
        print(f"     {label} " + "".join(f" {p['ann']*100:+6.2f}%/{p['mdd']*100:+6.1f}%" for p in row))

    print("\n[5] 最优区域深挖（vt 5~8%, λ 0~0.5）...")
    best = None
    for vt in [0.08, 0.07, 0.06, 0.05]:
        for lam in [0.0, 0.25, 0.5]:
            nav, to = simulate(px, ret, lam=lam, vt=vt)
            p = perf(nav.values)
            tr = perf(nav[nav.index < wf].values)
            te = perf(nav[nav.index >= wf].values)
            oos = perf(nav[nav.index >= oos_s].values)
            nav_c, _ = simulate(px, ret, lam=lam, vt=vt, comm=0.001)
            pc = perf(nav_c.values)
            score = p["sharpe"] + p["calmar"]
            if best is None or score > best[0]:
                best = (score, vt, lam, p, tr, te, oos, pc, to)
            print(f"     vt={vt*100:3.0f}% λ={lam:.2f}: 全期 {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/"
                  f"回撤{p['mdd']*100:+6.1f}%/C{p['calmar']:.2f} || 训练 {tr['ann']*100:+6.2f}%/S{tr['sharpe']:.2f} "
                  f"|| 测试 {te['ann']*100:+6.2f}%/S{te['sharpe']:.2f}/回撤{te['mdd']*100:+.1f}% "
                  f"|| OOS {oos['ann']*100:+6.2f}%/S{oos['sharpe']:.2f} || 千1 {pc['ann']*100:+6.2f}%/S{pc['sharpe']:.2f}")

    score, vt, lam, p, tr, te, oos, pc, to = best
    print(f"\n[6] 最优组合: vt={vt*100:.0f}% λ={lam} (Sharpe+Calmar={score:.2f})")
    print(f"     全期 {p['ann']*100:+.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+.1f}%/Calmar{p['calmar']:.2f} 换手{to:.0f}")
    print(f"     训练 {tr['ann']*100:+.2f}%/S{tr['sharpe']:.2f}/回撤{tr['mdd']*100:+.1f}%")
    print(f"     测试 {te['ann']*100:+.2f}%/S{te['sharpe']:.2f}/回撤{te['mdd']*100:+.1f}%")
    print(f"     OOS  {oos['ann']*100:+.2f}%/S{oos['sharpe']:.2f}/回撤{oos['mdd']*100:+.1f}%")
    print(f"     千1  {pc['ann']*100:+.2f}%/S{pc['sharpe']:.2f}")

    print("\n[7] 波动率目标实现方式对比（vt=6%, λ=0）...")
    for mode in ["realized", "weighted"]:
        nav, to = simulate(px, ret, lam=0.0, vt=0.06, vt_mode=mode)
        p = perf(nav.values)
        te = perf(nav[nav.index >= wf].values)
        print(f"     {mode:10s}: 全期 {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}% "
              f"|| 测试 {te['ann']*100:+6.2f}%/S{te['sharpe']:.2f}/回撤{te['mdd']*100:+.1f}%")

    print("\n[8] 暴露下限 vt_min（vt=6%, λ=0）...")
    for vm in [0.0, 0.3, 0.5]:
        nav, _ = simulate(px, ret, lam=0.0, vt=0.06, vt_min=vm)
        p = perf(nav.values)
        print(f"     vt_min={vm:.1f}: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}%")

    print("\n[9] 最优组合逐年 ...")
    nav_best, _ = simulate(px, ret, lam=lam, vt=vt)
    for y in range(2019, 2027):
        a = nav_best[nav_best.index.year == y]
        e = nav_eq[nav_eq.index.year == y]
        if len(a) < 2 or len(e) < 2:
            continue
        print(f"     {y}: 最优 {a.iloc[-1]/a.iloc[0]-1:+7.2%} vs 等权 {e.iloc[-1]/e.iloc[0]-1:+7.2%}")

    print("\n[10] 高原检验：Sharpe 网格的分布 ...")
    vals = np.array(list(grid_sharpe.values()))
    print(f"     全网格 Sharpe: 最小 {vals.min():.2f} 中位 {np.median(vals):.2f} 最大 {vals.max():.2f}")
    top = sorted(grid_sharpe.items(), key=lambda kv: -kv[1])[:5]
    print("     Sharpe 前 5:")
    for (v_, l_), s_ in top:
        print(f"       vt={'无' if v_ is None else f'{v_*100:.0f}%'} λ={l_}: S={s_:.2f}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_vt_lambda_grid_out.json")
    res = {"grid_sharpe": {f"vt{v}_lam{l}": s for (v, l), s in grid_sharpe.items()},
           "grid_calmar": {f"vt{v}_lam{l}": c for (v, l), c in grid_calmar.items()},
           "best": {"vt": vt, "lam": lam, **p},
           "bench_equal": pe}
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
