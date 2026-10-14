"""本轮新方向：**不依赖横截面分档**的信号族
=========================================

前两轮结论：横截面技术信号在"同规模档"口径下全负；每日分档更是未来函数。
本轮换思路，测**时间序列/组合层**信号——它们不依赖横截面排名，
只用单资产自身的历史，因此天然规避风格与分档问题。

在 36 只干净跨资产 ETF 池（2019-01~2026-09）上测：

**组合层（全池统一规则）**
  P1 时间序列动量（TSMOM）：`ret_12M > 0` 才持有，等权；月度调仓
  P2 TSMOM + 趋势过滤：`close > SMA200` 且 `ret_12M > 0`
  P3 风险平价（ERC）：用 60 日协方差等风险贡献，月度再平衡
  P4 最小方差（min-var）：60 日协方差，月度
  P5 TSMOM + 逆波动率（现役底盘的变体，作对照）
  P6 双动量（绝对+相对）：先 TSMOM 过滤，通过者按 12M 动量取 top5
  P7 波动率目标：组合波动超 10% 就按比例降暴露

**单资产层（逐只择时，然后等权持有通过者）**
  T1 close > SMA200
  T2 close > SMA200 且 SMA50 > SMA200（双均线）
  T3 12M 动量 > 0
  T4 12M 动量 > 0 且 close > SMA200
  T5 近 20 日波动率 < 其 60 日分位 0.5（低波环境才持有）

铁律：
  - 三口径基准：等权 / 中位数 / 最低 amount 档 / 最高 amount 档
  - walk-forward：2019-2022 训练 / 2023-2026 测试
  - 成本敏感：万1 / 千1
  - 参数成簇（动量窗口 6/9/12M，SMA 100/150/200/250）
  - 信号-收益对齐：T 日收盘出信号 → T+1 生效（我用次月首个交易日）
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
        select asset_code, trade_date, open, high, low, close, volume, amount
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


def build_px(df):
    """收盘价矩阵 + 日收益矩阵（按 calendar 对齐，缺失填 NaN）。"""
    px = df.pivot(index="date", columns="code", values="close").sort_index()
    ret = px.pct_change()
    return px, ret


def erc_weights(cov):
    """等风险贡献（ERC）迭代解。"""
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    w = np.ones(n) / n
    for _ in range(200):
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
        new_w = new_w / s
        if np.max(np.abs(new_w - w)) < 1e-8:
            w = new_w
            break
        w = new_w
    return w


def simulate(px, ret, rule, comm=0.0001, mom_win=252, sma_win=200, vol_win=60,
             topk=5, vol_target=None, exposure=1.0):
    """月度调仓。rule 决定持有子集与权重。

    rule 签名: f(hist_px, hist_ret) -> dict{code: weight}
      hist_px/hist_ret: 截至信号日（不含当日）的历史
    """
    dates = px.index
    months = dates.to_period("M")
    month_first = {}
    for i, d in enumerate(dates):
        m = months[i]
        if m not in month_first:
            month_first[m] = i
    nav = 1.0
    navs = []
    cur_w = {}
    turnover_sum = 0.0
    for i, d in enumerate(dates):
        is_month_start = (i == 0) or (months[i] != months[i - 1])
        if is_month_start and i > 0:
            hist_px = px.iloc[:i]
            hist_ret = ret.iloc[:i]
            new_w = rule(hist_px, hist_ret, mom_win=mom_win, sma_win=sma_win,
                         vol_win=vol_win, topk=topk)
            codes = set(new_w) | set(cur_w)
            turnover = sum(abs(new_w.get(c2, 0.0) - cur_w.get(c2, 0.0)) for c2 in codes)
            turnover_sum += turnover
            nav *= (1 - comm * turnover)
            cur_w = new_w
        # 当日收益
        day_ret = 0.0
        if cur_w:
            r = ret.iloc[i]
            for code, wt in cur_w.items():
                if code in r.index and pd.notna(r[code]):
                    day_ret += wt * r[code]
        port_ret = exposure * day_ret
        # 波动率目标：组合波动超限则降暴露（简单版，用已实现波动）
        if vol_target is not None and i >= 60:
            hist_port = pd.Series(navs[-60:]).pct_change().dropna() if len(navs) >= 61 else None
            if hist_port is not None and len(hist_port) > 20:
                v = hist_port.std() * np.sqrt(252)
                if v > vol_target:
                    port_ret *= vol_target / v
        nav *= (1 + port_ret)
        navs.append(nav)
    return pd.Series(navs, index=dates), turnover_sum


def rule_equal_all(px, ret, **kw):
    return {c: 1.0 / px.shape[1] for c in px.columns}


def rule_tsmom(px, ret, mom_win=252, **kw):
    mom = px.iloc[-1] / px.iloc[-min(mom_win, len(px))] - 1
    held = mom[mom > 0].index
    if len(held) == 0:
        return {}
    return {c: 1.0 / len(held) for c in held}


def rule_tsmom_sma(px, ret, mom_win=252, sma_win=200, **kw):
    sma = px.rolling(sma_win, min_periods=sma_win).mean().iloc[-1]
    last = px.iloc[-1]
    mom = last / px.iloc[-min(mom_win, len(px))] - 1
    held = [c for c in px.columns if pd.notna(sma[c]) and last[c] > sma[c] and mom[c] > 0]
    if not held:
        return {}
    return {c: 1.0 / len(held) for c in held}


def rule_sma_only(px, ret, sma_win=200, **kw):
    sma = px.rolling(sma_win, min_periods=sma_win).mean().iloc[-1]
    last = px.iloc[-1]
    held = [c for c in px.columns if pd.notna(sma[c]) and last[c] > sma[c]]
    if not held:
        return {}
    return {c: 1.0 / len(held) for c in held}


def rule_erc(px, ret, vol_win=60, **kw):
    r = ret.iloc[-vol_win:]
    r = r.dropna(axis=1, how="any")
    if r.shape[1] < 2 or len(r) < 20:
        return rule_equal_all(px, ret, **kw)
    cov = r.cov().values * 252
    w = erc_weights(cov)
    return {c: float(v) for c, v in zip(r.columns, w) if v > 1e-6}


def rule_minvar(px, ret, vol_win=60, **kw):
    r = ret.iloc[-vol_win:].dropna(axis=1, how="any")
    if r.shape[1] < 2 or len(r) < 20:
        return rule_equal_all(px, ret, **kw)
    cov = r.cov().values * 252
    try:
        inv = np.linalg.pinv(cov)
        w = inv @ np.ones(cov.shape[0])
        w = np.clip(w, 0, None)
        if w.sum() <= 0:
            return rule_equal_all(px, ret, **kw)
        w = w / w.sum()
    except Exception:
        return rule_equal_all(px, ret, **kw)
    return {c: float(v) for c, v in zip(r.columns, w) if v > 1e-6}


def rule_invvol(px, ret, vol_win=60, **kw):
    v = ret.iloc[-vol_win:].std() * np.sqrt(252)
    v = v[v > 0]
    if v.empty:
        return rule_equal_all(px, ret, **kw)
    inv = 1.0 / v
    w = inv / inv.sum()
    return {c: float(x) for c, x in w.items()}


def rule_dual_mom_topk(px, ret, mom_win=252, topk=5, **kw):
    mom = px.iloc[-1] / px.iloc[-min(mom_win, len(px))] - 1
    mom = mom[mom > 0].sort_values(ascending=False)
    held = mom.head(topk).index
    if len(held) == 0:
        return {}
    return {c: 1.0 / len(held) for c in held}


def rule_sma50_200(px, ret, sma_win=200, **kw):
    sma200 = px.rolling(200, min_periods=200).mean().iloc[-1]
    sma50 = px.rolling(50, min_periods=50).mean().iloc[-1]
    last = px.iloc[-1]
    held = [c for c in px.columns
            if pd.notna(sma200[c]) and last[c] > sma200[c] and pd.notna(sma50[c]) and sma50[c] > sma200[c]]
    if not held:
        return {}
    return {c: 1.0 / len(held) for c in held}


def rule_mom_only(px, ret, mom_win=252, **kw):
    return rule_tsmom(px, ret, mom_win=mom_win, **kw)


def rule_lowvol_env(px, ret, vol_win=20, **kw):
    """近20日波动率处其自身历史低分位（<0.5）才持有，等权。"""
    rv = ret.rolling(vol_win, min_periods=vol_win).std() * np.sqrt(252)
    cur = rv.iloc[-1]
    hist = rv.iloc[-252:]
    pct = hist.rank(pct=True).iloc[-1]
    held = [c for c in px.columns if pd.notna(pct[c]) and pct[c] < 0.5]
    if not held:
        return {}
    return {c: 1.0 / len(held) for c in held}


def main():
    print("[1] 加载 36 只干净跨资产 ETF ...")
    df = load_etf(START, END)
    df = df[~df["code"].isin(BAD)].copy()
    px, ret = build_px(df)
    print(f"     {px.shape[0]} 日 x {px.shape[1]} 只  {px.index.min().date()}~{px.index.max().date()}")

    print("\n[2] 四口径基准 ...")
    r = ret.iloc[1:]
    nav_eq = (1 + r.mean(axis=1)).cumprod()
    nav_md = (1 + r.median(axis=1)).cumprod()
    d2 = df.copy()
    d2["bkt"] = d2.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    # 稳定档（滞后60日均量）
    d2["amt60_prev"] = d2.groupby("code")["amount"].transform(
        lambda s: s.rolling(60, min_periods=60).mean()).groupby(d2["code"]).shift(1)
    d2["bkt_stab"] = d2.groupby("date")["amt60_prev"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    lo = d2[d2["bkt_stab"] == 0].pivot(index="date", columns="code", values="close").sort_index().pct_change()
    hi = d2[d2["bkt_stab"] == 9].pivot(index="date", columns="code", values="close").sort_index().pct_change()
    nav_lo = (1 + lo.mean(axis=1)).cumprod()
    nav_hi = (1 + hi.mean(axis=1)).cumprod()
    benches = {"等权": nav_eq, "中位数": nav_md, "最低稳定档": nav_lo, "最高稳定档": nav_hi}
    for k, v in benches.items():
        p = perf(v)
        print(f"     {k:10s}: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+.1f}%")

    print("\n[3] 组合层/择时层策略（万1, 月频）...")
    rules = [
        ("P1 TSMOM 12M 等权", rule_tsmom, {}),
        ("P2 TSMOM+SMA200", rule_tsmom_sma, {}),
        ("P3 风险平价 ERC", rule_erc, {}),
        ("P4 最小方差", rule_minvar, {}),
        ("P5 逆波动率(全池)", rule_invvol, {}),
        ("P6 双动量 top5", rule_dual_mom_topk, {}),
        ("P7b 等权全池(基线)", rule_equal_all, {}),
        ("T1 SMA200 择时", rule_sma_only, {}),
        ("T2 SMA50>SMA200", rule_sma50_200, {}),
        ("T3 12M动量", rule_mom_only, {}),
        ("T5 低波环境", rule_lowvol_env, {}),
    ]
    results = {}
    for label, rule, kw in rules:
        nav, to = simulate(px, ret, rule, comm=0.0001, **kw)
        p = perf(nav)
        # walk-forward
        wf = pd.Timestamp("2023-01-01")
        tr = perf(nav[nav.index < wf])
        te = perf(nav[nav.index >= wf])
        # 成本
        nav_hi_c, _ = simulate(px, ret, rule, comm=0.001, **kw)
        p_hi = perf(nav_hi_c)
        results[label] = {"full": p, "train": tr, "test": te, "cost10bp": p_hi, "turnover": float(to)}
        print(f"     {label:22s}: 全期 {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}% "
              f"|| 训练 {tr['ann']*100:+6.2f}%/S{tr['sharpe']:.2f} || 测试 {te['ann']*100:+6.2f}%/S{te['sharpe']:.2f} "
              f"|| 千1 {p_hi['ann']*100:+6.2f}%/S{p_hi['sharpe']:.2f}")

    print("\n[4] 参数成簇（动量窗口，TSMOM+SMA200）...")
    for mw in [126, 189, 252, 315, 378]:
        nav, _ = simulate(px, ret, rule_tsmom_sma, mom_win=mw)
        p = perf(nav)
        print(f"     动量{mw:3d}日({mw/21:.0f}M): {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+.1f}%")

    print("\n[5] 参数成簇（SMA 窗口，TSMOM+SMA）...")
    for sw in [100, 150, 200, 250]:
        nav, _ = simulate(px, ret, rule_tsmom_sma, sma_win=sw)
        p = perf(nav)
        print(f"     SMA{sw:3d}: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+.1f}%")

    print("\n[6] 暴露档（TSMOM+SMA200）...")
    for ex in [1.0, 0.8, 0.6, 0.5]:
        nav, _ = simulate(px, ret, rule_tsmom_sma, exposure=ex)
        p = perf(nav)
        print(f"     暴露 {ex*100:3.0f}%: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+.1f}%")

    print("\n[7] 逐年（TSMOM+SMA200 vs 等权基准）...")
    nav_a, _ = simulate(px, ret, rule_tsmom_sma)
    for y in range(2019, 2027):
        a = nav_a[nav_a.index.year == y]
        e = nav_eq[nav_eq.index.year == y]
        if len(a) < 2 or len(e) < 2:
            continue
        print(f"     {y}: 策略 {a.iloc[-1]/a.iloc[0]-1:+7.2%} | 等权 {e.iloc[-1]/e.iloc[0]-1:+7.2%}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_timeseries_out.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
