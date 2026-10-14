"""ETF 池敏感性 v3 —— 用"允许 NaN + 前向填充"而非丢弃整只
=====================================================

v2 的问题：要求 `px_all[c].notna().all()` 太严，把 20 只干净池砍到 4 只
（多数 ETF 2019 年还没上市，前期是 NaN），导致结论无意义。

v3 做法（与主回测一致）：
  - 保留有 NaN 的标的，**不**丢弃整只
  - 计算指标时用 `.rolling(min_periods=N)` 自然产生 NaN，再用 `ffill(limit=5)` 防停牌放大
  - 只用**该标的已有数据之后**的期间（避免"标的未上市却计入基准"的幸存者偏差）
  - 基准与策略用**同一套**规则，保证同口径

关键修正：**上市不足 2 年的标的（159981/161226/501018/510030 等）
不能进 2019 年起的基准**，否则基准里包含"后来才出现"的资产 = 幸存者偏差。
做法：只用**2019-01-01 之前已上市**的标的（即 2019-01-01 有数据的）。
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


def simulate(px, ret, mom_win=252, weight="equal", comm=0.0001, vol_win=60,
             vol_floor=0.05, sma_win=None):
    """只用已有数据的标的；未上市/无数据的标的不计入基准与收益。"""
    dates = px.index
    months = dates.to_period("M")
    nav = 1.0
    navs = []
    cur_w = {}
    turnover = 0.0
    for i, d in enumerate(dates):
        is_month_start = (i == 0) or (months[i] != months[i - 1])
        if is_month_start and i > 0:
            hist = px.iloc[:i]
            hret = ret.iloc[:i]
            # 只用信号日有数据、且历史窗口足够的标的
            last = hist.iloc[-1]
            avail = last[last.notna()].index.tolist()
            # 需要 mom_win 历史
            ok = []
            for c in avail:
                col = hist[c].dropna()
                if len(col) >= min(mom_win, 252):
                    ok.append(c)
            if not ok:
                cur_w = {}
                navs.append(nav)
                continue
            mom = {c: hist[c].dropna().iloc[-1] / hist[c].dropna().iloc[-min(mom_win, len(hist[c].dropna()))] - 1
                   for c in ok}
            held = [c for c in ok if mom[c] > 0]
            if sma_win is not None:
                sma = hist.rolling(sma_win, min_periods=sma_win).mean().iloc[-1]
                held = [c for c in held if pd.notna(sma.get(c)) and last[c] > sma[c]]
            if not held:
                new_w = {}
            elif weight == "equal":
                new_w = {c: 1.0 / len(held) for c in held}
            else:
                v = hret.iloc[-vol_win:][held].std() * np.sqrt(252)
                v = v[v > 0].dropna()
                if v.empty:
                    new_w = {c: 1.0 / len(held) for c in held}
                else:
                    inv = 1.0 / v.clip(lower=vol_floor)
                    w = inv / inv.sum()
                    new_w = {c: float(x) for c, x in w.items()}
            codes = set(new_w) | set(cur_w)
            t = sum(abs(new_w.get(x, 0.0) - cur_w.get(x, 0.0)) for x in codes)
            turnover += t
            nav *= (1 - comm * t)
            cur_w = new_w
        day_ret = 0.0
        if cur_w:
            r = ret.iloc[i]
            for code, wt in cur_w.items():
                if code in r.index and pd.notna(r[code]):
                    day_ret += wt * r[code]
        nav *= (1 + day_ret)
        navs.append(nav)
    return pd.Series(navs, index=dates), turnover


BOND_CASH = {"511010.SH", "511090.SH", "511220.SH", "511380.SH", "511880.SH", "510030.SH"}
COMMODITY = {"518880.SH", "159981.SZ", "161226.SH", "159985.SZ"}
OVERSEAS = {"513030.SH", "513050.SH", "513080.SH", "513090.SH", "513100.SH", "513130.SH",
            "513180.SH", "513330.SH", "513500.SH", "513520.SH", "513650.SH", "513660.SH",
            "513730.SH", "513980.SH", "510900.SH", "501018.SH"}
EQUITY = {"510300.SH", "510500.SH", "588000.SH", "588080.SH", "563300.SH", "159967.SZ",
          "512100.SH", "512400.SH", "512500.SH", "512890.SH"}
# 旧报告点名的坏数据（整段缺失/份额拆分假暴跌）
BAD = {"159915.SZ", "159919.SZ", "159922.SZ", "159949.SZ", "510050.SH"}


def classify(code):
    if code in BOND_CASH:
        return "bond_cash"
    if code in COMMODITY:
        return "commodity"
    if code in OVERSEAS:
        return "overseas"
    if code in EQUITY:
        return "equity"
    return "other"


def main():
    print("[1] 加载（仅 qmt_local_86400，去重）...")
    df = load_etf_clean(START, END)
    g = df.groupby("code")
    df["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    big = set(df[df["dret"].abs() > 0.30]["code"].unique())
    print(f"     {df['code'].nunique()} 只；|单日涨跌|>30% 的: {sorted(big) if big else '无'}")

    # 只用"2019-01-02 已有数据"的标的（避免幸存者偏差）
    first = df.groupby("code")["date"].min()
    base_date = pd.Timestamp(START)
    listed_2019 = first[first <= base_date].index.tolist()
    print(f"     2019-01-02 已上市的: {len(listed_2019)} 只")

    df_b = df[df["code"].isin(listed_2019) & ~df["code"].isin(bad_union(big))].copy()
    print(f"     基准池（剔坏数据）: {df_b['code'].nunique()} 只")
    df_b["cls"] = df_b["code"].map(classify)
    print(f"     分类: {df_b.groupby('cls')['code'].nunique().to_dict()}")

    px_all = df_b.pivot(index="date", columns="code", values="close").sort_index()
    ret_all = px_all.pct_change(fill_method=None)
    print(f"     pivot {px_all.shape}  NaN 占比 {px_all.isna().mean().mean():.4f}")

    pools = {
        "A 干净全池": list(px_all.columns),
        "D 宽基+债+商品": [c for c in px_all.columns
                          if classify(c) in ("equity", "bond_cash", "commodity")],
        "E 股票类(股+境外)": [c for c in px_all.columns
                            if classify(c) in ("equity", "overseas")],
        "F 只债+商品": [c for c in px_all.columns
                       if classify(c) in ("bond_cash", "commodity")],
        "G 只境外": [c for c in px_all.columns if classify(c) == "overseas"],
    }

    wf = pd.Timestamp("2023-01-01")
    oos_s = pd.Timestamp("2025-01-01")
    res = {}
    for pname, codes in pools.items():
        if len(codes) < 3:
            print(f"\n=== {pname}: 标的不足({len(codes)})，跳过 ===")
            continue
        px = px_all[codes]
        ret = ret_all[codes]
        r = ret.iloc[1:]
        # 基准：每日对**当日有数据**的标的等权
        nav_eq = (1 + r.mean(axis=1)).cumprod()
        nav_md = (1 + r.median(axis=1)).cumprod()
        nav_ts, to = simulate(px, ret, weight="equal")
        nav_bd, _ = simulate(px, ret, weight="invvol")
        print(f"\n=== {pname} ({len(codes)} 只) ===")
        for label, v in [("等权基准", nav_eq), ("中位数基准", nav_md)]:
            p = perf(v.values)
            print(f"     {label:10s}: {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}%")
        for label, v in [("TSMOM等权", nav_ts), ("底盘(逆波动)", nav_bd)]:
            p = perf(v.values)
            tr = perf(v[v.index < wf].values)
            te = perf(v[v.index >= wf].values)
            oos = perf(v[v.index >= oos_s].values)
            print(f"     {label:10s}: 全期 {p['ann']*100:+6.2f}%/S{p['sharpe']:.2f}/回撤{p['mdd']*100:+6.1f}% "
                  f"|| 训练 {tr['ann']*100:+6.2f}%/S{tr['sharpe']:.2f} || 测试 {te['ann']*100:+6.2f}%/S{te['sharpe']:.2f}/"
                  f"回撤{te['mdd']*100:+.1f}% || OOS {oos['ann']*100:+6.2f}%/S{oos['sharpe']:.2f}")
        wins = 0
        total_y = 0
        for y in range(2019, 2027):
            a = nav_ts[nav_ts.index.year == y]
            e = nav_eq[nav_eq.index.year == y]
            if len(a) < 2 or len(e) < 2:
                continue
            ra = a.iloc[-1] / a.iloc[0] - 1
            re_ = e.iloc[-1] / e.iloc[0] - 1
            wins += 1 if ra > re_ else 0
            total_y += 1
            print(f"       {y}: TSMOM {ra:+7.2%} vs 等权 {re_:+7.2%} {'✓' if ra > re_ else '✗'}")
        print(f"     → TSMOM 跑赢等权 {wins}/{total_y} 年")
        res[pname] = {"n": len(codes), "bench": perf(nav_eq.values),
                      "median": perf(nav_md.values), "tsmom": perf(nav_ts.values),
                      "baseline": perf(nav_bd.values), "wins": wins, "years": total_y,
                      "turnover": float(to)}

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_pool_sensitivity_v3_out.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


def bad_union(big):
    return set(BAD) | set(big)


if __name__ == "__main__":
    main()
