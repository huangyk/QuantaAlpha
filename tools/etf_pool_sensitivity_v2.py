"""ETF 池敏感性（修正版）—— 修复混源重复行导致的 NaN
==============================================

上一版 `etf_pool_sensitivity.py` 有两个 bug：
  1. **混源重复行**：库里 ETF 同时有 `qmt` 与 `qmt_local_86400` 两种 source，
     不过滤 source 会导致同一 (code, trade_date) 出现两行 → pivot 出 NaN →
     基准年化全变 nan、TSMOM 也从 +10.38% 掉到 +3.32%。
  2. 坏数据判定只用 |涨跌|>30%，但旧报告的坏数据是**整段缺失**（不是异常单日），
     必须额外查"最长连续有数据的天数占比"。

本版：
  - 只用 `source='qmt_local_86400'`（干净 QMT 回填），按 (code, date) 去重
  - 坏数据判定 = |单日涨跌|>30% **或** 数据覆盖率 <85%（2019-01~2026-09 应有 ~1879 日）
  - 池变体：A 干净全池 / B 含坏数据 / D 宽基+债+商品 / E 股票类(股+境外) / F 只债券+商品
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
MIN_COVER = 0.85


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
            else:
                v = hret.iloc[-vol_win:][held].std() * np.sqrt(252)
                v = v[v > 0]
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
    print("[1] 加载（仅 qmt_local_86400，按 (code,date) 去重）...")
    df = load_etf_clean(START, END)
    n_days = df["date"].nunique()
    print(f"     {df['code'].nunique()} 只 / {n_days} 个交易日")

    g = df.groupby("code")
    df["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    # 覆盖率
    cov = df.groupby("code").size() / n_days
    bad_cov = cov[cov < MIN_COVER].index.tolist()
    big_move = df[df["dret"].abs() > 0.30]["code"].unique().tolist()
    print(f"     覆盖率<{MIN_COVER:.0%} 的: {bad_cov}")
    print(f"     有 |单日涨跌|>30% 的: {big_move}")
    bad = set(bad_cov) | set(big_move)
    print(f"     判定坏数据共 {len(bad)} 只: {sorted(bad)}")

    df_clean = df[~df["code"].isin(bad)].copy()
    print(f"     干净池 {df_clean['code'].nunique()} 只")

    df_clean["cls"] = df_clean["code"].map(classify)
    print(f"     分类: {df_clean.groupby('cls')['code'].nunique().to_dict()}")

    px_all = df_clean.pivot(index="date", columns="code", values="close").sort_index()
    ret_all = px_all.pct_change(fill_method=None)
    print(f"     pivot: {px_all.shape}  NaN 占比 {px_all.isna().mean().mean():.4f}")

    pools = {
        "A 干净全池": list(px_all.columns),
        "D 宽基+债+商品": [c for c in px_all.columns
                          if classify(c) in ("equity", "bond_cash", "commodity")],
        "E 股票类(股+境外)": [c for c in px_all.columns
                            if classify(c) in ("equity", "overseas")],
        "F 只债+商品(防御)": [c for c in px_all.columns
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
        # 只用数据完整的标的（上市满 2 年 + 无 NaN 缺口），避免基准与策略口径不一致
        usable = [c for c in codes if px_all[c].notna().all()]
        if len(usable) < 3:
            print(f"\n=== {pname}: 去 NaN 后仅 {len(usable)} 只，跳过 ===")
            continue
        px = px_all[usable]
        ret = ret_all[usable]
        print(f"\n=== {pname} ({len(usable)} 只完整数据，原 {len(codes)} 只) ===")
        r = ret.iloc[1:]
        nav_eq = (1 + r.mean(axis=1)).cumprod()
        nav_md = (1 + r.median(axis=1)).cumprod()
        nav_ts, to = simulate(px, ret, weight="equal")
        nav_bd, _ = simulate(px, ret, weight="invvol")
        print(f"\n=== {pname} ({px.shape[1]} 只) ===")
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
        res[pname] = {"n": int(px.shape[1]), "bench": perf(nav_eq.values),
                      "median": perf(nav_md.values), "tsmom": perf(nav_ts.values),
                      "baseline": perf(nav_bd.values), "wins": wins, "years": total_y,
                      "turnover": float(to)}

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_pool_sensitivity_v2_out.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
