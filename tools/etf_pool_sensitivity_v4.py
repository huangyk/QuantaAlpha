"""ETF 池敏感性 v4 —— 正确做法：允许标的后期加入，但只用其已有数据
==============================================================

v3 的错：要求"2019-01-02 已上市"，只剩 5 只，完全失去意义。

正确做法（业界标准，也免幸存者偏差）：
  - 标的可以**后期上市**，但从其**首个有效数据日**起才计入基准与可选池
  - 基准 = 每日对"当日已有数据"的标的等权（自然形成，无未来信息）
  - 策略同样只用"当日已有数据"的标的
  - 这样既有 36 只的分散度，又不含"未来才出现的资产"的偏差

本版在 36 只干净池（剔 5 只坏数据）上做池变体敏感性：
  A 干净全池（36 只，后期上市的从上市日起算）
  B 加回坏数据（41 只，看坏数据影响）
  D 宽基+债+商品
  E 股票类（股+境外）
  F 只债+商品
  G 只境外
  H 剔除债券（只看风险资产，看 TSMOM 择时是否还有效）
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


def load_etf_clean(start, end, only_clean_source=True):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    if only_clean_source:
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
    else:
        cur.execute(
            """
            select distinct on (asset_code, trade_date)
                   asset_code, trade_date, close, amount
            from bars_1d
            where trade_date between %s and %s and asset_type = 'etf'
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
    """标的后期上市：从其首个有效数据日起才可选；基准与策略同口径。"""
    dates = px.index
    months = dates.to_period("M")
    nav = 1.0
    navs = []
    cur_w = {}
    turnover = 0.0
    # 每只标的的首个有效数据位置
    first_valid = {c: px[c].first_valid_index() for c in px.columns}
    for i, d in enumerate(dates):
        is_month_start = (i == 0) or (months[i] != months[i - 1])
        if is_month_start and i > 0:
            hist = px.iloc[:i]
            hret = ret.iloc[:i]
            # 信号日有数据、且历史窗口足够的标的
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
            if sma_win is not None:
                sma = hist.rolling(sma_win, min_periods=sma_win).mean().iloc[-1]
                last = hist.iloc[-1]
                held = [c for c in held if pd.notna(sma.get(c)) and last[c] > sma[c]]
            if not held:
                new_w = {}
            elif weight == "equal":
                new_w = {c: 1.0 / len(held) for c in held}
            else:
                sub = hret.iloc[-vol_win:][held]
                v = sub.std() * np.sqrt(252)
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


def run_pool(pname, df):
    px = df.pivot(index="date", columns="code", values="close").sort_index()
    ret = px.pct_change(fill_method=None)
    r = ret.iloc[1:]
    # 基准：每日对当日有数据的标的等权
    nav_eq = (1 + r.mean(axis=1)).cumprod()
    nav_md = (1 + r.median(axis=1)).cumprod()
    nav_ts, to = simulate(px, ret, weight="equal")
    nav_bd, _ = simulate(px, ret, weight="invvol")
    wf = pd.Timestamp("2023-01-01")
    oos_s = pd.Timestamp("2025-01-01")
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
    print(f"     → TSMOM 跑赢等权 {wins}/{total_y} 年")
    return {"n": int(px.shape[1]), "bench": perf(nav_eq.values), "median": perf(nav_md.values),
            "tsmom": perf(nav_ts.values), "baseline": perf(nav_bd.values),
            "wins": wins, "years": total_y, "turnover": float(to)}


def main():
    print("[1] 加载（仅 qmt_local_86400，去重，剔 5 只坏数据）...")
    df = load_etf_clean(START, END)
    df = df[~df["code"].isin(BAD)].copy()
    df["cls"] = df["code"].map(classify)
    print(f"     {df['code'].nunique()} 只  分类 {df.groupby('cls')['code'].nunique().to_dict()}")

    res = {}
    res["A_clean_all"] = run_pool("A 干净全池", df)
    res["D_eq_bond_cmd"] = run_pool(
        "D 宽基+债+商品", df[df["cls"].isin(["equity", "bond_cash", "commodity"])])
    res["E_equity_overseas"] = run_pool(
        "E 股票类(股+境外)", df[df["cls"].isin(["equity", "overseas"])])
    res["F_bond_cmd"] = run_pool(
        "F 只债+商品", df[df["cls"].isin(["bond_cash", "commodity"])])
    res["G_overseas"] = run_pool("G 只境外", df[df["cls"] == "overseas"])
    res["H_no_bond"] = run_pool(
        "H 剔除债券(风险资产)", df[df["cls"] != "bond_cash"])

    print("\n[2] 汇总对比 ...")
    print(f"     {'池':24s} {'只':>3s} {'基准年化':>9s} {'基准S':>6s} {'基准回撤':>8s} "
          f"{'TSMOM年化':>9s} {'TSMOM_S':>7s} {'TSMOM回撤':>9s} {'跑赢':>5s}")
    for k, v in res.items():
        b = v["bench"]
        t = v["tsmom"]
        print(f"     {k:24s} {v['n']:>3d} {b['ann']*100:>+8.2f}% {b['sharpe']:>6.2f} {b['mdd']*100:>+7.1f}% "
              f"{t['ann']*100:>+8.2f}% {t['sharpe']:>7.2f} {t['mdd']*100:>+8.1f}% "
              f"{v['wins']:>2d}/{v['years']}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_pool_sensitivity_v4_out.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
