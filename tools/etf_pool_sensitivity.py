"""ETF 池扩展到"股票宽基 + 更多资产"，检验 TSMOM 是否只是小样本幸运
==============================================================

前几轮 TSMOM 结论基于 36 只干净跨资产 ETF。疑问：
  1. 36 只样本是否太小？加更多 ETF（含非 qmt_local 源的、股票宽基）看结论是否稳。
  2. 现有池缺**A股宽基/行业**（只有 510300/510500/588000/588080），
     加 159901/159902/510880/512880/513500 等能否提升分散？
  3. 不同池下 TSMOM 等权 vs 等权基准 vs 底盘 的排序是否稳定？

池变体：
  A. 36 只（现用，干净 qmt_local）
  B. 全库 ETF（41 只，含 qmt 旧源）
  C. 36 只 + 库里其他可用 ETF（放宽到所有 asset_type='etf' 且有 2 年数据）
  D. 只含宽基/债券/商品（剔境外与行业，看是否依赖某类资产）
  E. 只含股票类（股+境外，剔债+货，看牛熊敏感性）

铁律：三/四口径基准 + walk-forward + 成本敏感，全部列出。
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


def load_all_etf(start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select asset_code, trade_date, close, amount
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


# 资产分类（粗略，用于池变体 D/E）
EQUITY = {"510300.SH", "510500.SH", "588000.SH", "588080.SH", "563300.SH",
          "159901.SZ", "159902.SZ", "510880.SH", "512880.SH", "512100.SH",
          "512400.SH", "512500.SH", "512890.SH", "159967.SZ", "159985.SZ"}
OVERSEAS = {"513030.SH", "513050.SH", "513080.SH", "513090.SH", "513100.SH",
            "513130.SH", "513180.SH", "513330.SH", "513500.SH", "513520.SH",
            "513650.SH", "513660.SH", "513730.SH", "513980.SH", "510900.SH",
            "501018.SH", "161226.SH", "159981.SZ"}
BOND_CASH = {"511010.SH", "511090.SH", "511220.SH", "511380.SH", "511880.SH",
             "510030.SH", "501018.SH"}
COMMODITY = {"518880.SH", "159981.SZ", "161226.SH", "159985.SZ"}


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
    print("[1] 加载全库 ETF ...")
    df = load_all_etf(START, END)
    # 每只至少有 400 个交易日才纳入（保证 SMA200 + 动量 252 有暖机）
    cnt = df.groupby("code").size()
    ok = cnt[cnt >= 400].index.tolist()
    df = df[df["code"].isin(ok)].copy()
    # 剔异常单日收益
    g = df.groupby("code")
    df["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    n_bad = int((df["dret"].abs() > 0.30).sum())
    print(f"     {df['code'].nunique()} 只（≥400日），剔 |涨跌|>30% 的行 {n_bad}")
    df = df[df["dret"].abs() <= 0.30].copy()
    df["cls"] = df["code"].map(classify)
    print(f"     分类: {df.groupby('cls')['code'].nunique().to_dict()}")

    px_all = df.pivot(index="date", columns="code", values="close").sort_index()
    ret_all = px_all.pct_change(fill_method=None)

    pools = {
        "A 36只干净(现用)": [c for c in px_all.columns if c not in BAD],
        "B 全库(含旧源)": list(px_all.columns),
        "D 宽基+债+商品": [c for c in px_all.columns
                          if classify(c) in ("equity", "bond_cash", "commodity")],
        "E 股票类(股+境外)": [c for c in px_all.columns
                            if classify(c) in ("equity", "overseas")],
    }

    wf = pd.Timestamp("2023-01-01")
    oos_s = pd.Timestamp("2025-01-01")

    for pname, codes in pools.items():
        if len(codes) < 3:
            print(f"\n=== {pname}: 标的不足({len(codes)})，跳过 ===")
            continue
        px = px_all[codes]
        ret = ret_all[codes]
        r = ret.iloc[1:]
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
                  f"回撤{te['mdd']*100:+.1f}% || OOS {oos['ann']*100:+6.2f}%/S{oos['sharpe']:.2f} || 换手{to:.0f}")
        # 逐年对比
        for y in range(2019, 2027):
            a = nav_ts[nav_ts.index.year == y]
            e = nav_eq[nav_eq.index.year == y]
            if len(a) < 2 or len(e) < 2:
                continue
            print(f"       {y}: TSMOM {a.iloc[-1]/a.iloc[0]-1:+7.2%} vs 等权 {e.iloc[-1]/e.iloc[0]-1:+7.2%}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "etf_pool_sensitivity_out.json")
    res = {}
    for pname, codes in pools.items():
        if len(codes) < 3:
            continue
        px = px_all[codes]
        ret = ret_all[codes]
        r = ret.iloc[1:]
        nav_eq = (1 + r.mean(axis=1)).cumprod()
        nav_ts, to = simulate(px, ret, weight="equal")
        nav_bd, _ = simulate(px, ret, weight="invvol")
        res[pname] = {"n": len(codes), "bench": perf(nav_eq.values),
                      "tsmom": perf(nav_ts.values), "baseline": perf(nav_bd.values),
                      "turnover": float(to)}
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
