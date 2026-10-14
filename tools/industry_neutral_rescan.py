"""行业中性化重扫 F4/F5/F6/F7 —— 补上"行业"这一层
=====================================================

背景：此前所有超额都用「等权 / 中位数 / 同 amount 十分位」三口径，
结论是正超额几乎全部来自**小盘风格**。但**行业轮动**这一层从未剥离过——
若某信号恰好扎堆在某几个行业（如低波动扎堆银行/公用事业），
正超额可能是行业 beta 而非个股 alpha。

本脚本在既有三口径之外，再加两口径：
  - 同行业：减「同日同 SW1 行业」的中位数收益
  - 同行业+同档：再减「同日同 amount 十分位」的中位数收益

行业映射：TimescaleDB `sector_membership` 的 `SW1*`（申万一级，非加权），
覆盖 5565 只 = 全A。注意只有 2026-07 的当前快照（无历史），
故存在轻微 look-ahead —— 但行业归属年际稳定，且此处仅用于**中性化**而非信号，
可接受；结论若显著才需补历史行业表。

规则与 is_2020_2024_tribench.py 完全一致：
  T 收盘出信号 -> T+1 开盘买 -> 出场 = 首个 {止损<=-8% / 止盈>=+10% / 最长 20 日}
  成本 0.25%，剔次新(<250日)/ST/open<=0；暖机从 2018 起，2020-01-02 起出信号。
"""
import json
import os
import tempfile
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
COST = 0.0025
STOP_LOSS = -0.08
TAKE_PROFIT = 0.10
MAX_HOLD = 20
LOAD_START = "2018-01-01"
SIM_START = "2020-01-02"
END = "2024-12-31"


def load_panel(start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close,
               b.volume, b.amount, coalesce(m.is_st, false) as is_st
        from bars_1d b
        left join asset_master m on m.asset_code = b.asset_code
        where b.trade_date between %s and %s
          and b.asset_type is distinct from 'etf' and b.close is not null
        order by b.asset_code, b.trade_date
        """,
        (start, end),
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close",
                                     "volume", "amount", "is_st"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def load_industry_map():
    """asset_code -> SW1 行业（每只取 start_date 最新的一条）。"""
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select distinct on (asset_code) asset_code, sector_code
        from sector_membership
        where sector_code like 'SW1%' and sector_code not like '%加权%'
        order by asset_code, start_date desc
        """
    )
    m = {r[0]: r[1] for r in cur.fetchall()}
    cur.close()
    c.close()
    return m


def build_benchmarks(panel):
    """四口径基准 NAV：等权 / 中位数 / 同amount档 / 同行业。"""
    d = panel.loc[panel["dret"].notna() & (panel["dret"].abs() < 0.5),
                  ["date", "dret", "size_bucket", "industry"]].copy()
    daily_broad = d.groupby("date")["dret"].mean()
    daily_median = d.groupby("date")["dret"].median()
    daily_dec = d.groupby(["date", "size_bucket"])["dret"].mean().unstack("size_bucket")
    daily_ind = d.groupby(["date", "industry"])["dret"].median().unstack("industry")
    return {
        "broad": (1 + daily_broad).cumprod(),
        "median": (1 + daily_median).cumprod(),
        "decile": (1 + daily_dec).cumprod(),
        "industry": (1 + daily_ind).cumprod(),
    }


def make_signals(panel):
    g = panel.groupby("code")
    panel["sma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    panel["sma60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    panel["sma10"] = g["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    panel["tsmin20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).min())
    panel["svol5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    panel["max60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).max())
    for col in ["sma20", "sma60", "sma10", "tsmin20", "max60"]:
        panel[col] = g[col].transform(lambda s: s.ffill(limit=5))
    c = panel["close"]
    p1 = panel["prev_close"]
    p2 = panel["prev2_close"]
    sma20, sma60, sma10 = panel["sma20"], panel["sma60"], panel["sma10"]
    tsmin20, svol5, max60 = panel["tsmin20"], panel["svol5"], panel["max60"]
    prev_sma20 = g["sma20"].shift(1)
    panel["F4"] = ((c - sma20).abs() / sma20 < 0.03) & (c > sma20) & (p1 <= prev_sma20)
    panel["F6"] = (tsmin20 < sma60 * 0.93) & (c > sma20) & (panel["volume"] > svol5 * 1.5)
    panel["F5"] = (p1 / p2 > 1.095) & (c < p1) & (panel["volume"] > svol5 * 1.5)
    panel["F7"] = (max60.shift(20) / max60 > 1.25) & (c > sma20) & (c > sma10) & (panel["volume"] > svol5 * 1.3)
    return panel


def simulate(panel, benches, signal_col, year=None):
    nav_b = {ts.date(): v for ts, v in benches["broad"].items()}
    nav_m = {ts.date(): v for ts, v in benches["median"].items()}
    nav_d = {b: {ts.date(): v for ts, v in benches["decile"][b].items()}
             for b in benches["decile"].columns}
    nav_i = {b: {ts.date(): v for ts, v in benches["industry"][b].items()}
             for b in benches["industry"].columns}
    trades = []
    for code, g in panel.groupby("code"):
        g = g.sort_values("date")
        ao, ah, al, ac = (g["open"].values, g["high"].values,
                          g["low"].values, g["close"].values)
        sig = g[signal_col].values
        ad = g["date"].values
        ab = g["size_bucket"].values
        ai = g["industry"].values
        n = len(g)
        for i in range(n):
            if i + 2 > n - 1:
                break
            if not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                continue
            buy = ao[i + 1]
            b = ab[i + 1]
            ind = ai[i + 1]
            last_j = min(i + 1 + MAX_HOLD, n - 1)
            ej = last_j
            exit_price = ac[last_j]
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(ac[j]) or ac[j] <= 0:
                    continue
                hi = ah[j] / buy - 1
                lo = al[j] / buy - 1
                if lo <= STOP_LOSS:
                    exit_price, ej = buy * (1 + STOP_LOSS), j
                    break
                if hi >= TAKE_PROFIT:
                    exit_price, ej = buy * (1 + TAKE_PROFIT), j
                    break
            if exit_price is None or not np.isfinite(exit_price):
                continue
            ret = exit_price / buy - 1 - COST
            d0 = pd.Timestamp(ad[i + 1]).date()
            d1 = pd.Timestamp(ad[ej]).date()
            entry = pd.Timestamp(ad[i + 1])
            if year is not None and entry.year != year:
                continue

            def ex(nav):
                return ret - (nav[d1] / nav[d0] - 1) if (d0 in nav and d1 in nav and nav[d0] > 0) else np.nan

            exb = ex(nav_b)
            exm = ex(nav_m)
            dm = nav_d.get(b)
            exd = ex(dm) if dm is not None else np.nan
            di = nav_i.get(ind)
            exi = ex(di) if di is not None else np.nan
            trades.append(dict(entry=entry, ex_broad=exb, ex_median=exm,
                               ex_decile=exd, ex_industry=exi, ret=ret))
    return pd.DataFrame(trades)


def stats(df, col):
    v = df[col].dropna().values
    if len(v) == 0:
        return dict(n=0)
    m = v.mean()
    t = m / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0.0
    lo_c, hi_c = np.percentile(v, [1, 99])
    trimmed = v[(v >= lo_c) & (v <= hi_c)]
    return dict(n=len(v), mean_bp=float(m * 1e4), t=float(t),
                median_bp=float(np.median(v) * 1e4),
                trimmed_bp=float(trimmed.mean() * 1e4), win=float((v > 0).mean()))


COLS = [("等权", "ex_broad"), ("中位数", "ex_median"),
        ("同amount档", "ex_decile"), ("★同行业", "ex_industry")]


def main():
    print(f"[1] 加载面板 {LOAD_START}~{END} ...")
    df = load_panel(LOAD_START, END)
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df["prev2_close"] = grp["close"].shift(2)
    df["dret"] = grp["close"].transform(lambda s: s / s.shift(1) - 1)

    imap = load_industry_map()
    print(f"     行业映射 {len(imap):,} 只")
    df["industry"] = df["code"].map(imap)

    panel = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    panel = panel[panel["date"] >= pd.Timestamp(SIM_START)].copy()
    panel["size_bucket"] = panel.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    hit = panel["industry"].notna().mean()
    print(f"     有效样本 {len(panel):,}  行业命中率 {hit:.1%}  "
          f"{panel['date'].min().date()} ~ {panel['date'].max().date()}")

    benches = build_benchmarks(panel)
    for k in ["broad", "median"]:
        nav = benches[k]
        print(f"     {k} NAV 终点 {nav.iloc[-1]:.3f}")
    panel = make_signals(panel)
    for col in ["F4", "F6", "F5", "F7"]:
        print(f"     {col} 信号天数 {int(panel[col].sum()):,}")

    out = {}
    print("\n[2] 2020-2024 四口径（新增同行业）...")
    for col in ["F4", "F6", "F5", "F7"]:
        tr = simulate(panel, benches, col)
        print(f"  === {col} 2020-2024 全期 ===")
        for name, c in COLS:
            s = stats(tr, c)
            if s["n"]:
                print(f"    [{name:>9s}] n={s['n']:,} {s['mean_bp']:+8.2f}bp t={s['t']:+7.2f} "
                      f"中位 {s['median_bp']:+8.1f} 去极值 {s['trimmed_bp']:+8.2f} 胜率 {s['win']:.1%}")
                out[f"{col}_all_{c}"] = s
        for y in range(2020, 2025):
            sub = simulate(panel, benches, col, year=y)
            if sub.empty:
                continue
            line = f"    {y}: "
            for name, c in COLS:
                s = stats(sub, c)
                line += f"{name} {s['mean_bp']:+7.1f}(t{s['t']:+5.1f}) | " if s["n"] else f"{name} -- | "
            print(line)
            out[f"{col}_{y}"] = {c: stats(sub, c) for _, c in COLS}

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "industry_neutral_rescan_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
