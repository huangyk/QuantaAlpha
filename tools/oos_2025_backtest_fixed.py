"""F4/F6/F5/F7 全 A 事件驱动回测 —— 2025-2026 真 OOS（修正版）
================================================================

与 tools/oos_2025_backtest.py 的差别（关键修正）：

旧版要求 `float_shares notna`，而 DB 里 2025-02~2026-03 共 14 个月的
float_shares **整段为 NULL**（见 tools/_db_float_shares_gap.py），
导致那 14 个月被整体丢弃，OOS 只剩 33 个入场日、结论不可信。

本版：
  - 不再要求 float_shares；市值分档基准改用 `amount`（成交额，99.7% 非空）
    的横截面十分位做**规模代理档**，并在输出中标注这是代理口径。
  - 覆盖 2024-01-01 ~ 2026-09-30 完整 OOS。
  - 其余规则与 dixi_fullpanel_backtest.py 逐条一致：
      T 日收盘出信号 -> T+1 开盘买 -> 最早 T+2 收盘出
      出场 = 首个 {止损<=-8% / 止盈>=+10% / 最长持有20交易日}
      基准 = 同池等权(每日全 A 有效宇宙等权)
      成本 = 双边 0.25%
      过滤 = 上市>=250日 / 非ST / open>0 / close 非空 / |日收益|<50%
  - 铁律：train 内不选参，本脚本只做 OOS 检验，不调任何参数。

输出：logs/ 下 JSON + 控制台明细。
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
LOAD_START = "2022-06-01"   # 多载 2.5 年喂饱 250日/60日 指标
SIM_START = "2024-01-02"    # 指标暖机后才允许出信号
OOS_START = pd.Timestamp("2025-01-01")


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
          and b.asset_type is distinct from 'etf'
          and b.close is not null
        order by b.asset_code, b.trade_date
        """,
        (start, end),
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "amount", "is_st"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def build_benchmark(panel):
    """同池等权基准(每日全 A 有效宇宙等权) + 规模代理十分位档基准。

    规模代理用 amount(成交额) 横截面十分位。注意：这是**规模代理**，
    不是严格市值档；结论里市值中性一列要带这个限定。
    """
    g = panel.groupby("code")
    panel["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    valid_b = panel["listed_n"] >= 250
    d = panel.loc[valid_b & panel["dret"].notna() & (panel["dret"].abs() < 0.5),
                  ["date", "dret", "size_bucket"]].copy()
    nav_broad = (1 + d.groupby("date")["dret"].mean()).cumprod()
    nav_size = (1 + d.groupby(["date", "size_bucket"])["dret"].mean().unstack("size_bucket")).cumprod()
    return {"broad": nav_broad, "size": nav_size}


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
    sma20 = panel["sma20"]
    sma60 = panel["sma60"]
    sma10 = panel["sma10"]
    tsmin20 = panel["tsmin20"]
    svol5 = panel["svol5"]
    max60 = panel["max60"]
    prev_sma20 = g["sma20"].shift(1)
    panel["F4"] = ((c - sma20).abs() / sma20 < 0.03) & (c > sma20) & (p1 <= prev_sma20)
    panel["F6"] = (tsmin20 < sma60 * 0.93) & (c > sma20) & (panel["volume"] > svol5 * 1.5)
    panel["F5"] = (p1 / p2 > 1.095) & (c < p1) & (panel["volume"] > svol5 * 1.5)
    panel["F7"] = (max60.shift(20) / max60 > 1.25) & (c > sma20) & (c > sma10) & (panel["volume"] > svol5 * 1.3)
    return panel


def simulate(panel, bench, signal_col, only_after=None, only_before=None):
    nav_map = {ts.date(): v for ts, v in bench["broad"].items()}
    size_maps = {b: {ts.date(): v for ts, v in bench["size"][b].items()} for b in bench["size"].columns}
    trades = []
    for code, g in panel.groupby("code"):
        g = g.sort_values("date")
        ao = g["open"].values
        ah = g["high"].values
        al = g["low"].values
        ac = g["close"].values
        sig = g[signal_col].values
        ad = g["date"].values
        ab = g["size_bucket"].values
        n = len(g)
        for i in range(n):
            if i + 2 > n - 1:
                break
            if not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                continue
            buy = ao[i + 1]
            b = ab[i + 1]
            last_j = min(i + 1 + MAX_HOLD, n - 1)
            exited = False
            ej = last_j
            exit_price = ac[last_j]
            reason = "timeout"
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(ac[j]) or ac[j] <= 0:
                    continue
                hi = ah[j] / buy - 1
                lo = al[j] / buy - 1
                if lo <= STOP_LOSS:
                    exit_price = buy * (1 + STOP_LOSS)
                    ej = j
                    exited = True
                    reason = "stop"
                    break
                if hi >= TAKE_PROFIT:
                    exit_price = buy * (1 + TAKE_PROFIT)
                    ej = j
                    exited = True
                    reason = "tp"
                    break
            if exit_price is None or not np.isfinite(exit_price):
                continue
            ret = exit_price / buy - 1 - COST
            d0 = pd.Timestamp(ad[i + 1]).date()
            d1 = pd.Timestamp(ad[ej]).date()
            nv0 = nav_map.get(d0)
            nv1 = nav_map.get(d1)
            if nv0 is None or nv1 is None or nv0 <= 0:
                continue
            ex = ret - (nv1 / nv0 - 1)
            sm = size_maps.get(b)
            if sm is not None:
                sv0 = sm.get(d0)
                sv1 = sm.get(d1)
                exs = ret - (sv1 / sv0 - 1) if (sv0 is not None and sv1 is not None and sv0 > 0) else np.nan
            else:
                exs = np.nan
            entry = pd.Timestamp(ad[i + 1])
            if only_after is not None and entry < only_after:
                continue
            if only_before is not None and entry >= only_before:
                continue
            trades.append(dict(entry=entry, exit=pd.Timestamp(ad[ej]), code=code,
                               ex=ex, exs=exs, ret=ret, reason=reason))
    return pd.DataFrame(trades)


def stats(df):
    if df.empty:
        return dict(n=0)
    ex = df["ex"].values
    exs = df["exs"].dropna().values
    m = ex.mean()
    t = m / (ex.std() / np.sqrt(len(ex))) if ex.std() > 0 else 0.0
    lo_c, hi_c = np.percentile(ex, [1, 99])
    trimmed = ex[(ex >= lo_c) & (ex <= hi_c)]
    return dict(
        n=len(ex),
        mean_bp=float(m * 1e4),
        t=float(t),
        median_bp=float(np.median(ex) * 1e4),
        trimmed_mean_bp=float(trimmed.mean() * 1e4),
        win=float((ex > 0).mean()),
        size_mean_bp=float(exs.mean() * 1e4) if len(exs) else None,
        size_t=float(exs.mean() / (exs.std() / np.sqrt(len(exs)))) if len(exs) and exs.std() > 0 else None,
        n_entry_dates=int(df["entry"].nunique()),
    )


def main():
    print(f"[1] 加载面板 {LOAD_START}~2026-09-30 ...")
    df = load_panel(LOAD_START, "2026-09-30")
    print(f"     行数 {len(df):,}  股票 {df['code'].nunique():,}")
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df["prev2_close"] = grp["close"].shift(2)
    valid = (df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)
    panel = df[valid].copy()
    panel = panel[panel["date"] >= pd.Timestamp(SIM_START)].copy()
    print(f"     有效样本 {len(panel):,}  {panel['date'].min().date()} ~ {panel['date'].max().date()}")
    # 规模代理档：amount 横截面十分位
    panel["size_bucket"] = panel.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    print(f"     amount 非空率 {panel['amount'].notna().mean():.4f}")

    print("[2] 建同池等权 + 规模代理十分位基准 ...")
    bench = build_benchmark(panel)
    nav = bench["broad"]
    print(f"     基准 NAV {nav.index.min().date()}~{nav.index.max().date()} 终点 {nav.iloc[-1]:.4f}")

    print("[3] 算指标 + 信号 ...")
    panel = make_signals(panel)
    for col in ["F4", "F6", "F5", "F7"]:
        print(f"     {col} 信号天数(全A) : {int(panel[col].sum()):,}")

    out = {"note": "size_bucket 用 amount(成交额) 十分位作规模代理；float_shares 在 2025-02~2026-03 整段 NULL，不可用"}
    print("\n[4] OOS(2025-01-01 起入场) 事件驱动回测 ...")
    for col in ["F4", "F6", "F5", "F7"]:
        tr = simulate(panel, bench, col, only_after=OOS_START)
        s = stats(tr)
        out[col] = s
        print(f"     {col}: 笔数 {s['n']:,} ({s['n_entry_dates']} 个入场日) | "
              f"逐笔超额 {s['mean_bp']:+.2f}bp t={s['t']:+.2f} | 中位数 {s['median_bp']:+.1f}bp | "
              f"去极值 {s['trimmed_mean_bp']:+.2f}bp | 胜率 {s['win']:.1%}")
        print(f"          └ 规模代理中性: {s['size_mean_bp']:+.2f}bp t={s['size_t']:+.2f}")
        # 分年
        for y in [2025, 2026]:
            sub = tr[tr["entry"].dt.year == y]
            if sub.empty:
                continue
            sy = stats(sub)
            out[f"{col}_{y}"] = sy
            print(f"          {y}: n={sy['n']:,} ({sy['n_entry_dates']}日) {sy['mean_bp']:+.2f}bp "
                  f"t={sy['t']:+.2f} 中位 {sy['median_bp']:+.1f} 去极值 {sy['trimmed_mean_bp']:+.2f} "
                  f"胜率 {sy['win']:.1%}")
        # 月度
        if not tr.empty:
            mo = tr.groupby(tr["entry"].dt.to_period("M"))["ex"].agg(["count", "mean"])
            mo["mean_bp"] = (mo["mean"] * 1e4).round(1)
            print("          月度:", {str(k): f"n={int(v['count'])}/{v['mean_bp']:+.0f}bp"
                                            for k, v in mo.iterrows()})

    print("\n[5] IS 对照(2024-01~2024-12) —— 只看方向一致性，不选参 ...")
    for col in ["F4", "F6"]:
        tr = simulate(panel, bench, col, only_before=OOS_START)
        s = stats(tr)
        out[f"{col}_is2024"] = s
        print(f"     {col} IS2024: 笔数 {s['n']:,} ({s['n_entry_dates']}日) | "
              f"{s['mean_bp']:+.2f}bp t={s['t']:+.2f} | 中位 {s['median_bp']:+.1f} | "
              f"去极值 {s['trimmed_mean_bp']:+.2f} | 胜率 {s['win']:.1%}")

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "oos_2025_fixed_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
