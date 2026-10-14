"""分离 IS/OOS：只跑 2025-2026（新回填段）验证 F4/F7 的样本外表现。

- 数据窗口固定为 2024-06-01 ~ 2026-09-30，保证 2025 起有足量历史（250日上市 + 60日指标）
- 以 2025-01-01 为 OOS 起点（此前从未在该段验证过）
- 复用 dixi_fullpanel_backtest 的全部规则（T+1 开盘买、≤8%止损/+10%止盈/最长20日、
  同池等权 + 同市值档基准、成本 0.25%、剔次新/ST/停牌）
- 铁律：train 内不选参；本脚本不调参，只做 OOS 检验
"""
import json
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
COST = 0.0025
STOP_LOSS = -0.08
TAKE_PROFIT = 0.10
MAX_HOLD = 20
LOAD_START = "2023-01-01"   # 多载 2 年喂饱 250日/60日 指标，避免前段 NaN
SIM_START = "2024-06-03"    # 从这天才允许出信号（保证指标暖机）
OOS_START = pd.Timestamp("2025-01-01")


def load_panel(start, end):
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close,
               b.volume, b.float_shares, coalesce(m.is_st, false) as is_st
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
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "float_shares", "is_st"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def build_benchmark(panel):
    g = panel.groupby("code")
    panel["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    valid_b = panel["listed_n"] >= 250
    d = panel.loc[valid_b & panel["dret"].notna() & (panel["dret"].abs() < 0.5), ["date", "dret", "size_bucket"]].copy()
    daily_broad = d.groupby("date")["dret"].mean()
    nav_broad = (1 + daily_broad).cumprod()
    daily_size = d.groupby(["date", "size_bucket"])["dret"].mean().unstack("size_bucket")
    nav_size = (1 + daily_size).cumprod()
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
    max60_20ago = max60.shift(20)
    panel["F7"] = (max60_20ago / max60 > 1.25) & (c > sma20) & (c > sma10) & (panel["volume"] > svol5 * 1.3)
    return panel


def simulate_trades(panel, bench, signal_col, use_stop=True, only_after=None):
    nav_broad = bench["broad"]
    nav_size = bench["size"]
    nav_map = {ts.date(): v for ts, v in nav_broad.items()}
    size_maps = {b: {ts.date(): v for ts, v in nav_size[b].items()} for b in nav_size.columns}
    trades = []
    for code, g in panel.groupby("code"):
        g = g.sort_values("date")
        arr_open = g["open"].values
        arr_high = g["high"].values
        arr_low = g["low"].values
        arr_close = g["close"].values
        arr_sig = g[signal_col].values
        arr_date = g["date"].values
        arr_bucket = g["size_bucket"].values
        n = len(g)
        for i in range(n):
            if i + 2 > n - 1:
                break
            if not arr_sig[i] or not np.isfinite(arr_open[i + 1]) or arr_open[i + 1] <= 0:
                continue
            buy = arr_open[i + 1]
            b = arr_bucket[i + 1]
            last_j = min(i + 1 + MAX_HOLD, n - 1)
            exited = False
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(arr_close[j]) or arr_close[j] <= 0:
                    continue
                hi = arr_high[j] / buy - 1
                lo = arr_low[j] / buy - 1
                if use_stop:
                    if lo <= STOP_LOSS:
                        exit_price = buy * (1 + STOP_LOSS)
                        ej = j
                        exited = True
                        break
                    if hi >= TAKE_PROFIT:
                        exit_price = buy * (1 + TAKE_PROFIT)
                        ej = j
                        exited = True
                        break
            if not exited:
                ej = last_j
                exit_price = arr_close[ej]
            if exit_price is None or not np.isfinite(exit_price):
                continue
            trade_ret = exit_price / buy - 1 - COST
            d0 = pd.Timestamp(arr_date[i + 1]).date()
            d1 = pd.Timestamp(arr_date[ej]).date()
            nv0 = nav_map.get(d0)
            nv1 = nav_map.get(d1)
            if nv0 is None or nv1 is None or nv0 <= 0:
                continue
            excess = trade_ret - (nv1 / nv0 - 1)
            sm = size_maps.get(b)
            if sm is not None:
                sv0 = sm.get(d0)
                sv1 = sm.get(d1)
                excess_size = trade_ret - (sv1 / sv0 - 1) if (sv0 is not None and sv1 is not None and sv0 > 0) else np.nan
            else:
                excess_size = np.nan
            entry_ts = pd.Timestamp(arr_date[i + 1])
            if only_after is not None and entry_ts < only_after:
                continue
            trades.append((pd.Timestamp(arr_date[ej]), excess, excess_size, trade_ret, entry_ts))
    return trades


def agg(trades):
    if not trades:
        return dict(n=0, pt_mean=0.0, pt_t=0.0, pt_size=0.0, pt_size_t=0.0, win=0.0)
    ex = np.array([t[1] for t in trades])
    exs = np.array([t[2] for t in trades])
    pt_mean = ex.mean()
    pt_t = pt_mean / (ex.std() / np.sqrt(len(ex))) if ex.std() > 0 else 0.0
    m = ~np.isnan(exs)
    if m.sum() > 0:
        ev = exs[m]
        pt_size = ev.mean()
        pt_size_t = pt_size / (ev.std() / np.sqrt(len(ev))) if ev.std() > 0 else 0.0
    else:
        pt_size = pt_size_t = 0.0
    return dict(n=len(trades), pt_mean=float(pt_mean), pt_t=float(pt_t),
                pt_size=float(pt_size), pt_size_t=float(pt_size_t), win=float((ex > 0).mean()))


def yearly(trades):
    df = pd.DataFrame(trades, columns=["exit", "ex", "exs", "ret", "entry"])
    df["yr"] = df["entry"].dt.year
    out = {}
    for yr, g in df.groupby("yr"):
        ex = g["ex"].values
        exs = g["exs"].values[~np.isnan(g["exs"].values)]
        m = ex.mean()
        t = m / (ex.std() / np.sqrt(len(ex))) if ex.std() > 0 else 0.0
        ms = exs.mean() if len(exs) else 0.0
        ts = ms / (exs.std() / np.sqrt(len(exs))) if len(exs) and exs.std() > 0 else 0.0
        out[int(yr)] = dict(n=len(g), pt_bp=float(m * 1e4), pt_t=float(t),
                            pt_size_bp=float(ms * 1e4), pt_size_t=float(ts))
    return out


def main():
    print(f"[1] 加载面板 {LOAD_START}~2026-09-30 ...")
    df = load_panel(LOAD_START, "2026-09-30")
    print(f"     行数 {len(df):,} 股票 {df['code'].nunique():,}")
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df["prev2_close"] = grp["close"].shift(2)
    df["float_mcap"] = df["close"] * df["float_shares"]
    valid = (df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0) & df["float_shares"].notna()
    panel = df[valid].copy()
    panel = panel[panel["date"] >= pd.Timestamp(SIM_START)].copy()
    print(f"     有效样本 {len(panel):,}  起始 {panel['date'].min().date()}  终止 {panel['date'].max().date()}")
    panel["size_bucket"] = panel.groupby("date")["float_mcap"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))

    print("[2] 建同池等权 + 分市值档基准 ...")
    bench = build_benchmark(panel)
    nav = bench["broad"]
    print(f"     基准 NAV 终点 {nav.iloc[-1]:.4f}  区间年化 {(nav.iloc[-1] ** (252 / len(nav)) - 1):+.2%}")

    print("[3] 算指标 + 信号 ...")
    panel = make_signals(panel)
    for col in ["F4", "F6", "F5", "F7"]:
        print(f"     {col} 信号天数(全A) : {int(panel[col].sum()):,}")

    out = {}
    print("[4] OOS(2025-01-01 起入场) 事件驱动回测, 规则同 dixi_fullpanel ...")
    for col in ["F4", "F6", "F5", "F7"]:
        tr = simulate_trades(panel, bench, col, use_stop=True, only_after=OOS_START)
        a = agg(tr)
        out[col] = a
        out[col + "_yearly"] = yearly(tr)
        print(f"     {col}: 笔数 {a['n']:,} | 逐笔超额 {a['pt_mean'] * 1e4:+.2f}bp t={a['pt_t']:+.2f} | "
              f"市值中性 {a['pt_size'] * 1e4:+.2f}bp t={a['pt_size_t']:+.2f} | 胜率 {a['win']:.1%}")
        for y, v in sorted(out[col + "_yearly"].items()):
            print(f"        {y}: n={v['n']:,} {v['pt_bp']:+.2f}bp (t={v['pt_t']:+.2f})  "
                  f"size {v['pt_size_bp']:+.2f}bp (t={v['pt_size_t']:+.2f})")

    print("[5] IS 对照(2024-06~2024-12, 短) —— 仅看方向一致性 ...")
    for col in ["F4", "F7"]:
        tr = simulate_trades(panel, bench, col, use_stop=True,
                             only_after=pd.Timestamp("2024-06-03"))
        tr = [t for t in tr if t[4] < OOS_START]
        a = agg(tr)
        out[col + "_is2024"] = a
        print(f"     {col} IS2024: 笔数 {a['n']:,} | 逐笔 {a['pt_mean'] * 1e4:+.2f}bp t={a['pt_t']:+.2f}")

    # 注意：本会话沙箱下工作区子目录不可写（D:\qqmt\QuantaAlpha\tools|logs 均拒写），
    # 故结果写到平台临时目录，由 agent 转存/展示。
    import tempfile
    import os as _os
    out_dir = _os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    out_path = _os.path.join(out_dir, "oos_2025_out.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"完成 -> {out_path}")

if __name__ == "__main__":
    main()
