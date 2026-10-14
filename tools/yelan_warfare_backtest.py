"""三篇公众号战法的全 A 事件驱动回测
==================================

来源：
  W1 低调使用倍量黄金梯量学三倍量战法升级（将军梯）
  W2 看到大阴不要怕要启动必然会有一根大阳急拉关键是找到多空线
  W3 下雨天有人撑伞有人骂娘【临界密度】主力高度控盘+大阳启动点

★ 数据可用性声明（重要）
  W1 需 OHLC + 成交量            → ✅ 有
  W2 需 OHLC + 成交量            → ✅ 有（"筹码单峰变双峰"❌ 无筹码数据，不做）
  W3 需筹码分布(85%/15%成本线)   → ❌ **本库无筹码数据**，模块1/2/4 不做
     模块3 的"威廉30日位置"        → ✅ 可用 30 日高低点位置替代
     模块5"临界加速点"            → ✅ 均线多头 + MACD 空中加油 + 缩量洗盘
  W3 需换手率                     → ⚠️ turnover_rate 与 float_shares 在
     2025-02~2026-03 整段 NULL，改用 **成交额 / 自身 60 日中位数** 作流动性代理

铁律（全脚本遵守）：
  1. T 日收盘出信号 → T+1 开盘买入 → 最早 T+2 收盘出
  2. 出场：≤8% 止损 / 最长持有 N 日 / 形态破坏（W1 的支撑线破位）
  3. 成本双边 0.25%（另有 0.5%/1% 敏感）
  4. 基准四口径：等权 / 中位数 / 同「滞后60日均量」十分位档
  5. 规模分档用**滞后** 60 日均量排名（每日同期排名是未来函数，已禁用）
  6. IS 2020-2024 / OOS 2025-2026
"""
import json
import os
import tempfile
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
COST = 0.0025
STOP = -0.08
MAX_HOLD = 20
LOAD_START = "2018-01-01"
SIM_START = "2020-01-02"
END = "2026-09-30"
OOS = pd.Timestamp("2025-01-01")


def limit_pct(code):
    """按板块给涨停幅度：主板 10%，创业板/科创板 20%，北交所 30%。"""
    c = code.split(".")[0]
    if c.startswith("688"):
        return 0.20
    if c.startswith("30"):
        return 0.20
    if c.startswith(("8", "4", "9")):
        return 0.30
    return 0.10


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
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "amount", "is_st"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def build_benchmarks(panel):
    g = panel.groupby("code")
    panel["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    d = panel.loc[panel["dret"].notna() & (panel["dret"].abs() < 0.5), ["date", "dret", "bkt"]].copy()
    return {
        "broad": (1 + d.groupby("date")["dret"].mean()).cumprod(),
        "median": (1 + d.groupby("date")["dret"].median()).cumprod(),
        "decile": (1 + d.groupby(["date", "bkt"])["dret"].mean().unstack("bkt")).cumprod(),
    }


# ══════════════════════════════════════════════════════════════════
# W1 将军梯（倍量黄金梯）
# ══════════════════════════════════════════════════════════════════
def w1_junjiangti(codes, o, h, l, c_, v, mul=2.0, lookback=90, min_gap=3):
    """将军柱 = 阳线 且 量 >= mul × MA5(量)。

    形态（逐根 bar 判断，state machine）：
      1. 记录最近的将军柱（阳线 + 倍量）
      2. 连续出现 ≥3 根将军柱，且两柱之间间隔 ≥min_gap 根
      3. 每根将军柱的 **开盘价** 作为支撑线，且逐根抬高
      4. 两柱之间的回调不跌破前一根将军柱的支撑
      5. 当前价回踩到最近一根将军柱支撑附近（回到支撑的 ±3% 内）
    返回 bool 数组 + 支撑价数组。
    """
    n = len(c_)
    is_yang = c_ > o
    vma5 = pd.Series(v).rolling(5, min_periods=5).mean().values
    is_jun = is_yang & (v >= mul * vma5) & np.isfinite(vma5) & (vma5 > 0)

    sig = np.zeros(n, dtype=bool)
    sup = np.full(n, np.nan)

    jun_idx = []       # 最近将军柱的下标
    jun_sup = []       # 对应支撑价（该柱开盘价）
    for i in range(n):
        if is_jun[i]:
            # 与上一根将军柱的间隔
            if not jun_idx or (i - jun_idx[-1]) >= min_gap:
                jun_idx.append(i)
                jun_sup.append(o[i])
            else:
                # 紧贴则替换为更高的那根
                if o[i] > jun_sup[-1]:
                    jun_idx[-1] = i
                    jun_sup[-1] = o[i]
        # 只保留窗口内的
        while jun_idx and (i - jun_idx[0]) > lookback:
            jun_idx.pop(0)
            jun_sup.pop(0)
        if len(jun_idx) >= 3:
            # 支撑逐级抬高（允许小幅容差）
            rising = all(jun_sup[k + 1] >= jun_sup[k] * 0.995 for k in range(len(jun_sup) - 1))
            # 回调不破前柱支撑
            holds = True
            for k in range(len(jun_idx) - 1):
                seg = l[jun_idx[k] + 1: jun_idx[k + 1] + 1]
                if len(seg) and np.nanmin(seg) < jun_sup[k] * 0.97:
                    holds = False
                    break
            last_sup = jun_sup[-1]
            cur = c_[i]
            if rising and holds and np.isfinite(last_sup) and abs(cur - last_sup) / last_sup <= 0.03:
                sig[i] = True
                sup[i] = last_sup
    return sig, sup


# ══════════════════════════════════════════════════════════════════
# W2 多空线大阴急拉
# ══════════════════════════════════════════════════════════════════
def w2_duokongxian(codes, o, h, l, c_, v, pit_drop=-0.03, recover=0.05, touch_min=2, lookback=120):
    """多空线 = 历史被多次触碰的价位（用 rolling 极值的触碰次数近似）。

    简化实现（可量化、无未来函数）：
      1. 多空线取过去 lookback 日的 **MA60**（长期多空分界，近似"多次支撑反弹的价位"）
      2. **坑**：某根 K 线最低价跌破 MA60 达 pit_drop 以上，且是根大阴线（close < open）
      3. **急拉**：其后 1~3 根内出现一根大阳线（涨幅 >= recover），最好触及涨停
      4. 信号日 = 该大阳线当日（T 日收盘出信号 → T+1 开盘买）
    """
    n = len(c_)
    sig = np.zeros(n, dtype=bool)
    ma60 = pd.Series(c_).rolling(60, min_periods=60).mean().values
    for i in range(n):
        if not np.isfinite(ma60[i]) or ma60[i] <= 0:
            continue
        # 找最近 6 根内的坑
        for k in range(max(0, i - 5), i):
            if not np.isfinite(ma60[k]):
                continue
            broke = l[k] < ma60[k] * (1 + pit_drop)
            is_bear = c_[k] < o[k]
            if not (broke and is_bear):
                continue
            # 急拉确认
            if c_[i] / c_[k] - 1 >= recover and c_[i] > o[i]:
                sig[i] = True
                break
    return sig


# ══════════════════════════════════════════════════════════════════
# W3 临界加速点（均线多头 + MACD 空中加油 + 缩量洗盘）
# ══════════════════════════════════════════════════════════════════
def w3_linjie(codes, o, h, l, c_, v, macd_fast=12, macd_slow=26, macd_sig=9):
    n = len(c_)
    s = pd.Series(c_)
    ma5 = s.rolling(5, min_periods=5).mean().values
    ma10 = s.rolling(10, min_periods=10).mean().values
    ma20 = s.rolling(20, min_periods=20).mean().values
    ema_f = s.ewm(span=macd_fast, adjust=False).mean()
    ema_s = s.ewm(span=macd_slow, adjust=False).mean()
    dif = (ema_f - ema_s).values
    dea = pd.Series(dif).ewm(span=macd_sig, adjust=False).mean().values
    hist = (dif - dea) * 2
    vma5 = pd.Series(v).rolling(5, min_periods=5).mean().values
    vma10 = pd.Series(v).rolling(10, min_periods=10).mean().values

    ma_bull = (ma5 > ma10) & (ma10 > ma20)
    dif_gt_dea = dif > dea
    hist_shrink = np.isfinite(hist) & np.isfinite(np.roll(hist, 1)) & (hist > 0) & (hist < np.roll(hist, 1))
    wash = v < vma5 * 1.0
    sig = ma_bull & dif_gt_dea & hist_shrink & wash & np.isfinite(c_)
    return sig.astype(bool)


# ══════════════════════════════════════════════════════════════════
# W3b 威廉 30 日位置（模块3 的可实现部分）
# ══════════════════════════════════════════════════════════════════
def w3b_wr30(codes, o, h, l, c_, v, win=30, thr=0.80, amt_liq=True, amt_win=60):
    n = len(c_)
    hh = pd.Series(h).rolling(win, min_periods=win).max().values
    ll = pd.Series(l).rolling(win, min_periods=win).min().values
    pos = (c_ - ll) / np.where(hh - ll > 0, hh - ll, np.nan)
    cond = pos >= thr
    if amt_liq:
        amt_med = pd.Series(pd.Series(v).rolling(amt_win, min_periods=20).median().values,
                            index=range(n)).values
        cond = cond & (v <= amt_med)
    return (cond & np.isfinite(c_)).astype(bool)


# ══════════════════════════════════════════════════════════════════
# W3c 涨停爆发（板块自适应涨跌停）
# ══════════════════════════════════════════════════════════════════
def w3c_limit_up(codes, o, h, l, c_, v, amt_liq=True, amt_win=60):
    n = len(c_)
    lims = np.array([limit_pct(x) for x in codes])
    prev_c = np.roll(c_, 1)
    prev_c[0] = np.nan
    is_up = (c_ / prev_c - 1) >= (lims * 0.98)
    if amt_liq:
        amt_med = pd.Series(pd.Series(v).rolling(amt_win, min_periods=20).median().values,
                            index=range(n)).values
        is_up = is_up & (v <= amt_med)
    return (is_up & np.isfinite(c_)).astype(bool)


# ══════════════════════════════════════════════════════════════════
# 通用事件驱动回测
# ══════════════════════════════════════════════════════════════════
def simulate(panel, benches, sig_col, sup_col=None, cost=COST,
             stop=STOP, take=None, max_hold=MAX_HOLD,
             only_after=None, only_before=None):
    nav_b = {ts.date(): v for ts, v in benches["broad"].items()}
    nav_m = {ts.date(): v for ts, v in benches["median"].items()}
    nav_d = {b: {ts.date(): v for ts, v in benches["decile"][b].items()} for b in benches["decile"].columns}
    trades = []
    for code, g in panel.groupby("code"):
        g = g.sort_values("date")
        ao = g["open"].values
        ah = g["high"].values
        al = g["low"].values
        ac = g["close"].values
        sig = g[sig_col].values
        sup = g[sup_col].values if sup_col else None
        ad = g["date"].values
        ab = g["bkt"].values
        n = len(g)
        for i in range(n):
            if i + 2 > n - 1:
                break
            if not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                continue
            buy = ao[i + 1]
            b = ab[i + 1]
            last_j = min(i + 1 + max_hold, n - 1)
            exited = False
            ej = last_j
            exit_price = ac[last_j]
            reason = "timeout"
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(ac[j]) or ac[j] <= 0:
                    continue
                hi = ah[j] / buy - 1
                lo = al[j] / buy - 1
                if stop is not None and lo <= stop:
                    exit_price = buy * (1 + stop)
                    ej = j
                    exited = True
                    reason = "stop"
                    break
                if take is not None and hi >= take:
                    exit_price = buy * (1 + take)
                    ej = j
                    exited = True
                    reason = "tp"
                    break
                # W1 形态破坏：收盘跌破最近将军柱支撑
                if sup is not None and sup[i] == sup[i] and np.isfinite(sup[i]) and ac[j] < sup[i] * 0.97:
                    exit_price = ac[j]
                    ej = j
                    exited = True
                    reason = "support_break"
                    break
            if exit_price is None or not np.isfinite(exit_price):
                continue
            ret = exit_price / buy - 1 - cost
            d0 = pd.Timestamp(ad[i + 1]).date()
            d1 = pd.Timestamp(ad[ej]).date()
            entry = pd.Timestamp(ad[i + 1])
            if only_after is not None and entry < only_after:
                continue
            if only_before is not None and entry >= only_before:
                continue
            exb = exm = exd = np.nan
            if d0 in nav_b and d1 in nav_b and nav_b[d0] > 0:
                exb = ret - (nav_b[d1] / nav_b[d0] - 1)
            if d0 in nav_m and d1 in nav_m and nav_m[d0] > 0:
                exm = ret - (nav_m[d1] / nav_m[d0] - 1)
            dm = nav_d.get(b)
            if dm is not None and d0 in dm and d1 in dm and dm[d0] > 0:
                exd = ret - (dm[d1] / dm[d0] - 1)
            trades.append(dict(entry=entry, code=code, reason=reason,
                               ex_broad=exb, ex_median=exm, ex_decile=exd, ret=ret))
    return pd.DataFrame(trades)


def st(df, col):
    v = df[col].dropna().values
    if len(v) == 0:
        return "n=0"
    m = v.mean()
    t = m / (v.std() / np.sqrt(len(v))) if v.std() > 0 else 0.0
    return f"n={len(v):>6,} {m*1e4:+8.2f}bp t={t:+6.2f} 胜率{(v>0).mean():.1%}"


def main():
    print("[1] 加载全 A ...")
    df = load_panel(LOAD_START, END)
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    df = df[df["date"] >= pd.Timestamp(SIM_START)].copy()
    df["amt60_prev"] = grp["amount"].transform(
        lambda s: s.rolling(60, min_periods=60).mean()).groupby(df["code"]).shift(1)
    df["bkt"] = df.groupby("date")["amt60_prev"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    print(f"     {len(df):,} 行 / {df['code'].nunique():,} 只")

    print("[2] 建四口径基准（滞后 60 日均量分档）...")
    benches = build_benchmarks(df)
    for k in ["broad", "median"]:
        nv = benches[k]
        yrs = len(nv) / 252
        print(f"     {k}: 年化 {(nv.iloc[-1]**(1/yrs)-1)*100:+.2f}%")

    print("\n[3] 逐股计算信号 ...")
    sig_cols = {}
    sup_cols = {}
    for code, g in df.groupby("code"):
        codes_arr = np.repeat(code, len(g))
        o = g["open"].values; h = g["high"].values; l = g["low"].values
        c = g["close"].values; v = g["volume"].values
        s1, sup1 = w1_junjiangti(codes_arr, o, h, l, c, v, mul=2.0)
        s1b, _ = w1_junjiangti(codes_arr, o, h, l, c, v, mul=2.5)
        s2 = w2_duokongxian(codes_arr, o, h, l, c, v)
        s3 = w3_linjie(codes_arr, o, h, l, c, v)
        s3b = w3b_wr30(codes_arr, o, h, l, c, v)
        s3c = w3c_limit_up(codes_arr, o, h, l, c, v)
        sig_cols[code] = dict(W1_mul2=s1, W1_mul25=s1b, W2_pit=s2,
                              W3_linjie=s3, W3b_wr30=s3b, W3c_limitup=s3c)
        sup_cols[code] = sup1
    print(f"     信号计算完成，{len(sig_cols)} 只")

    # 把信号贴回 panel
    names = ["W1_mul2", "W1_mul25", "W2_pit", "W3_linjie", "W3b_wr30", "W3c_limitup"]
    for nm in names:
        df[nm] = False
        df["sup_W1"] = np.nan
    for code, g in df.groupby("code"):
        idx = g.index
        for nm in names:
            df.loc[idx, nm] = sig_cols[code][nm]
        df.loc[idx, "sup_W1"] = sup_cols[code]
    for nm in names:
        print(f"     {nm}: 信号天数 {int(df[nm].sum()):,}")

    out = {}
    print("\n[4] IS(2020-2024) / OOS(2025-2026) 四口径 ...")
    for nm in names:
        sup = "sup_W1" if nm.startswith("W1") else None
        a = simulate(df, benches, nm, sup, only_before=OOS)
        b = simulate(df, benches, nm, sup, only_after=OOS)
        print(f"  {nm}")
        print(f"     IS  [等权] {st(a,'ex_broad')} | [中位] {st(a,'ex_median')} | [同档] {st(a,'ex_decile')}")
        print(f"     OOS [等权] {st(b,'ex_broad')} | [中位] {st(b,'ex_median')} | [同档] {st(b,'ex_decile')}")
        out[nm] = {"is_n": int(len(a)), "oos_n": int(len(b)),
                   "is_decile": float(a["ex_decile"].dropna().mean()) * 1e4 if not a.empty else None,
                   "oos_decile": float(b["ex_decile"].dropna().mean()) * 1e4 if not b.empty else None,
                   "is_median": float(a["ex_median"].dropna().mean()) * 1e4 if not a.empty else None,
                   "oos_median": float(b["ex_decile"].dropna().mean()) * 1e4 if not b.empty else None}
        if not a.empty:
            for r, gg in a.groupby("reason"):
                print(f"        IS 出场[{r}]: n={len(gg):,} {gg['ex_decile'].mean()*1e4:+.2f}bp")

    print("\n[5] 成本敏感性（同档口径, IS / OOS）...")
    for nm in names:
        sup = "sup_W1" if nm.startswith("W1") else None
        line = f"     {nm:14s}: "
        for cost in [0.0025, 0.005, 0.01]:
            a = simulate(df, benches, nm, sup, cost=cost, only_before=OOS)
            b = simulate(df, benches, nm, sup, cost=cost, only_after=OOS)
            ia = a["ex_decile"].mean() * 1e4 if not a.empty else 0
            ib = b["ex_decile"].mean() * 1e4 if not b.empty else 0
            line += f"{cost*100:.2f}%: IS{ia:+8.1f}/OOS{ib:+8.1f} | "
        print(line)
        for cost in [0.005]:
            a = simulate(df, benches, nm, sup, cost=cost, only_before=OOS)
            b = simulate(df, benches, nm, sup, cost=cost, only_after=OOS)
            out[nm][f"cost5_is"] = float(a["ex_decile"].dropna().mean()) * 1e4 if not a.empty else None
            out[nm][f"cost5_oos"] = float(b["ex_decile"].dropna().mean()) * 1e4 if not b.empty else None

    print("\n[6] 逐年（同档口径）...")
    for nm in names:
        sup = "sup_W1" if nm.startswith("W1") else None
        tr = simulate(df, benches, nm, sup)
        line = f"     {nm:14s}: "
        for y in range(2020, 2027):
            s = tr[tr["entry"].dt.year == y]
            if s.empty:
                continue
            line += f"{y}:{s['ex_decile'].mean()*1e4:+7.0f}(n={len(s)}) "
        print(line)
        out[nm]["yearly"] = {int(y): float(g["ex_decile"].dropna().mean()) * 1e4
                             for y, g in tr.groupby(tr["entry"].dt.year) if not g.empty}

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "yelan_warfare_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()