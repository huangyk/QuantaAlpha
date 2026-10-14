"""用「买入持有同池等权」基准重跑 F4/F5/F6/F7 —— IS + OOS
=========================================================

背景：2026-10-13 发现既有基准「每日横截面复利 NAV」有系统性偏差
（2020-2024：每日再平衡等权 +39~45% 含不可交易再平衡溢价；每日横截面中位数 −87.8% 严重低估；
 而**买入持有同池等权仅 +11.8%**，与上证 +8.6%/沪深300 −5.2% 同量级）。

本脚本用正确基准重跑 4 个战法信号，分 IS(2020-2024) / OOS(2025-2026-10)：
  基准 A 每日再平衡等权（既有）
  基准 B 每日横截面中位数（既有）
  基准 C 同amount档（既有）
  基准 D ★买入持有同池等权（正确）

固定池 = 2020-01-02 已上市标的（保留退市股，退市后按最后价 ffill≈变现）。
规则：T收盘出信号 → T+1开盘买 → 首个{止损≤-8% / 止盈≥+10% / 最长20日}；成本 0.25%。
"""
import json
import os
import tempfile
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
COST, STOP_LOSS, TAKE_PROFIT, MAX_HOLD = 0.0025, -0.08, 0.10, 20
LOAD_START, IS_START, IS_END, OOS_END = "2018-01-01", "2020-01-02", "2024-12-31", "2026-10-13"


def load_panel(start, end):
    c = psycopg2.connect(**DB)
    c.autocommit = True
    cur = c.cursor()
    cur.execute("set max_parallel_workers_per_gather=0")
    cur.execute(
        """select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close, b.volume, b.amount,
                  coalesce(m.is_st,false) as is_st
           from bars_1d b left join asset_master m on m.asset_code=b.asset_code
           where b.trade_date between %s and %s and b.asset_type='stock' and b.close>0
           order by b.asset_code, b.trade_date""", (start, end))
    rows = cur.fetchall()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close",
                                     "volume", "amount", "is_st"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def make_signals(df):
    g = df.groupby("code")
    df["listed_n"] = g.cumcount() + 1
    df["ret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    df["prev_close"] = g["close"].shift(1)
    df["prev2_close"] = g["close"].shift(2)
    for w in (10, 20, 60):
        df[f"sma{w}"] = g["close"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    df["tsmin20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).min())
    df["svol5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    df["max60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).max())
    for col in ["sma10", "sma20", "sma60", "tsmin20", "max60"]:
        df[col] = df.groupby("code")[col].transform(lambda s: s.ffill(limit=5))
    c, p1, p2 = df["close"], df["prev_close"], df["prev2_close"]
    sma10, sma20, sma60 = df["sma10"], df["sma20"], df["sma60"]
    prev_sma20 = df.groupby("code")["sma20"].shift(1)
    df["F4"] = ((c - sma20).abs() / sma20 < 0.03) & (c > sma20) & (p1 <= prev_sma20)
    df["F6"] = (df["tsmin20"] < sma60 * 0.93) & (c > sma20) & (df["volume"] > df["svol5"] * 1.5)
    df["F5"] = (p1 / p2 > 1.095) & (c < p1) & (df["volume"] > df["svol5"] * 1.5)
    df["F7"] = (df["max60"].shift(20) / df["max60"] > 1.25) & (c > sma20) & (c > sma10) & (df["volume"] > df["svol5"] * 1.3)
    return df


def main():
    print("[1] 加载面板 ...")
    df = load_panel(LOAD_START, OOS_END)
    df = make_signals(df)
    p = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    p = p[p["date"] >= pd.Timestamp(IS_START)].copy()
    p["size_bucket"] = p.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    print(f"     有效样本 {len(p):,}  {p['date'].min().date()} ~ {p['date'].max().date()}")

    # ---- 基准 ----
    d = p.loc[p["ret"].notna() & (p["ret"].abs() < 0.5), ["date", "ret", "size_bucket"]]
    nav_ew = (1 + d.groupby("date")["ret"].mean()).cumprod()
    nav_md = (1 + d.groupby("date")["ret"].median()).cumprod()
    nav_dc = (1 + d.groupby(["date", "size_bucket"])["ret"].mean().unstack()).cumprod()
    # 买入持有同池等权：固定池 = 首日已有标的
    d0 = p["date"].min()
    base = p[p["date"] == d0]["code"].unique()
    q = p[p["code"].isin(base)]
    piv = q.pivot_table(index="date", columns="code", values="close").ffill()
    nav_bh = (piv / piv.iloc[0]).mean(axis=1)
    # 退市/停牌统计
    last_valid = piv.notna().idxmax(axis=1)
    print(f"     固定池 {len(base)} 只; 买入持有等权 NAV 终点 {nav_bh.iloc[-1]:.3f} "
          f"({(nav_bh.iloc[-1]-1)*100:+.1f}%)")
    for nm, nav in [("每日再平衡等权", nav_ew), ("每日横截面中位数", nav_md)]:
        print(f"     {nm} NAV 终点 {nav.iloc[-1]:.3f} ({(nav.iloc[-1]-1)*100:+.1f}%)")

    def dmap(nav):
        return {ts.date(): v for ts, v in nav.items()}

    navs = {"A 每日再平衡等权": dmap(nav_ew), "B 每日横截面中位数": dmap(nav_md),
            "C 同amount档": {b: dmap(nav_dc[b]) for b in nav_dc.columns}, "D ★买入持有等权": dmap(nav_bh)}

    # ---- 模拟 ----
    out = {}
    for col in ["F4", "F6", "F5", "F7"]:
        trades = []
        for code, gg in p.groupby("code"):
            gg = gg.sort_values("date")
            ao, ah, al, ac = gg["open"].values, gg["high"].values, gg["low"].values, gg["close"].values
            sig, ad, ab = gg[col].values, gg["date"].values, gg["size_bucket"].values
            n = len(gg)
            for i in range(n):
                if i + 2 > n - 1 or not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                    continue
                buy = ao[i + 1]
                last_j = min(i + 1 + MAX_HOLD, n - 1)
                ej, exit_price = last_j, ac[last_j]
                for j in range(i + 2, last_j + 1):
                    if not np.isfinite(ac[j]) or ac[j] <= 0:
                        continue
                    if al[j] / buy - 1 <= STOP_LOSS:
                        exit_price, ej = buy * (1 + STOP_LOSS), j
                        break
                    if ah[j] / buy - 1 >= TAKE_PROFIT:
                        exit_price, ej = buy * (1 + TAKE_PROFIT), j
                        break
                ret = exit_price / buy - 1 - COST
                dd0, dd1 = pd.Timestamp(ad[i + 1]).date(), pd.Timestamp(ad[ej]).date()
                entry = pd.Timestamp(ad[i + 1])
                b = ab[i + 1]

                def ex(nav):
                    return ret - (nav[dd1] / nav[dd0] - 1) if (dd0 in nav and dd1 in nav and nav[dd0] > 0) else np.nan

                dm = navs["C 同amount档"].get(b)
                trades.append(dict(entry=entry, ret=ret, ex_ew=ex(navs["A 每日再平衡等权"]),
                                   ex_md=ex(navs["B 每日横截面中位数"]),
                                   ex_dc=ex(dm) if dm else np.nan, ex_bh=ex(navs["D ★买入持有等权"])))
        tr = pd.DataFrame(trades)
        print(f"\n  === {col}  交易 {len(tr):,} 笔 ===")
        for seg, lo, hi in [("IS 2020-2024", IS_START, IS_END), ("OOS 2025-2026", "2025-01-01", OOS_END)]:
            sub = tr[(tr["entry"] >= pd.Timestamp(lo)) & (tr["entry"] <= pd.Timestamp(hi))]
            line = f"    {seg:16s} n={len(sub):>6,} | "
            for nm, cc in [("等权", "ex_ew"), ("中位", "ex_md"), ("同档", "ex_dc"), ("★买入持有", "ex_bh")]:
                v = sub[cc].dropna().values
                if not len(v):
                    line += f"{nm} -- | "
                    continue
                m, t = v.mean() * 1e4, v.mean() / (v.std() / np.sqrt(len(v)))
                line += f"{nm} {m:+7.1f}bp(t{t:+6.1f}) | "
                out[f"{col}_{seg}_{cc}"] = dict(n=len(v), mean_bp=float(m), t=float(t))
            print(line)

    out_dir = os.environ.get("QA_OUT_DIR", tempfile.gettempdir())
    path = os.path.join(out_dir, "rebuild_bench_rerun_out.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {path}")


if __name__ == "__main__":
    main()
