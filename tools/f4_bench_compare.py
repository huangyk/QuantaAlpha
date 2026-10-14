"""F4 在「正确基准」下的超额 —— 校验既有基准偏差的影响
=====================================================

背景（2026-10-13 发现）：
既有所有回测的基准都用「每日横截面复利 NAV」：
  - 每日再平衡等权  2020-2024 5年 +44.6%
  - 每日横截面中位数 2020-2024 5年 **-87.3%**
但**真实的买入持有等权**只有 **+11.8%**（与上证 +8.6% / 沪深300 -5.2% 同量级）。

偏差来源：
  1. 每日再平衡等权含**再平衡溢价**（A股个股波动大，≈σ²/2×N），高估基准 ~33pp；
     该溢价需每日全额调仓才可得，A股 T+1 + 成本下**不可交易**。
  2. 每日横截面中位数是**不可交易指标**且系统性低估（60% 交易日逾半数个股下跌）。

本脚本对 F4 交易用 4 个基准算超额，看结论是否随基准改变：
  A. 每日再平衡等权（既有）
  B. 每日横截面中位数（既有）
  C. **买入持有等权**（固定池，close/首日 → 均值）★正确
  D. 逐笔配对：每笔 ret − 同期全池个股的平均区间收益

若 F4 在 C/D 下仍显著为正 → 真 alpha；若归零 → 既有正超额是基准偏差。
"""
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


def main():
    c = psycopg2.connect(**DB)
    c.autocommit = True
    cur = c.cursor()
    cur.execute("set max_parallel_workers_per_gather=0")
    cur.execute(
        """select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close, b.volume, b.amount
           from bars_1d b
           where b.trade_date between %s and %s and b.asset_type='stock' and b.close>0
           order by b.asset_code, b.trade_date""",
        (LOAD_START, END),
    )
    rows = cur.fetchall()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code")
    df["n"] = g.cumcount() + 1
    df["ret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    df["sma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["sma20"] = df.groupby("code")["sma20"].transform(lambda s: s.ffill(limit=5))
    df["prev_close"] = g["close"].shift(1)
    df["F4"] = ((df["close"] - df["sma20"]).abs() / df["sma20"] < 0.03) & (df["close"] > df["sma20"]) & (df["prev_close"] <= g["sma20"].shift(1))

    p = df[(df["n"] >= 250) & (df["open"] > 0) & (df["date"] >= SIM_START)].copy()
    p["size_bucket"] = p.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))

    # --- 基准 A/B: 每日横截面复利 ---
    d = p.loc[p["ret"].notna() & (p["ret"].abs() < 0.5), ["date", "ret", "size_bucket"]]
    nav_ew = (1 + d.groupby("date")["ret"].mean()).cumprod()
    nav_md = (1 + d.groupby("date")["ret"].median()).cumprod()
    nav_dc = (1 + d.groupby(["date", "size_bucket"])["ret"].mean().unstack()).cumprod()

    # --- 基准 C: 买入持有等权（固定池 = 首日已有标的，close 归一到首日）---
    d0 = p["date"].min()
    base = p[p["date"] == d0]["code"].unique()
    q = p[p["code"].isin(base)].copy()
    piv = q.pivot_table(index="date", columns="code", values="close").ffill()
    piv = piv.dropna(axis=1, how="any")
    nav_bh = (piv / piv.iloc[0]).mean(axis=1)          # 买入持有等权 NAV
    print(f"买入持有等权 {d0.date()}~{p['date'].max().date()} NAV {nav_bh.iloc[-1]:.3f} ({(nav_bh.iloc[-1]-1)*100:+.1f}%)")
    print(f"每日再平衡等权 NAV {nav_ew.iloc[-1]:.3f} ({(nav_ew.iloc[-1]-1)*100:+.1f}%)")
    print(f"每日横截面中位数 NAV {nav_md.iloc[-1]:.3f} ({(nav_md.iloc[-1]-1)*100:+.1f}%)\n")

    nav_bh_d = {ts.date(): v for ts, v in nav_bh.items()}
    nav_ew_d = {ts.date(): v for ts, v in nav_ew.items()}
    nav_md_d = {ts.date(): v for ts, v in nav_md.items()}
    nav_dc_d = {b: {ts.date(): v for ts, v in nav_dc[b].items()} for b in nav_dc.columns}

    # 逐笔配对基准 D: 同日同池个股平均区间收益 —— 用 nav_bh 的比值即等价，故 D≡C
    trades = []
    for code, gg in p.groupby("code"):
        gg = gg.sort_values("date")
        ao, ah, al, ac = gg["open"].values, gg["high"].values, gg["low"].values, gg["close"].values
        sig = gg["F4"].values
        ad = gg["date"].values
        ab = gg["size_bucket"].values
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
            b = ab[i + 1]

            def ex(nav):
                return ret - (nav[dd1] / nav[dd0] - 1) if (dd0 in nav and dd1 in nav and nav[dd0] > 0) else np.nan

            dm = nav_dc_d.get(b)
            trades.append(dict(ex_ew=ex(nav_ew_d), ex_md=ex(nav_md_d),
                               ex_bh=ex(nav_bh_d), ex_dc=ex(dm) if dm else np.nan,
                               hold=(pd.Timestamp(ad[ej]) - pd.Timestamp(ad[i + 1])).days))
    tr = pd.DataFrame(trades)
    print(f"F4 交易数 {len(tr):,}  平均持有 {tr['hold'].mean():.1f} 天\n")

    def st(col):
        v = tr[col].dropna().values
        if not len(v):
            return None
        m, t = v.mean(), v.mean() / (v.std() / np.sqrt(len(v)))
        lo, hi = np.percentile(v, [1, 99])
        return f"n={len(v):,} {m*1e4:+8.2f}bp t={t:+7.2f} 中位{np.median(v)*1e4:+8.1f} 去极值{v[(v>=lo)&(v<=hi)].mean()*1e4:+8.2f} 胜率{(v>0).mean():.1%}"

    print("=== F4 超额：不同基准 ===")
    for name, col in [("A 每日再平衡等权(既有)", "ex_ew"), ("B 每日横截面中位数(既有)", "ex_md"),
                      ("C ★买入持有等权(正确)", "ex_bh"), ("D 同amount档", "ex_dc")]:
        s = st(col)
        print(f"  [{name:24s}] {s}")


if __name__ == "__main__":
    main()
