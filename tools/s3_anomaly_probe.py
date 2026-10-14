"""S3 的致命疑点排查：为什么 decile5 的胜率高达 91%~92.6%？
====================================================

s3_deep_validation 显示 decile5 IS +2548bp(t=75.7) 胜率 91.1%、
OOS +2335bp(t=60.9) 胜率 92.6% —— 胜率高到不合常理，必须查是不是 bug。

疑点：
  1. 同档基准在 decile5 内，如果 decile5 整体在涨，任何信号都"赢"
  2. 信号与基准**用了不同的股票集合**（例如信号股恰好是 decile5 里涨得好的）
  3. 买入价/基准起点日不一致
  4. size_bucket 在信号日与买入日之间跳档（我用的是买入日 b = ab[i+1]）

本脚本逐项排查，并输出 decile5 的原始收益分布（不是超额）。
"""
import os
import tempfile
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
LOAD_START = "2018-01-01"
SIM_START = "2020-01-02"
END = "2026-09-30"
COST = 0.0025
STOP = -0.08
TAKE = 0.10
MAX_HOLD = 20


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


def main():
    print("[1] 加载 ...")
    df = load_panel(LOAD_START, END)
    grp = df.groupby("code")
    df["listed_n"] = grp.cumcount() + 1
    df["prev_close"] = grp["close"].shift(1)
    df = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0)].copy()
    df = df[df["date"] >= pd.Timestamp(SIM_START)].copy()
    df["size_bucket"] = df.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    panel = df[df["size_bucket"] >= 5].copy()

    g = panel.groupby("code")
    panel["sma10"] = g["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    panel["sv20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    prev_sma = g["sma10"].shift(1)
    prev_close = g["close"].shift(1)
    panel["S3"] = (panel["close"] > panel["sma10"]) & \
                  (panel["volume"] < panel["sv20"] * 0.7) & (prev_close < prev_sma)
    panel["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)

    # 基准：decile5 等权 NAV
    d5 = panel[panel["size_bucket"] == 5]
    nav5 = (1 + d5.groupby("date")["dret"].mean()).cumprod()
    print(f"     decile5 等权 NAV {nav5.index.min().date()}~{nav5.index.max().date()} 终点 {nav5.iloc[-1]:.3f}")
    yrs = len(nav5) / 252
    print(f"     decile5 年化 {(nav5.iloc[-1]**(1/yrs)-1)*100:+.2f}%")

    # 模拟
    nav_map = {ts.date(): v for ts, v in nav5.items()}
    trades = []
    for code, gg in panel.groupby("code"):
        gg = gg.sort_values("date")
        ao = gg["open"].values
        ah = gg["high"].values
        al = gg["low"].values
        ac = gg["close"].values
        sig = gg["S3"].values
        ad = gg["date"].values
        ab = gg["size_bucket"].values
        n = len(gg)
        for i in range(n):
            if i + 2 > n - 1:
                break
            if not sig[i] or not np.isfinite(ao[i + 1]) or ao[i + 1] <= 0:
                continue
            buy = ao[i + 1]
            b = ab[i + 1]
            if b != 5:
                continue
            last_j = min(i + 1 + MAX_HOLD, n - 1)
            exited = False
            ej = last_j
            exit_price = ac[last_j]
            for j in range(i + 2, last_j + 1):
                if not np.isfinite(ac[j]) or ac[j] <= 0:
                    continue
                hi = ah[j] / buy - 1
                lo = al[j] / buy - 1
                if lo <= STOP:
                    exit_price = buy * (1 + STOP)
                    ej = j
                    exited = True
                    break
                if hi >= TAKE:
                    exit_price = buy * (1 + TAKE)
                    ej = j
                    exited = True
                    break
            if exit_price is None or not np.isfinite(exit_price):
                continue
            ret = exit_price / buy - 1 - COST
            d0 = pd.Timestamp(ad[i + 1]).date()
            d1 = pd.Timestamp(ad[ej]).date()
            if d0 not in nav_map or d1 not in nav_map or nav_map[d0] <= 0:
                continue
            bench = nav_map[d1] / nav_map[d0] - 1
            trades.append(dict(entry=pd.Timestamp(ad[i + 1]), code=code,
                               ret=ret, bench=bench, ex=ret - bench, decile=b))
    tr = pd.DataFrame(trades)
    print(f"\n[2] decile5 内 S3 交易 {len(tr):,} 笔")
    print(f"     原始收益  均值 {tr['ret'].mean()*1e4:+8.2f}bp  中位 {tr['ret'].median()*1e4:+8.2f}bp  "
          f"胜率 {(tr['ret']>0).mean():.1%}")
    print(f"     基准收益  均值 {tr['bench'].mean()*1e4:+8.2f}bp  中位 {tr['bench'].median()*1e4:+8.2f}bp")
    print(f"     超额      均值 {tr['ex'].mean()*1e4:+8.2f}bp  中位 {tr['ex'].median()*1e4:+8.2f}bp  "
          f"胜率 {(tr['ex']>0).mean():.1%}")

    # 关键排查：decile5 的股票集合在信号日 vs 全期是否一致
    print("\n[3] 疑点排查 ...")
    sig_days = tr["entry"].nunique()
    print(f"     a) 信号入场日数 {sig_days}，每日 decile5 平均股票数 "
          f"{d5.groupby('date').size().mean():.0f}")
    # decile5 内的日收益分布
    d5r = d5.groupby("date")["dret"].mean()
    print(f"     b) decile5 日收益: 均值 {d5r.mean()*1e4:+.2f}bp 中位 {d5r.median()*1e4:+.2f}bp "
          f"正收益日占比 {(d5r>0).mean():.1%}")
    # 信号股的原始收益 vs decile5 全体原始收益（同期）
    print(f"     c) S3 信号股原始收益均值 {tr['ret'].mean()*1e4:+.2f}bp vs "
          f"decile5 全体原始收益 {d5r.mean()*1e4:+.2f}bp")
    # 检查：信号是否集中在少数股票
    cc = tr["code"].value_counts()
    print(f"     d) 涉及 {len(cc)} 只股票，最多 {cc.iloc[0]} 笔，前10只占 "
          f"{cc.head(10).sum()/len(tr)*100:.1f}%")
    # 检查：超额是否集中在少数日期
    dd = tr.groupby("entry")["ex"].mean()
    print(f"     e) 超额最好5日 {[f'{v*1e4:+.0f}bp' for v in dd.sort_values(ascending=False).head(5).values]}")
    print(f"        超额最差5日 {[f'{v*1e4:+.0f}bp' for v in dd.sort_values().head(5).values]}")
    print(f"        超额>0 的日期占比 {(dd>0).mean():.1%}")
    # 关键：decile5 的原始收益是否本身就极高（正漂移）
    print(f"\n     f) ★decile5 原始收益中位数 {tr['ret'].median()*1e4:+.2f}bp，"
          f"但胜率 {(tr['ret']>0).mean():.1%} → 中位与胜率矛盾，说明右尾极厚")
    q = tr["ret"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    for k, v in q.items():
        print(f"        原始收益分位 {k:.2f}: {v*1e4:+9.1f}bp")


if __name__ == "__main__":
    main()
