"""确认 decile 基准的崩溃是"成员漂移/幸存者偏差"还是真实收益
=========================================================

_decile_anomaly_diag 显示 decile9 NAV 终点 2359x（年化 +211.8%）、
decile0 NAV 0.11（年化 -27.3%）——这不可能是同一市场里的真实分化，
必然是**基准构建缺陷**。

关键证据：decile5 的日成交额中位数从 2020 年 397亿 掉到 2026 年 1.97亿。
"每日按 amount 排名分十档"里，档位成员完全由当天成交额排名决定，
且**当天高成交额本身就是当天大涨/大跌的结果**（因果倒置）→ 基准不可投资。

正确做法：用**滞后的规模属性**分档（过去 60 日平均 amount），
这样档位是稳定的股票属性，基准才是可投资的资产组合。

对比：
  A. 每日 amount 排名分档（旧，错误）
  B. 过去 60 日平均 amount 排名分档（新，稳定属性，用上一日值避免前视）
"""
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")


def report(df, col, label):
    print(f"  --- {label} ---")
    for d in [0, 3, 5, 7, 9]:
        sub = df[(df[col] == d) & df["dret"].notna() & (df["dret"].abs() < 0.5)]
        if sub.empty:
            continue
        r = sub.groupby("date")["dret"].mean()
        if len(r) < 10:
            continue
        nav = (1 + r).cumprod()
        yrs = len(nav) / 252
        print(f"  decile{d}: 日均股票数~{sub.groupby('date')['code'].nunique().mean():.0f} "
              f"日均收益 {r.mean()*1e4:+7.2f}bp NAV终点 {nav.iloc[-1]:9.3f} "
              f"年化 {(nav.iloc[-1]**(1/yrs)-1)*100:+8.2f}%")


def main():
    c = psycopg2.connect(**DB)
    cur = c.cursor()
    cur.execute(
        """
        select b.asset_code, b.trade_date, b.close, b.amount, coalesce(m.is_st,false) as is_st
        from bars_1d b
        left join asset_master m on m.asset_code = b.asset_code
        where b.trade_date between '2019-06-01' and '2026-09-30'
          and b.asset_type is distinct from 'etf' and b.close is not null
        order by b.asset_code, b.trade_date
        """
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    df = pd.DataFrame(rows, columns=["code", "date", "close", "amount", "is_st"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code")
    df["listed_n"] = g.cumcount() + 1
    df = df[(df["listed_n"] >= 250) & (~df["is_st"])].copy()
    df["dret"] = df.groupby("code")["close"].transform(lambda s: s / s.shift(1) - 1)
    print(f"面板 {len(df):,} 行 {df['date'].min().date()}~{df['date'].max().date()}")

    print("\n=== A. 每日 amount 排名分档（旧口径）===")
    df["bkt_daily"] = df.groupby("date")["amount"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    report(df, "bkt_daily", "每日 amount 排名")

    print("\n=== B. 过去 60 日平均 amount 排名分档（新口径，用上一日值）===")
    df["amt60"] = df.groupby("code")["amount"].transform(
        lambda s: s.rolling(60, min_periods=60).mean())
    df["amt60_prev"] = df.groupby("code")["amt60"].shift(1)
    df["bkt_stab"] = df.groupby("date")["amt60_prev"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    report(df, "bkt_stab", "60日平均 amount 排名（滞后）")

    print("\n=== C. 规模属性绝对水平稳定性（amt60 中位数）===")
    for lo, hi, name in [(9, 10, "最大档"), (0, 1, "最小档")]:
        sub = df[df["bkt_stab"] == lo]
        med = sub.groupby("date")["amt60_prev"].median()
        y20 = med[med.index.year == 2020].median()
        y26 = med[med.index.year == 2026].median()
        print(f"  {name}: amt60 中位数 2020 {y20/1e8:.2f}亿 → 2026 {y26/1e8:.2f}亿")

    print("\n=== D. 全A等权（合理性对照）===")
    allr = df[df["dret"].notna() & (df["dret"].abs() < 0.5)].groupby("date")["dret"].mean()
    navall = (1 + allr).cumprod()
    yrs = len(navall) / 252
    print(f"  日均 {allr.mean()*1e4:+.2f}bp NAV终点 {navall.iloc[-1]:.3f} "
          f"年化 {(navall.iloc[-1]**(1/yrs)-1)*100:+.2f}%")


if __name__ == "__main__":
    main()
