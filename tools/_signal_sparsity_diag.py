"""诊断为什么 F4/F6 在 2025-02 ~ 2026-03 完全没有信号。

数据是完整的（每月 ~4500 只股票、99.7% 有成交），所以问题在信号条件本身。
本脚本逐月统计：
  1. 满足 |C-MA20|/MA20 < 3% 且 C>MA20 的股票数（F4 第一部分）
  2. 进一步满足金叉（昨收<=昨MA20）的股票数（完整 F4）
  3. 满足 F6 三段的股票数
  4. 20日前60日高 > 今60日高*1.25 的股票数（F7 部分）
目的是判断是"条件太严"还是"指标计算被 NaN/停牌吃掉"。
"""
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")

c = psycopg2.connect(**DB)
cur = c.cursor()
cur.execute(
    """
    select b.asset_code, b.trade_date, b.open, b.high, b.low, b.close, b.volume, b.float_shares,
           coalesce(m.is_st, false) as is_st
    from bars_1d b
    left join asset_master m on m.asset_code = b.asset_code
    where b.trade_date between '2023-01-01' and '2026-09-30'
      and b.asset_type is distinct from 'etf' and b.close is not null
    order by b.asset_code, b.trade_date
    """
)
rows = cur.fetchall()
cur.close()
c.close()
df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "float_shares", "is_st"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["code", "date"]).reset_index(drop=True)
print(f"loaded {len(df):,} rows, {df['code'].nunique():,} codes")

g = df.groupby("code")
df["listed_n"] = g.cumcount() + 1
df["prev_close"] = g["close"].shift(1)
df["prev2_close"] = g["close"].shift(2)
df["float_mcap"] = df["close"] * df["float_shares"]
df = df[(df["listed_n"] >= 250) & (~df["is_st"]) & (df["open"] > 0) & df["float_shares"].notna()].copy()
print(f"valid {len(df):,}")

g = df.groupby("code")
df["sma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
df["sma60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
df["svol5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
df["tsmin20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).min())
df["max60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=60).max())
for col in ["sma20", "sma60", "svol5", "tsmin20", "max60"]:
    df[col] = g[col].transform(lambda s: s.ffill(limit=5))

c_ = df["close"]
p1 = df["prev_close"]
p2 = df["prev2_close"]
sma20 = df["sma20"]
sma60 = df["sma60"]
svol5 = df["svol5"]
tsmin20 = df["tsmin20"]
max60 = df["max60"]
prev_sma20 = g["sma20"].shift(1)

df["f4_a"] = ((c_ - sma20).abs() / sma20 < 0.03) & (c_ > sma20)
df["f4_b"] = df["f4_a"] & (p1 <= prev_sma20)
df["f6"] = (tsmin20 < sma60 * 0.93) & (c_ > sma20) & (df["volume"] > svol5 * 1.5)
df["f7_a"] = (max60.shift(20) / max60 > 1.25)

df["month"] = df["date"].dt.to_period("M")
agg = df.groupby("month").agg(
    n_rows=("close", "size"),
    n_sma20_ok=("sma20", lambda s: int(s.notna().sum())),
    n_f4a=("f4_a", "sum"),
    n_f4b=("f4_b", "sum"),
    n_f6=("f6", "sum"),
    n_f7a=("f7_a", "sum"),
)
agg["f4a_pct"] = (agg["n_f4a"] / agg["n_rows"] * 100).round(2)
pd.set_option("display.width", 220)
print("\n=== 月度信号条件命中（占当日有效股票比） ===")
print(agg.loc["2024-06":].to_string())
