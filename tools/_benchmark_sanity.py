"""校验"同池等权基准"本身是否合理 —— OOS 修正版里 F4/F6 全为负，
先怀疑基准口径而不是信号。

对比三种基准：
  1. 同池等权（当前用）：每日全 A 有效宇宙等权 NAV
  2. 规模代理十分位等权（amount 分档）
  3. 中位数股等权（去极值，防少数暴涨股主导）

如果"同池等权"在 2025-2026 因少数小盘妖股暴涨而虚高，
那么所有负超额都可能是基准假象。
"""
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")

c = psycopg2.connect(**DB)
cur = c.cursor()
cur.execute(
    """
    select b.asset_code, b.trade_date, b.close, b.volume, b.amount,
           coalesce(m.is_st, false) as is_st
    from bars_1d b
    left join asset_master m on m.asset_code = b.asset_code
    where b.trade_date between '2023-06-01' and '2026-09-30'
      and b.asset_type is distinct from 'etf' and b.close is not null
    order by b.asset_code, b.trade_date
    """
)
rows = cur.fetchall()
cur.close()
c.close()
df = pd.DataFrame(rows, columns=["code", "date", "close", "volume", "amount", "is_st"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["code", "date"]).reset_index(drop=True)

g = df.groupby("code")
df["listed_n"] = g.cumcount() + 1
df["dret"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
valid = (df["listed_n"] >= 250) & (~df["is_st"]) & df["dret"].notna() & (df["dret"].abs() < 0.5)
d = df[valid].copy()
d["size_bucket"] = d.groupby("date")["amount"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))

daily_broad = d.groupby("date")["dret"].mean()
daily_med = d.groupby("date")["dret"].median()
daily_top_dec = d[d["size_bucket"] == 9].groupby("date")["dret"].mean()
daily_bot_dec = d[d["size_bucket"] == 0].groupby("date")["dret"].mean()

res = pd.DataFrame({
    "broad": daily_broad,
    "median": daily_med,
    "top_dec_amount": daily_top_dec,
    "bot_dec_amount": daily_bot_dec,
})
nav = (1 + res).cumprod()
pd.set_option("display.width", 220)

print("=== 年化收益（按自然年，复利） ===")
for y in [2023, 2024, 2025, 2026]:
    sub = nav[nav.index.year == y]
    if len(sub) < 2:
        continue
    r = sub.iloc[-1] / sub.iloc[0]
    print(f"{y}: " + "  ".join(f"{k}={v:.4f}({(v-1)*100:+.1f}%)" for k, v in r.items()))

print("\n=== 全期(2023-06~2026-09) 年化 ===")
n_years = len(nav) / 252
for k in nav.columns:
    tot = nav[k].iloc[-1]
    print(f"{k}: 总 {tot:.3f}x  年化 {(tot ** (1 / n_years) - 1) * 100:+.2f}%")

print("\n=== 每日截面收益分布（看是否被少数暴涨股主导） ===")
for y in [2024, 2025, 2026]:
    sub = d[d["date"].dt.year == y]
    if sub.empty:
        continue
    q = sub.groupby("date")["dret"].agg(["mean", "median", lambda s: s.quantile(0.99)])
    q.columns = ["mean", "median", "p99"]
    print(f"{y}: 平均日截面 mean={q['mean'].mean()*1e4:+.1f}bp median={q['median'].mean()*1e4:+.1f}bp "
          f"p99={q['p99'].mean()*1e4:+.0f}bp  →  均值/中位数比 {(q['mean'].mean()/q['median'].mean()):.2f}")
