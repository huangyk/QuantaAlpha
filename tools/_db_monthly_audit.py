"""OOS 段"信号日"极度稀疏的根因诊断 + 数据可用性审计。

发现：F4/F6 在 2025-01 ~ 2026-07 只有 33 个入场日，且 2025 年只有 1 月有信号
（41笔/18笔，全负），之后 2025-02 ~ 2026-03 完全没有信号 → 这不合常理，
必须查是"数据缺"还是"信号条件真的没触发"。
"""
import numpy as np
import pandas as pd
import psycopg2

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")

c = psycopg2.connect(**DB)
cur = c.cursor()
cur.execute(
    """
    select date_trunc('month', trade_date)::date as m,
           count(*) as rows_,
           count(*) filter (where close is not null) as n_close,
           count(distinct asset_code) filter (where close is not null) as codes,
           count(*) filter (where volume > 0) as n_vol
    from bars_1d
    where trade_date >= '2024-06-01' and asset_type is distinct from 'etf'
    group by 1 order by 1
    """
)
df = pd.DataFrame(cur.fetchall(), columns=["month", "rows", "n_close", "codes", "n_vol"])
pd.set_option("display.width", 200)
print("=== bars_1d 月度可用性 (2024-06 起) ===")
print(df.to_string(index=False))
c.close()
