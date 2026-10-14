"""检查 2025-2026 回填数据的真实密度（可交易性），判断 OOS 是否可验证。

铁律 #6 说 2011-2019 过稀零成交、2025-2026 回填未完。
本脚本用 amount>0 的占比 + 每日有效股票数来量化。
"""
import psycopg2
import pandas as pd

c = psycopg2.connect(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
cur = c.cursor()
cur.execute(
    """
    select date_trunc('quarter', trade_date)::date as q,
           count(*) as n,
           count(*) filter (where close is not null) as n_close,
           count(*) filter (where amount > 0) as n_amt,
           count(distinct asset_code) filter (where amount > 0) as codes_amt
    from bars_1d
    where trade_date >= '2023-01-01'
    group by 1 order by 1
    """
)
rows = cur.fetchall()
df = pd.DataFrame(rows, columns=["q", "n", "n_close", "n_amt", "codes_amt"])
df["amt_ratio"] = (df["n_amt"] / df["n"]).round(4)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
c.close()
