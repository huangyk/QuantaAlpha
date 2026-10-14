"""统计每月实际交易日数，判断"某月无信号"是不是因为那个月根本没有数据。"""
import psycopg2
import pandas as pd

c = psycopg2.connect(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
cur = c.cursor()
cur.execute(
    """
    select date_trunc('month', trade_date)::date as m,
           count(distinct trade_date) as days,
           min(trade_date) as first_day,
           max(trade_date) as last_day
    from bars_1d
    where trade_date >= '2024-06-01' and asset_type is distinct from 'etf'
    group by 1 order by 1
    """
)
df = pd.DataFrame(cur.fetchall(), columns=["m", "days", "first", "last"])
pd.set_option("display.width", 200)
print(df.to_string(index=False))
c.close()
