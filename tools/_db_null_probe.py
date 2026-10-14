"""看 bars_1d 里 OHLC 为 NULL 的行长什么样（判断是否停牌/未复权占位）。"""
import psycopg2
import pandas as pd

c = psycopg2.connect(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
cur = c.cursor()
cur.execute(
    """
    select asset_code, trade_date, open, high, low, close, volume, amount, asset_type
    from bars_1d
    where close is null and trade_date >= '2025-01-01'
    order by trade_date desc limit 15
    """
)
df = pd.DataFrame(cur.fetchall(), columns=["code", "date", "o", "h", "l", "c", "v", "amt", "type"])
pd.set_option("display.width", 200)
print(df.to_string())
print()
cur.execute(
    """
    select asset_type, count(*) from bars_1d where close is null group by 1
    """
)
print("null-close by type:", cur.fetchall())
cur.execute(
    """
    select count(distinct asset_code) from bars_1d where close is null
    """
)
print("distinct codes with null close:", cur.fetchone())
c.close()
