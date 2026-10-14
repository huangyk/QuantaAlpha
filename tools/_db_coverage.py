"""快速检查 bars_1d 数据覆盖情况（只读）。"""
import psycopg2

c = psycopg2.connect(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
cur = c.cursor()
cur.execute("select max(trade_date), count(*) from bars_1d")
print("max_date, total_rows:", cur.fetchone())
cur.execute(
    "select count(distinct trade_date) from bars_1d where trade_date >= '2025-01-01'"
)
print("distinct days since 2025-01-01:", cur.fetchone())
cur.execute(
    """
    select date_trunc('year', trade_date)::date as y, count(*) as rows_, count(distinct asset_code) as codes
    from bars_1d group by 1 order by 1
    """
)
for r in cur.fetchall():
    print(r)
cur.execute("select count(distinct asset_code) from bars_1d")
print("total codes:", cur.fetchone())
c.close()
