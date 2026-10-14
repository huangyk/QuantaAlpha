"""检查 ETF 数据在库情况，为 ETF 池三口径复核做准备。"""
import psycopg2

c = psycopg2.connect(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
cur = c.cursor()
cur.execute("select count(*) from bars_1d where asset_type = 'etf'")
print("etf rows:", cur.fetchone())
cur.execute("select count(distinct asset_code) from bars_1d where asset_type = 'etf'")
print("etf codes:", cur.fetchone())
cur.execute("select min(trade_date), max(trade_date) from bars_1d where asset_type = 'etf'")
print("etf date range:", cur.fetchone())
cur.execute("select source, count(*) from bars_1d where asset_type = 'etf' group by 1")
print("by source:", cur.fetchall())
cur.execute(
    """
    select asset_code, count(*), min(trade_date), max(trade_date)
    from bars_1d where asset_type = 'etf' group by 1 order by 1
    """
)
rows = cur.fetchall()
print(f"etf list ({len(rows)}):")
for r in rows:
    print("   ", r)
c.close()
