"""确认 2025-02~2026-03 的 float_shares 全为 NULL 的规模与成因。

这解释了为什么 oos_2025_backtest 里那段"无信号"：load_panel 要求
float_shares notna，而这段的 float_shares 整段为 NULL，被全部过滤掉。
"""
import psycopg2

c = psycopg2.connect(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
cur = c.cursor()
cur.execute(
    """
    select date_trunc('month', trade_date)::date as m,
           count(*) as n,
           count(*) filter (where float_shares is null) as fs_null,
           count(*) filter (where float_shares is not null) as fs_ok,
           count(*) filter (where turnover_rate is null) as tr_null
    from bars_1d
    where trade_date >= '2024-06-01' and asset_type is distinct from 'etf'
    group by 1 order by 1
    """
)
for r in cur.fetchall():
    print(r)
print()
cur.execute(
    """
    select min(trade_date), max(trade_date), count(*)
    from bars_1d
    where float_shares is not null and trade_date >= '2025-01-01'
    """
)
print("rows with float_shares in 2025+:", cur.fetchone())
c.close()
