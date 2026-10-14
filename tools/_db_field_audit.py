"""检查 bars_1d 的 OHLC 字段完整度与量纲，确认回测数据可用（只读）。"""
import psycopg2

c = psycopg2.connect(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
cur = c.cursor()
for cond, label in [
    ("", "all"),
    ("where trade_date >= '2020-01-01'", "2020+"),
    ("where trade_date >= '2025-01-01'", "2025+"),
]:
    cur.execute(
        f"""
        select count(*) as n,
               count(*) filter (where open is null) as o_null,
               count(*) filter (where high is null) as h_null,
               count(*) filter (where low is null) as l_null,
               count(*) filter (where close is null) as c_null,
               count(*) filter (where volume is null) as v_null,
               count(*) filter (where amount is null) as amt_null,
               count(*) filter (where volume is not null and volume <= 0) as v_le0,
               count(*) filter (where amount is not null and amount <= 0) as amt_le0
        from bars_1d {cond}
        """
    )
    print(label, cur.fetchone())
# 量纲抽查：2026-10-08 若干只
cur.execute(
    """
    select asset_code, open, high, low, close, volume, amount, turnover_rate, float_shares
    from bars_1d where trade_date = '2026-10-08' and asset_type is distinct from 'etf'
    order by amount desc nulls last limit 5
    """
)
for r in cur.fetchall():
    print(r)
c.close()
