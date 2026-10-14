"""关键诊断：为什么 2025-02 ~ 2026-03 这 14 个月的 bars_1d 数据在
信号诊断里"消失"了？

现象：_signal_sparsity_diag 的月度表里 2025-02~2026-03 完全没出现，
但 _db_days_per_month 显示这些月每天都有 20+ 个交易日、~4500 只股票。
=> 说明这些行在 load_panel 的过滤条件里被全部丢掉了。
本脚本直接查这些月的原始行，看是哪个字段为 NULL。
"""
import psycopg2

c = psycopg2.connect(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
cur = c.cursor()
for m in ["2025-02-03", "2025-06-03", "2025-11-05", "2026-02-03"]:
    cur.execute(
        """
        select count(*) as n,
               count(*) filter (where close is null) as c_null,
               count(*) filter (where open is null) as o_null,
               count(*) filter (where float_shares is null) as fs_null,
               count(*) filter (where volume is null) as v_null,
               count(*) filter (where high is null) as h_null,
               count(*) filter (where low is null) as l_null,
               count(*) filter (where amount is null) as amt_null
        from bars_1d where trade_date = %s and asset_type is distinct from 'etf'
        """,
        (m,),
    )
    print(m, cur.fetchone())

# 看 asset_master 覆盖
cur.execute(
    """
    select count(*) from asset_master
    """
)
print("asset_master rows:", cur.fetchone())
cur.execute(
    """
    select count(distinct b.asset_code) from bars_1d b
    left join asset_master m on m.asset_code = b.asset_code
    where b.trade_date >= '2025-02-01' and b.trade_date < '2026-04-01'
      and b.asset_type is distinct from 'etf' and m.asset_code is null
    """
)
print("bars codes missing from asset_master (2025-02~2026-03):", cur.fetchone())
c.close()
