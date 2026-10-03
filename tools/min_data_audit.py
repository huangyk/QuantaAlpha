"""分钟级数据完成率审计（真实查 TimescaleDB，不估算、不编造）。

刻意避开两类会把人拖死的查询（本机实测会被 SIGTERM）：
  - `count(DISTINCT asset_code)`：3900 万行上跑不完 → 用 pg_stats.n_distinct 估算
  - `min(ts)/max(ts)`：PK 是 (asset_code, ts)，ts 不是前缀 → 全表扫 → 用 pg_stats 直方图边界
两者都只读系统表，秒级返回。需要精确值时再单独下钻。

用法：
    "D:/Users/lenovo/miniconda3/envs/qmt312/python.exe" tools/min_data_audit.py            # 快（系统表）
    "D:/Users/lenovo/miniconda3/envs/qmt312/python.exe" tools/min_data_audit.py --detail   # 慢：按交易日聚合（后台跑）
"""

from __future__ import annotations

import argparse
import sys

CONN = "host=127.0.0.1 port=5432 dbname=quant_data user=quant password=quant123456"
TABLES = ("bars_1d", "bars_5m", "bars_1m")

# 行数/标的数/区间，全部来自 pg_class + pg_stats（秒级）
FAST_SQL = """
SELECT c.relname,
       c.reltuples::bigint AS est_rows,
       (SELECT s.n_distinct FROM pg_stats s
         WHERE s.tablename = c.relname AND s.attname = 'asset_code') AS est_codes,
       -- histogram_bounds 是 anyarray，SQL 里不能下标（DatatypeMismatch），
       -- 只能整个转成 text 再在 Python 里解析首尾元素。
       (SELECT h.histogram_bounds::text FROM pg_stats h
         WHERE h.tablename = c.relname AND h.attname = 'trade_date') AS d_hist,
       (SELECT h.histogram_bounds::text FROM pg_stats h
         WHERE h.tablename = c.relname AND h.attname = 'ts') AS ts_hist,
       -- last_analyze 不在 pg_stats 里，在 pg_stat_user_tables
       sut.last_analyze::date AS analyzed
FROM pg_class c
LEFT JOIN pg_stat_user_tables sut ON sut.relname = c.relname
WHERE c.relname = ANY(%s) AND c.relkind = 'r'
ORDER BY c.relname;
"""

IDX_SQL = """
SELECT tablename, indexname FROM pg_indexes
WHERE tablename = ANY(%s) ORDER BY tablename, indexname;
"""


def _fmt(v):
    return "-" if v is None else str(v)


def _hist_bounds(hist):
    """pg_stats.histogram_bounds::text -> (min, max)。形如 '{2026-01-05,2026-02-01,...}'。"""
    if not hist:
        return None, None
    body = hist.strip().strip("{}")
    parts = [p.strip().strip('"') for p in body.split(",") if p.strip()]
    if not parts:
        return None, None
    return parts[0], parts[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true", help="按交易日聚合（全表扫描，请后台跑）")
    args = ap.parse_args()

    try:
        import psycopg
    except ImportError:
        print("[ERR] 当前环境无 psycopg，请用 qmt312 环境运行", file=sys.stderr)
        return 2

    with psycopg.connect(CONN) as conn:
        with conn.cursor() as cur:
            cur.execute(FAST_SQL, (list(TABLES),))
            rows = cur.fetchall()
            print(f"{'表':<9}{'估算行数':>13}{'估算标的':>9}  {'trade_date 区间':<26}{'analyzed'}")
            print("-" * 92)
            for (t, est, codes, d_hist, ts_hist, analyzed) in rows:
                dmin, dmax = _hist_bounds(d_hist)
                rng = f"{dmin} ~ {dmax}" if dmin else "(无统计/未 ANALYZE)"
                print(f"{t:<9}{est:>13,}{int(codes) if codes else 0:>9,}  {rng:<26}{_fmt(analyzed)}")

            if args.detail:
                cur.execute(IDX_SQL, (list(TABLES),))
                idx = {}
                for t, i in cur.fetchall():
                    idx.setdefault(t, []).append(i)
                for t in TABLES:
                    print(f"  {t} 索引: {idx.get(t)}")

            if not args.detail:
                print("\n(加 --detail 看索引 + 按交易日的覆盖分布，会全表扫描，请后台跑)")
                return 0

            # ---- 慢查询：按交易日聚合（只在 --detail 且建议后台运行时做）----
            # A 股每交易日理论 K 线数：5m = 48 根（9:30-11:30 + 13:00-15:00），1m = 240 根
            bars_per_day = {"bars_5m": 48, "bars_1m": 240}
            for t in ("bars_5m", "bars_1m"):
                cur.execute(f"""
                    SELECT date_trunc('month', trade_date)::date AS m,
                           count(DISTINCT trade_date) AS days,
                           count(*) AS rows
                    FROM {t} GROUP BY 1 ORDER BY 1;
                """)
                got = cur.fetchall()
                print(f"\n### {t}: {len(got)} 个月份有数据")
                bpd = bars_per_day.get(t, 0)
                for m, days, r in got:
                    # 反推日均标的数：行数 / 天数 / 每天理论K线数
                    codes = int(r / max(days, 1) / bpd) if bpd else 0
                    print(f"  {m}: {days:>2} 个交易日, {r:>11,} 行, 日均约 {codes:>5,} 只标的")
    return 0


if __name__ == "__main__":
    sys.exit(main())
