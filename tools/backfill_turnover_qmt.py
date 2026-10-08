"""
backfill_turnover_qmt.py — 用 QMT 把 2025-2026 缺失的换手率/流通股补进 bars_1d。

背景（2026-10-08 根因排查）：
- bars_1d.turnover_rate 在 2020-2024 覆盖 ~94%（来自 QMT，经 sync_to_db.py 的
  _add_turnover_rate 取 QMT 的 turnoverRate 列或 volume/floatVol*100 反算）。
- 2025 = 0%、2026 = 14.7% 为空：因为那批行是用 quant_datahub/datahub/qmt.py 进的库，
  其 fetch_bars 只映射 6 个字段、把 turnover_rate/float_shares 过滤掉了。
  —— 不是 QMT 没有，是入库 ETL 漏取。
- 本脚本复用 sync_to_db.py 的同源逻辑，只回填缺口（trade_date>=2025 且 turnover_rate 为 NULL），
  不重拉价格，安全幂等（ON CONFLICT DO UPDATE）。

注意：本脚本依赖 xtdata 历史下载服务可用（QMT 客户端登录且数据服务正常）。
在 WorkBuddy 沙箱里 xtdata.download_history_data2 当前会 segfault，需在 QMT 正常的机器上运行。
运行：python tools/backfill_turnover_qmt.py
"""
import warnings
warnings.filterwarnings("ignore")

import psycopg2
import pandas as pd

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
START = "20250101"
END = "20260930"          # 与 bars_1d 现有 2025-2026 价格区间对齐
BATCH = 200


def gap_codes(cur):
    """取需要回填的标的：2025-2026 且 turnover_rate 为 NULL 的 distinct 代码。"""
    cur.execute("""
        SELECT DISTINCT asset_code FROM bars_1d
        WHERE trade_date >= '2025-01-01' AND turnover_rate IS NULL
        ORDER BY asset_code
    """)
    return [r[0] for r in cur.fetchall()]


def fetch_qmt(codes):
    """从 QMT 取日线 + 换手率 + 流通股。返回 list of record dict。"""
    from xtquant import xtdata
    xtdata.enable_hello = False
    xtdata.connect()
    xtdata.download_history_data2(stock_list=codes, period="1d",
                                  start_time=START, end_time=END, incrementally=True)
    raw = xtdata.get_local_data(field_list=[], stock_list=codes, period="1d",
                                start_time=START, end_time=END,
                                dividend_type="none", fill_data=False)
    records = []
    for code, df in (raw or {}).items():
        if df is None or len(df) == 0:
            continue
        # turnover_rate：优先 QMT 原生 turnoverRate 列
        if "turnoverRate" in df.columns and not df["turnoverRate"].isna().all():
            tr = df["turnoverRate"]
        else:
            tr = pd.Series(index=df.index, dtype=float)  # 缺失则留空，下面用 floatVol 反算
        # 流通股：从合约详情取（只在 turnoverRate 缺失时反算需要）
        detail = {}
        try:
            detail = xtdata.get_instrument_detail(code) or {}
        except Exception:
            detail = {}
        float_vol = None
        for k in ("FloatVolume", "float_volume", "FloatShares", "float_shares"):
            if detail.get(k):
                float_vol = float(detail[k]); break
        for ts, row in df.iterrows():
            d = pd.Timestamp(ts).normalize()
            v = row.get("volume")
            trv = tr.loc[ts] if ts in tr.index else None
            if pd.notna(trv) and trv and trv == trv:
                turnover = float(trv)
            elif float_vol and v:
                # volume 单位可能手/股，统一按 sync_to_db 公式 volume/float_vol*100
                turnover = (float(v) / float_vol) * 100.0
            else:
                turnover = None
            total_v = None
            for k in ("TotalVolume", "total_volume", "TotalShares", "total_shares"):
                if detail.get(k):
                    total_v = float(detail[k]); break
            records.append({
                "asset_code": str(code),
                "ts": d,
                "turnover_rate": turnover,
                "float_shares": float_vol,
                "total_shares": total_v,
                "capital": float_vol,
            })
    return records


def upsert(cur, records):
    sql = """
        INSERT INTO bars_1d (ts, asset_code, turnover_rate, float_shares, total_shares, capital)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (asset_code, ts) DO UPDATE SET
            turnover_rate = COALESCE(EXCLUDED.turnover_rate, bars_1d.turnover_rate),
            float_shares = COALESCE(EXCLUDED.float_shares, bars_1d.float_shares),
            total_shares  = COALESCE(EXCLUDED.total_shares, bars_1d.total_shares),
            capital       = COALESCE(EXCLUDED.capital, bars_1d.capital)
    """
    rows = [(r["ts"], r["asset_code"], r["turnover_rate"], r["float_shares"],
            r["total_shares"], r["capital"]) for r in records]
    cur.executemany(sql, rows)


def coverage(cur):
    cur.execute("""
        SELECT substring(trade_date::text,1,4) yr,
               count(*) FILTER (WHERE turnover_rate IS NOT NULL) notnull,
               count(*) total
        FROM bars_1d WHERE trade_date >= '2025-01-01' GROUP BY 1 ORDER BY 1
    """)
    print("回填后 2025-2026 覆盖：")
    for yr, n, t in cur.fetchall():
        print(f"  {yr}: {n}/{t} ({100.0*n/t:.1f}%)")


def main():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    codes = gap_codes(cur)
    print(f"需回填标的数: {len(codes)}")
    if not codes:
        print("无缺口，无需回填。"); return
    done = 0
    for i in range(0, len(codes), BATCH):
        batch = codes[i:i+BATCH]
        try:
            recs = fetch_qmt(batch)
        except Exception as e:
            print(f"  batch {i}-{i+len(batch)} 失败: {e}")
            continue
        if recs:
            upsert(cur, recs)
            conn.commit()
        done += len(batch)
        print(f"  已处理 {done}/{len(codes)}，本批写入 {len(recs)} 行")
    coverage(cur)
    conn.close()
    print("回填完成。")


if __name__ == "__main__":
    main()
