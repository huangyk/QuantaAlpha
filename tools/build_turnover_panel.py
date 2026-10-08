"""
build_turnover_panel.py
合并「库内换手率」与「akshare 补的 2025-2026」，产出 2020-2026 统一换手率面板。

数据口径（已实测核对 2026-10-08）：
- bars_1d.turnover_rate：2020-2024 覆盖 ~94%，2025=0%，2026=14.7%，2011-2019=0%。
- bars_1d 另有 float_shares / volume（手）。验证 换手率 = volume*100 / float_shares
  与 turnover_rate 完全吻合（2023 样本 diff=0.0000），可用作精确反算。
- akshare_turnover_top2000.csv.gz：turnover_rate 为「百分比」(3.47=3.47%)，
  instrument 为 002428.SZ 库格式，datetime 为日期字符串。仅覆盖 2025-2026（1486 只）。

合并优先级：库内（直接读 + float_shares 反算）> akshare（÷100）。
输出：data/cache/turnover_unified_2020_2026.parquet （长表：asset_code, trade_date, turnover_rate）
"""
import psycopg2
import pandas as pd
import numpy as np

DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
AK_PATH = "data/cache/akshare_turnover_top2000.csv.gz"
OUT = "data/cache/turnover_unified_2020_2026.parquet"


def load_db():
    """库内统一换手率：优先直接读 turnover_rate，缺失则用 float_shares 反算。"""
    conn = psycopg2.connect(**DB)
    sql = """
        SELECT asset_code,
               trade_date,
               COALESCE(turnover_rate,
                        volume * 100.0 / NULLIF(float_shares, 0)) AS turnover_rate
        FROM bars_1d
        WHERE trade_date >= '2020-01-01'
          AND asset_code NOT LIKE '%.IND'          -- 排除指数
          AND asset_type <> 'index'
    """
    df = pd.read_sql(sql, conn)
    conn.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.rename(columns={"turnover_rate": "turnover_db"})
    return df


def load_ak():
    df = pd.read_csv(AK_PATH, compression="gzip")
    df = df.rename(columns={"datetime": "trade_date", "instrument": "asset_code"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # akshare 换手率是百分比 -> 转小数
    df["turnover_ak"] = df["turnover_rate"] / 100.0
    df = df[["asset_code", "trade_date", "turnover_ak"]]
    return df


def coverage(df, col, label):
    g = df.dropna(subset=[col]).groupby(df["trade_date"].dt.year)
    rows = []
    for yr, sub in g:
        rows.append((int(yr), sub["asset_code"].nunique(), len(sub)))
    print(f"\n[{label}] 每年覆盖（按 turnover 非空）：")
    print(f"  {'year':>6}  {'#assets':>8}  {'#rows':>9}")
    for yr, a, r in rows:
        print(f"  {yr:>6}  {a:>8}  {r:>9}")


def main():
    print("=== 1) 读取库内换手率（2020-2026）===")
    db = load_db()
    print(f"  DB 行数: {len(db):,}  非空: {db['turnover_db'].notna().sum():,}")
    coverage(db, "turnover_db", "库内(直接读+float_shares反算)")

    print("\n=== 2) 读取 akshare 换手率（2025-2026, ÷100）===")
    ak = load_ak()
    print(f"  AK 行数: {len(ak):,}  非空: {ak['turnover_ak'].notna().sum():,}")
    coverage(ak, "turnover_ak", "akshare")

    print("\n=== 3) 合并（库内优先，缺口用 akshare 补）===")
    merged = db.merge(ak, on=["asset_code", "trade_date"], how="outer")
    merged["turnover_rate"] = merged["turnover_db"].fillna(merged["turnover_ak"])
    src = np.where(merged["turnover_db"].notna(), "db",
                   np.where(merged["turnover_ak"].notna(), "ak", "none"))
    merged["source"] = src
    out = merged[["asset_code", "trade_date", "turnover_rate", "source"]].copy()
    out = out.dropna(subset=["turnover_rate"]).sort_values(["asset_code", "trade_date"])
    print(f"  合并后有效行: {len(out):,}")
    print(out["source"].value_counts().to_string())

    # 合并后逐年覆盖
    coverage(out, "turnover_rate", "合并后统一面板")

    print(f"\n=== 4) 写出 {OUT} ===")
    out.to_parquet(OUT, index=False)
    print("  done.")


if __name__ == "__main__":
    main()
