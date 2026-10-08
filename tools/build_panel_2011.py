"""把 TimescaleDB 导出的 CSV 转成宽表面板 parquet（2011-2026 全 A）。

为什么用 parquet 而不是 qlib bin：
    1. 最近所有实验都是纯 pandas 计算（战法 / 分池 / 权重层），qlib 的因子引擎用不上
    2. qlib bin 只支持标准字段，**装不下 amount / float_shares / turnover_rate**
       —— 而这三个字段正是这次要用的（真实成交额、流通股本、换手率）
    3. qlib 需要代码格式映射（600000.SH ↔ SH600000）且库里混着指数/ETF/板块指数，
       转换环节容易出错；parquet 直接保留原始代码 + asset_type 过滤已在 SQL 里做完

输出：data/cache/panel_2011.parquet（含 close/open/high/low/volume/amount/
      turnover_rate/float_shares/total_shares 八个 date × instrument 面板）
"""
from __future__ import annotations

from pathlib import Path
import time

import pandas as pd

SRC = Path('data/cache/stock_1d.csv.gz')
OUT = Path('data/cache/panel_2011.parquet')
FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount',
          'turnover_rate', 'float_shares', 'total_shares']


def main():
    if OUT.exists():
        print(f"缓存已存在：{OUT}（{OUT.stat().st_size/1e6:.0f} MB），跳过转换，直接做质量检查")
        P = pd.read_parquet(OUT)
        P.index = pd.DatetimeIndex(P.index.get_level_values(0))
        P.columns = pd.MultiIndex.from_tuples(
            [tuple(c) if isinstance(c, tuple) else (c, c) for c in P.columns])
        quality(P)
        return
    t0 = time.time()
    print(f"读取 {SRC} ...")
    df = pd.read_csv(SRC, parse_dates=['trade_date'])
    print(f"  {len(df):,} 行 × {df.shape[1]} 列，耗时 {time.time()-t0:.0f}s")

    df = df.rename(columns={'asset_code': 'instrument'})
    df = df.set_index(['trade_date', 'instrument']).sort_index()

    print("构建宽表面板 ...")
    panels = {}
    for f in FIELDS:
        s = df[f].unstack('instrument')
        panels[f] = s
        print(f"  {f:<14s} {s.shape}  非NaN {int(s.notna().sum().sum()):>12,}")
    P = pd.concat(panels, axis=1)
    P = P.sort_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    P.to_parquet(OUT)
    print(f"\n保存 {OUT}（{OUT.stat().st_size/1e6:.0f} MB）  总耗时 {time.time()-t0:.0f}s")

    quality(P)


def quality(P):
    print("\n=== 数据质量检查 ===")
    print(f"  区间: {P.index.min().date()} ~ {P.index.max().date()}")
    print(f"  交易日数: {len(P.index)}")
    print(f"  标的数: {P.columns.get_level_values(1).nunique()}")
    print("\n  逐年覆盖（当日有行情的标的数中位 / 当日全池成交额中位）：")
    cnt = P['close'].notna().sum(axis=1)
    amt_by_day = P['amount'].replace(0, pd.NA).median(axis=1)
    dfq = pd.DataFrame({'n': cnt, 'amt': amt_by_day})
    dfq['year'] = dfq.index.year
    g = dfq.groupby('year')
    for y, row in g:
        print(f"    {y}: 标的 {int(row['n'].median()):>5d} 只   "
              f"成交额中位 {row['amt'].median()/1e8:>8.2f} 亿")

    print("\n  ⚠️ 量纲校验（amount ÷ (close×volume×100)）：")
    chk = (P['amount'] / (P['close'] * P['volume'] * 100)).replace([float('inf')], pd.NA)
    med = chk.median()
    print(f"    逐股比值中位数的分布：P5={med.quantile(.05):.2f}  中位={med.median():.2f}  "
          f"P95={med.quantile(.95):.2f}   （应 ≈1.00；若 ≈100 则 volume 单位是「股」）")

    print("\n  股本字段可用性（这是 qlib 数据集里没有的）：")
    for f in ['float_shares', 'total_shares', 'turnover_rate']:
        nn = int(P[f].notna().sum().sum())
        print(f"    {f:<14s} 非NaN {nn:>12,}  覆盖率 {nn/P[f].size:.1%}")
    # 有股本数据的标的数
    have = P['float_shares'].notna().sum(axis=1)
    print(f"    每日有 float_shares 的标的数：中位 {int(have.median())} 只")


if __name__ == '__main__':
    main()