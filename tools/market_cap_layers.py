"""市值分层验证（2020-2024，用 DB 的 float_shares）—— 独立检验大小盘风格。

动机：长期验证已确认「流动性分层越低收益越低」且 2021 后转负，
但那混杂了流动性与市值两个因素。DB 的 `float_shares` 提供了真正的市值维度，
可以做一次**独立的大小盘检验**（qlib 数据集里没有这个字段）。

样本限制（必须知道）：
    `float_shares` 在 bars_1d 里**只有 2020-2024 有数据**（每年 80~102 万行），
    2011-2019 与 2025-2026 完全为空 → 本实验最多覆盖 5 年，且正好是
    「抱团白马主导」的行情，结论外推性有限。

分层：按建仓日流通市值（close × float_shares）从大到小分 5 档等权持有。

输出：data/results/market_cap_layers.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path('data/cache/panel_2011.parquet')
OUT = Path('data/results')
ACCOUNT = 1e7
COMMISSION_BP, STAMP_BP, ETA, SIGMA_BP = 5.0, 5.0, 0.1, 342.0
START, END = '2020-01-01', '2024-12-31'
Q = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# ⚠️ 第一个分位段取的是 <= q[0.2]，即**最小** 20%
LABELS = ['MC1_最小20%', 'MC2_20-40%', 'MC3_40-60%', 'MC4_60-80%', 'MC5_最大20%']


def main():
    print("载入面板 ...")
    P = pd.read_parquet(PANEL)
    if not isinstance(P.columns, pd.MultiIndex):
        P.columns = pd.MultiIndex.from_tuples(
            [tuple(c) if isinstance(c, tuple) else (c, c) for c in P.columns])
    P.index = pd.DatetimeIndex(P.index.get_level_values(0))
    P = P.loc[START:END]
    C, O, AMT = P['close'], P['open'], P['amount']
    FS = P['float_shares']

    cap = C * FS / 1e8                      # 流通市值（亿元）
    print(f"  区间 {P.index[0].date()} ~ {P.index[-1].date()}")
    print(f"  有 float_shares 的标的数：中位 {int(FS.notna().sum(axis=1).median())} 只")
    ok = cap.notna().sum(axis=1)
    print(f"  有市值数据的日数：{int((ok > 100).sum())} / {len(P)}")

    ret = (C.shift(-1) / C - 1.0).clip(-0.5, 0.5)
    listing = C.notna().cumsum()
    dates = P.index
    month_starts = [d for d in dates.to_series().groupby(
        [dates.year, dates.month]).min() if ok.get(d, 0) > 100]

    nav = {lb: [] for lb in LABELS}
    used, n_by_lab = [], {lb: 0 for lb in LABELS}

    for k, d0 in enumerate(month_starts):
        d_prev = month_starts[k - 1] if k > 0 else d0
        cap_now = cap.loc[d0].dropna()
        cap_now = cap_now[(cap_now > 0) & (listing.loc[:d_prev].iloc[-1].reindex(cap_now.index) >= 500)]
        cap_now = cap_now[~((O.loc[d0] / C.loc[d0].shift(1) - 1.0).reindex(cap_now.index) >= 0.098)]
        cap_now = cap_now.dropna()
        if len(cap_now) < 100:
            continue
        q = cap_now.quantile(Q)
        for i, lb in enumerate(LABELS):
            lo, hi = q.iloc[i], q.iloc[i + 1]
            sel = cap_now[(cap_now > lo) & (cap_now <= hi)].index if i else cap_now[cap_now <= hi].index
            if len(sel) < 20:
                nav[lb].append(0.0)
                continue
            w = pd.Series(1.0 / len(sel), index=sel)
            n_by_lab[lb] += len(sel)
            if k == 0:
                part = (ACCOUNT * w / AMT.loc[d0].reindex(w.index)).clip(0, 1).fillna(0)
                imp = float((w * ETA * SIGMA_BP * np.sqrt(part)).sum())
                cost = (COMMISSION_BP + STAMP_BP + 2 * imp) / 1e4
            else:
                cost = 0.0
            nav[lb].append(float((w * ret.loc[d0].reindex(w.index).fillna(0.0)).sum()) - cost)
        used.append(d0)

    idx_use = pd.DatetimeIndex(used)

    def metrics(ser):
        ser = ser[ser != 0]
        if len(ser) < 3:
            return None
        ann = (1 + ser).prod() ** (12 / len(ser)) - 1
        cum = (1 + ser).cumprod()
        return {'年化%': ann * 100,
                'IR': ser.mean() / ser.std() * np.sqrt(12) if ser.std() > 0 else np.nan,
                '回撤%': (cum / cum.cummax() - 1).min() * 100,
                '月数': len(ser)}

    print(f"\n=== 流通市值分层 buy & hold（{idx_use[0].date()} ~ {idx_use[-1].date()}，"
          f"{len(idx_use)} 个月） ===")
    print(f"  {'档位':<14s}{'平均只数':>9s}{'市值中位(亿)':>13s}{'年化%':>9s}{'IR':>7s}{'回撤%':>9s}")
    print("-" * 68)
    rows = []
    for lb in LABELS:
        ser = pd.Series(nav[lb], index=idx_use)
        m = metrics(ser)
        if m is None:
            continue
        print(f"  {lb:<14s}{n_by_lab[lb]//max(len(idx_use),1):>9d}{'-':>13s}"
              f"{m['年化%']:>+9.2f}{m['IR']:>+7.2f}{m['回撤%']:>9.2f}")
        rows.append({'layer': lb, 'avg_n': n_by_lab[lb] // max(len(idx_use), 1), **m})
    print("-" * 68)

    # 各档市值中位数（用最后一次建仓日）
    d_last = idx_use[-1]
    cap_last = cap.loc[d_last].dropna()
    print("\n各档流通市值中位数（亿元，建仓日口径）：")
    q = cap_last.quantile(Q)
    for i, lb in enumerate(LABELS):
        seg = cap_last[(cap_last > q.iloc[i]) & (cap_last <= q.iloc[i + 1])] if i else cap_last[cap_last <= q.iloc[i + 1]]
        print(f"  {lb:<14s} {seg.median():>10.1f} 亿   （{len(seg)} 只）")

    if len(rows) >= 2:
        small, big = rows[0]['年化%'], rows[-1]['年化%']
        print(f"\n  → 最小20% 市值年化 {small:+.2f}%  vs  最大20% 市值年化 {big:+.2f}%"
              f"   小减大 = {small-big:+.2f} pp")
        print("  → 小市值更优则「小盘跑赢」在市值维度上成立")

    pd.DataFrame(rows).to_csv(OUT / 'market_cap_layers.csv', index=False)
    print("\n保存: market_cap_layers.csv")


if __name__ == '__main__':
    main()