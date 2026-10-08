"""用 akshare 补来的 2025-2026 换手率，验证项目最强信号是否已死。

背景：
    换手率因子（f1 换手水平）在 DB 的 turnover_rate 上有 **IC −0.069~−0.081，|t| 8~11**，
    是项目里最强的信号，三段（train/valid/test = 2020-2021/2022/2023-2024）全部同号。
    但 DB 的 turnover_rate **只有 2020-2024**，而 2025-2026 正是战法失效的关键区间。

本脚本用 akshare 补来的 2025-2026 换手率做**完全独立的样本外检验**：
    train 2020-2021 / valid 2022 / test 2023-2024（旧数据）
    → OOS 2025 / OOS 2026（**新拉的数据，脚本运行前从未见过**）

这批 2025-2026 数据是脚本运行前刚从网上拉的，**不存在任何前视污染**，
是本项目第一次真正意义上的"未来数据"验证。

判定：若 2025/2026 的 IC 仍为负且 |t|≥2 → 因子有效，可继续深挖；
      若 IC 消失或反转 → 因子已死，2020-2024 的强 IC 是样本期特性。

输出：data/results/turnover_oos_2025_2026.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path('data/cache/panel_2011.parquet')
AK = Path('data/cache/akshare_turnover_top2000.csv.gz')
OUT = Path('data/results')
H = 5
SEGS = [('train_2020-21', '2020-01-01', '2021-12-31'),
        ('valid_2022', '2022-01-01', '2022-12-31'),
        ('test_2023-24', '2023-01-01', '2024-12-31'),
        ('OOS_2025', '2025-01-01', '2025-12-31'),
        ('OOS_2026', '2026-01-01', '2026-09-30')]


def main():
    if not AK.exists():
        raise SystemExit(f'缺少 {AK}，先跑 fetch_turnover_akshare.py --top 2000')
    print("载入 akshare 换手率 ...")
    ak = pd.read_csv(AK, parse_dates=['datetime'])
    TR_ak = ak.pivot_table(index='datetime', columns='instrument', values='turnover_rate')
    AMP_ak = ak.pivot_table(index='datetime', columns='instrument', values='amplitude')
    print(f"  akshare: {TR_ak.shape[0]} 天 × {TR_ak.shape[1]} 只  "
          f"{TR_ak.index.min().date()} ~ {TR_ak.index.max().date()}")

    P = pd.read_parquet(PANEL)
    if not isinstance(P.columns, pd.MultiIndex):
        P.columns = pd.MultiIndex.from_tuples(
            [tuple(c) if isinstance(c, tuple) else (c, c) for c in P.columns])
    P.index = pd.DatetimeIndex(P.index.get_level_values(0))
    C, O = P['close'], P['open']
    ret = (O.shift(-(H + 1)) / O.shift(-1) - 1.0).clip(-0.5, 0.5)
    # ⚠️ 两个来源必须先合并再算 IC：
    #   DB 的 turnover_rate 覆盖 2020-2024，akshare 覆盖 2025-2026
    TR = P['turnover_rate'].combine_first(TR_ak.reindex(columns=P['turnover_rate'].columns))
    AMP = AMP_ak.reindex(index=P.index, columns=TR.columns)
    print(f"  合并后换手率面板: {TR.shape[0]} 天 × {TR.shape[1]} 只")
    for nm, a, b in [('2020-2024(DB)', '2020-01-01', '2024-12-31'),
                     ('2025-2026(ak)', '2025-01-01', '2026-09-30')]:
        sub = TR.loc[pd.Timestamp(a):pd.Timestamp(b)]
        print(f"    {nm}: 覆盖 {int(sub.notna().sum().sum()):>10,} 个格子")

    # 因子
    f = {'f1_换手水平': TR,
         'f2_突然放量': TR / (TR.rolling(20, min_periods=5).median() + 1e-9),
         'f3_换手加速': TR / (TR.shift(5) + 1e-9),
         'f4_振幅': AMP}

    print("\n" + "=" * 96)
    print("五段 RankIC（2025-2026 为 akshare 新拉的真正样本外数据）")
    print("=" * 96)
    print(f"  {'因子':<14s}" + "".join(f"{s[0]:>16s}" for s in SEGS))
    print("  " + "-" * 94)
    rows = []
    for name, fac in f.items():
        line = f"  {name:<14s}"
        row = {'factor': name}
        for seg, a, b in SEGS:
            m_ = (ret.index >= pd.Timestamp(a)) & (ret.index <= pd.Timestamp(b))
            s = pd.concat([fac[m_].stack().rename('f'), ret[m_].stack().rename('r')],
                          axis=1).dropna()
            ics = []
            for dt, g in s.groupby(level=0):
                if len(g) < 50:
                    continue
                x, y = g['f'].values, g['r'].values
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    continue
                ics.append(pd.Series(x).corr(pd.Series(y), method='spearman'))
            ser = pd.Series(ics)
            if len(ser) > 20 and ser.std() > 0:
                ic = ser.mean()
                t = ic / ser.std() * np.sqrt(len(ser))
            else:
                ic = t = np.nan
            line += f"{ic:>+9.4f}({t:>+5.2f})" if not np.isnan(ic) else f"{'—':>16s}"
            row[f'{seg}_IC'] = ic
            row[f'{seg}_t'] = t
            row[f'{seg}_n'] = len(ser)
        print(line)
        rows.append(row)

    print("\n" + "=" * 96)
    print("判定")
    print("=" * 96)
    for row in rows:
        oos25, oos26 = row.get('OOS_2025_IC'), row.get('OOS_2026_IC')
        t25, t26 = row.get('OOS_2025_t'), row.get('OOS_2026_t')
        if np.isnan(oos25):
            continue
        keep = (not np.isnan(oos26)) and oos25 * oos26 > 0 and \
               abs(t25) >= 2 and abs(t26) >= 2
        print(f"  {row['factor']:<14s} 2025 IC {oos25:+.4f}(t={t25:+.2f})  "
              f"2026 IC {oos26:+.4f}(t={t26:+.2f})  "
              f"→ {'**通过：样本外仍有效**' if keep else '样本外未通过'}")

    # OOS 分位组
    print("\n=== 2025-2026 十分位组（D1 = 换手最低组） ===")
    for name, fac in f.items():
        grp = {}
        for dt in ret.index[(ret.index >= '2025-01-01')]:
            if dt not in fac.index:
                continue
            s = fac.loc[dt].dropna()
            y = ret.loc[dt].reindex(s.index)
            z = pd.concat([s.rename('s'), y.rename('y')], axis=1).dropna()
            if len(z) < 50:
                continue
            lab = pd.qcut(z['s'].rank(method='first'), 10, labels=False)
            grp[dt] = z.groupby(lab)['y'].mean()
        if not grp:
            continue
        G = pd.DataFrame(grp).T.sort_index()
        dec = [G[c].dropna().mean() * 1e4 for c in sorted(G.columns)]
        mono = all(dec[i] > dec[i + 1] for i in range(len(dec) - 1))
        print(f"  {name:<14s}" + "".join(f"{v:>+8.1f}" for v in dec))
        print(f"  {'':<14s}D1−D10={(dec[0]-dec[-1]):+8.1f}bp/5日  "
              f"{'单调 ✓' if mono else '不单调'}"
              + (f"  且 D10={dec[-1]:+.1f}bp（换手最高组）" if name == 'f1_换手水平' else ""))

    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'turnover_oos_2025_2026.csv', index=False)
    print("\n保存: turnover_oos_2025_2026.csv")


if __name__ == '__main__':
    main()