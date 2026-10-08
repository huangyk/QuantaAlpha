"""新数据维度测试：`turnover_rate`（换手率）是否有预测力。

背景：DB 的 `bars_1d` 提供了 qlib 数据集里**没有**的三个字段 ——
`amount`（已用于成本修正）、`turnover_rate`、`float_shares`。
前两个实验已用完 amount / float_shares，这里检验第三个。

经济含义（A 股的经典异象）：
    流动性溢价：低换手股长期跑赢高换手股（Amihud-Mendelson）
    反转效应：高换手 = 短期过热 → 未来收益差

样本限制（必须知道）：
    `turnover_rate` 在 bars_1d 里**只有 2020-2024 有数据**（约 38% 的行），
    2011-2019 与 2025-2026 为空 → 本实验最多 5 年，恰好覆盖「抱团白马」行情。

三个因子（全部只用 T 日及之前的信息）：
    f1_换手水平    turnover_rate（低 → 高）
    f2_换手变化    turnover_rate / 过去20日中位（突然放量）
    f3_换手反转    过去5日换手变化率

判定铁律：
    1) RankIC 的 |t| ≥ 2
    2) 十分位组单调
    3) 分 train(2020-2021) / valid(2022) / test(2023-2024) 三段同号

输出：data/results/turnover_factor_test.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path('data/cache/panel_2011.parquet')
OUT = Path('data/results')
START, END = '2020-01-01', '2024-12-31'
H = 5
SEGS = [('train', '2020-01-01', '2021-12-31'),
        ('valid', '2022-01-01', '2022-12-31'),
        ('test', '2023-01-01', '2024-12-31')]


def main():
    print("载入面板 ...")
    P = pd.read_parquet(PANEL)
    if not isinstance(P.columns, pd.MultiIndex):
        P.columns = pd.MultiIndex.from_tuples(
            [tuple(c) if isinstance(c, tuple) else (c, c) for c in P.columns])
    P.index = pd.DatetimeIndex(P.index.get_level_values(0))
    P = P.loc[START:END]
    C, O = P['close'], P['open']
    TR = P['turnover_rate'].where(P['turnover_rate'] > 0)

    print(f"  区间 {P.index[0].date()} ~ {P.index[-1].date()}")
    print(f"  turnover_rate 覆盖：每日中位 {int(TR.notna().sum(axis=1).median())} 只，"
          f"占比 {TR.notna().sum().sum()/TR.size:.1%}")

    # 因子
    f = {}
    f['f1_换手水平'] = TR
    f['f2_突然放量'] = TR / (TR.rolling(20, min_periods=5).median() + 1e-9)
    f['f3_换手加速'] = TR / (TR.shift(5) + 1e-9)

    # 收益：T 日收盘出分 → T+1 开盘买 → T+6 开盘卖（正确对齐）
    ret = (O.shift(-(H + 1)) / O.shift(-1) - 1.0).clip(-0.5, 0.5)

    print("\n=== RankIC 三段检验（正确对齐口径）===")
    print(f"  {'因子':<16s}{'train':>18s}{'valid':>18s}{'test':>18s}{'三段同号':>10s}")
    print("  " + "-" * 80)
    rows = []
    for name, fac in f.items():
        ic_res = {}
        for seg, a, b in SEGS:
            m = (ret.index >= pd.Timestamp(a)) & (ret.index <= pd.Timestamp(b))
            s = pd.concat([fac[m].stack().rename('f'), ret[m].stack().rename('r')],
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
            ic_res[seg] = (ser.mean(), ser.mean() / ser.std() * np.sqrt(len(ser))
                           if len(ser) > 20 and ser.std() > 0 else np.nan, len(ser))
        signs = {np.sign(ic_res[s][0]) for s in ic_res if not np.isnan(ic_res[s][0])}
        same = '是' if len(signs) == 1 else '否'
        line = f"  {name:<16s}"
        for seg in ('train', 'valid', 'test'):
            mm, tt, nn = ic_res[seg]
            line += f"{mm:>+10.4f}({tt:>+5.2f})" if not np.isnan(mm) else f"{'—':>18s}"
        print(line + f"{same:>10s}")
        rows.append({'factor': name, **{f'{s}_IC': ic_res[s][0] for s in ic_res},
                     **{f'{s}_t': ic_res[s][1] for s in ic_res},
                     **{f'{s}_n': ic_res[s][2] for s in ic_res},
                     '三段同号': same})

    # 十分位组（test 段）
    print("\n=== test 段十分位组（D1 = 因子值最低组；判断单调性） ===")
    a, b = '2023-01-01', '2024-12-31'
    m = (ret.index >= pd.Timestamp(a)) & (ret.index <= pd.Timestamp(b))
    for name, fac in f.items():
        grp = {}
        for dt in ret[m].index:
            s = fac.loc[dt].dropna()
            y = ret.loc[dt].reindex(s.index)
            z = pd.concat([s.rename('s'), y.rename('y')], axis=1).dropna()
            if len(z) < 80:
                continue
            lab = pd.qcut(z['s'].rank(method='first'), 10, labels=False)
            grp[dt] = z.groupby(lab)['y'].mean()
        if not grp:
            continue
        G = pd.DataFrame(grp).T.sort_index()
        dec = [G[c].dropna().mean() for c in sorted(G.columns)]
        dec_bp = [v * 1e4 for v in dec]
        mono = all(dec[i] > dec[i + 1] for i in range(len(dec) - 1))
        print(f"  {name:<16s}" + "".join(f"{v:>+8.1f}" for v in dec_bp[:5])
              + "  ...  " + "".join(f"{v:>+8.1f}" for v in dec_bp[5:]))
        print(f"  {'':<16s}D1−D10 = {(dec[0]-dec[-1])*1e4:+8.2f} bp/5日  "
              f"({'单调' if mono else '不单调 → 按铁律无效'})")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'turnover_factor_test.csv', index=False)

    print("\n=== 判定 ===")
    ok = R[(R['train_t'].abs() >= 2) & (R['三段同号'] == '是')]
    print(f"  train 段 |t|>=2 且三段同号的因子: {list(ok['factor']) if len(ok) else '无'}")
    print("\n保存: turnover_factor_test.csv")


if __name__ == '__main__':
    main()