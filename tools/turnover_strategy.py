"""换手率因子的策略级验证 —— 检验「避开最高换手」能否成为可执行规则。

上一轮发现（新数据维度，qlib 数据集里没有 turnover_rate）：
    f1_换手水平  RankIC train −0.0691 / valid −0.0814 / test −0.0735（三段同号，|t| 8~11）
    f2_突然放量  −0.0145 / −0.0250 / −0.0364
    f3_换手加速  −0.0141 / −0.0206 / −0.0203
    **但十分位组非单调**：D1~D8 都在 +18~+29bp，只有 D10（换手最高那 10%）是 −54.4bp
    → 关系不是单调的，而是「中等换手最好、极高换手最差」
    → 真正可用的规则不是「买低换手」，而是「**避开最高换手的那一组**」

本脚本检验三条规则（都含真实成本，月频调仓）：
    R1  等权全池（基准）
    R2  等权持有**最低换手**的 K 只      （单调因子的直觉做法）
    R3  等权持有**剔除最高换手 10%** 后的池 （分位组形状指向的做法）

⚠️ 样本限制（必须在结论里说明）：
    turnover_rate 只有 **2020-2024** 有数据。2025-2026 为空，
    因此**无法做 2025 之后的样本外验证**。本实验的 test 段是 2023-2024。

输出：data/results/turnover_strategy.csv
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
TRAIN_END, VALID_END = '2021-12-31', '2022-12-31'
KS = [20, 30, 50]


def main():
    P = pd.read_parquet(PANEL)
    if not isinstance(P.columns, pd.MultiIndex):
        P.columns = pd.MultiIndex.from_tuples(
            [tuple(c) if isinstance(c, tuple) else (c, c) for c in P.columns])
    P.index = pd.DatetimeIndex(P.index.get_level_values(0))
    P = P.loc[START:END]
    C, O, AMT = P['close'], P['open'], P['amount']
    TR = P['turnover_rate'].where(P['turnover_rate'] > 0)
    ret = (C.shift(-1) / C - 1.0).clip(-0.5, 0.5)
    dates = P.index
    mstart = [d for d in dates.to_series().groupby([dates.year, dates.month]).min()]

    def build(d0, K=None, drop_top_frac=None):
        """返回目标权重。K=None 表示全池等权；drop_top_frac 表示剔除换手最高的那一档。"""
        tr = TR.loc[d0].dropna()
        amt = AMT.loc[d0]
        ok = tr.index[(amt.reindex(tr.index) > 0) & (C.loc[d0].reindex(tr.index) > 0)]
        tr = tr.reindex(ok).dropna()
        if len(tr) < 100:
            return None
        if drop_top_frac:
            cut = tr.quantile(1 - drop_top_frac)
            tr = tr[tr <= cut]
        if K and len(tr) > K:
            tr = tr.nsmallest(K)            # 最低换手 K 只
        if len(tr) < 30:
            return None
        return pd.Series(1.0 / len(tr), index=tr.index)

    strategies = {'R1_全池等权': lambda d: build(d)}
    for K in KS:
        strategies[f'R2_最低换手{K}'] = (lambda k: (lambda d: build(d, K=k)))(K)
    for f in (0.1, 0.2, 0.3):
        strategies[f'R3_剔除最高换手{int(f*100)}%'] = (
            lambda ff: (lambda d: build(d, drop_top_frac=ff)))(f)

    nav = {k: [] for k in strategies}
    nav_cost = {k: [] for k in strategies}
    used = []
    for d0 in mstart:
        used.append(d0)          # 与 nav 同步 append，保证长度一致
        for name, fn in strategies.items():
            w = fn(d0)
            if w is None:
                nav[name].append(0.0); nav_cost[name].append(0.0)
                continue
            part = (ACCOUNT * w / AMT.loc[d0].reindex(w.index)).clip(0, 1).fillna(0)
            imp = float((w * ETA * SIGMA_BP * np.sqrt(part)).sum())
            cost = (COMMISSION_BP + STAMP_BP + 2 * imp) / 1e4     # 首期建仓
            nav[name].append(float((w * ret.loc[d0].reindex(w.index).fillna(0.0)).sum()) - cost)
            nav_cost[name].append(cost)
    idx = pd.DatetimeIndex(used)
    assert all(len(v) == len(idx) for v in nav.values()), \
        f'长度不一致: nav={ {k: len(v) for k, v in nav.items()} } vs idx={len(idx)}'

    def ann(s):
        s = s[s != 0]
        return np.nan if len(s) < 3 else ((1 + s).prod() ** (12 / len(s)) - 1) * 100

    base = pd.Series(nav['R1_全池等权'], index=idx)
    print("=== 换手率策略（月频调仓，持有 1 月，首期扣建仓成本）===")
    print(f"  {'策略':<22s}{'全期':>9s}{'train':>9s}{'valid':>9s}{'test':>9s}"
          f"{'test超额%':>11s}{'test t':>9s}")
    print("  " + "-" * 82)
    rows = []
    for name in strategies:
        ser = pd.Series(nav[name], index=idx)
        a = ann(ser)
        a_tr = ann(ser[ser.index <= pd.Timestamp(TRAIN_END)])
        a_va = ann(ser[(ser.index > pd.Timestamp(TRAIN_END)) & (ser.index <= pd.Timestamp(VALID_END))])
        a_te = ann(ser[ser.index > pd.Timestamp(VALID_END)])
        e = (ser - base)[ser.index > pd.Timestamp(VALID_END)]
        t = e.mean() / e.std() * np.sqrt(len(e)) if e.std() > 0 else np.nan
        print(f"  {name:<22s}{a:>+9.2f}{a_tr:>+9.2f}{a_va:>+9.2f}{a_te:>+9.2f}{a_te-ann(base[base.index>pd.Timestamp(VALID_END)]):>+11.2f}{t:>+9.2f}")
        rows.append({'strategy': name, '全期': a, 'train': a_tr, 'valid': a_va, 'test': a_te,
                     'test_超额': a_te - ann(base[base.index > pd.Timestamp(VALID_END)]),
                     'test_t': t,
                     '建仓成本bp': float(np.mean(nav_cost[name])) * 1e4})

    # 逐年
    print("\n=== 逐年年化（%） ===")
    print(f"  {'策略':<22s}" + "".join(f"{y:>8d}" for y in range(2020, 2025)))
    for name in strategies:
        ser = pd.Series(nav[name], index=idx)
        line = f"  {name:<22s}"
        for y in range(2020, 2025):
            line += f"{ann(ser[ser.index.year == y]):>+8.2f}"
        print(line)

    # 最优规则的月度分布
    print("\n=== 「剔除最高换手10%」相对全池的月度超额分布（bp） ===")
    s = pd.Series(nav['R3_剔除最高换手10%'], index=idx)
    e = (s - base) * 1e4
    e = e[e != 0]
    if len(e):
        print(f"  均值 {e.mean():+.2f}  中位 {e.median():+.2f}  为正占比 {(e>0).mean():.1%}"
              f"  年化均值 {e.mean()*12/100:+.2f}%")
        print(f"  → 若「为正占比」显著高于 50% 且年化均值为正，说明剔除高换手有稳定增益")

    pd.DataFrame(rows).to_csv(OUT / 'turnover_strategy.csv', index=False)
    print("\n保存: turnover_strategy.csv")
    print("\n⚠️ 样本限制：turnover_rate 只有 2020-2024，**2025-2026 无法验证**。")


if __name__ == '__main__':
    main()