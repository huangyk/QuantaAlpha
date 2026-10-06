"""调仓频率实验：多久再平衡一次最划算？（不需要任何选股 alpha）

为什么这个实验重要：
    前面已证明「一次换手要花 10bp 固定 + 9.5~30bp 冲击」，
    而日内时点优化最多只能省 ~7bp。
    所以**少换一次手，比挑个好时点更值钱**。本实验把这个判断量化。

载体：等权持有一个固定名单（按流动性选的 Top N），比较不同再平衡周期：
    T = 1(每日) / 2 / 3 / 5 / 10 / 20 / 60 / 从不
    - 频率高：权重不漂移，但成本大
    - 频率低：省成本，但权重向赢家漂移（这会带来一个额外的动量/反转效应）
    两者的权衡点就是答案。

严格点：
    - 名单用**滚动 60 日中位成交额**选 Top N（无前视）
    - 成本用真实 participation 冲击（Almgren: 0.1 × σ日 × sqrt(participation)）
    - 每日记账，输出毛收益/换手/成本/净收益/回撤/IR

数据：cn_real_full，2020-01 ~ 2026-09（1635 天）
输出：data/results/rebalance_freq.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = 'D:/qqmt/qlib_data/cn_real_full'
START, END = '2020-01-01', '2026-09-30'
OUT = Path('data/results')
ACCOUNT = 1e7
COMMISSION_BP = 5.0
STAMP_BP = 5.0
ETA = 0.1
SIGMA_BP = 342.0          # 实测池内个股日波动中位（turnover_cost_curve.py 算出）
PERIODS = [1, 2, 3, 5, 10, 20, 60, 10 ** 9]     # 最后一项 = 从不


def load():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    codes = D.list_instruments(D.instruments('liquid500'), start_time=START, end_time=END, as_list=True)
    df = D.features(codes, ['$close', '$volume'], start_time=START, end_time=END, freq='day').sort_index()
    P = df['$close'].unstack('instrument').sort_index()
    V = df['$volume'].unstack('instrument').sort_index().reindex_like(P)
    return P, V


def rolling_amt_median(P, V, win=60, minp=20):
    """每票每日的「过去 win 日中位成交额」，用于无前视地选流动性最好的 N 只。"""
    amt = P * V
    return amt.rolling(win, min_periods=minp).median()


def run(P, V, n_hold, every, account=ACCOUNT):
    amt = (P * V)
    amt_med = rolling_amt_median(P, V)
    rets = P.pct_change()
    dates = P.index
    # 每月初确定名单（用当时可见的信息）
    month_starts = set(pd.Series(dates).groupby([dates.year, dates.month]).min())
    uni = None
    w = None
    gross, net, turn, cost = [], [], [], []
    for k, d in enumerate(dates):
        # 月首（且有足够历史）时刷新名单
        if d in month_starts and k >= 61:
            row = amt_med.loc[d].dropna() if d in amt_med.index else pd.Series(dtype=float)
            if len(row) >= n_hold:
                uni = list(row.nlargest(n_hold).index)
        if uni is None or len(uni) == 0:
            continue
        r = rets.loc[d]
        if w is None:
            sel = [x for x in uni if x in P.columns]
            if len(sel) == 0:
                continue
            w = pd.Series(1.0 / len(sel), index=sel)
            gross.append(0.0); net.append(0.0); turn.append(0.0); cost.append(0.0)
            continue
        rr = r.reindex(w.index).fillna(0.0)
        pgross = float((w * rr).sum())
        # 权重漂移
        w_drift = w * (1 + rr)
        w_drift /= w_drift.sum()
        pcost, pturn = 0.0, 0.0
        if k % every == 0:
            sel = [x for x in uni if x in P.columns]
            target = pd.Series(1.0 / len(sel), index=sel)
            allidx = target.index.union(w_drift.index)
            a = target.reindex(allidx).fillna(0.0)
            b = w_drift.reindex(allidx).fillna(0.0)
            pturn = 0.5 * float((a - b).abs().sum())
            # 冲击：按每票买入额 / 当日成交额
            buy_amt = (a - b).clip(lower=0) * account
            day_amt = amt.loc[d].reindex(allidx)
            part = (buy_amt / day_amt.replace(0, np.nan)).clip(0, 1)
            imp = ETA * SIGMA_BP * np.sqrt(part.fillna(0))        # bp，逐票
            avg_imp = float(((a - b).clip(lower=0).abs() * imp.fillna(0)).sum() /
                            max(float((a - b).clip(lower=0).abs().sum()), 1e-9))
            pcost = pturn * (COMMISSION_BP + STAMP_BP + 2 * avg_imp) / 1e4
            w = target
        else:
            w = w_drift
        gross.append(pgross)
        cost.append(pcost)
        net.append(pgross - pcost)
        turn.append(pturn)
    idx = dates[:len(gross)]
    return pd.DataFrame({'gross': gross, 'net': net, 'cost': cost, 'turn': turn}, index=idx)


def metrics(R, bench=None):
    g, n = R['gross'], R['net']
    def ann(s):
        return (1 + s.mean()) ** 252 - 1
    cum = (1 + n).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    out = {'毛年化%': ann(g) * 100, '净年化%': ann(n) * 100,
           '成本年化%': (ann(g) - ann(n)) * 100,
           '日均换手': R['turn'].mean(),
           '最大回撤%': mdd * 100,
           '净IR': n.mean() / n.std() * np.sqrt(252)}
    if bench is not None:
        e = (n - bench.reindex(n.index).fillna(0))
        out['超额年化%'] = ann(e) * 100
        out['超额t'] = e.mean() / e.std() * np.sqrt(len(e))
    return out


def main():
    print("载入日线 ...")
    P, V = load()
    print(f"  {P.shape[0]} 天 × {P.shape[1]} 只   {P.index[0].date()} ~ {P.index[-1].date()}")
    bench = P.pct_change().mean(axis=1)          # 全池等权（每日再平衡）

    for n_hold in (30, 50):
        print(f"\n{'='*78}")
        print(f"持仓 {n_hold} 只（按流动性滚动选 Top{n_hold}），账户 {ACCOUNT/1e4:,.0f} 万")
        print(f"  {'调仓周期':<10s}{'毛年化%':>10s}{'净年化%':>10s}{'成本年化%':>11s}"
              f"{'日均换手':>10s}{'最大回撤%':>11s}{'净IR':>8s}{'超额t':>8s}")
        rows = []
        for ev in PERIODS:
            R = run(P, V, n_hold, ev)
            m = metrics(R, bench)
            label = '从不' if ev > 1e6 else f'{ev}天'
            rows.append({'持仓数': n_hold, '调仓周期': label, **m})
            print(f"  {label:<10s}{m['毛年化%']:>+10.2f}{m['净年化%']:>+10.2f}"
                  f"{m['成本年化%']:>11.2f}{m['日均换手']:>10.3f}{m['最大回撤%']:>11.2f}"
                  f"{m['净IR']:>+8.2f}{m.get('超额t', float('nan')):>+8.2f}")
        rows_df = pd.DataFrame(rows)
        f = OUT / f'rebalance_freq_{n_hold}.csv'
        rows_df.to_csv(f, index=False)
        print(f"  保存 {f.name}")
        best = rows_df.loc[rows_df['净年化%'].idxmax()]
        print(f"  → 最优调仓周期: {best['调仓周期']}  净年化 {best['净年化%']:+.2f}%  "
              f"成本 {best['成本年化%']:.2f}%/年  换手 {best['日均换手']:.3f}")

    print(f"\n参考：全池等权每日再平衡（无成本）年化 "
          f"{((1+bench.mean())**252-1)*100:+.2f}%")


if __name__ == '__main__':
    main()
