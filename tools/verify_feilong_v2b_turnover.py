"""飞龙 V2 续：分位组单调性 + 换手诊断——区分「因子无用」与「有 alpha 被成本吃光」。

V2 结论：同池等权基准下 A 年化超额 -11.44%（跑输池等权），测试期 IC=-0.0047(t=-0.83)。
但 A 的日均换手 0.734 × 成本 0.002 ≈ 0.147%/天 ≈ 年化 -37% 的成本拖累，
即毛超额可能仍为正。本脚本回答三个问题：

    Q1 分位组单调性：按 pred 分 10 组，看毛收益是否从 D1(最高) 到 D10(最低) 单调递减。
       单调 → 有真实截面 alpha；无序 → 因子确为噪声。
    Q2 换手到底吃掉多少：毛 vs 净 超额对比。
    Q3 降换手能否救回：持仓周期 N={1,2,3,5} × 缓冲带 buffer={0,20,40}
       （缓冲：跌出 topk+buffer 才卖，而不是跌出 topk 就卖）。

沿用 V2 的面板缓存与模型（M3_LGB_shallow，random_state=42 确定性）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from quantaalpha.strategy.feilong import feilong_factors

CACHE = Path('data/results/_feilong_panel_v2.parquet')
OUT = Path('data/results')
TOPK = 30
COST = 0.002
SPLIT = '2025-10-01'
FEATS = ['feilong_entry', 'feilong_exit', 'feilong_rr']


def cs_rank(df, cols=FEATS):
    return df[cols].groupby(level='datetime').rank(pct=True)


def metrics_ex(rets, bench):
    rets = rets.dropna()
    if len(rets) == 0:
        return {}
    ex = rets - bench.reindex(rets.index).fillna(0)
    cum = rets.cumsum()
    ec = ex.cumsum()
    return {
        '年化': rets.mean() * 252,
        '年化超额': ex.mean() * 252,
        '超额IR': ex.mean() / ex.std(ddof=1) * np.sqrt(252) if ex.std(ddof=1) > 0 else np.nan,
        '超额t': ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if ex.std(ddof=1) > 0 else np.nan,
        '最大回撤': (cum - cum.cummax()).min(),
    }


def decile_groups(panel, dates, ngroup=10):
    """每天按 pred 分 ngroup 组，组内等权，返回 (date × group) 毛收益矩阵。"""
    rows = {}
    for dt in dates:
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=['pred', 'fwd', 'bench_eq'])
        if len(sub) < ngroup * 5:
            continue
        try:
            sub = sub.assign(g=pd.qcut(sub['pred'].rank(method='first'), ngroup, labels=False))
        except ValueError:
            continue
        rows[dt] = sub.groupby('g')['fwd'].mean()
    return pd.DataFrame(rows).T.sort_index()


def run_buffered(panel, dates, topk=TOPK, buffer=0, every=1):
    """带缓冲带 + 调仓周期的 top-K 回测。"""
    hold, rows = set(), []
    for k, dt in enumerate(dates):
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=['pred', 'fwd', 'bench_eq'])
        if len(sub) < topk:
            continue
        if k % every != 0:
            # 非调仓日：继续持有（收益照记）
            cur = hold & set(sub.index)
            if len(cur) == 0:
                continue
            sel = sub.loc[list(cur)]
            rows.append({'date': dt, 'gross': sel['fwd'].mean(), 'net': sel['fwd'].mean(),
                         'turnover': 0.0, 'bench_eq': sub['bench_eq'].iloc[0]})
            continue
        rk = sub['pred'].rank(ascending=False, method='first')
        keep = [i for i in hold if i in rk.index and rk[i] <= topk + buffer]
        need = topk - len(keep)
        if need > 0:
            cand = rk.drop(index=[i for i in keep if i in rk.index]).nsmallest(need)
            cur = set(keep) | set(cand.index)
        else:
            cur = set(keep[:topk])
        sel = sub.loc[list(cur)]
        gross = sel['fwd'].mean()
        turn = 1.0 if not hold else 1.0 - len(cur & hold) / max(len(cur), 1)
        rows.append({'date': dt, 'gross': gross, 'net': gross - turn * COST,
                     'turnover': turn, 'bench_eq': sub['bench_eq'].iloc[0]})
        hold = cur
    return pd.DataFrame(rows).set_index('date')


def main():
    panel = pd.read_parquet(CACHE)
    bq = panel.groupby(level='datetime')['fwd'].mean().rename('bench_eq')
    panel = panel.join(bq, on='datetime')

    all_dates = sorted(set(panel.index.get_level_values('datetime')))
    tr_dates = [d for d in all_dates if d < pd.Timestamp(SPLIT)]
    te_dates = [d for d in all_dates if d >= pd.Timestamp(SPLIT)]

    tr_all = panel.loc[tr_dates].dropna(subset=FEATS + ['fwd'])
    te = panel.loc[te_dates].dropna(subset=FEATS + ['fwd'])
    model = lgb.LGBMRegressor(n_estimators=60, learning_rate=0.05, num_leaves=7, subsample=0.8,
                              colsample_bytree=0.8, verbose=-1, random_state=42)
    model.fit(cs_rank(tr_all), tr_all['fwd'])
    panel['pred'] = np.nan
    panel.loc[te.index, 'pred'] = model.predict(cs_rank(te))
    bench_te = bq.reindex(te_dates).dropna()

    # ---- Q1 分位组单调性（毛收益，未扣成本） ----
    G = decile_groups(panel, te_dates)
    print("=== Q1 pred 十分位组（D1=预测最高）测试期毛收益 ===")
    out = []
    for g in sorted(G.columns):
        s = G[g].dropna()
        m = metrics_ex(s, bench_te.reindex(s.index))
        out.append({'组': f'D{g+1}', '毛年化': m.get('年化'), '毛年化超额': m.get('年化超额'),
                    '超额t': m.get('超额t')})
        print(f"  D{g+1:<2d} 毛年化={m.get('年化', float('nan'))*100:+7.2f}%  "
              f"毛超额={m.get('年化超额', float('nan'))*100:+7.2f}%  t={m.get('超额t', float('nan')):+.2f}")
    d1 = metrics_ex(G[0].dropna(), bench_te.reindex(G[0].dropna().index))
    d10 = metrics_ex(G[9].dropna(), bench_te.reindex(G[9].dropna().index))
    print(f"  → 多空 D1-D10 毛年化价差 = {(d1['年化'] - d10['年化'])*100:+.2f}%  "
          f"(t 见上两组各自显著性；同向且 D1>D10 说明有截面区分度)")
    pd.DataFrame(out).to_csv(OUT / 'feilong_v2_decile.csv', index=False)

    # ---- Q2 换手吃掉了多少 ----
    A = run_buffered(panel, te_dates, buffer=0, every=1)
    mA_g = metrics_ex(A['gross'], bench_te.reindex(A.index))
    mA_n = metrics_ex(A['net'], bench_te.reindex(A.index))
    print("\n=== Q2 换手成本拆解（top30 每日全换） ===")
    print(f"  毛年化超额 = {mA_g['年化超额']*100:+.2f}%   净年化超额 = {mA_n['年化超额']*100:+.2f}%   "
          f"成本拖累 = {(mA_n['年化超额']-mA_g['年化超额'])*100:+.2f}%")
    print(f"  日均换手 = {A['turnover'].mean():.3f}  （成本 {COST*100:.2f}%/次全换 → 年化 "
          f"{-A['turnover'].mean()*COST*252*100:.1f}%）")

    # ---- Q3 降换手方案 ----
    print("\n=== Q3 降换手方案（年化超额 vs 池等权；成本已扣） ===")
    rows = []
    for every in (1, 2, 3, 5):
        for buf in (0, 20, 40):
            R = run_buffered(panel, te_dates, buffer=buf, every=every)
            mg = metrics_ex(R['gross'], bench_te.reindex(R.index))
            mn = metrics_ex(R['net'], bench_te.reindex(R.index))
            rows.append({'调仓周期': every, '缓冲': buf, '日均换手': R['turnover'].mean(),
                         '毛年化超额': mg.get('年化超额'), '净年化超额': mn.get('年化超额'),
                         '超额IR': mn.get('超额IR'), '超额t': mn.get('超额t'),
                         '最大回撤': mn.get('最大回撤')})
            print(f"  every={every} buffer={buf:<3d} 换手={R['turnover'].mean():.3f}  "
                  f"毛超额={mg.get('年化超额', float('nan'))*100:+7.2f}%  "
                  f"净超额={mn.get('年化超额', float('nan'))*100:+7.2f}%  "
                  f"IR={mn.get('超额IR', float('nan')):+.3f}  t={mn.get('超额t', float('nan')):+.2f}")
    pd.DataFrame(rows).to_csv(OUT / 'feilong_v2_turnover_grid.csv', index=False)
    print("\n保存: feilong_v2_decile.csv / feilong_v2_turnover_grid.csv")


if __name__ == '__main__':
    main()
