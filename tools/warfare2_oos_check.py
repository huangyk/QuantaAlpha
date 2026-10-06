"""战法2 参数在**训练期**选定 → 测试期只跑一次（干净的样本外，不从测试期网格挑最优点）。

同时输出：
    1. 数据覆盖诊断（每日有效样本数，确认结论可信度）
    2. f4 分年 IC（地基因子稳定性）
    3. 训练期网格选 (K, 调仓周期) → 测试期单次验证 + 分年

上一轮网格是在测试期上挑的（K=30/every=5 → t=1.65），存在选择偏差，本脚本消除它。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

FEATS = ['f3_bull', 'f4_pullback', 'f2_vol']
COST = 0.002
SPLIT = '2025-01-01'
OUT = Path('data/results')
PICKLE = OUT / '_warfare2_score_panel.pkl'


def cs_rank(df, cols=FEATS):
    return df[cols].groupby(level='datetime').rank(pct=True)


def ic_series(panel, dates, pred_col='pred'):
    out = {}
    for dt in dates:
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=[pred_col, 'fwd'])
        if len(sub) < 50:
            continue
        out[dt] = sub[pred_col].corr(sub['fwd'], method='spearman')
    return pd.Series(out).dropna()


def ic_stats(s):
    if len(s) < 5:
        return {}
    return {'IC均值': s.mean(), 'ICIR': s.mean() / s.std(ddof=1),
            't': s.mean() / (s.std(ddof=1) / np.sqrt(len(s))), '天数': len(s)}


def run_topk(panel, dates, topk=30, every=1, pred_col='pred', mask_bull=True):
    hold, rows = None, []
    for k, dt in enumerate(dates):
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=[pred_col, 'fwd'])
        if mask_bull:
            sub = sub[sub['is_bull'].fillna(False)]
        if len(sub) < topk:
            continue
        if k % every != 0:
            if hold is None:
                continue
            cur = hold & set(sub.index)
            if not cur:
                continue
            rows.append({'date': dt, 'net': sub.loc[list(cur), 'fwd'].mean(), 'turnover': 0.0})
            continue
        sel = sub.nlargest(topk, pred_col)
        cur = set(sel.index)
        turn = 1.0 if hold is None else 1.0 - len(cur & hold) / max(len(cur), 1)
        rows.append({'date': dt, 'net': sel['fwd'].mean() - turn * COST, 'turnover': turn})
        hold = cur
    return pd.DataFrame(rows).set_index('date')


def metrics(rets, bench):
    rets = rets.dropna()
    if len(rets) == 0:
        return {}
    ex = rets - bench.reindex(rets.index).fillna(0)
    return {'年化': rets.mean() * 252, '年化超额': ex.mean() * 252,
            '超额IR': ex.mean() / ex.std(ddof=1) * np.sqrt(252) if ex.std(ddof=1) > 0 else np.nan,
            '超额t': ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if ex.std(ddof=1) > 0 else np.nan,
            '最大回撤': (rets.cumsum() - rets.cumsum().cummax()).min(),
            '日胜率': float((rets > 0).mean()), '天数': len(rets)}


def grid(panel, dates, bench, pred_col='pred'):
    rows = []
    for topk in (10, 20, 30, 50):
        for every in (1, 2, 3, 5):
            R = run_topk(panel, dates, topk=topk, every=every, pred_col=pred_col)
            m = metrics(R['net'], bench)
            rows.append({'topk': topk, '调仓周期': every, '日均换手': R['turnover'].mean(),
                         '年化超额': m.get('年化超额'), '超额IR': m.get('超额IR'),
                         '超额t': m.get('超额t'), '最大回撤': m.get('最大回撤')})
    return pd.DataFrame(rows)


def main():
    panel = pd.read_pickle(PICKLE)
    bq = panel.groupby(level='datetime')['fwd'].mean()
    panel = panel.join(bq.rename('bench_eq'), on='datetime')
    dates = sorted(set(panel.index.get_level_values('datetime')))
    tr = [d for d in dates if d < pd.Timestamp(SPLIT)]
    te = [d for d in dates if d >= pd.Timestamp(SPLIT)]

    # ---------- 1. 数据覆盖诊断 ----------
    print("=== 1. 数据覆盖：每日有效个股数（按年均值） ===")
    cov = pd.DataFrame({
        '有fwd': panel['fwd'].notna().groupby(level='datetime').sum(),
        'f2(5日)': panel['f2_vol'].notna().groupby(level='datetime').sum(),
        'f4(20日)': panel['f4_pullback'].notna().groupby(level='datetime').sum(),
        'f3(60日)': panel['f3_bull'].notna().groupby(level='datetime').sum(),
        'is_bull': panel['is_bull'].fillna(False).groupby(level='datetime').sum(),
    })
    print(cov.groupby(cov.index.year).mean().round(0).to_string())

    # ---------- 2. f4 分年 IC ----------
    print("\n=== 2. f4_pullback（贴近MA20）分年 rank IC ===")
    for y in sorted(set(pd.Timestamp(d).year for d in dates)):
        dy = [d for d in dates if pd.Timestamp(d).year == y]
        st = ic_stats(ic_series(panel.assign(pred=panel['f4_pullback']), dy))
        if st:
            print(f"  {y}  IC={st['IC均值']:+.4f}  ICIR={st['ICIR']:+.3f}  t={st['t']:+.2f}  n={st['天数']}")

    # ---------- 3. 训练期选参 → 测试期单次 ----------
    tr_all = panel.loc[tr].dropna(subset=FEATS + ['fwd'])
    te_df = panel.loc[te].dropna(subset=FEATS + ['fwd'])
    lg = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=15, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1, random_state=42)
    lg.fit(cs_rank(tr_all), tr_all['fwd'])
    panel['pred'] = np.nan
    panel.loc[te_df.index, 'pred'] = lg.predict(cs_rank(te_df))
    # 训练期打分（用训练期自身拟合的模型，避免用到测试期信息）
    lg_tr = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=15, subsample=0.8,
                              colsample_bytree=0.8, verbose=-1, random_state=42)
    vcut = tr[int(len(tr) * 0.8)]
    fit_d = [d for d in tr if d < vcut]
    lg_tr.fit(cs_rank(panel.loc[fit_d].dropna(subset=FEATS + ['fwd'])),
              panel.loc[fit_d].dropna(subset=FEATS + ['fwd'])['fwd'])
    panel['pred_tr'] = np.nan
    tri = panel.loc[tr].dropna(subset=FEATS + ['fwd']).index
    panel.loc[tri, 'pred_tr'] = lg_tr.predict(cs_rank(panel.loc[tri]))

    bench_tr, bench_te = bq.reindex(tr), bq.reindex(te)
    G = grid(panel, tr, bench_tr, pred_col='pred_tr')
    print("\n=== 3a. 训练期网格（2020-2024，参数在此选定） ===")
    print(G.round(4).to_string(index=False))
    best = G.sort_values('超额IR', ascending=False).iloc[0]
    print(f"  → 按超额IR 选定: K={int(best['topk'])}, 调仓周期={int(best['调仓周期'])}"
          f"（训练期 年化超额={best['年化超额']*100:+.2f}% IR={best['超额IR']:+.2f} t={best['超额t']:+.2f}）")

    print("\n=== 3b. 测试期单次验证（参数来自训练期，未回挑） ===")
    R = run_topk(panel, te, topk=int(best['topk']), every=int(best['调仓周期']), pred_col='pred')
    m = metrics(R['net'], bench_te)
    mb = metrics(bench_te.reindex(R.index).dropna(), bench_te)
    print(f"  K={int(best['topk'])} every={int(best['调仓周期'])}  日均换手={R['turnover'].mean():.2f}")
    for k in ['年化', '年化超额', '超额IR', '超额t', '最大回撤', '日胜率']:
        print(f"    {k:<8} {m.get(k, float('nan')):.4f}")
    print(f"    （池等权基准同期年化 {mb['年化']*100:+.2f}%，日胜率 {mb['日胜率']:.1%}）")

    # 分年（测试期）
    print("\n=== 3c. 测试期分年 ===")
    rows = []
    for y in sorted(set(pd.Timestamp(d).year for d in te)):
        dy = [d for d in te if pd.Timestamp(d).year == y]
        Ry = run_topk(panel, dy, topk=int(best['topk']), every=int(best['调仓周期']), pred_col='pred')
        my = metrics(Ry['net'], bq.reindex(dy))
        rows.append({'年份': y, '年化超额': my.get('年化超额'), '超额t': my.get('超额t'),
                     '天数': my.get('天数')})
        print(f"  {y}  年化超额={my.get('年化超额', float('nan'))*100:+7.2f}%  "
              f"t={my.get('超额t', float('nan')):+.2f}  天数={my.get('天数')}")
    pd.DataFrame(rows).to_csv(OUT / 'warfare2_oos_by_year.csv', index=False)
    G.to_csv(OUT / 'warfare2_train_grid.csv', index=False)
    print("\n保存: warfare2_train_grid.csv / warfare2_oos_by_year.csv")


if __name__ == '__main__':
    main()
