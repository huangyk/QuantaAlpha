"""飞龙 V2 续2：闸门重定义——用「选股池自身状态」替代沪深300作环境指标。

V2 ③-a 暴露的问题：以沪深300 定义的 regime 下，
    bullish 日池等权年化 +42% / neutral +78% / bearish +168%（仅12天）
即"越看空的日子池子涨得越猛"——完全反向。根因是**环境指标与交易标的不匹配**：
沪深300 是大盘，选股池 liquid500 是流动性/中小盘，两者 2025-10~2026-09 走势背离
（沪深300 年化 -7.7%，池等权 +43.4%）。用大盘状态去管小盘仓位，自然是反向闸门。

本脚本用「池等权指数」自身（池内全部个股等权净值）重算同一套四条件 regime，
检验：
    R1 池自身 regime 下，池等权日收益是否 bullish>neutral>bearish（尤其 bearish 是否显著为负）
    R2 池自身 regime 下，策略在各 regime 相对池等权的超额（bearish 是否真的更差）
    R3 两套 regime 的 scale 组合扫描对照（探索性）

输出：data/results/feilong_v2_gate_pool.csv / feilong_v2_gate_grid_pool.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from quantaalpha.strategy.feilong import market_regime, compute_breadth

CACHE = Path('data/results/_feilong_panel_v2.parquet')
OUT = Path('data/results')
TOPK, COST, SPLIT = 30, 0.002, '2025-10-01'
FEATS = ['feilong_entry', 'feilong_exit', 'feilong_rr']


def cs_rank(df, cols=FEATS):
    return df[cols].groupby(level='datetime').rank(pct=True)


def run_backtest(panel, dates, topk=TOPK, gate=None):
    hold, rows = None, []
    for dt in dates:
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=['pred', 'fwd', 'bench_eq'])
        if len(sub) < topk:
            continue
        sel = sub.nlargest(topk, 'pred')
        cur = set(sel.index)
        turn = 1.0 if hold is None else 1.0 - len(cur & hold) / max(len(cur), 1)
        scale = 1.0 if gate is None else float(gate.get(dt, 1.0))
        rows.append({'date': dt, 'net': sel['fwd'].mean() - turn * COST, 'turnover': turn,
                     'bench_eq': sub['bench_eq'].iloc[0], 'scale': scale})
        hold = cur
    return pd.DataFrame(rows).set_index('date')


def metrics_ex(rets, bench=None):
    rets = rets.dropna()
    if len(rets) == 0:
        return {}
    out = {'年化': rets.mean() * 252, '天数': len(rets)}
    if bench is not None:
        ex = rets - bench.reindex(rets.index).fillna(0)
        out['年化超额'] = ex.mean() * 252
        out['超额IR'] = ex.mean() / ex.std(ddof=1) * np.sqrt(252) if ex.std(ddof=1) > 0 else np.nan
        out['超额t'] = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if ex.std(ddof=1) > 0 else np.nan
    return out


def main():
    panel = pd.read_parquet(CACHE)
    bq = panel.groupby(level='datetime')['fwd'].mean().rename('bench_eq')
    panel = panel.join(bq, on='datetime')
    all_dates = sorted(set(panel.index.get_level_values('datetime')))
    te_dates = [d for d in all_dates if d >= pd.Timestamp(SPLIT)]
    tr_dates = [d for d in all_dates if d < pd.Timestamp(SPLIT)]

    # 池宽度：站上 MA20 的个股占比（用池自身，不用大盘）
    pool_panels = {}
    for i, g in panel.groupby(level='instrument'):
        gg = g.droplevel('instrument').copy()
        gg['$close'] = gg['close']          # compute_breadth 约定列名
        pool_panels[i] = gg
    breadth = compute_breadth(pool_panels, 20)
    # 池等权指数：用池内等权日收益累乘（fwd 是 T+1开盘→T+2开盘，对齐到 T+1 更贴切，此处用同口径即可）
    pool_ret = bq.dropna()
    pool_idx = (1 + pool_ret).cumprod()
    pool_df = pd.DataFrame({'$close': pool_idx})

    reg_pool = market_regime(pool_df, breadth.reindex(pool_idx.index).ffill(),
                             p={'index_vol_pct_window': 120})
    valid = [d for d in reg_pool.index if pd.notna(reg_pool.loc[d, 'vol_pct'])]
    print(f"池自身 regime 有效区间: {valid[0].date()} ~ {valid[-1].date()} ({len(valid)}天)")

    # ---- R1 池自身 regime 下池等权的表现 ----
    print("\n=== R1 池自身 regime → 池等权日收益（对照 V2 的沪深300 regime） ===")
    g = pd.DataFrame({'bench_eq': bq.reindex(valid),
                      'regime': reg_pool.loc[valid, 'regime']}).dropna()
    rows = []
    for rg in ['bullish', 'neutral', 'bearish']:
        s = g[g['regime'] == rg]
        if len(s) < 2:
            print(f"  {rg:8s} 样本不足 ({len(s)})")
            rows.append({'regime': rg, '天数': len(s)})
            continue
        t = s['bench_eq'].mean() / (s['bench_eq'].std(ddof=1) / np.sqrt(len(s)))
        print(f"  {rg:8s} 天数={len(s):3d}  池等权日均={s['bench_eq'].mean()*100:+.4f}%  "
              f"年化={s['bench_eq'].mean()*252*100:+.2f}%  t={t:+.2f}")
        rows.append({'regime': rg, '天数': len(s), '池等权年化': s['bench_eq'].mean() * 252, 't': t})
    pd.DataFrame(rows).to_csv(OUT / 'feilong_v2_gate_pool.csv', index=False)

    # ---- R2 策略在各 regime 的超额 ----
    tr_all = panel.loc[tr_dates].dropna(subset=FEATS + ['fwd'])
    te = panel.loc[te_dates].dropna(subset=FEATS + ['fwd'])
    model = lgb.LGBMRegressor(n_estimators=60, learning_rate=0.05, num_leaves=7, subsample=0.8,
                              colsample_bytree=0.8, verbose=-1, random_state=42)
    model.fit(cs_rank(tr_all), tr_all['fwd'])
    panel['pred'] = np.nan
    panel.loc[te.index, 'pred'] = model.predict(cs_rank(te))

    A = run_backtest(panel, te_dates)
    A['regime'] = reg_pool.reindex(A.index)['regime']
    A['ex'] = A['net'] - A['bench_eq']
    print("\n=== R2 策略A（无闸门）按池自身 regime 的超额 ===")
    for rg in ['bullish', 'neutral', 'bearish']:
        s = A[A['regime'] == rg]
        if len(s) < 2:
            print(f"  {rg:8s} 样本不足 ({len(s)})")
            continue
        t = s['ex'].mean() / (s['ex'].std(ddof=1) / np.sqrt(len(s)))
        print(f"  {rg:8s} 天数={len(s):3d}  年化超额={s['ex'].mean()*252*100:+.2f}%  t={t:+.2f}")

    # ---- R3 scale 扫描（池 regime） ----
    bench_te = bq.reindex(te_dates).dropna()
    print("\n=== R3 scale 组合扫描（池自身 regime，探索性） ===")
    out = []
    for bear, neu in [(1.0, 1.0), (0.5, 1.0), (0.0, 1.0), (0.0, 0.75), (0.0, 0.5), (0.5, 0.5)]:
        sc = {d: (1.0 if reg_pool['regime'].get(d, 'neutral') == 'bullish'
                  else (neu if reg_pool['regime'].get(d, 'neutral') == 'neutral' else bear))
              for d in te_dates}
        R = run_backtest(panel, te_dates, gate=sc)
        m = metrics_ex(R['net'] * R['scale'], bench_te.reindex(R.index))
        out.append({'bearish': bear, 'neutral': neu, '仓位均值': R['scale'].mean(),
                    '年化超额': m.get('年化超额'), '超额IR': m.get('超额IR'), '超额t': m.get('超额t')})
        print(f"  bear={bear:<4} neutral={neu:<4} 仓位均值={R['scale'].mean():.2f}  "
              f"年化超额={m.get('年化超额', float('nan'))*100:+7.2f}%  "
              f"IR={m.get('超额IR', float('nan')):+.3f}  t={m.get('超额t', float('nan')):+.2f}")
    pd.DataFrame(out).to_csv(OUT / 'feilong_v2_gate_grid_pool.csv', index=False)
    print("\n保存: feilong_v2_gate_pool.csv / feilong_v2_gate_grid_pool.csv")


if __name__ == '__main__':
    main()
