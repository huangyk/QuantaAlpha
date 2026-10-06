"""扫描最近 N 个交易日的信号条件通过率，定位信号日（真实数据诊断）。"""
import qlib, pandas as pd, numpy as np
from quantaalpha.strategy.daily_trade_plan import (
    PlanParams, load_daily_panel, compute_indicators, screen_universe, confirm_signal)

def main():
    p = PlanParams()
    qlib.init(provider_uri='D:/qqmt/qlib_data/cn_real_full', region='cn')
    from qlib.data import D
    insts = D.instruments('liquid500')
    codes = D.list_instruments(insts, start_time='2024-06-01', end_time='2026-09-30', as_list=True)
    df = load_daily_panel(codes, '2024-06-01', '2026-09-30')
    panels = compute_indicators(df, p)
    rej = screen_universe(panels, p)
    alldates = sorted(set(df.index.get_level_values('datetime')))[-60:]
    keys = ['trend', 'pull', 'vol', 'calm', 'signal']
    stat = {k: 0 for k in keys}; per = {}
    for sd in alldates:
        c = {k: 0 for k in keys}
        for inst, g in panels.items():
            if inst in rej or sd not in g.index: continue
            idx = list(g.index).index(sd)
            if idx == 0: continue
            row = g.loc[sd]; prev = g.iloc[idx - 1]
            if np.isnan(row['close']): continue
            s = confirm_signal(inst, row, prev, p)['checks']
            c['trend'] += s['trend_ma_bull']; c['pull'] += s['pullback_ma']
            c['vol'] += s['vol_surge']; c['calm'] += s['calm_range']
            c['signal'] += s['trend_ma_bull'] and s['pullback_ma'] and s['vol_surge'] and s['calm_range']
        per[sd.strftime('%Y-%m-%d')] = c
        for k in keys: stat[k] += c[k]
    print("最近60交易日 累计通过标的日数:", stat)
    print("日均: " + "  ".join(f"{k}={stat[k]/60:.1f}" for k in keys))
    print("\n信号最多的日期:")
    for d, c in sorted(per.items(), key=lambda x: -x[1]['signal'])[:10]:
        print(" ", d, c)

if __name__ == '__main__':
    main()
