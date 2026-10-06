"""验证飞龙三因子 + 大盘regime：当前状态、历史regime、按regime分组的飞龙表现。"""
import qlib, pandas as pd, numpy as np
from quantaalpha.strategy.feilong import feilong_factors, market_regime, compute_breadth

FW = 5  # 前瞻天数

def main():
    qlib.init(provider_uri='D:/qqmt/qlib_data/cn_real_full', region='cn')
    from qlib.data import D
    start, end = '2024-06-01', '2026-09-30'
    fields = ['$open','$high','$low','$close','$volume','$factor']
    idx_df = D.features(['SH000300'], fields, start_time=start, end_time=end, freq='day').droplevel('instrument').sort_index()
    codes = D.list_instruments(D.instruments('liquid500'), start_time=start, end_time=end, as_list=True)
    df = D.features(codes, fields, start_time=start, end_time=end, freq='day').sort_index()
    panels = {}
    for inst, g in df.groupby(level='instrument'):
        if isinstance(g.index, pd.MultiIndex): g = g.droplevel('instrument')
        panels[inst] = g.sort_index()
    print(f"股票池 {len(panels)} 只, 指数 {len(idx_df)} 天, 区间 {idx_df.index[0].date()}~{idx_df.index[-1].date()}")

    breadth = compute_breadth(panels, 20).reindex(idx_df.index).ffill()
    reg = market_regime(idx_df, breadth)
    last = reg.iloc[-1]
    print(f"\n=== 当前大盘检测 ({last.name.date()}) ===")
    print(f"  regime={last['regime']}  score={last['regime_score']:.0f}/4  scale={last['regime_scale']:.1f}")
    print(f"  指数MA20={last['ma20']:.1f} MA60={last['ma60']:.1f} 距60日高={last['dd60']*100:.1f}% 波动分位={last['vol_pct']:.2f} 宽度={last['breadth']*100:.0f}%")
    vc = reg['regime'].value_counts()
    print(f"\n=== regime 分布 ({len(reg)}天) ===")
    for k in ['bullish','neutral','bearish']:
        print(f"  {k}: {vc.get(k,0)} 天 ({vc.get(k,0)/len(reg)*100:.0f}%)")

    # 长表：索引 (datetime, instrument)
    frames=[]
    for inst,g in panels.items():
        f=feilong_factors(g)
        d=pd.DataFrame({'entry':f['feilong_entry'],'exit':f['feilong_exit'],'rr':f['feilong_rr'],
                        'close':g['$close']})
        d['fwd']=d['close'].shift(-FW)/d['close']-1
        d=d.reset_index(); d['instrument']=inst
        frames.append(d.set_index(['datetime','instrument']))
    long=pd.concat(frames)

    recs=[]
    for dt in reg.index:
        try: sub=long.xs(dt, level='datetime')
        except KeyError: continue
        sub=sub.dropna(subset=['entry','fwd'])
        if len(sub)<50: continue
        top=sub.nlargest(max(1,int(len(sub)*0.1)),'entry')
        recs.append({'date':dt,'regime':reg.loc[dt,'regime'],
                     'top_fwd':top['fwd'].mean(),'all_fwd':sub['fwd'].mean()})
    R=pd.DataFrame(recs).set_index('date')
    print(f"\n=== 飞龙 entry top10% 的 {FW}日收益，按大盘regime分组 (共{len(R)}天) ===")
    for k in ['bullish','neutral','bearish']:
        s=R[R['regime']==k]
        if len(s)==0: continue
        rel=(s['top_fwd']-s['all_fwd']).mean()
        print(f"  {k:8s} 天数={len(s):3d}  top10%={s['top_fwd'].mean()*100:+.3f}%  全池={s['all_fwd'].mean()*100:+.3f}%  超额={rel*100:+.3f}%")
    R.to_csv('data/results/feilong_regime_forward.csv', encoding='utf-8')
    reg.to_csv('data/results/market_regime.csv', encoding='utf-8')
    print("\n保存: data/results/feilong_regime_forward.csv, data/results/market_regime.csv")

if __name__=='__main__':
    main()
