"""稳健性检验：飞龙多因子预测IC + 降换手(rebalance每N日)对照，验证闸门结论。"""
import numpy as np, pandas as pd, lightgbm as lgb, qlib
from quantaalpha.strategy.feilong import feilong_factors, market_regime, compute_breadth

P='D:/qqmt/qlib_data/cn_real_full'; S,E='2024-06-01','2026-09-30'
F=['$open','$high','$low','$close','$volume','$factor']; TOPK=30; COST=0.002
def main():
    qlib.init(provider_uri=P, region='cn')
    from qlib.data import D
    idx=D.features(['SH000300'],F,start_time=S,end_time=E,freq='day').droplevel('instrument').sort_index()
    codes=D.list_instruments(D.instruments('liquid500'),start_time=S,end_time=E,as_list=True)
    df=D.features(codes,F,start_time=S,end_time=E,freq='day').sort_index()
    panels={}
    for i,g in df.groupby(level='instrument'):
        if isinstance(g.index,pd.MultiIndex): g=g.droplevel('instrument')
        panels[i]=g.sort_index()
    br=compute_breadth(panels,20).reindex(idx.index).ffill()
    reg=market_regime(idx,br)
    # panel
    fr=[]
    for inst,g in panels.items():
        f=feilong_factors(g)
        d=pd.DataFrame({'e':f['feilong_entry'],'x':f['feilong_exit'],'r':f['feilong_rr'],'c':g['$close']})
        d['fwd1']=d['c'].shift(-1)/d['c']-1
        d=d.reset_index(); d['instrument']=inst; fr.append(d.set_index(['datetime','instrument']))
    panel=pd.concat(fr)
    dates=sorted(set(idx.index)); split=int(len(dates)*0.6)
    tr_d,te_d=dates[:split],dates[split:]
    feats=['e','x','r']
    tr=panel.loc[tr_d].dropna(subset=feats+['fwd1']); te=panel.loc[te_d].dropna(subset=feats+['fwd1'])
    cs=lambda d: d[feats].groupby(level='datetime').rank(pct=True)
    m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,subsample=.8,colsample_bytree=.8,verbose=-1,random_state=42)
    m.fit(cs(tr),tr['fwd1'])
    panel['pred']=np.nan; panel.loc[te.index,'pred']=m.predict(cs(te))
    # 预测IC(测试期, 逐日截面rank相关)
    ics=[]
    for dt in te_d:
        try: s=panel.xs(dt,level='datetime').dropna(subset=['pred','fwd1'])
        except KeyError: continue
        if len(s)>30: ics.append(s['pred'].corr(s['fwd1'],method='spearman'))
    print(f"多因子预测 IC(测试期日均)={np.nanmean(ics):.4f}  ICIR={np.nanmean(ics)/np.nanstd(ics):.2f}")
    # 降换手: 每N日调仓一次, 持仓不动
    def bt(reb):
        hold=None; rows=[]
        for i,dt in enumerate(te_d):
            if i%reb!=0 and hold is not None:
                # 持有不动, 收益用持仓次日均值近似=整体? 简化: 跳过(不逐日复利)
                continue
            try: s=panel.xs(dt,level='datetime').dropna(subset=['pred','fwd1'])
            except KeyError: continue
            if len(s)<TOPK: continue
            sel=s.nlargest(TOPK,'pred'); cur=set(sel.index)
            turn=1.0 if hold is None else 1-len(cur&hold)/max(len(cur),1)
            rows.append({'date':dt,'net':sel['fwd1'].mean()-turn*COST,'bench':idx['$close'].pct_change().get(dt,np.nan),
                         'turn':turn,'scale':reg['regime_scale'].get(dt,1.0)})
            hold=cur
        return pd.DataFrame(rows).set_index('date')
    print("\n=== 降换手对照(每5日调仓) ===")
    for nm,gate in [('A无闸门',None),('B有闸门',reg['regime_scale'].to_dict())]:
        r=bt(5)
        net=r['net'] if gate is None else r['net']*r['scale']
        ex=net-r['bench'].fillna(0)
        print(f"  {nm}: 年化超额={ex.mean()*252*100:+.2f}%  IR={(ex.mean()/ex.std()*np.sqrt(252)) if ex.std()>0 else np.nan:.2f}  累计={ex.cumsum().iloc[-1]*100:+.2f}%")
if __name__=='__main__': main()
