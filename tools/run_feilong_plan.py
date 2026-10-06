import qlib, pandas as pd, json
from quantaalpha.strategy.feilong import feilong_factors, market_regime, compute_breadth, generate_feilong_plan
def main():
    qlib.init(provider_uri='D:/qqmt/qlib_data/cn_real_full', region='cn')
    from qlib.data import D
    start,end='2024-06-01','2026-09-30'; F=['$open','$high','$low','$close','$volume','$factor']
    idx=D.features(['SH000300'],F,start_time=start,end_time=end,freq='day').droplevel('instrument').sort_index()
    codes=D.list_instruments(D.instruments('liquid500'),start_time=start,end_time=end,as_list=True)
    df=D.features(codes,F,start_time=start,end_time=end,freq='day').sort_index()
    panels={}
    for i,g in df.groupby(level='instrument'):
        if isinstance(g.index,pd.MultiIndex): g=g.droplevel('instrument')
        panels[i]=g.sort_index()
    br=compute_breadth(panels,20).reindex(idx.index).ffill()
    reg=market_regime(idx,br)
    out={}
    for d in ['2026-09-30','2026-09-04']:
        r=generate_feilong_plan(panels,reg,d,top_n=5)
        out[d]=r
        print(f"\n===== 飞龙交易计划 {d} | regime={r['regime']}(score {r.get('regime_score')}/4, scale {r['regime_scale']}) 指数MA20={r.get('index_ma20')} MA60={r.get('index_ma60')} 宽度={r.get('breadth')}% =====")
        if not r['plans']: print("  ", r.get('note')); continue
        for p in r['plans']:
            print(f"  {p['instrument']} entry={p['feilong_entry']:.3f} rr={p['feilong_rr']:.2f} exitP={p['feilong_exit_pressure']:.3f} | 收{p['close_T']} 止损{p['stop_loss']}(-{p['stop_pct']}%) 目标{p['take_profit']}(+{p['tp_pct']}%) 仓位{p['position_pct']}%")
    json.dump(out,open('data/results/feilong_trade_plan.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print("\n保存: data/results/feilong_trade_plan.json")
if __name__=='__main__': main()
