"""最小探针: 钉死低波动信号的符号 (IC 与 多空价差 是否同号)"""
import numpy as np, pandas as pd, psycopg2
DB = dict(host='127.0.0.1', port=5432, dbname='quant_data', user='quant', password='quant123456')
c = psycopg2.connect(**DB); cur = c.cursor()
cur.execute("""select b.asset_code, b.trade_date, b.open, b.close, b.float_shares,
               coalesce(m.is_st,false) as is_st
               from bars_1d b left join asset_master m on m.asset_code=b.asset_code
               where b.trade_date between '2022-01-01' and '2026-09-30'
                 and (b.asset_type is distinct from 'etf') order by b.asset_code, b.trade_date""")
df = pd.DataFrame(cur.fetchall(), columns=['code','date','open','close','float_shares','is_st'])
df['date']=pd.to_datetime(df['date']); cur.close(); c.close()
grp=df.groupby('code')
df['listed_n']=grp.cumcount()+1
df['prev_close']=grp['close'].shift(1)
df['ret_fwd']=grp['open'].shift(-1)/df['open']-1.0
df['logret']=np.log(grp['close'].transform(lambda s:s/s.shift(1)))
df['vol20']=grp['logret'].transform(lambda s:s.rolling(20,min_periods=20).std())
df['float_mcap']=df['close']*df['float_shares']
df['vol20']=grp['vol20'].transform(lambda s:s.ffill(limit=5))
df['float_mcap']=grp['float_mcap'].transform(lambda s:s.ffill(limit=5))
df['limit_up']=(df['open']/df['prev_close']>=1.098); df['limit_dn']=(df['open']/df['prev_close']<=0.902)
valid=(df['listed_n']>=250)&(~df['limit_up'])&(~df['limit_dn'])&(~df['is_st'])&df['vol20'].notna()&df['float_mcap'].notna()&df['ret_fwd'].notna()&(df['open']>0)&(df['ret_fwd'].abs()<0.5)
p=df[valid].copy()
# 信号: 低波动=高 sig
p['sig']=-p.groupby('date')['vol20'].rank(pct=True)
p['tgt']=p.groupby('date')['ret_fwd'].rank(pct=True)
# 行业+市值中性化(简化: 仅市值, 先看符号)
def neu(g):
    x=g['float_mcap'].rank(pct=True).values; y=g['sig'].values
    X=np.column_stack([np.ones_like(x),x])
    b,*_=np.linalg.lstsq(X,y,rcond=None); return y-X@b
p['sig_neu']=p.groupby('date').apply(lambda g: pd.Series(neu(g),index=g.index)).reset_index(level=0,drop=True)
# 对齐列
p=p.sort_values(['date','code'])
# 逐日: corr(sig_neu,tgt) 与 (高sig十分位均值ret - 低sig十分位均值ret)
corrs=[]; spreads=[]
for d,g in p.groupby('date'):
    if len(g)<50: continue
    corrs.append(g['sig_neu'].corr(g['tgt']))
    sr=g['sig_neu'].rank(pct=True)
    top=g[sr>=0.9]['ret_fwd'].mean(); bot=g[sr<=0.1]['ret_fwd'].mean()
    spreads.append(top-bot)
corrs=np.array(corrs); spreads=np.array(spreads)
print('日度 IC(sig_neu vs ret排名): 均值 %.4f  t=%.2f'%(corrs.mean(), corrs.mean()/(corrs.std()/np.sqrt(len(corrs)))))
print('日度 多空价差(高sig-低sig):  均值 %+.4f  t=%.2f'%(spreads.mean(), spreads.mean()/(spreads.std()/np.sqrt(len(spreads)))))
print()
print('样本日 5 个:')
i=0
for d,g in p.groupby('date'):
    if len(g)<50: continue
    sr=g['sig_neu'].rank(pct=True)
    top=g[sr>=0.9]['ret_fwd'].mean(); bot=g[sr<=0.1]['ret_fwd'].mean()
    print('  %s  corr=%.3f  价差=%+.4f'%(d.date(), g['sig_neu'].corr(g['tgt']), top-bot))
    i+=1
    if i>=5: break
print()
print('结论: IC 与 价差 同号 => 信号方向自洽; 异号 => 仍有bug')
