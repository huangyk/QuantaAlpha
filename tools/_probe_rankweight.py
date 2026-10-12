"""
验证: 正 rank IC 是否应推出正 rank-weighted 多空收益?
若两者符号相反, 说明有 bug 或尾部效应, 必须查清才能下判词。
"""
import numpy as np, pandas as pd, psycopg2
DB = dict(host='127.0.0.1', port=5432, dbname='quant_data', user='quant', password='quant123456')

c = psycopg2.connect(**DB); cur = c.cursor()
cur.execute("""
    select b.asset_code, b.trade_date, b.open, b.close, b.float_shares,
           coalesce(m.is_st,false) as is_st
    from bars_1d b left join asset_master m on m.asset_code=b.asset_code
    where b.trade_date between '2022-01-01' and '2026-09-30'
      and b.asset_type is distinct from 'etf'
    order by b.asset_code, b.trade_date
""")
rows = cur.fetchall(); cur.close(); c.close()
df = pd.DataFrame(rows, columns=['code','date','open','close','float_shares','is_st'])
df['date'] = pd.to_datetime(df['date'])
grp = df.groupby('code')
df['listed_n'] = grp.cumcount()+1
df['prev_close'] = grp['close'].shift(1)
df['ret_fwd'] = grp['open'].shift(-1)/df['open']-1.0
df['logret'] = np.log(grp['close'].transform(lambda s: s/s.shift(1)))
df['vol20'] = grp['logret'].transform(lambda s: s.rolling(20,min_periods=20).std())
df['float_mcap'] = df['close']*df['float_shares']
df['vol20'] = grp['vol20'].transform(lambda s: s.ffill(limit=5))
df['float_mcap'] = grp['float_mcap'].transform(lambda s: s.ffill(limit=5))
df['limit_up'] = (df['open']/df['prev_close']>=1.098)
df['limit_dn'] = (df['open']/df['prev_close']<=0.902)
valid = (df['listed_n']>=250)&(~df['limit_up'])&(~df['limit_dn'])&(~df['is_st']) \
    & df['vol20'].notna()&df['float_mcap'].notna()&df['ret_fwd'].notna() \
    & (df['open']>0)&(df['ret_fwd'].abs()<0.5)
panel = df[valid].copy()
panel['sig_raw'] = -panel.groupby('date')['vol20'].rank(pct=True)
panel['size_rank'] = panel.groupby('date')['float_mcap'].rank(pct=True)
panel['tgt_rank'] = panel.groupby('date')['ret_fwd'].rank(pct=True)
resid=[]
for d,g in panel.groupby('date'):
    x=g['size_rank'].values; y=g['sig_raw'].values
    X=np.column_stack([np.ones_like(x),x])
    beta,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@beta
    resid.append(pd.Series(r,index=g.index))
panel['sig_ind']=pd.concat(resid)

sp_list=[]; rw_list=[]; corr_ret_list=[]
for d,g in panel.groupby('date'):
    if len(g)<50: continue
    sr = g['sig_ind'].rank(pct=True)
    sp = g['sig_ind'].corr(g['tgt_rank'])          # Spearman (rank vs rank)
    cr = g['sig_ind'].corr(g['ret_fwd'])            # Pearson of signal vs raw ret
    w = sr-0.5
    rw = (w*g['ret_fwd']).sum()/len(g)              # rank-weighted zero-cost P&L
    sp_list.append(sp); corr_ret_list.append(cr); rw_list.append(rw)

sp=np.array(sp_list); cr=np.array(corr_ret_list); rw=np.array(rw_list)
print(f"日均 Spearman(sig_ind, rank_ret) = {sp.mean():+.4f}  t={sp.mean()/(sp.std()/np.sqrt(len(sp))):+.2f}")
print(f"日均 Pearson(sig_ind, ret_fwd)  = {cr.mean():+.4f}")
print(f"日均 rank-weighted P&L           = {rw.mean()*1e4:+.3f} bp/日  -> 年化 {(1+rw.mean())**252-1:+.2%}")
print(f"两者符号关系: Spearman>0 ? {sp.mean()>0}   rank-weighted>0 ? {rw.mean()>0}")
print(f"corr(spearman每日, rankP&L每日) = {np.corrcoef(sp,rw)[0,1]:+.3f}")
print(f"corr(pearson_ret每日, rankP&L每日) = {np.corrcoef(cr,rw)[0,1]:+.3f}")
# 拆解: rank-weighted P&L 的多/空两腿各自均值
ls_leg=[]; sh_leg=[]
for d,g in panel.groupby('date'):
    if len(g)<50: continue
    sr=g['sig_ind'].rank(pct=True)
    long_ret = g[sr>=0.5]['ret_fwd'].mean()   # 低波动半
    short_ret= g[sr<0.5]['ret_fwd'].mean()    # 高波动半
    ls_leg.append(long_ret); sh_leg.append(short_ret)
print(f"低波动半日均收益 {np.mean(ls_leg)*1e4:+.3f} bp | 高波动半日均收益 {np.mean(sh_leg)*1e4:+.3f} bp | 差 { (np.mean(ls_leg)-np.mean(sh_leg))*1e4:+.3f} bp")
