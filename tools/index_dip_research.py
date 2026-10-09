"""
指数"顺趋势抄跌" 真 edge 研究 (续 index_bounce_research 的疑点)
=================================================================
上一版"无脑抄5%回撤"因固定tp/sl使按年均值退化成同一值(伪象)。本版改用干净前向收益法:
  - 信号: 价格自近期高回撤>=depth% 且为"首次触达"(rising-edge触发, 带冷却) -> 次日均不重叠
  - 三类条件: A)任意抄跌(无趋势过滤) B)上升趋势中抄跌(收>MA60) C)下降趋势中抄跌(收<MA60)
  - 持有 H=10/20/40 日后收盘退出, Forward Return = close[i+1+H]/open[i+1]-1 (含万3成本)
  - 跨6指数 pooling 求 N/均值bp/t/胜率, 并按年拆(看regime稳定性, 无伪象)
目的: 回答"别人蛮准"最省解释——"顺趋势抄跌"是否有可测edge, 还是连这也脆弱。
"""
import sys, psycopg2, numpy as np, pandas as pd
sys.path.insert(0, r'D:/qqmt/QuantaAlpha/tools')
DB = dict(host='127.0.0.1', port=5432, dbname='quant_data', user='quant', password='quant123456')

def load_ohlc(code, start='2018-01-01'):
    con = psycopg2.connect(**DB); cur = con.cursor()
    cur.execute("SELECT trade_date, open, high, low, close FROM bars_1d WHERE asset_code=%s AND trade_date>=%s ORDER BY trade_date", (code, start))
    r = cur.fetchall(); cur.close(); con.close()
    df = pd.DataFrame(r, columns=['date','open','high','low','close']).set_index('date')
    df.index = pd.to_datetime(df.index)
    return df.dropna()

INDICES = [('上证指数','000001.SH'),('沪深300','000300.SH'),('上证50','000016.SH'),
           ('中证500','000905.SH'),('创业板指','399006.SZ'),('科创50','000688.SH')]

def dip_signals(df, depth=0.05, ma_w=60):
    c = df['close'].values.astype(float); n = len(c)
    rollmax = pd.Series(c).rolling(20).max().values
    ma = pd.Series(c).rolling(ma_w).mean().values
    cond_any = c <= (1-depth)*rollmax
    cond_up = cond_any & (c > ma)
    cond_dn = cond_any & (c < ma)
    def triggers(mask):
        sig = np.zeros(n, bool); cool = False
        for i in range(n):
            if cool:
                if not mask[i]: cool = False
                continue
            if mask[i]:
                sig[i] = True; cool = True
        return sig
    return triggers(cond_any), triggers(cond_up), triggers(cond_dn)

def fwd_returns(df, sig, H=20, cost=0.0003):
    o = df['open'].values.astype(float); c = df['close'].values.astype(float); dates = df.index
    n = len(df); out = []
    for i in range(n):
        if not sig[i] or i+1+H >= n: continue
        entry = o[i+1]
        if not np.isfinite(entry) or entry <= 0: continue
        ex = c[i+1+H]
        if not np.isfinite(ex): continue
        out.append((ex/entry - 1 - cost, dates[i+1].year))
    return out

def main():
    print('加载 6 指数 ...')
    data = {nm: load_ohlc(code) for nm, code in INDICES}
    horizons = [10, 20, 40]
    depths = [0.03, 0.05, 0.08]
    print(f'\n===== 顺趋势抄跌 vs 任意抄跌 vs 下降抄跌 (pooling 6指数) =====')
    print(f'{"条件":<10}{"回撤":>6}{"H":>4}{"N":>7}{"均bp":>8}{"t":>7}{"胜率":>7}')
    allrows = []
    for depth in depths:
        for H in horizons:
            rows = []
            for label, fn in [('A任意', lambda d: dip_signals(d, depth)[0]),
                              ('B升势', lambda d: dip_signals(d, depth)[1]),
                              ('C降势', lambda d: dip_signals(d, depth)[2])]:
                allg = []; ally = []
                for nm, df in data.items():
                    sig = fn(df)
                    for g, y in fwd_returns(df, sig, H):
                        allg.append(g); ally.append(y)
                g = np.array(allg)*1e4; N=len(g)
                if N < 30: continue
                mean = g.mean(); std = g.std()
                t = mean/(std/np.sqrt(N)) if std>0 else 0
                wr = (g>0).mean()
                rows.append((label, depth*100, H, N, mean, t, wr))
                allrows.append((label, depth*100, H, N, mean, t, wr))
            for r in rows:
                print(f'{r[0]:<10}{r[1]:>5.0f}%{r[2]:>4}{r[3]:>7}{r[4]:>+8.0f}{r[5]:>+7.1f}{r[6]*100:>6.0f}%')
            print()
    # 稳健最佳: 仅 N>=100 且按 t 显著性排序
    big = [r for r in allrows if r[3] >= 100]
    big.sort(key=lambda r: -r[5])
    print(f'>>> 稳健最佳(N>=100, 按t排序): '
          f'{big[0][0]} 回撤{big[0][1]:.0f}% H{big[0][2]} 均{big[0][4]:+.0f}bp t={big[0][5]:+.1f} 胜率{big[0][6]*100:.0f}% N={big[0][3]}')
    sig = [r for r in big if r[5] > 2]
    print(f'>>> 达到 t>2 显著的组合数: {len(sig)} (0 即无任何显著edge)')
    # 按年拆 稳健最佳(无伪象)
    bl, bd, bH = big[0][0], big[0][1]/100, big[0][2]
    fn = {'A任意':0,'B升势':1,'C降势':2}[bl]
    print(f'\n===== 稳健最佳({bl} 回撤{bd*100:.0f}% H{bH}) 按年(regime稳定性) =====')
    allg=[]; ally=[]
    for nm, df in data.items():
        sig_vec = dip_signals(df, bd)[fn]
        for g,y in fwd_returns(df, sig_vec, bH):
            allg.append(g); ally.append(y)
    s = pd.Series(np.array(allg)*1e4, index=ally).groupby(level=0).agg(['mean','count'])
    for y,row in s.iterrows():
        print(f'  {y}: 均 {row["mean"]:+.0f}bp (n={int(row["count"])})')

if __name__ == '__main__':
    main()
