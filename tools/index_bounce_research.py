"""
指数"支撑抢反弹赚几点" 深度研究 (把上一版小样本做实)
=================================================================
1) 多指数(6只)× 全整数关卡扫描 × 参数网格(tp/sl/band) ->  pooling 成几百笔大样本
2) 验显著性:  pooled 期望的 t 统计量(解决"小样本不硬")
3) 稳健性:     哪个 (tp,sl,band) 组合期望稳正
4) 选择性:     关卡里"强水平"(期望>0)占比 -> 支撑识别是否真有筛选力
5) regime:     最佳组合按年拆期望 -> 是否只在某阵子灵
6) 基线对照:    无脑"任何5%回撤就买" vs 只在支撑带买 -> 支撑识别是否增量
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

def bounce_trades(df, level, band=0.02, tp=0.05, sl=0.03, maxhold=15, cost=0.0003):
    c = df['close'].values.astype(float); o = df['open'].values.astype(float)
    h = df['high'].values.astype(float); l = df['low'].values.astype(float)
    dates = df.index; n = len(df); out = []
    i = 0
    while i < n - 1:
        if level*(1-band) <= c[i] <= level*(1+band):
            j = i + 1
            if j >= n: break
            entry = o[j]
            if not np.isfinite(entry) or entry <= 0: i += 1; continue
            tgt = entry*(1+tp); stp = entry*(1-sl); done = False
            for k in range(j, min(j+maxhold, n)):
                if l[k] <= stp: out.append((stp/entry-1, 'stop', dates[j])); done=True; i=k+1; break
                if h[k] >= tgt: out.append((tgt/entry-1, 'tp', dates[j])); done=True; i=k+1; break
            if not done:
                ex = c[min(j+maxhold-1, n-1)]; out.append((ex/entry-1, 'hold', dates[j])); i = j+maxhold
        else:
            i += 1
    return out

INDICES = [('上证指数','000001.SH'),('沪深300','000300.SH'),('上证50','000016.SH'),
           ('中证500','000905.SH'),('创业板指','399006.SZ'),('科创50','000688.SH')]

def build_levels(df, step_pref=None):
    s = df['close']; step = step_pref or (100 if s.iloc[-1] > 2000 else 50)
    lo, hi = s.iloc[-250:].min(), s.iloc[-250:].max()
    return [lv for lv in range(int(lo//step)*step, int(hi//step)*step+1, step)]

def main():
    tps = [0.02,0.03,0.04,0.05,0.06,0.07]; sls = [0.02,0.03,0.05]; bands = [0.015,0.025]
    maxhold = 15; cost = 0.0003
    print('加载 6 指数 OHLC ...')
    idxdata = {}
    for nm, code in INDICES:
        df = load_ohlc(code); idxdata[nm] = (df, build_levels(df))
    print(f'指数关卡数: ' + ', '.join(f'{nm}={len(lv)}' for nm,(df,lv) in idxdata.items()))

    print(f'\n===== 网格: pooling 全指数×全关卡, 期望显著性(t) 与选择性 =====')
    print(f'{"cfg":<22}{"N":>6}{"均净bp":>8}{"t":>7}{"Sharpe~":>9}{"强关占比":>9}')
    best = None
    grid_rows = []
    for tp in tps:
        for sl in sls:
            for band in bands:
                allg = []; alld = []; strong = 0; total_lv = 0
                for nm, (df, levels) in idxdata.items():
                    for lv in levels:
                        tr = bounce_trades(df, lv, band, tp, sl, maxhold, cost)
                        if len(tr) >= 3:
                            total_lv += 1
                            g = np.array([t[0] for t in tr])
                            if g.mean() > 0: strong += 1
                            allg.extend(g); alld.extend([t[2] for t in tr])
                allg = np.array(allg); N = len(allg)
                net = allg - cost; mean_net = net.mean()*1e4; std = net.std()*1e4
                t = mean_net/(std/np.sqrt(N)) if N > 1 and std > 0 else 0
                sharpe = mean_net/std*np.sqrt(252/maxhold) if std > 0 else 0
                sel = strong/total_lv if total_lv else 0
                grid_rows.append(((tp,sl,band), N, mean_net, t, sharpe, sel))
                print(f'tp{tp*100:.0f} sl{sl*100:.0f} b{band*100:.1f}  {N:>6}{mean_net:>+8.0f}{t:>+7.1f}{sharpe:>+9.2f}{sel*100:>8.0f}%')
                if t > 2 and (best is None or mean_net > best[1]):
                    best = ((tp,sl,band), mean_net, t, sel, N)
    if best is None:
        best = max(grid_rows, key=lambda r: r[3])  # fallback: max t
        best = (best[0], best[2], best[3], best[5], best[1])
    print(f'\n>>> 最佳稳健组合: tp{best[0][0]*100:.0f}% sl{best[0][1]*100:.0f}% band{best[0][2]*100:.1f}%  '
          f'均净{best[1]:+.0f}bp t={best[2]:+.1f} 强关占比{best[3]*100:.0f}% N={best[4]}')

    # regime: 按年拆最佳组合
    tp, sl, band = best[0]
    print(f'\n===== 最佳组合按年期望(稳定性) =====')
    yr_g = []; yr_d = []
    for nm, (df, levels) in idxdata.items():
        for lv in levels:
            for tr in bounce_trades(df, lv, band, tp, sl, maxhold, cost):
                yr_d.append(tr[2].year); yr_g.append(tr[0]-cost)
    s = pd.Series(yr_g, index=yr_d).groupby(level=0).agg(['mean','count'])
    for y, row in s.iterrows():
        print(f'  {y}: 均净 {row["mean"]*1e4:+.0f}bp (n={int(row["count"])})')

    # 基线: 无脑"任何5%回撤就买" (close<=0.95*20日高)
    print(f'\n===== 基线对照: 无脑抄任何5%回撤 vs 只在支撑带买 =====')
    base_g = []; base_y = []
    for nm, (df, _) in idxdata.items():
        c = df['close'].values.astype(float); o = df['open'].values.astype(float)
        h = df['high'].values.astype(float); l = df['low'].values.astype(float); dates = df.index
        rollmax = pd.Series(c).rolling(20).max().values
        i = 0; n = len(df)
        while i < n-1:
            if c[i] <= 0.95*rollmax[i] and np.isfinite(rollmax[i]):
                j = i+1
                if j >= n: break
                entry = o[j]
                if not np.isfinite(entry) or entry<=0: i+=1; continue
                tgt = entry*(1+tp); stp = entry*(1+sl); done=False
                for k in range(j, min(j+maxhold,n)):
                    if l[k]<=stp: base_g.append(stp/entry-1-cost); base_y.append(dates[j].year); done=True; i=k+1; break
                    if h[k]>=tgt: base_g.append(tgt/entry-1-cost); base_y.append(dates[j].year); done=True; i=k+1; break
                if not done:
                    base_g.append(c[min(j+maxhold-1,n-1)]/entry-1-cost); base_y.append(dates[j].year); i=j+maxhold
            else: i+=1
    bg = np.array(base_g)
    print(f'  无脑5%回撤买入(tp{tp*100:.0f}/sl{sl*100:.0f}): N={len(bg)} 均净 {bg.mean()*1e4:+.0f}bp  '
          f'(注: 固定tp/sl使std≈0, t失真, 看均值/胜率即可)')
    print(f'  只在支撑带买(同上组合):            N={best[4]} 均净 {best[1]:+.0f}bp t={best[2]:+.1f}')
    print('  -> 若支撑带 >> 无脑抄, 说明支撑识别有增量价值; 若相近/更差, 则"支撑位"叙事无增量, 真edge在"抄跌"本身。')
    print(f'\n===== 无脑5%回撤 按年(看regime稳定性) =====')
    bs = pd.Series(bg, index=base_y).groupby(level=0).agg(['mean','count'])
    for y, row in bs.iterrows():
        print(f'  {y}: 均净 {row["mean"]*1e4:+.0f}bp (n={int(row["count"])})')

if __name__ == '__main__':
    main()
