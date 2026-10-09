"""
指数"支撑位抢反弹、赚几个点就走" 回测 (回应"别人赚几个点就走"的玩法)
=================================================================
玩法: 价格触达支撑带(level±band) -> 次日开盘买入 -> 触 +tp% 止盈 或 -sl% 止损 或 持有maxhold日退出。
检验: 在用户引用的4个关口(上证3741/3700, 创业板2900, 科创501200)上, 这个"几个点"打法期望是否为正。
对比: 真支撑 vs 假支撑(数据已证伪的) -> 说明"赚几个点就走"只在真支撑上成立。
成本: 指数ETF 双边~万3(万1佣金+微小冲击), 远小于个股小盘30bp。
"""
import sys, argparse, psycopg2, numpy as np, pandas as pd
sys.path.insert(0, r'D:/qqmt/QuantaAlpha/tools')
from index_support_check import IDX
DB = dict(host='127.0.0.1', port=5432, dbname='quant_data', user='quant', password='quant123456')

def load_ohlc(code, start='2018-01-01'):
    con = psycopg2.connect(**DB); cur = con.cursor()
    cur.execute("SELECT trade_date, open, high, low, close FROM bars_1d WHERE asset_code=%s AND trade_date>=%s ORDER BY trade_date", (code, start))
    r = cur.fetchall(); cur.close(); con.close()
    df = pd.DataFrame(r, columns=['date','open','high','low','close']).set_index('date')
    df.index = pd.to_datetime(df.index)
    return df.dropna()

def bounce_backtest(df, level, band=0.02, tp=0.03, sl=0.03, maxhold=20, cost=0.0003):
    """信号日收盘落入支撑带 -> 次日开盘买 -> 路径退出。返回逐笔。"""
    c = df['close'].values.astype(float); o = df['open'].values.astype(float)
    h = df['high'].values.astype(float); l = df['low'].values.astype(float)
    n = len(df); trades = []
    i = 0
    while i < n - 1:
        if c[i] >= level*(1-band) and c[i] <= level*(1+band):
            j = i + 1
            if j >= n: break
            entry = o[j]
            if entry <= 0 or not np.isfinite(entry):
                i += 1; continue
            tgt = entry*(1+tp); stp = entry*(1-sl)
            exited = False
            for k in range(j, min(j+maxhold, n)):
                if l[k] <= stp:
                    gross = stp/entry - 1; trades.append((gross, 'stop')); exited=True; i=k+1; break
                if h[k] >= tgt:
                    gross = tgt/entry - 1; trades.append((gross, 'tp')); exited=True; i=k+1; break
            if not exited:
                ex = c[min(j+maxhold-1, n-1)]
                trades.append((ex/entry - 1, 'hold')); i = j + maxhold
        else:
            i += 1
    if not trades:
        return dict(n=0)
    g = np.array([t[0] for t in trades]); typ = [t[1] for t in trades]
    net = g - cost
    wr = (g > 0).mean()
    return dict(n=len(trades), win_rate=float(wr), mean_gross=float(g.mean()*1e4),
                mean_net=float(net.mean()*1e4), expectancy=float(net.mean()*1e4),
                tp_share=float((np.array(typ)=='tp').mean()),
                stop_share=float((np.array(typ)=='stop').mean()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tp', type=float, default=0.03)
    ap.add_argument('--sl', type=float, default=0.03)
    ap.add_argument('--band', type=float, default=0.02)
    ap.add_argument('--maxhold', type=int, default=20)
    args = ap.parse_args()
    print(f'\n===== 指数"支撑抢反弹 赚{args.tp*100:.0f}点走/亏{args.sl*100:.0f}点砍" 回测 '
          f'(band±{args.band*100:.0f}%, 持有<= {args.maxhold}日, 成本万3) =====')
    targets = [('上证指数','000001.SH',3741,'~边际'),('上证指数','000001.SH',3700,'✗假支撑'),
               ('创业板指','399006.SZ',2900,'✓真支撑'),('科创50','000688.SH',1200,'✗假支撑')]
    print(f'  {"指数":<8}{"关口":>7}{"判定":>8}{"笔数":>6}{"胜率":>7}{"均净bp":>9}{"期望bp":>8}{"止盈%":>7}{"止损%":>7}')
    for nm,code,lv,tag in targets:
        df = load_ohlc(code)
        r = bounce_backtest(df, lv, args.band, args.tp, args.sl, args.maxhold)
        if r['n']==0:
            print(f'  {nm:<8}{lv:>7}{tag:>8}  无触达'); continue
        print(f'  {nm:<8}{lv:>7}{tag:>8}{r["n"]:>6}{r["win_rate"]*100:>6.0f}%'
              f'{r["mean_net"]:>9.0f}{r["expectancy"]:>8.0f}{r["tp_share"]*100:>7.0f}{r["stop_share"]*100:>7.0f}')
    # 网格: tp 2/3/4/5%, sl=3%
    print(f'\n--- 网格(真支撑 创业板2900 / 上证3741): tp vs 期望bp ---')
    for nm,code,lv in [('创业板2900','399006.SZ',2900),('上证3741','000001.SH',3741)]:
        df = load_ohlc(code)
        row=[]
        for tp in [0.02,0.03,0.04,0.05]:
            r=bounce_backtest(df,lv,args.band,tp,args.sl,args.maxhold)
            row.append(f'tp{tp*100:.0f}%:{r["expectancy"]:+.0f}bp' if r["n"] else 'NA')
        print(f'  {nm:<12}' + '  '.join(row))

if __name__ == '__main__':
    main()
