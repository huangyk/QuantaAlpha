"""
指数 动量/趋势跟踪 研究 (抄跌证伪后的对照方向)
=================================================================
学术上时间序列动量(TSMOM)/趋势跟踪在指数层有正预期, 与刚证伪的"抄跌/均值回归"相反。
干净法: 信号用前一日收盘算 -> 次日均(open)进场, 避免前视。现金日收益=0。
A) MA择时(仅多): 收>MA(W) 持仓, 否则空仓
B) TSMOM(多空): 仓位=sign(收_{t-1}/收_{t-1-L}-1), 日收益=仓位*指数日收益
指标: 年化收益 / Sharpe / 最大回撤, 与买入持有对比。
目的: 指数层到底有没有"可测显著edge"——若有, 在趋势跟踪而非抄跌。
"""
import sys, psycopg2, numpy as np, pandas as pd
sys.path.insert(0, r'D:/qqmt/QuantaAlpha/tools')
from index_dip_research import load_ohlc, INDICES
DB = dict(host='127.0.0.1', port=5432, dbname='quant_data', user='quant', password='quant123456')

def metrics(rets, switch_cost=0.0003):
    rets = np.array(rets, dtype=float)
    nav = np.cumprod(1 + rets)
    n = len(rets)
    ann = nav[-1]**(252/n) - 1 if nav[-1] > 0 else -1
    sharpe = rets.mean()/rets.std()*np.sqrt(252) if rets.std() > 0 else 0
    peak = np.maximum.accumulate(nav); mdd = (nav/peak - 1).min()
    # 切换成本已在rets里扣除(调用方处理)
    return dict(ann=ann, sharpe=sharpe, mdd=mdd, nav=nav[-1])

def ma_timing_rets(df, W=120, cost=0.0003):
    c = df['close'].values.astype(float); o = df['open'].values.astype(float)
    ma = pd.Series(c).rolling(W).mean().values
    pos = np.zeros(len(c))
    for t in range(1, len(c)):
        # 用前一日收盘 vs 前一日MA 判定, 次日均(open)进场 -> 无前视
        if np.isfinite(ma[t-1]) and c[t-1] > ma[t-1]:
            pos[t] = 1.0
    rets = []
    for t in range(1, len(c)):
        r = pos[t] * (c[t]/o[t] - 1)
        if t >= 2 and pos[t] != pos[t-1]:
            r -= cost
        rets.append(r)
    return np.array(rets)

def tsmom_rets(df, L=250, cost=0.0003):
    c = df['close'].values.astype(float); o = df['open'].values.astype(float)
    pos = np.zeros(len(c))
    for t in range(L+1, len(c)):
        sig = c[t-1]/c[t-1-L] - 1
        pos[t] = 1.0 if sig > 0 else (-1.0 if sig < 0 else 0.0)
    rets = []
    for t in range(1, len(c)):
        r = pos[t] * (c[t]/o[t] - 1)
        if t >= 2 and pos[t] != pos[t-1]:
            r -= cost
        rets.append(r)
    return np.array(rets)

def buyhold_rets(df):
    c = df['close'].values.astype(float); o = df['open'].values.astype(float)
    return np.array([c[t]/o[t]-1 for t in range(1, len(c))])

def main():
    data = {nm: load_ohlc(code) for nm, code in INDICES}
    print(f'{"指数":<10}{"策略":<16}{"年化":>8}{"Sharpe":>8}{"最大回撤":>10}')
    # 汇总(跨指数均值)
    agg = {}
    for nm, df in data.items():
        bh = metrics(buyhold_rets(df))
        ma60 = metrics(ma_timing_rets(df, 60)); ma120 = metrics(ma_timing_rets(df,120)); ma200 = metrics(ma_timing_rets(df,200))
        ts20 = metrics(tsmom_rets(df,20)); ts120 = metrics(tsmom_rets(df,120)); ts250 = metrics(tsmom_rets(df,250))
        print(f'{nm:<10}{"买入持有":<16}{bh["ann"]*100:>7.1f}%{bh["sharpe"]:>8.2f}{bh["mdd"]*100:>9.1f}%')
        for lbl, m in [("MA60择时",ma60),("MA120择时",ma120),("MA200择时",ma200),
                       ("TSMOM20",ts20),("TSMOM120",ts120),("TSMOM250",ts250)]:
            print(f'{"":<10}{lbl:<16}{m["ann"]*100:>7.1f}%{m["sharpe"]:>8.2f}{m["mdd"]*100:>9.1f}%')
        print()
        agg[nm] = dict(bh=bh, ma60=ma60, ma120=ma120, ma200=ma200, ts20=ts20, ts120=ts120, ts250=ts250)
    # 跨指数平均 Sharpe / 回撤 改善
    print('===== 跨6指数平均(趋势跟踪 vs 买入持有) =====')
    def avg(key, k): return np.mean([agg[n][key][k] for n in agg])
    print(f'{"策略":<16}{"平均年化":>10}{"平均Sharpe":>12}{"平均最大回撤":>14}')
    print(f'{"买入持有":<16}{avg("bh","ann")*100:>9.1f}%{avg("bh","sharpe"):>12.2f}{avg("bh","mdd")*100:>13.1f}%')
    for lbl,key in [("MA120择时","ma120"),("MA200择时","ma200"),("TSMOM120","ts120"),("TSMOM250","ts250")]:
        print(f'{lbl:<16}{avg(key,"ann")*100:>9.1f}%{avg(key,"sharpe"):>12.2f}{avg(key,"mdd")*100:>13.1f}%')
    print('\n解读: 趋势跟踪价值通常在"降回撤+提Sharpe"而非更高年化; 若平均Sharpe>买入持有且回撤更小, 则指数层确有可测edge(且方向与抄跌相反)。')

if __name__ == '__main__':
    main()
