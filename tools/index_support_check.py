"""
指数级 支撑/压力 检验工具 (回答"指数关口分析我们能不能做")
=================================================================
复用 TimescaleDB 的 index 数据(上证/创业板/科创50 等), 验证某"关口"是否真有支撑:
  1) 当前价位相对关口的位置(上方/下方/距离)
  2) 历史触达该关口±band 后, 未来 fwd 日收益分布(均值/中位/胜率) -> 是否真"支撑"
  3) 通用整数关口扫描 -> 该指数哪些百点/五十点真有支撑(避免被单个数字误导)

与个股横截面 alpha 不同: 指数时序动量/支撑是另一类问题, 本工具专门验它。
"""
import argparse, re, psycopg2, numpy as np, pandas as pd
DB = dict(host='127.0.0.1', port=5432, dbname='quant_data', user='quant', password='quant123456')

IDX = {'上证指数':'000001.SH','创业板指':'399006.SZ','科创50':'000688.SH',
       '上证50':'000016.SH','沪深300':'000300.SH','中证500':'000905.SH'}

def load(code, start='2018-01-01'):
    con = psycopg2.connect(**DB); cur = con.cursor()
    cur.execute("SELECT trade_date, close FROM bars_1d WHERE asset_code=%s AND trade_date>=%s ORDER BY trade_date",
                (code, start))
    r = cur.fetchall(); cur.close(); con.close()
    s = pd.DataFrame(r, columns=['date','close']).set_index('date')['close']
    s.index = pd.to_datetime(s.index)
    return s

def support_test(s, level, band=0.015, fwd=20):
    s = s.dropna()
    touch = s[(s >= level*(1-band)) & (s <= level*(1+band))]
    if len(touch) < 3:
        return dict(n=len(touch), mean=np.nan, med=np.nan, wr=np.nan)
    rets = []
    for d in touch.index:
        i = s.index.get_loc(d)
        if i + fwd < len(s):
            rets.append(s.iloc[i+fwd]/s.iloc[i] - 1)
    rets = np.array(rets)
    return dict(n=len(rets), mean=rets.mean(), med=np.median(rets), wr=(rets>0).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default='上证指数')
    ap.add_argument('--levels', default='', help='逗号分隔, 如 3700,3741')
    ap.add_argument('--band', type=float, default=0.015)
    ap.add_argument('--fwd', type=int, default=20)
    args = ap.parse_args()
    code = IDX.get(args.index, args.index)
    s = load(code)
    latest = s.iloc[-1]; d0 = s.index.max().date()
    print(f'\n### {args.index} ({code})  最新 {d0} 收盘 {latest:.2f}  '
          f'近250日[{s.iloc[-250:].min():.0f},{s.iloc[-250:].max():.0f}]')

    levels = [float(x) for x in args.levels.split(',') if x] if args.levels else []
    print(f'\n--- 指定关口支撑检验 (触达±{args.band*100:.1f}%, 未来{args.fwd}日) ---')
    for lv in levels:
        r = support_test(s, lv, args.band, args.fwd)
        pos = '上方' if latest > lv else ('下方' if latest < lv else '持平')
        dist = (latest/lv - 1)*100
        print(f'  关口 {lv:.0f}: 现价 {latest:.0f} 在其{pos} ({dist:+.1f}%) | '
              f'历史触达 {r["n"]} 次, 未来{args.fwd}日均值 {r["mean"]*100:+.1f}%, '
              f'中位 {r["med"]*100:+.1f}%, 胜率 {r["wr"]*100:.0f}%')
        verdict = '✓真支撑' if (r["wr"] and r["wr"]>=0.55 and r["mean"]>0) else ('✗非支撑(易破)' if (r["wr"] and r["wr"]<0.5) else '～中性')
        print(f'        判定: {verdict}')

    # 通用整数关口扫描
    print(f'\n--- 通用整数关口扫描 (找该指数哪些关口真有支撑) ---')
    lo, hi = s.iloc[-250:].min(), s.iloc[-250:].max()
    step = 100 if latest > 2000 else 50
    res = []
    for lv in range(int(lo//step)*step, int(hi//step)*step + 1, step):
        r = support_test(s, lv, args.band, args.fwd)
        if r['n'] >= 3 and np.isfinite(r['mean']):
            res.append((lv, r['n'], r['mean'], r['wr']))
    res.sort(key=lambda x: -x[2])
    good = [x for x in res if x[3] >= 0.55 and x[2] > 0]
    bad  = [x for x in res if x[3] < 0.5]
    print(f'  真支撑关口(胜率>=55%且均值>0): ' + (', '.join(f'{l}(+{m*100:.1f}%/{w*100:.0f}%)' for l,n,m,w in good) or '无'))
    print(f'  易破关口(胜率<50%):          ' + (', '.join(f'{l}({m*100:+.1f}%/{w*100:.0f}%)' for l,n,m,w in bad) or '无'))
    if res:
        print(f'  全部{len(res)}个关口均值收益平均 {np.mean([x[2] for x in res])*100:+.1f}%, '
              f'胜率>0占比 {(np.mean([x[3]>0.5 for x in res])*100):.0f}%')

if __name__ == '__main__':
    main()
