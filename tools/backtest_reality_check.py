"""回测「可实现性」体检 —— 现有回测里有多少收益是**现实中买不到的**。

动机：前面所有日线回测（飞龙、战法2）都用 `open(T+1) 买入` 的口径，
但从未检查过这些标的在 T+1 开盘**是否真的买得进**。A 股一字涨停/大幅高开的票
在开盘集合竞价几乎买不到足额筹码，如果回测大量持有这类票，收益就是假的。

本脚本量化四件事：
    A. 每日「开盘买不进」的标的有多少（一字板 / 涨停开盘 / 停牌）
    B. 这些买不进的票，事后看贡献了多少收益（= 回测的虚假收益）
    C. 流动性：有多少标的日成交额不足，按 1e7 账户等权买入会砸盘
    D. 卖出端：跌停卖不掉的比例

口径：liquid500 池，日线 2020-01 ~ 2026-09（cn_real_full）。
输出：控制台 + data/results/reality_check_*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = 'D:/qqmt/qlib_data/cn_real_full'
START, END = '2020-01-01', '2026-09-30'
FIELDS = ['$open', '$high', '$low', '$close', '$volume']
OUT = Path('data/results')
ACCOUNT = 1e7          # 参考账户规模（元）
TOPK = 30              # 参考持仓只数


def load():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    codes = D.list_instruments(D.instruments('liquid500'), start_time=START, end_time=END, as_list=True)
    df = D.features(codes, FIELDS, start_time=START, end_time=END, freq='day').sort_index()
    panels = {}
    for i, g in df.groupby(level='instrument'):
        panels[i] = g.droplevel('instrument').sort_index() if isinstance(g.index, pd.MultiIndex) else g.sort_index()
    return panels


def build(panels):
    frames = []
    for inst, g in panels.items():
        o, h, l, c, v = g['$open'], g['$high'], g['$low'], g['$close'], g['$volume']
        pc = c.shift(1)
        d = pd.DataFrame(index=g.index)
        d['open'] = o
        d['close'] = c
        d['prev_close'] = pc
        d['open_ret'] = o / pc - 1.0                       # 开盘跳空
        d['amt'] = c * v * 100.0                            # 成交额（元）；volume 单位是「手」
        # T+1 收盘 → T+2 开盘的收益（对应"T+1开盘买 T+2开盘卖"的持有段）
        d['fwd_hold'] = o.shift(-1) / o - 1.0
        # 一字板：四价合一且涨幅达板
        limit = np.where(pc.notna() & (o / pc - 1.0) >= 0.098, 1, 0)
        d['is_yiziban'] = ((o == h) & (h == l) & (l == c)).astype(int) * limit
        # 涨停开盘（含一字）：主板 10% / 双创 20%
        d['is_limit_up'] = ((d['open_ret'] >= 0.098)).astype(int)
        d['is_limit_dn'] = ((d['open_ret'] <= -0.098)).astype(int)
        d['is_suspend'] = o.isna().astype(int)
        frames.append(d.assign(inst=inst).reset_index().set_index(['datetime', 'inst']))
    return pd.concat(frames).sort_index()


def main():
    panels = load()
    P = build(panels)
    dates = sorted(P.index.get_level_values('datetime').unique())
    print(f"池: liquid500  区间 {dates[0].date()} ~ {dates[-1].date()}  {len(dates)} 天")
    print(f"总样本行: {len(P):,}")

    # ---------- A. 每日「买不进」比例 ----------
    print("\n=== A. T+1 开盘买入场景中，买不进的标的占比（按日） ===")
    daily = []
    for dt in dates:
        try:
            s = P.xs(dt, level='datetime')
        except KeyError:
            continue
        s = s.dropna(subset=['open'])
        if len(s) < 50:
            continue
        daily.append({
            'date': dt, 'n': len(s),
            'yiziban': s['is_yiziban'].mean(),
            'limit_up': s['is_limit_up'].mean(),
            'limit_dn': s['is_limit_dn'].mean(),
        })
    D1 = pd.DataFrame(daily).set_index('date')
    print(f"  一字涨停（完全买不进）  日均 {D1['yiziban'].mean():.2%}   最高 {D1['yiziban'].max():.2%}")
    print(f"  涨停开盘（含一字，难买）日均 {D1['limit_up'].mean():.2%}   最高 {D1['limit_up'].max():.2%}")
    print(f"  跌停开盘（卖不掉）      日均 {D1['limit_dn'].mean():.2%}   最高 {D1['limit_dn'].max():.2%}")
    # 组合视角：随机持 30 只，期望有几只买不进
    exp_unbuyable = D1['limit_up'].mean() * TOPK
    print(f"  → 若持仓 {TOPK} 只，期望有 {exp_unbuyable:.1f} 只在开盘买不进"
          f"（占仓位 {D1['limit_up'].mean():.1%}）")
    D1.to_csv(OUT / 'reality_check_daily.csv')

    # ---------- B. 买不进的票贡献了多少虚假收益 ----------
    print("\n=== B. 「买不进」的票事后收益 vs 正常票（虚假收益有多肥） ===")
    allp = P.dropna(subset=['open'])
    rows = []
    for name, mask in [
        ('一字涨停开盘', allp['is_yiziban'] == 1),
        ('涨停开盘(含一字)', allp['is_limit_up'] == 1),
        ('正常可买', allp['is_limit_up'] == 0),
    ]:
        s = allp.loc[mask, 'fwd_hold'].dropna()
        rows.append({'类别': name, '样本数': len(s), '次日期望收益': s.mean(),
                     '中位数': s.median(), '胜率': (s > 0).mean()})
    B = pd.DataFrame(rows)
    for r in rows:
        print(f"  {r['类别']:<16s} n={r['样本数']:>7,}  次日收益均值={r['次日期望收益']*100:+.3f}%  "
              f"中位={r['中位数']*100:+.3f}%  胜率={r['胜率']:.1%}")
    gap = B.loc[B['类别'] == '涨停开盘(含一字)', '次日期望收益'].iloc[0] - \
          B.loc[B['类别'] == '正常可买', '次日期望收益'].iloc[0]
    print(f"  → 涨停开盘票比正常票次日高 {gap*100:+.3f} 个百分点；"
          f"若持仓中 {D1['limit_up'].mean():.1%} 是这类票，"
          f"回测虚增 ≈ {gap * D1['limit_up'].mean() * 252 * 100:+.1f}%/年（单利粗估）")
    B.to_csv(OUT / 'reality_check_bias.csv', index=False)

    # ---------- C. 流动性：按 1e7 账户等权买入的冲击 ----------
    print(f"\n=== C. 流动性（账户 {ACCOUNT:,.0f} 元，等权持 {TOPK} 只 → 单票 {ACCOUNT/TOPK:,.0f} 元） ===")
    per_stock = ACCOUNT / TOPK
    amt = allp['amt'].dropna()
    print(f"  日成交额分布（元）：")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        print(f"    P{int(q*100):>2d}: {amt.quantile(q):>15,.0f}")
    for thr in (5e6, 1e7, 3e7, 1e8):
        print(f"  成交额 < {thr/1e8:.2f}亿 ({thr:,.0f}) 的占比: {(amt < thr).mean():.2%}")
    # 参与率 = 单票买入额 / 日成交额，>5% 即显著冲击
    ratio = per_stock / amt
    print(f"  单票买入占日成交额比例：中位 {ratio.median():.2%}  "
          f"P90 {ratio.quantile(0.9):.2%}  >5% 的占比 {(ratio > 0.05).mean():.2%}")
    cliq = pd.DataFrame({'per_stock_buy': per_stock, 'amt_median': amt.median(),
                         'ratio_median': ratio.median(), 'ratio_p90': ratio.quantile(0.9),
                         'ratio_gt5pct': (ratio > 0.05).mean()}, index=[0])
    cliq.to_csv(OUT / 'reality_check_liquidity.csv', index=False)

    # ---------- D. 建议的可交易过滤后，池等权基准变化 ----------
    print("\n=== D. 加上「可交易过滤」后，池等权基准怎么变（基准也要同口径） ===")
    def eq_bench(frame):
        r = frame['close'].groupby(level='inst').pct_change()
        return r.groupby(level='datetime').mean()
    r_all = eq_bench(allp).dropna()
    buyable = allp[allp['is_limit_up'] == 0]
    r_buy = eq_bench(buyable).dropna()
    def ann(s):
        return (1 + s.mean()) ** 252 - 1
    print(f"  全池等权        年化 {ann(r_all)*100:+.2f}%  （{len(r_all)} 天）")
    print(f"  剔除涨停开盘后  年化 {ann(r_buy)*100:+.2f}%  （{len(r_buy)} 天）")
    print(f"  差 {(ann(r_buy)-ann(r_all))*100:+.2f} 个百分点/年")
    pd.DataFrame({'bench_all': r_all, 'bench_buyable': r_buy}).to_csv(OUT / 'reality_check_bench.csv')

    print("\n结论要点：回测若不过滤涨停开盘，等于每天在买不到的票上记账收益。")


if __name__ == '__main__':
    main()
