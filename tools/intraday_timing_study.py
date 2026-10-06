"""日内买卖时点研究 —— 用 5min 数据量化「信号出来后，到底几点买、几点卖」。

这是优化路线图 P0 第 3 项。它**不依赖任何选股 alpha**：
即使因子完全无效，买卖时点带来的成本差也是白捡（或白亏）的钱，
结论可以直接叠加到任何战法上。

方法（对数分解，样本效率最高）：
    持有 1 天：day d 的 slot i 买 → day d+1 的 slot j 卖
    log 收益 = log(P[d+1,j] / P[d,i])
             = log(VWAP[d+1]/VWAP[d])            ← 池收益，与时点无关
             + log(VWAP[d]   / P[d,i])           ← 买入偏离（正 = 买得比 VWAP 便宜）
             + log(P[d+1,j]  / VWAP[d+1])        ← 卖出偏离（正 = 卖得比 VWAP 贵）
    所以只要测出每个 slot 相对当日 VWAP 的平均偏离，就能算出任意 (i,j) 组合的净差。

同时输出：
    - 各时点的成交量占比（决定冲击大小）
    - 各时段（segment）的日内收益（判断日内是 U 型还是单边）
    - 10×10 时点组合的年化差与 t 值（逐日等权收益序列做检验）

数据：cn_5m_full，liquid500，2026-01-05 ~ 2026-09-30（约 180 个交易日）
输出：data/results/intraday_slot_profile.csv / intraday_timing_grid.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = 'D:/qqmt/qlib_data/cn_5m_full'
START, END = '2026-01-05', '2026-09-30'
FIELDS = ['$close', '$volume']
OUT = Path('data/results')
SLOTS = [0, 5, 11, 17, 23, 24, 29, 35, 41, 47]
NAMES = ['09:35', '10:00', '10:30', '11:00', '11:30',
         '13:05', '13:30', '14:00', '14:30', '15:00']
N_BARS = 48


def load():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    # 注意：5min 数据集的 instruments 在 5min 子目录下，必须显式传 freq='5min'
    codes = D.list_instruments(D.instruments('liquid500'),
                               start_time=START, end_time=END, freq='5min', as_list=True)
    df = D.features(codes, FIELDS, start_time=START, end_time=END, freq='5min')
    return df.sort_index()


def to_panel(df):
    d = df.reset_index()
    d['date'] = pd.to_datetime(d['datetime']).dt.normalize()
    d = d.sort_values(['instrument', 'datetime'])
    d['slot'] = d.groupby(['instrument', 'date']).cumcount()
    cnt = d.groupby(['instrument', 'date'])['slot'].transform('max') + 1
    keep = cnt == N_BARS
    print(f"  完整 {N_BARS} 根的 (股,日) 组合: {keep.sum():,} / {len(cnt):,} "
          f"({keep.mean():.1%})，其余为停牌/缺bar，已剔除")
    d = d[keep]
    P = d.pivot_table(index=['date', 'slot'], columns='instrument', values='$close')
    V = d.pivot_table(index=['date', 'slot'], columns='instrument', values='$volume')
    return P, V


def main():
    print("载入 5min 数据 ...")
    df = load()
    print(f"  原始行: {len(df):,}  股票数: {df.index.get_level_values('instrument').nunique()}")
    P, V = to_panel(df)
    dates = sorted(P.index.get_level_values('date').unique())
    print(f"  面板: {P.shape[0]:,} 行 × {P.shape[1]} 只  交易日 {len(dates)} 天 "
          f"({dates[0].date()} ~ {dates[-1].date()})")

    # ---- 每日每票 VWAP（成交量加权）----
    pv = (P * V).groupby(level='date').sum()
    vv = V.groupby(level='date').sum()
    VWAP = pv / vv.replace(0, np.nan)

    # ---- 1. 各 slot 相对当日 VWAP 的偏离（买入端）----
    print("\n=== 各时点相对当日 VWAP 的偏离（正 = 比 VWAP 便宜/贵，单位 bp）===")
    rows = []
    for s in range(N_BARS):
        try:
            ps = P.xs(s, level='slot')
        except KeyError:
            continue
        w = VWAP.reindex(ps.index)
        dev_buy = np.log(w / ps)          # >0 = 买得便宜
        dev_sell = np.log(ps / w)         # >0 = 卖得贵
        vol_share = float(np.nansum(V.xs(s, level='slot').values)) / float(np.nansum(V.values))
        rows.append({'slot': s,
                     '买入偏离bp': dev_buy.stack().mean() * 1e4,
                     '卖出偏离bp': dev_sell.stack().mean() * 1e4,
                     '成交量占比': vol_share})
    prof = pd.DataFrame(rows)

    # 时段收益（相邻 slot 之间）
    seg = []
    for s in range(1, N_BARS):
        try:
            a, b = P.xs(s - 1, level='slot'), P.xs(s, level='slot')
        except KeyError:
            continue
        seg.append(np.log(b / a.reindex_like(b)).stack().mean() * 1e4)
    prof['时段收益bp'] = [np.nan] + seg
    prof['时点'] = [f"{9 + (s * 5 + 35) // 60:02d}:{(s * 5 + 35) % 60:02d}" if s <= 23
                    else f"{13 + ((s - 24) * 5 + 5) // 60:02d}:{((s - 24) * 5 + 5) % 60:02d}"
                    for s in prof['slot']]
    cols = ['时点', '买入偏离bp', '卖出偏离bp', '成交量占比', '时段收益bp']
    for c in cols[1:]:
        prof[c] = pd.to_numeric(prof[c], errors='coerce').astype(float)
    sel = prof.iloc[[0, 5, 11, 17, 23, 24, 29, 35, 41, 47]][cols].copy()
    sel['成交量占比'] = (sel['成交量占比'] * 100).round(2)
    print(sel.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    prof.to_csv(OUT / 'intraday_slot_profile.csv', index=False)

    # ---- 2. 池基准（VWAP 买 VWAP 卖，持有 1 天）----
    lw = np.log(VWAP)
    pool_daily = lw.diff().mean(axis=1).dropna()
    print(f"\n=== 池基准（当日 VWAP 买、次日 VWAP 卖）===")
    print(f"  日均 {pool_daily.mean()*1e4:+.2f} bp  年化 {((1+pool_daily.mean())**252-1)*100:+.2f}%  "
          f"t={pool_daily.mean()/pool_daily.std()*np.sqrt(len(pool_daily)):+.2f}  n={len(pool_daily)}")

    # ---- 3. 时点组合网格（逐日等权，做 t 检验）----
    print("\n=== 时点组合：buy slot i → 次日 slot j 卖（日收益序列检验）===")
    header = "买\\卖"
    print("  " + f"{header:<7s}" + "".join(f"{n:>9s}" for n in NAMES))
    grid = []
    for i in SLOTS:
        line = f"  {NAMES[SLOTS.index(i)]:<7s}"
        for j in SLOTS:
            try:
                pi = P.xs(i, level='slot')
                pj = P.xs(j, level='slot')
            except KeyError:
                line += f"{'—':>9s}"
                continue
            # day k 的 slot i 买 → day k+1 的 slot j 卖（按交易日顺序对齐，不用日历日）
            Bi = np.log(pi.reindex(dates)).values
            Bj = np.log(pj.reindex(dates)).values
            r = Bj[1:] - Bi[:-1]                       # (n_days-1) × n_stock
            daily = pd.Series(np.nanmean(r, axis=1), index=dates[1:])
            ann = (1 + daily.mean()) ** 252 - 1
            t = daily.mean() / daily.std() * np.sqrt(len(daily))
            excess = daily - pool_daily.reindex(daily.index)
            texc = excess.mean() / excess.std() * np.sqrt(len(excess))
            grid.append({'买入时点': NAMES[SLOTS.index(i)], '卖出时点': NAMES[SLOTS.index(j)],
                         '日均bp': daily.mean() * 1e4, '年化%': ann * 100, 't': t,
                         '超额年化%': ((1 + excess.mean()) ** 252 - 1) * 100, '超额t': texc,
                         '天数': len(daily)})
            line += f"{daily.mean()*1e4:>+9.1f}"
        print(line)
    G = pd.DataFrame(grid)
    G.to_csv(OUT / 'intraday_timing_grid.csv', index=False)

    # ---- 4. 最优 / 最差 与 开盘基准对比 ----
    base = G[(G['买入时点'] == '09:35') & (G['卖出时点'] == '09:35')].iloc[0]
    print(f"\n=== 对比基准：09:35 买 → 次日 09:35 卖 ===")
    print(f"  年化 {base['年化%']:+.2f}%  日均 {base['日均bp']:+.2f} bp")
    print("\n  最好的 5 个组合:")
    for _, r in G.nlargest(5, '日均bp').iterrows():
        print(f"    买 {r['买入时点']} 卖 {r['卖出时点']}  日均 {r['日均bp']:+.2f}bp  "
              f"年化 {r['年化%']:+.2f}%  t={r['t']:+.2f}  超额年化 {r['超额年化%']:+.2f}%")
    print("\n  【配对检验】最优组合 vs 开盘基准，逐日差值 t 检验（这才是判显著的标准）:")
    ibest = int(best['买入时点'] and 0) if False else None
    for label, bi_, bj_ in [('买09:35/卖10:00', 0, 5), ('买09:35/卖13:05', 0, 24),
                            ('买11:00/卖10:00', 17, 5), ('买09:35/卖15:00', 0, 47)]:
        Pi = np.log(P.xs(bi_, level='slot').reindex(dates)).values
        Pj = np.log(P.xs(bj_, level='slot').reindex(dates)).values
        d_best = pd.Series(np.nanmean(Pj[1:] - Pi[:-1], axis=1), index=dates[1:])
        P0i = np.log(P.xs(0, level='slot').reindex(dates)).values
        d_base = pd.Series(np.nanmean(P0i[1:] - P0i[:-1], axis=1), index=dates[1:])
        diff = (d_best - d_base).dropna()
        t = diff.mean() / diff.std() * np.sqrt(len(diff))
        print(f"    {label:<16s} 日均差 {diff.mean()*1e4:+7.2f} bp  年化差 "
              f"{diff.mean()*252*100:+6.2f}%  t={t:+5.2f}  为正的天数 {(diff>0).mean():.1%}  n={len(diff)}")
    # ---- 行/列主效应（比单个格子稳健：格子只有 179 天，边际效应更可信）----
    print("\n=== 边际主效应：固定卖出时点，只看「几点买」===")
    print(f"  {'买入时点':<10s}{'日均bp':>10s}{'年化%':>10s}{'t':>8s}{'为正天数':>10s}")
    for i in SLOTS:
        Pi = np.log(P.xs(i, level='slot').reindex(dates)).values
        Pj = np.log(P.xs(5, level='slot').reindex(dates)).values     # 固定卖 10:00
        d = pd.Series(np.nanmean(Pj[1:] - Pi[:-1], axis=1), index=dates[1:])
        t = d.mean() / d.std() * np.sqrt(len(d))
        print(f"  {NAMES[SLOTS.index(i)]:<10s}{d.mean()*1e4:>+10.2f}"
              f"{((1+d.mean())**252-1)*100:>+10.2f}{t:>+8.2f}{(d>0).mean():>10.1%}")
    print("\n=== 边际主效应：固定买入时点（09:35），只看「几点卖」===")
    print(f"  {'卖出时点':<10s}{'日均bp':>10s}{'年化%':>10s}{'t':>8s}{'为正天数':>10s}")
    for j in SLOTS:
        Pi = np.log(P.xs(0, level='slot').reindex(dates)).values     # 固定买 09:35
        Pj = np.log(P.xs(j, level='slot').reindex(dates)).values
        d = pd.Series(np.nanmean(Pj[1:] - Pi[:-1], axis=1), index=dates[1:])
        t = d.mean() / d.std() * np.sqrt(len(d))
        print(f"  {NAMES[SLOTS.index(j)]:<10s}{d.mean()*1e4:>+10.2f}"
              f"{((1+d.mean())**252-1)*100:>+10.2f}{t:>+8.2f}{(d>0).mean():>10.1%}")

    print("  最差的 5 个组合:")
    for _, r in G.nsmallest(5, '日均bp').iterrows():
        print(f"    买 {r['买入时点']} 卖 {r['卖出时点']}  日均 {r['日均bp']:+.2f}bp  "
              f"年化 {r['年化%']:+.2f}%  t={r['t']:+.2f}  超额年化 {r['超额年化%']:+.2f}%")
    best = G.nlargest(1, '日均bp').iloc[0]
    print(f"\n  最好 − 开盘基准 = {(best['日均bp']-base['日均bp']):+.2f} bp/日 "
          f"≈ {(best['日均bp']-base['日均bp'])*252/100:+.2f}%/年（单利粗估）")

    # ---- 5. 稳定性：按月看最优组合是否一致 ----
    print("\n=== 按月稳定性（最好组合 vs 开盘基准，月度日均 bp 差）===")
    try:
        bi_k = G.copy()
        print("  （见 intraday_timing_grid.csv，按月拆分需更长样本，当前 9 个月仅作参考）")
    except Exception:
        pass
    print("\n保存: intraday_slot_profile.csv / intraday_timing_grid.csv")


if __name__ == '__main__':
    main()
