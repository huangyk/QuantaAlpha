"""战法定稿：统一口径重算各池 buy&hold，给出可直接执行的规则与真实指标。

前面所有实验的口径都不统一（开盘价 vs 收盘价、日频 vs 月频、扣不扣成本），
导致数字互相矛盾。本脚本统一为**最保守也最标准的口径**：

    收益   = close[T+1]/close[T] - 1        （收盘→次日收盘，日频）
    建仓   = 每月首个交易日等权买入，只建仓一次；之后**永不卖、永不调仓**
    成本   = 建仓/换股时付 10bp（佣金5+印花税5）+ 真实 participation 冲击
    基准   = 同池 buy&hold（不扣成本，因为战法的成本只在建仓时发生）

池子（全部要求：上市≥500 日，即已剔除次新与妖股）：
    liquid500          现有池
    全A成熟            全部成熟股
    流动性分层         前/中/后 1/3
    分层+规模代理      用成交额分层的 6 档
    容量上限           日均成交额 ≥3000万 / ≥1000万 / 全池（用于确定能上多少资金）

输出：data/results/strategy_final.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path('data/results/_all_a_panel.parquet')
OUT = Path('data/results')
ACCOUNT = 1e7
COMMISSION_BP, STAMP_BP, ETA, SIGMA_BP = 5.0, 5.0, 0.1, 342.0
SEGS = [('2021', '2021-01-01', '2021-12-31'), ('2022', '2022-01-01', '2022-12-31'),
        ('2023', '2023-01-01', '2023-12-31'), ('2024', '2024-01-01', '2024-12-31'),
        ('2025', '2025-01-01', '2025-12-31'), ('2026', '2026-01-01', '2026-09-30')]


def main():
    d = pd.read_parquet(CACHE)
    C = d['$close'].unstack('instrument').sort_index()
    V = d['$volume'].unstack('instrument').reindex_like(C)
    O = d['$open'].unstack('instrument').sort_index()
    AMT = C * V
    ret = (C.shift(-1) / C - 1.0).clip(-0.5, 0.5)
    listing = C.notna().cumsum()
    mature = listing >= 500
    amt60 = AMT.rolling(60, min_periods=20).median()
    print(f"  面板 {C.shape}  成熟股 {int(mature.iloc[-1].sum())} 只")

    liq = amt60.ffill().iloc[-1]
    mcodes = [c for c in C.columns if mature[c].max()]
    order = liq[mcodes].sort_values(ascending=False)
    N = len(order)

    p0 = Path('D:/qqmt/qlib_data/cn_real_full/instruments/liquid500.txt')
    liq500 = [l.split('\t')[0].strip() for l in open(p0) if l.strip()] if p0.exists() else []

    pools = {
        # ⚠️ liquid500 也必须做成熟度过滤，否则建仓日可买的 362 只里混着次新股
        #    （实测次新股 5 日收益 >±50% 的有 800 个样本，会污染 buy&hold 收益）
        'liquid500(现有池)': [c for c in liq500 if c in C.columns and mature[c].max()],
        'liquid500未过滤': [c for c in liq500 if c in C.columns],
        '全A成熟': mcodes,
        '流动性前1/3': list(order.index[:N // 3]),
        '流动性中1/3': list(order.index[N // 3:2 * N // 3]),
        '流动性后1/3': list(order.index[2 * N // 3:]),
        '流动性后600': list(order.index[-600:]),
    }

    dates = C.index
    month_starts = set(pd.Series(dates).groupby([dates.year, dates.month]).min())

    def first_day_universe(codes, min_amt, tag=''):
        """首次建仓日（2020-01 之后第一个满足条件的交易日）的可交易池。"""
        dbg = []
        for d in dates[63:]:
            if d not in month_starts:
                continue
            a = AMT.loc[d].reindex(codes)
            ok = [c for c in codes if a.get(c, 0) >= min_amt and not (O.at[d, c] / C.at[d, c] - 1 >= 0.098)]
            # ⚠️ 数据完整性检查：某些交易日（如 2020-05-05）全市场只有 ~120 只更新，
            #    拿去建仓等于"在残缺横截面里等权"，会严重扭曲结果 → 必须跳过
            if len(ok) < 0.5 * len(codes):
                dbg.append((d.date(), len(ok), 'skip'))
                continue
            dbg.append((d.date(), len(ok)))
            if len(ok) >= 50:
                print(f"    [建仓日 {d.date()}] {tag} 可买 {len(ok)} 只"
                      f"（候选 {len(codes)}，跳过 {sum(1 for x in dbg if len(x) > 2)} 个残缺日）")
                return d, ok
        return None, None

    print(f"\n{'池':<20s}{'只数':>6s}{'建仓成本bp':>11s}{'年化%':>9s}{'IR':>7s}"
          f"{'回撤%':>9s}{'2025%':>9s}{'2026%':>9s}")
    print("-" * 80)
    rows = []
    for name, codes in pools.items():
        for min_amt, tag in [(0, ''), (1000e4, '≥1000万'), (3000e4, '≥3000万')]:
            d0, uni = first_day_universe(codes, min_amt, name)
            if uni is None:
                continue
            label = name + (f' [{tag}]' if tag else '')
            w = pd.Series(1.0 / len(uni), index=uni)
            # 建仓成本（真实冲击）
            buy = ACCOUNT * w
            part = (buy / AMT.loc[d0].reindex(w.index)).clip(0, 1).fillna(0)
            imp = float((w * ETA * SIGMA_BP * np.sqrt(part)).sum())
            entry_cost = (COMMISSION_BP + STAMP_BP + 2 * imp)
            nav = []
            for d in dates[dates >= d0]:
                if d not in ret.index:
                    nav.append(0.0)
                    continue
                rr = ret.loc[d].reindex(w.index).fillna(0.0)
                nav.append(float((w * rr).sum()))
            ser = pd.Series(nav, index=dates[dates >= d0])
            ann = (1 + ser).prod() ** (252 / len(ser)) - 1
            cum = (1 + ser).cumprod()
            mdd = (cum / cum.cummax() - 1).min()
            ir = ser.mean() / ser.std() * np.sqrt(252) if ser.std() > 0 else np.nan
            per = {}
            for nm, a, b_ in SEGS:
                m = (ser.index >= pd.Timestamp(a)) & (ser.index <= pd.Timestamp(b_))
                per[nm] = ((1 + ser[m]).prod() ** (252 / max(m.sum(), 1)) - 1) * 100 if m.sum() > 5 else np.nan
            net_ann = (1 + ser).prod() ** (252 / len(ser)) * (1 - entry_cost / 1e4) - 1
            print(f"{label:<20s}{len(uni):>6d}{entry_cost:>11.1f}{ann*100:>+9.2f}{ir:>+7.2f}"
                  f"{mdd*100:>+9.2f}{per.get('2025', np.nan):>+9.2f}{per.get('2026', np.nan):>+9.2f}")
            rows.append({'pool': label, 'n': len(uni), 'entry_cost_bp': entry_cost,
                         'ann%': ann * 100, 'net_ann%': net_ann * 100, 'IR': ir, 'mdd%': mdd * 100,
                         **{f'y{nm}': v for nm, v in per.items()}})
    print("-" * 80)
    print("（年化为日频几何年化；净年化 = 再扣一次性建仓成本）")

    R = pd.DataFrame(rows).sort_values('net_ann%', ascending=False)
    print("\n按「净年化」排序（= 已扣建仓成本，这才是能拿到的）：")
    for _, r in R.iterrows():
        print(f"  {r['pool']:<22s}{r['n']:>6d} 只  净年化 {r['net_ann%']:>+7.2f}%  "
              f"IR {r['IR']:>+5.2f}  回撤 {r['mdd%']:>+7.2f}%  建仓成本 {r['entry_cost_bp']:.1f}bp")
    R.to_csv(OUT / 'strategy_final.csv', index=False)
    print("\n保存: strategy_final.csv")


if __name__ == '__main__':
    main()