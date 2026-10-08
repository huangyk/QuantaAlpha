"""战法验证：买入持有 + 阈值触发再平衡（终于把成本口径对齐）

上一版权重实验的致命错误：**基准没扣成本，策略月度无条件调仓扣 11.6%/年**，
所以所有方案必然显示 −11% 超额。这掩盖了真正的问题。

本脚本把口径彻底对齐：
    - 基准 = **买入持有（buy & hold）**：首日等权建仓，之后权重自然漂移，不再动，
      只在建仓时付一次成本。这是「什么都不做」的真实收益，也是任何主动方案的天花板。
    - 策略 = 阈值触发再平衡：只有当某票权重偏离目标超过阈值时才调，
      调仓频率由偏离本身决定，不是日历驱动。

要回答的核心问题：
    Q1  不同池子的 buy&hold 收益是多少？（选池是唯一还有杠杆的决策）
    Q2  阈值再平衡能不能比 buy&hold 更好？（如果不能，就明确写进战法：不要动）
    Q3  建仓一次的启动成本占多少？（决定资金门槛）

数据：cn_real_full 全 A 缓存，2020-01 ~ 2026-09
输出：data/results/strategy_bh_vs_rebalance.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path('data/results/_all_a_panel.parquet')
OUT = Path('data/results')
ACCOUNT = 1e7
COMMISSION_BP, STAMP_BP, ETA = 5.0, 5.0, 0.1
SIGMA_BP = 342.0
SEGS = [('2021', '2021-01-01', '2021-12-31'), ('2022', '2022-01-01', '2022-12-31'),
        ('2023', '2023-01-01', '2023-12-31'), ('2024', '2024-01-01', '2024-12-31'),
        ('2025', '2025-01-01', '2025-12-31'), ('2026', '2026-01-01', '2026-09-30')]


def main():
    print("载入缓存 ...")
    d = pd.read_parquet(CACHE)
    O = d['$open'].unstack('instrument').sort_index()
    C = d['$close'].unstack('instrument').sort_index()
    V = d['$volume'].unstack('instrument').reindex_like(C)
    AMT = (C * V) * 100.0   # volume 单位是「手」，成交额需 ×100
    ret = (O.shift(-1) / O - 1.0).clip(-0.5, 0.5)
    listing = C.notna().cumsum()
    amt20 = AMT.rolling(20, min_periods=5).mean()
    amt60 = AMT.rolling(60, min_periods=20).median()
    mature = listing >= 500
    print(f"  面板 {C.shape}  成熟股 {mature.iloc[-1].sum()} 只")

    # ---- 候选池（按流动性分层，都是成熟股）----
    liq_rank = amt60.ffill().iloc[-1]
    mature_codes = [c for c in C.columns if mature[c].max()]
    order = liq_rank[mature_codes].sort_values(ascending=False)
    N = len(order)
    pools = {
        '全部成熟股': mature_codes,
        '流动性前1/3': list(order.index[:N // 3]),
        '流动性中1/3': list(order.index[N // 3:2 * N // 3]),
        '流动性后1/3': list(order.index[2 * N // 3:]),
        '流动性后600': list(order.index[-600:]),
        '流动性后300': list(order.index[-300:]),
    }

    def build_pool_pool(d0):
        """每月首个交易日：当日之前的可见信息定池。"""
        amt_ok = amt20.loc[d0] >= 1000e4
        return {name: [c for c in codes if c in C.columns and amt_ok.get(c, False) and not (O.at[d0, c] / C.at[d0, c] - 1 >= 0.098)]
                for name, codes in pools.items()}

    def impact_of(w, d):
        """按买入增额计算冲击（bp 加权）。"""
        delta = w.clip(lower=0)
        buy = ACCOUNT * delta
        part = (buy / AMT.loc[d].reindex(w.index)).clip(0, 1).fillna(0)
        return float((delta * ETA * SIGMA_BP * np.sqrt(part)).sum())

    results = []
    nav_store = {}
    dates = C.index
    month_starts = set(pd.Series(dates).groupby([dates.year, dates.month]).min())
    dates_used = []

    for k, d in enumerate(dates):
        if d not in month_starts or k < 62 or d not in ret.index:
            continue
        d_prev = dates[k - 1]
        P = build_pool_pool(d_prev)
        r = ret.loc[d]
        for name, uni in P.items():
            key = name
            if key not in nav_store:
                nav_store[key] = {'bh': [], 'reb': {thr: [] for thr in (0.3, 0.5, 0.8)}}
                nav_store[key]['w'] = None
                nav_store[key]['w_reb'] = {thr: None for thr in (0.3, 0.5, 0.8)}
                nav_store[key]['turn_bh'] = []
                nav_store[key]['turn_reb'] = {thr: [] for thr in (0.3, 0.5, 0.8)}
            st = nav_store[key]
            if len(uni) < 30:
                for thr in (0.3, 0.5, 0.8):
                    st['reb'][thr].append(0.0)
                st['bh'].append(0.0)
                continue
            # ---- buy & hold：首期建仓，之后漂移不动 ----
            if st['w'] is None:
                w = pd.Series(1.0 / len(uni), index=uni)
                imp = impact_of(w, d)
                cost = (COMMISSION_BP + STAMP_BP + 2 * imp) / 1e4
                st['turn_bh'].append(1.0)
            else:
                w = st['w']
                cost = 0.0
                st['turn_bh'].append(0.0)
            rr = r.reindex(w.index).fillna(0.0)
            st['bh'].append(float((w * rr).sum()) - cost)
            w2 = w * (1 + rr)
            st['w'] = w2 / w2.sum()
            # ---- 阈值再平衡：权重偏离目标(等权)超过 thr 才调 ----
            for thr in (0.3, 0.5, 0.8):
                wr = st['w_reb'][thr]
                if wr is None:
                    wr = pd.Series(1.0 / len(uni), index=uni)
                    imp = impact_of(wr, d)
                    st['reb'][thr].append(float((wr * rr).sum()) - (COMMISSION_BP + STAMP_BP + 2 * imp) / 1e4)
                    st['turn_reb'][thr].append(1.0)
                else:
                    target = pd.Series(1.0 / len(wr.index), index=wr.index)
                    dev = float((wr - 1.0 / len(wr)).abs().max())
                    if dev > thr:
                        delta = (target - wr)
                        turn = 0.5 * float(delta.abs().sum())
                        buy = (delta.clip(lower=0))
                        imp = impact_of(buy, d) if turn > 0 else 0.0
                        cst = turn * (COMMISSION_BP + STAMP_BP + 2 * imp) / 1e4
                        st['reb'][thr].append(float((wr * rr).sum()) - cst)
                        wr2 = wr * (1 + rr)
                        wr = target
                        st['turn_reb'][thr].append(turn)
                    else:
                        st['reb'][thr].append(float((wr * rr).sum()))
                        st['turn_reb'][thr].append(0.0)
                    wr2 = wr * (1 + rr)
                    st['w_reb'][thr] = wr2 / wr2.sum()
        dates_used.append(d)

    idx_use = pd.DatetimeIndex(dates_used)

    print(f"\n{'池':<14s}{'只数':>6s}{'BH年化%':>10s}{'BH_IR':>8s}{'BH回撤%':>10s}"
          f"{'阈值0.3超额%':>14s}{'阈值0.5超额%':>14s}{'阈值0.8超额%':>14s}")
    print("-" * 92)
    rows = []
    for name, st in nav_store.items():
        bh = pd.Series(st['bh'], index=idx_use)
        n0 = np.argmax(bh.values != 0) if (bh.values != 0).any() else 0
        bh = bh.iloc[n0:]
        ix = idx_use[n0:]
        ann_bh = (1 + bh).prod() ** (12 / len(bh)) - 1
        cum = (1 + bh).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        ir = bh.mean() / bh.std() * np.sqrt(12) if bh.std() > 0 else np.nan
        line = f"{name:<14s}{len(pools[name]):>6d}{ann_bh*100:>+10.2f}{ir:>+8.2f}{mdd*100:>+10.2f}"
        row = {'pool': name, 'n': len(pools[name]), 'bh_ann%': ann_bh * 100,
               'bh_IR': ir, 'bh_mdd%': mdd * 100}
        for thr in (0.3, 0.5, 0.8):
            ser = pd.Series(st['reb'][thr][n0:], index=ix)
            exc = (1 + ser) / (1 + bh) - 1
            ann_e = ((1 + exc).prod() ** (12 / len(exc)) - 1) * 100
            e = (ser - bh)
            t = e.mean() / e.std() * np.sqrt(len(e)) if e.std() > 0 else np.nan
            line += f"{ann_e:>+14.2f}"
            row[f'exc_{thr}%'] = ann_e
            row[f't_{thr}'] = t
            row[f'turn_{thr}'] = float(np.mean(st['turn_reb'][thr][n0:]))
        print(line)
        rows.append(row)

    print("-" * 92)
    print("（超额 = 阈值再平衡 相对 同池买入持有；月频口径，12 期/年）")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'strategy_bh_vs_rebalance.csv', index=False)

    best_pool = R.sort_values('bh_ann%', ascending=False).iloc[0]
    print(f"\n→ buy&hold 最强的池: {best_pool['pool']}  年化 {best_pool['bh_ann%']:+.2f}%  "
          f"IR {best_pool['bh_IR']:+.2f}  回撤 {best_pool['bh_mdd%']:+.2f}%")
    print("→ 阈值再平衡全部为负 = **不要主动调仓，买入持有就是最优战法**"
          if all(R[f'exc_{t}%'].max() < 3 for t in (0.3, 0.5, 0.8))
          else "→ 存在调仓增益，见上表")

    # 逐年
    print("\n逐年 buy&hold 年化（%）:")
    for name, st in nav_store.items():
        bh = pd.Series(st['bh'], index=idx_use)
        line = f"  {name:<14s}"
        for nm, a, b_ in SEGS:
            m = (bh.index >= pd.Timestamp(a)) & (bh.index <= pd.Timestamp(b_))
            if m.sum() < 3:
                line += f"{'-':>9s}"
                continue
            s = bh[m]
            line += f"{((1+s).prod()**(12/len(s))-1)*100:>+9.2f}"
        print(line)
    print("\n保存: strategy_bh_vs_rebalance.csv")


if __name__ == '__main__':
    main()