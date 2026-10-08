"""风格可预测性检验：2021 年后的失效是「风格切换」，那风格能预测吗？

背景（上一轮诊断）：
    2013-2020 流动性 Top500 池跑赢所有宽基（+6.43% / +4.47%）
    2021-2026 池子 −1.35%，同期沪深300 +0.20%、中证1000 −3.55%
    「300−1000」风格差从 −0.40/−0.90 变成 **+3.75**（历史最大）→ 大盘价值占优

关键问题：**风格是随机游走，还是可预测的？**
    - 若可预测 → 风格择时就是新 alpha，能修复战法
    - 若随机 → 风格切换是结构性变化，战法需要换池而非择时

检验设计（严格遵守铁律）：
    信号   style = 沪深300 / 中证1000 的比值（上升 = 大盘占优）
    判据   比值的 N 日动量 > 0 → 持 300；否则持 1000
    参数   N ∈ {20, 60, 120, 250, 0(=基准)}，**只在 train 段选**
    验证   test 段单次，不挑最优点
    基准   固定持有 300 / 固定持有 1000 / 50-50 / 池等权 Top500

额外诊断：比值本身的自相关（若近似 0 → 随机游走，择时无效）。

数据：panel_index.parquet（指数）+ panel_2011.parquet（池）
输出：data/results/style_timing.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STOCKS = Path('data/cache/panel_2011.parquet')
INDEX = Path('data/cache/panel_index.parquet')
OUT = Path('data/results')
ACCOUNT = 1e7
COMMISSION_BP, STAMP_BP, ETA, SIGMA_BP = 5.0, 5.0, 0.1, 342.0
MIN_LIST_DAYS = 500
TRAIN_END, VALID_END = '2019-12-31', '2022-12-31'
NS = [20, 60, 120, 250]


def load():
    P = pd.read_parquet(STOCKS)
    if not isinstance(P.columns, pd.MultiIndex):
        P.columns = pd.MultiIndex.from_tuples(
            [tuple(c) if isinstance(c, tuple) else (c, c) for c in P.columns])
    P.index = pd.DatetimeIndex(P.index.get_level_values(0))
    I = pd.read_parquet(INDEX)
    if not isinstance(I.columns, pd.MultiIndex):
        I.columns = pd.MultiIndex.from_tuples(
            [tuple(c) if isinstance(c, tuple) else (c, c) for c in I.columns])
    I.index = pd.DatetimeIndex(I.index.get_level_values(0))
    return P, I


def main():
    P, I = load()
    C, O, AMT = P['close'], P['open'], P['amount']
    idx_c = I['close']
    px300, px1000 = idx_c['000300.SH'], idx_c['000852.SH']
    style = (px300 / px1000).dropna()
    dates = style.index
    # 只保留指数与个股数据都完整的交易日
    common = dates[(dates >= P.index[0]) & (dates <= P.index[-1])]
    style = style.reindex(common).dropna()

    print("=== 1. 风格比值（沪深300 / 中证1000）的自相关 ===")
    r = style.pct_change().dropna()
    print(f"  日收益均值 {r.mean():+.4f}  年化波动 {r.std()*np.sqrt(252):.2%}")
    for lag in (1, 5, 10, 20, 60, 120):
        ac = r.autocorr(lag)
        print(f"  自相关 lag={lag:>3d}: {ac:+.4f}"
              + ("   ← 有可预测性" if abs(ac) > 0.05 else ""))
    print("  → 自相关接近 0 = 风格近似随机游走，择时难有稳定 alpha")

    # ---- 池等权 Top500（作为对照基准）----
    listing = C.notna().cumsum()
    ret = (C.shift(-1) / C - 1.0).clip(-0.5, 0.5)
    ok_cnt = C.notna().sum(axis=1)
    ref = ok_cnt.rolling(250, min_periods=20).median()
    good = set(ok_cnt[(ok_cnt > 0.6 * ref) & (ok_cnt > 500)].index)
    all_d = P.index
    mstart = [d for d in all_d.to_series().groupby([all_d.year, all_d.month]).min() if d in good]
    pool_nav = []
    for k, d0 in enumerate(mstart):
        if d0 not in style.index:
            pool_nav.append(0.0)
            continue
        d_prev = mstart[k - 1] if k > 0 else d0
        liq = AMT.loc[:d_prev].tail(60).median().dropna()
        mature = listing.loc[:d_prev].iloc[-1] >= MIN_LIST_DAYS
        limit_up = (O.loc[d0] / C.loc[d0].shift(1) - 1.0) >= 0.098
        sel = [c for c in liq.index[:500]
               if mature.get(c, False) and pd.notna(C.at[d0, c]) and C.at[d0, c] > 0
               and not limit_up.get(c, False)]
        if len(sel) < 30:
            pool_nav.append(0.0)
            continue
        w = pd.Series(1.0 / len(sel), index=sel)
        if k == 0:
            part = (ACCOUNT * w / AMT.loc[d0].reindex(w.index)).clip(0, 1).fillna(0)
            imp = float((w * ETA * SIGMA_BP * np.sqrt(part)).sum())
            cost = (COMMISSION_BP + STAMP_BP + 2 * imp) / 1e4
        else:
            cost = 0.0
        pool_nav.append(float((w * ret.loc[d0].reindex(w.index).fillna(0.0)).sum()) - cost)

    # ---- 策略：按风格动量在 300 / 1000 / 池 之间切换（月频调仓）----
    COST_SW = (COMMISSION_BP + STAMP_BP) / 1e4        # 一次切换（指数无冲击）

    def next_ret(px, d0):
        """d0 → style 序列中的下一个交易日的指数收益。"""
        if d0 not in style.index:
            return 0.0
        loc = style.index.get_loc(d0)
        if loc + 1 >= len(style.index):
            return 0.0
        nxt = style.index[loc + 1]
        if d0 not in px.index or nxt not in px.index:
            return 0.0
        return float(px.loc[nxt] / px.loc[d0] - 1)

    def strat(n):
        """n=0 表示基准（固定 50/50）；n>0 用风格动量择时。"""
        rows = []
        prev_w = None
        for d0 in mstart:
            if d0 not in style.index:
                continue
            if n == 0:
                wts = {'300': 0.5, '1000': 0.5}
            else:
                hist = style.loc[:d0].tail(n)
                long_big = len(hist) >= n and hist.iloc[-1] > hist.iloc[0]
                wts = {'300': 1.0, '1000': 0.0} if long_big else {'300': 0.0, '1000': 1.0}
            turn = 0.0 if prev_w is None else 0.5 * sum(
                abs(wts.get(k, 0) - prev_w.get(k, 0)) for k in set(wts) | set(prev_w))
            cost = turn * 2 * COST_SW
            r300 = next_ret(px300, d0)
            r1000 = next_ret(px1000, d0)
            if r300 == 0.0 and r1000 == 0.0:
                continue
            gross = wts.get('300', 0) * r300 + wts.get('1000', 0) * r1000
            rows.append({'date': d0, 'net': gross - cost, 'turn': turn})
            prev_w = wts
        return pd.DataFrame(rows).set_index('date') if rows else None

    pool = pd.Series(pool_nav, index=pd.DatetimeIndex(mstart))
    fixed300 = pd.Series([next_ret(px300, d0) for d0 in mstart],
                         index=pd.DatetimeIndex(mstart))
    fixed1000 = pd.Series([next_ret(px1000, d0) for d0 in mstart],
                          index=pd.DatetimeIndex(mstart))

    def ann(s):
        s = s[s != 0]
        return np.nan if len(s) < 3 else ((1 + s).prod() ** (12 / len(s)) - 1) * 100

    TV_END = pd.Timestamp(VALID_END)
    TR_END = pd.Timestamp(TRAIN_END)

    print("\n=== 2. 风格择时策略（train 选 N，test 单次验证） ===")
    print(f"  {'N':<6s}{'train+valid年化%':>17s}{'超额t':>9s}{'换手':>8s}{'test年化%':>11s}")
    print("  " + "-" * 56)
    cands = []
    for n in NS:
        S = strat(n)
        if S is None or len(S) < 24:
            continue
        ser = S['net']
        b = fixed300.reindex(ser.index).fillna(0)
        sel_mask = ser.index <= TV_END
        e = (ser - b)[sel_mask]
        t = e.mean() / e.std() * np.sqrt(len(e)) if e.std() > 0 else np.nan
        a_tv = ann(ser[sel_mask])
        cands.append({'N': n, 'tv_ann': a_tv, 'tv_t': t, 'turn': S['turn'].mean(),
                       'test': ann(ser[ser.index > TV_END]),
                       'S': ser})
        print(f"  {n:<6d}{a_tv:>+17.2f}{t:>+9.2f}{S['turn'].mean():>8.3f}"
              f"{cands[-1]['test']:>+11.2f}")
    if not cands:
        print("  无有效候选")
        return
    best = max(cands, key=lambda x: x['tv_ann'])
    print(f"  → train+valid 选出 N={best['N']}")

    print("\n=== 3. 同期对照（月频年化 %） ===")
    S = best['S']
    rows = []
    for nm, ser in [('风格择时(选中N)', S), ('固定沪深300', fixed300),
                    ('固定中证1000', fixed1000), ('50/50', strat(0)['net']),
                    ('池等权Top500', pool)]:
        e = (ser - fixed300.reindex(ser.index).fillna(0))
        a = ann(ser[ser.index <= TR_END])
        a2 = ann(ser[(ser.index > TR_END) & (ser.index <= TV_END)])
        a3 = ann(ser[ser.index > TV_END])
        t = e.mean() / e.std() * np.sqrt(len(e)) if e.std() > 0 else np.nan
        print(f"  {nm:<18s}{a:>+9.2f}{a2:>+9.2f}{a3:>+9.2f}{t:>+9.2f}")
        rows.append({'strategy': nm, 'train': a, 'valid': a2, 'test': a3, 'excess_t_vs300': t})
    pd.DataFrame(rows).to_csv(OUT / 'style_timing.csv', index=False)

    bt, vt = ann(S[S.index > TV_END]), ann(fixed300[fixed300.index > TV_END])
    print(f"\n  判定：风格择时 test {bt:+.2f}%  vs  固定沪深300 test {vt:+.2f}%  差 {bt-vt:+.2f} pp")
    if bt - vt > 3:
        print("  → **风格可预测，择时有效** —— 这是 2021 失效问题的解法")
    else:
        print("  → 风格近似随机游走，择时无显著增益 → 风格切换应视为不可预测的结构性变化")
    print("\n保存: style_timing.csv")


if __name__ == '__main__':
    main()