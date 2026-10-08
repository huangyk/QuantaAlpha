"""诊断：池等权为什么 2021 年之后失效？

上一轮长周期验证发现：动态池（每月按当时成交额重选 Top500）buy&hold
2013-2020 每年为正、2021 后多数为负。这是最值得追的线索 —— 原因里可能藏着新信号。

三种可能，用数据区分：
    A. **市场问题**：整个 A 股都跌 → 池子没错，只是行情差
    B. **选池问题**：大盘/宽基在涨，而「流动性 Top500」在跌 → 选池逻辑错了
    C. **结构问题**：池子成分频繁更替/权重漂移导致失效

对照基准（数据库自带指数，2011-2026 完整）：
    000300.SH 沪深300   000905.SH 中证500   000852.SH 中证1000
    000001.SH 上证      399006.SZ 创业板指   000016.SH 上证50
    000688.SH 科创50（2019 起）

输出：data/results/diagnose_2021.csv
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
IDX = {'沪深300': '000300.SH', '中证500': '000905.SH', '中证1000': '000852.SH',
       '上证': '000001.SH', '创业板指': '399006.SZ', '上证50': '000016.SH'}
CUT = pd.Timestamp('2021-01-01')


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
    ret = (C.shift(-1) / C - 1.0).clip(-0.5, 0.5)
    listing = C.notna().cumsum()

    ok_cnt = C.notna().sum(axis=1)
    ref = ok_cnt.rolling(250, min_periods=20).median()
    good = set(ok_cnt[(ok_cnt > 0.6 * ref) & (ok_cnt > 500)].index)
    dates = P.index
    mstart = [d for d in dates.to_series().groupby([dates.year, dates.month]).min() if d in good]

    # ---- 建池：流动性 Top500 / 中盘500-2000 / 全成熟 ----
    pools = {'Top500': (0, 500), '501-2000': (500, 2000), '全成熟': (0, 10 ** 9)}
    nav = {k: [] for k in pools}
    idx_nav = {k: [] for k in IDX}
    member_log = []      # (date, set)
    member_dates = []    # 与 member_log 一一对应
    for k, d0 in enumerate(mstart):
        d_prev = mstart[k - 1] if k > 0 else d0
        liq = AMT.loc[:d_prev].tail(60).median().dropna().sort_values(ascending=False)  # ⚠️ 必须排序，否则取到的是字母序前 N 只
        mature = listing.loc[:d_prev].iloc[-1] >= MIN_LIST_DAYS
        limit_up = (O.loc[d0] / C.loc[d0].shift(1) - 1.0) >= 0.098
        for name, (lo, hi) in pools.items():
            sel = [c for c in liq.index[lo:hi]
                   if mature.get(c, False) and pd.notna(C.at[d0, c]) and C.at[d0, c] > 0
                   and not limit_up.get(c, False)]
            if len(sel) < 30:
                nav[name].append(0.0)
                continue
            w = pd.Series(1.0 / len(sel), index=sel)
            if k == 0:
                part = (ACCOUNT * w / AMT.loc[d0].reindex(w.index)).clip(0, 1).fillna(0)
                imp = float((w * ETA * SIGMA_BP * np.sqrt(part)).sum())
                cost = (COMMISSION_BP + STAMP_BP + 2 * imp) / 1e4
            else:
                cost = 0.0
            nav[name].append(float((w * ret.loc[d0].reindex(w.index).fillna(0.0)).sum()) - cost)
            if name == 'Top500':
                member_log.append((d0, set(sel)))
                member_dates.append(d0)
        # 指数收益
        for nm, code in IDX.items():
            try:
                px = I['close'][code]
                idx_nav[nm].append(float(px.loc[d0 + pd.Timedelta(days=1)] / px.loc[d0] - 1)
                                    if (d0 + pd.Timedelta(days=1)) in px.index else 0.0)
            except KeyError:
                idx_nav[nm].append(0.0)

    idx_use = pd.DatetimeIndex(mstart)

    def ann(ser):
        ser = ser[ser != 0]
        if len(ser) < 3:
            return np.nan
        return ((1 + ser).prod() ** (12 / len(ser)) - 1) * 100

    # ---- 1. 池 vs 指数，分 2021 前后 ----
    print("=" * 92)
    print("诊断 1：池 vs 宽基指数，分 2021 年前后（月频年化 %）")
    print("=" * 92)
    rows = []
    for name in list(pools) + list(IDX):
        ser = pd.Series(nav[name] if name in nav else idx_nav[name], index=idx_use)
        pre = ser[ser.index < CUT]
        post = ser[ser.index >= CUT]
        rows.append({'series': name, '全期': ann(ser), '2021前': ann(pre), '2021后': ann(post),
                     '前后差': ann(post) - ann(pre) if ann(post) == ann(post) else np.nan})
    D = pd.DataFrame(rows)
    print(f"  {'序列':<12s}{'全期':>10s}{'2021前':>10s}{'2021后':>10s}{'前后差':>10s}")
    print("-" * 56)
    for _, r in D.iterrows():
        print(f"  {r['series']:<12s}{r['全期']:>+10.2f}{r['2021前']:>+10.2f}"
              f"{r['2021后']:>+10.2f}{r['前后差']:>+10.2f}")

    top_pre = D.loc[D['series'] == 'Top500', '2021前'].iloc[0]
    top_post = D.loc[D['series'] == 'Top500', '2021后'].iloc[0]
    best_post = D[D['series'].isin(IDX)].nlargest(1, '2021后')['series'].iloc[0]
    best_post_v = D.loc[D['series'] == best_post, '2021后'].iloc[0]
    print("-" * 56)
    print(f"  判定：")
    print(f"    2021 前：Top500 {top_pre:+.2f}%  vs 最好的宽基 "
          f"{D[D['series'].isin(IDX)].nlargest(1, '2021前')['series'].iloc[0]} "
          f"{D[D['series'].isin(IDX)].nlargest(1, '2021前')['2021前'].iloc[0]:+.2f}%")
    print(f"    2021 后：Top500 {top_post:+.2f}%  vs 最好的宽基 {best_post} {best_post_v:+.2f}%")
    if best_post_v > top_post + 5:
        print(f"    → **选池问题**：{best_post} 同期比流动性Top500 高 {best_post_v-top_post:.1f}pp")
    else:
        print(f"    → **市场问题**：宽基同期也不差，不是池子选错")

    # ---- 2. Top500 成分更替速度 ----
    print("\n" + "=" * 92)
    print("诊断 2：Top500 成分更替速度（结构稳定性）")
    print("=" * 92)
    turn_ = []
    for i in range(1, len(member_log)):
        a, b = member_log[i - 1][1], member_log[i][1]
        turn_.append(1 - len(a & b) / max(len(b), 1))
    T = pd.Series(turn_, index=pd.DatetimeIndex(member_dates[1:]))
    for nm, a, b in [('2013-2020', '2013-01-01', '2020-12-31'),
                     ('2021-2026', '2021-01-01', '2026-09-30')]:
        m = (T.index >= pd.Timestamp(a)) & (T.index <= pd.Timestamp(b))
        print(f"  {nm} 月度成分更替率：均值 {T[m].mean():.1%}  中位 {T[m].median():.1%}")
    rows.append({'月度更替_全期': T.mean(), '月度更替_2021后': T[T.index >= CUT].mean()})

    # ---- 3. 逐年对照 ----
    print("\n" + "=" * 92)
    print("诊断 3：逐年年化（%），池 vs 指数")
    print("=" * 92)
    yrs = list(range(2013, 2027))
    print(f"  {'序列':<12s}" + "".join(f"{y:>7d}" for y in yrs))
    for name in list(pools) + ['沪深300', '中证500', '中证1000', '创业板指']:
        ser = pd.Series(nav[name] if name in nav else idx_nav[name], index=idx_use)
        line = f"  {name:<12s}"
        for y in yrs:
            m = ser.index.year == y
            sub = ser[m]
            v = ann(sub) if len(sub[sub != 0]) >= 3 else np.nan
            line += f"{v:>+7.1f}" if v == v else f"{'-':>7s}"
        print(line)
    print("\n  → 看 2021 年起：若池子连续为负而某类指数稳定为正 → 该放弃「流动性Top500」这个池")

    # ---- 4. 风格诊断：大盘 vs 中小盘，谁在赢 ----
    print("\n" + "=" * 92)
    print("诊断 4：风格 —— 大盘（沪深300）相对中小盘（中证500/1000）的超额")
    print("=" * 92)
    s300 = pd.Series(idx_nav['沪深300'], index=idx_use)
    s500 = pd.Series(idx_nav['中证500'], index=idx_use)
    s1000 = pd.Series(idx_nav['中证1000'], index=idx_use)
    top5 = pd.Series(nav['Top500'], index=idx_use)
    print(f"  {'区间':<12s}{'沪深300':>10s}{'中证500':>10s}{'中证1000':>11s}{'Top500':>10s}{'300-1000':>10s}")
    for nm, a, b in [('2013-2016', '2013-01-01', '2016-12-31'),
                     ('2017-2020', '2017-01-01', '2020-12-31'),
                     ('2021-2026', '2021-01-01', '2026-09-30')]:
        m = (s300.index >= pd.Timestamp(a)) & (s300.index <= pd.Timestamp(b))
        v300, v500, v1000, vtop = ann(s300[m]), ann(s500[m]), ann(s1000[m]), ann(top5[m])
        print(f"  {nm:<12s}{v300:>+10.2f}{v500:>+10.2f}{v1000:>+11.2f}{vtop:>+10.2f}{v300-v1000:>+10.2f}")
    print("\n  → 「300−1000」为正 = 大盘占优；为负 = 中小盘占优")
    print("  → 若 2021 起该值转正且持续，则「流动性Top500」失效是**风格切换**所致，属可修复问题")

    pd.DataFrame(D).to_csv(OUT / 'diagnose_2021.csv', index=False)
    print("\n保存: diagnose_2021.csv")


if __name__ == '__main__':
    main()