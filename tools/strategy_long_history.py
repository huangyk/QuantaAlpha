"""战法长期验证：2011-2026（15.8 年）——「小盘/低流动性长期跑赢」是风格规律还是 2020 后才有？

这是换池方案的关键验证。之前只在 2020-2026（6.3 年）测过，得到
「池子越偏中小盘越集中，长期收益越高」。但 6 年只覆盖一轮牛熊，
无法区分「风格规律」与「2020 后特有现象」。

设计要点（全部为消除前视）：
    建池   每月首个交易日，按**过去 60 日成交额中位数**排序分层，
          而非使用静态的 liquid500 清单（那是 2026 年的清单，用它回测 2011 属幸存者偏差）
    过滤   剔除上市不足 500 个交易日（数据长度近似）、剔除当日涨停开盘
    持有   等权买入后**永久持有**，不再平衡（与战法一致）
    成本   用真实 amount 字段算 participation（0.1×σ×√part，σ=342bp）
    基准   同池等权（同建仓日、同过滤规则），确保同口径

分层（按建仓日流动性从高到低）：
    L1 Top500 / L2 501-1000 / L3 1001-2000 / L4 2000+ / LALL 全部

输出：data/results/long_history_pool.csv（分段与逐年）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path('data/cache/panel_2011.parquet')
OUT = Path('data/results')
ACCOUNT = 1e7
COMMISSION_BP, STAMP_BP, ETA, SIGMA_BP = 5.0, 5.0, 0.1, 342.0
MIN_LIST_DAYS = 500
LAYERS = [('L1_Top500', 0, 500), ('L2_501-1000', 500, 1000),
          ('L3_1001-2000', 1000, 2000), ('L4_2000以上', 2000, 10 ** 9),
          ('LALL_全部', 0, 10 ** 9)]
SEGS = [('2011-2016', '2011-01-01', '2016-12-31'),
        ('2017-2020', '2017-01-01', '2020-12-31'),
        ('2021-2026', '2021-01-01', '2026-09-30')]


def main():
    print("载入 2011-2026 面板 ...")
    P = pd.read_parquet(PANEL)
    if not isinstance(P.columns, pd.MultiIndex):
        P.columns = pd.MultiIndex.from_tuples(
            [tuple(c) if isinstance(c, tuple) else (c, c) for c in P.columns])
    P.index = pd.DatetimeIndex(P.index.get_level_values(0))
    C = P['close']; O = P['open']; AMT = P['amount']
    ret = (C.shift(-1) / C - 1.0).clip(-0.5, 0.5)
    print(f"  {P.index[0].date()} ~ {P.index[-1].date()}  {len(P.index)} 天 × {C.shape[1]} 只")

    listing = C.notna().cumsum()
    # ⚠️ 数据完整性检查：某些交易日全市场只有百来只更新，必须跳过
    ok_cnt = C.notna().sum(axis=1)
    # ⚠️ 完整性检查必须相对于「当时实际存在的股票数」——2011 年全 A 只有 ~1900 只，
    #    若拿全表 5224 只做阈值会把整个 2011-2016 全部误杀。
    #    正确做法：与近 250 日的覆盖中位数比较。
    ref = ok_cnt.rolling(250, min_periods=20).median()
    good_days = set(ok_cnt[(ok_cnt > 0.6 * ref) & (ok_cnt > 500)].index)
    dates = P.index
    month_starts = [d for d in dates.to_series().groupby(
        [dates.year, dates.month]).min() if d in good_days]
    print(f"  可用建仓日（月首且数据完整）: {len(month_starts)} 个")
    if not month_starts:
        print("  ✗ 没有可用建仓日，样本不足")
        return
    print(f"  首个建仓日 {month_starts[0].date()}  末个 {month_starts[-1].date()}")

    # ---- 逐个建仓日跑 buy & hold ----
    nav = {name: [] for name, _, _ in LAYERS}
    sel_sizes = {}
    turn_log = {name: [] for name, _, _ in LAYERS}
    used = []

    for k, d0 in enumerate(month_starts):
        d_prev = month_starts[k - 1] if k > 0 else d0
        hist = AMT.loc[:d_prev].tail(60).median()          # 过去 60 日成交额中位
        mature = listing.loc[:d_prev].iloc[-1] >= MIN_LIST_DAYS
        limit_up = (O.loc[d0] / C.loc[d0].shift(1) - 1.0) >= 0.098
        liq = hist.dropna().sort_values(ascending=False)   # ⚠️ 必须排序，否则取到字母序前 N 只
        for name, lo, hi in LAYERS:
            sel = [c for c in liq.index[lo:hi]
                   if mature.get(c, False) and pd.notna(C.at[d0, c])
                   and C.at[d0, c] > 0 and not limit_up.get(c, False)]
            if len(sel) < 30:
                nav[name].append(0.0); turn_log[name].append(0.0)
                continue
            w = pd.Series(1.0 / len(sel), index=sel)
            sel_sizes[name] = sel_sizes.get(name, 0) + len(sel)
            if k == 0:
                part = (ACCOUNT * w / AMT.loc[d0].reindex(w.index)).clip(0, 1).fillna(0)
                imp = float((w * ETA * SIGMA_BP * np.sqrt(part)).sum())
                cost = (COMMISSION_BP + STAMP_BP + 2 * imp) / 1e4
                turn_log[name].append(1.0)
            else:
                cost = 0.0
                turn_log[name].append(0.0)
            # 建仓当期收益：d0 收盘建仓 → d0+1 收盘
            r = ret.loc[d0].reindex(w.index).fillna(0.0)
            nav[name].append(float((w * r).sum()) - cost)
        used.append(d0)

    idx_use = pd.DatetimeIndex(used)

    def metrics(ser):
        ser = ser[ser != 0]
        if len(ser) < 3:
            return None
        # ⚠️ ser 是**月度**收益（每个建仓日一条），年化必须用 12 期，不能用 252
        ann = (1 + ser).prod() ** (12 / len(ser)) - 1
        cum = (1 + ser).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        ir = ser.mean() / ser.std() * np.sqrt(12) if ser.std() > 0 else np.nan
        return {'年化%': ann * 100, 'IR': ir, '回撤%': mdd * 100, '月数': len(ser)}

    print(f"\n{'分层':<14s}{'只数':>7s}{'全期年化%':>11s}{'IR':>7s}{'回撤%':>9s}"
          f"{'2011-2016':>11s}{'2017-2020':>11s}{'2021-2026':>11s}")
    print("-" * 90)
    rows = []
    layer_sizes = {}
    for name, lo, hi in LAYERS:
        ser = pd.Series(nav[name], index=idx_use)
        m = metrics(ser)
        if m is None:
            continue
        seg_txt = {}
        for nm, a, b in SEGS:
            mask = (ser.index >= pd.Timestamp(a)) & (ser.index <= pd.Timestamp(b))
            mm = metrics(ser[mask])
            seg_txt[nm] = mm['年化%'] if mm else np.nan
        print(f"{name:<14s}{layer_sizes.get(name, 0):>7d}{m['年化%']:>+11.2f}{m['IR']:>+7.2f}{m['回撤%']:>+9.2f}"
              + "".join(f"{seg_txt[s[0]]:>+11.2f}" for s in SEGS))
        layer_sizes[name] = int(sel_sizes.get(name, 0))
        rows.append({'layer': name, 'avg_n': int(sel_sizes.get(name, 0)), **m, **seg_txt})

    print("-" * 90)
    print("口径：买入并永久持有；首期扣一次建仓成本（含真实 participation 冲击）")

    # 逐年
    print("\n逐年年化（%）：")
    print(f"  {'分层':<14s}" + "".join(f"{y:>8d}" for y in range(2011, 2027)))
    for name, _, _ in LAYERS:
        ser = pd.Series(nav[name], index=idx_use)
        line = f"  {name:<14s}"
        for y in range(2011, 2027):
            mask = (ser.index.year == y)
            mm = metrics(ser[mask])
            line += f"{mm['年化%']:>+8.1f}" if mm else f"{'-':>8s}"
        print(line)


    # ---- 滚动起点：不同建仓起点的年化（检验结论是否稳健）----
    print("\n=== 滚动起点：从 YYYY 开始建仓，到 2026-09 的年化（%） ===")
    print(f"  {'起点':<8s}" + "".join(f"{y:>8d}" for y in (2013, 2015, 2017, 2019, 2021, 2023, 2025)))
    print(f"  {'—':<8s}" + "-" * 56)
    roll_rows = []
    for name, lo, hi in LAYERS:
        ser = pd.Series(nav[name], index=idx_use)
        line = f"  {name:<8s}"
        row = {'layer': name}
        for y0 in (2013, 2015, 2017, 2019, 2021, 2023, 2025):
            mask = ser.index >= pd.Timestamp(f'{y0}-01-01')
            sub = ser[mask]
            mm = metrics(sub) if len(sub[sub != 0]) >= 3 else None
            line += f"{mm['年化%']:>+8.2f}" if mm else f"{'-':>8s}"
            row[f'from{y0}'] = mm['年化%'] if mm else np.nan
        print(line)
        roll_rows.append(row)
    print("\n  → 若不同起点的结果差异巨大，说明该策略依赖特定市场阶段，稳健性差")
    pd.DataFrame(roll_rows).to_csv(OUT / 'long_history_rolling.csv', index=False)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'long_history_pool.csv', index=False)
    print("\n保存: long_history_pool.csv")


if __name__ == '__main__':
    main()