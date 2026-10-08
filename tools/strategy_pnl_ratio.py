"""战法交易计划核心指标：盈亏比（平均盈利 / 平均亏损）+ 胜率 + 最大回撤。

策略定义（与 docs/trading_plan.html 一致）：
    选池   每月首个交易日，按过去 60 日成交额中位数排序，取流动性中后段 [lo, hi]
    过滤   上市 >= 500 日、剔除当日涨停开盘、剔除停牌/无价
    持有   等权买入，持有到下一月首再平衡（每月 1 个"交易回合"）
    成本   真实 participation 冲击 + 双边手续费（佣金5+印花5+冲击），按换手率计

输出：data/results/pnl_ratio.csv  + 打印盈亏比/胜率/回撤
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PANEL = Path('data/cache/panel_2011.parquet')
OUT = Path('data/results')
ACCOUNT = 1e7
COMM_BP, STAMP_BP, ETA, SIGMA_BP = 5.0, 5.0, 0.1, 342.0
MIN_LIST_DAYS = 500
POOLS = [('P1_501-1000', 500, 1000), ('P2_1001-2000', 1000, 2000),
         ('P3_1001-3000', 1000, 3000), ('P4_2001-3000', 2000, 3000)]


def build_baskets(C, O, AMT, dates, good_days):
    listing = C.notna().cumsum()
    ok_cnt = C.notna().sum(axis=1)
    ref = ok_cnt.rolling(250, min_periods=20).median()
    valid = set(ok_cnt[(ok_cnt > 0.6 * ref) & (ok_cnt > 500)].index)
    month_starts = [d for d in dates.to_series().groupby(
        [dates.year, dates.month]).min() if d in valid]
    baskets, mstarts = [], []
    for k, d0 in enumerate(month_starts):
        d_prev = month_starts[k - 1] if k > 0 else d0
        hist = AMT.loc[:d_prev].tail(60).median().dropna()
        liq = hist.sort_values(ascending=False)
        mature = listing.loc[:d_prev].iloc[-1] >= MIN_LIST_DAYS
        limit_up = (O.loc[d0] / C.loc[d0].shift(1) - 1.0) >= 0.098
        baskets.append((d0, liq, mature, limit_up))
        mstarts.append(d0)
    return baskets, mstarts


def metrics(returns: pd.Series):
    """returns: 每个交易回合的净收益序列"""
    r = returns.dropna()
    if len(r) < 5:
        return None
    wins = r[r > 0]; losses = r[r < 0]
    win_rate = len(wins) / len(r)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = -losses.mean() if len(losses) else 0.0
    pl_ratio = (avg_win / avg_loss) if avg_loss > 0 else np.inf
    # 期望（每回合）
    expect = win_rate * avg_win - (1 - win_rate) * avg_loss
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return dict(n=len(r), win_rate=win_rate, avg_win=avg_win,
                avg_loss=avg_loss, pl_ratio=pl_ratio, expect=expect,
                ann=ann, mdd=mdd)


def main():
    print("载入面板 ...")
    P = pd.read_parquet(PANEL)
    if not isinstance(P.columns, pd.MultiIndex):
        P.columns = pd.MultiIndex.from_tuples(
            [(c, c) if not isinstance(c, tuple) else c for c in P.columns])
    P.index = pd.DatetimeIndex(P.index.get_level_values(0))
    C = P['close']; O = P['open']; AMT = P['amount']
    dates = P.index
    print(f"  {dates[0].date()} ~ {dates[-1].date()}  {len(dates)} 天")

    baskets, mstarts = build_baskets(C, O, AMT, dates, None)
    print(f"  建仓月首: {len(mstarts)} 个")

    rows = []
    for pname, lo, hi in POOLS:
        net_ret = []
        turn_log = []
        prev_sel = None
        for k, (d0, liq, mature, limit_up) in enumerate(baskets):
            if k + 1 >= len(baskets):
                break
            d1 = mstarts[k + 1]
            sel = [c for c in liq.index[lo:hi]
                   if mature.get(c, False) and pd.notna(C.at[d0, c])
                   and C.at[d0, c] > 0 and not limit_up.get(c, False)]
            if len(sel) < 30:
                net_ret.append(np.nan); turn_log.append(np.nan)
                prev_sel = set(sel); continue
            w = pd.Series(1.0 / len(sel), index=sel)
            # 持有收益：d0 收盘买 -> d1 收盘卖（再平衡）
            r_gross = (C.loc[d1].reindex(sel) / C.loc[d0].reindex(sel) - 1.0).fillna(0.0)
            ret = float((w * r_gross).sum())
            # 换手率：与上一期 basket 的差异
            if prev_sel is not None and prev_sel:
                turn = 1 - len(set(sel) & prev_sel) / len(sel)
            else:
                turn = 1.0
            prev_sel = set(sel)
            # 成本：双边（卖旧+买新）。冲击按 participation
            part = (ACCOUNT * w / AMT.loc[d0].reindex(w.index)).clip(0, 1).fillna(0)
            imp = float((w * ETA * SIGMA_BP * np.sqrt(part)).sum())   # 单边冲击 bp
            cost_bp = (COMM_BP + STAMP_BP + 2 * imp)  # 双边：佣金双边+印花卖+冲击双边
            cost = cost_bp / 1e4 * turn
            net_ret.append(ret - cost)
            turn_log.append(turn)

        ser = pd.Series(net_ret, index=mstarts[:-1])
        m = metrics(ser)
        if m is None:
            print(f"  {pname}: 样本不足"); continue
        print(f"\n=== {pname}（流动性 {lo+1}-{hi}）===")
        print(f"  交易回合数      : {m['n']} 个月")
        print(f"  胜率            : {m['win_rate']*100:5.1f}%")
        print(f"  平均盈利(月)    : {m['avg_win']*100:6.2f}%")
        print(f"  平均亏损(月)    : {m['avg_loss']*100:6.2f}%")
        print(f"  ★ 盈亏比        : {m['pl_ratio']:.2f}  (平均盈利/平均亏损)")
        print(f"  期望月收益      : {m['expect']*100:6.3f}%")
        print(f"  年化(复利)      : {m['ann']*100:6.2f}%")
        print(f"  最大回撤        : {m['mdd']*100:6.2f}%")
        avg_turn = np.nanmean(turn_log)
        print(f"  月均换手率      : {avg_turn*100:5.1f}%")
        rows.append(dict(pool=pname, n=m['n'], win_rate=m['win_rate'],
                         avg_win=m['avg_win'], avg_loss=m['avg_loss'],
                         pl_ratio=m['pl_ratio'], expect=m['expect'],
                         ann=m['ann'], mdd=m['mdd'], avg_turn=avg_turn))

    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'pnl_ratio.csv', index=False)
    print("\n保存: data/results/pnl_ratio.csv")

    # 额外：每回合盈亏比的稳定性（分牛熊段）
    print("\n=== 盈亏比按市场段（P3_1001-3000）===")
    ser_all = None
    for pname, lo, hi in POOLS:
        if pname == 'P3_1001-3000':
            # 重算序列（上面已 append 但没存序列）；简单重跑
            pass
    # 直接重算 P3 序列分段
    net_ret = []; prev_sel = None
    for k, (d0, liq, mature, limit_up) in enumerate(baskets):
        if k + 1 >= len(baskets):
            break
        d1 = mstarts[k + 1]
        sel = [c for c in liq.index[lo:hi]
               if mature.get(c, False) and pd.notna(C.at[d0, c])
               and C.at[d0, c] > 0 and not limit_up.get(c, False)]
        if len(sel) < 30:
            net_ret.append(np.nan); prev_sel = set(sel); continue
        w = pd.Series(1.0 / len(sel), index=sel)
        r_gross = (C.loc[d1].reindex(sel) / C.loc[d0].reindex(sel) - 1.0).fillna(0.0)
        ret = float((w * r_gross).sum())
        turn = 1 - len(set(sel) & prev_sel) / len(sel) if prev_sel else 1.0
        prev_sel = set(sel)
        part = (ACCOUNT * w / AMT.loc[d0].reindex(w.index)).clip(0, 1).fillna(0)
        imp = float((w * ETA * SIGMA_BP * np.sqrt(part)).sum())
        cost = (COMM_BP + STAMP_BP + 2 * imp) / 1e4 * turn
        net_ret.append(ret - cost)
    ser_all = pd.Series(net_ret, index=mstarts[:-1])
    for seg, a, b in [('2011-2016', '2011-01-01', '2016-12-31'),
                      ('2017-2020', '2017-01-01', '2020-12-31'),
                      ('2021-2026', '2021-01-01', '2026-09-30')]:
        mm = metrics(ser_all[(ser_all.index >= a) & (ser_all.index <= b)])
        if mm:
            print(f"  {seg}: 盈亏比={mm['pl_ratio']:.2f}  胜率={mm['win_rate']*100:.1f}%  "
                  f"年化={mm['ann']*100:.1f}%  回撤={mm['mdd']*100:.1f}%")


if __name__ == '__main__':
    main()
