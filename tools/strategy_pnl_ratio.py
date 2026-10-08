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
COMM_RATE, STAMP_RATE, MIN_COMM, ETA, SIGMA_BP = 0.00025, 0.0005, 5.0, 0.1, 342.0
MIN_LIST_DAYS = 500
POOLS = [('P1_501-1000', 500, 1000), ('P2_1001-2000', 1000, 2000),
         ('P3_1001-3000', 1000, 3000), ('P4_2001-3000', 2000, 3000)]


def round_trip_cost(w, amt_at, account, turn):
    """单次再平衡的真实成本（占账户比例）。考虑券商最低5元佣金。

    w: 目标权重 Series(票->权重); amt_at: 建仓日成交额 Series;
    account: 账户规模; turn: 换手率(0~1, 与上一期差异)。
    成本 = 卖出旧的 + 买入新的；两边各付 max(金额*0.025%, 5元) 佣金，
          卖出另付 0.05% 印花税，再加 Almgren 冲击。
    """
    n_i = (w * account).clip(lower=0)
    c = 0.0
    for code, ni in n_i.items():
        if ni <= 0:
            continue
        buy = max(ni * COMM_RATE, MIN_COMM)
        sell = max(ni * COMM_RATE, MIN_COMM) + ni * STAMP_RATE
        c += buy + sell
    # 冲击（单边）按 participation
    part = (n_i / amt_at.reindex(w.index)).clip(0, 1).fillna(0)
    imp = float((w * ETA * SIGMA_BP * np.sqrt(part)).sum())
    c += account * imp / 1e4 * 2  # 双边冲击
    return (c / account) * turn


def build_baskets(C, O, AMT, dates, good_days, freq='m'):
    listing = C.notna().cumsum()
    ok_cnt = C.notna().sum(axis=1)
    ref = ok_cnt.rolling(250, min_periods=20).median()
    valid = set(ok_cnt[(ok_cnt > 0.6 * ref) & (ok_cnt > 500)].index)
    if freq == 'q':
        # 季度首：按年+季分组取每组第一个交易日
        grp = dates.to_series().groupby([dates.year, dates.quarter])
    else:
        grp = dates.to_series().groupby([dates.year, dates.month])
    rb_starts = [d for d in grp.min() if d in valid]
    baskets, mstarts = [], []
    for k, d0 in enumerate(rb_starts):
        d_prev = rb_starts[k - 1] if k > 0 else d0
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--freq', choices=['m', 'q'], default='m', help='m=月频, q=季频')
    ap.add_argument('--account', type=float, default=1e7, help='账户规模(元)')
    ap.add_argument('--max-positions', type=int, default=0, help='持仓上限(只), 0=窗口全部')
    args = ap.parse_args()
    FREQ = args.freq
    ACCT = args.account
    MAXPOS = args.max_positions
    print(f"回测口径: {'月频' if FREQ=='m' else '季频'}再平衡  账户规模 {ACCT/1e4:.0f}万  最低佣金{MIN_COMM}元")

    print("载入面板 ...")
    P = pd.read_parquet(PANEL)
    if not isinstance(P.columns, pd.MultiIndex):
        P.columns = pd.MultiIndex.from_tuples(
            [(c, c) if not isinstance(c, tuple) else c for c in P.columns])
    P.index = pd.DatetimeIndex(P.index.get_level_values(0))
    C = P['close']; O = P['open']; AMT = P['amount']
    dates = P.index
    print(f"  {dates[0].date()} ~ {dates[-1].date()}  {len(dates)} 天")

    baskets, mstarts = build_baskets(C, O, AMT, dates, None, freq=FREQ)
    print(f"  再平衡点: {len(mstarts)} 个")

    rows = []
    equity = {}  # pool -> Series(净值, 起始=ACCT)
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
            # 小资金：只保留「买得起100股」且流动性最高的前 N 只（与 make_order_list 对齐）
            if MAXPOS and MAXPOS > 0:
                per = ACCT / MAXPOS
                buyable = [c for c in sel if C.at[d0, c] * 100 <= per * 1.3]
                liq_at = AMT.loc[d0].reindex(buyable).fillna(0)
                sel = list(liq_at.sort_values(ascending=False).index[:MAXPOS])
            need = MAXPOS if (MAXPOS and MAXPOS > 0) else 30
            if len(sel) < need:
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
            cost = round_trip_cost(w, AMT.loc[d0], ACCT, turn)
            net_ret.append(ret - cost)
            turn_log.append(turn)

        ser = pd.Series(net_ret, index=mstarts[:-1])
        eq = ACCT * (1 + ser.fillna(0)).cumprod()
        equity[pname] = eq
        m = metrics(ser)
        if m is None:
            print(f"  {pname}: 样本不足"); continue
        label = f"{pname} [{'月' if FREQ=='m' else '季'}]"
        print(f"\n=== {label}（流动性 {lo+1}-{hi}）===")
        print(f"  交易回合        : {m['n']} 个（{'月' if FREQ=='m' else '季'}）")
        print(f"  胜率            : {m['win_rate']*100:5.1f}%")
        print(f"  平均盈利        : {m['avg_win']*100:6.2f}%")
        print(f"  平均亏损        : {m['avg_loss']*100:6.2f}%")
        print(f"  ★ 盈亏比        : {m['pl_ratio']:.2f}  (平均盈利/平均亏损)")
        print(f"  期望每期收益    : {m['expect']*100:6.3f}%")
        print(f"  年化(复利)      : {m['ann']*100:6.2f}%")
        print(f"  最大回撤        : {m['mdd']*100:6.2f}%")
        print(f"  期末净值        : {eq.iloc[-1]/ACCT*100:6.1f}%  (={eq.iloc[-1]/1e4:.1f}万)")
        avg_turn = np.nanmean(turn_log)
        print(f"  均换手率/期     : {avg_turn*100:5.1f}%")
        rows.append(dict(pool=label, n=m['n'], win_rate=m['win_rate'],
                         avg_win=m['avg_win'], avg_loss=m['avg_loss'],
                         pl_ratio=m['pl_ratio'], expect=m['expect'],
                         ann=m['ann'], mdd=m['mdd'], avg_turn=avg_turn,
                         final_nav=eq.iloc[-1]/ACCT))

    R = pd.DataFrame(rows)
    suf = f"_{FREQ}{int(ACCT):d}" if ACCT != 1e7 else f"_{FREQ}"
    R.to_csv(OUT / f'pnl_ratio{suf}.csv', index=False)
    print(f"\n保存: data/results/pnl_ratio{suf}.csv")

    # 净值曲线导出（供画图）
    eq_df = pd.DataFrame(equity)
    eq_df.index.name = 'date'
    eq_df.to_csv(OUT / f'equity_curve{suf}.csv')

    # 盈亏比按市场段（P3）
    print("\n=== 盈亏比按市场段（P3_1001-3000）===")
    # 重算 P3 序列分段（用 ACCT 成本）
    net_ret = []; prev_sel = None
    for k, (d0, liq, mature, limit_up) in enumerate(baskets):
        if k + 1 >= len(baskets):
            break
        d1 = mstarts[k + 1]
        sel = [c for c in liq.index[1000:3000]
               if mature.get(c, False) and pd.notna(C.at[d0, c])
               and C.at[d0, c] > 0 and not limit_up.get(c, False)]
        if len(sel) < 30:
            net_ret.append(np.nan); prev_sel = set(sel); continue
        w = pd.Series(1.0 / len(sel), index=sel)
        r_gross = (C.loc[d1].reindex(sel) / C.loc[d0].reindex(sel) - 1.0).fillna(0.0)
        ret = float((w * r_gross).sum())
        turn = 1 - len(set(sel) & prev_sel) / len(sel) if prev_sel else 1.0
        prev_sel = set(sel)
        cost = round_trip_cost(w, AMT.loc[d0], ACCT, turn)
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
