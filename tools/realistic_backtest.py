#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
realistic_backtest.py — 5步框架「带真实执行摩擦」回测 + 泄漏归因
=====================================================================
把交易计划的 5 步框架(看大盘→选个股→看位置→买点→止损止盈)做成可回测的
月频多空(仅多头)策略，并建模此前所有回测都忽略的真实摩擦：

  1. 涨停买不进   open >= prev_close*1.098  → 当天无法买入，信号作废(漏买)
  2. 跌停卖不出   open <= prev_close*0.902  → 当天无法卖出，持仓顺延(漏卖)
  3. 流动性截断   单票下单额 > 当日成交额*LIQ_CAP(默认5%) → 只能买到上限(额度漏动)
  4. 滑点         参与率模型 impact_bp = IMPACT_K * sqrt(participation)，买卖双向
  5. 手续费       佣金 0.025%(最低5元,双边) + 印花税 0.05%(仅卖) + 过户费 0.001%(双边)

双路径对比：
  - 理想路径(friction=False)：信号价精确成交、无滑点无费、涨停跌停也"照常"成交
  - 真实路径(friction=True) ：上述所有摩擦 + 涨停跌停不可成交
  → 泄漏 = 理想NAV / 真实NAV - 1，并拆成 漏买/漏卖/流动性/滑点/手续费 五项

数据：TimescaleDB bars_1d（全A日线，含 open/high/low/close/volume(手)/amount(元)）。
口径局限：bars_5m仅2026，本回测用日线，分时痕迹不建模；筹码分布(CYQ)仅作参考未接入选股；
         创业板指(399006)库空，大盘闸门用 沪深300/中证500/中证1000/上证。

用法：
  python tools/realistic_backtest.py                      # 默认8万/6只/2021起
  python tools/realistic_backtest.py --capital 1e6        # 100万看流动性泄漏
  python tools/realistic_backtest.py --start 2023-01-01 --n-positions 8
"""
import argparse, sys, json
import numpy as np
import pandas as pd
import psycopg2

# ── DB ──
DB = dict(host="127.0.0.1", port=5432, dbname="quant_data", user="quant", password="quant123456")
def conn():
    return psycopg2.connect(**DB)
def sql(cur, q, params=None):
    cur.execute(q, params); return cur.fetchall()

# ── 常量 ──
INDICES = {"000300.SH": "沪深300", "000905.SH": "中证500",
           "000852.SH": "中证1000", "000001.SH": "上证指数"}
MIN_LIST_DAYS = 250          # 上市满 250 日(剔除次新)
LIMIT_UP = 0.098             # 涨停(主板10%板，用9.8%判定开盘即板)
LIMIT_DOWN = 0.098
IMPACT_K = 34.2              # 滑点系数：impact_bp = IMPACT_K*sqrt(participation)，participation=100%时34bp
LIQ_CAP = 0.05               # 单票下单额 ≤ 当日成交额5%(超过截断)
COMM_RATE = 0.00025          # 佣金 0.025%
COMM_MIN = 5.0               # 最低5元
STAMP_RATE = 0.0005          # 印花税 0.05%(仅卖)
TRANSFER_RATE = 0.00001      # 过户费 0.001%(双边)


def load_market(cur):
    """指数序列(近130日, 截断到最新交易日)，用于大盘闸门。"""
    rows = sql(cur, """select asset_code, trade_date, close from bars_1d
                       where asset_code = any(%s) order by asset_code, trade_date""",
               (list(INDICES),))
    df = pd.DataFrame(rows, columns=["code", "date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def regime(env, breadth):
    """返回目标仓位暴露(0/0.5/0.8)。弱市(指数全弱 且 宽度<50%)→0(空仓)。"""
    weak = all(e["label"] == "弱" for e in env.values()) and (breadth is None or breadth < 50)
    strong = breadth is not None and breadth >= 70 and all(e["close"] >= e["ma60"] for e in env.values())
    if weak:
        return 0.0, "弱市(空仓)"
    if strong:
        return 0.8, "强市(80%)"
    return 0.5, "震荡(50%)"


def compute_env(idx, as_of):
    """step1 大盘环境(复用 trading_checklist 的口径，已修偷看未来)。"""
    env = {}
    for code, name in INDICES.items():
        s = idx[(idx.code == code) & (idx.date <= as_of)].sort_values("date").set_index("date")["close"]
        if len(s) < 60:
            continue
        ma20 = s.rolling(20).mean().iloc[-1]
        ma60 = s.rolling(60).mean().iloc[-1]
        ret20 = s.iloc[-1] / s.iloc[-20] - 1
        above = s.iloc[-1] > ma20
        label = "强" if (above and ret20 > 0) else ("弱" if (not above and ret20 < 0) else "震荡")
        env[name] = dict(close=s.iloc[-1], ma20=ma20, ma60=ma60, ret20=ret20, label=label)
    return env


def compute_breadth(stock_df, as_of):
    """全市宽度：站上MA20的票占比(截断到as_of)。"""
    sub = stock_df[stock_df.date <= as_of]
    last = sub[sub.date == as_of]
    if last.empty:
        return None
    ma20 = sub.groupby("code")["close"].apply(lambda x: x.tail(20).mean())
    m = last.set_index("code")["close"]
    both = m.index.intersection(ma20.index)
    if not len(both):
        return None
    return float((m[both] > ma20[both]).mean() * 100)


def build_by_code(cur, load_start, end):
    """加载股票日线(全样本窗口)，按 code 拆成 {code: DataFrame(date索引)}，
    预计算滚动指标(ma20/ma60/ret20/hi20/lo5/amt_med/above_ma20) + prev_close，
    并返回 above_mat(全市站上MA20的布尔矩阵, 用于O(1)算宽度)。"""
    rows = sql(cur, """select asset_code, trade_date, open, high, low, close, volume, amount
                       from bars_1d where asset_type='stock'
                       and trade_date >= %s and trade_date <= %s
                       order by asset_code, trade_date""", (load_start, end))
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    by_code = {}
    above_series = {}
    for code, g in df.groupby("code"):
        g = g.sort_values("date").set_index("date")
        g["prev_close"] = g["close"].shift(1)
        g["ma20"] = g["close"].rolling(20).mean()
        g["ma60"] = g["close"].rolling(60).mean()
        g["ret20"] = g["close"].pct_change(20, fill_method=None)
        g["hi20"] = g["high"].rolling(20).max()
        g["lo5"] = g["low"].rolling(5).min()
        g["amt_med"] = g["amount"].rolling(20).median()
        g["above_ma20"] = g["close"] > g["ma20"]
        by_code[code] = g
        above_series[code] = g["above_ma20"]
    above_mat = pd.DataFrame(above_series)
    trading_days = sorted(df["date"].unique())
    return by_code, above_mat, trading_days


def select(by_code, d, n_pos, min_amount):
    """位置优先选股(贴近支撑 + 动量正)，返回候选列表。O(1)行查找。"""
    cand = []
    for code, g in by_code.items():
        if d not in g.index:
            continue
        if len(g) < MIN_LIST_DAYS:
            continue
        r = g.loc[d]
        c = r["close"]
        ma20 = r["ma20"]; ma60 = r["ma60"]; ret20 = r["ret20"]
        hi20 = r["hi20"]; lo5 = r["lo5"]; amt_med = r["amt_med"]
        if not pd.notna(ma20) or not pd.notna(ret20) or not pd.notna(amt_med):
            continue
        if not (amt_med >= min_amount):
            continue
        if ret20 <= 0:
            continue
        pct_ma20 = (c / ma20 - 1) * 100
        buy_trigger = ma20 * 0.995 if c > ma20 else c
        stop = max(ma20 * 0.985, lo5 * 0.999, c * 0.85)
        stop = min(stop, c * 0.999)
        tp1 = c * 1.10
        tp2 = hi20
        cand.append(dict(code=code, close=c, pct_ma20=pct_ma20, ret20=ret20,
                         buy_trigger=buy_trigger, stop=stop, tp1=tp1, tp2=tp2,
                         near_support=(pct_ma20 <= 15)))
    cand.sort(key=lambda x: (0 if x["near_support"] else 1, -x["ret20"]))
    return cand[:n_pos]


def run_backtest(by_code, above_mat, trading_days, idx, start, capital, n_pos, min_amount, friction):
    """执行回测。friction=True 建模全部摩擦+涨停跌停不可成交；False=理想。"""
    rb_days = set()
    for d in trading_days:
        # 月末最后一个交易日
        nxt = pd.Timestamp(d) + pd.offsets.MonthEnd(0)
        if d == max((x for x in trading_days if x <= nxt), default=None):
            rb_days.add(d)
    cash = capital
    holdings = {}            # code -> dict(shares, entry, stop, tp1, tp2, half, buy_day)
    pending = None           # (targets_list, exposure, weak)
    nav = []
    leaks = dict(limit_up_miss=0, limit_up_miss_ret=0.0,
                 limit_down_hold=0, liq_trunc_notional=0.0,
                 slippage_cost=0.0, fee_cost=0.0,
                 suspension_days=0, trades_filled=0, trades_attempted=0)

    def prev_c(code, t):
        g = by_code[code]
        if t in g.index:
            pc = g.loc[t, "prev_close"]
            return pc if pd.notna(pc) else np.nan
        return np.nan

    for i, t in enumerate(trading_days):
        as_of_d = t
        # --- 计算大盘环境(月末定调) ---
        if t in rb_days:
            env = compute_env(idx, t)
            breadth = compute_breadth_stock(above_mat, t)
            exposure, reg_label = regime(env, breadth)
            targets = select(by_code, t, n_pos, min_amount)
            # 只保留有买点的(次日开盘<=buy_trigger才可能成交；理想路径也需判断)
            pending = (targets, exposure, reg_label)
            exec_day = trading_days[i + 1] if i + 1 < len(trading_days) else None

        # --- 执行 pending(次交易日开盘) ---
        if pending is not None and i > 0 and trading_days[i - 1] in rb_days:
            targets, exposure, reg_label = pending
            pending = None
            # 卖出不在新篮子里的旧持仓(弱市则全平)
            for code in list(holdings.keys()):
                in_new = any(x["code"] == code for x in targets)
                if exposure == 0 or not in_new:
                    g = by_code[code]
                    if t in g.index:
                        bar = g.loc[t]
                        pc = prev_c(code, t)
                        if friction and pd.notna(pc) and bar["open"] <= pc * (1 - LIMIT_DOWN):
                            leaks["limit_down_hold"] += 1
                            continue  # 跌停卖不出，顺延
                        sell_px = bar["open"]
                        part = (holdings[code]["shares"] * 100 * sell_px) / bar["amount"] if bar["amount"] > 0 else 1
                        slip = IMPACT_K * np.sqrt(min(part, 1)) if friction else 0
                        fill = sell_px * (1 - slip / 1e4) if friction else sell_px
                        proceeds = holdings[code]["shares"] * 100 * fill
                        comm = max(COMM_MIN, proceeds * COMM_RATE) if friction else 0
                        stamp = proceeds * STAMP_RATE if friction else 0
                        trans = proceeds * TRANSFER_RATE if friction else 0
                        cash += proceeds - comm - stamp - trans
                        if friction:
                            leaks["fee_cost"] += comm + stamp + trans
                            leaks["slippage_cost"] += slip / 1e4 * proceeds
                        del holdings[code]
            # 买入新篮子
            if exposure > 0:
                per = exposure * capital / max(n_pos, 1)
                for x in targets:
                    code = x["code"]
                    g = by_code[code]
                    if t not in g.index:
                        continue
                    bar = g.loc[t]
                    pc = prev_c(code, t)
                    leaks["trades_attempted"] += 1
                    # 涨停买不进
                    if friction and pd.notna(pc) and bar["open"] >= pc * (1 + LIMIT_UP):
                        leaks["limit_up_miss"] += 1
                        continue
                    # 追高过滤：开盘较信号日收盘涨超3%则不追(非泄漏，仅不追高)
                    if bar["open"] > x["close"] * 1.03:
                        continue
                    notional = per
                    part = notional / bar["amount"] if bar["amount"] > 0 else 1
                    slip = IMPACT_K * np.sqrt(min(part, 1)) if friction else 0
                    fill = bar["open"] * (1 + slip / 1e4) if friction else bar["open"]
                    if not pd.notna(fill) or fill <= 0:
                        continue
                    # 流动性截断
                    cap = bar["amount"] * LIQ_CAP
                    if friction and notional > cap:
                        leaks["liq_trunc_notional"] += notional - cap
                        notional = cap
                    shares = int(notional // (fill * 100))
                    if shares < 1:
                        continue
                    cost = shares * 100 * fill
                    comm = max(COMM_MIN, cost * COMM_RATE) if friction else 0
                    trans = cost * TRANSFER_RATE if friction else 0
                    if cost + comm + trans > cash:
                        # 现金不足，按比例减
                        affordable = cash / (fill * 100 * (1 + (COMM_RATE if friction else 0)))
                        shares = int(affordable)
                        if shares < 1:
                            continue
                        cost = shares * 100 * fill
                        comm = max(COMM_MIN, cost * COMM_RATE) if friction else 0
                        trans = cost * TRANSFER_RATE if friction else 0
                    cash -= cost + comm + trans
                    if friction:
                        leaks["fee_cost"] += comm + trans
                        leaks["slippage_cost"] += slip / 1e4 * cost
                    holdings[code] = dict(shares=shares, entry=fill, stop=x["stop"],
                                          tp1=x["tp1"], tp2=x["tp2"], half=False, buy_day=t)
                    leaks["trades_filled"] += 1

        # --- 每日止损/止盈检查 ---
        for code in list(holdings.keys()):
            h = holdings[code]
            if t <= h["buy_day"]:
                continue
            g = by_code[code]
            if t not in g.index:
                leaks["suspension_days"] += 1
                continue
            bar = g.loc[t]
            pc = prev_c(code, t)
            low, high, open_p, close = bar["low"], bar["high"], bar["open"], bar["close"]
            sell_px, reason = None, None
            if low <= h["stop"]:
                sell_px = open_p if open_p < h["stop"] else h["stop"]
                reason = "stop"
            elif not h["half"] and high >= h["tp1"]:
                sell_px = open_p if open_p > h["tp1"] else h["tp1"]
                reason = "tp1"
            elif high >= h["tp2"]:
                sell_px = open_p if open_p > h["tp2"] else h["tp2"]
                reason = "tp2"
            if sell_px is not None:
                if friction and pd.notna(pc) and open_p <= pc * (1 - LIMIT_DOWN):
                    leaks["limit_down_hold"] += 1
                    continue  # 跌停卖不出
                part = (h["shares"] * 100 * sell_px) / bar["amount"] if bar["amount"] > 0 else 1
                slip = IMPACT_K * np.sqrt(min(part, 1)) if friction else 0
                fill = sell_px * (1 - slip / 1e4) if friction else sell_px
                sell_shares = h["shares"] // 2 if reason == "tp1" else h["shares"]
                proceeds = sell_shares * 100 * fill
                comm = max(COMM_MIN, proceeds * COMM_RATE) if friction else 0
                stamp = proceeds * STAMP_RATE if friction else 0
                trans = proceeds * TRANSFER_RATE if friction else 0
                cash += proceeds - comm - stamp - trans
                if friction:
                    leaks["fee_cost"] += comm + stamp + trans
                    leaks["slippage_cost"] += slip / 1e4 * proceeds
                if reason == "tp1":
                    h["half"] = True
                    h["shares"] -= sell_shares
                else:
                    del holdings[code]

        # --- 记账 NAV ---
        pos_val = sum(h["shares"] * 100 * by_code[code].loc[t, "close"]
                      for code, h in holdings.items() if t in by_code[code].index)
        nav.append((t, cash + pos_val))

    nav = pd.DataFrame(nav, columns=["date", "nav"])
    return nav, leaks


def compute_breadth_stock(above_mat, t):
    """全市宽度(站上MA20占比)，O(1)列均值。"""
    if t not in above_mat.index:
        return None
    col = above_mat.loc[t].dropna()
    return float(col.mean() * 100) if len(col) else None


def metrics(nav):
    nav = nav.set_index("date")["nav"]
    ret = nav.iloc[-1] / nav.iloc[0] - 1
    days = (nav.index[-1] - nav.index[0]).days
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (365 / max(days, 1)) - 1
    peak = nav.cummax()
    dd = (nav / peak - 1).min()
    # 日收益
    dret = nav.pct_change().dropna()
    sharpe = dret.mean() / dret.std() * np.sqrt(242) if dret.std() > 0 else 0
    return dict(total=ret, ann=ann, maxdd=dd, sharpe=sharpe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--capital", type=float, default=8e4)
    ap.add_argument("--n-positions", type=int, default=6)
    ap.add_argument("--min-amount", type=float, default=3e7)
    ap.add_argument("--commission", type=float, default=0.00025,
                   help="佣金率(小数), 默认万2.5=0.00025; 万1填 0.0001")
    ap.add_argument("--min-commission", type=float, default=5.0,
                   help="最低佣金(元), 默认5; 无最低限制填 0")
    args = ap.parse_args()

    # 用命令行费率覆盖全局常量(单进程内生效)
    global COMM_RATE, COMM_MIN
    COMM_RATE = args.commission
    COMM_MIN = args.min_commission

    c = conn(); cur = c.cursor()
    idx = load_market(cur)
    end = pd.Timestamp(idx["date"].max())
    load_start = pd.Timestamp(args.start) - pd.Timedelta(days=400)  # 指标预热
    by_code, above_mat, trading_days = build_by_code(cur, load_start, end)
    trading_days = [d for d in trading_days if d >= pd.Timestamp(args.start)]
    cur.close(); c.close()

    print(f"账户 {args.capital:,.0f}元 | 持仓上限 {args.n_positions} 只 | 起点 {args.start} | 终点 {end.date()}")
    print(f"交易日 {len(trading_days)} 个 | 样本股票 {len(by_code)} 只\n")

    nav_ideal, _ = run_backtest(by_code, above_mat, trading_days, idx, args.start, args.capital,
                                args.n_positions, args.min_amount, friction=False)
    nav_real, leaks = run_backtest(by_code, above_mat, trading_days, idx, args.start, args.capital,
                                   args.n_positions, args.min_amount, friction=True)

    mi = metrics(nav_ideal); mr = metrics(nav_real)
    print("=" * 64)
    print("【收益对比】")
    print(f"  理想(无摩擦/涨停跌停照成交) : 总 {mi['total']*100:+.1f}%  年化 {mi['ann']*100:+.1f}%  回撤 {mi['maxdd']*100:.1f}%  Sharpe {mi['sharpe']:.2f}")
    print(f"  真实(全摩擦+不可成交)       : 总 {mr['total']*100:+.1f}%  年化 {mr['ann']*100:+.1f}%  回撤 {mr['maxdd']*100:.1f}%  Sharpe {mr['sharpe']:.2f}")
    leak_total = nav_ideal["nav"].iloc[-1] / nav_real["nav"].iloc[-1] - 1
    print(f"  → 总泄漏(理想/真实-1)        : {leak_total*100:+.1f}%")
    print("=" * 64)
    print("【泄漏拆解】")
    print(f"  涨停漏买       : {leaks['limit_up_miss']:>5d} 次 (占尝试 {leaks['trades_attempted']} 次)")
    print(f"  跌停卖不出     : {leaks['limit_down_hold']:>5d} 次 (持仓被锁)")
    print(f"  流动性截断金额 : {leaks['liq_trunc_notional']:>12,.0f} 元 (单票>日成交{LIQ_CAP*100:.0f}%)")
    print(f"  滑点成本       : {leaks['slippage_cost']:>12,.0f} 元")
    print(f"  手续费成本     : {leaks['fee_cost']:>12,.0f} 元 (佣金+印花税+过户费)")
    print(f"  停牌交易日     : {leaks['suspension_days']:>5d} 天")
    print(f"  实际成交笔数   : {leaks['trades_filled']} / 尝试 {leaks['trades_attempted']}")
    print("=" * 64)
    print("说明: 此前所有回测(含 strategy_pnl_ratio/etf_backtest)默认『开盘/收盘必成交』")
    print("     未计 涨停买不进/跌停卖不出/流动性截断/滑点/最低5元佣金 → 结果偏乐观。")

    nav_real.to_csv("data/results/realistic_nav_real.csv", index=False)
    nav_ideal.to_csv("data/results/realistic_nav_ideal.csv", index=False)
    with open("data/results/realistic_backtest_report.txt", "w", encoding="utf-8") as f:
        f.write(f"===== 5步框架真实摩擦回测报告 =====\n")
        f.write(f"账户 {args.capital:,.0f}元 | 持仓上限 {args.n_positions} 只 | 流动性门槛 {args.min_amount/1e4:.0f}万 | {args.start}~{end.date()}\n")
        f.write(f"交易日 {len(trading_days)} 个 | 样本 {len(by_code)} 只\n\n")
        f.write(f"理想(无摩擦/涨停跌停照成交): 总 {mi['total']*100:+.1f}%  年化 {mi['ann']*100:+.1f}%  回撤 {mi['maxdd']*100:.1f}%  Sharpe {mi['sharpe']:.2f}\n")
        f.write(f"真实(全摩擦+不可成交)      : 总 {mr['total']*100:+.1f}%  年化 {mr['ann']*100:+.1f}%  回撤 {mr['maxdd']*100:.1f}%  Sharpe {mr['sharpe']:.2f}\n")
        f.write(f"总泄漏(理想/真实-1)        : {leak_total*100:+.1f}%\n\n")
        f.write(f"泄漏拆解:\n")
        f.write(f"  涨停漏买       : {leaks['limit_up_miss']} 次 (占尝试 {leaks['trades_attempted']})\n")
        f.write(f"  跌停卖不出     : {leaks['limit_down_hold']} 次 (持仓被锁)\n")
        f.write(f"  流动性截断金额 : {leaks['liq_trunc_notional']:,.0f} 元 (单票>日成交{LIQ_CAP*100:.0f}%)\n")
        f.write(f"  滑点成本       : {leaks['slippage_cost']:,.0f} 元\n")
        f.write(f"  手续费成本     : {leaks['fee_cost']:,.0f} 元 (佣金+印花税+过户费)\n")
        f.write(f"  停牌交易日     : {leaks['suspension_days']} 天\n")
        f.write(f"  实际成交       : {leaks['trades_filled']} / 尝试 {leaks['trades_attempted']}\n")
        f.write(f"\n说明: 此前 strategy_pnl_ratio/etf_backtest 默认『开盘必成交、无摩擦』，未计涨停买不进/\n")
        f.write(f"      跌停卖不出/流动性截断/滑点/最低5元佣金 → 结果系统性偏乐观。\n")


if __name__ == "__main__":
    main()
