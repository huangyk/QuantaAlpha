"""交易计划生成器（CLI，可下单、含买点/止损/止盈/仓位）

把 trading_checklist 的「5 步框架」落成一份能照着下单的计划：
    1 看大盘  → 定环境(强/弱/震荡)与总仓位
    2 选个股  → 流动性/次新/涨停过滤 + 20日动量排序候选池
    3 看位置  → 价 vs MA20/MA60、距高低 → 决定买点(市价附近 / 回踩MA20)
    4 筹码    → CYQ 成本分布做上下文(获利盘/密集区)
    5 止盈    → 尾随止损(MA20下/5日低/≤15%) + 两档止盈(+10% / 20日高) + R:R
    + 仓位    → 按账户资金(默认8万)算每只下单手数与占比，弱市自动降仓

复用 trading_checklist.py 的数据源与已验证口径（5m VWAP×100、CYQ 重建等）。

口径限制（务必看清）：
    - 创业板指(399006)在 bars_1d 为空，大盘环境只用 沪深300/中证500/中证1000/上证。
    - 筹码分布(CYQ)为收盘价+成交量累加的近似，非券商真实持仓成本。
    - 分时痕迹仅 bars_5m(2026 年起)有数据。
    - 本计划按用户「5 步框架」生成，是「可执行清单」而非「已历史验证盈利的策略」：
      我们没有跑过「信号日→买入→止损止盈闭环」的回测，故其胜率/期望收益未经证实。
      它的最大价值是「纪律化」——把模糊的盘感变成可核对、可执行的价位与仓位。

用法：
    python tools/make_trading_plan.py                       # 默认 8万、6 只候选
    python tools/make_trading_plan.py --capital 8e4 --n-positions 5
    python tools/make_trading_plan.py --watch 600000.SH 000001.SZ
    python tools/make_trading_plan.py --min-amount 5e7 --no-5m
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trading_checklist import (  # noqa: E402
    conn, sql, get_latest_date, INDICES,
    step1_market, step2_universe, fetch_history,
    step3_position, step4_chips, step4_intraday,
)


# ─────────────────────── 大盘定调 → 总仓位 ───────────────────────
def market_posture(env, breadth):
    strong = sum(1 for e in env.values() if e["label"] == "强")
    weak = sum(1 for e in env.values() if e["label"] == "弱")
    br = breadth if pd.notna(breadth) else 50.0
    if strong >= 3 and br >= 60:
        return "强", 0.80, "指数多头、宽度扩张，可重仓"
    if weak >= 3 and br < 40:
        return "弱", 0.50, "指数空头、宽度萎缩，控仓防守"
    return "震荡", 0.60, "方向不明、结构分化，中等仓位"


# ─────────────────────── 买点触发价 ───────────────────────
def step_buy(position):
    out = {}
    for code, p in position.items():
        pct = p["pct_ma20"]
        if pct <= 10:
            buy = round(p["close"] * 0.995, 2)          # 已在支撑区，略低于现价挂单
            typ = "市价附近(已在支撑区)"
        else:
            buy = round(p["ma20"], 2)                   # 远离支撑，等回踩 MA20
            typ = f"回踩MA20(现距MA20 {pct:+.1f}%)"
        out[code] = dict(buy=buy, typ=typ)
    return out


# ─────────────────────── 止损 / 两档止盈 ───────────────────────
def plan_exit(position, buy):
    out = {}
    for code, p in position.items():
        b = buy[code]["buy"]
        floor = b * (1 - 0.15)
        stop = min(p["support_ma20"] * 0.99, p["low5"] * 0.999, floor)
        stop = min(stop, b * 0.999)                     # 必须低于买点
        stop = round(stop, 2)
        t1 = round(b * 1.10, 2)                         # 第一目标 +10%
        t2 = round(max(p["resist_hi20"], b * 1.10), 2)  # 第二目标 20日高
        risk = b - stop
        r1, r2 = t1 - b, t2 - b
        rr = (r2 / risk) if risk > 0 else float("nan")
        out[code] = dict(
            buy=b, stop=stop, t1=t1, t2=t2,
            risk_pct=risk / b * 100, r1_pct=r1 / b * 100, r2_pct=r2 / b * 100,
            rr=rr, wait_for_pullback=(b > p["close"] * 1.001),  # 买点高于现价=需等回踩
        )
    return out


# ─────────────────────── 选股重排：位置优先 + 动量正 ───────────────────────
def refine_universe(uni, position, n):
    """弱市里纯动量最强=已暴涨=远离支撑(难买)。改为优先「位置好(贴近MA20)+动量正」的票，
    让买点尽量=市价附近、可立即执行；超买的退为次选。"""
    rows = []
    for code in uni.index:
        if code not in position:
            continue
        mom = uni.loc[code, "ret20"] if code in uni.index else 0
        if mom <= 0:
            continue
        rows.append((code, position[code]["pct_ma20"], mom))
    rows.sort(key=lambda x: (x[1], -x[2]))   # 位置好(距MA20小)优先，动量次
    return [c for c, _, _ in rows[:n]]


# ─────────────────────── 仓位(账户资金→手数) ───────────────────────
def plan_sizes(codes, ex, capital, pos_pct):
    plan_cap = capital * pos_pct
    n = max(1, len(codes))
    per = plan_cap / n
    sizes, total_actual = {}, 0.0
    for code in codes:
        b = ex[code]["buy"]
        lots = max(1, int(per // (b * 100)))
        actual = lots * 100 * b
        total_actual += actual
        sizes[code] = dict(lots=lots, actual=actual, per=per, small=(actual < 5000))
    return sizes, plan_cap, total_actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--capital", type=float, default=8e4, help="账户资金(元)，默认8万")
    ap.add_argument("--n-positions", type=int, default=6, help="候选/持仓只数")
    ap.add_argument("--min-amount", type=float, default=3e7, help="日均成交额下限(元)")
    ap.add_argument("--watch", nargs="*", default=None)
    ap.add_argument("--no-5m", action="store_true")
    args = ap.parse_args()

    out_path = None
    try:
        res_dir = Path("data/results")
        res_dir.mkdir(parents=True, exist_ok=True)
        out_path = res_dir / f"trading_plan_{args.as_of or 'latest'}.txt"

        class Tee:
            def __init__(self, f):
                self.f = f; self._o = sys.stdout
            def write(self, t):
                self._o.write(t); self.f.write(t)
            def flush(self):
                self._o.flush(); self.f.flush()
        sys.stdout = Tee(open(out_path, "w", encoding="utf-8"))
    except Exception:
        pass

    c = conn(); cur = c.cursor()
    as_of = (pd.Timestamp(args.as_of) if args.as_of
             else pd.Timestamp(get_latest_date(cur)))
    cap = args.capital
    print(f"═══ 交易计划 ═══  as-of: {as_of.date()}  账户资金: {cap/1e4:.0f}万  ═══\n")

    # Step 1 大盘 → 仓位
    env, breadth = step1_market(cur, as_of)
    posture, pos_pct, note = market_posture(env, breadth)
    print("【大盘定调 / 总仓位】")
    for name, e in env.items():
        print(f"  {name:6s} {e['label']}  收{e['close']:.2f}  "
              f"vsMA20{(e['close']/e['ma20']-1)*100:+.1f}%  20日{e['ret20']*100:+.1f}%")
    print(f"  全市宽度(站上MA20): {breadth:.1f}%" if pd.notna(breadth) else "  全市宽度: n/a")
    print(f"  → 环境【{posture}】总仓位 {pos_pct*100:.0f}%（{cap*pos_pct/1e4:.1f}万）  {note}\n")

    # Step 2 候选
    if args.watch:
        codes = [x.upper() for x in args.watch]
        _pos = None
        print(f"【选股】--watch 模式，指定 {len(codes)} 只：{', '.join(codes)}\n")
    else:
        uni = step2_universe(cur, as_of, args.min_amount, max(args.n_positions * 2, 12))
        if uni.empty:
            print("【选股】无候选（流动性/次新/涨停过滤后为空）\n")
            cur.close(); c.close()
            print(f"\n计划已保存: {out_path}"); sys.stdout = sys.__stdout__; return
        # 先取历史算位置，按「位置优先(贴近支撑)+动量正」重排，挑可执行的买点
        _hist = fetch_history(cur, list(uni.index), as_of, max(255, 130))
        _pos = step3_position(_hist, as_of)
        codes = refine_universe(uni, _pos, args.n_positions)
        if not codes:
            print("【选股】无「动量为正且位置可用」的候选（弱市多为超买，建议观望）\n")
            cur.close(); c.close()
            print(f"\n计划已保存: {out_path}"); sys.stdout = sys.__stdout__; return
        print(f"【选股】候选 {len(codes)} 只（流动性≥{args.min_amount/1e4:.0f}万, 剔除次新/涨停, "
              f"位置优先+动量正）:\n")

    # Step 3/4
    hist = fetch_history(cur, codes, as_of, max(255, 130))
    position = step3_position(hist, as_of)
    chips = step4_chips(hist, as_of)
    intraday = {} if args.no_5m else step4_intraday(cur, codes, as_of)
    codes = [x for x in codes if x in position]          # 仅保留有完整历史的
    if not codes:
        print("  候选股历史不足60日，无法生成计划。\n")
        cur.close(); c.close()
        print(f"\n计划已保存: {out_path}"); sys.stdout = sys.__stdout__; return

    buy = step_buy(position)
    ex = plan_exit(position, buy)
    sizes, plan_cap, total_actual = plan_sizes(codes, ex, cap, pos_pct)

    print(f"  总计划仓位 {plan_cap/1e4:.1f}万 / {len(codes)} 只 ≈ 每只 {plan_cap/len(codes)/1e4:.2f}万\n")
    print("  代码        现价    买点      止损      止盈1(+10%) 止盈2(20日高)  R:R   手数  实投    占比")
    print("  " + "-" * 92)
    for code in codes:
        p = position[code]; b = buy[code]; e = ex[code]; s = sizes[code]
        flag = " ⏳等回踩" if e["wait_for_pullback"] else ""
        small = " ⚠单笔<5k" if s["small"] else ""
        print(f"  {code:10s} {p['close']:7.2f} {b['buy']:7.2f}  {e['stop']:7.2f}  "
              f"{e['t1']:7.2f}      {e['t2']:7.2f}       {e['rr']:.2f}  "
              f"{s['lots']:3d}手 {s['actual']/1e4:5.2f}万 {s['actual']/cap*100:4.1f}%{flag}{small}")

    print()
    print("【个股明细：位置 / 筹码 / 分时】\n")
    for code in codes:
        p = position[code]; b = buy[code]; e = ex[code]; ch = chips.get(code, {}); idy = intraday.get(code, {})
        print(f"  ▶ {code}  买点{b['buy']:.2f}({b['typ']})")
        print(f"    位置: MA20{p['pct_ma20']:+.1f}%  MA60{p['pct_ma60']:+.1f}%  "
              f"距20日高{p['pct_from_hi20']:+.1f}%  距52周高{p['pct_from_hi252']:+.1f}%")
        if ch:
            print(f"    筹码: 获利盘{ch['profit_ratio']:.0f}%  平均成本{ch['avg_cost']:.2f}  "
                  f"90%成本[{ch['cost90_low']:.2f},{ch['cost90_high']:.2f}]  峰{ch['peak']:.2f}")
        if idy:
            print(f"    分时: 最大量能{idy['heaviest_bucket']}  收-VWAP{idy['vs_vwap']:+.0f}bp  尾盘30分{idy['eod_ret']:+.2f}%")
        print(f"    纪律: 买点{b['buy']:.2f} → 止损{e['stop']:.2f}(-{e['risk_pct']:.1f}%)  "
              f"止盈1 {e['t1']:.2f}(+{e['r1_pct']:.1f}%) / 止盈2 {e['t2']:.2f}(+{e['r2_pct']:.1f}%)  R:R={e['rr']:.2f}")
        if e["wait_for_pullback"]:
            print("    ⚠ 买点高于现价，需等回踩至买点再挂单，不到不买。")
        if pd.notna(e["rr"]) and e["rr"] < 1:
            print("    ⚠ R:R<1，风险收益不划算，谨慎或放弃。")
        print()

    print("【总纪律】")
    print(f"  1. 总仓位 {pos_pct*100:.0f}%（{plan_cap/1e4:.1f}万），现金留 {cap-plan_cap:.0f}元应对回踩加仓/补仓。")
    print(f"  2. 到点才买：买点高于现价的票必须等回踩，禁止追高。")
    print(f"  3. 破位即走：收盘跌破止损价→次日开盘无条件止损，不摊平、不侥幸。")
    print(f"  4. 止盈分步：到止盈1 减仓 1/2 锁利，余仓用移动止损跟到止盈2 或破位。")
    print(f"  5. 单笔<5000元会被最低5元佣金吃掉(≈0.1%/笔)，已标⚠的票建议合并或减少只数。")
    print(f"  6. 本计划按你「5步框架」生成，未经历史回测验证胜率；先小资金跑、记录每笔对错再迭代。")

    n_wait = sum(1 for code in codes if ex[code]["wait_for_pullback"])
    if n_wait == len(codes):
        print(f"\n  ⚠ 全部 {len(codes)} 只候选都需大幅回踩才到买点 → 当前无舒适买点，建议观望、仅留现金。")
    elif n_wait:
        print(f"\n  ※ {n_wait}/{len(codes)} 只需等回踩，余 {len(codes)-n_wait} 只可市价附近买入。")
    print(f"\n  实际下单合计: {total_actual/1e4:.2f}万（计划 {plan_cap/1e4:.1f}万）")

    cur.close(); c.close()
    print(f"\n计划已保存: {out_path}")
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
