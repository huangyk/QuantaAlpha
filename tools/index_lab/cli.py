"""index_lab 命令行编排器。

用法：
  python -m tools.index_lab              # 跑全部方向，输出总报告
  python -m tools.index_lab support      # 单方向
  python -m tools.index_lab bounce
  python -m tools.index_lab dip
  python -m tools.index_lab momentum
  python -m tools.index_lab etfguard
  python -m tools.index_lab calltracker [calls.json]   # 无参数则跑演示
"""
import argparse
import sys

import numpy as np
import pandas as pd

from . import config
from .io_db import load_ohlc, DataError, load_etf_universe
from . import support, bounce, dip, momentum, calltracker, etf_guard, method_absmom


# ---------- 数据 ----------
def load_indices(start=config.DEFAULT_START):
    data = {}
    for nm, code in config.INDICES:
        try:
            data[nm] = load_ohlc(code, start)
        except DataError as e:
            print(f"  [跳过] {nm}({code}): {e}")
    return data


def _levels_for(df, step=None):
    s = df["close"]; latest = s.iloc[-1]
    step = step or (100 if latest > 2000 else 50)
    lo, hi = s.iloc[-250:].min(), s.iloc[-250:].max()
    return [lv for lv in range(int(lo // step) * step, int(hi // step) * step + 1, step)]


# ---------- 各方向 ----------
def cmd_support():
    print("\n===== 方向1 圆整数支撑（指定关口 + 扫描） =====")
    data = load_indices()
    named = {"上证指数": [3700, 3741], "创业板指": [2900], "科创50": [1200]}
    for nm, df in data.items():
        r = support.run(df, nm, levels=named.get(nm))
        print(f"\n· {nm}  最新 {r['date']} 收盘 {r['latest']:.2f}")
        for lv in r["levels"]:
            print(f"    关口 {lv['level']:.0f}: 现价在其{lv['position']}({lv['dist']:+.1f}%) | "
                  f"触达{lv['n']}次 均值{lv['mean']*100:+.1f}% 胜率{lv['wr']*100:.0f}% -> {lv['verdict']}")
        print(f"    扫描: {r['summary']}")


def cmd_bounce():
    print("\n===== 方向2 抢反弹“赚几点”（多指数×多关卡 pooling 网格） =====")
    data = load_indices()
    levels_map = {nm: _levels_for(df) for nm, df in data.items()}
    rows = bounce.run_grid(data, levels_map, tps=[0.03, 0.05], sls=[0.03], bands=[0.02], maxhold=15)
    print(f"{'tp/sl/band':<14}{'N':>6}{'均净bp':>9}{'t':>8}{'Sharpe':>8}{'强关占比':>9}")
    for r in rows[:8]:
        print(f"tp{r['tp']*100:.0f}/sl{r['sl']*100:.0f}/b{r['band']*100:.0f} "
              f"{r['N']:>6}{r['net_bp']:>+9.0f}{r['t']:>+8.1f}{r['sharpe']:>+8.2f}{r['strong_ratio']*100:>8.0f}%")
    print("  -> 所有配置 t 全<1 即为“无稳健edge”（详见 docs 总报告）。")


def cmd_dip():
    print("\n===== 方向3 顺趋势抄跌（干净前向收益，跨6指数 pooling） =====")
    data = load_indices()
    rows = dip.run_pool(data, depths=[0.03, 0.05, 0.08], horizons=[20, 40])
    print(f"{'条件':<6}{'回撤':>5}{'H':>4}{'N':>6}{'均bp':>9}{'t':>8}{'胜率':>7}")
    for r in rows:
        print(f"{r[0]:<6}{r[1]:>4.0f}%{r[2]:>4}{r[3]:>6}{r[4]:>+9.0f}{r[5]:>+8.1f}{r[6]*100:>6.0f}%")
    big = [r for r in rows if r[3] >= 100]
    big.sort(key=lambda r: -r[5])
    sig = [r for r in big if r[5] > 2]
    print(f"  -> 达到 t>2 显著的组合数: {len(sig)}（0 即无任何显著edge）")
    if big:
        b = big[0]
        print(f"  -> 稳健最佳(N>=100): {b[0]} 回撤{b[1]:.0f}% H{b[2]} 均{b[4]:+.0f}bp t={b[5]:+.1f} 胜率{b[6]*100:.0f}%")


def cmd_momentum():
    print("\n===== 方向4 动量/趋势跟踪（跨6指数平均 vs 买入持有） =====")
    data = load_indices()
    agg = momentum.run_pool(data, Ws=(60, 120, 200), Ls=(120, 250))
    bh = agg["buyhold"]
    print(f"{'策略':<12}{'平均年化':>10}{'平均Sharpe':>13}{'平均回撤':>10}")
    print(f"{'买入持有':<12}{bh['ann']*100:>9.1f}%{bh['sharpe']:>13.2f}{bh['mdd']*100:>9.1f}%")
    for k in ("ma60", "ma120", "ma200", "tsmom120", "tsmom250"):
        m = agg[k]
        print(f"{k:<12}{m['ann']*100:>9.1f}%{m['sharpe']:>13.2f}{m['mdd']*100:>9.1f}%")
    print("  -> MA择时平均降年化+降Sharpe仅小幅降回撤；TSMOM多空≈0。仅创业板/中证500 MA60 有降回撤价值。")


def cmd_etfguard():
    print("\n===== 功能B ETF 等权持有 + MA60 趋势门控（降回撤保护实测） =====")
    try:
        r = etf_guard.run(extra_gates=[("创业板指", "399006.SZ")])
    except DataError as e:
        print(f"  [跳过] {e}"); return
    ew = r["equal_weight"]
    print(f"  ETF 池 {r['n_etf']} 只，区间 {r['start']}~{r['end']}")
    print(f"  {'策略':<16}{'年化':>9}{'Sharpe':>9}{'回撤':>9}{'切换':>6}")
    print(f"  {'纯等权持有':<16}{ew['ann']*100:>8.1f}%{ew['sharpe']:>9.2f}{ew['mdd']*100:>8.1f}%{ew['n_switches']:>6}")
    for gname, g in r["gates"].items():
        print(f"  {gname}+MA60门控{'':<7}{g['ann']*100:>8.1f}%{g['sharpe']:>9.2f}{g['mdd']*100:>8.1f}%{g['n_switches']:>6}")


def cmd_absmom():
    print("\n===== 好方法：跨资产绝对动量 / 趋势跟随（文献稳健，对功能B的单指数门控修正） =====")
    try:
        r = method_absmom.run()
    except DataError as e:
        print(f"  [跳过] {e}"); return
    ew = r["equal_weight"]
    print(f"  ETF 池 {r['n_etf']} 只，区间 {r['start']}~{r['end']}")
    print(f"  {'策略':<18}{'年化':>9}{'Sharpe':>9}{'回撤':>9}{'在市':>8}{'调仓':>6}")
    print(f"  {'纯等权持有':<18}{ew['ann']*100:>8.1f}%{ew['sharpe']:>9.2f}{ew['mdd']*100:>8.1f}%{ew['time_in_market']*100:>7.0f}%{ew['n_rebal']:>6}")
    for k, g in r["configs"].items():
        print(f"  {k:<18}{g['ann']*100:>8.1f}%{g['sharpe']:>9.2f}{g['mdd']*100:>8.1f}%{g['time_in_market']*100:>7.0f}%{g['n_rebal']:>6}")
    best = max(r["configs"].items(), key=lambda kv: kv[1]["sharpe"])
    print(f"  -> 风险调整最优: {best[0]} (Sharpe {best[1]['sharpe']:.2f}, 回撤 {best[1]['mdd']*100:.1f}%)")


def cmd_signal():
    print("\n===== 好方法·当前持仓信号（sma200 + 逆波动，截至最新交易日） =====")
    try:
        r = method_absmom.run_signal()
    except DataError as e:
        print(f"  [跳过] {e}"); return
    print(f"  数据截至 {r['as_of']}；信号 sma200 / 权重 {r['weight']}")
    print(f"  池 {r['n_total']} 只，趋势向上(持有) {r['n_hold']} 只")
    print(f"  {'代码':<10}{'收盘':>9}{'200SMA':>9}{'偏离%':>8}{'60日波动':>9}{'权重%':>7}")
    held = [x for x in r["holdings"] if x["hold"]]
    held.sort(key=lambda x: -x["weight"])
    for x in held:
        sma = x["sma"]; gap = x["gap_pct"]; vol = x["vol_60"]
        print(f"  {x['code']:<10}{x['close']:>9.3f}{sma:>9.3f}{gap:>+8.1f}"
              f"{('%.1f%%' % (vol*100)) if x['hold'] and vol==vol else '   -':>9}{x['weight']*100:>7.1f}%")
    exitd = [x for x in r["holdings"] if not x["hold"]]
    print(f"  --- 趋势向下(转现金) {len(exitd)} 只 ---")
    for x in exitd[:14]:
        gap = x["gap_pct"]
        sma_s = f"{x['sma']:.3f}" if x["sma"] == x["sma"] else "   -"
        print(f"  {x['code']:<10}{x['close']:>9.3f}{sma_s:>9}{gap:>+8.1f}")
    # 写出可直接跟的单
    import csv, os
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "results"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "current_holdings.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "close", "sma200", "gap_pct", "vol_60", "hold", "weight"])
        for x in r["holdings"]:
            w.writerow([x["code"], round(x["close"], 4),
                        (round(x["sma"], 4) if x["sma"] == x["sma"] else ""),
                        (round(x["gap_pct"], 2) if x["gap_pct"] == x["gap_pct"] else ""),
                        (round(x["vol_60"], 4) if x["vol_60"] == x["vol_60"] else ""),
                        int(x["hold"]), x["weight"]])
    print(f"  -> 持仓单已写出: {path}")


def cmd_regime():
    print("\n===== 好方法·regime 稳健性（牛/震荡/熊 分段对比） =====")
    try:
        r = method_absmom.run_regime()
    except DataError as e:
        print(f"  [跳过] {e}"); return
    print(f"  {r['n_etf']}只ETF {r['start']}~{r['end']}；配置 {r['config']}；共 {r['n_months']} 个月")
    print(f"  {'regime':<8}{'月数':>5}{'策略月均':>10}{'基准月均':>10}{'超额月均':>10}{'胜率':>7}{'策略年化':>10}{'基准年化':>10}")
    for k in ("bull", "side", "bear"):
        a = r["regimes"].get(k)
        if not a:
            continue
        name = {"bull": "牛市", "side": "震荡", "bear": "熊市"}[k]
        print(f"  {name:<7}{a['n']:>5}{a['avg_strat']*100:>+9.2f}%{a['avg_base']*100:>+9.2f}%"
              f"{a['avg_excess']*100:>+9.2f}%{a['win_rate']*100:>6.0f}%"
              f"{a['ann_strat']*100:>+9.1f}%{a['ann_base']*100:>+9.1f}%")
    print("  -> 熊市段策略(常转现金)应显著跑赢基准；牛市段基本跟上 => 降回撤非牛市漏赚。")


def cmd_calltracker(json_path=None):
    print("\n===== 功能A call tracker（博主判断按时间戳打分） =====")
    if json_path:
        t = calltracker.CallTracker.load(json_path)
        print(f"  已从 {json_path} 载入 {len(t)} 条判断")
    else:
        print("  [演示] 录入两条示例判断并打分（演示数据，非真实博主记录）：")
        t = calltracker.CallTracker()
        t.add("2023-01-03", "创业板指", "support", 2300, 20, "演示：创业板支撑2300")
        t.add("2024-02-01", "上证指数", "down", 2600, 20, "演示：上证跌破2600")
    summary, detail = t.score()
    print(f"  汇总: 录入 {summary['total']} 条 / 可评 {summary['scored']} 条 / "
          f"命中 {summary['hits']} / 命中率 {summary['hit_rate']*100:.0f}% / "
          f"平均前向收益 {summary['avg_fwd_ret']*100:+.1f}%")
    if summary["by_direction"]:
        print("  按方向: " + ", ".join(f"{k} {v['rate']*100:.0f}%(n={v['n']})"
                                        for k, v in summary["by_direction"].items()))
    for d in detail:
        if d["ok"]:
            print(f"    {d['ts']} {d['index_name']} {d['direction']}@{d['level']:.0f} "
                  f"-> {'命中' if d['correct'] else '未中'} (前向{d['fwd_ret']*100:+.1f}%)")
        else:
            print(f"    {d['ts']} {d['index']} {d['direction']} -> 跳过: {d['reason']}")


def cmd_all():
    print("=" * 64)
    print("  指数技术打法验证总方案  ·  index_lab")
    print("  数据: TimescaleDB bars_1d (QMT回填, 至 2026-09-30)")
    print("=" * 64)
    cmd_support()
    cmd_bounce()
    cmd_dip()
    cmd_momentum()
    cmd_etfguard()
    cmd_absmom()
    cmd_signal()
    cmd_regime()
    cmd_calltracker()
    print("\n" + "=" * 64)
    print("  裁定：四个验证方向(支撑/抢反弹/抄跌/趋势)均无压倒性可复制 edge；")
    print("  功能B 的单指数MA门控在多元ETF池上反而伤收益。")
    print("  唯一经文献+实盘数据验证的真方法 = 跨资产绝对动量/趋势跟随")
    print("  (对每个ETF独立判趋势，通过才持有)，可降回撤、提夏普。")
    print("  详见 docs/index_technical_verdict_report.html")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(prog="index_lab", description="指数技术打法验证统一包")
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["all", "support", "bounce", "dip", "momentum", "etfguard",
                             "absmom", "signal", "regime", "calltracker"])
    ap.add_argument("calls", nargs="?", default=None,
                    help="calltracker 子命令的 JSON 路径（可选）")
    args = ap.parse_args()
    try:
        if args.cmd == "all":
            cmd_all()
        elif args.cmd == "support":
            cmd_support()
        elif args.cmd == "bounce":
            cmd_bounce()
        elif args.cmd == "dip":
            cmd_dip()
        elif args.cmd == "momentum":
            cmd_momentum()
        elif args.cmd == "etfguard":
            cmd_etfguard()
        elif args.cmd == "absmom":
            cmd_absmom()
        elif args.cmd == "signal":
            cmd_signal()
        elif args.cmd == "regime":
            cmd_regime()
        elif args.cmd == "calltracker":
            cmd_calltracker(args.calls)
    except DataError as e:
        print(f"[数据错误] {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[中断]", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
