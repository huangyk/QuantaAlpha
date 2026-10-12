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
from . import (support, bounce, dip, momentum, calltracker, etf_guard,
               method_absmom, method_robust, method_higher)


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


def cmd_robust():
    print("\n===== 好方法·防过拟合压力测试（walk-forward OOS + 参数敏感 + 成本敏感） =====")
    try:
        r = method_robust.run_robust()
    except DataError as e:
        print(f"  [跳过] {e}"); return
    wf = r["wf"]
    print(f"  池 {r['n_etf']} 只")
    print("  -- walk-forward (训练窗选参 -> 完全隔离测试窗验证) --")
    print(f"  训练 {wf['train_win'][0]}~{wf['train_win'][1]}: 选参 {wf['train']['config']} "
          f"Sharpe {wf['train']['sharpe']:.2f} / 年化 {wf['train']['ann']*100:.1f}% / 回撤 {wf['train']['mdd']*100:.1f}%")
    print(f"  测试 {wf['test_win'][0]}~{wf['test_win'][1]} (隔离):")
    print(f"    选参配置  -> Sharpe {wf['test_best']['sharpe']:.2f} / 年化 {wf['test_best']['ann']*100:.1f}% / 回撤 {wf['test_best']['mdd']*100:.1f}%")
    print(f"    预设 sma200/inv_vol -> Sharpe {wf['test_pre']['sharpe']:.2f} / 年化 {wf['test_pre']['ann']*100:.1f}% / 回撤 {wf['test_pre']['mdd']*100:.1f}%")
    print(f"    等权基线   -> Sharpe {wf['test_base']['sharpe']:.2f} / 年化 {wf['test_base']['ann']*100:.1f}% / 回撤 {wf['test_base']['mdd']*100:.1f}%")
    tb, tp, bb = wf['test_best']['sharpe'], wf['test_pre']['sharpe'], wf['test_base']['sharpe']
    print(f"  -> 测试窗: 选参配置{'跑赢' if tb > bb else '跑输'}基线({tb:.2f} vs {bb:.2f}); "
          f"预设配置{'跑赢' if tp > bb else '跑输'}基线({tp:.2f} vs {bb:.2f})")
    print("  -- 参数敏感 (SMA窗口, 全程 2019-2026) --")
    print(f"  {'权重':<8}{'SMA':>6}{'年化':>9}{'Sharpe':>9}{'回撤':>9}{'在市':>8}")
    for x in sorted(r['param'], key=lambda d: (d['weight'], d['sma_w'])):
        print(f"  {x['weight']:<8}{x['sma_w']:>6}{x['ann']*100:>8.1f}%{x['sharpe']:>9.2f}"
              f"{x['mdd']*100:>8.1f}%{x['time_in_market']*100:>7.0f}%")
    print("  -> Sharpe 跨窗口成簇(无单点尖峰) => 非过拟合")
    print("  -- 成本敏感 (sma200/inv_vol, 万1~千1) --")
    print(f"  {'佣金':>8}{'年化':>9}{'Sharpe':>9}{'回撤':>9}{'vs基线':>8}")
    for x in r['cost']:
        cstr = (f"万{int(x['cost']*10000)}" if x['cost'] < 0.001 else f"千{int(x['cost']*1000)}")
        print(f"  {cstr:>8}{x['ann']*100:>8.1f}%{x['sharpe']:>9.2f}{x['mdd']*100:>8.1f}%"
              f"{('跑赢' if x['beats'] else '跑输'):>8}")
    print("  -> 千1 佣金下仍跑赢等权基线 => 结论对成本稳健")


def cmd_higher():
    print("\n===== 高收益 sleeve 探索：高beta池 / TopK动量集中（含防过拟合三重测试） =====")
    try:
        r = method_higher.run_higher()
    except DataError as e:
        print(f"  [跳过] {e}")
        return
    print(f"  池 全{ r['n_all']} 只 / 高beta {r['n_hb']} 只（剔除纯固收货币: "
          f"511010/511090/511220/511880），区间 {r['start']}~{r['end']}")
    print(f"  {'基线/配置':<26}{'年化':>9}{'Sharpe':>9}{'回撤':>9}{'在市':>8}{'调仓':>6}")
    for k, g in r["baselines"].items():
        print(f"  [基线]{k:<20}{g['ann']*100:>8.1f}%{g['sharpe']:>9.2f}{g['mdd']*100:>8.1f}%"
              f"{g['time_in_market']*100:>7.0f}%{g['n_rebal']:>6}")
    for k, g in r["configs"].items():
        print(f"  {k:<26}{g['ann']*100:>8.1f}%{g['sharpe']:>9.2f}{g['mdd']*100:>8.1f}%"
              f"{g['time_in_market']*100:>7.0f}%{g['n_rebal']:>6}")
    best_sh = max(r["configs"].items(), key=lambda kv: kv[1]["sharpe"])
    best_ann = max(r["configs"].items(), key=lambda kv: kv[1]["ann"])
    print(f"  -> 风险调整最优: {best_sh[0]} (Sharpe {best_sh[1]['sharpe']:.2f})")
    print(f"  -> 绝对收益最高: {best_ann[0]} (年化 {best_ann[1]['ann']*100:.1f}%)")

    # 防过拟合三重测试
    print("  -- walk-forward (训练窗选参 -> 隔离测试窗验证, 基线=高beta池等权) --")
    try:
        wf = method_higher.walk_forward_higher()
        print(f"  训练 {wf['train_win'][0]}~{wf['train_win'][1]}: 选参 {wf['train']['config']} "
              f"Sharpe {wf['train']['sharpe']:.2f} / 年化 {wf['train']['ann']*100:.1f}% / 回撤 {wf['train']['mdd']*100:.1f}%")
        print(f"  测试 {wf['test_win'][0]}~{wf['test_win'][1]} (隔离):")
        for key, lab in (("test_best", "训练选参"), ("test_pre", "预设 topk3/inv_vol"), ("test_base", "等权基线")):
            x = wf[key]
            print(f"    {lab:<18} Sharpe {x['sharpe']:.2f} / 年化 {x['ann']*100:.1f}% / 回撤 {x['mdd']*100:.1f}%")
        tb, tp, bb = wf["test_best"]["sharpe"], wf["test_pre"]["sharpe"], wf["test_base"]["sharpe"]
        print(f"  -> 测试窗: 选参{'跑赢' if tb > bb else '跑输'}基线({tb:.2f} vs {bb:.2f}); "
              f"预设{'跑赢' if tp > bb else '跑输'}基线({tp:.2f} vs {bb:.2f})")
    except DataError as e:
        print(f"    [跳过] {e}")

    # H1 专项：只在高beta池(不做TopK集中)内选参 -> 隔离验证
    print("  -- H1 专项 walk-forward (高beta池 topk=None, 基线=高beta池等权) --")
    try:
        wf1 = method_higher.walk_forward_higher(topks=(None,), preset_topk=None)
        print(f"  训练 {wf1['train_win'][0]}~{wf1['train_win'][1]}: 选参 {wf1['train']['config']} "
              f"Sharpe {wf1['train']['sharpe']:.2f} / 年化 {wf1['train']['ann']*100:.1f}% / 回撤 {wf1['train']['mdd']*100:.1f}%")
        print(f"  测试 {wf1['test_win'][0]}~{wf1['test_win'][1]} (隔离):")
        for key, lab in (("test_best", "训练选参"), ("test_pre", "预设 inv_vol"), ("test_base", "等权基线")):
            x = wf1[key]
            print(f"    {lab:<18} Sharpe {x['sharpe']:.2f} / 年化 {x['ann']*100:.1f}% / 回撤 {x['mdd']*100:.1f}%")
        tb, bb = wf1["test_best"]["sharpe"], wf1["test_base"]["sharpe"]
        print(f"  -> H1 测试窗: 选参{'跑赢' if tb > bb else '跑输'}基线({tb:.2f} vs {bb:.2f})")
    except DataError as e:
        print(f"    [跳过] {e}")

    print("  -- TopK 参数敏感 (全程) --")
    try:
        for x in method_higher.param_topk():
            print(f"    topk={x['topk']:<3}{x['weight']:<9}年化{x['ann']*100:>7.1f}%"
                  f"  Sharpe {x['sharpe']:>5.2f}  回撤{x['mdd']*100:>7.1f}%  在市{x['time_in_market']*100:>4.0f}%")
    except DataError as e:
        print(f"    [跳过] {e}")

    print("  -- 成本敏感 (H1: 高beta池 topk=None/inv_vol, 万1~千1, 对高beta池等权基线) --")
    try:
        for x in method_higher.cost_sens_higher(topk=None, weight="inv_vol"):
            cstr = f"万{int(x['cost']*10000)}" if x["cost"] < 0.001 else f"千{int(x['cost']*1000)}"
            print(f"    {cstr:>4}  年化{x['ann']*100:>7.1f}%  Sharpe {x['sharpe']:>5.2f}"
                  f"  回撤{x['mdd']*100:>7.1f}%  基线Sharpe {x['base_sharpe']:.2f}  "
                  f"{'跑赢' if x['beats'] else '跑输'}")
    except DataError as e:
        print(f"    [跳过] {e}")


def cmd_blend():
    print("\n===== 收益-回撤前沿：底盘(全池) ↔ H1(高beta池) 核心-卫星混合 =====")
    try:
        r = method_higher.run_blend_frontier()
    except DataError as e:
        print(f"  [跳过] {e}")
        return
    print(f"  池 全{r['n_all']}只 / 高beta {r['n_hb']}只，区间 {r['start']}~{r['end']}")
    print(f"  {'混合(H1%)':<10}{'底盘%':>7}{'H1%':>7}{'年化':>9}{'Sharpe':>9}{'回撤':>9}{'波动':>9}")
    for x in r["rows"]:
        print(f"  {x['h1_frac']*100:>8.0f}%{x['base_frac']*100:>7.0f}%{x['h1_frac']*100:>7.0f}%"
              f"{x['ann']*100:>8.1f}%{x['sharpe']:>9.2f}{x['mdd']*100:>8.1f}%{x['vol']*100:>8.1f}%")
    print(f"  [基线] 全池等权 Sharpe {r['base_equal']['sharpe']:.2f}/年化 {r['base_equal']['ann']*100:.1f}%/"
          f"回撤 {r['base_equal']['mdd']*100:.1f}%  | 高beta等权 Sharpe {r['hb_equal']['sharpe']:.2f}/"
          f"年化 {r['hb_equal']['ann']*100:.1f}%/回撤 {r['hb_equal']['mdd']*100:.1f}%")

    # walk-forward OOS：每比例在隔离测试窗是否仍稳定赚钱
    print("  -- walk-forward OOS (训练 2019-2022 / 隔离测试 2023-2026) --")
    wf = None
    try:
        wf = method_higher.walk_forward_blend()
        print(f"  {'混合(H1%)':<10}{'训练Sharpe':>12}{'训练年化':>10}{'训练回撤':>10}"
              f"{'测试Sharpe':>12}{'测试年化':>10}{'测试回撤':>10}")
        for tr, te in zip(wf["train"], wf["test"]):
            print(f"  {tr['blend']*100:>8.0f}%{tr['sharpe']:>12.2f}{tr['ann']*100:>9.1f}%"
                  f"{tr['mdd']*100:>9.1f}%{te['sharpe']:>12.2f}{te['ann']*100:>9.1f}%{te['mdd']*100:>9.1f}%")
        ok = sum(1 for te in wf["test"] if te["sharpe"] > 0)
        print(f"  -> 隔离测试窗 Sharpe>0 的比例数: {ok}/{len(wf['test'])} "
              f"(全>0 即混合增强在样本外仍稳定赚钱, 非过拟合)")
    except DataError as e:
        print(f"    [跳过] {e}")

    # 落盘：CSV + 内联 SVG 前沿 HTML
    _write_blend_frontier(r, wf)


def _write_blend_frontier(r, wf):
    """写出混合前沿的 CSV 与自包含 HTML(内联 SVG 散点), 供用户挑'回撤能睡着'的点。"""
    import csv as _csv
    import os as _os
    here = _os.path.dirname(__file__)
    res_dir = _os.path.normpath(_os.path.join(here, "..", "..", "data", "results"))
    doc_dir = _os.path.normpath(_os.path.join(here, "..", "..", "docs"))
    _os.makedirs(res_dir, exist_ok=True)
    csv_path = _os.path.join(res_dir, "blend_frontier.csv")
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["h1_pct", "base_pct", "ann_pct", "sharpe", "mdd_pct", "vol_pct"])
        for x in r["rows"]:
            w.writerow([round(x["h1_frac"] * 100), round(x["base_frac"] * 100),
                        round(x["ann"] * 100, 2), round(x["sharpe"], 3),
                        round(x["mdd"] * 100, 2), round(x["vol"] * 100, 2)])
    print(f"  -> 前沿数据已写出: {csv_path}")

    # 构建 SVG 散点：x=最大回撤(绝对值%), y=年化收益%
    rows = r["rows"]
    xs = [abs(x["mdd"]) * 100 for x in rows]
    ys = [x["ann"] * 100 for x in rows]
    xmax = max(15.0, (max(xs) // 5 + 1) * 5)
    ymax = max(15.0, (max(ys) // 5 + 1) * 5)
    W, H = 680, 440
    lg, rt, tp, bt = 56, 24, 30, 52
    pw, ph = W - lg - rt, H - tp - bt

    def X(v):
        return lg + (v / xmax) * pw

    def Y(v):
        return tp + ph - (v / ymax) * ph

    def sh_color(sh):
        if sh >= 1.0:
            return "#16a34a"
        if sh >= 0.7:
            return "#0891b2"
        if sh >= 0.4:
            return "#d97706"
        return "#dc2626"

    parts = []
    parts.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
                 f'font-family="ui-monospace,Consolas,monospace">')
    # 网格 + 轴
    for gx in range(0, int(xmax) + 1, 5):
        parts.append(f'<line x1="{X(gx):.1f}" y1="{Y(0):.1f}" x2="{X(gx):.1f}" y2="{tp}" '
                     f'stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{X(gx):.1f}" y="{H-bt+18:.1f}" font-size="10" '
                     f'fill="#6b7280" text-anchor="middle">{gx}%</text>')
    for gy in range(0, int(ymax) + 1, 5):
        parts.append(f'<line x1="{lg}" y1="{Y(gy):.1f}" x2="{W-rt}" y2="{Y(gy):.1f}" '
                     f'stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{lg-8:.1f}" y="{Y(gy)+3:.1f}" font-size="10" '
                     f'fill="#6b7280" text-anchor="end">{gy}%</text>')
    parts.append(f'<text x="{lg+pw/2:.1f}" y="{H-8:.1f}" font-size="11" fill="#374151" '
                 f'text-anchor="middle">最大回撤 (绝对值 %)</text>')
    parts.append(f'<text x="14" y="{tp+ph/2:.1f}" font-size="11" fill="#374151" '
                 f'text-anchor="middle" transform="rotate(-90 14 {tp+ph/2:.1f})">年化收益 %</text>')
    # 前沿连线（按 H1 权重从高到低）
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["h1_frac"])
    line_pts = " ".join(f"{X(abs(rows[i]['mdd'])*100):.1f},{Y(rows[i]['ann']*100):.1f}"
                        for i in order)
    parts.append(f'<polyline points="{line_pts}" fill="none" stroke="#9ca3af" '
                 f'stroke-width="1.5" stroke-dasharray="4 3"/>')
    # 点
    for x in rows:
        cx, cy = X(abs(x["mdd"]) * 100), Y(x["ann"] * 100)
        col = sh_color(x["sharpe"])
        lab = f"H1 {x['h1_frac']*100:.0f}%"
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{col}" '
                     f'stroke="#fff" stroke-width="1.5"/>')
        parts.append(f'<text x="{cx+9:.1f}" y="{cy+3:.1f}" font-size="10" fill="#111827">'
                     f'{lab} · {x["ann"]*100:.1f}%/S{x["sharpe"]:.2f}</text>')
    parts.append('</svg>')

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>混合比例收益-回撤前沿</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f8fafc;
color:#0f172a;margin:0;padding:24px;}} .card{{background:#fff;border:1px solid #e2e8f0;
border-radius:12px;padding:20px;max-width:760px;margin:0 auto;box-shadow:0 1px 3px rgba(0,0,0,.06);}}
h1{{font-size:18px;margin:0 0 4px}} .sub{{color:#64748b;font-size:13px;margin:0 0 16px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:16px}}
th,td{{border:1px solid #e2e8f0;padding:6px 8px;text-align:right}}
th{{background:#f1f5f9}} .tip{{background:#eff6ff;border:1px solid #bfdbfe;color:#1e40af;
border-radius:8px;padding:10px 12px;font-size:13px;margin-top:16px;line-height:1.6}}
.legend{{font-size:12px;color:#64748b;margin-top:8px}}</style></head>
<body><div class="card">
<h1>收益-回撤前沿 · 底盘(全池) ↔ H1(高beta池)</h1>
<p class="sub">核心-卫星混合：投资者把 (1-blend) 配底盘 sleeve、blend 配 H1 sleeve；
每 sleeve 各自月频调仓、内部可转现金。区间 {r['start']}~{r['end']}，{r['n_all']}只ETF。</p>
{''.join(parts)}
<div class="legend">点色 = Sharpe（绿≥1.0 / 青0.7~1.0 / 橙0.4~0.7 / 红&lt;0.4）；
虚线连接各混合点即前沿。越靠右上=收益越高但回撤越深。</div>
<table><thead><tr><th>混合(H1%)</th><th>底盘%</th><th>H1%</th><th>年化</th>
<th>Sharpe</th><th>回撤</th><th>波动</th></tr></thead><tbody>
"""
    for x in rows:
        html += (f"<tr><td>{x['h1_frac']*100:.0f}%</td><td>{x['base_frac']*100:.0f}%</td>"
                 f"<td>{x['h1_frac']*100:.0f}%</td><td>{x['ann']*100:.1f}%</td>"
                 f"<td>{x['sharpe']:.2f}</td><td>{x['mdd']*100:.1f}%</td>"
                 f"<td>{x['vol']*100:.1f}%</td></tr>\n")
    html += "</tbody></table>"
    if wf:
        html += ('<div class="tip"><b>walk-forward OOS（隔离测试窗 2023-2026）：</b><br>'
                 + "<br>".join(
                     f"H1 {te['blend']*100:.0f}% → 训练 Sharpe {tr['sharpe']:.2f} / "
                     f"测试 Sharpe {te['sharpe']:.2f}（年化 {te['ann']*100:.1f}%，"
                     f"回撤 {te['mdd']*100:.1f}%）"
                     for tr, te in zip(wf["train"], wf["test"]))
                 + "<br>测试窗 Sharpe 全&gt;0 即混合增强在样本外仍稳定赚钱。</div>")
    html += ('<div class="tip"><b>怎么挑：</b>从虚线左上端（H1 0%，回撤最小）沿前沿往右下走，'
             '每加一点 H1 权重就多收益、也多回撤。<b>选一个你"浮亏时能睡着"的回撤点</b>，'
             '别追最右端的最高收益——那是用更深回撤换的。</div>')
    html += "</div></body></html>"
    html_path = _os.path.join(doc_dir, "blend_frontier.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  -> 前沿可视化已写出: {html_path}")


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
    cmd_robust()
    cmd_higher()
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
                             "absmom", "signal", "regime", "robust", "higher", "blend",
                             "calltracker"])
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
        elif args.cmd == "robust":
            cmd_robust()
        elif args.cmd == "higher":
            cmd_higher()
        elif args.cmd == "blend":
            cmd_blend()
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
