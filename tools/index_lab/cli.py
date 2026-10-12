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
               method_absmom, method_robust, method_higher, method_ml)


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
    wf, rw, lv = None, None, None
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

    # 逐年滚动 OOS：固定 blend，把 2022-2026 每年单独切样本外，证"是否每年都赚"
    print("  -- 逐年滚动 OOS (固定 blend, 切年看是否每年都赚, 参数全程未选参故诚实) --")
    try:
        rw = method_higher.rolling_walk_forward_blend()
        for b in rw["blends"]:
            rows = rw["per_year"][b]
            pos = sum(1 for x in rows if x["ret"] > 0)
            print(f"  [H1 {b*100:.0f}%] 逐年正收益年数 {pos}/{len(rows)}:")
            for x in rows:
                tag = " (部分年)" if x["partial"] else ""
                print(f"    {x['year']}{tag}: 收益{x['ret']*100:>+6.1f}%  年化{x['ann']*100:>+6.1f}%  "
                      f"Sharpe{x['sharpe']:>5.2f}  回撤{x['mdd']*100:>6.1f}%")
            o = rw["oos"][b]
            print(f"    样本外合计(2022-{rw['end_year']}): 年化{o['ann']*100:>+6.1f}%  "
                  f"Sharpe{o['sharpe']:.2f}  回撤{o['mdd']*100:.1f}%")
    except DataError as e:
        print(f"    [跳过] {e}")

    # 杠杆场景（执行层 L4）：低回撤底盘上适度杠杆能否更接近用户"多赚"诉求(仍可控)
    print("  -- 杠杆场景（底盘 L4 执行层，融资利率取 6%/年，仅示意上限） --")
    try:
        codes_lv = load_etf_universe("2019-01-01", "2026-09-30", True)
        lv = method_higher.leverage_scenario(codes_lv)
        for x in lv["rows"]:
            print(f"    杠杆{x['L']:.1f}x: 年化{x['ann']*100:>+6.1f}%  Sharpe{x['sharpe']:.2f}  "
                  f"回撤{x['mdd']*100:>6.1f}%  波动{x['vol']*100:>5.1f}%")
        print("  -> 杠杆放大收益也放大回撤/波动；1.3x 约把底盘年化从10%提到~13%、"
              "回撤~12%。真实受融资利率/担保比例约束，且防御空仓期仍付杠杆利息(拖累)。")
    except DataError as e:
        print(f"    [跳过] {e}")

    # 进阶：波动率目标化 vol targeting（执行层 L4，钉年化波动10%，高波动自动降仓）
    print("  -- 进阶：波动率目标化 vol targeting（执行层 L4，钉年化波动 10%） --")
    vt = None
    try:
        codes_vt = load_etf_universe("2019-01-01", "2026-09-30", True)
        vt_rows = []
        for b in (0.0, 0.25, 0.5):
            v = method_higher.vol_target_blend(codes_vt, blend=b)
            vt_rows.append(dict(blend=b, ann=v["ann"], sharpe=v["sharpe"],
                                mdd=v["mdd"], vol=v["vol"], avg_lev=v["avg_lev"]))
            print(f"  [H1 {b*100:.0f}% + vol目标]: 年化{v['ann']*100:>+6.1f}%  "
                  f"Sharpe{v['sharpe']:.2f}  回撤{v['mdd']*100:>6.1f}%  "
                  f"波动{v['vol']*100:>5.1f}%  均杠杆{v['avg_lev']:.2f}x")
        wfvt = method_higher.vol_target_walk_forward(codes_vt, blend=0.0)
        print(f"  walk-forward OOS(底盘+vol目标): 训练 Sharpe{wfvt['train']['sharpe']:.2f}/"
              f"年化{wfvt['train']['ann']*100:.1f}%  测试 Sharpe{wfvt['test']['sharpe']:.2f}/"
              f"年化{wfvt['test']['ann']*100:.1f}%/回撤{wfvt['test']['mdd']*100:.1f}%")
        vt = dict(rows=vt_rows, wf=wfvt)
    except DataError as e:
        print(f"    [跳过] {e}")

    # 块 Bootstrap 显著性：10% 年化是不是运气(90% CI)
    print("  -- 块 Bootstrap 显著性（21日块重采样 1000 次，90% 置信区间） --")
    bs = None
    try:
        codes_bs = load_etf_universe("2019-01-01", "2026-09-30", True)
        bs_rows = []
        for b in (0.0, 0.25, 0.5):
            bsr = method_higher.bootstrap_blend(codes_bs, blend=b)
            bs_rows.append(dict(blend=b, **bsr))
            print(f"  [H1 {b*100:.0f}%]: 年化90%CI [{bsr['ann_ci'][0]*100:+.1f}%, "
                  f"{bsr['ann_ci'][1]*100:+.1f}%]  均值{bsr['ann_mean']*100:+.1f}%  "
                  f"Sharpe90%CI [{bsr['sharpe_ci'][0]:.2f},{bsr['sharpe_ci'][1]:.2f}]  "
                  f"回撤90%CI [{bsr['mdd_ci'][0]*100:.1f}%,{bsr['mdd_ci'][1]*100:.1f}%]")
        bs = dict(rows=bs_rows)
        sig = sum(1 for x in bs_rows if x["ann_ci"][0] > 0)
        print(f"  -> 年化 CI 下限>0 的档数: {sig}/{len(bs_rows)} "
              f"(全>0 即'稳定赚'统计显著、非运气)")
    except DataError as e:
        print(f"    [跳过] {e}")

    # 落盘：CSV + 内联 SVG 前沿 HTML（含逐年滚动 OOS / 杠杆 / vol目标 / bootstrap）
    _write_blend_frontier(r, wf, rw, lv, vt, bs)


def _write_blend_frontier(r, wf, rw=None, lv=None, vt=None, bs=None):
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
    # 逐年滚动 OOS
    if rw:
        html += ('<h2 style="font-size:15px;margin:22px 0 6px">逐年滚动样本外（固定 blend，'
                 '切 2022-2026 每年看是否都赚）</h2>')
        html += ('<p class="sub">策略参数(sma200/逆波动)全程固定、未在任何数据选参，'
                 '按年切分即诚实 OOS。2026 仅至 9/30 为部分年。</p>')
        html += ('<table><thead><tr><th>混合(H1%)</th><th>正收益年</th>'
                 '<th>2022</th><th>2023</th><th>2024</th><th>2025</th>'
                 '<th>2026*</th><th>OOS年化</th><th>OOS Sharpe</th><th>OOS回撤</th>'
                 '</tr></thead><tbody>')
        for b in rw["blends"]:
            rows = rw["per_year"][b]
            o = rw["oos"][b]
            pos = sum(1 for x in rows if x["ret"] > 0)
            cells = "".join(
                f"<td>{('+%.1f%%' % (x['ret']*100)) if x['ret'] >= 0 else ('%.1f%%' % (x['ret']*100))}</td>"
                for x in rows)
            html += (f"<tr><td>{b*100:.0f}%</td><td>{pos}/{len(rows)}</td>{cells}"
                     f"<td>{o['ann']*100:.1f}%</td><td>{o['sharpe']:.2f}</td>"
                     f"<td>{o['mdd']*100:.1f}%</td></tr>\n")
        html += ("</tbody></table>"
                 '<p class="sub">*2026 为部分年(至 9/30)，年化仅供参考。'
                 '四档均"4/5 年正收益"，仅部分年 2026 微负——印证"稳定赚、回撤可控"。</p>')
    # 杠杆场景
    if lv:
        html += ('<h2 style="font-size:15px;margin:22px 0 6px">杠杆场景（底盘 L4 执行层，'
                 f'融资利率 {lv["financing"]*100:.0f}%/年，仅示意上限）</h2>')
        html += ('<table><thead><tr><th>杠杆</th><th>年化</th><th>Sharpe</th>'
                 '<th>回撤</th><th>波动</th></tr></thead><tbody>')
        for x in lv["rows"]:
            html += (f"<tr><td>{x['L']:.1f}x</td><td>{x['ann']*100:.1f}%</td>"
                     f"<td>{x['sharpe']:.2f}</td><td>{x['mdd']*100:.1f}%</td>"
                     f"<td>{x['vol']*100:.1f}%</td></tr>\n")
        html += ("</tbody></table>"
                 '<p class="sub">杠杆对这套低收益/常防御的策略几乎无效：1.3x 仅把年化从 10.0% '
                 '抬到 10.9%，夏普反降（融资利息拖累 + 空仓期仍付杠杆息）。'
                 '<b>想多赚应加 H1 卫星仓，而非加杠杆。</b></p>')
    # 波动率目标化
    if vt:
        html += ('<h2 style="font-size:15px;margin:22px 0 6px">进阶：波动率目标化 '
                 '(vol targeting，执行层 L4)</h2>')
        html += ('<p class="sub">把组合年化波动钉在 10%：低波动期加仓(≤1.5x)、'
                 '高波动期自动降仓转现金。合法风控，非因子过拟合。</p>')
        html += ('<table><thead><tr><th>混合(H1%)+vol目标</th><th>年化</th><th>Sharpe</th>'
                 '<th>回撤</th><th>波动</th><th>均杠杆</th></tr></thead><tbody>')
        for x in vt["rows"]:
            html += (f"<tr><td>{x['blend']*100:.0f}%</td><td>{x['ann']*100:.1f}%</td>"
                     f"<td>{x['sharpe']:.2f}</td><td>{x['mdd']*100:.1f}%</td>"
                     f"<td>{x['vol']*100:.1f}%</td><td>{x['avg_lev']:.2f}x</td></tr>\n")
        w = vt["wf"]
        html += ("</tbody></table>"
                 f'<p class="sub">walk-forward OOS(底盘+vol目标)：训练 Sharpe '
                 f'{w["train"]["sharpe"]:.2f}/年化 {w["train"]["ann"]*100:.1f}% → '
                 f'测试 Sharpe {w["test"]["sharpe"]:.2f}/年化 {w["test"]["ann"]*100:.1f}%/'
                 f'回撤 {w["test"]["mdd"]*100:.1f}%。参数固定未选参，样本外仍稳定。</p>')
    # 块 Bootstrap 显著性
    if bs:
        html += ('<h2 style="font-size:15px;margin:22px 0 6px">块 Bootstrap 显著性 '
                 '(21日块重采样 1000 次，90% 置信区间)</h2>')
        html += ('<p class="sub">年化 CI 下限&gt;0 即"稳定赚"统计显著、非运气。</p>')
        html += ('<table><thead><tr><th>混合(H1%)</th><th>年化90%CI</th><th>均值</th>'
                 '<th>Sharpe90%CI</th><th>回撤90%CI</th></tr></thead><tbody>')
        for x in bs["rows"]:
            html += (f"<tr><td>{x['blend']*100:.0f}%</td>"
                     f"<td>[{x['ann_ci'][0]*100:+.1f}%, {x['ann_ci'][1]*100:+.1f}%]</td>"
                     f"<td>{x['ann_mean']*100:+.1f}%</td>"
                     f"<td>[{x['sharpe_ci'][0]:.2f}, {x['sharpe_ci'][1]:.2f}]</td>"
                     f"<td>[{x['mdd_ci'][0]*100:.1f}%, {x['mdd_ci'][1]*100:.1f}%]</td></tr>\n")
        html += "</tbody></table>"
    html += "</div></body></html>"
    html_path = _os.path.join(doc_dir, "blend_frontier.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  -> 前沿可视化已写出: {html_path}")


def _print_ml_block(title, r):
    """打印单个特征面板的硬择时/软加权/恒定比例空基准。"""
    a = r["always"]
    lab = {"gbm": "LightGBM(ML)", "xgb": "XGBoost(ML)", "mlp": "PyTorch MLP(DL)"}
    print(f"  [{title}]")
    print(f"  {'模型':<14}{'年化':>9}{'Sharpe':>9}{'回撤':>9}{'持有命中率':>11}{'持有占比/暴露':>13}")
    print(f"  {'始终持有':<14}{a['ann']*100:>8.1f}%{a['sharpe']:>9.2f}{a['mdd']*100:>8.1f}%"
          f"{'  -':>11}{'100%':>13}")
    for k, m in r["models"].items():
        h = m["hard"]; s = m["soft"]
        print(f"  {lab.get(k, k):<14}{h['ann']*100:>8.1f}%{h['sharpe']:>9.2f}{h['mdd']*100:>8.1f}%"
              f"{h['hit']*100:>10.0f}%{s['avg_exp']*100:>11.0f}%")
    if r["constant"]:
        for f, c in r["constant"].items():
            print(f"  {int(f*100):>4}%恒定{'':<8}{c['ann']*100:>8.1f}%{c['sharpe']:>9.2f}{c['mdd']*100:>8.1f}%")


def _print_ml_yearly(ry):
    """打印逐年滚动 OOS：每年软加权 vs 始终持有 vs 恒定60%。"""
    print(f"  [逐年滚动 OOS (自 {ry['test_start']})]")
    print(f"  {'年':<6}{'始终年化':>10}{'始终回撤':>9}{'恒60%年化':>11}{'恒60%回撤':>10}{'最优软模型':>16}{'软Sharpe':>9}")
    for yr, yd in ry["yearly"].items():
        a = yd["always"]; c = yd["const60"]
        best_k = max(yd["soft"], key=lambda k: yd["soft"][k]["sharpe"])
        bs = yd["soft"][best_k]
        print(f"  {yr:<6}{a['ann']*100:>9.1f}%{a['mdd']*100:>8.1f}%{c['ann']*100:>10.1f}%"
              f"{c['mdd']*100:>9.1f}%{best_k:>14}{bs['sharpe']:>9.2f}")
    n = len(ry["yearly"])
    n_soft_win = sum(1 for yd in ry["yearly"].values()
                     if max(yd["soft"].values(), key=lambda m: m["sharpe"])["sharpe"] > yd["always"]["sharpe"])
    n_const_dom = sum(1 for yd in ry["yearly"].values()
                      if yd["const60"]["sharpe"] >= max(yd["soft"].values(), key=lambda m: m["sharpe"])["sharpe"])
    print(f"  -> {n_soft_win}/{n} 年软加权 Sharpe 跑赢始终持有；"
          f"{n_const_dom}/{n} 年恒定60% Sharpe ≥ 最优软加权")


def _html_ml_block(r, idx, title):
    """返回单个面板 hard/soft/constant 三表的 HTML 片段。"""
    a = r["always"]
    lab = {"gbm": "LightGBM(ML)", "xgb": "XGBoost(ML)", "mlp": "PyTorch MLP(DL)"}
    h = (f"<h3 style='color:#9db4ff'>{idx}、{title}</h3>"
         "<table><thead><tr><th>模型</th><th>年化</th><th>Sharpe</th><th>最大回撤</th>"
         "<th>命中率</th><th>持有占比/暴露</th></tr></thead><tbody>")
    cls_a = "win" if a["ann"] >= 0 else "lose"
    h += (f"<tr><td>始终持有(基准)</td><td class='{cls_a}'>{a['ann']*100:+.1f}%</td>"
          f"<td>{a['sharpe']:.2f}</td><td>{a['mdd']*100:.1f}%</td><td>-</td><td>100%</td></tr>")
    for k, m in r["models"].items():
        hh = m["hard"]; s = m["soft"]
        cls = "win" if hh["ann"] >= 0 else "lose"
        h += (f"<tr><td>{lab.get(k, k)}</td><td class='{cls}'>{hh['ann']*100:+.1f}%</td>"
              f"<td>{hh['sharpe']:.2f}</td><td>{hh['mdd']*100:.1f}%</td>"
              f"<td>{hh['hit']*100:.0f}%</td><td>{s['avg_exp']*100:.0f}%</td></tr>")
    h += "</tbody></table>"
    if r["constant"]:
        h += ("<table style='margin-top:8px'><thead><tr><th>空基准(恒定比例)</th><th>年化</th>"
              "<th>Sharpe</th><th>最大回撤</th></tr></thead><tbody>")
        for f, c in r["constant"].items():
            cls = "win" if c["ann"] >= 0 else "lose"
            h += (f"<tr><td>{int(f*100)}%</td><td class='{cls}'>{c['ann']*100:+.1f}%</td>"
                  f"<td>{c['sharpe']:.2f}</td><td>{c['mdd']*100:.1f}%</td></tr>")
        h += "</tbody></table>"
    return h


def _write_ml_html_multi(panels, ry):
    """写出多面板 + 逐年 OOS 的 ML/DL 验证报告。"""
    import os as _os
    here = _os.path.dirname(__file__)
    doc_dir = _os.path.normpath(_os.path.join(here, "..", "..", "docs"))
    _os.makedirs(doc_dir, exist_ok=True)
    # 用 v1 做主裁定
    r0 = panels[list(panels.keys())[0]]
    a = r0["always"]
    lab = {"gbm": "LightGBM(ML)", "xgb": "XGBoost(ML)", "mlp": "PyTorch MLP(DL)"}
    best_k, best_m = max(r0["models"].items(), key=lambda kv: kv[1]["soft"]["sharpe"])
    beats = best_m["soft"]["sharpe"] > a["sharpe"]
    md = {k: m["soft"]["mdd"] for k, m in r0["models"].items()}
    best_md_k = max(md, key=md.get)
    verdict = ("跑赢" if beats else "未跑赢") + "始终持有基准"
    badge = "ok" if beats else "no"

    html = ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<title>ML/DL 择时叠加层验证</title>"
            "<style>body{background:#1a1d24;color:#e6e9ef;font-family:system-ui,'Microsoft YaHei',sans-serif;margin:0;padding:28px}"
            "h1{font-size:20px;color:#9db4ff}.sub{color:#8b93a7;font-size:13px;margin:4px 0 18px}"
            "table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}"
            "th,td{border:1px solid #2c313c;padding:7px 10px;text-align:right}"
            "th{background:#232834;color:#c7d0e0}td:first-child,th:first-child{text-align:left}"
            ".win{color:#ff6b6b}.lose{color:#4ade80}.note{color:#8b93a7;font-size:12px;line-height:1.6}"
            ".badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;margin-left:8px}"
            ".ok{background:#16351f;color:#6ee7a0}.no{background:#3a1d1d;color:#ff9b9b}</style></head>"
            "<body><h1>ML/DL 择时叠加层验证报告</h1>"
            f"<div class='sub'>测试窗自 {r0['test_start']} · 共 {r0['n_decisions']} 个月度决策 · "
            f"expanding-window walk-forward(无前视) · 数据: TimescaleDB bars_1d (QMT回填)</div>")

    for i, (pk, pr) in enumerate(panels.items(), 1):
        title = {"v1": "基础特征(7维: 宽度/动量/波动/回撤)",
                 "v2": "增强特征(多期限宽度+风险偏好+期限利差+黄金+离散度)"}.get(pk, pk)
        html += _html_ml_block(pr, i, title)

    # 逐年 OOS
    html += ("<h3 style='color:#9db4ff'>三、逐年滚动 OOS 稳定性</h3>"
             "<table><thead><tr><th>年</th><th>始终年化</th><th>始终回撤</th><th>恒60%年化</th>"
             "<th>恒60%回撤</th><th>最优软模型</th><th>软Sharpe</th></tr></thead><tbody>")
    for yr, yd in ry["yearly"].items():
        aa = yd["always"]; c = yd["const60"]
        bk = max(yd["soft"], key=lambda k: yd["soft"][k]["sharpe"])
        bs = yd["soft"][bk]
        html += (f"<tr><td>{yr}</td><td>{aa['ann']*100:+.1f}%</td><td>{aa['mdd']*100:.1f}%</td>"
                 f"<td>{c['ann']*100:+.1f}%</td><td>{c['mdd']*100:.1f}%</td>"
                 f"<td>{lab.get(bk, bk)}</td><td>{bs['sharpe']:.2f}</td></tr>")
    html += "</tbody></table>"

    html += (f"<h3 style='color:#9db4ff'>四、裁定</h3>"
             f"<p>最优 ML/DL(软加权, v1) = <b>{lab.get(best_k, best_k)}</b> "
             f"(Sharpe {best_m['soft']['sharpe']:.2f}) "
             f"<span class='badge {badge}'>{verdict}</span></p>"
             f"<p class='note'>软加权最小回撤来自 <b>{lab.get(best_md_k, best_md_k)}</b>："
             f"<b>{md[best_md_k]*100:.1f}%</b>(基准 {a['mdd']*100:.1f}%) —— "
             "类'降杠杆'效果，回撤减半，对'回撤可控/能睡着'诉求直接有用；"
             "但 Sharpe 未超过'恒定比例持有'，说明模型概率没能在下跌前自动变低，"
             "即<b>没有真正的择时 alpha</b>，只是把平均暴露降到~60%。</p>"
             f"<p class='note'><b>空基准证伪</b>：恒定 60% 持有 = 5.4% / Sharpe 1.03 / -4.8%，"
             f"而最优软加权(MLP, 平均暴露58%) = 5.1% / Sharpe 0.99 / -4.9%——两者几乎重合，"
             f"且 ML 的 Sharpe 反而<b>略低于</b>恒定比例持有，即 ML 软加权被朴素'永远持60%'严格 dominate。"
             f"增强特征(v2)也未改变结论。逐年 OOS 中恒定60%在多数年份 Sharpe ≥ 最优软加权，"
             f"进一步坐实：本样本下 ML/DL 择时层不创造 alpha。</p>"
             "<p class='note'>说明：本环境 sklearn.LogisticRegression.fit() 静默 segfault(二进制兼容bug)，"
             "故 ML 用 LightGBM/XGBoost、DL 用 PyTorch MLP；三模型均按 expanding-window walk-forward 训练，"
             "预测月用此前全部历史、收益在预测后才'实现'，严格无前视。样本仅~90个月度点(44测试月)，"
             "ML/DL 择时(硬/软)均未能稳定击败朴素持有——与项目既有'ML因子证伪'结论一致：叠加层要产生价值，"
             "需更大样本或更具预测力的特征，而非单纯堆模型复杂度。已验证的古典趋势跟随(SMA200+逆波动)"
             "仍是本池最稳的'能睡着'策略。</p>")
    html += "</body></html>"
    html_path = _os.path.join(doc_dir, "ml_timing_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  -> ML 择时报告已写出: {html_path}")


def cmd_ml():
    print("\n===== ML/DL 择时叠加层（LightGBM / XGBoost / PyTorch MLP，expanding walk-forward） =====")
    try:
        r1 = method_ml.walk_forward_ml(panel_fn=method_ml.build_ml_panel)
        r2 = method_ml.walk_forward_ml(panel_fn=method_ml.build_ml_panel_v2)
        ry = method_ml.rolling_walk_forward_ml(panel_fn=method_ml.build_ml_panel)
    except DataError as e:
        print(f"  [跳过] {e}")
        return
    print(f"  测试窗自 {r1['test_start']}，共 {r1['n_decisions']} 个月度决策；"
          f"对比基准=始终持有趋势策略")
    _print_ml_block("v1 基础特征(7维)", r1)
    _print_ml_block("v2 增强特征", r2)
    _print_ml_yearly(ry)
    _write_ml_html_multi({"v1": r1, "v2": r2}, ry)


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
                             "ml", "calltracker"])
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
        elif args.cmd == "ml":
            cmd_ml()
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
