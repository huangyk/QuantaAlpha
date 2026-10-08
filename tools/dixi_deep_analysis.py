"""低吸（战法2 均线回踩）深度分析 —— 补齐此前从未做过的三个缺口。

背景（2026-10-07 已裁定，勿推翻，本轮只做增量）：
    战法2 三因子中只有 f4（贴近 MA20）有信息，且**强依赖市场风格**：
    2022-2024 IC 显著正、2025-2026 反向。当时留下「路线A：风格择时开关」待定夺，从未实现。
    且此前**所有**低吸结论都是「固定持有期」口径，从未做过真正的
    止损/止盈路径回测 —— 而低吸战法的真实形态是路径依赖的，两者不等价。

本脚本补齐：

  Part 1  干净口径基线重算
          - 剔除次新股（上市 < 250 交易日）      ← 2026-10-07 定的铁律，warfare2 脚本早于它
          - 过滤 T+1 涨停买不进                  ← 用户指出的泄漏点之一
          - 真实双边成本（万1 + 印花0.05% + 过户0.001% + 冲击）
  Part 2  风格择时开关（从未做过）
          - 状态变量必须用【池自身状态】（沪深300 管中小盘方向会反，已有教训）
          - 检验：f4 的 IC 在强弱两市下是否**符号相反且各自 |t|>=2**，否则是后见之明
  Part 3  止损/止盈路径回测（从未做过）
          - T 日收盘出信号 → T+1 开盘买 → 按 {止损 / 止盈 / 最大持有期} 路径出场
          - ⚠ A股 T+1：买入当日不可卖出（判断从次日起）
          - 基准 = 同一 T 日、全池所有股票**走完全相同路径规则**的平均收益（严格同口径）

运行： conda quantaalpha; python tools/dixi_deep_analysis.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = 'D:/qqmt/qlib_data/cn_real_full'
START, END = '2020-01-01', '2026-09-30'
FIELDS = ['$open', '$high', '$low', '$close', '$volume']
POOL = 'liquid500'
SPLIT = '2025-01-01'
NEW_DAYS = 250                      # 上市不足 250 个交易日视为次新股，剔除
MAX_HOLD = 20
OUT = Path('data/results')

# ---- 真实交易成本（用户实际：万1，无最低佣金；ETF外的股票有印花/过户）----
COMM = 0.0001                       # 佣金（双边）
STAMP = 0.0005                      # 印花税（仅卖出）
TRANSFER = 0.00001                  # 过户费（双边）
IMPACT_BP = 9.5                     # 单边冲击 bp（2026-10-07 实测：中小盘 ~9.5bp）
COST_ONEWAY = COMM + TRANSFER + IMPACT_BP / 1e4
COST_ROUND = COMM * 2 + STAMP + TRANSFER * 2 + IMPACT_BP * 2 / 1e4   # ≈ 0.0255


# ============================================================ 面板构建
def load():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    codes = D.list_instruments(D.instruments(POOL), start_time=START, end_time=END, as_list=True)
    df = D.features(codes, FIELDS, start_time=START, end_time=END, freq='day').sort_index()
    panels = {}
    for i, g in df.groupby(level='instrument'):
        panels[i] = g.droplevel('instrument').sort_index()
    return panels


def build(panels):
    """构建长表面板。

    ⚠️ 铁律1（停牌 NaN 放大）：先 ffill(limit=5) 补停牌价再算均线，
       前瞻收益/成交判断一律用**原始** OHLC，保证只统计真实可交易日。
    ⚠️ 铁律5（次新股）：上市不足 250 交易日的样本必须剔除，否则基准被污染。
    ⚠️ 涨停买入过滤：T+1 开盘相对 T 收盘 >= +9.8% 视为买不进。
    """
    frames = []
    for inst, g in panels.items():
        ro, rh, rl, rc = g['$open'], g['$high'], g['$low'], g['$close']
        f = g.ffill(limit=5)
        c, v = f['$close'], f['$volume']
        ma20 = c.rolling(20, min_periods=20).mean()
        ma60 = c.rolling(60, min_periods=60).mean()
        ma5v = v.rolling(5, min_periods=5).mean()

        d = pd.DataFrame(index=g.index)
        d['f3'] = (ma20 - ma60) / ma60                  # 多头排列强度
        d['f4'] = -((c - ma20).abs() / ma20)            # 贴近 MA20（低吸核心）
        d['f2'] = v / ma5v                              # 放量强度
        d['is_bull'] = (ma20 > ma60) & (c > ma20)
        d['dist_ma20'] = (c - ma20) / ma20

        # ---- 可交易性（全部基于原始 OHLC）----
        buy_px = ro.shift(-1)                           # T+1 开盘价
        d['buy'] = buy_px
        d['fwd'] = ro.shift(-2) / ro.shift(-1) - 1.0    # T+1买→T+2卖（1日持有，铁律：T日出分从T+1计）
        d['can_buy'] = buy_px.notna() & (buy_px / rc < 1.098)   # 剔除涨停买不进
        d['mature'] = rc.notna().cumsum() >= NEW_DAYS   # 次新股剔除

        d['ro'], d['rh'], d['rl'], d['rc'] = ro, rh, rl, rc
        d['close'] = c
        d = d.reset_index()
        d['instrument'] = inst
        frames.append(d.set_index(['datetime', 'instrument']))
    panel = pd.concat(frames).sort_index()

    # 统一应用剔除条件（旧脚本缺失的一步）
    panel = panel[panel['mature'] & panel['can_buy']]
    return panel


def add_regime(panel):
    """用【池自身状态】给每个交易日打市场环境标签（T 日收盘可知，无前视）。

    三个维度（都不用沪深300 —— 已有教训：沪深300 管中小盘池方向会反）：
      breadth 池内站上 MA20 的占比（宽度，衡量普涨/普跌）
      trend   池等权净值 / 其 MA60 - 1（趋势）
      vol     池等权 20 日已实现年化波动（恐慌度）
    """
    eq_ret = (panel['rc'].groupby(level='instrument').pct_change()   # ⚠ 必须先按个股求日收益
              .groupby(level='datetime').mean())                     # 再做截面等权
    idx = (1.0 + eq_ret.fillna(0.0)).cumprod()
    trend = idx / idx.rolling(60, min_periods=60).mean() - 1.0
    # 宽度：池内站上自身 MA20 的占比（dist_ma20 已在 build 里按个股算好）
    above = (panel['dist_ma20'] > 0).groupby(level='datetime').mean()
    vol = eq_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    R = pd.DataFrame({'eq_ret': eq_ret, 'breadth': above, 'trend': trend, 'vol': vol})
    return panel.join(R, on='datetime'), R


# ============================================================ 统计工具
def ic_by_day(panel, dates, col='pred', fwd='fwd', min_n=50):
    out = {}
    for dt in dates:
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=[col, fwd])
        if len(sub) < min_n:
            continue
        out[dt] = sub[col].corr(sub[fwd], method='spearman')
    return pd.Series(out).dropna()


def icstat(s):
    if len(s) < 5:
        return {'n': len(s)}
    return {'IC': s.mean(), 'ICIR': s.mean() / s.std(ddof=1),
            't': s.mean() / (s.std(ddof=1) / np.sqrt(len(s))),
            'pos%': float((s > 0).mean()), 'n': len(s)}


def show(tag, st):
    if 'IC' not in st:
        print(f"  {tag:<34} 样本不足 n={st['n']}")
        return
    print(f"  {tag:<34} IC={st['IC']:+.4f}  ICIR={st['ICIR']:+.3f}  "
          f"t={st['t']:+6.2f}  IC>0={st['pos%']:5.1%}  n={st['n']}")


# ============================================================ Part 3 路径回测
def path_pnl(sub, stop, target, max_hold, exit_mode='trigger', cost_round=COST_ROUND):
    """单只股票：从每个 d 交易日结算子。

    规则：T 日收盘出信号 → buy = open[T+1]
          ⚠ A股 T+1：buy 当日（j=0）不可卖 → 出场判断从 j=1 起
          j = 1..max_hold-1：先判止损（保守，再判止盈）
          j = max_hold-1 仍未触发 → 当日收盘强制平仓
    返回 Series(index=date)：单笔净收益率（已扣双边成本）
    """
    o, h, l, c = sub['ro'], sub['rh'], sub['rl'], sub['rc']
    buy = sub['buy']
    res = pd.Series(np.nan, index=sub.index)
    if buy.notna().sum() == 0:
        return res

    stopPx = buy * (1.0 - stop) if stop else pd.Series(-np.inf, index=sub.index)
    tgtPx = buy * (1.0 + target) if target else pd.Series(np.inf, index=sub.index)

    exitpx = pd.Series(np.nan, index=sub.index)
    live = buy.notna()

    for j in range(1, max_hold):                  # j = 持有第 j 天；日期偏移 1+j
        off = 1 + j
        op, hp, lp, cp = o.shift(-off), h.shift(-off), l.shift(-off), c.shift(-off)
        act = live & exitpx.isna() & cp.notna()
        if not act.any():
            break
        hit_s = act & ((op <= stopPx) | (lp <= stopPx))          # 触及止损（含跳空低开）
        hit_t = act & ~hit_s & ((op >= tgtPx) | (hp >= tgtPx))   # 触及止盈（保守：先判止损）
        # ⚠ 用 mask 而非 where 链式：where 的第二次调用会把已成交的止损行
        #    错误地覆盖成止盈价（负债累累的 bug，务必保留此处写法）
        if exit_mode == 'trigger':
            px_s = np.minimum(op, stopPx)          # 跳空低开则按开盘价止损
            px_t = np.maximum(op, tgtPx)           # 跳空高开则按开盘价止盈
        else:                                      # 'close'：触发日收盘成交（中性/保守）
            px_s = px_t = cp
        exitpx = exitpx.mask(hit_s, px_s)
        exitpx = exitpx.mask(hit_t, px_t)

    forced = c.shift(-max_hold)                   # 到期强平（最后持有日收盘）
    exitpx = exitpx.fillna(forced)

    ok = buy.notna() & exitpx.notna() & (buy > 0)
    res[ok] = (exitpx[ok] / buy[ok] - 1.0) - cost_round
    return res


def run_paths(panel, combos):
    """对池内每只股票跑路径，返回 list of Series[(date,instrument)] -> net return"""
    out = {}
    for name, (stop, target, mh, mode) in combos.items():
        allser = []
        for inst, g in panel.groupby(level='instrument'):
            s = path_pnl(g.droplevel('instrument').sort_index(), stop, target, mh, mode)
            s.index = pd.MultiIndex.from_product([[inst], s.index], names=['instrument', 'datetime'])
            allser.append(s)
        out[name] = pd.concat(allser).reorder_levels(['datetime', 'instrument']).sort_index()
    return out


# ============================================================ 主流程
def part1(panel, dates):
    print("\n" + "=" * 78)
    print("Part 1  干净口径基线重算（剔次新 + 剔涨停买不进 + 真实成本）")
    print("=" * 78)
    print(f"  样本 {len(panel):,} 行 / {panel.index.get_level_values('instrument').nunique()} 只 / "
          f"{len(dates)} 天")
    for f in ['f3', 'f4', 'f2']:
        show(f"全样本 {f}", icstat(ic_by_day(panel, dates, col=f)))
    print("\n  --- f4 分年 IC（核心地基因子，看风格切换是否依旧）---")
    for y in sorted({pd.Timestamp(d).year for d in dates}):
        dy = [d for d in dates if pd.Timestamp(d).year == y]
        show(f"{y}  f4", icstat(ic_by_day(panel, dy, col='f4')))


def part2(panel, dates, R):
    print("\n" + "=" * 78)
    print("Part 2  风格择时开关（从未做过）—— f4 是否真的是一个「有条件的真因子」")
    print("=" * 78)
    print("  判定标准：强弱两市下 f4 的 IC 必须**符号相反且各自 |t| >= 2**，否则只是后见之明。\n")

    valid = R.dropna().index
    for var in ['breadth', 'trend', 'vol']:
        s = R[var].dropna()
        lo, hi = s.quantile(0.33), s.quantile(0.67)
        print(f"  --- 按 {var} 三分组（低 ≤{lo:.3f} / 中 / 高 ≥{hi:.3f}）---")
        grp = pd.cut(R[var], [-np.inf, lo, hi, np.inf], labels=['低', '中', '高'])
        for lb in ['低', '中', '高']:
            ds = [d for d in dates if d in valid and grp.get(d) == lb]
            show(f"    {var}={lb}", icstat(ic_by_day(panel, ds, col='f4')))
        print()

    # 复合状态：弱市（trend<中位数 且 breadth<中位数）vs 强市（两者都高）
    rb = R.reindex(dates)
    medt, medb = rb['trend'].median(), rb['breadth'].median()
    weak = [d for d in dates if rb.loc[d, 'trend'] < medt and rb.loc[d, 'breadth'] < medb]
    strong = [d for d in dates if rb.loc[d, 'trend'] >= medt and rb.loc[d, 'breadth'] >= medb]
    print(f"  --- 复合状态：弱市(trend↓ & breadth↓) n={len(weak)} 天 / "
          f"强市(trend↑ & breadth↑) n={len(strong)} 天 ---")
    show("    弱市 f4", icstat(ic_by_day(panel, weak, col='f4')))
    show("    强市 f4", icstat(ic_by_day(panel, strong, col='f4')))
    show("    弱市 f3", icstat(ic_by_day(panel, weak, col='f3')))
    show("    强市 f3", icstat(ic_by_day(panel, strong, col='f3')))
    return weak, strong


def part3(panel, paths, weak, strong):
    print("\n" + "=" * 78)
    print("Part 3  止损/止盈路径回测（从未做过）—— 低吸的真实形态")
    print("=" * 78)
    print(f"  成本：双边 {COST_ROUND*100:.3f}%（万1+印花0.05%+过户+冲击9.5bp×2）")
    print("  基准：同一天全池所有股票走**完全相同路径规则**的平均收益（严格同口径）\n")

    rows = []
    weak_s, strong_s = set(weak), set(strong)
    for name, s in paths.items():
        s = s.dropna()
        # ⚠ 不能用 df.loc[日期列表]：MultiIndex 部分索引要求全部命中，
        #   而某些交易日已因 net=NaN 被删掉 → 用 isin 布尔掩码
        df = panel.join(s.rename('net'), how='inner').dropna(subset=['net'])
        if df.empty:
            continue
        bench = df.groupby(level='datetime')['net'].mean()        # 同路径全池均值（基准）
        groups = dict(tuple(df.groupby(level='datetime')))        # 预分组，避免三层循环重复开销

        for gname, gd in [('全期', None), ('弱市', weak_s), ('强市', strong_s)]:
            for tag, pick in [('f4前20%', 'top20'), ('f4前10只', 'top10'), ('事件型', 'event')]:
                sel = []
                for dt, gday in groups.items():
                    if gd is not None and dt not in gd:
                        continue
                    gday = gday.dropna(subset=['f4', 'net'])
                    if len(gday) < 30:
                        continue
                    if pick == 'top20':
                        try:
                            lab = pd.qcut(gday['f4'].rank(ascending=False, method='first'),
                                          5, labels=False)
                        except ValueError:
                            continue
                        sel.append(gday[lab == 0]['net'])
                    elif pick == 'top10':
                        sel.append(gday.nlargest(10, 'f4')['net'])
                    else:
                        e = gday[(gday['f3'] > 0) & (gday['f4'] > -0.03) & (gday['f2'] > 1.5)]
                        if len(e):
                            sel.append(e['net'])
                if not sel:
                    continue
                r = pd.concat(sel)
                b = bench.reindex(r.index.get_level_values('datetime'))
                ex = (r.values - b.values)
                if len(ex) < 30 or np.std(ex) == 0:
                    continue
                t = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))
                rows.append({'路径': name, '市场': gname, '选法': tag, '笔数': len(r),
                             '单笔收益': r.mean(), '基准': np.nanmean(b.values),
                             '单笔超额': ex.mean(), 't': t, '胜率': float((ex > 0).mean())})
    if not rows:
        print("  无有效样本")
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(res.assign(单笔收益=lambda x: x['单笔收益'] * 100, 基准=lambda x: x['基准'] * 100,
                     单笔超额=lambda x: x['单笔超额'] * 100).round(3).to_string(index=False))
    return res


def daily_series(panel, net, pick):
    """把逐笔结果按**交易日聚合**成日度超额序列。

    ⚠️ 逐笔统计的 t 值是虚高的：同一天几百只票 + 相邻交易日同一只票，样本高度重叠自相关，
       n=10 万笔但真实独立观测只有约 1361 天。必须用日度序列重算显著性。
    """
    df = panel.join(net.rename('net'), how='inner').dropna(subset=['net'])
    bench = df.groupby(level='datetime')['net'].mean()
    e = {}
    for dt, g in df.groupby(level='datetime'):
        g = g.dropna(subset=['f4', 'net'])
        if len(g) < 30:
            continue
        if pick == 'top20':
            try:
                lab = pd.qcut(g['f4'].rank(ascending=False, method='first'), 5, labels=False)
            except ValueError:
                continue
            sel = g[lab == 0]['net']
        elif pick == 'top10':
            sel = g.nlargest(10, 'f4')['net']
        else:
            sel = g[(g['f3'] > 0) & (g['f4'] > -0.03) & (g['f2'] > 1.5)]['net']
        if len(sel) >= 3:
            e[dt] = sel.mean() - bench[dt]
    return pd.Series(e).sort_index()


def dstat(s):
    if len(s) < 20:
        return {}
    return {'天数': len(s), '日均超额': s.mean(),
            't(日度)': s.mean() / (s.std(ddof=1) / np.sqrt(len(s))),
            'IR年化': s.mean() / s.std(ddof=1) * np.sqrt(252) if s.std(ddof=1) > 0 else np.nan,
            '日胜率': float((s > 0).mean())}


def part4(panel, paths, dates):
    print("\n" + "=" * 78)
    print("Part 4  日度聚合的真实性检验 —— 逐笔 t 值是虚高的，必须用日度重算")
    print("=" * 78)
    print("  逐笔样本高度重叠（同日数百只 + 相邻日同一只），n=10万笔其实只有 ~1361 个独立观测。\n")
    tr = [d for d in dates if d < pd.Timestamp(SPLIT)]
    te = [d for d in dates if d >= pd.Timestamp(SPLIT)]

    cases = [
        ('持有20 无止损止盈', 'f4前20%', '对照：不设止损'),
        ('持有10 无止损止盈', 'f4前20%', '对照：不设止损'),
        ('止损5% 止盈8% 最长10', 'f4前20%', '低吸+tight止损'),
        ('止损8% 止盈10% 最长20', 'f4前20%', '低吸+标准止损'),
        ('止损8% 止盈10% 最长10', '事件型', '原战法布尔+止损'),
        ('只止损8% 最长20', '事件型', '原战法+仅止损'),
    ]
    pick_map = {'f4前20%': 'top20', 'f4前10只': 'top10', '事件型': 'event'}
    rows = []
    for pname, p, note in cases:
        if pname not in paths:
            continue
        s = daily_series(panel, paths[pname], pick_map[p])
        if s.empty:
            continue
        all_ = dstat(s)
        trs = dstat(s.reindex(tr).dropna())
        tes = dstat(s.reindex(te).dropna())
        rows.append({'路径': pname, '选法': p, '备注': note,
                     '全期超额bp': all_.get('日均超额', np.nan) * 1e4, '全期t': all_.get('t(日度)'),
                     '全期IR': all_.get('IR年化'), '胜率': all_.get('日胜率'),
                     '训练t(21-24)': trs.get('t(日度)'), '测试t(25-26)': tes.get('t(日度)'),
                     '测试超额bp': tes.get('日均超额', np.nan) * 1e4, '天数': all_.get('天数')})
        print(f"  {pname} × {p}  | {note}")
        print(f"      全期 {all_.get('天数')}天 超额={all_.get('日均超额',0)*1e4:+6.2f}bp/日  "
              f"t={all_.get('t(日度)',np.nan):+5.2f}  IR={all_.get('IR年化',np.nan):+5.2f}  "
              f"胜率={all_.get('日胜率',0):.1%}")
        if trs:
            print(f"      训练(21-24) t={trs['t(日度)']:+5.2f}  |  "
                  f"测试(25-26) 超额={tes.get('日均超额',0)*1e4:+6.2f}bp/日 t={tes.get('t(日度)',np.nan):+5.2f}")
        print()
    res = pd.DataFrame(rows)
    if not res.empty:
        res.to_csv(OUT / 'dixi_daily_verdict.csv', index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/results/_dixi_panel.pkl')
    ap.add_argument('--rebuild', action='store_true')
    args = ap.parse_args()

    if args.rebuild or not Path(args.cache).exists():
        print("构建面板（含次新股剔除 + 涨停买入过滤）...")
        panels = load()
        panel = build(panels)
        panel, R = add_regime(panel)
        panel.to_pickle(args.cache)
        R.to_pickle(args.cache.replace('.pkl', '_R.pkl'))
    else:
        panel = pd.read_pickle(args.cache)
        R = pd.read_pickle(args.cache.replace('.pkl', '_R.pkl'))
        print(f"读缓存 {args.cache}")

    dates = sorted(set(panel.index.get_level_values('datetime')))
    print(f"面板 {len(panel):,} 行 / {len(dates)} 天")

    part1(panel, dates)
    weak, strong = part2(panel, dates, R)

    combos = {
        '持有5 无止损止盈': (None, None, 5, 'trigger'),
        '持有10 无止损止盈': (None, None, 10, 'trigger'),
        '持有20 无止损止盈': (None, None, 20, 'trigger'),
        '止损8% 止盈10% 最长10': (0.08, 0.10, 10, 'trigger'),
        '止损8% 止盈10% 最长20': (0.08, 0.10, 20, 'trigger'),
        '止损5% 止盈8% 最长10': (0.05, 0.08, 10, 'trigger'),
        '止损10% 止盈20% 最长20': (0.10, 0.20, 20, 'trigger'),
        '只止损8% 最长20': (0.08, None, 20, 'trigger'),
        '只止盈10% 最长20': (None, 0.10, 20, 'trigger'),
        '止损8% 止盈10% 最长20(收盘价成交)': (0.08, 0.10, 20, 'close'),
    }
    print("\n计算路径（每只股票 × 每条规则）...")
    paths = run_paths(panel, combos)
    res = part3(panel, paths, weak, strong)
    if not res.empty:
        res.to_csv(OUT / 'dixi_path_results.csv', index=False)
        print(f"\n保存 data/results/dixi_path_results.csv")
    part4(panel, paths, dates)


if __name__ == '__main__':
    main()
