"""飞龙多因子验证 V2 —— 修掉「池/基准错配」后重跑，并重新裁定大盘闸门。

V1 的问题（已确认）：基准用沪深300，选股池用 liquid500（小盘/流动性池），
两者 2025-10~2026-09 相差 84 个百分点，"年化超额 +33%" 是错配假象，
IC≈0.0009 说明选股本身接近随机。

V2 三项改动：
    ① 同池等权基准（bench_eq）：池内全部个股次日收益的等权均值 —— 与选股池同口径，
       超额才是真正的选股 alpha；沪深300 只作为"市场环境"参照，不再当基准。
    ② 可交易收益口径：T 收盘出信号 → T+1 开盘成交 → T+2 开盘卖出
       （fwd = Ref($open,-2)/Ref($open,-1)-1），避免 V1 里 T 收盘价对 T 收盘价的理想化。
       同时保留 V1 口径（T→T+1 收盘）做对照。
    ③ 模型防过拟合：3 个特征上 300 棵树的 LGB 极易过拟合。改为在训练期内部再切
       验证段挑选模型（线性等权 / LGB-raw / LGB-rank），测试期只报结果，不再回挑。

闸门（③）：先做「闸门理论有效性」事件研究——不看策略，只看**池等权基准**在
bullish/neutral/bearish 日的收益是否显著分化（样本用全部 regime 有效日，不只测试期）；
再看策略在各 regime 下相对同池基准的超额；最后扫描几档 scale 组合，标注为探索性。

输出：data/results/feilong_v2_*.csv + docs/feilong_v2_verdict.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import qlib

from quantaalpha.strategy.feilong import feilong_factors, market_regime, compute_breadth

PROVIDER = 'D:/qqmt/qlib_data/cn_real_full'
START, END = '2024-06-01', '2026-09-30'
FIELDS = ['$open', '$high', '$low', '$close', '$volume', '$factor']
TOPK = 30
COST = 0.002          # 单边? 不：往返总成本率（佣金+印花+冲击近似）
CACHE = Path('data/results/_feilong_panel_v2.parquet')
SPLIT = '2025-10-01'  # 训练/测试分界（与 V1 一致，便于对照）
OUT = Path('data/results')


def load():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    idx = D.features(['SH000300'], FIELDS, start_time=START, end_time=END,
                     freq='day').droplevel('instrument').sort_index()
    codes = D.list_instruments(D.instruments('liquid500'), start_time=START, end_time=END, as_list=True)
    df = D.features(codes, FIELDS, start_time=START, end_time=END, freq='day').sort_index()
    panels = {}
    for i, g in df.groupby(level='instrument'):
        if isinstance(g.index, pd.MultiIndex):
            g = g.droplevel('instrument')
        panels[i] = g.sort_index()
    return idx, panels


def build_panel(panels):
    frames = []
    for inst, g in panels.items():
        f = feilong_factors(g)
        d = pd.DataFrame({
            'feilong_entry': f['feilong_entry'],
            'feilong_exit': f['feilong_exit'],
            'feilong_rr': f['feilong_rr'],
            'close': g['$close'], 'open': g['$open'],
        })
        # 可交易口径：T 收盘信号 → T+1 开盘买 → T+2 开盘卖
        d['fwd'] = d['open'].shift(-2) / d['open'].shift(-1) - 1.0
        # V1 口径（对照）：T 收盘 → T+1 收盘
        d['fwd_c2c'] = d['close'].shift(-1) / d['close'] - 1.0
        d = d.reset_index()
        d['instrument'] = inst
        frames.append(d.set_index(['datetime', 'instrument']))
    return pd.concat(frames).sort_index()


def cs_rank(df, cols):
    return df[cols].groupby(level='datetime').rank(pct=True)


def ic_series(panel, dates, pred_col='pred', ret_col='fwd'):
    """逐日 Spearman rank IC。"""
    out = {}
    for dt in dates:
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=[pred_col, ret_col])
        if len(sub) < 50:
            continue
        out[dt] = sub[pred_col].corr(sub[ret_col], method='spearman')
    s = pd.Series(out).dropna()
    return s


def ic_stats(s):
    if len(s) == 0:
        return {}
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if s.std(ddof=1) > 0 else np.nan
    return {'IC均值': s.mean(), 'IC标准差': s.std(ddof=1), 'ICIR': s.mean() / s.std(ddof=1),
            't统计': t, 'IC>0占比': float((s > 0).mean()), '天数': len(s)}


def run_backtest(panel, dates, topk=TOPK, gate=None, scale_col=None):
    """日频多头 top-K 等权。gate: dict(date->scale)。"""
    hold, rows = None, []
    for dt in dates:
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=['pred', 'fwd', 'bench_eq'])
        if len(sub) < topk:
            continue
        sel = sub.nlargest(topk, 'pred')
        gross = sel['fwd'].mean()
        cur = set(sel.index)
        turn = 1.0 if hold is None else 1.0 - len(cur & hold) / max(len(cur), 1)
        net = gross - turn * COST
        scale = 1.0 if gate is None else float(gate.get(dt, 1.0))
        rows.append({'date': dt, 'gross': gross, 'net': net, 'turnover': turn,
                     'bench_eq': sub['bench_eq'].iloc[0], 'scale': scale})
        hold = cur
    return pd.DataFrame(rows).set_index('date')


def metrics(rets, bench=None):
    rets = rets.dropna()
    if len(rets) == 0:
        return {}
    cum = rets.cumsum()
    out = {'年化': rets.mean() * 252, '最大回撤': (cum - cum.cummax()).min(),
           'IR': (rets.mean() / rets.std(ddof=1) * np.sqrt(252)) if rets.std(ddof=1) > 0 else np.nan,
           '日胜率': float((rets > 0).mean()), '累计': cum.iloc[-1], '天数': len(rets)}
    if bench is not None:
        ex = rets - bench.reindex(rets.index).fillna(0)
        ec = ex.cumsum()
        out['年化超额'] = ex.mean() * 252
        out['超额最大回撤'] = (ec - ec.cummax()).min()
        out['超额IR'] = (ex.mean() / ex.std(ddof=1) * np.sqrt(252)) if ex.std(ddof=1) > 0 else np.nan
        out['超额日胜率'] = float((ex > 0).mean())
        # 超额 t 统计（日频，考虑自相关弱，保守不调整）
        out['超额t统计'] = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if ex.std(ddof=1) > 0 else np.nan
    return out


def fmt(m, keys):
    return {k: round(float(m.get(k, np.nan)), 4) for k in keys}


def main():
    idx, panels = load()
    if CACHE.exists():
        panel = pd.read_parquet(CACHE)
    else:
        panel = build_panel(panels)
        panel.to_parquet(CACHE)

    # ---- 同池等权基准：池内全部个股次日(fwd)收益的等权均值 ----
    bq = panel.groupby(level='datetime')['fwd'].mean().rename('bench_eq')
    panel = panel.join(bq, on='datetime')
    # 沪深300 日收益（仅作市场环境参照）
    hs300 = idx['$close'].pct_change()

    # ---- regime：缩短波动分位窗口，让 regime 更早有效，扩大诊断样本 ----
    breadth = compute_breadth(panels, 20)
    reg = market_regime(idx, breadth.reindex(idx.index).ffill(),
                        p={'index_vol_pct_window': 120})
    valid_reg = reg.index[reg['regime_score'].notna() & reg['vol_pct'].notna()]
    print(f"regime 有效区间: {valid_reg.min().date()} ~ {valid_reg.max().date()} ({len(valid_reg)}天)")

    all_dates = sorted(set(panel.index.get_level_values('datetime')))
    tr_dates = [d for d in all_dates if d < pd.Timestamp(SPLIT)]
    te_dates = [d for d in all_dates if d >= pd.Timestamp(SPLIT)]
    print(f"训练 {tr_dates[0].date()}~{tr_dates[-1].date()} ({len(tr_dates)}天) | "
          f"测试 {te_dates[0].date()}~{te_dates[-1].date()} ({len(te_dates)}天)")

    feats = ['feilong_entry', 'feilong_exit', 'feilong_rr']

    # ---- 训练期内部再切验证段选模型（防测试期回挑） ----
    vcut = tr_dates[int(len(tr_dates) * 0.8)]
    fit_dates = [d for d in tr_dates if d < vcut]
    val_dates = [d for d in tr_dates if d >= vcut]
    tr_all = panel.loc[tr_dates].dropna(subset=feats + ['fwd'])
    fit = panel.loc[fit_dates].dropna(subset=feats + ['fwd'])
    val = panel.loc[val_dates].dropna(subset=feats + ['fwd'])
    print(f"  训练内 fit {len(fit_dates)}天 / valid {len(val_dates)}天")

    cands = {}
    # M0: 无模型，等权合成（entry 高 & exit 低 & rr 高）
    rank_val = cs_rank(val, feats)
    cands['M0_等权合成'] = rank_val['feilong_entry'] - rank_val['feilong_exit'] + rank_val['feilong_rr']
    # M1: LGB-raw（V1 口径）
    m1 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1, random_state=42)
    m1.fit(cs_rank(fit, feats), fit['fwd'])
    cands['M1_LGB_raw'] = pd.Series(m1.predict(cs_rank(val, feats)), index=val.index)
    # M2: LGB-rank（label 截面 rank，抑制极端值）
    m2 = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1, random_state=42)
    yfit = fit['fwd'].groupby(level='datetime').rank(pct=True)
    m2.fit(cs_rank(fit, feats), yfit)
    cands['M2_LGB_rank'] = pd.Series(m2.predict(cs_rank(val, feats)), index=val.index)
    # M3: LGB-raw 但浅树少棵（正则更强）
    m3 = lgb.LGBMRegressor(n_estimators=60, learning_rate=0.05, num_leaves=7, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1, random_state=42)
    m3.fit(cs_rank(fit, feats), fit['fwd'])
    cands['M3_LGB_shallow'] = pd.Series(m3.predict(cs_rank(val, feats)), index=val.index)

    val_ic = {}
    for name, pred in cands.items():
        v = val.copy()
        v['pred'] = pred
        val_ic[name] = ic_stats(ic_series(v, val_dates))
    print("\n=== 训练内验证段 IC（用于选模型，不看测试期） ===")
    for k, v in val_ic.items():
        print(f"  {k:16s} IC={v['IC均值']:+.4f} ICIR={v['ICIR']:+.3f} t={v['t统计']:+.2f}")

    best = max(val_ic, key=lambda k: abs(val_ic[k]['ICIR']) if val_ic[k] else 0)
    print(f"  → 选定模型: {best}")

    # ---- 用选定模型在测试期打分 ----
    if best == 'M0_等权合成':
        te_rank = cs_rank(panel.loc[te_dates], feats)
        panel['pred'] = te_rank['feilong_entry'] - te_rank['feilong_exit'] + te_rank['feilong_rr']
    else:
        model = {'M1_LGB_raw': m1, 'M2_LGB_rank': m2, 'M3_LGB_shallow': m3}[best]
        te = panel.loc[te_dates].dropna(subset=feats + ['fwd'])
        # 用训练期全量重训（V1 做法保持一致：训练期拟合，测试期预测）
        Xtr = cs_rank(tr_all, feats)
        ytr = tr_all['fwd'].groupby(level='datetime').rank(pct=True) if best == 'M2_LGB_rank' else tr_all['fwd']
        model.fit(Xtr, ytr)
        panel['pred'] = np.nan
        panel.loc[te.index, 'pred'] = model.predict(cs_rank(te, feats))

    te = panel.loc[te_dates]
    ic_te = ic_stats(ic_series(te, te_dates))
    print(f"\n=== 测试期真实 IC（{best}，vs 可交易收益） ===")
    print(f"  IC={ic_te['IC均值']:+.4f} ICIR={ic_te['ICIR']:+.3f} t={ic_te['t统计']:+.2f} "
          f"IC>0占比={ic_te['IC>0占比']:.2%} 天数={ic_te['天数']}")

    # ---- ① 同池等权基准下的 A/B 回测 ----
    bench_eq_te = bq.reindex(te_dates).dropna()
    A = run_backtest(panel, te_dates)
    gateB = reg['regime_scale'].to_dict()
    B = run_backtest(panel, te_dates, gate=gateB)

    keys = ['年化', '年化超额', '最大回撤', '超额最大回撤', 'IR', '超额IR', '日胜率', '超额日胜率', '累计']
    mA = metrics(A['net'], bench_eq_te)
    mB = metrics(B['net'] * B['scale'], bench_eq_te.reindex(B.index))
    mBench = metrics(bench_eq_te.reindex(A.index).dropna())
    print("\n=== ① 同池等权基准下（liquid500 等权），测试期 ===")
    print(f"{'指标':<12}{'A无闸门':>12}{'B有闸门':>12}{'池等权基准':>14}")
    for k in keys:
        print(f"{k:<12}{mA.get(k, np.nan):>12.4f}{mB.get(k, np.nan):>12.4f}{mBench.get(k, np.nan):>14.4f}")
    print(f"{'日均换手':<12}{A['turnover'].mean():>12.4f}{(B['turnover'] * B['scale']).mean():>12.4f}")
    print(f"（对照）沪深300 同期年化 {metrics(hs300.reindex(A.index).dropna())['年化']:.4f} "
          f"—— 与池等权基准的差即 V1 的错配来源")

    # ---- ③ 闸门诊断 ----
    # (a) 闸门理论有效性：池等权基准在各 regime 下的日收益（全 regime 有效期，不只测试期）
    vr = [d for d in all_dates if d in reg.index and pd.notna(reg.loc[d, 'vol_pct'])]
    g = pd.DataFrame({'bench_eq': bq.reindex(vr), 'hs300': hs300.reindex(vr),
                      'regime': reg.loc[vr, 'regime']}).dropna()
    print("\n=== ③-a 闸门理论有效性：池等权基准在各 regime 的日收益（全样本 %d 天） ===" % len(g))
    lines = []
    for rg in ['bullish', 'neutral', 'bearish']:
        s = g[g['regime'] == rg]
        if len(s) < 2:
            print(f"  {rg:8s} 样本不足 ({len(s)})")
            lines.append((rg, len(s), np.nan, np.nan, np.nan, np.nan))
            continue
        t_eq = s['bench_eq'].mean() / (s['bench_eq'].std(ddof=1) / np.sqrt(len(s)))
        t_hs = s['hs300'].mean() / (s['hs300'].std(ddof=1) / np.sqrt(len(s)))
        print(f"  {rg:8s} 天数={len(s):3d}  池等权日均={s['bench_eq'].mean()*100:+.4f}% (t={t_eq:+.2f})  "
              f"沪深300日均={s['hs300'].mean()*100:+.4f}% (t={t_hs:+.2f})  年化池={s['bench_eq'].mean()*252*100:+.2f}%")
        lines.append((rg, len(s), s['bench_eq'].mean() * 252, t_eq, s['hs300'].mean() * 252, t_hs))
    pd.DataFrame(lines, columns=['regime', '天数', '池等权年化', '池t', '沪深300年化', '沪深t']).to_csv(
        OUT / 'feilong_v2_regime_base.csv', index=False)

    # (b) 策略 A 在各 regime 下相对同池基准的超额
    A2 = A.copy()
    A2['regime'] = reg.reindex(A.index)['regime']
    A2['ex'] = A2['net'] - A2['bench_eq']
    print("\n=== ③-b 策略A（无闸门）在各 regime 下相对同池基准的超额 ===")
    bl = []
    for rg in ['bullish', 'neutral', 'bearish']:
        s = A2[A2['regime'] == rg]
        if len(s) < 2:
            print(f"  {rg:8s} 样本不足 ({len(s)})")
            bl.append((rg, len(s), np.nan, np.nan))
            continue
        t_ex = s['ex'].mean() / (s['ex'].std(ddof=1) / np.sqrt(len(s)))
        print(f"  {rg:8s} 天数={len(s):3d}  年化超额={s['ex'].mean()*252*100:+.2f}%  t={t_ex:+.2f}  "
              f"日均净={s['net'].mean()*100:+.4f}%")
        bl.append((rg, len(s), s['ex'].mean() * 252, t_ex))
    pd.DataFrame(bl, columns=['regime', '天数', '年化超额', 't']).to_csv(
        OUT / 'feilong_v2_regime_strategy.csv', index=False)

    # (c) scale 组合扫描（探索性，标注不用于最终选参）
    print("\n=== ③-c scale 组合扫描（探索性：样本小，仅看方向） ===")
    grid = [(1.0, 1.0), (0.5, 1.0), (0.0, 1.0), (0.0, 0.5), (0.5, 0.5), (0.0, 0.75)]
    rows = []
    for bear, neu in grid:
        sc = {}
        for d in te_dates:
            r = reg['regime'].get(d, 'neutral')
            sc[d] = 1.0 if r == 'bullish' else (neu if r == 'neutral' else bear)
        R = run_backtest(panel, te_dates, gate=sc)
        m = metrics(R['net'] * R['scale'], bench_eq_te.reindex(R.index))
        rows.append({'bearish': bear, 'neutral': neu, '年化': m.get('年化'), '年化超额': m.get('年化超额'),
                     '超额IR': m.get('超额IR'), '最大回撤': m.get('最大回撤'),
                     '超额最大回撤': m.get('超额最大回撤'), '仓位均值': R['scale'].mean()})
        print(f"  bear={bear:<4} neutral={neu:<4} 年化超额={m.get('年化超额', float('nan'))*100:+7.2f}%  "
              f"超额IR={m.get('超额IR', float('nan')):+.3f}  回撤={m.get('最大回撤', float('nan'))*100:+.2f}%")
    pd.DataFrame(rows).to_csv(OUT / 'feilong_v2_gate_grid.csv', index=False)

    # ---- 落盘 ----
    A.to_csv(OUT / 'feilong_v2_A_nogate.csv')
    B.to_csv(OUT / 'feilong_v2_B_gate.csv')
    pd.concat([bq.rename('bench_eq'), hs300.rename('hs300')], axis=1).to_csv(OUT / 'feilong_v2_bench.csv')
    summary = {
        '模型选择': best,
        '验证段IC': {k: {kk: round(float(vv), 5) for kk, vv in v.items()} for k, v in val_ic.items()},
        '测试期IC': {k: round(float(v), 5) for k, v in ic_te.items()},
        'A无闸门': {k: round(float(v), 5) for k, v in mA.items()},
        'B有闸门': {k: round(float(v), 5) for k, v in mB.items()},
        '池等权基准': {k: round(float(v), 5) for k, v in mBench.items()},
    }
    (OUT / 'feilong_v2_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print("\n保存: feilong_v2_A_nogate.csv / feilong_v2_B_gate.csv / feilong_v2_gate_grid.csv / "
          "feilong_v2_regime_*.csv / feilong_v2_summary.json")


if __name__ == '__main__':
    main()
