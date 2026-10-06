"""飞龙多因子验证：LGB 组合三因子 × 大盘闸门（开/关对比）。

用项目真实数据（qlib cn_real_full 日线）+ LightGBM 多因子组合，
复现「飞龙 entry/exit/rr」的多因子打分能力，并对比：
    A. 无闸门（飞龙原样，每天满仓 top-K）
    B. 有闸门（bearish空仓 / neutral半仓 / bullish满仓）
输出：年化超额、最大回撤、IR、日胜率、换手、IC，并按 regime 分组。

严格防未来函数：因子只用 ≤t 数据；label = t→t+1 收益；模型只用训练期拟合，测试期预测。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import qlib

from quantaalpha.strategy.feilong import feilong_factors, market_regime, compute_breadth

PROVIDER = 'D:/qqmt/qlib_data/cn_real_full'
START, END = '2024-06-01', '2026-09-30'
FIELDS = ['$open', '$high', '$low', '$close', '$volume', '$factor']
TOPK = 30
COST = 0.0005 + 0.0015  # 开+平，一次性往返成本率


def load():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    idx = D.features(['SH000300'], FIELDS, start_time=START, end_time=END, freq='day').droplevel('instrument').sort_index()
    codes = D.list_instruments(D.instruments('liquid500'), start_time=START, end_time=END, as_list=True)
    df = D.features(codes, FIELDS, start_time=START, end_time=END, freq='day').sort_index()
    panels = {}
    for i, g in df.groupby(level='instrument'):
        if isinstance(g.index, pd.MultiIndex):
            g = g.droplevel('instrument')
        panels[i] = g.sort_index()
    return idx, panels


def build_panel(panels):
    """长表 (datetime, instrument): 三因子 + fwd1收益。"""
    frames = []
    for inst, g in panels.items():
        f = feilong_factors(g)
        d = pd.DataFrame({
            'feilong_entry': f['feilong_entry'],
            'feilong_exit': f['feilong_exit'],
            'feilong_rr': f['feilong_rr'],
            'close': g['$close'],
        })
        d['fwd1'] = d['close'].shift(-1) / d['close'] - 1.0
        d = d.reset_index()
        d['instrument'] = inst
        frames.append(d.set_index(['datetime', 'instrument']))
    return pd.concat(frames)


def run_backtest(panel, dates, bench_ret, gate=None, topk=TOPK):
    """日频多头 top-K 等权。gate: Series(date->scale) 或 None。逐日只看当日预测排前 topk。"""
    dates = list(dates)
    hold = None
    rows = []
    for dt in dates:
        try:
            sub = panel.xs(dt, level='datetime')
        except KeyError:
            continue
        sub = sub.dropna(subset=['pred', 'fwd1'])
        if len(sub) < topk:
            continue
        sel = sub.nlargest(topk, 'pred')
        gross = sel['fwd1'].mean()
        cur = set(sel.index)
        if hold is None:
            turn = 1.0
        else:
            turn = 1.0 - len(cur & hold) / max(len(cur), 1)
        cost = turn * COST
        net = gross - cost
        scale = 1.0 if gate is None else float(gate.get(dt, 1.0))
        rows.append({'date': dt, 'gross': gross, 'net': net, 'turnover': turn,
                     'bench': bench_ret.get(dt, np.nan), 'scale': scale})
        hold = cur
    return pd.DataFrame(rows).set_index('date')


def metrics(rets, bench=None):
    rets = rets.dropna()
    n = len(rets)
    if n == 0:
        return {}
    cum = rets.cumsum()
    dd = (cum - cum.cummax()).min()
    ann = rets.mean() * 252
    std = rets.std(ddof=1)
    ir = (rets.mean() / std * np.sqrt(252)) if std > 0 else np.nan
    out = {'年化': ann, '最大回撤': dd, 'IR': ir, '日胜率': float((rets > 0).mean()),
           '累计': cum.iloc[-1], 'n': n}
    if bench is not None:
        ex = rets - bench.reindex(rets.index).fillna(0)
        ex_cum = ex.cumsum()
        out['年化超额'] = ex.mean() * 252
        out['超额最大回撤'] = (ex_cum - ex_cum.cummax()).min()
    return out


def main():
    idx, panels = load()
    breadth = compute_breadth(panels, 20).reindex(idx.index).ffill()
    reg = market_regime(idx, breadth)
    panel = build_panel(panels)

    # 训练/测试按时间 60/40
    dates = sorted(set(idx.index))
    split = int(len(dates) * 0.6)
    tr_dates, te_dates = dates[:split], dates[split:]
    print(f"训练 {tr_dates[0].date()}~{tr_dates[-1].date()} ({len(tr_dates)}天) | "
          f"测试 {te_dates[0].date()}~{te_dates[-1].date()} ({len(te_dates)}天)")

    # LGB 多因子：截面标准化后训练
    tr = panel.loc[tr_dates].dropna(subset=['feilong_entry', 'feilong_exit', 'feilong_rr', 'fwd1'])
    te = panel.loc[te_dates].dropna(subset=['feilong_entry', 'feilong_exit', 'feilong_rr', 'fwd1'])
    feats = ['feilong_entry', 'feilong_exit', 'feilong_rr']
    # 训练期截面 rank 归一（0~1），避免量纲失衡
    def cs_rank(df):
        return df[feats].groupby(level='datetime').rank(pct=True)
    Xtr, ytr = cs_rank(tr), tr['fwd1']
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42)
    model.fit(Xtr, ytr)
    panel['pred'] = np.nan
    te_idx = te.index
    panel.loc[te_idx, 'pred'] = model.predict(cs_rank(te))

    # 测试期基准：沪深300 日收益
    bench_ret = (idx['$close'].pct_change()).reindex(te_dates)

    # A 无闸门 vs B 有闸门
    A = run_backtest(panel, te_dates, bench_ret, gate=None)
    B = run_backtest(panel, te_dates, bench_ret, gate=reg['regime_scale'].to_dict())

    bench_te = bench_ret.reindex(te_dates)
    print("\n=== 飞龙多因子(LGB三因子) 测试期对比 ===")
    mA = metrics(A['net'], bench_te)
    mB = metrics(B['net'] * B['scale'], bench_te)
    print(f"{'指标':<12}{'A无闸门':>12}{'B有闸门':>12}")
    for k in ['年化', '年化超额', '最大回撤', '超额最大回撤', 'IR', '日胜率', '累计']:
        print(f"{k:<12}{mA.get(k,np.nan):>12.4f}{mB.get(k,np.nan):>12.4f}")
    print(f"{'日均换手':<12}{(A['turnover'].mean()):>12.4f}{(B['turnover'].mean()*B['scale'].mean()):>12.4f}")

    # 测试期 regime 分布
    print("\n=== 测试期 regime 分布 ===")
    print(reg.loc[te_dates, 'regime'].value_counts().to_string())

    # 按 regime 看 A 的表现（验证差行情飞龙差）
    print("\n=== 无闸门策略 A 按测试期 regime 的日收益 ===")
    A2 = A.copy(); A2['regime'] = reg.loc[A.index, 'regime']
    for rg in ['bullish', 'neutral', 'bearish']:
        s = A2[A2['regime'] == rg]
        if len(s) == 0: continue
        ex = (s['net'] - s['bench']).mean() * 252
        print(f"  {rg:8s} 天数={len(s):3d}  日均净={s['net'].mean()*100:+.4f}%  年化超额={ex*100:+.2f}%")

    A.to_csv('data/results/feilong_multi_A_nogate.csv')
    B.to_csv('data/results/feilong_multi_B_gate.csv')
    print("\n保存: feilong_multi_A_nogate.csv / feilong_multi_B_gate.csv")


if __name__ == '__main__':
    main()
