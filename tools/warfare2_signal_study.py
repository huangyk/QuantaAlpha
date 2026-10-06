"""战法2（均线回踩低吸）信号层事件研究 —— 回答「什么时候有信号、信号值不值钱」。

信号定义（严格照 `data/factorlib/tdx_warfare2_factors.json`，T 日收盘确认）：
    F3 过滤：MA20 > MA60
    F4 信号：|C-MA20|/MA20 < 3% 且 C > MA20 且 昨日 C <= 昨日 MA20（金叉回踩）
    F2 确认：VOL > MA5(VOL)*1.5 且 VOL > 昨日 VOL
    → signal = F3 & F4 & F2

执行口径（可交易）：T 日收盘出信号 → **T+1 开盘**买入 → 持有 h 日后 **开盘**卖出
    ret_h = open(T+1+h) / open(T+1) - 1

基准：**同池等权** —— 同一 T 日池内全部个股的同期 ret_h 均值（与选股池同口径，
避免此前飞龙 V1「池 vs 沪深300」制造假超额的教训）。

输出：分持有期/分年度的信号统计（样本数、收益、超额、t值、胜率），
以及信号频次分布。落盘 data/results/warfare2_signal_study.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = 'D:/qqmt/qlib_data/cn_real_full'
START, END = '2020-01-01', '2026-09-30'
FIELDS = ['$open', '$high', '$low', '$close', '$volume']
HOLDS = [1, 2, 3, 5, 10, 20]
OUT = Path('data/results')


def load():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    codes = D.list_instruments(D.instruments('liquid500'), start_time=START, end_time=END, as_list=True)
    df = D.features(codes, FIELDS, start_time=START, end_time=END, freq='day').sort_index()
    panels = {}
    for i, g in df.groupby(level='instrument'):
        panels[i] = g.droplevel('instrument').sort_index() if isinstance(g.index, pd.MultiIndex) else g.sort_index()
    return panels


def compute(panels):
    """返回长表 (datetime, instrument): signal + 各持有期前瞻收益 + 信号相关特征。

    ⚠️ 同 score_study 的修复：先 ffill(limit=5) 补停牌价再算均线，否则
    rolling(60, min_periods=60) 会因单个停牌日让其后 59 天全为 NaN，
    信号被大量漏掉。前瞻收益仍用**原始 open**，保证只统计真实可交易日。
    """
    frames = []
    for inst, g in panels.items():
        raw_open = g['$open'].copy()
        g = g.ffill(limit=5)
        c, o, v = g['$close'], g['$open'], g['$volume']
        ma20 = c.rolling(20, min_periods=20).mean()
        ma60 = c.rolling(60, min_periods=60).mean()
        ma5v = v.rolling(5, min_periods=5).mean()
        f3 = ma20 > ma60
        f4 = ((c - ma20).abs() / ma20 < 0.03) & (c > ma20) & (c.shift(1) <= ma20.shift(1))
        f2 = (v > ma5v * 1.5) & (v > v.shift(1))
        sig = (f3 & f4 & f2).fillna(False)

        d = pd.DataFrame(index=g.index)
        d['signal'] = sig
        # 可交易收益：T+1 开盘买 → T+1+h 开盘卖
        o_next = raw_open.shift(-1)
        for h in HOLDS:
            d[f'ret{h}'] = raw_open.shift(-(1 + h)) / o_next - 1.0
        # 信号质量特征（用于后续闸门切片）
        d['dist_ma20'] = (c - ma20) / ma20
        d['vol_ratio'] = v / ma5v
        d['ret20'] = c / c.shift(20) - 1.0
        d['dd60'] = c / c.rolling(60, min_periods=60).max() - 1.0
        d['amp'] = (g['$high'] - g['$low']) / c.shift(1)
        d['turnover_vol'] = v / v.rolling(60, min_periods=60).mean()
        d['rv20'] = c.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)
        d['amount'] = v * c                       # 成交额（流动性代理，无市值字段）
        d['gap'] = o.shift(-1) / c - 1.0          # T+1 开盘相对 T 收盘的跳空
        d['close'] = c
        d = d.reset_index()
        d['instrument'] = inst
        frames.append(d.set_index(['datetime', 'instrument']))
    return pd.concat(frames).sort_index()


def stat(sig_ret, all_ret):
    """给定信号组收益与同期池等权基准（同样本对齐），算超额统计。"""
    s = sig_ret.dropna()
    if len(s) < 5:
        return {}
    ex = s - all_ret.reindex(s.index)
    ex = ex.dropna()
    t_ex = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if ex.std(ddof=1) > 0 else np.nan
    t_abs = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if s.std(ddof=1) > 0 else np.nan
    return {'样本数': len(s), '收益均值': s.mean(), '中位数': s.median(), '胜率': float((s > 0).mean()),
            '绝对t': t_abs, '基准均值': ex.mean() * 0 + all_ret.reindex(s.index).mean(),
            '超额均值': ex.mean(), '超额t': t_ex, '超额胜率': float((ex > 0).mean())}


def main():
    panels = load()
    print(f"载入 {len(panels)} 只标的")
    df = compute(panels)
    df.to_pickle('data/results/_warfare2_signal_panel.pkl')

    dates = sorted(set(df.index.get_level_values('datetime')))
    print(f"区间 {dates[0].date()} ~ {dates[-1].date()} ({len(dates)} 天)")

    sig = df[df['signal']]
    print(f"\n信号总数 {len(sig)} ；有信号的交易日 {sig.index.get_level_values('datetime').nunique()} 天"
          f"（占比 {sig.index.get_level_values('datetime').nunique()/len(dates):.1%}）；"
          f"日均信号 {len(sig)/len(dates):.2f} 只")

    # 池等权基准：每日全部个股各持有期收益的均值
    bench = df.groupby(level='datetime')[[f'ret{h}' for h in HOLDS]].mean()

    print("\n=== 战法2 信号前瞻收益（T+1开盘买 → 持有h日开盘卖），vs 同池等权 ===")
    rows = []
    for h in HOLDS:
        col = f'ret{h}'
        b = df.join(bench[col].rename('bench'), on='datetime')['bench']
        m = stat(sig[col], b)
        rows.append({'持有期': h, **{k: (round(float(v), 5) if isinstance(v, (int, float, np.floating)) else v)
                                     for k, v in m.items()}})
        if m:
            print(f"  h={h:<3d} 样本={m['样本数']:>6d}  收益均值={m['收益均值']*100:+6.2f}%  "
                  f"中位={m['中位数']*100:+6.2f}%  胜率={m['胜率']:.1%}  "
                  f"超额={m['超额均值']*100:+6.2f}%  超额t={m['超额t']:+.2f}  超额胜率={m['超额胜率']:.1%}")
    pd.DataFrame(rows).to_csv(OUT / 'warfare2_signal_study.csv', index=False)

    # 分年度稳健性（h=5 与 h=1）
    print("\n=== 分年度（h=5） ===")
    yr_rows = []
    for y, g in sig.groupby(sig.index.get_level_values('datetime').year):
        b = df.join(bench['ret5'].rename('bench'), on='datetime')['bench']
        m = stat(g['ret5'], b)
        if not m:
            continue
        yr_rows.append({'年份': y, '样本数': m['样本数'], '收益均值': m['收益均值'],
                        '超额均值': m['超额均值'], '超额t': m['超额t'], '胜率': m['胜率']})
        print(f"  {y}  样本={m['样本数']:>5d}  收益={m['收益均值']*100:+6.2f}%  "
              f"超额={m['超额均值']*100:+6.2f}%  t={m['超额t']:+.2f}  胜率={m['胜率']:.1%}")
    pd.DataFrame(yr_rows).to_csv(OUT / 'warfare2_signal_by_year.csv', index=False)

    # 信号频次分布
    cnt = sig.groupby(level='datetime').size().reindex(dates, fill_value=0)
    print(f"\n=== 每日信号个数分布 ===")
    print(f"  0 个: {(cnt==0).mean():.1%} 天 | 1-3 个: {((cnt>=1)&(cnt<=3)).mean():.1%} 天 | "
          f"4-10 个: {((cnt>=4)&(cnt<=10)).mean():.1%} 天 | >10 个: {(cnt>10).mean():.1%} 天 | "
          f"最大 {cnt.max()} 个")
    print("\n保存: warfare2_signal_study.csv / warfare2_signal_by_year.csv / _warfare2_signal_panel.pkl")


if __name__ == '__main__':
    main()
