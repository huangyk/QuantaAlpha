"""对齐审计：把「因子收益口径」的影响量化 —— 修正前后逐因子对比。

审计起因（2026-10-07）：
    `tools/eval_factors.py`（项目所有因子评估的主脚本）原式用
        fwd = close.shift(-h) / close        ← 因子在 T 日收盘算出，却拿 T 日收盘价当买入价
    这是**未来函数**：T 日收盘才知道信号，无法用 T 日收盘价成交。
    正确口径：
        fwd = close.shift(-(h+1)) / close.shift(-1)   ← T+1 收盘买入，持有 h 天
    本脚本对同一批因子用两种口径各算一遍 RankIC，直接量化虚高幅度。

顺带审计历史结论是否受影响：
    verify_feilong_v2.py    : open.shift(-2)/open.shift(-1)     → T+1 开盘买，**正确**
    warfare2_score_study.py: raw_open.shift(-2)/raw_open.shift(-1) → **正确**
    verify_feilong_multifactor.py / verify_feilong.py: close.shift(-1)/close → **前视**
    oos_validate.py        : Ref(close,-(h+1))/Ref(close,-h)   → **正确**

数据：cn_real_full / liquid500 / 2020-01-01 ~ 2026-09-30
分段：train 2020-2023 / valid 2024 / test 2025-2026
输出：data/results/alignment_audit.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = 'D:/qqmt/qlib_data/cn_real_full'
START, END = '2020-01-01', '2026-09-30'
OUT = Path('data/results')
H = 5                      # 持有 5 个交易日（与 eval_factors 默认一致）
SEG = {'train': ('2020-01-01', '2023-12-31'),
       'valid': ('2024-01-01', '2024-12-31'),
       'test':  ('2025-01-01', '2026-09-30')}


def load():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    codes = D.list_instruments(D.instruments('liquid500'), start_time=START, end_time=END, as_list=True)
    df = D.features(codes, ['$open', '$high', '$low', '$close', '$volume'],
                    start_time=START, end_time=END, freq='day').sort_index()
    return df


def build_factors(g):
    """在单只票的日线序列上构造一批标准因子（停牌日先 ffill 补价再算滚动）。"""
    o, h, l, c, v = g['$open'], g['$high'], g['$low'], g['$close'], g['$volume']
    cf = c.ffill(limit=5)
    ret = c.pct_change()
    f = {}
    # 动量 / 反转
    for n in (1, 5, 20, 60, 120):
        f[f'mom_{n}'] = c / cf.shift(n) - 1.0
    # 波动率
    f['vol_20'] = ret.rolling(20).std()
    f['vol_60'] = ret.rolling(60).std()
    f['vol_ratio'] = f['vol_20'] / (f['vol_60'] + 1e-8)
    # 量能
    amt = cf * v
    f['amt_r5_20'] = amt.rolling(5).mean() / (amt.rolling(20).mean() + 1e-8)
    f['vol_r5_20'] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-8)
    # 位置
    lo, hi = cf.rolling(60).min(), cf.rolling(60).max()
    f['pos_60'] = (cf - lo) / (hi - lo + 1e-8)
    f['pos_20'] = (cf - cf.rolling(20).min()) / (cf.rolling(20).max() - cf.rolling(20).min() + 1e-8)
    # 均线关系
    ma20, ma60 = cf.rolling(20).mean(), cf.rolling(60).mean()
    f['c_ma20'] = cf / ma20 - 1.0
    f['ma20_ma60'] = ma20 / ma60 - 1.0
    # 振幅
    f['atr_ratio'] = (h - l).rolling(20).mean() / (cf + 1e-8)
    # 日内收益 / 隔夜收益
    f['intraday'] = (cf / o - 1.0).rolling(5).mean()
    f['overnight'] = (o / cf.shift(1) - 1.0).rolling(5).mean()
    # 战法2 三因子（修复停牌 NaN 版）
    f['f3_bull'] = (ma20 - ma60) / (ma60 + 1e-8)
    f['f4_pullback'] = -(cf - ma20).abs() / (ma20 + 1e-8)
    f['f2_vol'] = v.rolling(5).mean() / (v - v).abs() if False else v / (v.rolling(5).mean() + 1e-8)
    return pd.DataFrame(f)


def main():
    print("载入日线 ...")
    df = load()
    print(f"  {len(df):,} 行  {df.index.get_level_values('instrument').nunique()} 只")

    # ---- 构造因子面板（长表）----
    frames = []
    for i, (inst, g) in enumerate(df.groupby(level='instrument')):
        g = g.droplevel('instrument') if isinstance(g.index, pd.MultiIndex) else g
        g = g.sort_index()
        f = build_factors(g)
        f['instrument'] = inst
        f = f.reset_index().set_index(['datetime', 'instrument'])
        frames.append(f)
    F = pd.concat(frames).sort_index()
    print(f"  因子面板: {F.shape}")

    # ---- 两种收益口径 ----
    C = df['$close'].unstack('instrument').sort_index()
    fwd_old = C.shift(-H) / C - 1.0            # ✗ 前视：用 T 日收盘价买
    fwd_new = C.shift(-(H + 1)) / C.shift(-1) - 1.0   # ✓ T+1 收盘买
    print(f"  fwd_old / fwd_new 样本数: {int(fwd_old.notna().sum().sum())} / {int(fwd_new.notna().sum().sum())}")

    def rank_ic(fac_long, fwd):
        s = pd.concat([fac_long.rename('f'), fwd.stack().rename('r')], axis=1).dropna()
        out = []
        for dt, g in s.groupby(level=0):
            if len(g) < 50:
                continue
            x, y = g['f'].values, g['r'].values
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            out.append(pd.Series(x).corr(pd.Series(y), method='spearman'))
        return pd.Series(out)

    print("\n=== 逐因子 RankIC 对比（持有 5 日）===")
    print(f"  {'因子':<14s}{'口径A(前视)':>12s}{'t_A':>7s}{'口径B(正确)':>12s}{'t_B':>7s}{'虚高倍数':>10s}")
    rows = []
    for col in F.columns:
        fl = F[col]
        ic_old, ic_new = rank_ic(fl, fwd_old), rank_ic(fl, fwd_new)
        if len(ic_old) < 50 or len(ic_new) < 50:
            continue

        def stat(s):
            t = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else np.nan
            return s.mean(), t
        m_a, t_a = stat(ic_old)
        m_b, t_b = stat(ic_new)
        infl = m_a / m_b if abs(m_b) > 1e-6 else np.nan
        rows.append({'factor': col, 'IC_old': m_a, 't_old': t_a, 'IC_new': m_b, 't_new': t_b,
                     'inflate': infl})
        flag = ' ← 前视口径把它变成了假信号' if abs(t_a) >= 2 and abs(t_b) < 2 and m_a * m_b > 0 else ''
        print(f"  {col:<14s}{m_a:>+12.4f}{t_a:>+7.2f}{m_b:>+12.4f}{t_b:>+7.2f}"
              f"{(infl if np.isfinite(infl) else 0):>10.2f}x{flag}")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'alignment_audit.csv', index=False)

    # ---- 汇总：有多少因子在修正后失去显著性 ----
    print("\n=== 汇总 ===")
    lost = R[(R['t_old'].abs() >= 2) & (R['t_new'].abs() < 2)]
    gained = R[(R['t_old'].abs() < 2) & (R['t_new'].abs() >= 2)]
    print(f"  前视口径下 |t|>=2 的因子: {(R['t_old'].abs()>=2).sum()} 个")
    print(f"  正确口径下 |t|>=2 的因子: {(R['t_new'].abs()>=2).sum()} 个")
    print(f"  因口径修正而**失去显著性**: {len(lost)} 个 → {list(lost['factor'])}")
    print(f"  因口径修正而**新变得显著**: {len(gained)} 个 → {list(gained['factor'])}")
    same = ((R['t_old'] * R['t_new']) > 0).sum()
    flip = ((R['t_old'] * R['t_new']) < 0).sum()
    print(f"  方向一致 {same} 个 / 方向反转 {flip} 个")

    # ---- test 段单独看（当前决策依据）----
    print("\n=== test 段（2025-01 ~ 2026-09）单因子 RankIC ===")
    print(f"  {'因子':<14s}{'口径A':>10s}{'t_A':>7s}{'口径B(正确)':>13s}{'t_B':>7s}")
    sub_rows = []
    for col in F.columns:
        fl = F[col]
        s_new = pd.concat([fl.rename('f'), fwd_new.stack().rename('r')], axis=1).dropna()
        s_old = pd.concat([fl.rename('f'), fwd_old.stack().rename('r')], axis=1).dropna()
        res = {}
        for nm, ss in (('A', s_old), ('B', s_new)):
            ss = ss[(ss.index.get_level_values('datetime') >= pd.Timestamp(SEG['test'][0]))
                    & (ss.index.get_level_values('datetime') <= pd.Timestamp(SEG['test'][1]))]
            ics = []
            for dt, g in ss.groupby(level=0):
                if len(g) < 50:
                    continue
                ics.append(pd.Series(g['f'].values).corr(pd.Series(g['r'].values), method='spearman'))
            ser = pd.Series(ics)
            res[nm] = (ser.mean(), ser.mean() / ser.std() * np.sqrt(len(ser))) if len(ser) > 30 and ser.std() > 0 else (np.nan, np.nan)
        if np.isnan(res['B'][0]):
            continue
        sub_rows.append({'factor': col, 'test_IC_old': res['A'][0], 'test_t_old': res['A'][1],
                         'test_IC_new': res['B'][0], 'test_t_new': res['B'][1]})
        mark = '  ★正确口径下依然显著' if abs(res['B'][1]) >= 2 else ''
        print(f"  {col:<14s}{res['A'][0]:>+10.4f}{res['A'][1]:>+7.2f}"
              f"{res['B'][0]:>+13.4f}{res['B'][1]:>+7.2f}{mark}")
    pd.DataFrame(sub_rows).to_csv(OUT / 'alignment_audit_test.csv', index=False)
    sig = [r for r in sub_rows if abs(r['test_t_new']) >= 2]
    print(f"\n  → test 段在**正确口径**下 |t|>=2 的因子: {[r['factor'] for r in sig] or '无'}")


if __name__ == '__main__':
    main()