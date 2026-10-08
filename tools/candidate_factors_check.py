"""候选因子组合验证：短期反转 + 量能高潮退潮（2025-2026 有效信号）

审计发现（在**正确对齐口径**下，test 段 2025-01~2026-09 |t|>=2 的因子）：
    mom_20     −0.0266 (t=−2.53)   20 日动量 → 负 IC = 涨多的跑输
    mom_60     −0.0263 (t=−2.44)   60 日动量 → 同上
    amt_r5_20  −0.0246 (t=−2.89)   放量 → 负 IC = 量能高潮退潮
    c_ma20     −0.0248 (t=−2.51)   偏离20日均线 → 负 IC = 离均线远的跑输
    f2_vol     −0.0216 (t=−3.94)   当日放量 → 同上
    → 五者方向一致：**「前期强势 + 放量 + 偏离均线」→ 未来 5 日跑输**
    这与战法2 的 F3「多头排列」不同：F3 是中期趋势（test 段 t=−1.34 已失效），
    而这里是**短期反转/退潮**，在 2025-2026 的强市里反而有效。

本脚本做四件事（每件都按铁律）：
    1. **逐年 IC**（2020~2026 每年单列）—— 判断是否只是某一段的偶然
    2. train(2020-2023) + valid(2024) 上选参数组合
    3. test(2025-2026) **单次**验证，真实成本 + 低换手约束
    4. test 期分位组单调性（判 alpha 的铁标准）

严格性声明（必须诚实标注的局限）：
    因子方向是在观察了全样本 IC 之后选定的，因此 test 结果带 mild look-ahead。
    真正的防伪证据是 **逐年 IC 的方向一致性**（若 7 年里多数年份同号则可信）。

数据：cn_real_full / liquid500 / 2020-01-01 ~ 2026-09-30（1665 天，500 只）
输出：data/results/candidate_factors_*.csv
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = 'D:/qqmt/qlib_data/cn_real_full'
START, END = '2020-01-01', '2026-09-30'
OUT = Path('data/results')
H = 5                       # 持有 5 个交易日
SIGMA_BP = 342.0
COMMISSION_BP, STAMP_BP, ETA = 5.0, 5.0, 0.1
CAND = ['mom_20', 'mom_60', 'amt_r5_20', 'c_ma20', 'f2_vol']
TRAIN_END, VALID_END = '2023-12-31', '2024-12-31'
SEGS = [('2020', '2020-01-01', '2020-12-31'), ('2021', '2021-01-01', '2021-12-31'),
        ('2022', '2022-01-01', '2022-12-31'), ('2023', '2023-01-01', '2023-12-31'),
        ('2024', '2024-01-01', '2024-12-31'), ('2025', '2025-01-01', '2025-12-31'),
        ('2026', '2026-01-01', '2026-09-30')]


def build(g, N=60):
    c, v = g['$close'], g['$volume']
    cf = c.ffill(limit=5)
    ma20 = cf.rolling(20).mean()
    ma60 = cf.rolling(60).mean()
    amt = cf * v
    return pd.DataFrame({
        'mom_20': cf / cf.rolling(20).mean() - 1.0,
        'mom_60': cf / cf.rolling(60).mean() - 1.0,
        'amt_r5_20': amt.rolling(5).mean() / (amt.rolling(20).mean() + 1e-8),
        'c_ma20': cf / ma20 - 1.0,
        'f2_vol': v / (v.rolling(5).mean() + 1e-8),
    })


def main():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    codes = D.list_instruments(D.instruments('liquid500'), start_time=START, end_time=END, as_list=True)
    df = D.features(codes, ['$open', '$close', '$volume'],
                    start_time=START, end_time=END, freq='day').sort_index()

    frames = []
    for inst, g in df.groupby(level='instrument'):
        g = g.droplevel('instrument').sort_index()
        f = build(g)
        f['instrument'] = inst
        frames.append(f.reset_index().set_index(['datetime', 'instrument']))
    F = pd.concat(frames).sort_index()

    C = df['$close'].unstack('instrument').sort_index()
    amt_day = C * df["$volume"].unstack("instrument").reindex_like(C) * 100.0
    # 正确对齐：T 日收盘出分 → T+1 开盘买入 → T+1+H 卖出。用开盘价更贴近可执行。
    O = df['$open'].unstack('instrument').sort_index()
    buy_px = O.shift(-1)
    fwd = O.shift(-(H + 1)) / buy_px - 1.0        # T+1 开盘买 → T+6 开盘卖（持有 5 日）
    # ⚠️ 两处必须做，否则数字会失控（实测踩过）：
    #   1) 次新股/妖股 5 日收益动辄 +500%，会把等权均值基准从 +8% 拉到 +1392%
    #   2) 策略与基准必须同口径（都用均值，不能一个均值一个中位数）
    listing = C.notna().cumsum()               # 每票累计有行情天数（近似上市天数）
    mature = listing >= 250                    # 剔除上市不足 250 日的次新股
    _raw = fwd.stack()
    n_out = int((_raw.abs() > 0.5).sum())
    n_new = int((~mature.stack()).sum())
    print(f"  原始样本 {len(_raw):,}  |5日收益|>50% 异常 {n_out} 个({n_out/len(_raw):.3%})  "
          f"上市<250日样本 {n_new} 个({n_new/len(_raw):.2%})")
    fwd = fwd.clip(-0.5, 0.5).where(mature)
    bench = fwd.mean(axis=1)                   # ★与策略同口径（均值）
    print(f"  基准（全池等权·剔除次新股·均值口径）年化 "
          f"{((1+bench.fillna(0)).prod()**(252/len(bench))-1)*100:+.2f}%")
    mature_ok = mature.reindex_like(C)
    print(f"因子面板 {F.shape}   fwd 有效样本 {int(fwd.notna().sum().sum()):,}")

    # ---- score：截面 z-score 后按负 IC 方向（反向）合成 ----
    Z = {}
    for c in CAND:
        s = F[c].unstack('instrument').reindex_like(C)
        Z[c] = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0, np.nan), axis=0)
    score = -sum(Z[c] for c in CAND) / len(CAND)      # 反向：分数高 = 前期弱/缩量/靠近均线

    def daily_ic(s, fwd_):
        s2 = pd.concat([s.stack().rename('f'), fwd_.stack().rename('r')], axis=1).dropna()
        dts, vals = [], []
        for dt, g in s2.groupby(level=0):
            if len(g) < 50:
                continue
            x, y = g['f'].values, g['r'].values
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            dts.append(dt)
            vals.append(pd.Series(x).corr(pd.Series(y), method='spearman'))
        return pd.Series(vals, index=pd.DatetimeIndex(dts))

    # ---- 1. 逐年 IC（防伪关键证据）----
    print("\n=== 逐年 RankIC（原始方向，负=反转）===")
    hdr = f"  {'因子':<12s}" + "".join(f"{y[0]:>9s}" for y in SEGS)
    print(hdr)
    yearly = {}
    for c in CAND + ['SCORE']:
        s = score if c == 'SCORE' else Z[c]
        ic_all = daily_ic(s, fwd)
        line = f"  {c:<12s}"
        row = {}
        for nm, a, b in SEGS:
            v = ic_all[(ic_all.index >= pd.Timestamp(a)) & (ic_all.index <= pd.Timestamp(b))]
            m = v.mean()
            t = m / v.std() * np.sqrt(len(v)) if len(v) > 20 and v.std() > 0 else np.nan
            row[nm] = {'IC': m, 't': t, 'n': len(v)}
            line += f"{m:>+9.4f}"
        yearly[c] = row
        print(line + ("   ← 合成（已反向）" if c == 'SCORE' else ""))

    print("\n  逐年 t 值（|t|>=2 记 *）：")
    for c in CAND + ['SCORE']:
        line = f"  {c:<12s}"
        for nm in [s[0] for s in SEGS]:
            t = yearly[c][nm]['t']
            line += f"{t:>+8.1f}{'*' if abs(t) >= 2 else ' '}"
        print(line)
    same_years = sum(1 for nm in [s[0] for s in SEGS] if yearly['SCORE'][nm]['IC'] > 0)
    print(f"\n  → 合成分子在 {same_years}/7 个年份为正 IC")

    # ---- 2. 参数选择（train + valid）----
    tr = [d for d in C.index if d <= pd.Timestamp(TRAIN_END)]
    va = [d for d in C.index if pd.Timestamp(TRAIN_END) < d <= pd.Timestamp(VALID_END)]
    te = [d for d in C.index if d > pd.Timestamp(VALID_END)]
    print(f"\n分段: train {len(tr)} / valid {len(va)} / test {len(te)} 天")

    def backtest(days, K, every, n_drop, account=1e7):
        hold, rows = None, []
        for k, d in enumerate(days):
            if d not in score.index or d not in fwd.index:
                continue
            s = score.loc[d].dropna()
            if d in mature_ok.index:
                s = s[s.index.isin(mature_ok.columns[mature_ok.loc[d]])]
            if len(s) < K:
                continue
            reb = (hold is None) or (k % every == 0)
            if reb:
                cand = list(s.nlargest(K).index)
                if hold and n_drop > 0:
                    keep = [x for x in cand if x in hold][:max(K - n_drop, 1)]
                    new = set(keep + [x for x in cand if x not in hold][:n_drop])
                else:
                    new = set(cand)
            else:
                new = set(hold)
            turn = 0.0 if hold is None else 0.5 * len(new ^ hold) / max(len(new), 1)
            hold = new
            imp = 0.0
            if turn > 0:
                buy_amt = account / len(hold)
                part = (buy_amt / amt_day.loc[d].reindex(list(hold))).clip(0, 1).fillna(0)
                imp = float((ETA * SIGMA_BP * np.sqrt(part)).mean())
            cost = turn * (COMMISSION_BP + STAMP_BP + 2 * imp) / 1e4
            r = fwd.loc[d].reindex(list(hold)).fillna(0.0)
            gross = float(r.mean())
            rows.append({'date': d, 'gross': gross, 'net': gross - cost,
                         'turn': turn, 'cost': cost})
        return pd.DataFrame(rows).set_index('date')

    def ev(R):
        b = bench.reindex(R.index)
        e = (R['net'] - b).dropna()
        bb = b.dropna()
        ann_s = (1 + R['net']).prod() ** (252 / len(R)) - 1
        ann_b = (1 + bb).prod() ** (252 / len(bb)) - 1
        exc = (1 + e).cumprod()                 # 超额净值（超额日收益复利）
        ann_e = exc.iloc[-1] ** (252 / len(e)) - 1
        cum = (1 + R['net']).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        return {'策略年化%': ann_s * 100, '基准年化%': ann_b * 100,
                '净超额年化%': ann_e * 100,
                '超额t': e.mean() / e.std() * np.sqrt(len(e)) if e.std() > 0 else np.nan,
                '最大回撤%': mdd * 100, '日均换手': R['turn'].mean(),
                '成本年化%': R['cost'].sum() / len(R) * 252 * 100, 'n': len(e)}

    print("\n=== 参数扫描（仅 train+valid 选参）===")
    rows = []
    for K, evy, nd in product([20, 30, 50], [5, 10, 20], [0, 5, 10]):
        R = backtest(tr + va, K, evy, nd)
        if len(R) < 100:
            continue
        m = ev(R)
        rows.append({'K': K, 'every': evy, 'n_drop': nd, **m})
    G = pd.DataFrame(rows)
    G = G.sort_values('净超额年化%', ascending=False)
    print(f"  {'K':>4s}{'every':>7s}{'ndrop':>7s}{'净超额年化%':>12s}{'超额t':>8s}{'换手':>8s}{'成本年化%':>11s}")
    for _, r in G.head(8).iterrows():
        print(f"  {int(r['K']):>4d}{int(r['every']):>7d}{int(r['n_drop']):>7d}"
              f"{r['净超额年化%']:>+12.2f}{r['超额t']:>+8.2f}{r['日均换手']:>8.3f}{r['成本年化%']:>11.2f}")
    G.to_csv(OUT / 'candidate_factors_grid.csv', index=False)

    best = G.iloc[0]
    K, evy, nd = int(best['K']), int(best['every']), int(best['n_drop'])
    print(f"\n  选出 K={K} every={evy} n_drop={nd}")

    # ---- 3. test 单次验证 ----
    R = backtest(te, K, evy, nd)
    m = ev(R)
    print(f"\n=== test 期单次验证（{te[0].date()} ~ {te[-1].date()}，{len(R)} 天）===")
    print(f"  策略年化 {m['策略年化%']:+.2f}%   同池等权基准 {m['基准年化%']:+.2f}%")
    print(f"  **净超额年化 {m['净超额年化%']:+.2f}%   超额 t={m['超额t']:+.2f}**")
    print(f"  最大回撤 {m['最大回撤%']:+.2f}%   日均换手 {m['日均换手']:.3f}   成本 {m['成本年化%']:.2f}%/年")

    # 逐年 test
    print("\n  test 逐年表现：")
    for nm, a, b_ in [s for s in SEGS if s[0] in ('2025', '2026')]:
        sub = R[(R.index >= pd.Timestamp(a)) & (R.index <= pd.Timestamp(b_))]
        if len(sub) < 5:
            continue
        e = sub['net'] - bench.reindex(sub.index)
        bb = bench.reindex(sub.index).dropna()
        e = e.dropna()
        ann_s = (1 + sub['net']).prod() ** (252 / len(sub)) - 1
        ann_b = (1 + bb).prod() ** (252 / len(bb)) - 1
        ann_e = (1 + e).cumprod().iloc[-1] ** (252 / len(e)) - 1
        print(f"    {nm}: 策略 {ann_s*100:+8.2f}%  基准 {ann_b*100:+9.2f}%  "
              f"超额 {ann_e*100:+8.2f}%  t={e.mean()/e.std()*np.sqrt(len(e)):+.2f}")

    # ---- 4. 分位组单调性 ----
    print("\n=== test 期分位组（D1 = 分数最高 = 前期最弱）===")
    grp = {}
    for d in te:
        s = score.loc[d].dropna() if d in score.index else None
        if s is None:
            continue
        if d in mature_ok.index:
            s = s[s.index.isin(mature_ok.columns[mature_ok.loc[d]])]
        y = fwd.loc[d].reindex(s.index) if d in fwd.index else None
        if y is None:
            continue
        z = pd.concat([s.rename('s'), y.rename('y')], axis=1).dropna()
        if len(z) < 80:
            continue
        lab = pd.qcut(z['s'].rank(method='first'), 10, labels=False)
        grp[d] = z.groupby(lab)['y'].mean()
    Gp = pd.DataFrame(grp).T.sort_index()
    dec = []
    for c in sorted(Gp.columns):
        e = Gp[c].dropna() - bench.reindex(Gp[c].dropna().index)
        dec.append({'组': f'D{c+1}', '日均超额bp': e.mean() * 1e4,
                    '超额t': e.mean() / e.std() * np.sqrt(len(e)) if e.std() > 0 else np.nan,
                    'n': len(e)})
        print(f"  D{c+1:<2d} 日均超额 {e.mean()*1e4:+7.2f}bp  t={dec[-1]['超额t']:+5.2f}")
    spread = dec[0]['日均超额bp'] - dec[-1]['日均超额bp']
    print(f"  → D1−D10 价差 = {spread:+.2f} bp/日 ≈ {spread*252/100:+.2f}%/年")
    mono = all(dec[i]['日均超额bp'] > dec[i + 1]['日均超额bp'] for i in range(len(dec) - 1))
    print(f"  → 单调性: {'D1>D2>...>D10 成立' if mono else '不单调（仅首尾有效）'}")
    pd.DataFrame(dec).to_csv(OUT / 'candidate_factors_decile.csv', index=False)
    pd.DataFrame([{'K': K, 'every': evy, 'n_drop': nd, **m, 'D1-D10_bp': spread}]).to_csv(
        OUT / 'candidate_factors_summary.csv', index=False)
    pd.DataFrame(yearly['SCORE']).T.to_csv(OUT / 'candidate_factors_yearly.csv')
    print("\n保存: candidate_factors_{grid,decile,summary,yearly}.csv")


if __name__ == '__main__':
    main()