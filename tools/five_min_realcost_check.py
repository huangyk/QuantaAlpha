"""用「真实成本 + 换手约束」口径重跑 5min 低波动族因子 —— 验证能否转正。

背景（昨夜三实验的结论）：
    - 一次**换股**真实成本 = 10bp 固定 + 2×市场冲击(9.5~30bp) = **29~70bp**
    - 可承受换股率：alpha 20%/年 → 0.056/日；alpha 50% → 0.141/日
    - 之前 5min 因子层换手 **9.6x/日** → 必死；这里把它压到 ≤0.15/日再看能否转正

因子（昨夜 mining_5m_final.md 单因子 RankIC Top，均「三段同号」）：
    f1 Volatility_Adj_Efficiency_20D  =+0.0236  波动率调整后的效率（越高越好）
    f2 High_Day_Rank_Divergence_15D   =−0.0231  排名变化波动 → 取负
    f3 Return_Rank_Consistency_10D    =−0.0229  排名水平波动 → 取负
    f4 Vol_Scaled_Return_Std_10D      =−0.0123  单位成交额收益率波动 → 取负
    共识：「波动/不稳定 → 次日收益差」，四个因子同向。

执行口径（与昨夜执行层结论一致）：
    - T 日 15:00 收盘算分 → **T+1 09:35 开盘买**（开盘比 VWAP 便宜 12bp）
    - 持有 1 个交易日 → T+2 09:35 卖出
    - 换手控制三件套：持仓数 K / 调仓周期 every / 缓冲带 n_drop（只换最差的 n_drop 只）
    - 成本：10bp 固定 + 2 × 真实 participation 冲击（σ=342bp 实测值）

严格性（遵守铁律）：
    **参数只在 train+valid 上选，test 期单次验证**，不从测试期挑最优点。
    基准 = 同池等权（每日再平衡）。

分段（重划，保证 test 有足够天数）：
    train 2026-01-05~04-30 / valid 05-01~06-30 / test 07-01~09-30

输出：data/results/five_min_realcost_*.csv
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = 'D:/qqmt/qlib_data/cn_5m_full'
START, END = '2026-01-05', '2026-09-30'
OUT = Path('data/results')
SIGMA_BP = 342.0          # 实测池内个股日波动中位
COMMISSION_BP = 5.0
STAMP_BP = 5.0
ETA = 0.1
N_BARS = 48
TRAIN_END, VALID_END = '2026-04-30', '2026-06-30'
FACTORS = ['f1_efficiency', 'f2_rank_div', 'f3_rank_std', 'f4_vol_scaled']
SIGN = {'f1_efficiency': +1, 'f2_rank_div': -1, 'f3_rank_std': -1, 'f4_vol_scaled': -1}


def load_panels():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    codes = D.list_instruments(D.instruments('liquid500'), start_time=START, end_time=END,
                               freq='5min', as_list=True)
    df = D.features(codes, ['$open', '$high', '$low', '$close', '$volume'],
                    start_time=START, end_time=END, freq='5min').sort_index()
    d = df.reset_index()
    d['date'] = pd.to_datetime(d['datetime']).dt.normalize()
    d = d.sort_values(['instrument', 'datetime'])
    d['slot'] = d.groupby(['instrument', 'date']).cumcount()
    cnt = d.groupby(['instrument', 'date'])['slot'].transform('max') + 1
    d = d[cnt == N_BARS]
    out = {}
    for f in ['$open', '$high', '$low', '$close', '$volume']:
        out[f] = d.pivot_table(index=['date', 'slot'], columns='instrument', values=f)
    return out


def compute_factors(p):
    C, O, H, L, V = p['$close'], p['$open'], p['$high'], p['$low'], p['$volume']
    R = C.pct_change()
    R_rank = R.rank(axis=1)
    f = {}
    # f1 = TS_MEAN(|C-O|/(H-L), 20) / TS_STD(return, 20)
    A = (C - O).abs() / (H - L + 1e-8)
    f['f1_efficiency'] = A.rolling(20).mean() / (R.rolling(20).std() + 1e-8)
    # f2 = TS_STD(RANK(ret) - DELAY(RANK(ret),5), 15)     （方向取负）
    f['f2_rank_div'] = (R_rank - R_rank.shift(5)).rolling(15).std()
    # f3 = TS_STD(RANK(ret), 10)                            （方向取负）
    f['f3_rank_std'] = R_rank.rolling(10).std()
    # f4 = TS_STD(ret,10) / TS_MEAN(volume,10)              （方向取负）
    f['f4_vol_scaled'] = R.rolling(10).std() / (V.rolling(10).mean() + 1e-8)
    return f, R


def daily_scores(f):
    """取每天 15:00（slot 47）的因子值，做截面 z-score 后按符号等权合成。"""
    z = {}
    for k in FACTORS:
        d = f[k].xs(N_BARS - 1, level='slot')
        mu, sd = d.mean(axis=1), d.std(axis=1).replace(0, np.nan)
        z[k] = d.sub(mu, axis=0).div(sd, axis=0) * SIGN[k]
    score = sum(z[k] for k in FACTORS) / len(FACTORS)
    return score, z


def main():
    print("载入 5min 数据并计算因子 ...")
    p = load_panels()
    f, R = compute_factors(p)
    score, z = daily_scores(f)
    C = p['$close']
    V = p['$volume']
    amt_day = (C * V).groupby(level='date').sum()          # 每日全池成交额矩阵
    dates = sorted(score.index)
    print(f"  交易日 {len(dates)} 天  {dates[0].date()} ~ {dates[-1].date()}  "
          f"因子面板 {score.shape}")

    # ---- 单因子 RankIC（铁律口径）：**符号不在此处假定**，先看原始方向 ----
    print("\n=== 单因子 RankIC 原始方向（日频，T+1 09:35 买 → T+2 09:35 卖）===")
    print("   注意：原记录 mining_5m_final.md 的方向是「因子表直接秩相关」口径，")
    print("   本脚本用「每日 15:00 因子值 → 次日开盘到再次日开盘收益」，口径不同，方向可能相反。")
    buy = C.xs(0, level='slot')                            # 每日 09:35
    # ⚠️ 对齐铁律：score[d] 来自 T 日 15:00 收盘，T+1 09:35 才能买，T+2 09:35 卖。
    #    所以 fwd 必须是 buy[d+2]/buy[d+1]，而不是 buy[d+1]/buy[d]
    #    （后者会让 T 日收盘的信号去买 T 日开盘 = 未来函数，第一版就是这么错的，已作废）
    fwd = buy.shift(-2) / buy.shift(-1) - 1.0

    tr = [d for d in dates if d <= pd.Timestamp(TRAIN_END)]
    va = [d for d in dates if pd.Timestamp(TRAIN_END) < d <= pd.Timestamp(VALID_END)]
    te = [d for d in dates if d > pd.Timestamp(VALID_END)]
    print(f"\n分段: train {len(tr)} 天 / valid {len(va)} 天 / test {len(te)} 天 "
          f"(test {te[0].date()} ~ {te[-1].date()})")

    def ic_stats(s):
        v = s.dropna()
        n = len(v)
        return v.mean(), (v.mean() / v.std() * np.sqrt(n) if v.std() > 0 else np.nan), n

    def fac_ic(k, days):
        s = pd.concat([z[k].stack(), fwd.stack()], axis=1).dropna()
        s.columns = ['f', 'y']
        s = s[s.index.get_level_values(0).isin(days)]
        return s.groupby(level=0).apply(lambda g: g['f'].corr(g['y'], method='spearman'))

    # 方向只由 train 段决定（避免事后挑符号）
    SIGN.clear()
    ic_rows = []
    for k in FACTORS:
        ic_all = fac_ic(k, dates)
        m, t, n = ic_stats(ic_all)
        m_tr, t_tr, _ = ic_stats(fac_ic(k, tr))
        sign = 1 if m_tr > 0 else -1
        SIGN[k] = sign
        m_v, t_v, nv = ic_stats(fac_ic(k, va))
        m_t, t_t, nt2 = ic_stats(fac_ic(k, te))
        print(f"  {k:<16s} train IC={m_tr:+.4f}(t={t_tr:+.1f}) → 方向 {sign:+d}  | "
              f"valid {m_v:+.4f}(t={t_v:+.1f})  test {m_t:+.4f}(t={t_t:+.1f})")
        ic_rows.append({'factor': k, 'train_IC': m_tr, 'train_t': t_tr, 'sign': sign,
                        'valid_IC': m_v, 'valid_t': t_v,
                        'test_IC': m_t, 'test_t': t_t, 'all_IC': m, 'all_t': t})
    print("  → 四因子在 train 段是否同号:",
          "是" if len(set(SIGN.values())) == 1 else f"否 {SIGN}")
    # 用 train 决定的符号重建合成
    for k in FACTORS:
        z[k] = z[k] * SIGN[k]
    score = sum(z[k] for k in FACTORS) / len(FACTORS)
    for k in FACTORS:
        m, t, n = ic_stats(fac_ic(k, dates))
    s = pd.concat([score.stack(), fwd.stack()], axis=1).dropna()
    s.columns = ['f', 'y']
    ic = s.groupby(level=0).apply(lambda g: g['f'].corr(g['y'], method='spearman'))
    m, t, n = ic_stats(ic)
    print(f"  {'合成 score(已按train定方向)':<24s} RankIC={m:+.4f}  t={t:+5.2f}  n={n}")
    # 合成分段 IC（判断信号是否在 test 失效）
    s2 = pd.concat([score.stack(), fwd.stack()], axis=1).dropna()
    s2.columns = ['f', 'y']
    seg_txt = []
    for nm, dd in [('train', tr), ('valid', va), ('test', te)]:
        sub = s2[s2.index.get_level_values(0).isin(dd)]
        icx = sub.groupby(level=0).apply(lambda g: g['f'].corr(g['y'], method='spearman'))
        mm, tt, _ = ic_stats(icx)
        seg_txt.append(f"{nm} {mm:+.4f}(t={tt:+.1f})")
        ic_rows.append({'factor': f'SCORE_{nm}', 'train_IC': np.nan, 'train_t': np.nan,
                        'sign': 1, 'valid_IC': np.nan, 'valid_t': np.nan,
                        'test_IC': np.nan, 'test_t': np.nan, 'all_IC': mm, 'all_t': tt})
    print(f"  {'合成 score 分段':<24s} " + " | ".join(seg_txt))
    ic_rows.append({'factor': 'SCORE', 'train_IC': np.nan, 'train_t': np.nan,
                    'sign': 1, 'valid_IC': np.nan, 'valid_t': np.nan,
                    'test_IC': np.nan, 'test_t': np.nan, 'all_IC': m, 'all_t': t})
    pd.DataFrame(ic_rows).to_csv(OUT / 'five_min_realcost_ic.csv', index=False)

    # ---- 基准：同池等权（每日再平衡，与信号同口径可交易）----
    bench = fwd.mean(axis=1)

    def backtest(sel_days, K, every, n_drop, account=1e7):
        hold, rows = None, []
        for k, d in enumerate(sel_days):
            try:
                s = score.loc[d].dropna()
                px_buy = buy.loc[d]
                amt = amt_day.loc[d]
            except KeyError:
                continue
            if len(s) < K:
                continue
            cand = list(s.nlargest(K).index)
            if hold is None:
                new = set(cand)
            elif k % every == 0:
                # 缓冲带：保留原持仓里排名最好的 K-n_drop 只，只换掉最差的 n_drop 只
                keep_old = [x for x in cand if x in hold][:max(K - n_drop, 1)]
                add_new = [x for x in cand if x not in hold]
                new = set(keep_old + add_new[:max(n_drop, 0)]) if n_drop > 0 else set(cand)
                new = set(list(new)[:K])
            else:
                new = set(hold)
            # 成本
            turn, w = 0.0, []
            if hold:
                old = set(hold)
                turn = 0.5 * (len(new - old) + len(old - new)) / max(len(new), 1)
            if turn > 0:
                buy_amt = account / max(len(new), 1) * 1.0
                part = (buy_amt / amt.reindex(list(new)).replace(0, np.nan)).clip(0, 1).fillna(0)
                imp = ETA * SIGMA_BP * np.sqrt(part)
                imp_avg = float(imp.mean())
            else:
                imp_avg = 0.0
            cost = turn * (COMMISSION_BP + STAMP_BP + 2 * imp_avg) / 1e4
            hold = sorted(new)
            w = [1.0 / len(hold)] * len(hold)
            r = fwd.loc[d].reindex(hold).fillna(0.0)
            rows.append({'date': d, 'gross': float((r * w).sum()),
                         'net': float((r * w).sum()) - cost,
                         'turn': turn, 'cost': cost})
        return pd.DataFrame(rows).set_index('date') if rows else pd.DataFrame()

    def ev(R, days):
        b = bench.reindex(R.index)
        e = R['net'] - b
        n = max(len(R), 1)
        ann_s = (1 + R['net']).prod() ** (252 / n) - 1
        ann_b = (1 + b).prod() ** (252 / n) - 1
        t = e.mean() / e.std() * np.sqrt(len(e)) if e.std() > 0 else np.nan
        return {'策略年化%': ann_s * 100, '基准年化%': ann_b * 100,
                '净超额年化%': (ann_s - ann_b) * 100, '超额t': t,
                '日均换手': R['turn'].mean(), '成本年化%': (R['cost'].sum() / len(R)) * 252 * 100}

    # ---- 参数选择：只在 train+valid 上 ----
    grid = list(product([20, 30, 50], [1, 2, 3, 5, 10], [0, 5, 10]))
    print(f"\n=== 参数扫描（{len(grid)} 组，仅用 train+valid 选参）===")
    rows = []
    for K, evy, nd in grid:
        R = backtest(tr + va, K, evy, nd)
        if len(R) < 20:
            continue
        m = ev(R, tr + va)
        rows.append({'K': K, 'every': evy, 'n_drop': nd, **m})
    G = pd.DataFrame(rows)
    best = G.loc[G['净超额年化%'].idxmax()]
    print(f"  选出: K={int(best['K'])} every={int(best['every'])} n_drop={int(best['n_drop'])}  "
          f"train+valid 净超额年化 {best['净超额年化%']:+.2f}%  t={best['超额t']:+.2f}  "
          f"换手 {best['日均换手']:.3f}")
    G.to_csv(OUT / 'five_min_realcost_grid.csv', index=False)

    # ---- test 期单次验证 ----
    K, evy, nd = int(best['K']), int(best['every']), int(best['n_drop'])
    R = backtest(te, K, evy, nd)
    m = ev(R, te)
    print(f"\n=== test 期单次验证（{te[0].date()} ~ {te[-1].date()}，{len(R)} 天）===")
    print(f"  净超额年化 {m['净超额年化%']:+.2f}%  超额 t={m['超额t']:+.2f}  "
          f"日均换手 {m['日均换手']:.3f}  成本 {m['成本年化%']:.2f}%/年")

    # ---- test 期分位组单调性 ----
    print("\n=== test 期分位组（铁律：D1 最高分 → D10 最低分）===")
    grp = {}
    for d in te:
        try:
            s = score.loc[d].dropna()
            y = fwd.loc[d].reindex(s.index)
        except KeyError:
            continue
        s = s.to_frame('s').join(y.to_frame('y')).dropna()
        if len(s) < 50:
            continue
        lab = pd.qcut(s['s'].rank(method='first', ascending=False), 10, labels=False)
        grp[d] = s.groupby(lab)['y'].mean()
    Gp = pd.DataFrame(grp).T.sort_index()
    b_te = bench.reindex(Gp.index)
    dec = []
    for c in sorted(Gp.columns):
        e = Gp[c].dropna() - b_te.reindex(Gp[c].dropna().index)
        ann = e.mean() * 252          # 算术年化超额（超额不可复利，几何年化会严重放大）
        dec.append({'组': f'D{c+1}', '超额年化%': ann * 100,
                    '超额t': e.mean() / e.std() * np.sqrt(len(e)) if e.std() > 0 else np.nan,
                    'n': len(e)})
        print(f"  D{c+1:<2d} 日均超额 {e.mean()*1e4:+7.2f}bp  年化 {ann*100:+7.2f}%  t={dec[-1]['超额t']:+5.2f}")
    d1, d10 = dec[0], dec[-1]
    print(f"  → D1−D10 = {(d1['超额年化%']-d10['超额年化%']):+.2f} 个百分点/年"
          f"（{'单调有效' if d1['超额年化%']>d10['超额年化%'] else '反向/无效'}）")
    pd.DataFrame(dec).to_csv(OUT / 'five_min_realcost_decile.csv', index=False)

    summary = {'参数': f"K={K}, every={evy}, n_drop={nd}",
               'test净超额年化%': m['净超额年化%'], 'test超额t': m['超额t'],
               'test日均换手': m['日均换手'], 'test成本年化%': m['成本年化%'],
               'D1-D10': d1['超额年化%'] - d10['超额年化%'],
               'test天数': len(R)}
    pd.DataFrame([summary]).to_csv(OUT / 'five_min_realcost_summary.csv', index=False)
    print("\n保存: five_min_realcost_ic/grid/decile/summary.csv")


if __name__ == '__main__':
    main()
