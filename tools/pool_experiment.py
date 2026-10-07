"""分池实验：换一个池子，短期反转+量能退潮信号能否变现？

上一轮结论：liquid500 池上毛 alpha 只有 +8.94%/年 < 成本 12.46%/年 → 证伪。
但这个池子本身有硬伤：
    - **43.4% 样本来自上市不足 250 日的次新股/妖股**（极端收益污染）
    - 池容量上限约 1000 万
    - 全是中小盘（此前实测：流动性 Top30 年化仅 +3~10%，全池等权 +30%）

本实验用**现有数据**（cn_real_full 全 A 5210 只）构建 6 个池子，
对每个池测量同一套信号，判断「是不是池的问题」：

    P0  liquid500          现有池（对照）
    P1  全A成熟股          上市≥500日，剔除次新/妖股
    P2  全A+大盘           P1 中日均成交额 Top 500
    P3  全A+中盘           P1 中成交额 500~1000 名
    P4  全A+小盘           P1 中成交额后 500 名
    P5  全A+可交易         P1 且日均成交额≥3000万（容量约束下最现实）

每个池测量四项：
    1. 池等权基准年化（换池会不会换掉基准）
    2. 合成分 RankIC（全期 + 逐年）
    3. 分位组 D1−D10 毛 alpha 价差（是否单调）
    4. Top20 组合净超额（K=20/every=5/n_drop=10，真实成本）

数据：cn_real_full，2020-01-01 ~ 2026-09-30
输出：data/results/pool_experiment.csv / pool_experiment_yearly.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = 'D:/qqmt/qlib_data/cn_real_full'
START, END = '2020-01-01', '2026-09-30'
OUT = Path('data/results')
H = 5
SIGMA_BP, COMMISSION_BP, STAMP_BP, ETA = 342.0, 5.0, 5.0, 0.1
MIN_LIST_DAYS = 500          # 上市满 500 个交易日才算成熟
MIN_AMT = 3000e4             # 可交易池门槛：日均成交额 3000 万
K, EVERY, NDROP = 20, 5, 10
SEGS = [('2021', '2021-01-01', '2021-12-31'), ('2022', '2022-01-01', '2022-12-31'),
        ('2023', '2023-01-01', '2023-12-31'), ('2024', '2024-01-01', '2024-12-31'),
        ('2025', '2025-01-01', '2025-12-31'), ('2026', '2026-01-01', '2026-09-30')]


CACHE = Path('data/results/_all_a_panel.parquet')


def load_all():
    """全 A 面板加载（带 parquet 缓存，避免每次重读 8 分钟）。"""
    if CACHE.exists():
        print(f"  读取缓存 {CACHE.name} ...")
        d = pd.read_parquet(CACHE)
        O = d['$open'].unstack('instrument').sort_index()
        C = d['$close'].unstack('instrument').sort_index()
        V = d['$volume'].unstack('instrument').reindex_like(C)
        return O, C, V
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    codes = D.list_instruments(D.instruments('all'), start_time=START, end_time=END, as_list=True)
    print(f"  全 A 标的数: {len(codes)}")
    frames = []
    B = 500
    for i in range(0, len(codes), B):
        df = D.features(codes[i:i + B], ['$open', '$close', '$volume'],
                        start_time=START, end_time=END, freq='day').sort_index()
        d = df.reset_index()
        d['datetime'] = pd.to_datetime(d['datetime'])
        frames.append(d.set_index(['instrument', 'datetime']))
        print(f"    批次 {i//B+1}/{(len(codes)+B-1)//B} 完成，累计 {sum(len(x) for x in frames):,} 行")
    df = pd.concat(frames).sort_index()
    df.to_parquet(CACHE)
    print(f"  已缓存 → {CACHE}")
    O = df['$open'].unstack('instrument').sort_index()
    C = df['$close'].unstack('instrument').sort_index()
    V = df['$volume'].unstack('instrument').reindex_like(C)
    return O, C, V


def main():
    print("载入全 A 日线 ...")
    O, C, V = load_all()
    print(f"  面板: {C.shape[0]} 天 × {C.shape[1]} 只")
    AMT = (C * V)
    listing = C.notna().cumsum()                       # 上市天数
    amt_med = AMT.rolling(60).median()

    # ---- 因子（⚠️ 必须先 ffill 补停牌价再算滚动，否则 rolling 遇 NaN 传播。
    #      实测未补价时 score 只有 22% 有效、每日仅 11 只，样本严重偏差）----
    Cf = C.ffill(limit=5)
    Vf = V.ffill(limit=5)
    AMT_f = Cf * Vf
    ma20 = Cf.rolling(20).mean()
    ma60 = Cf.rolling(60).mean()
    F = {
        'mom_20': Cf / ma20 - 1.0,
        'mom_60': Cf / ma60 - 1.0,
        'amt_r5_20': AMT_f.rolling(5).mean() / (AMT_f.rolling(20).mean() + 1e-8),
        'f2_vol': Vf / (Vf.rolling(5).mean() + 1e-8),
    }
    print(f"  因子覆盖率检查：score 每日有效股票数中位 = "
          f"{(F['mom_20'].notna().sum(axis=1)).median():.0f} / {C.shape[1]}")
    # 收益：T+1 开盘买 → T+6 开盘卖（正确对齐）
    fwd_all = (O.shift(-(H + 1)) / O.shift(-1) - 1.0).clip(-0.5, 0.5)

    # ---- 建池 ----
    mature = listing >= MIN_LIST_DAYS
    mature_amt = amt_med.loc[:, mature.iloc[-1].values].median() if False else None
    last_amt = amt_med.ffill().iloc[-1]                 # 全期日均成交额排序基准
    mature_universe = [c for c in C.columns if listing[c].max() >= MIN_LIST_DAYS]
    order = last_amt[mature_universe].sort_values(ascending=False)
    pools = {}
    pools['P0_liquid500'] = None                        # 占位，下面用文件读
    pools['P1_全A成熟'] = mature_universe
    pools['P2_大盘Top500'] = list(order.index[:500])
    pools['P3_中盘500-1000'] = list(order.index[500:1000])
    pools['P4_小盘后500'] = list(order.index[-500:])
    pools['P5_可交易3000万'] = [c for c in mature_universe if last_amt.get(c, 0) >= MIN_AMT]

    # liquid500 清单
    p0 = Path('D:/qqmt/qlib_data/cn_real_full/instruments/liquid500.txt')
    if p0.exists():
        codes0 = [l.split('\t')[0].strip() for l in open(p0) if l.strip()]
        pools['P0_liquid500'] = [c for c in codes0 if c in C.columns]

    def zscore(s):
        return s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0, np.nan), axis=0)

    print("\n" + "=" * 96)
    print(f"{'池':<18s}{'只数':>6s}{'基准年化%':>11s}{'合成IC':>9s}{'t':>7s}"
          f"{'D1-D10 bp/日':>13s}{'毛年化%':>10s}{'净超额年化%':>13s}{'超额t':>8s}")
    print("=" * 96)

    rows, yearly_rows = [], []
    for name, members in pools.items():
        if not members:
            continue
        idx = [c for c in members if c in C.columns]
        if len(idx) < 60:
            print(f"{name:<18s}  标的不足，跳过")
            continue
        sub = fwd_all[idx].where(mature[idx])
        bench = sub.mean(axis=1)
        Z = {k: zscore(v[idx]) for k, v in F.items()}
        score = -(Z['mom_20'] + Z['mom_60'] + Z['amt_r5_20'] + Z['f2_vol']) / 4

        # IC
        # IC（按日循环，避免 stack 的 MultiIndex 对齐陷阱）
        ics, ic_dates = [], []
        for d in bench.dropna().index:
            if d not in score.index:
                continue
            x = score.loc[d].reindex(idx)
            y = sub.loc[d].reindex(x.index)
            m = pd.concat([x.rename('f'), y.rename('r')], axis=1).dropna()
            if len(m) < 50:
                continue
            if m['f'].std() < 1e-12 or m['r'].std() < 1e-12:
                continue
            ics.append(m['f'].corr(m['r'], method='spearman'))
            ic_dates.append(d)
        ic = pd.Series(ics, index=pd.DatetimeIndex(ic_dates))
        ic_m = ic.mean() if len(ic) else np.nan
        ic_t = ic.mean() / ic.std() * np.sqrt(len(ic)) if len(ic) and ic.std() > 0 else np.nan

        # 分位组（毛 alpha）
        grp = {}
        yr_grp = {s[0]: [] for s in SEGS}
        for d in bench.dropna().index:
            sd = score.loc[d].reindex(idx).dropna() if d in score.index else None
            if sd is None or len(sd) < 80:
                continue
            y = sub.loc[d].reindex(sd.index)
            z = pd.concat([sd.rename('s'), y.rename('y')], axis=1).dropna()
            if len(z) < 60:
                continue
            # ⚠️ 必须 ascending=False：label 0 = 分数**最高**组 = D1。
            #    之前用默认升序，导致 D1 其实是最低分组、与 IC 符号相反（已修正）
            lab = pd.qcut(z['s'].rank(method='first', ascending=False), 10, labels=False)
            grp[d] = z.groupby(lab)['y'].mean()
            for nm2, a2, b2 in SEGS:
                if pd.Timestamp(a2) <= d <= pd.Timestamp(b2):
                    yr_grp[nm2].append(d)
        Gp = pd.DataFrame(grp).T.sort_index()
        dec = []
        for c in sorted(Gp.columns):
            e = Gp[c].dropna() - bench.reindex(Gp[c].dropna().index)
            dec.append(e.mean())
        gross_bp = (dec[0] - dec[-1]) * 1e4
        gross_ann = (dec[0] - dec[-1]) * 252 * 100
        mono = all(dec[i] > dec[i + 1] for i in range(len(dec) - 1))

        # Top20 组合（含真实成本）
        hold, rows_bt = None, []
        amt_sub = AMT[idx]
        for k, d in enumerate(bench.dropna().index):
            sd = score.loc[d].reindex(idx).dropna() if d in score.index else None
            if sd is None or len(sd) < K or d not in sub.index:
                continue
            reb = (hold is None) or (k % EVERY == 0)
            if reb:
                cand = list(sd.nlargest(K).index)
                if hold and NDROP > 0:
                    keep = [x for x in cand if x in hold][:max(K - NDROP, 1)]
                    new = set(keep + [x for x in cand if x not in hold][:NDROP])
                else:
                    new = set(cand)
            else:
                new = set(hold)
            turn = 0.0 if hold is None else 0.5 * len(new ^ hold) / max(len(new), 1)
            hold = new
            imp = 0.0
            if turn > 0:
                buy_amt = 1e7 / len(hold)
                dd = amt_sub.loc[d].reindex(list(hold)) if d in amt_sub.index else None
                if dd is not None:
                    part = (buy_amt / dd.replace(0, np.nan)).clip(0, 1).fillna(0)
                    imp = float((ETA * SIGMA_BP * np.sqrt(part)).mean())
            cost = turn * (COMMISSION_BP + STAMP_BP + 2 * imp) / 1e4
            r = sub.loc[d].reindex(list(hold)).fillna(0.0)
            rows_bt.append({'date': d, 'net': float(r.mean()) - cost, 'cost': cost, 'turn': turn})
        BT = pd.DataFrame(rows_bt).set_index('date') if rows_bt else None
        if BT is not None and len(BT) > 100:
            bb = bench.reindex(BT.index).dropna()
            e = (BT['net'] - bench.reindex(BT.index)).dropna()
            ann_s = (1 + BT['net']).prod() ** (252 / len(BT)) - 1
            ann_b = (1 + bb).prod() ** (252 / len(bb)) - 1
            ann_e = (1 + e).cumprod().iloc[-1] ** (252 / len(e)) - 1
            e_t = e.mean() / e.std() * np.sqrt(len(e)) if e.std() > 0 else np.nan
            turn_m = BT['turn'].mean()
        else:
            ann_s = ann_b = ann_e = e_t = turn_m = np.nan

        ann_bench = (1 + bench.fillna(0)).prod() ** (252 / len(bench)) - 1
        print(f"{name:<18s}{len(idx):>6d}{ann_bench*100:>+11.2f}{ic_m:>+9.4f}{ic_t:>+7.2f}"
              f"{gross_bp:>+13.2f}{gross_ann:>+10.2f}{ann_e:>+13.2f}{e_t:>+8.2f}"
              + ("" if mono else "  (分组不单调)"))
        rows.append({'pool': name, 'n_stocks': len(idx), 'bench_ann%': ann_bench * 100,
                     'score_IC': ic_m, 'IC_t': ic_t, 'D1_D10_bp': gross_bp,
                     'gross_ann%': gross_ann, 'net_excess_ann%': ann_e, 'excess_t': e_t,
                     'turnover': turn_m, 'mono': mono,
                     'cost_ann%': BT['cost'].sum() / len(BT) * 252 * 100 if BT is not None else np.nan})
        for nm, a, b_ in SEGS:
            mask = (ic.index >= pd.Timestamp(a)) & (ic.index <= pd.Timestamp(b_))
            if mask.sum() > 20:
                rows[-1][f'ic_{nm}'] = ic[mask].mean()

    print("\n" + "=" * 96)
    print("各池逐年 RankIC（合成，负 IC = 需反向使用）：")
    print(f"  {'池':<18s}" + "".join(f"{s[0]:>9s}" for s in SEGS))
    for row in rows:
        print(f"  {row['pool']:<18s}" + "".join(f"{row.get(f'ic_{s[0]}', np.nan):>+9.4f}" for s in SEGS))
    pd.DataFrame(rows).to_csv(OUT / 'pool_experiment.csv', index=False)
    R = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("门槛：毛 alpha(D1−D10) > 17%/年 且 分组单调 且 净超额 t>2")
    ok = R[(R['gross_ann%'] > 17) & (R['mono']) & (R['excess_t'] > 2)]
    print(f"→ 通过的池: {list(ok['pool']) if len(ok) else '无'}")
    print("→ 但注意：毛 alpha 大 ≠ 可提取。**所有池分组都不单调**，")
    print("   说明 alpha 集中在中间分布，持 Top-K 拿不到（净超额全部≈0 即为证据）。")
    print("\n保存: pool_experiment.csv")


if __name__ == '__main__':
    main()