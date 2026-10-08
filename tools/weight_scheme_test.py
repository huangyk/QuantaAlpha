"""权重层实验：不做选股，只调权重，能不能跑赢池等权？

定位：选股层已被反复证伪（分池实验：6 个池净超额全部 ≈0）。
但**权重层从未被测过**。已有线索指向权重层有价值：
    - 实验3：流动性 Top30/50 年化仅 +3~10%，全池等权 +30.08% → 小盘/低流动性更好
    - 分池实验：IC 与小盘程度单调正相关（小盘 0.0809 > 可交易池 0.0102）
    → 「偏向小盘/低流动性」这条线索在**连续权重**上从未被验证过。

七种权重方案（同一可交易池，低换手月度调仓，真实成本）：
    W1 等权          基准
    W2 流动性正比    w ∝ 成交额^1        （偏向流动性好的）
    W3 流动性反比    w ∝ 成交额^-0.5     （偏向流动性差的 = 小盘，倾斜）
    W4 低波动        w ∝ 1/波动率^0.5
    W5 高波动        w ∝ 波动率^0.5
    W6 动量          w ∝ 过去60日收益     （追强）
    W7 反动量        w ∝ 1/(1+过去60日收益)
    W8 混合倾斜      0.5×流动性反比 + 0.5×低波动（最贴合已发现的规律）

约束（全部方案统一，保证可执行）：
    - 单票权重 ≤ 5%
    - 单票买入额 ≤ 2% × 该票当日成交额（流动性配额）
    - 月度调仓（换手天然很低）
执行口径：T 日收盘定权重 → T+1 **开盘**买入（开盘比当日 VWAP 便宜 12bp，已实测）

数据：cn_real_full 全 A（读缓存），2020-01 ~ 2026-09
输出：data/results/weight_scheme.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path('data/results/_all_a_panel.parquet')
OUT = Path('data/results')
MIN_LIST_DAYS = 500          # 上市满 1 年
MIN_AMT = 1000e4             # 日成交额 ≥ 1000 万（3000万在 A 股只筛出百来只，池子太小）
MAX_W = 0.05                 # 单票权重上限 5%
LIQ_CAP = 0.02               # 单票买入额 ≤ 2% 日成交额
ACCOUNT = 1e7
COMMISSION_BP, STAMP_BP, ETA = 5.0, 5.0, 0.1
SEGS = [('2021', '2021-01-01', '2021-12-31'), ('2022', '2022-01-01', '2022-12-31'),
        ('2023', '2023-01-01', '2023-12-31'), ('2024', '2024-01-01', '2024-12-31'),
        ('2025', '2025-01-01', '2025-12-31'), ('2026', '2026-01-01', '2026-09-30')]


def main():
    print("载入缓存 ...")
    d = pd.read_parquet(CACHE)
    O = d['$open'].unstack('instrument').sort_index()
    C = d['$close'].unstack('instrument').sort_index()
    V = d['$volume'].unstack('instrument').reindex_like(C)
    AMT = C * V * 100.0   # volume 单位是「手」，成交额需 ×100
    listing = C.notna().cumsum()
    print(f"  面板 {C.shape}")

    # 日收益（T+1 开盘 → 次日开盘），截断极端值防 −100%
    ret = (O.shift(-1) / O - 1.0).clip(-0.5, 0.5)
    # ⚠️ 两个池必须分开：
    #   base_pool  基准用「成熟 + 流动性」——基准衡量的是"持有"，不能剔除涨停股票
    #              （上一版把涨停开盘也剔掉，等于每天系统性剔除当天最强的票，
    #                基准年化被压成 −3.92%，全盘错误）
    #   buy_pool   买入用「base_pool + 非涨停开盘」——涨停开盘确实买不进
    amt20 = AMT.rolling(20, min_periods=5).mean()
    is_mature = listing >= MIN_LIST_DAYS
    is_liquid = amt20 >= MIN_AMT
    limit_up = (O / C.shift(1) - 1.0) >= 0.098
    base_pool = is_mature & is_liquid
    buy_pool = base_pool & (~limit_up)
    print(f"  基准池中位 {base_pool.sum(axis=1).median():.0f} 只  |  "
          f"买入池中位 {buy_pool.sum(axis=1).median():.0f} 只  "
          f"(日成交额门槛 {MIN_AMT/1e4:,.0f} 万)")

    # 因子面板（ffill 补停牌后算）
    Cf = C.ffill(limit=5)
    AMT_f = (Cf * V.ffill(limit=5))
    vol20 = (Cf.pct_change().rolling(20).std())
    mom60 = Cf / Cf.rolling(60).mean() - 1.0
    liq = AMT_f.rolling(20).mean()

    dates = C.index
    month_starts = set(pd.Series(dates).groupby([dates.year, dates.month]).min())

    def build_weights(scheme, uni, d_prev):
        """在池内构造权重方案（上一交易日的数据，避免前视）。"""
        if len(uni) < 30:
            return None
        raw = {}
        if scheme == 'W1_等权':
            w = pd.Series(1.0, index=uni)
        elif scheme == 'W2_流动性正比':
            w = liq.loc[d_prev, uni].clip(lower=0) + 1.0
        elif scheme == 'W3_流动性反比':
            w = 1.0 / np.sqrt(liq.loc[d_prev, uni].clip(lower=0) + 1.0)
        elif scheme == 'W4_低波动':
            w = 1.0 / np.sqrt(vol20.loc[d_prev, uni].clip(lower=1e-4) + 1e-8)
        elif scheme == 'W5_高波动':
            w = np.sqrt(vol20.loc[d_prev, uni].clip(lower=1e-4))
        elif scheme == 'W6_动量':
            w = mom60.loc[d_prev, uni].clip(lower=-0.9) + 1.0
        elif scheme == 'W7_反动量':
            w = 1.0 / (1.0 + mom60.loc[d_prev, uni].clip(lower=-0.9))
        elif scheme == 'W8_混合倾斜':
            a = 1.0 / np.sqrt(liq.loc[d_prev, uni].clip(lower=0) + 1.0)
            b = 1.0 / np.sqrt(vol20.loc[d_prev, uni].clip(lower=1e-4) + 1e-8)
            w = 0.5 * a / a.mean() + 0.5 * b / b.mean()
        else:
            return None
        w = w.replace([np.inf, -np.inf], np.nan).dropna()
        if len(w) < 30:
            return None
        w = w.clip(lower=0)
        if w.sum() <= 0:
            return None
        w = w / w.sum()
        # 权重上限
        w = w.clip(upper=MAX_W)
        w = w / w.sum()
        # 流动性配额：单票买入额 ≤ 2% 日成交额 → w_i ≤ 2%×amt_i/账户
        cap = (LIQ_CAP * AMT_f.loc[d_prev, w.index].fillna(0)) / ACCOUNT
        over = w > cap
        if over.any():
            excess = (w - cap[over]).clip(lower=0)
            w = w.clip(upper=cap)
            w = w + excess / excess.sum() * w[w.index.isin(cap.index)].clip(upper=cap).sum() if excess.sum() > 0 else w
            w = w[w > 0]
            w = w / w.sum()
        return w

    SCHEMES = ['W1_等权', 'W2_流动性正比', 'W3_流动性反比', 'W4_低波动',
               'W5_高波动', 'W6_动量', 'W7_反动量', 'W8_混合倾斜']

    print(f"\n{'方案':<16s}{'年化%':>10s}{'IR':>7s}{'回撤%':>10s}{'换手':>8s}"
          f"{'成本年化%':>11s}{'超额年化%':>11s}{'超额t':>8s}")
    print("-" * 82)

    results = {}
    prev_weights = {}
    nav = {s: [] for s in SCHEMES}
    bench_nav = []
    dates_used = []
    months = sorted(month_starts)
    turnover_log = {s: [] for s in SCHEMES}

    for k, d in enumerate(dates):
        if d not in month_starts or k < 61:
            continue
        d_prev = dates[k - 1]
        if d not in ret.index:
            continue
        uni_all = base_pool.loc[d_prev].reindex(C.columns).fillna(False)
        uni_all = [c for c in C.columns if uni_all[c]]
        uni_buy = [c for c in uni_all if not limit_up.at[d, c]]
        r = ret.loc[d].reindex(C.columns)
        for s in SCHEMES:
            w = build_weights(s, uni_buy, d_prev)
            prev = prev_weights.get(s)
            if w is None:
                nav[s].append(0.0)
                turnover_log[s].append(0.0)
                prev_weights[s] = prev
                continue
            turn = 0.0 if prev is None else 0.5 * float(
                (w - prev.reindex(w.index).fillna(0)).abs().sum()
                + prev.drop(w.index, errors='ignore').sum())
            imp = 0.0
            if turn > 0:
                delta = (w - prev.reindex(w.index).fillna(0)).clip(lower=0) if prev is not None else w
                buy_amt = ACCOUNT * delta
                part = (buy_amt / AMT_f.loc[d, w.index]).clip(0, 1).fillna(0)
                imp_w = ETA * 342.0 * np.sqrt(part)
                imp = float((delta * imp_w).sum())
            cost = turn * (COMMISSION_BP + STAMP_BP + 2 * imp) / 1e4
            gross = float((w * r.reindex(w.index).fillna(0.0)).sum())
            nav[s].append(gross - cost)
            turnover_log[s].append(turn)
            prev_weights[s] = w
        # 等权基准（同池，不剔除涨停 —— 基准衡量的是"持有"）
        bench_nav.append(float(r.reindex(uni_all).fillna(0.0).mean()) if uni_all else 0.0)
        dates_used.append(d)

    n = min(len(v) for v in nav.values())
    idx_use = pd.DatetimeIndex(dates_used[:n])

    for s in SCHEMES:
        ser = pd.Series(nav[s][:n], index=idx_use)
        turn = pd.Series(turnover_log[s][:n], index=idx_use)
        cost_ann = (turn.mean() * 252 * (COMMISSION_BP + STAMP_BP) / 1e4) * 100
        ann = (1 + ser).prod() ** (12 / len(ser)) - 1        # 月频调仓 → 12 期/年
        cum = (1 + ser).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        ir = ser.mean() / ser.std() * np.sqrt(12) if ser.std() > 0 else np.nan
        results[s] = {'年化%': ann * 100, 'IR': ir, '回撤%': mdd * 100,
                      '换手': turn.mean(), '成本年化%': cost_ann, '月数': len(ser)}
        B = pd.Series(bench_nav[:n], index=idx_use)
        exc = (1 + ser) / (1 + B) - 1
        results[s]['超额年化%'] = ((1 + exc).prod() ** (12 / len(exc)) - 1) * 100
        e = ser - B
        results[s]['超额t'] = e.mean() / e.std() * np.sqrt(len(e)) if e.std() > 0 else np.nan

    for s in SCHEMES:
        r = results[s]
        print(f"{s:<16s}{r['年化%']:>+10.2f}{r['IR']:>+7.2f}{r['回撤%']:>+10.2f}"
              f"{r['换手']:>8.3f}{r['成本年化%']:>+11.2f}{r['超额年化%']:>+11.2f}{r['超额t']:>+8.2f}")

    R = pd.DataFrame(results).T
    R.to_csv(OUT / 'weight_scheme.csv')
    best = R.sort_values('超额年化%', ascending=False).iloc[0]
    print("-" * 82)
    print(f"最佳方案: {best.name}  超额年化 {best['超额年化%']:+.2f}%  t={best['超额t']:+.2f}")
    print(f"\n同池等权基准（同期）: 年化 {results['W1_等权']['年化%']:+.2f}%")

    # 分年
    print("\n逐年超额（vs 同池等权）：")
    hdr = f"  {'方案':<16s}" + "".join(f"{s[0]:>9s}" for s in SEGS)
    print(hdr)
    for s in SCHEMES:
        ser = pd.Series(nav[s][:n], index=idx_use)
        B = pd.Series(bench_nav[:n], index=idx_use)
        line = f"  {s:<16s}"
        for nm, a, b_ in SEGS:
            m = (ser.index >= pd.Timestamp(a)) & (ser.index <= pd.Timestamp(b_))
            if m.sum() < 3:
                line += f"{'-':>9s}"
                continue
            e1 = (1 + ser[m]) / (1 + B[m]) - 1
            line += f"{(e1.prod()-1)*100:>+9.2f}"
        print(line)
    print("\n保存: weight_scheme.csv")


if __name__ == '__main__':
    main()