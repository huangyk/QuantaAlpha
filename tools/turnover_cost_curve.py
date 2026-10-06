"""换手—成本曲线：在 liquid500 这种中小盘池上，多大的换手率是可承受的？

这是优化路线图 P0 第 4 项（换手惩罚 + 流动性配额）的**定量地基**：
在改策略之前，先搞清楚「换手 1 次要花多少钱、一年能吃掉多少收益」。

成本拆解（A 股，2023-08 印花税减半后）：
    - 佣金：双边约 5 bp（万 2.5 × 2）
    - 印花税：卖出单边 5 bp
    - 过户费：沪深 0.1 bp（忽略量级）
    - **市场冲击**：participation = 买入额 / 该时段成交额，
      用 Almgren 简化式  impact_bp = eta * sigma_daily_bp * sqrt(participation)
      sigma_daily 用真实数据算（不是假设值）

关键输出：
    1. 池内个股真实日波动率（用于冲击模型）
    2. 不同账户规模 × 持仓只数 下的单票 participation 分布（决定冲击）
    3. 换手率 → 年化成本曲线（含盈亏平衡表）
    4. 结论：在给定毛 alpha 下，可承受的最大换手率

数据：cn_real_full（日线，全期）用于波动率与成交额；5min 用于时段成交额占比。
输出：data/results/turnover_cost_*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import qlib

PROVIDER = 'D:/qqmt/qlib_data/cn_real_full'
START, END = '2020-01-01', '2026-09-30'
FIELDS = ['$close', '$volume']
OUT = Path('data/results')

# 成本参数（bp，1 bp = 0.01%）
COMMISSION_BP = 5.0      # 双边佣金
STAMP_BP = 5.0           # 卖出印花税
ETA = 0.1                # 冲击系数（Almgren 实证常见 0.1~0.3；取 1.0 会算出荒谬值）
SLOT_SHARE_5M = 0.10     # 若集中在单一 5min 时段成交，该时段约占全天成交 10%
MAX_PART = 0.20          # participation 超过 20% 视为「不可执行」，而非给一个虚假冲击数


def load():
    qlib.init(provider_uri=PROVIDER, region='cn')
    from qlib.data import D
    codes = D.list_instruments(D.instruments('liquid500'), start_time=START, end_time=END, as_list=True)
    df = D.features(codes, FIELDS, start_time=START, end_time=END, freq='day').sort_index()
    return df


def main():
    df = load()
    inst = df.index.get_level_values('instrument')
    c = df['$close']
    v = df['$volume']
    amt = (c * v)
    print(f"样本: {len(df):,} 行  {inst.nunique()} 只  "
          f"{df.index.get_level_values('datetime').min().date()} ~ "
          f"{df.index.get_level_values('datetime').max().date()}")

    # ---- 1. 真实日波动率 ----
    ret = c.groupby(level='instrument').pct_change()
    sig = ret.groupby(level='instrument').std() * 1e4      # bp
    print(f"\n=== 1. 池内个股日波动率（真实值，冲击模型的输入） ===")
    print(f"  中位 {sig.median():.0f} bp（{sig.median()/100:.2f}%）  "
          f"均值 {sig.mean():.0f} bp  P25 {sig.quantile(.25):.0f} / P75 {sig.quantile(.75):.0f}")
    sigma_bp = sig.median()

    # ---- 2. 各账户规模 × 持仓只数的 participation 分布 ----
    amt_med = amt.median()
    amt_pool = amt.dropna()
    print(f"\n=== 2. 单票买入额占日成交额的比重（participation）===")
    print(f"  池内单票日成交额：中位 {amt_med/1e4:,.0f} 万元")
    rows = []
    for acct in (5e6, 1e7, 5e7, 1e8):
        for k in (10, 20, 30, 50, 100):
            per = acct / k
            part = per / amt_pool
            # 集中在单一时段时 participation 放大 1/SLOT_SHARE
            part_slot = part / SLOT_SHARE_5M
            imp_bp = ETA * sigma_bp * np.sqrt(np.clip(part, 0, 1))
            imp_slot_bp = ETA * sigma_bp * np.sqrt(np.clip(part_slot, 0, 1))
            rows.append({'账户(万)': acct / 1e4, '持仓数': k, '单票买入(万)': per / 1e4,
                         'participation中位': part.median(), 'participation_P90': part.quantile(.9),
                         '全日成交冲击bp(中位)': imp_bp.median(),
                         '集中单时段冲击bp(中位)': imp_slot_bp.median(),
                         '集中单时段冲击bp(P90)': imp_slot_bp.quantile(.9),
                         '可执行': ('否(超20%)' if part.median() > MAX_PART else
                                   ('勉强' if part.median() > 0.10 else '可行'))})
    L = pd.DataFrame(rows)
    print(L.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    L.to_csv(OUT / 'turnover_cost_participation.csv', index=False)

    # ---- 3. 换手率 → 年化成本曲线 ----
    print(f"\n=== 3. 换手率 → 年化成本（账户 1000 万、持仓 30 只 的单票冲击为基准）===")
    per = 1e7 / 30
    part = (per / amt_pool).median()
    part_slot = part / SLOT_SHARE_5M
    imp = ETA * sigma_bp * np.sqrt(min(part, 1.0))
    imp_slot = ETA * sigma_bp * np.sqrt(min(part_slot, 1.0))
    fixed = COMMISSION_BP + STAMP_BP
    print(f"  单票买入 {per/1e4:,.0f} 万元 → participation {part:.2%}（全天）/ {part_slot:.2%}（单时段）")
    print(f"  固定成本 {fixed:.0f} bp/次换手（佣金+印花税）")
    print(f"  冲击成本：全天分散 {imp:.1f} bp / 集中单时段 {imp_slot:.1f} bp")
    print(f"\n  {'日换手':>8s}{'年化成本(分散)':>16s}{'年化成本(单时段)':>18s}{'覆盖所需毛alpha':>18s}")
    rows = []
    for tau in (0.05, 0.1, 0.2, 0.3, 0.5, 0.734, 1.0, 2.0):
        c_disp = (fixed + 2 * imp) * tau * 252 / 1e4        # 双边都有冲击
        c_slot = (fixed + 2 * imp_slot) * tau * 252 / 1e4
        rows.append({'日换手': tau, '年化成本_分散%': c_disp * 100, '年化成本_单时段%': c_slot * 100})
        flag = "  ← 飞龙实测换手" if abs(tau - 0.734) < 1e-6 else ""
        print(f"  {tau:>8.3f}{c_disp*100:>15.2f}%{c_slot*100:>17.2f}%{c_disp*100:>17.2f}%{flag}")
    pd.DataFrame(rows).to_csv(OUT / 'turnover_cost_curve.csv', index=False)

    # ---- 4. 校准：与飞龙实测对照 ----
    print(f"\n=== 4. 与飞龙实测对照（校验模型）===")
    feilong_tau, feilong_cost = 0.734, -36.98
    m10 = 10 * feilong_tau * 252 / 1e4 * 100
    m20 = 20 * feilong_tau * 252 / 1e4 * 100
    mreal = (fixed + 2 * imp) * feilong_tau * 252 / 1e4 * 100
    print(f"  飞龙：换手 {feilong_tau}/日 → 实测吞噬 {feilong_cost}%/年")
    print(f"  模型 @10bp/次（纯佣金+印花税）：{m10:.2f}%/年")
    print(f"  模型 @20bp/次（飞龙实际用的口径）：{m20:.2f}%/年  ← 与实测 {feilong_cost}% 吻合，模型可信")
    print(f"  加上真实冲击（全天分散 {imp:.1f}bp/单边）后：{mreal:.2f}%/年  ← 飞龙真实成本远高于当时的估计")

    # ---- 5. 可承受换手率 ----
    print(f"\n=== 5. 给定毛 alpha，可承受的最大日换手（成本<毛alpha 的一半）===")
    for alpha in (5, 10, 20, 30, 50):
        budget = alpha / 2                                  # 成本不超过毛 alpha 的一半
        per_turn_bp = fixed + 2 * imp_slot
        tau_max = budget / 100 * 1e4 / (per_turn_bp * 252)
        print(f"  毛 alpha {alpha:>3d}%/年 → 成本预算 {budget:>5.1f}%/年 → 最大日换手 "
              f"{tau_max:.3f}（约每 {max(1, int(1/max(tau_max, 1e-6)))} 天调仓一次）")
    print("\n保存: turnover_cost_participation.csv / turnover_cost_curve.csv")


if __name__ == '__main__':
    main()
