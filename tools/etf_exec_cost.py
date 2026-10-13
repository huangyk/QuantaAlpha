"""ETF 轮动策略的执行层（L4）成本模型
=====================================

背景（诚实定位）
---------------
`etf_timeseries_signals.py` 的 `simulate()` 只扣了**固定佣金**（万1/千1），
既没算市场冲击（Almgren），也没算**执行时点**差异。
而 `pool_hold.py` 实测：开盘集合竞价比当日 VWAP 便宜 ~12bp，
10:00/13:05 当日最贵；拆单集中单时段冲击 9.5→30.1bp（差 3 倍）。
本模块把这套执行成本**正确建模**，用于量化 ETF 轮动策略的真实净收益与可优化空间。

与股票池的关键区别（为什么 ETF 冲击很小）
----------------------------------------
- ETF **免印花税**（股票卖出 5bp），只有佣金。
- ETF 流动性极好（如 510300 日成交几十亿），零售账户 participation 极小
  → Almgren 冲击 ≈ 0。**所以 ETF 的执行拖累主要来自佣金 + 执行时点，不是冲击。**
- 这正是 L4 在 ETF 上该优化的点：① 执行时点（开盘 vs VWAP）② 降低调仓频率。

成本构成（单边，bp）
-------------------
  cost_bp = commission_bp
          + stamp_bp(ETF=0)
          + 2 * almgren_impact_bp(part, sigma)     # 买+卖
          + timing_bp(exec_session)                 # 相对 VWAP 的偏差

单位：1bp = 0.01%。

⚠️ 本文件的 `__main__` 用**合成数据**做单元自测（验证函数能跑、数量级合理），
   **不是真实回测结果**。真实回测需接 `etf_timeseries_signals.load_etf`（依赖 TimescaleDB）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── ETF 默认成本参数（bp，单边）─────────────────────────────────────────
COMMISSION_BP_ETF = 1.0      # 万1 单边佣金（ETF 无印花税）
STAMP_BP_ETF = 0.0           # ETF 免印花税
SIGMA_BP_ETF_DEF = 160.0     # ETF 日波动中位（跨资产，低于个股 342bp）
IMPACT_ETA = 0.1             # Almgren 冲击系数（同 pool_hold.py）

# 执行时点相对 VWAP 的偏差（bp，来自 pool_hold.py 个股实测；
# 对 ETF 为保守估计——ETF 更厚，开盘-VWAP 差通常更小，此处用个股实测作上限）
TIMING_BP = {
    "open": -12.0,     # 09:30-09:35 集合竞价，比 VWAP 便宜 12bp（最优）
    "11:00": -11.6,    # 次优
    "10:00": +5.3,     # 当日最贵之一
    "13:05": +5.8,     # 当日最贵之一
    "15:00": +3.0,     # 尾盘，近似 VWAP
    "vwap": 0.0,       # 基准
}


def almgren_impact_bp(participation: float, sigma_bp: float = SIGMA_BP_ETF_DEF,
                      eta: float = IMPACT_ETA) -> float:
    """单边市场冲击（bp）。participation = 单笔成交额 / ETF 当日成交额。

    ETF 日成交几十亿，零售账户 participation 通常 <0.1% → 冲击 <1bp，可忽略。
    """
    if participation <= 0:
        return 0.0
    return eta * sigma_bp * (participation ** 0.5)


def timing_bp(exec_session: str) -> float:
    """执行时点的相对 VWAP 偏差（bp）。负=比 VWAP 便宜（省）。"""
    return TIMING_BP.get(exec_session, 0.0)


def trade_cost_bp(participation: float, exec_session: str = "open",
                  commission_bp: float = COMMISSION_BP_ETF,
                  sigma_bp: float = SIGMA_BP_ETF_DEF) -> float:
    """单笔单边交易成本（bp，含冲击+时点）。"""
    impact = 2 * almgren_impact_bp(participation, sigma_bp)  # 买+卖双边冲击
    return commission_bp + STAMP_BP_ETF + impact + timing_bp(exec_session)


def annual_cost_from_turnover(cum_turnover: float, years: float,
                              exec_session: str = "open",
                              account: float = 1e7,
                              etf_median_daily_amt: float = 5e8) -> dict:
    """用**真实测得的累计换手**估算 ETF 轮动的年化执行成本。

    参数
    ----
    cum_turnover : 全期单边累计换手（Σ|Δw|），来自回测统计。
                   本策略实测 2019-2026 累计换手 ≈ 36.7。
    years        : 回测年数。
    exec_session : 执行时点（'open' 最优 / 'vwap' / '10:00' 最贵）。
    account      : 账户规模（元）。
    etf_median_daily_amt : ETF 日成交中位（元），用于算 participation。

    返回
    ----
    dict: 年化成本 bp、累计成本 bp、冲击占比、时点占比。
    """
    # 平均单笔 participation（用账户规模近似单笔成交额 ~ 账户/持仓数，这里保守取账户规模占比）
    part = (account / 36) / etf_median_daily_amt  # 36 只 ETF 等权
    per_trade = trade_cost_bp(part, exec_session)
    # 累计换手对应的"笔数"不显式建模：直接按换手额 × 单边成本率
    # cum_turnover 是组合权重的 L1 变动和，等价于"单边成交了 cum_turnover 倍组合"
    cum_cost_bp = cum_turnover * per_trade
    ann_cost_bp = cum_cost_bp / years if years > 0 else float("nan")
    return {
        "participation_pct": part * 100,
        "per_trade_cost_bp": per_trade,
        "cum_cost_bp": cum_cost_bp,
        "annual_cost_bp": ann_cost_bp,
        "annual_cost_pct": ann_cost_bp / 100,
        "timing_contrib_bp": cum_turnover * timing_bp(exec_session),
        "impact_contrib_bp": cum_turnover * 2 * almgren_impact_bp(part),
    }


def replay_with_exec_cost(nav: pd.Series, turnover_per_rebal: float,
                           n_rebal: int, exec_session: str = "open",
                           account: float = 1e7,
                           etf_median_daily_amt: float = 5e8) -> pd.Series:
    """在已有 NAV 序列上叠加**执行时点**成本（佣金已在 simulate 扣，这里补时点差）。

    说明：simulate() 已扣固定佣金；本函数只补"执行时点相对 VWAP 的偏差"
    （开盘省 12bp / 10:00 贵 5.3bp），按每次调仓的换手额计。
    """
    part = (account / 36) / etf_median_daily_amt
    per_rebal_bp = turnover_per_rebal * (timing_bp(exec_session) + 2 * almgren_impact_bp(part))
    total_bp = per_rebal_bp * n_rebal
    factor = (1 + total_bp / 10000.0) ** -1  # 时点成本是额外摩擦，乘性扣除
    return nav * factor


if __name__ == "__main__":
    # ── 合成数据单元自测（非真实回测）────────────────────────────────
    print("=== ETF 执行成本单元自测（合成数据，仅验证数量级）===")
    # 1) 冲击随 participation 变化（ETF 应极小）
    for part in [0.001, 0.01, 0.05, 0.1]:
        print(f"  participation={part*100:.1f}% -> 单边冲击 {almgren_impact_bp(part):.3f}bp")
    # 2) 执行时点对比
    for s in ["open", "vwap", "10:00"]:
        print(f"  执行时点 {s:6s} -> 相对 VWAP {timing_bp(s):+.1f}bp")
    # 3) 用真实测得累计换手 36.7 / 7.75 年 估算年化损益（相对 VWAP；负=省）
    r = annual_cost_from_turnover(cum_turnover=36.7, years=7.75, exec_session="open")
    print(f"\n  实测口径(累计换手36.7, 7.75年, 相对 VWAP 的年化执行损益):")
    print(f"    开盘执行: {r['annual_cost_bp']:+.2f} bp/年 ({r['annual_cost_pct']:+.3f}%)  [负=比VWAP省]")
    print(f"      其中时点贡献 {r['timing_contrib_bp']:+.1f}bp, 冲击贡献 {r['impact_contrib_bp']:.3f}bp")
    r_vwap = annual_cost_from_turnover(cum_turnover=36.7, years=7.75, exec_session="vwap")
    r_bad = annual_cost_from_turnover(cum_turnover=36.7, years=7.75, exec_session="10:00")
    print(f"    VWAP 执行: {r_vwap['annual_cost_bp']:+.2f} bp/年 ({r_vwap['annual_cost_pct']:+.3f}%)")
    print(f"    10:00最贵执行: {r_bad['annual_cost_bp']:+.2f} bp/年 ({r_bad['annual_cost_pct']:+.3f}%)")
    print(f"    → 开盘 vs 10:00 每年差约 {(r_bad['annual_cost_bp']-r['annual_cost_bp'])/100:+.3f}%")
    print("\n  （以上时点/冲击为 pool_hold.py 个股实测移植，对 ETF 为保守上限；真实需 DB 回测复核）")
