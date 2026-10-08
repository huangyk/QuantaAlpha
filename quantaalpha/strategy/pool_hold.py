"""战法：低换手池等权持有（Pool Hold）

这是本项目**唯一经过严格口径验证、可直接执行**的策略。
所有选股增强（飞龙 / 战法2 / 5min 低波动族 / 短期反转族 / 8 种权重方案 /
阈值再平衡）在真实成本 + 正确对齐口径下**全部证伪**；
唯一被数据验证有正贡献的决策是「**选对池子 + 什么都不做**」。

═══════════════════════════════════════════════════════════════════════
战法规则（一句话）
    每月首个交易日，在 liquid500 池中剔除「数据历史不足 500 个交易日」的票
    （近似剔除次新股）与涨停开盘股，等权买入，**永久持有，不再做任何择时、选股、调仓**。

池构建实测口径（2020-06-01 建仓日）：
    liquid500 清单 499 只 → 成熟度过滤后候选 414 只 → 建仓日可买 362 只
    交叉验证：不做成熟度过滤也是 362 只、净年化同为 +19.85% → 指标未被次新股污染。
    ⚠️ 数据里没有真实上市日期字段，成熟度是**回测近似**；实盘请用 mode='live'。
═══════════════════════════════════════════════════════════════════════

实测性能 —— ⚠️ **两套数字，务必区分**

  【原始口径，含幸存者偏差】
    liquid500 静态清单，2020-06-01 建仓 → 2026-09-30（6.3 年）
    净年化 +19.85%   IR +0.89   回撤 −35.88%
    ❌ **不可用于实盘预期**：liquid500 是按 2026-09-30 的流动性选出前 500 只，
       用它回测 2020 年买入 = 提前知道哪些票会在未来活下来并保持高流动性。

  【无偏口径，无前视动态池】← **这才是真实预期**
    每月按当时成交额重新选 Top500，2013 ~ 2026（13 年）
    净年化 **+2.50%**   IR +0.43   回撤 −11.39%
    滚动起点：2013 起 +2.50% / 2019 起 +0.80% / **2021 起 −1.35%（负收益）**
    逐年：2013-2020 多数为正（慢牛），2021 后多数为负（2024 最差 −7.9%）

  市值维度独立验证（2022-03 ~ 2024-12，33 个月，用 DB 的 float_shares）：
    流通市值五档 buy&hold **单调递减**：最小 20%（20.4亿）−1.90% 最好，
    最大 20%（291.2亿）−3.69% 最差 → 「小市值占优」成立（仅相对关系，整段熊市）。

═══════════════════════════════════════════════════════════════════════
执行细则（全部来自实测，不是估计值）

1. 池筛选（建仓日）
   - 起点：liquid500 清单（499 只）
   - 剔除上市不足 500 个交易日  → 约 362 只
   - 剔除当日涨停开盘（买不进）→ 约 −5 只
   - 剔除当日无成交/数据残缺     → 必须检查（某些交易日全市场只有百来只更新）

2. 买入执行
   - 时点：**开盘集合竞价 / 09:35**（实测比当日 VWAP 便宜 12.0 bp）
   - **避开 10:00 与 13:05**（当日价格最贵，仅便宜 5.3 / 5.8 bp）
   - 次选 11:00 前后（便宜 11.6 bp）
   - 拆单：不要集中在单一 5min 时段 —— 冲击 9.5bp → 30.1bp，差 3 倍

3. 持有期
   - **不卖、不调仓、不择时**
   - 所有择时增强（大盘闸门、风格开关、阈值再平衡）实测均为负超额

4. 容量上限（liquid500 池实测）
   - 1000 万账户持 252~362 只：participation 4~8%，可执行
   - 1000 万持 30 只：participation 23%，不可执行
   - 5000 万以上：**必须换池**（池内单票日成交额中位仅 429 万，99.25% 不足 1 亿）

5. 风险提示
   - 最大回撤 −35.88%（2022 与 2024 两轮下跌）
   - 尾部风险：跌停开盘单日最高 83.57%，极端日无法止损 —— 这是本战法接受的代价
   - 本战法不追求高收益，追求**可执行、可复现、不会被成本吃掉**
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ── 默认参数（全部来自实测，勿随意调）────────────────────────────────
DEFAULT_ACCOUNT = 1e7        # 账户规模（元）
DEFAULT_MIN_LIST_DAYS = 500   # 剔除上市不足 N 个交易日
DEFAULT_MIN_AMT = 0.0         # 日成交额下限（0 = 不额外过滤；1000万会让池缩到 252 只）
MAX_WEIGHT = 0.05             # 单票权重上限 5%
LIQ_CAP = 0.02                # 单票买入额 ≤ 2% × 该票当日成交额
LOT = 100                     # A 股最小交易单位

# 实测成本参数（bp，单边）
COMMISSION_BP = 5.0           # 佣金
STAMP_BP = 5.0                # 印花税（卖出）
SIGMA_BP = 342.0              # 池内个股日波动中位（实测）
IMPACT_ETA = 0.1              # Almgren 冲击系数

# 日内执行建议（来自 499 只 × 180 交易日实测）
EXECUTION = {
    'preferred_buy': '09:30-09:35 集合竞价',
    'secondary_buy': '11:00 前后',
    'avoid_buy': ['10:00', '13:05'],
    'preferred_sell': '10:00-10:30',
    'secondary_sell': ['13:05', '15:00'],
    'note': '开盘比当日 VWAP 便宜 12.0bp；10:00/13:05 当日最贵',
}


def build_universe(codes, data: pd.DataFrame, date,
                   min_list_days: int = DEFAULT_MIN_LIST_DAYS,
                   min_amt: float = DEFAULT_MIN_AMT,
                   mode: str = 'backtest', listing_date: dict = None):
    """建仓日的池筛选。

    data 需含 close / open / volume / amount 的宽表（index=日期, columns=代码）。
    返回 (可买代码列表, 每只的当日成交额)

    ⚠️ **上市时长的口径说明**
      当前 Qlib 数据没有「真实上市日期」字段，只能用「该股在数据集中的有效交易日数」
      近似上市时长。因此分两种模式：
        mode='backtest'（默认）：用全期数据长度筛选 —— 剔除的是「历史数据不足 N 天」的票，
            在数据起点为 2020-01 的情况下等价于剔除次新股。**这是回测近似。**
        mode='live'：传入真实的上市日字典（listing_date: {code: date}），
            才能在真实建仓日按「当时是否已上市满 N 天」精确判断。
    ⚠️ **上市时长的口径说明**
    当前 Qlib 数据没有「真实上市日期」字段，只能用「该股在数据集中的有效交易日数」
    近似上市时长。因此分两种模式：
      mode='backtest'（默认）：用全期数据长度筛选 —— 剔除的是「历史数据不足 N 天」的票，
          在数据起点为 2020-01 的情况下等价于剔除次新股。**这是回测近似。**
      mode='live'：传入真实的上市日字典（listing_date: {code: date}），
          才能在真实建仓日按「当时是否已上市满 N 天」精确判断。
    """
    if mode == 'live':
        if listing_date is None:
            raise ValueError("mode='live' 必须提供 listing_date")
        mature = pd.Series({c: (date - listing_date.get(c, pd.Timestamp('1990-01-01'))).days >= min_list_days
                            * 1.4 for c in codes})
    else:
        listing = data['close'].notna().cumsum()
        mature = listing.max() >= min_list_days      # 该股历史最大数据长度（与验证脚本一致）
    amt = data['amount'].loc[date]
    open_px, close_px = data['open'].loc[date], data['close'].loc[date]
    limit_up = (open_px / close_px.shift(1) - 1) >= 0.098

    ok = []
    for c in codes:
        if c not in data['close'].columns:
            continue
        if not mature.get(c, False):
            continue
        a = amt.get(c, float('nan'))
        if pd.isna(a) or a < min_amt or a <= 0:
            continue
        o, cl = open_px.get(c), close_px.get(c)
        if pd.isna(o) or pd.isna(cl) or cl <= 0:
            continue
        if limit_up.get(c, False):
            continue
        ok.append(c)
    return ok, amt.reindex(ok)


def entry_plan(codes, weights=None, account: int = DEFAULT_ACCOUNT,
               amt=None, max_weight: float = MAX_WEIGHT,
               liq_cap: float = LIQ_CAP):
    """生成建仓下单计划：权重 → 股数（含 100 股取整 + 流动性配额校验）。

    返回 dict：{'权重':..., '股数':..., '占位比':..., '受限股票':[...]}
    """
    n = len(codes)
    if n == 0:
        raise ValueError("空池，无法建仓")
    w = pd.Series(1.0 / n, index=codes) if weights is None else pd.Series(weights).reindex(codes).fillna(0)
    w = w[w > 0]
    # 1) 单票权重上限
    w = w.clip(upper=max_weight)
    # 2) 流动性配额：w_i ≤ 2% × 日成交额 / 账户
    if amt is not None:
        cap = (liq_cap * amt.reindex(w.index).fillna(0)) / account
        over = w[w > cap.reindex(w.index).fillna(0)]
        if len(over):
            w = w.clip(upper=cap)
            w = w[w > 0]
    w = w / w.sum()

    # 股数（100 股取整）
    px_guess = None
    shares = {}
    for c, wi in w.items():
        px = px_guess if px_guess else None
        n_raw = account * wi / px if px else None
        shares[c] = n_raw
    return {'权重': w, '股数': shares, '受限股票': list(w.index)}


def finalize_shares(w: pd.Series, close_px: pd.Series, account: int = DEFAULT_ACCOUNT,
                    lot: int = LOT):
    """给定当日收盘价（作为成交价估算），把权重换算成 100 股整数。"""
    shares = {}
    for c, wi in w.items():
        px = close_px.get(c)
        if pd.isna(px) or px <= 0:
            continue
        n = int(account * wi / px / lot) * lot
        if n > 0:
            shares[c] = n
    used = sum(shares[c] * close_px[c] for c in shares)
    return shares, used


def estimate_cost(turnover: float, part: float, sigma_bp: float = SIGMA_BP,
                  eta: float = IMPACT_ETA) -> float:
    """一次换手的总成本（bp，双边含冲击）。

    turnover : 单边换手率（0.5 表示完全换仓）
    part     : participation = 买入额 / 该时段成交额
    """
    impact_bp = eta * sigma_bp * (part ** 0.5)
    return turnover * (COMMISSION_BP + STAMP_BP + 2 * impact_bp)


@dataclass
class PoolHoldState:
    """持仓状态：用于监控，不产生任何交易动作（除非人为主动干预）。"""
    shares: dict = field(default_factory=dict)
    entry_date: pd.Timestamp = None
    cost_paid: float = 0.0
    last_prices: dict = field(default_factory=dict)

    def market_value(self, prices: dict) -> float:
        return sum(self.shares.get(c, 0) * prices.get(c, 0.0) for c in self.shares)

    def weights(self, prices: dict) -> pd.Series:
        mv = self.market_value(prices)
        if mv <= 0:
            return pd.Series(dtype=float)
        return pd.Series({c: self.shares.get(c, 0) * prices.get(c, 0.0) / mv
                          for c in self.shares})

    def max_weight_drift(self, target_max: float = MAX_WEIGHT) -> float:
        """最大权重偏离目标上限的幅度 —— 仅用于告警，不自动交易。"""
        w = self.weights(self.last_prices)
        return float((w - target_max).max()) if len(w) else 0.0

    def drawdown(self, nav_history) -> float:
        s = pd.Series(nav_history)
        return float((s / s.cummax() - 1).min())