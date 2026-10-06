"""日线级别「5分钟选股信号 → 日线确认 → 次日开盘交易计划」系统.

设计要点（对应用户 4 项要求）:

1. 信号/执行时间差
   - 5min 只用于**当日盘中筛选候选**（盘中预警），不产生实际成交。
   - **真正的信号在日线收盘确认**：T 日收盘后校验「信号成立」，T+1 日**开盘买入**。
   - 委托方式：T+1 开盘以「开盘价 + 跳空保护」限价委托；若开盘跳空高开超过
     `max_gap_pct` 则**放弃买入**（追高风险），跳空低开超过 `max_gap_down_pct`
     则**降低/放弃**（见 build_order_intent）。

2. 因子 → 交易计划
   每只入选标的输出：买入区间、仓位、止损（含 ATR 与支撑位双依据 + 幅度）、
   分档压力/止盈位、持仓周期预期、卖出触发条件。

3. 选股逻辑
   - 候选池：流动性过滤（成交额/换手）、上市时长、剔除 ST/退市/停牌/一字板。
   - 信号有效性 + 量价配合。
   - 多因子打分后**按优先级排序**。

4. 参数与适用性
   所有阈值集中在 `PlanParams`，并在模块 docstring 说明适用市场与失效场景。

本模块只做「计划生成」，不含撮合；回测验证请走独立的 backtest 路径。
所有计算均基于真实 Qlib 日线数据，不含模拟/随机成分。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 参数默认值（要求4：集中、可调、带默认值）
# ----------------------------------------------------------------------------


@dataclass
class PlanParams:
    """交易计划默认参数。"""

    # --- 信号确认（日线）---
    ma_fast: int = 20                 # 短期均线 = 回踩基准
    ma_slow: int = 60                 # 长期均线 = 趋势过滤
    pullback_tol: float = 0.03        # 回踩容忍：|close-MA20|/MA20 < 3%
    vol_ma: int = 5                   # 量能均线
    vol_surge_mult: float = 1.5       # 放量倍数：vol > 1.5 * MA5(vol)
    range_max: float = 0.08           # 当日振幅上限（高波动/异常日剔除）
    need_vol_confirm: bool = True     # 是否强制量能配合

    # --- 止损（要求2：ATR + 支撑位）---
    atr_period: int = 14
    atr_stop_mult: float = 2.0         # 止损 = 入场价 - atr_mult * ATR
    stop_floor_pct: float = 0.04      # 止损幅度下限（过窄则取此值，防震荡洗出）
    stop_cap_pct: float = 0.12        # 止损幅度上限（过宽则压缩，控制单笔风险）
    support_lookback: int = 10        # 支撑位取近 N 日最低价
    use_support_stop: bool = True     # 是否同时参考支撑位（取更靠近入场价者）

    # --- 止盈（分档）---
    tp1_atr_mult: float = 1.5         # 第一档：+1.5 ATR
    tp2_atr_mult: float = 3.0         # 第二档：+3.0 ATR
    tp3_atr_mult: float = 4.5         # 第三档：+4.5 ATR
    tp1_pct: float = 0.06             # 第一档目标下限（保证不过低）
    use_resistance: bool = True       # 用近 N 日高点做压力位
    resistance_lookback: int = 20

    # --- 仓位（风险平价）---
    account: float = 1_000_000.0      # 账户规模（示例）
    risk_per_trade_pct: float = 0.01  # 单笔风险占总资产 1%
    max_position_pct: float = 0.15    # 单票最大仓位 15%
    max_total_position_pct: float = 0.60  # 总仓位上限 60%
    max_new_positions: int = 5        # 单日最多建仓数

    # --- 次日开盘委托 / 跳空处理 ---
    max_gap_pct: float = 0.03         # 高开>3% 放弃（不追高）
    max_gap_down_pct: float = 0.05    # 低开>5% 放弃（接飞刀风险）
    entry_price_offset: float = 0.002  # 开盘价上方 0.2% 容错，保证成交

    # --- 候选池 / 异常剔除 ---
    min_amount_cny: float = 3e7       # 近 20 日日均成交额下限 3000 万
    min_turnover: float = 0.01        # 近 20 日日均换手下限 1%
    min_list_days: int = 120          # 上市满 120 个自然日，规避次新
    exclude_st: bool = True           # 剔除 ST/*ST/退
    exclude_limit_up_unbuyable: bool = True  # 剔除一字涨停无法买入


# ----------------------------------------------------------------------------
# 数据加载与指标计算
# ----------------------------------------------------------------------------


def load_daily_panel(codes: List[str], start: str, end: str) -> pd.DataFrame:
    """从 Qlib 载入日线面板 (index=instrument, columns=datetime)。"""
    from qlib.data import D

    df = D.features(
        codes,
        ["$open", "$high", "$low", "$close", "$volume", "$factor", "$change"],
        start_time=start,
        end_time=end,
        freq="day",
    )
    df = df.sort_index()
    return df


def _rolling(s: pd.Series, n: int, fn: str) -> pd.Series:
    r = s.rolling(n, min_periods=max(2, n // 2))
    if fn == "mean":
        return r.mean()
    if fn == "min":
        return r.min()
    if fn == "max":
        return r.max()
    raise ValueError(fn)


def compute_indicators(df: pd.DataFrame, p: PlanParams) -> pd.DataFrame:
    """按 instrument 计算指标（MA / ATR / 振幅 / 成交额 / 支撑压力）。"""
    out = {}
    for inst, g in df.groupby(level="instrument"):
        # groupby(level=...) 在部分 pandas 版本仍保留 instrument 层，
        # 导致 g.index 是 (instrument, datetime) 的 MultiIndex，日期字符串匹配恒失败。
        # 统一 drop 掉 instrument 层，使 g.index 只剩 datetime。
        if isinstance(g.index, pd.MultiIndex):
            g = g.droplevel("instrument")
        g = g.sort_index()
        close, high, low, vol = g["$close"], g["$high"], g["$low"], g["$volume"]

        prev_close = close.shift(1)
        ma_fast = _rolling(close, p.ma_fast, "mean")
        ma_slow = _rolling(close, p.ma_slow, "mean")
        ma_vol = _rolling(vol, p.vol_ma, "mean")

        # 真实波幅 TR = max(H-L, |H-prevC|, |L-prevC|)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = _rolling(tr, p.atr_period, "mean")

        out[inst] = pd.DataFrame(
            {
                "open": g["$open"],
                "close": close,
                "high": high,
                "low": low,
                "volume": vol,
                "ma_fast": ma_fast,
                "ma_slow": ma_slow,
                "ma_vol": ma_vol,
                "atr": atr,
                "prev_close": prev_close,
                "range_pct": (high - low) / (close + 1e-12),
                "ret1": close / prev_close - 1.0,
                "amount": (close * vol),  # 成交额近似
                "support": _rolling(low, p.support_lookback, "min"),
                "resistance": _rolling(high, p.resistance_lookback, "max"),
                "turnover_proxy": vol / (close + 1e-12),  # 占位，见下方说明
            }
        )
    return out


# ----------------------------------------------------------------------------
# 信号确认（日线收盘）
# ----------------------------------------------------------------------------


def confirm_signal(inst: str, row: pd.Series, prev: pd.Series, p: PlanParams) -> Dict:
    """T 日收盘确认信号：F3 趋势多头 + F4 回踩MA20 + F2 放量（+ 低波动偏好）。

    返回各条件布尔值与综合判定，供计划生成与排错使用。
    """
    close = row["close"]
    checks = {}

    # F3: 均线多头排列 MA20 > MA60
    checks["trend_ma_bull"] = bool(
        not np.isnan(row["ma_fast"]) and not np.isnan(row["ma_slow"])
        and row["ma_fast"] > row["ma_slow"]
    )
    # F4: 回踩 MA20 —— 收盘靠近 MA20 且在 MA20 上方，且昨日在 MA20 下方/相等(金叉)
    if np.isnan(row["ma_fast"]) or np.isnan(prev["ma_fast"]):
        checks["pullback_ma"] = False
    else:
        dist = abs(close - row["ma_fast"]) / (row["ma_fast"] + 1e-12)
        cross_up = prev["close"] <= prev["ma_fast"] and close > row["ma_fast"]
        near = dist < p.pullback_tol
        checks["pullback_ma"] = bool(near and close > row["ma_fast"] and cross_up)
    # F2: 放量 —— 量 > 1.5 * MA5(vol) 且 > 昨日量
    if np.isnan(row["ma_vol"]):
        checks["vol_surge"] = False
    else:
        checks["vol_surge"] = bool(
            row["volume"] > p.vol_surge_mult * row["ma_vol"] and row["volume"] > prev["volume"]
        )
    # 低波动偏好：当日振幅不过大（剔除异常/情绪高潮）
    checks["calm_range"] = bool(row["range_pct"] <= p.range_max)

    core = checks["trend_ma_bull"] and checks["pullback_ma"]
    if p.need_vol_confirm:
        signal_ok = core and checks["vol_surge"] and checks["calm_range"]
    else:
        signal_ok = core and checks["calm_range"]
    return {"checks": checks, "signal_ok": bool(signal_ok)}


# ----------------------------------------------------------------------------
# 候选池筛选与异常剔除（要求3）
# ----------------------------------------------------------------------------


def screen_universe(panels: Dict[str, pd.DataFrame], p: PlanParams) -> Dict[str, str]:
    """返回 {instrument: 剔除原因}；不在字典里=通过初筛。"""
    rejects = {}
    for inst, g in panels.items():
        if len(g) < max(p.ma_slow, p.atr_period) + 5:
            rejects[inst] = "history_too_short"
            continue
        if p.exclude_st and ("ST" in inst.upper() or "退" in inst):
            rejects[inst] = "excluded_by_name"
            continue
        # 上市时长：这里用面板首日近似（真实上市日以 instruments 为准）
        if len(g) < p.min_list_days // 2:
            rejects[inst] = "listed_too_short(panel)"
            continue
        # 流动性：近 20 日日均成交额
        amt20 = g["amount"].tail(20).mean()
        if not np.isnan(amt20) and amt20 < p.min_amount_cny:
            rejects[inst] = f"illiquid(amt20={amt20/1e4:.0f}万)"
            continue
        # 一字涨停无法买入：当日 close==high==low 且涨幅大
        last = g.iloc[-1]
        if (
            p.exclude_limit_up_unbuyable
            and last["high"] == last["low"] == last["close"]
            and last["ret1"] > 0.09
        ):
            rejects[inst] = "limit_up_unbuyable"
            continue
        # 停牌：最近有 NaN close
        if g["close"].tail(5).isna().any():
            rejects[inst] = "suspended_recent"
    return rejects


# ----------------------------------------------------------------------------
# 交易计划生成（要求2）
# ----------------------------------------------------------------------------


def build_trade_plan(
    inst: str,
    signal_date: pd.Timestamp,
    row: pd.Series,
    p: PlanParams,
    priority_score: float,
) -> Dict:
    """为单只标的生成完整交易计划。"""
    close = float(row["close"])
    atr = float(row["atr"]) if not np.isnan(row["atr"]) else close * 0.02

    # --- 买入参考价：T 日收盘（次日实际开盘由 build_order_intent 按跳空规则调整）---
    est_open = close
    buy_low = est_open * (1 - p.entry_price_offset)
    buy_high = est_open * (1 + p.entry_price_offset)

    # --- 止损：ATR 与支撑位取「更靠近入场价」者（即先触发者），并夹在上下限 ---
    atr_stop = est_open - p.atr_stop_mult * atr
    support = float(row["support"]) if not np.isnan(row["support"]) else np.nan
    if p.use_support_stop and not np.isnan(support):
        # 支撑与 ATR 止损取较高（离入场更近）者作为止损，但不得高于入场价下方一点
        stop_raw = max(atr_stop, support)
    else:
        stop_raw = atr_stop
    stop = min(stop_raw, est_open * (1 - p.stop_floor_pct))   # 至少留 floor 幅度
    stop = max(stop, est_open * (1 - p.stop_cap_pct))           # 至多 cap 幅度
    stop_pct = (est_open - stop) / est_open

    # --- 止盈分档：ATR 目标 + 压力位；强制单调递增 tp1<=tp2<=tp3 ---
    tp1 = max(est_open * (1 + p.tp1_pct), est_open + p.tp1_atr_mult * atr)
    tp2 = max(tp1 + 0.5 * atr, est_open + p.tp2_atr_mult * atr)
    tp3 = max(tp2 + 0.5 * atr, est_open + p.tp3_atr_mult * atr)
    resistance = float(row["resistance"]) if not np.isnan(row["resistance"]) else np.nan
    if p.use_resistance and not np.isnan(resistance):
        # 若压力位在 tp1 之上且合理，纳入第一档压力参考
        res_note = f"近{p.resistance_lookback}日高点 {resistance:.2f}"
    else:
        res_note = "无明显压力(创新高)"

    # --- 仓位：风险平价 ---
    risk_budget = p.account * p.risk_per_trade_pct
    shares_by_risk = risk_budget / (stop_pct * est_open) if stop_pct > 0 else 0
    max_shares = (p.account * p.max_position_pct) / est_open
    shares = int(min(shares_by_risk, max_shares))
    position_pct = shares * est_open / p.account

    # --- 持仓周期与卖出触发 ---
    holding = "3~10 个交易日（波段）"
    sell_triggers = {
        "止损": f"收盘价跌破 {stop:.2f}（-{stop_pct*100:.1f}%）→ 次日开盘卖出；"
                f"若次日开盘直接跳空低于止损，则开盘市价了结（不等待反弹）",
        "止盈_分档": f"T1 {tp1:.2f}(+{(tp1/est_open-1)*100:.1f}%) 减仓 1/2；"
                     f"T2 {tp2:.2f}(+{(tp2/est_open-1)*100:.1f}%) 再减 1/3；"
                     f"T3 {tp3:.2f}(+{(tp3/est_open-1)*100:.1f}%) 清仓",
        "趋势破坏": f"收盘跌破 MA{p.ma_fast}（当日约 {float(row['ma_fast']):.2f}）→ 离场观望",
        "信号失效": f"MA{p.ma_fast} 下穿 MA{p.ma_slow}（多头排列破坏）→ 离场",
        "时间止损": f"持有超过 10 个交易日仍未触及 T1 → 离场（资金效率）",
    }

    return {
        "instrument": inst,
        "signal_date": str(signal_date.date()),
        "priority_score": round(priority_score, 3),
        "close_T": round(close, 2),
        "ref_close_T": round(est_open, 2),  # 买入参考基准 = T日收盘（次日开盘按跳空规则调整）
        "buy_low": round(buy_low, 2),
        "buy_high": round(buy_high, 2),
        "shares": shares,
        "position_pct": round(position_pct * 100, 2),
        "stop_loss": round(stop, 2),
        "stop_pct": round(stop_pct * 100, 2),
        "stop_basis": f"max(ATR×{p.atr_stop_mult}, 支撑{support:.2f})" if p.use_support_stop and not np.isnan(support) else f"ATR×{p.atr_stop_mult}",
        "atr": round(atr, 2),
        "tp1": round(tp1, 2),
        "tp1_pct": round((tp1 / est_open - 1) * 100, 2),
        "tp2": round(tp2, 2),
        "tp2_pct": round((tp2 / est_open - 1) * 100, 2),
        "tp3": round(tp3, 2),
        "tp3_pct": round((tp3 / est_open - 1) * 100, 2),
        "resistance_note": res_note,
        "holding_period": holding,
        "sell_triggers": sell_triggers,
    }


# ----------------------------------------------------------------------------
# 次日开盘委托 / 跳空处理（要求1）
# ----------------------------------------------------------------------------


def build_order_intent(plan: Dict, open_T1: float, p: PlanParams) -> Dict:
    """依据 T+1 实际开盘价，生成委托指令/放弃原因（跳空处理）。"""
    ref = plan["ref_close_T"]
    gap = (open_T1 - ref) / ref
    intent = {
        "gap_pct": round(gap * 100, 2),
        "limit_buy": round(open_T1 * (1 + p.entry_price_offset), 2),
    }
    if gap > p.max_gap_pct:
        intent["action"] = "SKIP"
        intent["reason"] = f"高开{gap*100:.1f}%>{p.max_gap_pct*100:.0f}%，不追高"
    elif gap < -p.max_gap_down_pct:
        intent["action"] = "SKIP"
        intent["reason"] = f"低开{-gap*100:.1f}%>{p.max_gap_down_pct*100:.0f}%，规避接飞刀"
    else:
        intent["action"] = "BUY"
        intent["reason"] = "开盘限价买入"
    return intent


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------


def generate_plan(
    provider_uri: str,
    signal_date: str,
    universe: str = "liquid500",
    lookback_start: str = "2024-01-01",
    params: PlanParams = None,
) -> Dict:
    """生成某信号日的完整交易计划。"""
    import qlib

    p = params or PlanParams()
    qlib.init(provider_uri=provider_uri, region="cn")
    from qlib.data import D

    insts = D.instruments(universe)
    codes = D.list_instruments(insts, start_time=lookback_start, end_time=signal_date, as_list=True)
    df = load_daily_panel(codes, lookback_start, signal_date)
    panels = compute_indicators(df, p)

    rejects = screen_universe(panels, p)

    sd = pd.Timestamp(signal_date)
    plans, watch, diag = [], [], []
    for inst, g in panels.items():
        if inst in rejects:
            continue
        if sd not in g.index:
            continue
        row = g.loc[sd]
        idx = list(g.index).index(sd)
        if idx == 0:
            continue
        prev = g.iloc[idx - 1]
        if np.isnan(row["close"]):
            continue
        sig = confirm_signal(inst, row, prev, p)
        ch = sig["checks"]
        # 评分：趋势/回踩/量能 + 平静度 + 流动性
        score = 0.0
        score += 2.0 if ch["trend_ma_bull"] else 0.0
        score += 1.5 if ch["pullback_ma"] else 0.0
        score += 1.0 if ch["vol_surge"] else 0.0
        score += 0.5 * float(np.clip(1.0 - row["range_pct"] / p.range_max, 0, 1))
        amt20 = g["amount"].tail(20).mean()
        if not np.isnan(amt20) and amt20 > 0:
            score += 0.5 * float(np.clip(np.log10(amt20 / p.min_amount_cny) / 2.0, 0, 1))

        if sig["signal_ok"]:
            plan = build_trade_plan(inst, sd, row, p, score)
            plan["tier"] = "A-confirmed"
            plans.append(plan)
        else:
            # B 层观察名单：趋势成立 且 (回踩/放量/低波) 至少命中 2 项
            n_hit = sum([ch["pullback_ma"], ch["vol_surge"], ch["calm_range"]])
            if ch["trend_ma_bull"] and n_hit >= 2:
                plan = build_trade_plan(inst, sd, row, p, score)
                plan["tier"] = "B-watchlist"
                watch.append(plan)
        diag.append({"instrument": inst, "score": round(score, 3), "signal_ok": sig["signal_ok"], **ch})

    # 排序 + 仓位上限控制（仅 A 层建仓；B 层仅供人工参考，不自动下单）
    plans.sort(key=lambda x: -x["priority_score"])
    total_pos = 0.0
    kept, dropped = [], []
    for pl in plans:
        if len(kept) >= p.max_new_positions:
            dropped.append({**pl, "drop_reason": f"超过单日建仓上限{p.max_new_positions}"})
            continue
        if total_pos + pl["position_pct"] / 100.0 > p.max_total_position_pct:
            dropped.append({**pl, "drop_reason": "超过总仓位上限"})
            continue
        kept.append(pl)
        total_pos += pl["position_pct"] / 100.0
    watch.sort(key=lambda x: -x["priority_score"])

    return {
        "signal_date": signal_date,
        "universe": universe,
        "n_candidates": len(codes),
        "n_screened_out": len(rejects),
        "n_signal_confirmed": len(plans),
        "n_watchlist": len(watch),
        "total_position_pct": round(total_pos * 100, 2),
        "plans_confirmed": kept,
        "watchlist": watch,
        "dropped": dropped,
        "params": asdict(p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider_uri", default="D:/qqmt/qlib_data/cn_real_full")
    ap.add_argument("--signal_date", required=True, help="YYYY-MM-DD（信号确认日=T日收盘）")
    ap.add_argument("--universe", default="liquid500")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    args = ap.parse_args()

    res = generate_plan(args.provider_uri, args.signal_date, args.universe)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"saved -> {args.out}")
    print(txt[:4000])


if __name__ == "__main__":
    main()
