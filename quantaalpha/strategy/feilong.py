"""飞龙交易体系（Loop11 SOTA）+ 外部大盘检测闸门。

来源：`reference/rd-agent-research/quantaalpha-bridge/BASELINE_FROM_RD_AGENT.md`
基线锚点（cn_real_r500 + LGB，RD-Agent 会话）：IC≈0.0259 / 含成本年化≈42.4% / 回撤≈-30.9%。

三因子（严格照抄 Loop11 公式，含 eps/clip 防爆）：
  entry = feilong_geo_squeeze_regime
  exit  = exit_pressure_additive
  rr    = rr_atr_regime

**本模块新增：外部大盘检测（Market Regime Gate）** —— 飞龙体系本身没有任何
"外部检查"，在震荡/下跌行情里表现很差（基线回撤 -30.9% 即证据）。这里用
沪深300 指数趋势 + 波动 + 全市场宽度合成 regime 分数，对**建仓与总仓位**做
外部闸门：
    bullish  → 正常建仓（regime_scale=1.0）
    neutral  → 降档建仓（regime_scale=0.5）
    bearish  → 禁止新开仓（regime_scale=0.0），只允许持仓退出
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8


# ---------------------------------------------------------------------------
# 飞龙 Loop11 三因子
# ---------------------------------------------------------------------------

def feilong_factors(g: pd.DataFrame) -> pd.DataFrame:
    """对单只股票的日线面板计算飞龙 entry/exit/rr 三因子。index 需为 datetime。"""
    close, high, low, vol = g["$close"], g["$high"], g["$low"], g["$volume"]

    # --- entry: feilong_geo_squeeze_regime ---
    ema89 = close.ewm(span=89, adjust=False).mean()
    vwap89 = (close * vol).rolling(89, min_periods=45).sum() / vol.rolling(89, min_periods=45).sum().replace(0, np.nan)
    geo_feilong_dist = np.sqrt(
        (close / ema89.replace(0, np.nan)) * (close / vwap89.replace(0, np.nan))
    ) - 1.0
    bb_width = close.rolling(20, min_periods=10).std() / close.rolling(20, min_periods=10).mean()
    bb_width_mean60 = bb_width.rolling(60, min_periods=30).mean()
    # 分母加 eps 并 clip，防 BB_width→0 爆炸（基线文档明确的踩坑点）
    squeeze_intensity = bb_width_mean60 / bb_width.clip(lower=1e-6)
    squeeze_intensity = squeeze_intensity.clip(upper=20.0)
    entry = geo_feilong_dist * squeeze_intensity

    # --- exit: exit_pressure_additive ---
    lo20, hi20 = low.rolling(20, min_periods=10).min(), high.rolling(20, min_periods=10).max()
    range_position_20d = (close - lo20) / (hi20 - lo20 + EPS)
    vwap_extension = (close - vwap89) / vwap89.replace(0, np.nan)
    vol_ratio = vol / vol.rolling(20, min_periods=10).mean().replace(0, np.nan)
    volume_regime = 1.0 - (vol_ratio - 1).abs() / (1.0 + (vol_ratio - 1).abs())
    exit_pressure = (range_position_20d + vwap_extension + volume_regime) / 3.0

    # --- rr: rr_atr_regime ---
    atr14 = (high - low).rolling(14, min_periods=7).mean()
    support = lo20 - atr14
    target = high.rolling(12, min_periods=6).max() + atr14
    raw_rr = (close - support) / (target - close + EPS)
    ret = close.pct_change()
    realized_vol = ret.rolling(60, min_periods=30).std()
    vol_z = (realized_vol - realized_vol.rolling(60, min_periods=30).mean()) / (
        realized_vol.rolling(60, min_periods=30).std() + EPS
    )
    # 关键：vol_z clip。基线文档说「vol_z 再当激进分母/叠除易翻车」，这里只做温和裁剪
    rr = raw_rr / vol_z.clip(lower=-2.0, upper=2.0).replace(0, np.nan)
    rr = rr.clip(lower=-5.0, upper=10.0)

    return pd.DataFrame(
        {"feilong_entry": entry, "feilong_exit": exit_pressure, "feilong_rr": rr, "atr14": atr14},
        index=g.index,
    )


# ---------------------------------------------------------------------------
# 外部大盘检测（Market Regime Gate）—— 本次要补的关键
# ---------------------------------------------------------------------------

def market_regime(
    index_df: pd.DataFrame,
    breadth: pd.Series,
    p: dict | None = None,
) -> pd.DataFrame:
    """用沪深300 + 市场宽度合成 regime。

    breadth: 每个交易日「站上 MA20 的个股占比」(0~1)，index 与之对齐。
    判定四项各 1 分（趋势/位置/波动/宽度），总分 0~4：
        >=3  bullish   正常
        1~2  neutral   降档
        0    bearish   禁新开仓
    """
    p = p or {}
    ma20_n, ma60_n = p.get("index_ma20", 20), p.get("index_ma60", 60)
    vol_win = p.get("index_vol_window", 60)
    dd_win = p.get("index_dd_window", 60)
    vol_pct_window = p.get("index_vol_pct_window", 250)

    close = index_df["$close"]
    idx = pd.DataFrame(index=index_df.index)
    idx["ma20"] = close.rolling(ma20_n, min_periods=ma20_n // 2).mean()
    idx["ma60"] = close.rolling(ma60_n, min_periods=ma60_n // 2).mean()
    idx["dd60"] = close / close.rolling(dd_win, min_periods=dd_win // 2).max() - 1.0
    ret = close.pct_change()
    idx["vol60"] = ret.rolling(vol_win, min_periods=vol_win // 2).std()
    # 波动分位：当前波动在滚动窗口内的分位（越高越危险）
    vp = idx["vol60"].rolling(vol_pct_window, min_periods=60).apply(
        lambda x: float((x[-20:][np.isfinite(x[-20:])] <= x[-1]).mean()) if np.isfinite(x[-1]) else np.nan,
        raw=True,
    )
    idx["vol_pct"] = vp
    idx = idx.join(breadth.rename("breadth"))

    # 四项打分
    s1 = (idx["ma20"] > idx["ma60"]).astype(float)                       # 指数多头排列
    s2 = (idx["dd60"] > -0.08).astype(float)                            # 距60日高点回撤<8%
    s3 = (idx["vol_pct"] < 0.75).astype(float)                          # 波动分位<75%（不是高波恐慌）
    s4 = (idx["breadth"] > 0.45).astype(float)                          # 宽度>45%个股站上MA20
    score = s1 + s2 + s3 + s4
    idx["regime_score"] = score
    regime = np.where(score >= 3, "bullish", np.where(score >= 1, "neutral", "bearish"))
    idx["regime"] = regime
    idx["regime_scale"] = np.where(regime == "bullish", 1.0, np.where(regime == "neutral", 0.5, 0.0))
    return idx


def compute_breadth(panels: dict[str, pd.DataFrame], ma: int = 20) -> pd.Series:
    """市场宽度：每个交易日站上 MA(ma) 的个股占比。"""
    above_total = {}
    for inst, g in panels.items():
        ma_line = g["$close"].rolling(ma, min_periods=ma // 2).mean()
        ok = (g["$close"] > ma_line) & g["$close"].notna() & ma_line.notna()
        above_total[inst] = ok
    if not above_total:
        return pd.Series(dtype=float)
    df = pd.DataFrame(above_total)
    return df.sum(axis=1) / df.notna().sum(axis=1)


# ---------------------------------------------------------------------------
# 飞龙交易计划（叠加外部大盘闸门）
# ---------------------------------------------------------------------------

def generate_feilong_plan(
    panels: dict[str, pd.DataFrame],
    reg: pd.DataFrame,
    signal_date: str,
    top_n: int = 5,
    account: float = 1_000_000.0,
    risk_per_trade: float = 0.01,
    max_position_pct: float = 0.15,
    atr_stop_mult: float = 2.0,
    rr_target_mult: float = 1.0,
    exclude_limit_up: bool = True,
) -> dict:
    """按飞龙 entry 选 top-N，叠加大盘 regime 闸门，产出交易计划。

    regime_scale: bullish=1.0 / neutral=0.5 / bearish=0.0(禁新开仓)。
    仓位 = 风险平价仓位 × regime_scale（bearish 直接 0，不建仓）。
    """
    sd = pd.Timestamp(signal_date)
    if sd not in reg.index:
        raise ValueError(f"{signal_date} 不在 regime 序列内")
    r = reg.loc[sd]
    regime, scale = r["regime"], float(r["regime_scale"])

    # 截面：算三因子，取 entry 最大 top_n
    cand = []
    for inst, g in panels.items():
        if sd not in g.index or len(g) < 90:
            continue
        f = feilong_factors(g)
        if sd not in f.index:
            continue
        row, frow = g.loc[sd], f.loc[sd]
        if any(pd.isna(frow[k]) for k in ("feilong_entry", "feilong_rr", "feilong_exit")):
            continue
        if exclude_limit_up and row["$high"] == row["$low"] == row["$close"] and row["$close"] / g["$close"].iloc[-2] - 1 > 0.09:
            continue
        cand.append({
            "instrument": inst, "entry": float(frow["feilong_entry"]),
            "exit_pressure": float(frow["feilong_exit"]), "rr": float(frow["feilong_rr"]),
            "close": float(row["$close"]), "atr": float(frow["atr14"]) if not pd.isna(frow["atr14"]) else float(row["$close"]) * 0.02,
        })
    if not cand:
        return {"signal_date": signal_date, "regime": regime, "regime_scale": scale,
                "note": "当日无有效飞龙候选", "plans": []}
    C = pd.DataFrame(cand).sort_values("entry", ascending=False).head(top_n)

    plans = []
    for _, x in C.iterrows():
        close, atr = x["close"], max(x["atr"], 1e-6)
        stop = close * (1 - np.clip(atr_stop_mult * atr / close, 0.04, 0.12))
        # 目标：受 rr 因子调节（rr 越高目标越远，但受 ATR 兜底）
        tgt_mult = np.clip(rr_target_mult * (1.0 + 0.15 * x["rr"]), 1.0, 3.0)
        tp = close + tgt_mult * atr * 2.0
        risk_px = max(close - stop, 1e-6)
        # 风险平价：股数 = min(风险预算/每股风险, 单票上限/股价)
        shares_full = min(risk_per_trade * account / risk_px, max_position_pct * account / close)
        shares = shares_full * scale
        pos_value = shares * close  # 仓位市值（元）
        plans.append({
            "instrument": x["instrument"], "feilong_entry": round(x["entry"], 4),
            "feilong_exit_pressure": round(x["exit_pressure"], 4), "feilong_rr": round(x["rr"], 3),
            "close_T": round(close, 2), "stop_loss": round(stop, 2),
            "stop_pct": round((close - stop) / close * 100, 2),
            "take_profit": round(tp, 2), "tp_pct": round((tp / close - 1) * 100, 2),
            "shares": int(shares),
            "position_value": round(pos_value, 0),
            "position_pct": round(pos_value / account * 100, 2),
            "regime": regime, "regime_scale": scale,
        })
    return {
        "signal_date": signal_date, "regime": regime, "regime_score": int(r["regime_score"]),
        "regime_scale": scale, "index_ma20": round(float(r["ma20"]), 1), "index_ma60": round(float(r["ma60"]), 1),
        "index_dd60": round(float(r["dd60"]) * 100, 1), "breadth": round(float(r["breadth"]) * 100, 0),
        "n_candidates": len(cand), "plans": plans,
    }
