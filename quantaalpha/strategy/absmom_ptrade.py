# -*- coding: utf-8 -*-
"""
跨资产趋势跟随（200日SMA + 逆波动率）— Ptrade 版
================================================
数据：Ptrade 回测/实盘自有数据（attribute_history）
逻辑（与 QMT 版 absmom_qmt.py 完全一致，逐行对应）：
  - 每月末对 POOL 每只 ETF 判：close > 200日SMA -> 持有，否则该份额转现金
  - 持有子集内按逆波动率分配权重：w_i = 1/σ_i（σ=60日年化波动，下限0.05），归一
  - 月频调仓，佣金万1；信号用前一日收盘，次月生效（无前视）
运行：直接丢进 Ptrade 回测器加载本文件即可；实盘同理。
"""
import numpy as np
import pandas as pd

# ---------------- 参数 ----------------
POOL = [
    "159915.SZ", "159949.SZ", "159967.SZ", "159981.SZ", "159985.SZ", "161226.SZ",
    "501018.SH", "510030.SH", "510300.SH", "510500.SH", "510900.SH", "511010.SH",
    "511090.SH", "511220.SH", "511380.SH", "511880.SH", "512100.SH", "512400.SH",
    "512500.SH", "512890.SH", "513030.SH", "513050.SH", "513080.SH", "513090.SH",
    "513100.SH", "513130.SH", "513180.SH", "513330.SH", "513500.SH", "513520.SH",
    "513650.SH", "513660.SH", "513730.SH", "513980.SH", "518880.SH", "563300.SH",
    "588000.SH", "588080.SH",
]
SMA_W = 200
VOL_WINDOW = 60
INV_VOL_FLOOR = 0.05
COMM = 0.0001
EXPOSURE = 1.0   # 风险暴露比例：1.0=满仓(底盘)；0.6=低回撤档。与 QMT 版一致。


def initialize(context):
    context.pool = POOL
    context.last_month = None
    context.weights = {}
    set_option("use_real_price", True)
    set_benchmark("000300.SH")          # 沪深300 仅作环境参照


def _signal(context):
    """返回 (held_codes, weights_dict)。"""
    held, vols = [], {}
    for code in context.pool:
        closes = attribute_history(code, SMA_W, "1d", "close", df=False)
        if closes is None or len(closes) < SMA_W:
            continue
        closes = np.array(closes, dtype=float)
        sma = float(np.mean(closes[-SMA_W:]))
        if closes[-1] > sma:
            held.append(code)
            ret = pd.Series(closes).pct_change().dropna()
            vols[code] = float(ret.tail(VOL_WINDOW).std() * np.sqrt(252))
    if not held:
        return [], {}
    inv = {x: 1.0 / max(vols[x], INV_VOL_FLOOR) for x in held}
    tot = sum(inv.values())
    return held, {x: inv[x] / tot for x in held}


def handle_data(context, data):
    """每个交易日触发；月末调仓。"""
    today = context.current_dt
    if context.last_month is not None \
            and today.month == context.last_month.month \
            and today.year == context.last_month.year:
        return
    context.last_month = today

    held, w = _signal(context)
    context.weights = w
    total = context.portfolio.total_value
    investable = total * EXPOSURE
    for code in context.pool:
        if code in w:
            order_target_value(code, investable * w[code])
        else:
            order_target_value(code, 0)
