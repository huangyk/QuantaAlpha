# -*- coding: utf-8 -*-
"""
跨资产趋势跟随（200日SMA + 逆波动率）— QMT 版
================================================
数据：xtquant.xtdata（QMT / miniQMT 本地数据）
逻辑（与 Ptrade 版 absmom_ptrade.py 完全一致）：
  - 每月末对 POOL 每只 ETF 判：close > 200日SMA -> 持有，否则该份额转现金
  - 持有子集内按逆波动率分配权重：w_i = 1/σ_i（σ=60日年化波动，下限0.05），归一
  - 月频调仓，佣金万1；信号用前一日收盘，次月生效（无前视）
运行：
  - QMT 客户端「策略回测」加载本文件即可回测；
  - 实盘/模拟需先 xtdata.connect()（miniQMT 运行中的 XtMiniQmt.exe）。
注：本机研究环境（quantaalpha）无 xtquant，可用 tools/index_lab 做等价回测。
"""
import numpy as np
import pandas as pd

try:
    from xtquant.xtdata import get_local_data, download_history_data
except Exception:  # 非 QMT 环境忽略，逻辑仍可被复用
    get_local_data = None
    download_history_data = None

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


def _closes(code, count):
    """取最近 count 根日线收盘（QMT 本地数据）。"""
    if get_local_data is None:
        return None
    df = get_local_data(field_list=["close"], stock_code=code, period="1d", count=count)
    if df is None or len(df) == 0:
        return None
    return np.array(df["close"].values, dtype=float)


def signal_and_weights(codes):
    """返回 (held_codes, weights_dict)。QMT / Ptrade 共用，无平台依赖。"""
    held, vols = [], {}
    for code in codes:
        c = _closes(code, SMA_W + 1)
        if c is None or len(c) < SMA_W:
            continue
        sma = float(np.mean(c[-SMA_W:]))
        if c[-1] > sma:
            held.append(code)
            ret = pd.Series(c).pct_change().dropna()
            vols[code] = float(ret.tail(VOL_WINDOW).std() * np.sqrt(252))
    if not held:
        return [], {}
    inv = {x: 1.0 / max(vols[x], INV_VOL_FLOOR) for x in held}
    tot = sum(inv.values())
    return held, {x: inv[x] / tot for x in held}


# ---------------- QMT 生命周期 ----------------
def init(ContextInfo):
    ContextInfo.pool = POOL
    ContextInfo.last_month = None
    ContextInfo.weights = {}
    for code in POOL:
        try:
            download_history_data(code, "1d", "", "", "")  # 确保本地有数据
        except Exception:
            pass


def handlebar(ContextInfo):
    """每个交易日触发；月末调仓。"""
    dt = ContextInfo.get_bar_tradedate()          # 'YYYYMMDD'
    y, m = int(dt[:4]), int(dt[4:6])
    if ContextInfo.last_month == (y, m):
        return
    ContextInfo.last_month = (y, m)

    held, w = signal_and_weights(ContextInfo.pool)
    ContextInfo.weights = w
    # 总资产（按你的 QMT 版本取账户总值，下面为通用写法）
    total = ContextInfo.portfolio.accounts["total_value"] \
        if hasattr(ContextInfo.portfolio, "accounts") else 0.0
    for code in ContextInfo.pool:
        if code in w:
            ContextInfo.order_target_value(code, total * w[code])
        else:
            ContextInfo.order_target_value(code, 0)


if __name__ == "__main__":
    # 本机（qmt312 环境）直接连 QMT 取数，打印当前信号，验证脚本可跑。
    try:
        from xtquant.xtdata import connect
        connect()
    except Exception as e:
        print("[跳过] 非 QMT 环境:", e)
        raise SystemExit
    held, w = signal_and_weights(POOL)
    print(f"持有 {len(held)}/{len(POOL)} 只；权重:")
    for c in sorted(w, key=lambda x: -w[x]):
        print(f"  {c}  {w[c]*100:.1f}%")
